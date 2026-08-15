mod agent;
mod api;
mod config;
mod lsp;
mod mcp;
mod rag;
mod tools;
mod tui;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::agent::subagent::DispatchAgent;
use crate::agent::{AgentEvent, AgentLoop, default_system_prompt};
use crate::api::client::Client;
use crate::config::Config;
use crate::rag::embedder::Embedder;
use crate::rag::indexer::{collection_name, index_project, is_indexable};
use crate::rag::store::Store;
use crate::tools::Registry;
use crate::tools::lsp::{LspDefinition, LspHover, LspManager, LspReferences};
use crate::tools::mcp::register_mcp_tools;
use crate::tools::rag::RagSearch;
use crate::tools::shell::{ConsoleGate, PermissionGate};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("index") => cmd_index().await,
        Some("ask") if args.len() > 1 => cmd_ask(args[1..].join(" ")).await,
        None => cmd_tui().await,
        _ => {
            eprintln!("usage: claus [index | ask <question>]");
            std::process::exit(2);
        }
    }
}

/// Index (or re-index) the current project into Qdrant for rag_search.
async fn cmd_index() -> Result<()> {
    let config = Config::from_env()?;
    let root = std::env::current_dir()?;
    let embedder = build_embedder(&config)?;
    let store = Store::connect(&config.qdrant_url, collection_name(&root)).await?;

    println!("indexing {} ...", root.display());
    let report = index_project(&root, &embedder, &store).await?;
    println!(
        "indexed {} file(s) ({} chunks), {} unchanged, {} removed",
        report.indexed_files, report.chunks, report.unchanged_files, report.removed_files
    );
    Ok(())
}

/// One-shot question without the TUI; prints agent activity to stdout.
async fn cmd_ask(question: String) -> Result<()> {
    let config = Config::from_env()?;
    let root = std::env::current_dir()?;
    let (registry, notes, _) = build_registry(&config, &root, Arc::new(ConsoleGate)).await;
    for note in notes {
        eprintln!("[claus] {note}");
    }

    let client = Client::new(
        config.anthropic_api_key.clone(),
        config.model.clone(),
        config.max_tokens,
    );
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let printer = tokio::spawn(async move {
        use std::io::Write;
        while let Some(event) = event_rx.recv().await {
            match event {
                AgentEvent::TextDelta(fragment) => {
                    print!("{fragment}");
                    let _ = std::io::stdout().flush();
                }
                AgentEvent::AssistantText(_) => println!(),
                AgentEvent::ToolCall { name, input } => {
                    let mut summary = input.to_string();
                    summary.truncate(120);
                    eprintln!("[tool] {name} {summary}");
                }
                AgentEvent::ToolResult {
                    name,
                    is_error: true,
                    output,
                } => {
                    eprintln!("[tool] {name} failed: {}", output.lines().next().unwrap_or(""));
                }
                _ => {}
            }
        }
    });

    let system = default_system_prompt(&root.display().to_string());
    let mut agent = AgentLoop::new(client, registry, system).with_events(event_tx);
    agent.run(question).await?;
    printer.abort();
    Ok(())
}

/// Interactive TUI: the agent runs in its own task and keeps conversation
/// state across turns; the UI exchanges prompts and events over channels.
async fn cmd_tui() -> Result<()> {
    let config = Config::from_env()?;
    let root = std::env::current_dir()?;
    let (permission_tx, permission_rx) = mpsc::unbounded_channel();
    let (registry, notes, rag) = build_registry(&config, &root, Arc::new(tui::TuiGate::new(permission_tx))).await;

    let client = Client::new(
        config.anthropic_api_key.clone(),
        config.model.clone(),
        config.max_tokens,
    );
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<tui::UiCommand>();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let event_tx_watcher = event_tx.clone();

    let system = default_system_prompt(&root.display().to_string());
    let mut agent = AgentLoop::new(client, registry, system).with_events(event_tx.clone());
    tokio::spawn(async move {
        while let Some(command) = prompt_rx.recv().await {
            match command {
                tui::UiCommand::Prompt(prompt) => {
                    // Racing the turn against an Interrupt command lets Esc
                    // cancel mid-flight; the dropped future leaves history in
                    // a possibly invalid state that repair_interrupted fixes.
                    let mut interrupted = false;
                    tokio::select! {
                        result = agent.run(prompt) => {
                            if let Err(error) = result {
                                let _ = event_tx.send(AgentEvent::Error(format!("{error:#}")));
                            }
                        }
                        _ = wait_for_interrupt(&mut prompt_rx) => interrupted = true,
                    }
                    if interrupted {
                        agent.repair_interrupted();
                        let _ = event_tx.send(AgentEvent::Interrupted);
                    }
                }
                tui::UiCommand::Clear => agent.clear(),
                tui::UiCommand::Interrupt => {} // nothing running
            }
        }
    });

    if let Some(handles) = rag {
        spawn_reindex_watcher(root.clone(), handles, event_tx_watcher);
    }

    tui::App::new(config.model, notes, prompt_tx, event_rx, permission_rx)
        .run()
        .await
}

/// Resolve only when an Interrupt arrives; other commands cannot be sent
/// while a turn is running (the TUI blocks them), so they are ignored.
async fn wait_for_interrupt(commands: &mut mpsc::UnboundedReceiver<tui::UiCommand>) {
    loop {
        match commands.recv().await {
            Some(tui::UiCommand::Interrupt) => return,
            Some(_) => continue,
            None => std::future::pending::<()>().await, // channel closed: let the turn finish
        }
    }
}

/// RAG backends shared between the search tool and the reindex watcher.
struct RagHandles {
    embedder: Embedder,
    store: Arc<Store>,
}

/// Watch the project and incrementally re-index after edits settle. Events
/// arrive on notify's own thread and are debounced (2s of quiet) before one
/// sequential `index_project` run, so bursts of saves coalesce.
fn spawn_reindex_watcher(root: std::path::PathBuf, handles: RagHandles, events: mpsc::UnboundedSender<AgentEvent>) {
    use notify::{RecursiveMode, Watcher};

    let (raw_tx, mut raw_rx) = mpsc::unbounded_channel();
    let watch_root = root.clone();
    tokio::spawn(async move {
        let mut watcher = {
            let root = watch_root.clone();
            match notify::recommended_watcher(move |result: Result<notify::Event, notify::Error>| {
                if let Ok(event) = result
                    && event
                        .paths
                        .iter()
                        .any(|path| is_indexable(path.strip_prefix(&root).unwrap_or(path)))
                {
                    let _ = raw_tx.send(());
                }
            }) {
                Ok(watcher) => watcher,
                Err(_) => return,
            }
        };
        if watcher.watch(&watch_root, RecursiveMode::Recursive).is_err() {
            return;
        }

        while raw_rx.recv().await.is_some() {
            // Debounce: wait for 2s without further relevant events.
            while let Ok(Some(())) = tokio::time::timeout(std::time::Duration::from_secs(2), raw_rx.recv()).await {}
            match index_project(&watch_root, &handles.embedder, &handles.store).await {
                Ok(report) if report.indexed_files + report.removed_files > 0 => {
                    let _ = events.send(AgentEvent::Info(format!(
                        "reindexed {} changed file(s)",
                        report.indexed_files + report.removed_files
                    )));
                }
                Ok(_) => {}
                Err(error) => {
                    let _ = events.send(AgentEvent::Info(format!("background reindex failed: {error:#}")));
                }
            }
        }
    });
}

fn build_embedder(config: &Config) -> Result<Embedder> {
    let key = config
        .voyage_api_key
        .clone()
        .context("VOYAGE_API_KEY is not set (required for embeddings)")?;
    Ok(Embedder::new(key))
}

/// Assemble the tool set. Optional capabilities (RAG, MCP) degrade to a
/// startup note instead of failing the whole app.
async fn build_registry(
    config: &Config,
    root: &Path,
    gate: Arc<dyn PermissionGate>,
) -> (Registry, Vec<String>, Option<RagHandles>) {
    let mut registry = Registry::default();
    let mut notes = Vec::new();
    let mut rag_handles = None;

    registry.register(Arc::new(tools::fs::ReadFile));
    registry.register(Arc::new(tools::fs::WriteFile));
    registry.register(Arc::new(tools::fs::EditFile));
    registry.register(Arc::new(tools::fs::ListDir));
    registry.register(Arc::new(tools::shell::Shell::new(Some(gate))));
    registry.register(Arc::new(tools::search::SearchText));

    let lsp = LspManager::new(root.to_path_buf());
    registry.register(Arc::new(LspDefinition(Arc::clone(&lsp))));
    registry.register(Arc::new(LspReferences(Arc::clone(&lsp))));
    registry.register(Arc::new(LspHover(lsp)));

    match build_embedder(config) {
        Ok(embedder) => match Store::connect(&config.qdrant_url, collection_name(root)).await {
            Ok(store) => {
                let store = Arc::new(store);
                registry.register(Arc::new(RagSearch::new(
                    embedder.clone(),
                    Arc::clone(&store),
                    root.to_path_buf(),
                )));
                rag_handles = Some(RagHandles { embedder, store });
                notes.push("rag_search ready (index kept fresh by the file watcher)".to_string());
            }
            Err(error) => notes.push(format!("rag_search disabled: {error:#}")),
        },
        Err(error) => notes.push(format!("rag_search disabled: {error:#}")),
    }

    let mcp_config = mcp::McpConfig::load(root);
    notes.extend(register_mcp_tools(&mut registry, &mcp_config).await);

    // The dispatcher captures the registry built so far; sub-agents get every
    // tool except dispatch_agent itself.
    let client = Client::new(
        config.anthropic_api_key.clone(),
        config.model.clone(),
        config.max_tokens,
    );
    let dispatch = DispatchAgent::new(client, registry.clone(), root.display().to_string());
    registry.register(Arc::new(dispatch));

    (registry, notes, rag_handles)
}
