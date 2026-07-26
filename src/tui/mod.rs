use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use tokio::sync::mpsc;

use crate::agent::AgentEvent;
use crate::api::types::Usage;

const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    User,
    Assistant,
    Tool,
    Info,
    Error,
}

struct Entry {
    kind: Kind,
    text: String,
}

/// Terminal chat interface: renders the conversation, forwards prompts to the
/// agent task and displays its events as they arrive.
pub struct App {
    entries: Vec<Entry>,
    input: String,
    scroll_from_bottom: usize,
    busy: bool,
    spinner_tick: usize,
    model: String,
    usage: Usage,
    /// Tokens of the latest API round-trip: approximates current context size.
    context_tokens: u32,
    context_window: u32,
    cwd: String,
    git_branch: Option<String>,
    prompt_tx: mpsc::UnboundedSender<String>,
    agent_rx: mpsc::UnboundedReceiver<AgentEvent>,
}

impl App {
    pub fn new(
        model: String,
        startup_notes: Vec<String>,
        prompt_tx: mpsc::UnboundedSender<String>,
        agent_rx: mpsc::UnboundedReceiver<AgentEvent>,
    ) -> Self {
        let mut entries = vec![Entry {
            kind: Kind::Info,
            text: "claus — terminal coding agent. Enter sends, Esc quits, ↑/↓ scroll.".to_string(),
        }];
        entries.extend(startup_notes.into_iter().map(|text| Entry { kind: Kind::Info, text }));
        let context_window = context_window_for(&model);
        Self {
            entries,
            input: String::new(),
            scroll_from_bottom: 0,
            busy: false,
            spinner_tick: 0,
            model,
            usage: Usage::default(),
            context_tokens: 0,
            context_window,
            cwd: abbreviated_cwd(),
            git_branch: current_git_branch(),
            prompt_tx,
            agent_rx,
        }
    }

    pub async fn run(mut self) -> Result<()> {
        let mut terminal = ratatui::init();
        let result = self.event_loop(&mut terminal).await;
        ratatui::restore();
        result
    }

    async fn event_loop(&mut self, terminal: &mut ratatui::DefaultTerminal) -> Result<()> {
        // Blocking crossterm reads happen on a plain thread; events are
        // forwarded into the async world through a channel.
        let (input_tx, mut input_rx) = mpsc::unbounded_channel();
        std::thread::spawn(move || {
            while let Ok(event) = crossterm::event::read() {
                if input_tx.send(event).is_err() {
                    break;
                }
            }
        });
        let mut ticker = tokio::time::interval(std::time::Duration::from_millis(120));

        loop {
            terminal.draw(|frame| self.draw(frame))?;
            tokio::select! {
                Some(event) = input_rx.recv() => {
                    if self.handle_input(event) {
                        return Ok(());
                    }
                }
                Some(event) = self.agent_rx.recv() => self.handle_agent_event(event),
                _ = ticker.tick() => {
                    if self.busy {
                        self.spinner_tick = self.spinner_tick.wrapping_add(1);
                    }
                }
            }
        }
    }

    /// Returns true when the app should quit.
    fn handle_input(&mut self, event: Event) -> bool {
        let Event::Key(key) = event else {
            return false;
        };
        if key.kind != KeyEventKind::Press {
            return false;
        }
        match key.code {
            KeyCode::Esc => return true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return true,
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Up => self.scroll_from_bottom += 1,
            KeyCode::Down => self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(1),
            KeyCode::PageUp => self.scroll_from_bottom += 10,
            KeyCode::PageDown => self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(10),
            KeyCode::Char(c) => self.input.push(c),
            _ => {}
        }
        false
    }

    fn submit(&mut self) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.busy {
            return;
        }
        self.input.clear();
        self.scroll_from_bottom = 0;
        self.entries.push(Entry {
            kind: Kind::User,
            text: prompt.clone(),
        });
        self.busy = true;
        let _ = self.prompt_tx.send(prompt);
    }

    fn handle_agent_event(&mut self, event: AgentEvent) {
        self.scroll_from_bottom = 0;
        match event {
            AgentEvent::AssistantText(text) => self.entries.push(Entry {
                kind: Kind::Assistant,
                text,
            }),
            AgentEvent::ToolCall { name, input } => {
                let mut summary = serde_json::to_string(&input).unwrap_or_default();
                summary.truncate(100);
                self.entries.push(Entry {
                    kind: Kind::Tool,
                    text: format!("⚙ {name} {summary}"),
                });
            }
            AgentEvent::ToolResult { name, output, is_error } => {
                let first_line = output
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(100)
                    .collect::<String>();
                self.entries.push(Entry {
                    kind: if is_error { Kind::Error } else { Kind::Tool },
                    text: format!("  ↳ {name}: {first_line}"),
                });
            }
            AgentEvent::ApiUsage { usage } => {
                self.context_tokens = usage.input_tokens + usage.output_tokens;
                self.usage.input_tokens += usage.input_tokens;
                self.usage.output_tokens += usage.output_tokens;
            }
            AgentEvent::TurnComplete => self.busy = false,
            AgentEvent::Error(message) => {
                self.entries.push(Entry {
                    kind: Kind::Error,
                    text: format!("error: {message}"),
                });
                self.busy = false;
            }
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let [chat_area, input_area, status_area] =
            Layout::vertical([Constraint::Min(1), Constraint::Length(3), Constraint::Length(1)]).areas(frame.area());

        // Chat pane, auto-scrolled to the bottom unless the user scrolled up.
        let width = chat_area.width.max(1) as usize;
        let lines: Vec<Line> = self
            .entries
            .iter()
            .flat_map(|entry| entry_lines(entry, width))
            .collect();
        let total_rows: usize = lines
            .iter()
            .map(|line| {
                let chars: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
                chars.div_ceil(width).max(1)
            })
            .sum();
        let viewport = chat_area.height as usize;
        let max_offset = total_rows.saturating_sub(viewport);
        let offset = max_offset.saturating_sub(self.scroll_from_bottom.min(max_offset));
        let chat = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((offset as u16, 0));
        frame.render_widget(chat, chat_area);

        // Input box.
        let input =
            Paragraph::new(format!("{}█", self.input)).block(Block::default().borders(Borders::ALL).title(" prompt "));
        frame.render_widget(input, input_area);

        frame.render_widget(Paragraph::new(self.status_line()), status_area);
    }

    /// Claude-Code-style status bar:
    /// `📁 ~/Projects/x (main) · model · ▰▰▱▱▱▱▱▱▱▱ 22% · $0.53`
    fn status_line(&self) -> Line<'_> {
        let dim = Style::default().fg(Color::DarkGray);
        let mut spans = vec![Span::styled(
            format!("📁 {}", self.cwd),
            Style::default().fg(Color::Blue),
        )];
        if let Some(branch) = &self.git_branch {
            spans.push(Span::styled(
                format!(" ({branch})"),
                Style::default().fg(Color::Magenta),
            ));
        }
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled(
            self.model.clone(),
            Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(" · ", dim));

        let percent = (self.context_tokens as f64 / self.context_window as f64 * 100.0).min(100.0);
        let filled = (percent / 10.0).round() as usize;
        spans.push(Span::styled("▰".repeat(filled), Style::default().fg(Color::Yellow)));
        spans.push(Span::styled("▱".repeat(10 - filled.min(10)), dim));
        spans.push(Span::styled(format!(" {percent:.0}%"), dim));

        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled(
            format!("${:.2}", session_cost_usd(&self.model, &self.usage)),
            Style::default().fg(Color::Green),
        ));

        if self.busy {
            spans.push(Span::styled(
                format!("  {} working", SPINNER[self.spinner_tick % SPINNER.len()]),
                Style::default().fg(Color::Yellow),
            ));
        }
        Line::from(spans)
    }
}

/// Context window of the configured model, for the usage bar.
fn context_window_for(model: &str) -> u32 {
    if model.contains("haiku") { 200_000 } else { 1_000_000 }
}

/// Approximate cost in USD from (input, output) prices per million tokens.
fn session_cost_usd(model: &str, usage: &Usage) -> f64 {
    let (input_per_m, output_per_m) = if model.contains("fable") || model.contains("mythos") {
        (10.0, 50.0)
    } else if model.contains("opus") {
        (5.0, 25.0)
    } else if model.contains("sonnet-5") {
        (2.0, 10.0)
    } else if model.contains("sonnet") {
        (3.0, 15.0)
    } else if model.contains("haiku") {
        (1.0, 5.0)
    } else {
        (5.0, 25.0)
    };
    usage.input_tokens as f64 / 1e6 * input_per_m + usage.output_tokens as f64 / 1e6 * output_per_m
}

fn abbreviated_cwd() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let home = std::env::var("HOME").unwrap_or_default();
    let text = cwd.display().to_string();
    match text.strip_prefix(&home) {
        Some(rest) if !home.is_empty() => format!("~{rest}"),
        _ => text,
    }
}

/// Current branch from `.git/HEAD`, without spawning git.
fn current_git_branch() -> Option<String> {
    let head = std::fs::read_to_string(".git/HEAD").ok()?;
    head.trim()
        .strip_prefix("ref: refs/heads/")
        .map(str::to_string)
        .or_else(|| Some(head.trim().chars().take(8).collect())) // detached HEAD
}

fn entry_lines(entry: &Entry, _width: usize) -> Vec<Line<'_>> {
    let style = match entry.kind {
        Kind::User => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        Kind::Assistant => Style::default(),
        Kind::Tool => Style::default().fg(Color::Yellow),
        Kind::Info => Style::default().fg(Color::DarkGray),
        Kind::Error => Style::default().fg(Color::Red),
    };
    let prefix = match entry.kind {
        Kind::User => "you › ",
        Kind::Assistant => "claus › ",
        _ => "",
    };
    let mut lines = Vec::new();
    for (i, text_line) in entry.text.lines().enumerate() {
        let content = if i == 0 {
            format!("{prefix}{text_line}")
        } else {
            text_line.to_string()
        };
        lines.push(Line::from(Span::styled(content, style)));
    }
    if entry.kind == Kind::Assistant || entry.kind == Kind::User {
        lines.push(Line::from(""));
    }
    lines
}
