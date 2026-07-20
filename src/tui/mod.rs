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
        Self {
            entries,
            input: String::new(),
            scroll_from_bottom: 0,
            busy: false,
            spinner_tick: 0,
            model,
            usage: Usage::default(),
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
            AgentEvent::TurnComplete { usage } => {
                self.usage.input_tokens += usage.input_tokens;
                self.usage.output_tokens += usage.output_tokens;
                self.busy = false;
            }
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

        // Status bar.
        let spinner = if self.busy {
            format!("{} working ", SPINNER[self.spinner_tick % SPINNER.len()])
        } else {
            String::new()
        };
        let status = Line::from(vec![
            Span::styled(spinner, Style::default().fg(Color::Yellow)),
            Span::styled(
                format!(
                    " {} · in {} out {} tokens ",
                    self.model, self.usage.input_tokens, self.usage.output_tokens
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
        frame.render_widget(Paragraph::new(status), status_area);
    }
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
