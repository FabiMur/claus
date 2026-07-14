use tree_sitter::Parser;

/// A contiguous piece of a source file, sized for embedding. Lines are 1-based.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Target upper bound for one chunk; small adjacent items are grouped up to this.
const MAX_CHUNK_CHARS: usize = 3000;
/// Fallback window for files without a tree-sitter grammar.
const WINDOW_LINES: usize = 80;
const WINDOW_OVERLAP: usize = 10;

/// Split a source file into chunks: syntax-aware (top-level items) for Rust and
/// Python, fixed line windows for everything else.
pub fn chunk_source(extension: &str, source: &str) -> Vec<Chunk> {
    let language = match extension {
        "rs" => Some(tree_sitter_rust::LANGUAGE),
        "py" => Some(tree_sitter_python::LANGUAGE),
        _ => None,
    };
    match language {
        Some(lang) => chunk_syntax(lang.into(), source).unwrap_or_else(|| chunk_windows(source)),
        None => chunk_windows(source),
    }
}

fn chunk_syntax(language: tree_sitter::Language, source: &str) -> Option<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(source, None)?;
    let root = tree.root_node();

    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut group: Option<(usize, usize)> = None; // (start_row, end_row), 0-based inclusive

    let mut cursor = root.walk();
    for node in root.named_children(&mut cursor) {
        let (start, end) = (node.start_position().row, node.end_position().row);
        match group {
            Some((gs, ge)) if span_chars(&lines, gs, end) <= MAX_CHUNK_CHARS => {
                group = Some((gs, ge.max(end)));
            }
            Some((gs, ge)) => {
                push_span(&mut chunks, &lines, gs, ge);
                group = Some((start, end));
            }
            None => group = Some((start, end)),
        }
        // A single oversized item becomes its own (possibly split) chunk.
        if let Some((gs, ge)) = group
            && span_chars(&lines, gs, ge) > MAX_CHUNK_CHARS
        {
            push_span(&mut chunks, &lines, gs, ge);
            group = None;
        }
    }
    if let Some((gs, ge)) = group {
        push_span(&mut chunks, &lines, gs, ge);
    }
    Some(chunks)
}

fn span_chars(lines: &[&str], start: usize, end: usize) -> usize {
    lines[start..=end.min(lines.len().saturating_sub(1))]
        .iter()
        .map(|l| l.len() + 1)
        .sum()
}

/// Append the span as one chunk, splitting into line windows when oversized.
fn push_span(chunks: &mut Vec<Chunk>, lines: &[&str], start: usize, end: usize) {
    let end = end.min(lines.len().saturating_sub(1));
    if span_chars(lines, start, end) <= MAX_CHUNK_CHARS {
        chunks.push(make_chunk(lines, start, end));
        return;
    }
    let mut row = start;
    while row <= end {
        let window_end = (row + WINDOW_LINES - 1).min(end);
        chunks.push(make_chunk(lines, row, window_end));
        if window_end == end {
            break;
        }
        row = window_end + 1 - WINDOW_OVERLAP.min(window_end - row);
    }
}

fn chunk_windows(source: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut row = 0;
    while row < lines.len() {
        let end = (row + WINDOW_LINES - 1).min(lines.len() - 1);
        chunks.push(make_chunk(&lines, row, end));
        if end == lines.len() - 1 {
            break;
        }
        row = end + 1 - WINDOW_OVERLAP;
    }
    chunks
}

fn make_chunk(lines: &[&str], start: usize, end: usize) -> Chunk {
    Chunk {
        text: lines[start..=end].join("\n"),
        start_line: start + 1,
        end_line: end + 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_whole_file_with_overlap() {
        let source = (1..=200).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let chunks = chunk_windows(&source);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.first().unwrap().start_line, 1);
        assert_eq!(chunks.last().unwrap().end_line, 200);
        // Consecutive windows overlap so no boundary context is lost.
        assert!(chunks[1].start_line < chunks[0].end_line);
    }

    #[test]
    fn rust_items_are_grouped_by_syntax() {
        let source = "fn a() -> u32 {\n    1\n}\n\nfn b() -> u32 {\n    2\n}\n";
        let chunks = chunk_source("rs", source);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("fn a"));
        assert!(chunks[0].text.contains("fn b"));
    }

    #[test]
    fn oversized_function_is_split() {
        let body = (0..300)
            .map(|i| format!("    let x{i} = {i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!("fn big() {{\n{body}\n}}\n");
        let chunks = chunk_source("rs", &source);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn unknown_extension_falls_back_to_windows() {
        let chunks = chunk_source("md", "# title\n\nsome text\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
    }
}
