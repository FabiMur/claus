//! Minimal BM25 ranking over the chunk corpus, used as the lexical half of
//! hybrid search (fused with vector results via reciprocal rank fusion).

use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Lowercased alphanumeric terms; identifiers split on `_` and non-alphanumerics.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|term| term.len() > 1)
        .map(str::to_lowercase)
        .collect()
}

/// Score every document against the query; returns `(doc_index, score)` for
/// matching documents, best first, at most `top_n`.
pub fn rank(documents: &[String], query: &str, top_n: usize) -> Vec<(usize, f32)> {
    let query_terms = tokenize(query);
    if query_terms.is_empty() || documents.is_empty() {
        return Vec::new();
    }

    let tokenized: Vec<Vec<String>> = documents.iter().map(|d| tokenize(d)).collect();
    let total_docs = documents.len() as f32;
    let average_length = tokenized.iter().map(Vec::len).sum::<usize>() as f32 / total_docs;

    // Document frequency per query term.
    let mut doc_frequency: HashMap<&str, f32> = HashMap::new();
    for term in &query_terms {
        let count = tokenized.iter().filter(|doc| doc.iter().any(|t| t == term)).count();
        doc_frequency.insert(term, count as f32);
    }

    let mut scores: Vec<(usize, f32)> = tokenized
        .iter()
        .enumerate()
        .map(|(index, doc)| {
            let doc_length = doc.len() as f32;
            let score = query_terms
                .iter()
                .map(|term| {
                    let df = doc_frequency[term.as_str()];
                    if df == 0.0 {
                        return 0.0;
                    }
                    let tf = doc.iter().filter(|t| *t == term).count() as f32;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    let idf = ((total_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
                    idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doc_length / average_length.max(1.0)))
                })
                .sum();
            (index, score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_n);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_exact_term_matches_first() {
        let docs = vec![
            "fn retry_request with exponential backoff".to_string(),
            "fn render_status_bar for the tui".to_string(),
            "retry retry retry everywhere".to_string(),
        ];
        let results = rank(&docs, "retry backoff", 3);
        assert!(!results.is_empty());
        // Doc 0 matches both terms; it must beat doc 2 (one term, repeated).
        assert_eq!(results[0].0, 0);
        // Doc 1 matches nothing and is filtered out.
        assert!(results.iter().all(|(index, _)| *index != 1));
    }

    #[test]
    fn empty_query_or_corpus_returns_nothing() {
        assert!(rank(&[], "query", 5).is_empty());
        assert!(rank(&["doc".to_string()], "", 5).is_empty());
    }

    #[test]
    fn tokenizer_splits_identifiers() {
        assert_eq!(tokenize("read_file(path)"), vec!["read", "file", "path"]);
    }
}
