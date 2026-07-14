use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

use crate::rag::chunker::chunk_source;
use crate::rag::embedder::{Embedder, InputType};
use crate::rag::store::Store;

const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".venv",
    "__pycache__",
    ".claus",
    "qdrant_storage",
];
const TEXT_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "jsx", "ts", "tsx", "go", "java", "kt", "c", "h", "cpp", "hpp", "md", "toml", "yaml", "yml",
    "json", "sh", "sql", "html", "css",
];
const MAX_FILE_BYTES: u64 = 1_000_000;
const MANIFEST_PATH: &str = ".claus/manifest.json";

#[derive(Debug, Default)]
pub struct IndexReport {
    pub indexed_files: usize,
    pub removed_files: usize,
    pub unchanged_files: usize,
    pub chunks: usize,
}

/// Name the per-project collection after the absolute project path so
/// different checkouts never share vectors.
pub fn collection_name(project_root: &Path) -> String {
    let hash = blake3::hash(project_root.to_string_lossy().as_bytes());
    format!("claus-{}", &hash.to_hex()[..12])
}

/// Incrementally index the project: only files whose content hash changed are
/// re-chunked, re-embedded and re-upserted; deleted files are purged.
pub async fn index_project(root: &Path, embedder: &Embedder, store: &Store) -> Result<IndexReport> {
    let manifest_file = root.join(MANIFEST_PATH);
    let mut manifest: HashMap<String, String> = match std::fs::read_to_string(&manifest_file) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => HashMap::new(),
    };

    let mut report = IndexReport::default();
    let mut seen = Vec::new();

    for path in collect_files(root) {
        let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().to_string();
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue; // non-UTF-8 despite the extension filter
        };
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        seen.push(relative.clone());

        if manifest.get(&relative) == Some(&hash) {
            report.unchanged_files += 1;
            continue;
        }

        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
        let chunks = chunk_source(extension, &content);
        if chunks.is_empty() {
            continue;
        }
        let texts: Vec<String> = chunks
            .iter()
            .map(|c| format!("// {relative}:{}\n{}", c.start_line, c.text))
            .collect();
        let vectors = embedder.embed(&texts, InputType::Document).await?;

        store.delete_file(&relative).await?;
        store.upsert_chunks(&relative, &chunks, vectors).await?;

        manifest.insert(relative, hash);
        report.indexed_files += 1;
        report.chunks += chunks.len();
    }

    // Purge vectors of files that no longer exist.
    let gone: Vec<String> = manifest.keys().filter(|k| !seen.contains(k)).cloned().collect();
    for path in gone {
        store.delete_file(&path).await?;
        manifest.remove(&path);
        report.removed_files += 1;
    }

    if let Some(parent) = manifest_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&manifest_file, serde_json::to_string_pretty(&manifest)?).context("writing index manifest")?;
    Ok(report)
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| {
            entry
                .file_name()
                .to_str()
                .is_none_or(|name| !SKIP_DIRS.contains(&name) && (!name.starts_with('.') || name.len() <= 1))
        })
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_type().is_file()
                && entry.metadata().map(|m| m.len() <= MAX_FILE_BYTES).unwrap_or(false)
                && entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| TEXT_EXTENSIONS.contains(&ext))
        })
        .map(|entry| entry.into_path())
        .collect()
}
