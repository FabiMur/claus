use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use walkdir::WalkDir;

use crate::rag::chunker::{Chunk, chunk_source};
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
const CHUNKS_PATH: &str = ".claus/chunks.jsonl";

/// One indexed chunk, persisted locally as the BM25 corpus for hybrid search
/// (Qdrant holds the same data, but lexical scoring needs it all in memory).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ChunkRecord {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

/// Load the local chunk corpus written by the indexer (empty if not indexed).
pub fn load_chunk_corpus(root: &Path) -> Vec<ChunkRecord> {
    let Ok(content) = std::fs::read_to_string(root.join(CHUNKS_PATH)) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

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

/// One changed file waiting to be embedded, together with the offset of its
/// chunk texts inside the run's single batched embedding request.
struct Pending {
    relative: String,
    hash: String,
    chunks: Vec<Chunk>,
    offset: usize,
}

/// Incrementally index the project: only files whose content hash changed are
/// re-chunked, re-embedded and re-upserted; deleted files are purged. Chunks
/// from every changed file are embedded in a single batched pass, so a run
/// costs a handful of requests instead of one per file.
pub async fn index_project(root: &Path, embedder: &Embedder, store: &Store) -> Result<IndexReport> {
    let manifest_file = root.join(MANIFEST_PATH);
    let mut manifest: HashMap<String, String> = match std::fs::read_to_string(&manifest_file) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => HashMap::new(),
    };
    // Sidecar BM25 corpus, keyed by file so unchanged files keep their chunks.
    let mut corpus: HashMap<String, Vec<ChunkRecord>> = HashMap::new();
    for record in load_chunk_corpus(root) {
        corpus.entry(record.path.clone()).or_default().push(record);
    }

    let mut report = IndexReport::default();
    let mut seen = Vec::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut texts: Vec<String> = Vec::new();

    for path in collect_files(root) {
        let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().to_string();
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue; // non-UTF-8 despite the extension filter
        };
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        // Every readable file counts as seen, even when it is unchanged or
        // yields no chunks, or the purge below would drop a live file.
        seen.push(relative.clone());

        if manifest.get(&relative) == Some(&hash) {
            report.unchanged_files += 1;
            continue;
        }

        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
        let chunks = chunk_source(extension, &content);
        if chunks.is_empty() {
            continue; // nothing to embed: stays out of the manifest, retried next run
        }
        let offset = texts.len();
        texts.extend(
            chunks
                .iter()
                .map(|chunk| format!("// {relative}:{}\n{}", chunk.start_line, chunk.text)),
        );
        pending.push(Pending {
            relative,
            hash,
            chunks,
            offset,
        });
    }

    // One embedding pass for the whole run; `Embedder::embed` splits it into
    // requests of at most BATCH_SIZE chunks / MAX_BATCH_CHARS characters.
    let vectors = if texts.is_empty() {
        Vec::new()
    } else {
        embedder.embed(&texts, InputType::Document).await?
    };
    anyhow::ensure!(
        vectors.len() == texts.len(),
        "embedder returned {} vectors for {} chunk texts",
        vectors.len(),
        texts.len()
    );

    for file in pending {
        // Slice by the recorded offset: `upsert_chunks` zips chunks with
        // vectors, so a misaligned slice would silently pair a chunk with
        // another chunk's embedding instead of failing.
        let file_vectors = vectors[file.offset..file.offset + file.chunks.len()].to_vec();

        store.delete_file(&file.relative).await?;
        store.upsert_chunks(&file.relative, &file.chunks, file_vectors).await?;

        corpus.insert(
            file.relative.clone(),
            file.chunks
                .iter()
                .map(|chunk| ChunkRecord {
                    path: file.relative.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    text: chunk.text.clone(),
                })
                .collect(),
        );
        report.indexed_files += 1;
        report.chunks += file.chunks.len();
        manifest.insert(file.relative, file.hash);
    }

    // Purge vectors of files that no longer exist.
    let gone: Vec<String> = manifest.keys().filter(|k| !seen.contains(k)).cloned().collect();
    for path in gone {
        store.delete_file(&path).await?;
        manifest.remove(&path);
        corpus.remove(&path);
        report.removed_files += 1;
    }

    if let Some(parent) = manifest_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&manifest_file, serde_json::to_string_pretty(&manifest)?).context("writing index manifest")?;
    let corpus_lines: Vec<String> = corpus
        .into_values()
        .flatten()
        .filter_map(|record| serde_json::to_string(&record).ok())
        .collect();
    std::fs::write(root.join(CHUNKS_PATH), corpus_lines.join("\n")).context("writing chunk corpus")?;
    Ok(report)
}

/// Whether a path would be picked up by the indexer — used by the file
/// watcher to ignore churn in build artifacts, VCS metadata and our own
/// manifest writes.
pub fn is_indexable(path: &Path) -> bool {
    let in_skipped_dir = path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|name| SKIP_DIRS.contains(&name) || (name.starts_with('.') && name.len() > 1))
    });
    !in_skipped_dir
        && path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| TEXT_EXTENSIONS.contains(&ext))
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
