use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use fxhash::FxHashMap;
use rayon::prelude::*;
use onig::Regex;
use smallvec::SmallVec;
use lru::LruCache;
use std::num::NonZeroUsize;

#[derive(Debug)]
pub struct OptimizedBpeTokenizer {
    vocab: FxHashMap<Vec<u8>, usize>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    merge_ranks: FxHashMap<(Vec<u8>, Vec<u8>), usize>,
    regex: Regex,
    word_cache: Arc<Mutex<LruCache<String, Vec<usize>>>>,
    special_tokens: Vec<String>,
}

impl OptimizedBpeTokenizer {
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("Loading optimized BPE tokenizer...");
        
        // Load vocabulary with FxHashMap for faster hashing
        let vocab_content = std::fs::read_to_string(vocab_path)?;
        let vocab_json: serde_json::Value = serde_json::from_str(&vocab_content)?;
        
        let mut vocab = FxHashMap::default();
        let mut special_tokens = Vec::new();
        
        if let Some(vocab_obj) = vocab_json.as_object() {
            for (token_id_str, token_str_value) in vocab_obj {
                let token_id: usize = token_id_str.parse()?;
                let token_str = token_str_value.as_str().ok_or("Invalid token string")?;
                let token_bytes = parse_json_string(token_str)?;
                
                vocab.insert(token_bytes.clone(), token_id);
                
                // Detect special tokens
                if token_id >= 256 && (token_str.contains("<|") || token_str.contains("|>")) {
                    special_tokens.push(String::from_utf8_lossy(&token_bytes).to_string());
                }
            }
        }
        
        // Load merges and build rank map
        let merges_content = std::fs::read_to_string(merges_path)?;
        let mut merges = Vec::new();
        let mut merge_ranks = FxHashMap::default();
        
        for (rank, line) in merges_content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let first = parts[0].as_bytes().to_vec();
                let second = parts[1].as_bytes().to_vec();
                merge_ranks.insert((first.clone(), second.clone()), rank);
                merges.push((first, second));
            }
        }
        
        // Initialize GPT-2 regex pattern
        let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        let regex = Regex::new(pattern)?;
        
        // Create LRU cache with 100K capacity
        let word_cache = Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(100_000).unwrap())));
        
        log::info!("✓ Loaded {} vocab entries, {} merges", vocab.len(), merges.len());
        
        Ok(Self {
            vocab,
            merges,
            merge_ranks,
            regex,
            word_cache,
            special_tokens,
        })
    }
    
    pub fn encode(&self, text: &str) -> Vec<u16> {
        // For large texts, use parallel processing
        if text.len() > 1_000_000 {
            self.encode_parallel(text)
        } else {
            self.encode_sequential(text)
        }
    }
    
    pub fn encode_with_progress(&self, text: &str, show_progress: bool) -> Vec<u16> {
        // Always use sequential with progress for better logging
        self.encode_with_progress_impl(text, show_progress)
    }
    
    fn encode_sequential(&self, text: &str) -> Vec<u16> {
        self.encode_with_progress_impl(text, false)
    }
    
    fn encode_with_progress_impl(&self, text: &str, show_progress: bool) -> Vec<u16> {
        let mut tokens = Vec::new();
        let mut pos = 0;
        let text_len = text.len();
        let start_time = std::time::Instant::now();
        let mut last_log_mb = 0;
        let mut word_count = 0;
        
        if show_progress {
            log::info!("Starting tokenization: {:.1} MB text", text_len as f64 / 1024.0 / 1024.0);
        }
        
        while pos < text.len() {
            let search_content = &text[pos..];
            if let Some((start, end)) = self.regex.find(search_content) {
                let actual_start = pos + start;
                let actual_end = pos + end;
                let word = &text[actual_start..actual_end];
                
                // Check cache first
                let word_tokens = {
                    let mut cache = self.word_cache.lock().unwrap();
                    if let Some(cached) = cache.get(word) {
                        cached.clone()
                    } else {
                        drop(cache); // Release lock before computation
                        let computed = self.encode_word_optimized(word.as_bytes());
                        let mut cache = self.word_cache.lock().unwrap();
                        cache.put(word.to_string(), computed.clone());
                        computed
                    }
                };
                
                for token_id in word_tokens {
                    if token_id <= u16::MAX as usize {
                        tokens.push(token_id as u16);
                    }
                }
                
                pos = actual_end;
                word_count += 1;
                
                // Progress logging for large texts
                if show_progress && text_len > 10_000_000 { // 10MB threshold
                    let processed_mb = pos / (1024 * 1024);
                    if processed_mb > last_log_mb + 50 { // Every 50MB
                        let progress = (pos as f64 / text_len as f64) * 100.0;
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let throughput = pos as f64 / elapsed / 1024.0 / 1024.0;
                        log::info!("📝 Tokenization progress: {:.1}% | {:.1}MB | {} words → {} tokens | {:.1}MB/s", 
                                  progress, processed_mb as f64, word_count, tokens.len(), throughput);
                        last_log_mb = processed_mb;
                    }
                }
            } else {
                break;
            }
        }
        
        if show_progress {
            let duration = start_time.elapsed();
            log::info!("✅ Tokenization complete: {} words → {} tokens in {:.2}s ({:.1}MB/s)",
                      word_count, tokens.len(), duration.as_secs_f64(),
                      text_len as f64 / duration.as_secs_f64() / 1024.0 / 1024.0);
        }
        
        tokens
    }
    
    fn encode_parallel(&self, text: &str) -> Vec<u16> {
        // Split text into chunks for parallel processing
        const CHUNK_SIZE: usize = 100_000;
        let mut chunks = Vec::new();
        let mut start = 0;
        
        while start < text.len() {
            let mut end = (start + CHUNK_SIZE).min(text.len());
            
            // Find a char boundary
            while end < text.len() && !text.is_char_boundary(end) {
                end += 1;
            }
            
            chunks.push(&text[start..end]);
            start = end;
        }
        
        // Process chunks in parallel
        let chunk_tokens: Vec<Vec<u16>> = chunks
            .par_iter()
            .map(|chunk| self.encode_sequential(chunk))
            .collect();
        
        // Combine results
        let mut all_tokens = Vec::new();
        for tokens in chunk_tokens {
            all_tokens.extend(tokens);
        }
        
        all_tokens
    }
    
    fn encode_word_optimized(&self, word: &[u8]) -> Vec<usize> {
        // Use SmallVec to avoid heap allocation for small words
        type TokenVec = SmallVec<[Vec<u8>; 32]>;
        
        // Start with byte-level tokens
        let mut tokens: TokenVec = word.iter()
            .map(|&b| vec![b])
            .collect();
        
        // Rank-based merging: find and apply lowest-rank merges
        loop {
            if tokens.len() < 2 {
                break;
            }
            
            // Find the lowest-rank merge applicable to current tokens
            let mut best_merge: Option<(usize, usize)> = None;
            let mut best_rank = usize::MAX;
            
            for i in 0..tokens.len() - 1 {
                let pair = (&tokens[i], &tokens[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&(pair.0.clone(), pair.1.clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_merge = Some((i, rank));
                    }
                }
            }
            
            // If no merge found, we're done
            let Some((merge_pos, _)) = best_merge else {
                break;
            };
            
            // Apply the best merge
            let mut new_tokens = TokenVec::new();
            let mut i = 0;
            
            while i < tokens.len() {
                if i == merge_pos {
                    // Merge tokens at position i and i+1
                    let mut merged = tokens[i].clone();
                    merged.extend_from_slice(&tokens[i + 1]);
                    new_tokens.push(merged);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            
            tokens = new_tokens;
        }
        
        // Convert byte sequences to token IDs
        tokens.into_iter()
            .map(|token_bytes| {
                self.vocab.get(&token_bytes).copied().unwrap_or(0)
            })
            .collect()
    }
}

fn parse_json_string(json_str: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    let mut chars = json_str.chars().peekable();
    
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => bytes.push(b'\\'),
                Some('"') => bytes.push(b'"'),
                Some('n') => bytes.push(b'\n'),
                Some('r') => bytes.push(b'\r'),
                Some('t') => bytes.push(b'\t'),
                Some('u') => {
                    // Parse \uXXXX unicode escape
                    let mut hex_str = String::new();
                    for _ in 0..4 {
                        if let Some(hex_char) = chars.next() {
                            hex_str.push(hex_char);
                        }
                    }
                    if let Ok(unicode_val) = u32::from_str_radix(&hex_str, 16) {
                        if unicode_val <= 255 {
                            bytes.push(unicode_val as u8);
                        }
                    }
                }
                Some(c) => bytes.push(c as u8),
                None => break,
            }
        } else {
            bytes.push(ch as u8);
        }
    }
    
    Ok(bytes)
}