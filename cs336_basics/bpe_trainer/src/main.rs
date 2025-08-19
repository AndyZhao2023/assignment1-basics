use clap::Parser;
use onig::Regex;
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::sync::Arc;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;
use log::{info, error};
use rayon::prelude::*;
use indexmap::IndexMap;

#[derive(Parser)]
#[command(name = "bpe_trainer")]
#[command(about = "Train a Byte Pair Encoding (BPE) tokenizer")]
#[command(version = "1.0")]
struct Args {
    /// Input text file path
    #[arg(short, long)]
    input: String,
    
    /// Target vocabulary size (must be >= 256)
    #[arg(short, long)]
    vocab_size: usize,
    
    /// Output directory for results
    #[arg(short, long)]
    output_dir: String,
    
    /// Special tokens (comma-separated)
    #[arg(short, long, default_value = "<|endoftext|>")]
    special_tokens: String,
    
    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(if args.verbose { log::LevelFilter::Info } else { log::LevelFilter::Warn })
        .format_timestamp_secs()
        .init();
    
    // Validate arguments
    if args.vocab_size < 256 {
        error!("vocab_size must be >= 256, got {}", args.vocab_size);
        std::process::exit(1);
    }
    
    if !Path::new(&args.input).exists() {
        error!("Input file does not exist: {}", args.input);
        std::process::exit(1);
    }
    
    // Parse special tokens
    let special_tokens: Vec<String> = args.special_tokens
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    
    println!("🦀 Pure Rust BPE Trainer");
    println!("📁 Input: {}", args.input);
    println!("🎯 Vocab size: {}", args.vocab_size);
    println!("📂 Output: {}", args.output_dir);
    println!("🏷️  Special tokens: {:?}", special_tokens);
    
    // Run BPE training
    match train_bpe_tokenizer(&args.input, args.vocab_size, &special_tokens, &args.output_dir) {
        Ok(_) => {
            println!("✅ Training completed successfully!");
            println!("📁 Results saved to: {}", args.output_dir);
        }
        Err(e) => {
            error!("Training failed: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

fn train_bpe_tokenizer(
    file_path: &str,
    vocab_size: usize,
    special_tokens: &[String],
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    
    // Create output directory
    fs::create_dir_all(output_dir)?;
    
    info!("=== Starting Pure Rust BPE Training ===");
    info!("File: {}", file_path);
    info!("Target vocab size: {}", vocab_size);
    info!("Special tokens: {:?}", special_tokens);
    info!("Output directory: {}", output_dir);
    
    let file_size = fs::metadata(file_path)?.len();
    info!("File size: {:.1} GB", file_size as f64 / 1e9);
    
    // Phase 1: Pre-tokenization
    info!("Phase 1: Pre-tokenization");
    let pretokenize_start = Instant::now();
    
    let word_counts = pretokenize_file(file_path, file_size)?;
    
    let pretokenize_duration = pretokenize_start.elapsed();
    info!("Pre-tokenization completed in {:.2}s", pretokenize_duration.as_secs_f64());
    info!("Found {} unique word types", word_counts.len());
    
    // Phase 2: Initialize vocabulary
    info!("Phase 2: Vocabulary initialization");
    let mut vocab: HashMap<usize, Vec<u8>> = HashMap::new();
    
    // Add byte tokens (0-255)
    for i in 0..256 {
        vocab.insert(i, vec![i as u8]);
    }
    
    // Add special tokens
    let mut next_id = 256;
    for special_token in special_tokens {
        let token_bytes = special_token.as_bytes().to_vec();
        vocab.insert(next_id, token_bytes);
        next_id += 1;
    }
    
    let initial_vocab_size = next_id;
    info!("Initial vocabulary size: {}", initial_vocab_size);
    
    // Phase 3: Convert words to token sequences
    info!("Phase 3: Converting words to token sequences");
    let convert_start = Instant::now();
    
    let mut word_token_counts: HashMap<Vec<usize>, u32> = HashMap::new();
    for (word_bytes, count) in word_counts {
        let token_sequence: Vec<usize> = word_bytes.iter().map(|&b| b as usize).collect();
        word_token_counts.insert(token_sequence, count);
    }
    
    let convert_duration = convert_start.elapsed();
    info!("Conversion completed in {:.2}s", convert_duration.as_secs_f64());
    
    // Phase 4: Ultra-optimized BPE merging with incremental updates
    info!("Phase 4: Ultra-optimized BPE merging with incremental updates");
    let merge_start = Instant::now();
    
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let target_merges = vocab_size - initial_vocab_size;
    
    info!("Performing {} merges with ultra-optimized incremental algorithm...", target_merges);
    
    // Use simple HashMap - avoid expensive IndexMap operations
    let mut word_token_counts: HashMap<Vec<usize>, u32> = word_token_counts;
    
    // Initial pair statistics with heap for efficient updates
    let mut pair_stats = compute_pair_statistics_simple(&word_token_counts);
    let mut pair_heap = create_pair_heap(&pair_stats, &vocab);
    
    for merge_idx in 0..target_merges {
        // Get best pair from heap
        let best_pair = match get_best_pair_from_heap(&mut pair_heap, &pair_stats) {
            Some(pair) => pair,
            None => {
                info!("No more pairs to merge at iteration {}", merge_idx);
                break;
            }
        };
        
        let new_token_id = initial_vocab_size + merge_idx;
        
        // Create new vocabulary entry
        let first_token = vocab[&best_pair.0].clone();
        let second_token = vocab[&best_pair.1].clone();
        let merged_token = [first_token.clone(), second_token.clone()].concat();
        
        vocab.insert(new_token_id, merged_token);
        merges.push((first_token, second_token));
        
        // Apply merge and update statistics incrementally with heap maintenance
        apply_merge_simple(&mut word_token_counts, &mut pair_stats, &mut pair_heap, &best_pair, new_token_id, &vocab);
        
        // Progress reporting
        if (merge_idx + 1) % 100 == 0 || merge_idx < 10 || (merge_idx + 1) % 1000 == 0 {
            let elapsed = merge_start.elapsed().as_secs_f64();
            let rate = (merge_idx + 1) as f64 / elapsed;
            let eta = (target_merges - merge_idx - 1) as f64 / rate;
            info!("Merge {}/{} ({:.1}%) - Rate: {:.1} merges/s - ETA: {:.1}s - Heap size: {}", 
                  merge_idx + 1, target_merges, 
                  (merge_idx + 1) as f64 / target_merges as f64 * 100.0,
                  rate, eta, pair_heap.len());
        }
    }
    
    // word_token_counts is already HashMap
    
    let merge_duration = merge_start.elapsed();
    info!("BPE merging completed in {:.2}s", merge_duration.as_secs_f64());
    info!("Performed {} merges", merges.len());
    
    // Phase 5: Save outputs
    info!("Phase 5: Saving outputs");
    let save_start = Instant::now();
    
    save_vocabulary(&vocab, &format!("{}/vocab.json", output_dir))?;
    save_merges(&merges, &format!("{}/merges.txt", output_dir))?;
    
    let total_duration = total_start.elapsed();
    save_performance_log(
        output_dir,
        file_size,
        pretokenize_duration,
        convert_duration,
        merge_duration,
        total_duration,
        vocab.len(),
        merges.len()
    )?;
    
    let save_duration = save_start.elapsed();
    info!("Outputs saved in {:.2}s", save_duration.as_secs_f64());
    
    info!("=== Training Complete ===");
    info!("Total time: {:.2}s", total_duration.as_secs_f64());
    info!("Final vocabulary size: {}", vocab.len());
    
    Ok(())
}

fn pretokenize_file(file_path: &str, file_size: u64) -> Result<HashMap<Vec<u8>, u32>, Box<dyn std::error::Error>> {
    // GPT-2 regex pattern (preserved exactly)
    let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let regex = Regex::new(pattern)?;
    
    if file_size > 100_000_000 { // 100MB threshold - use parallel chunk processing
        info!("Using parallel chunk-based processing for large file");
        parallel_pretokenize_file(file_path, &regex, file_size)
    } else {
        info!("Using in-memory processing for small file");
        let content = fs::read_to_string(file_path)?;
        let mut word_counts = HashMap::new();
        extract_tokens_from_text(&content, &regex, &mut word_counts);
        Ok(word_counts)
    }
}


fn find_safe_boundary(text: &str) -> usize {
    // Find a safe boundary to avoid splitting tokens
    if text.len() <= 1000 {
        return text.len();
    }
    
    // Find a safe UTF-8 start position for search
    let mut start_search = text.len() - 1000;
    while start_search > 0 && !text.is_char_boundary(start_search) {
        start_search -= 1;
    }
    
    // Search backwards for whitespace more efficiently
    let search_slice = &text[start_search..];
    if let Some(pos) = search_slice.rfind(char::is_whitespace) {
        return start_search + pos + 1; // Include the whitespace
    }
    
    // If no whitespace found, find a safe UTF-8 boundary at 90% of text
    let target_pos = (text.len() * 9) / 10;
    
    // Ensure we're at a valid UTF-8 character boundary
    for i in (0..=target_pos).rev() {
        if text.is_char_boundary(i) {
            return i;
        }
    }
    
    // Fallback to beginning if no valid boundary found
    0
}

fn extract_tokens_from_text(text: &str, regex: &Regex, counts: &mut HashMap<Vec<u8>, u32>) {
    let mut pos = 0;
    while pos < text.len() {
        let search_content = &text[pos..];
        if let Some((start, end)) = regex.find(search_content) {
            let actual_start = pos + start;
            let actual_end = pos + end;
            let token_bytes = text[actual_start..actual_end].as_bytes().to_vec();
            *counts.entry(token_bytes).or_insert(0) += 1;
            pos = actual_end;
        } else {
            break;
        }
    }
}

#[derive(Clone, Debug)]
struct PairEntry {
    pair: (usize, usize),
    count: u32,
    tie_breaker: (Vec<u8>, Vec<u8>), // For deterministic ordering
}

impl PartialEq for PairEntry {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.tie_breaker == other.tie_breaker
    }
}

impl Eq for PairEntry {}

impl PartialOrd for PairEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: by count (higher is better)
        match self.count.cmp(&other.count) {
            Ordering::Equal => {
                // Tie-breaker: lexicographically smaller bytes (deterministic)
                other.tie_breaker.cmp(&self.tie_breaker)
            }
            other => other,
        }
    }
}

fn create_pair_heap(pair_stats: &HashMap<(usize, usize), u32>, vocab: &HashMap<usize, Vec<u8>>) -> BinaryHeap<PairEntry> {
    pair_stats.iter().map(|(&pair, &count)| {
        let tie_breaker = (vocab[&pair.0].clone(), vocab[&pair.1].clone());
        PairEntry { pair, count, tie_breaker }
    }).collect()
}

fn get_best_pair_from_heap(
    heap: &mut BinaryHeap<PairEntry>, 
    pair_stats: &HashMap<(usize, usize), u32>
) -> Option<(usize, usize)> {
    while let Some(entry) = heap.pop() {
        // Verify the entry is still valid
        if let Some(&current_count) = pair_stats.get(&entry.pair) {
            if current_count == entry.count && current_count > 0 {
                return Some(entry.pair);
            }
        }
    }
    None
}

fn apply_merge_with_inverted_index(
    word_token_counts: &mut IndexMap<Vec<usize>, u32>,
    pair_stats: &mut HashMap<(usize, usize), u32>,
    pair_heap: &mut BinaryHeap<PairEntry>,
    pair_to_words: &mut HashMap<(usize, usize), Vec<usize>>,
    best_pair: &(usize, usize),
    new_token_id: usize,
    vocab: &HashMap<usize, Vec<u8>>,
) {
    // Use inverted index to get words containing the best pair - O(1) lookup!
    let affected_word_indices = pair_to_words.get(best_pair).cloned().unwrap_or_default();
    
    // Collect affected words by index
    let affected_words: Vec<(Vec<usize>, u32)> = affected_word_indices
        .iter()
        .filter_map(|&word_idx| {
            let (word, &count) = word_token_counts.get_index(word_idx)?;
            Some((word.clone(), count))
        })
        .collect();
    
    // Track pairs that will need heap updates
    let mut affected_pairs = HashSet::new();
    
    // Remove old statistics and update inverted index
    for (word, count) in &affected_words {
        // Remove this word's contribution to pair statistics and inverted index
        for i in 0..word.len().saturating_sub(1) {
            let pair = (word[i], word[i + 1]);
            affected_pairs.insert(pair);
            
            // Update pair statistics
            if let Some(stat) = pair_stats.get_mut(&pair) {
                *stat = stat.saturating_sub(*count);
                if *stat == 0 {
                    pair_stats.remove(&pair);
                }
            }
            
            // Update inverted index - remove word indices that no longer contain this pair
            if let Some(word_indices) = pair_to_words.get_mut(&pair) {
                word_indices.retain(|&idx| {
                    if let Some((existing_word, _)) = word_token_counts.get_index(idx) {
                        existing_word != word
                    } else {
                        false
                    }
                });
                if word_indices.is_empty() {
                    pair_to_words.remove(&pair);
                }
            }
        }
        
        // Remove the old word from word_token_counts
        word_token_counts.shift_remove(word);
    }
    
    // Apply merge and add new statistics
    for (word, count) in affected_words {
        let merged_word = merge_word_tokens(&word, best_pair, new_token_id);
        
        // Add new word's contribution to pair statistics and inverted index
        let new_word_idx = word_token_counts.len(); // Index where new word will be inserted
        for i in 0..merged_word.len().saturating_sub(1) {
            let pair = (merged_word[i], merged_word[i + 1]);
            affected_pairs.insert(pair);
            *pair_stats.entry(pair).or_insert(0) += count;
            
            // Update inverted index
            pair_to_words.entry(pair).or_insert_with(Vec::new).push(new_word_idx);
        }
        
        // Add the merged word
        *word_token_counts.entry(merged_word).or_insert(0) += count;
    }
    
    // Remove the merged pair from statistics and inverted index
    pair_stats.remove(best_pair);
    pair_to_words.remove(best_pair);
    affected_pairs.remove(best_pair);
    
    // Update heap with affected pairs
    for pair in affected_pairs {
        if let Some(&count) = pair_stats.get(&pair) {
            if count > 0 {
                let tie_breaker = (vocab[&pair.0].clone(), vocab[&pair.1].clone());
                pair_heap.push(PairEntry { pair, count, tie_breaker });
            }
        }
    }
}

fn apply_merge_with_heap_update(
    word_token_counts: &mut IndexMap<Vec<usize>, u32>,
    pair_stats: &mut HashMap<(usize, usize), u32>,
    pair_heap: &mut BinaryHeap<PairEntry>,
    best_pair: &(usize, usize),
    new_token_id: usize,
    vocab: &HashMap<usize, Vec<u8>>,
) {
    // Collect words that contain the best pair
    let affected_words: Vec<(Vec<usize>, u32)> = word_token_counts
        .iter()
        .filter(|(word, _)| {
            word.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1)
        })
        .map(|(word, &count)| (word.clone(), count))
        .collect();
    
    // Track pairs that will need heap updates
    let mut affected_pairs = HashSet::new();
    
    // Remove old statistics for affected words
    for (word, count) in &affected_words {
        // Collect pairs from this word that will change
        for i in 0..word.len().saturating_sub(1) {
            let pair = (word[i], word[i + 1]);
            affected_pairs.insert(pair);
            
            if let Some(stat) = pair_stats.get_mut(&pair) {
                *stat = stat.saturating_sub(*count);
                if *stat == 0 {
                    pair_stats.remove(&pair);
                }
            }
        }
        
        // Remove the old word from word_token_counts
        word_token_counts.shift_remove(word);
    }
    
    // Apply merge and add new statistics
    for (word, count) in affected_words {
        let merged_word = merge_word_tokens(&word, best_pair, new_token_id);
        
        // Add new word's contribution to pair statistics
        for i in 0..merged_word.len().saturating_sub(1) {
            let pair = (merged_word[i], merged_word[i + 1]);
            affected_pairs.insert(pair);
            *pair_stats.entry(pair).or_insert(0) += count;
        }
        
        // Add the merged word
        *word_token_counts.entry(merged_word).or_insert(0) += count;
    }
    
    // Remove the merged pair from statistics
    pair_stats.remove(best_pair);
    affected_pairs.remove(best_pair);
    
    // Update heap with affected pairs
    for pair in affected_pairs {
        if let Some(&count) = pair_stats.get(&pair) {
            if count > 0 {
                let tie_breaker = (vocab[&pair.0].clone(), vocab[&pair.1].clone());
                pair_heap.push(PairEntry { pair, count, tie_breaker });
            }
        }
    }
}

fn compute_pair_statistics_simple(word_token_counts: &HashMap<Vec<usize>, u32>) -> HashMap<(usize, usize), u32> {
    // Use parallel processing for large vocabularies
    if word_token_counts.len() > 10000 {
        let words_vec: Vec<_> = word_token_counts.iter().collect();
        words_vec
            .par_iter()
            .map(|(word, &count)| {
                let mut local_stats = HashMap::new();
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i], word[i + 1]);
                    *local_stats.entry(pair).or_insert(0) += count;
                }
                local_stats
            })
            .reduce(HashMap::new, |mut acc, local| {
                for (pair, count) in local {
                    *acc.entry(pair).or_insert(0) += count;
                }
                acc
            })
    } else {
        // Use single-threaded for small vocabularies
        let mut stats = HashMap::new();
        for (word, &count) in word_token_counts {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i], word[i + 1]);
                *stats.entry(pair).or_insert(0) += count;
            }
        }
        stats
    }
}

fn apply_merge_simple(
    word_token_counts: &mut HashMap<Vec<usize>, u32>,
    pair_stats: &mut HashMap<(usize, usize), u32>,
    pair_heap: &mut BinaryHeap<PairEntry>,
    best_pair: &(usize, usize),
    new_token_id: usize,
    vocab: &HashMap<usize, Vec<u8>>,
) {
    // Collect words that contain the best pair
    let affected_words: Vec<(Vec<usize>, u32)> = word_token_counts
        .iter()
        .filter(|(word, _)| {
            word.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1)
        })
        .map(|(word, &count)| (word.clone(), count))
        .collect();
    
    // Track pairs that will need heap updates
    let mut affected_pairs = HashSet::new();
    
    // Remove old statistics for affected words
    for (word, count) in &affected_words {
        // Remove this word's contribution to pair statistics
        for i in 0..word.len().saturating_sub(1) {
            let pair = (word[i], word[i + 1]);
            affected_pairs.insert(pair);
            
            if let Some(stat) = pair_stats.get_mut(&pair) {
                *stat = stat.saturating_sub(*count);
                if *stat == 0 {
                    pair_stats.remove(&pair);
                }
            }
        }
        
        // Remove the old word from word_token_counts
        word_token_counts.remove(word);
    }
    
    // Apply merge and add new statistics
    for (word, count) in affected_words {
        let merged_word = merge_word_tokens(&word, best_pair, new_token_id);
        
        // Add new word's contribution to pair statistics
        for i in 0..merged_word.len().saturating_sub(1) {
            let pair = (merged_word[i], merged_word[i + 1]);
            affected_pairs.insert(pair);
            *pair_stats.entry(pair).or_insert(0) += count;
        }
        
        // Add the merged word
        *word_token_counts.entry(merged_word).or_insert(0) += count;
    }
    
    // Remove the merged pair from statistics
    pair_stats.remove(best_pair);
    affected_pairs.remove(best_pair);
    
    // Update heap with affected pairs
    for pair in affected_pairs {
        if let Some(&count) = pair_stats.get(&pair) {
            if count > 0 {
                let tie_breaker = (vocab[&pair.0].clone(), vocab[&pair.1].clone());
                pair_heap.push(PairEntry { pair, count, tie_breaker });
            }
        }
    }
}

fn compute_pair_statistics_optimized(word_token_counts: &IndexMap<Vec<usize>, u32>) -> HashMap<(usize, usize), u32> {
    // Use parallel processing for large vocabularies
    if word_token_counts.len() > 10000 {
        let words_vec: Vec<_> = word_token_counts.iter().collect();
        words_vec
            .par_iter()
            .map(|(word, &count)| {
                let mut local_stats = HashMap::new();
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i], word[i + 1]);
                    *local_stats.entry(pair).or_insert(0) += count;
                }
                local_stats
            })
            .reduce(HashMap::new, |mut acc, local| {
                for (pair, count) in local {
                    *acc.entry(pair).or_insert(0) += count;
                }
                acc
            })
    } else {
        // Use single-threaded for small vocabularies
        let mut stats = HashMap::new();
        for (word, &count) in word_token_counts {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i], word[i + 1]);
                *stats.entry(pair).or_insert(0) += count;
            }
        }
        stats
    }
}

fn apply_merge_optimized(
    word_token_counts: &mut IndexMap<Vec<usize>, u32>,
    pair_stats: &mut HashMap<(usize, usize), u32>,
    best_pair: &(usize, usize),
    new_token_id: usize,
) {
    // Collect words that contain the best pair
    let affected_words: Vec<(Vec<usize>, u32)> = word_token_counts
        .iter()
        .filter(|(word, _)| {
            word.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1)
        })
        .map(|(word, &count)| (word.clone(), count))
        .collect();
    
    // Remove old statistics for affected words
    for (word, count) in &affected_words {
        // Remove this word's contribution to pair statistics
        for i in 0..word.len().saturating_sub(1) {
            let pair = (word[i], word[i + 1]);
            if let Some(stat) = pair_stats.get_mut(&pair) {
                *stat = stat.saturating_sub(*count);
                if *stat == 0 {
                    pair_stats.remove(&pair);
                }
            }
        }
        
        // Remove the old word from word_token_counts
        word_token_counts.shift_remove(word);
    }
    
    // Apply merge and add new statistics
    for (word, count) in affected_words {
        let merged_word = merge_word_tokens(&word, best_pair, new_token_id);
        
        // Add new word's contribution to pair statistics
        for i in 0..merged_word.len().saturating_sub(1) {
            let pair = (merged_word[i], merged_word[i + 1]);
            *pair_stats.entry(pair).or_insert(0) += count;
        }
        
        // Add the merged word
        *word_token_counts.entry(merged_word).or_insert(0) += count;
    }
    
    // Remove the merged pair from statistics
    pair_stats.remove(best_pair);
}

fn find_best_pair(stats: &HashMap<(usize, usize), u32>, vocab: &HashMap<usize, Vec<u8>>) -> (usize, usize) {
    stats
        .iter()
        .max_by_key(|(&pair, &count)| {
            // Tie-breaking: prefer lexicographically smaller bytes for determinism
            (count, std::cmp::Reverse((&vocab[&pair.0], &vocab[&pair.1])))
        })
        .map(|(&pair, _)| pair)
        .expect("No pairs found")
}


fn merge_word_tokens(word: &[usize], target_pair: &(usize, usize), new_token_id: usize) -> Vec<usize> {
    if word.len() < 2 {
        return word.to_vec();
    }
    
    let mut result = Vec::new();
    let mut i = 0;
    
    while i < word.len() {
        if i < word.len() - 1 && word[i] == target_pair.0 && word[i + 1] == target_pair.1 {
            // Found the target pair, replace with new token
            result.push(new_token_id);
            i += 2; // Skip both tokens in the pair
        } else {
            // Keep the current token
            result.push(word[i]);
            i += 1;
        }
    }
    
    result
}

fn save_vocabulary(vocab: &HashMap<usize, Vec<u8>>, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_path)?;
    
    writeln!(file, "{{")?;
    let mut entries: Vec<_> = vocab.iter().collect();
    entries.sort_by_key(|(id, _)| *id);
    
    for (i, (id, token_bytes)) in entries.iter().enumerate() {
        let token_str = String::from_utf8_lossy(token_bytes);
        let comma = if i == entries.len() - 1 { "" } else { "," };
        writeln!(file, "  \"{}\": \"{}\"{}",  id, token_str.escape_debug(), comma)?;
    }
    writeln!(file, "}}")?;
    
    Ok(())
}

fn save_merges(merges: &[(Vec<u8>, Vec<u8>)], file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_path)?;
    
    for (first, second) in merges {
        let first_str = String::from_utf8_lossy(first);
        let second_str = String::from_utf8_lossy(second);
        writeln!(file, "{} {}", first_str, second_str)?;
    }
    
    Ok(())
}

fn parallel_pretokenize_file(
    file_path: &str,
    regex: &Regex,
    file_size: u64,
) -> Result<HashMap<Vec<u8>, u32>, Box<dyn std::error::Error>> {
    use std::sync::Mutex;
    
    const CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB chunks for better parallelization
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    
    let mut chunks = Vec::new();
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut overlap_buffer = String::new();
    let mut total_processed = 0u64;
    
    info!("Reading file into {}MB chunks for parallel processing", CHUNK_SIZE / 1024 / 1024);
    
    // Read file into chunks
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            if !overlap_buffer.is_empty() {
                chunks.push(overlap_buffer.clone());
            }
            break;
        }
        
        let chunk_str = match std::str::from_utf8(&buffer[..bytes_read]) {
            Ok(s) => s,
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                std::str::from_utf8(&buffer[..valid_up_to])?
            }
        };
        
        let full_text = if overlap_buffer.is_empty() {
            chunk_str.to_string()
        } else {
            overlap_buffer.clone() + chunk_str
        };
        
        // Find safe boundary at <|endoftext|> markers when possible, ensuring UTF-8 safety
        let safe_boundary = find_endoftext_boundary(&full_text)
            .unwrap_or_else(|| find_safe_boundary(&full_text));
        
        // Ensure safe_boundary is at a valid UTF-8 character boundary
        let safe_boundary = ensure_char_boundary(&full_text, safe_boundary);
        let (process_text, remaining) = full_text.split_at(safe_boundary);
        
        chunks.push(process_text.to_string());
        overlap_buffer = remaining.to_string();
        
        total_processed += bytes_read as u64;
    }
    
    info!("Split file into {} chunks, processing in parallel", chunks.len());
    
    // Process chunks in parallel
    let word_counts = Arc::new(Mutex::new(HashMap::new()));
    let processed_chunks = Arc::new(Mutex::new(0));
    
    chunks.par_iter().for_each(|chunk| {
        let mut local_counts = HashMap::new();
        extract_tokens_from_text(chunk, regex, &mut local_counts);
        
        // Merge local counts into global counts
        {
            let mut global_counts = word_counts.lock().unwrap();
            for (token, count) in local_counts {
                *global_counts.entry(token).or_insert(0) += count;
            }
        }
        
        // Progress tracking
        {
            let mut processed = processed_chunks.lock().unwrap();
            *processed += 1;
            if *processed % 10 == 0 || *processed == chunks.len() {
                let percent = (*processed as f64 / chunks.len() as f64) * 100.0;
                info!("Processed {}/{} chunks ({:.1}%)", *processed, chunks.len(), percent);
            }
        }
    });
    
    let final_counts = Arc::try_unwrap(word_counts).unwrap().into_inner().unwrap();
    info!("Parallel processing complete. Found {} unique tokens", final_counts.len());
    
    Ok(final_counts)
}

fn find_endoftext_boundary(text: &str) -> Option<usize> {
    // Look for <|endoftext|> markers in the last 10% of text for optimal chunking
    let mut start_search = (text.len() * 9) / 10;
    
    // Ensure start_search is at a valid char boundary
    while start_search > 0 && !text.is_char_boundary(start_search) {
        start_search -= 1;
    }
    
    if let Some(pos) = text[start_search..].find("<|endoftext|>") {
        Some(start_search + pos + "<|endoftext|>".len())
    } else {
        None
    }
}

fn ensure_char_boundary(text: &str, mut boundary: usize) -> usize {
    // Ensure boundary is at a valid UTF-8 character boundary
    if boundary >= text.len() {
        return text.len();
    }
    
    // Move boundary backward until we find a valid char boundary
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    
    boundary
}

fn save_performance_log(
    output_dir: &str,
    file_size: u64,
    pretokenize_duration: std::time::Duration,
    convert_duration: std::time::Duration,
    merge_duration: std::time::Duration,
    total_duration: std::time::Duration,
    final_vocab_size: usize,
    num_merges: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let log_path = format!("{}/performance.log", output_dir);
    let mut file = File::create(&log_path)?;
    
    writeln!(file, "=== BPE Training Performance Report ===")?;
    writeln!(file, "File size: {:.1} GB", file_size as f64 / 1e9)?;
    writeln!(file, "Final vocabulary size: {}", final_vocab_size)?;
    writeln!(file, "Number of merges: {}", num_merges)?;
    writeln!(file, "")?;
    writeln!(file, "=== Timing Breakdown ===")?;
    writeln!(file, "Pre-tokenization: {:.2}s ({:.1}%)", 
             pretokenize_duration.as_secs_f64(),
             pretokenize_duration.as_secs_f64() / total_duration.as_secs_f64() * 100.0)?;
    writeln!(file, "Token conversion: {:.2}s ({:.1}%)", 
             convert_duration.as_secs_f64(),
             convert_duration.as_secs_f64() / total_duration.as_secs_f64() * 100.0)?;
    writeln!(file, "BPE merging: {:.2}s ({:.1}%)", 
             merge_duration.as_secs_f64(),
             merge_duration.as_secs_f64() / total_duration.as_secs_f64() * 100.0)?;
    writeln!(file, "Total time: {:.2}s", total_duration.as_secs_f64())?;
    writeln!(file, "")?;
    writeln!(file, "=== Performance Metrics ===")?;
    writeln!(file, "Pre-tokenization throughput: {:.1} GB/hour", 
             file_size as f64 / 1e9 / pretokenize_duration.as_secs_f64() * 3600.0)?;
    writeln!(file, "BPE merge rate: {:.1} merges/second", 
             num_merges as f64 / merge_duration.as_secs_f64())?;
    
    info!("Performance log saved to: {}", log_path);
    Ok(())
}