use clap::Parser;
use onig::Regex;
use std::collections::{HashMap, BTreeMap};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;
use log::{info, error};
use rayon::prelude::*;
use indexmap::IndexMap;
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(name = "bpe_trainer")]
#[command(about = "Train a Byte Pair Encoding (BPE) tokenizer")]
#[command(version = "2.0")]
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
    
    println!("🦀 Optimized Rust BPE Trainer v2.0");
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
    
    info!("=== Starting Optimized BPE Training v2.0 ===");
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
    
    // Use BTreeMap for better cache locality with large vocabularies
    let mut word_token_counts: BTreeMap<Vec<usize>, u32> = BTreeMap::new();
    for (word_bytes, count) in word_counts {
        let token_sequence: Vec<usize> = word_bytes.iter().map(|&b| b as usize).collect();
        word_token_counts.insert(token_sequence, count);
    }
    
    let convert_duration = convert_start.elapsed();
    info!("Conversion completed in {:.2}s", convert_duration.as_secs_f64());
    
    // Phase 4: Ultra-optimized BPE merging
    info!("Phase 4: Ultra-optimized BPE merging");
    let merge_start = Instant::now();
    
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let target_merges = vocab_size - initial_vocab_size;
    
    info!("Performing {} merges with ultra-optimized algorithm...", target_merges);
    
    // Convert to Arc<Mutex> for thread-safe access
    let word_token_counts = Arc::new(Mutex::new(word_token_counts));
    
    for merge_idx in 0..target_merges {
        // Compute pair statistics in parallel with batching
        let pair_stats = compute_pair_statistics_ultra(&word_token_counts);
        
        if pair_stats.is_empty() {
            info!("No more pairs to merge at iteration {}", merge_idx);
            break;
        }
        
        // Find best pair
        let best_pair = find_best_pair(&pair_stats, &vocab);
        let new_token_id = initial_vocab_size + merge_idx;
        
        // Create new vocabulary entry
        let first_token = vocab[&best_pair.0].clone();
        let second_token = vocab[&best_pair.1].clone();
        let merged_token = [first_token.clone(), second_token.clone()].concat();
        
        vocab.insert(new_token_id, merged_token);
        merges.push((first_token, second_token));
        
        // Apply merge in parallel
        apply_merge_ultra(&word_token_counts, &best_pair, new_token_id);
        
        // Progress reporting
        if (merge_idx + 1) % 100 == 0 || merge_idx < 10 || (merge_idx + 1) % 1000 == 0 {
            let elapsed = merge_start.elapsed().as_secs_f64();
            let rate = (merge_idx + 1) as f64 / elapsed;
            let eta = (target_merges - merge_idx - 1) as f64 / rate;
            info!("Merge {}/{} ({:.1}%) - Rate: {:.1} merges/s - ETA: {:.1}s", 
                  merge_idx + 1, target_merges, 
                  (merge_idx + 1) as f64 / target_merges as f64 * 100.0,
                  rate, eta);
        }
    }
    
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

fn compute_pair_statistics_ultra(
    word_token_counts: &Arc<Mutex<BTreeMap<Vec<usize>, u32>>>
) -> HashMap<(usize, usize), u32> {
    let words = word_token_counts.lock().unwrap();
    
    // Process in parallel chunks for better performance
    const CHUNK_SIZE: usize = 10000;
    let words_vec: Vec<_> = words.iter().map(|(w, c)| (w.clone(), *c)).collect();
    
    words_vec
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut local_stats = HashMap::new();
            for (word, count) in chunk {
                for i in 0..word.len().saturating_sub(1) {
                    let pair = (word[i], word[i + 1]);
                    *local_stats.entry(pair).or_insert(0) += count;
                }
            }
            local_stats
        })
        .reduce(HashMap::new, |mut acc, local| {
            for (pair, count) in local {
                *acc.entry(pair).or_insert(0) += count;
            }
            acc
        })
}

fn apply_merge_ultra(
    word_token_counts: &Arc<Mutex<BTreeMap<Vec<usize>, u32>>>,
    best_pair: &(usize, usize),
    new_token_id: usize,
) {
    let mut words = word_token_counts.lock().unwrap();
    
    // Collect affected words
    let affected_words: Vec<(Vec<usize>, u32)> = words
        .iter()
        .filter(|(word, _)| {
            word.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1)
        })
        .map(|(word, &count)| (word.clone(), count))
        .collect();
    
    // Process merges in parallel
    let merged_words: Vec<(Vec<usize>, u32)> = affected_words
        .par_iter()
        .map(|(word, count)| {
            (merge_word_tokens(word, best_pair, new_token_id), *count)
        })
        .collect();
    
    // Remove old words and add merged ones
    for (word, _) in &affected_words {
        words.remove(word);
    }
    
    for (merged_word, count) in merged_words {
        *words.entry(merged_word).or_insert(0) += count;
    }
}

fn find_best_pair(stats: &HashMap<(usize, usize), u32>, vocab: &HashMap<usize, Vec<u8>>) -> (usize, usize) {
    stats
        .iter()
        .max_by_key(|(&pair, &count)| {
            (count, std::cmp::Reverse((&vocab[&pair.0], &vocab[&pair.1])))
        })
        .map(|(&pair, _)| pair)
        .expect("No pairs found")
}

fn merge_word_tokens(word: &[usize], target_pair: &(usize, usize), new_token_id: usize) -> Vec<usize> {
    if word.len() < 2 {
        return word.to_vec();
    }
    
    let mut result = Vec::with_capacity(word.len());
    let mut i = 0;
    
    while i < word.len() {
        if i < word.len() - 1 && word[i] == target_pair.0 && word[i + 1] == target_pair.1 {
            result.push(new_token_id);
            i += 2;
        } else {
            result.push(word[i]);
            i += 1;
        }
    }
    
    result
}

fn pretokenize_file(file_path: &str, file_size: u64) -> Result<HashMap<Vec<u8>, u32>, Box<dyn std::error::Error>> {
    // GPT-2 regex pattern (preserved exactly)
    let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let regex = Regex::new(pattern)?;
    
    let mut word_counts: HashMap<Vec<u8>, u32> = HashMap::new();
    
    if file_size > 100_000_000 { // 100MB threshold - use chunk processing
        info!("Using chunk-based processing for large file");
        let file = File::open(file_path)?;
        process_file_in_chunks(file, &regex, &mut word_counts, file_size)?;
    } else {
        info!("Using in-memory processing for small file");
        let content = fs::read_to_string(file_path)?;
        extract_tokens_from_text(&content, &regex, &mut word_counts);
    }
    
    Ok(word_counts)
}

fn process_file_in_chunks(
    file: File,
    regex: &Regex,
    word_counts: &mut HashMap<Vec<u8>, u32>,
    file_size: u64
) -> Result<(), Box<dyn std::error::Error>> {
    const CHUNK_SIZE: usize = 8 * 1024 * 1024; // 8MB chunks
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut overlap_buffer = String::new();
    let mut total_processed = 0u64;
    let mut last_progress_gb = 0;
    
    info!("Starting chunk-based processing with {}MB chunks", CHUNK_SIZE / 1024 / 1024);
    
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        
        if bytes_read == 0 {
            if !overlap_buffer.is_empty() {
                extract_tokens_from_text(&overlap_buffer, regex, word_counts);
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
        
        let safe_boundary = find_safe_boundary(&full_text);
        let (process_text, remaining) = full_text.split_at(safe_boundary);
        
        extract_tokens_from_text(process_text, regex, word_counts);
        
        overlap_buffer = remaining.to_string();
        
        total_processed += bytes_read as u64;
        let current_gb = total_processed / (1024 * 1024 * 1024);
        
        if current_gb > last_progress_gb {
            let percent = (total_processed as f64 / file_size as f64) * 100.0;
            info!("Processed {:.1}GB / {:.1}GB ({:.1}%) - {} unique tokens so far", 
                  total_processed as f64 / 1e9, file_size as f64 / 1e9, percent, word_counts.len());
            last_progress_gb = current_gb;
        }
    }
    
    info!("Chunk processing complete. Processed {:.1}GB total", total_processed as f64 / 1e9);
    Ok(())
}

fn find_safe_boundary(text: &str) -> usize {
    if text.len() <= 1000 {
        return text.len();
    }
    
    let mut start_search = text.len() - 1000;
    while start_search > 0 && !text.is_char_boundary(start_search) {
        start_search -= 1;
    }
    
    let search_slice = &text[start_search..];
    if let Some(pos) = search_slice.rfind(char::is_whitespace) {
        return start_search + pos + 1;
    }
    
    let target_pos = (text.len() * 9) / 10;
    
    for i in (0..=target_pos).rev() {
        if text.is_char_boundary(i) {
            return i;
        }
    }
    
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

fn save_vocabulary(vocab: &HashMap<usize, Vec<u8>>, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_path)?;
    
    writeln!(file, "{{")?;
    let mut entries: Vec<_> = vocab.iter().collect();
    entries.sort_by_key(|(id, _)| *id);
    
    for (i, (id, token_bytes)) in entries.iter().enumerate() {
        let token_str = String::from_utf8_lossy(token_bytes);
        let comma = if i == entries.len() - 1 { "" } else { "," };
        writeln!(file, "  \"{}\": \"{}\"{}", id, token_str.escape_debug(), comma)?;
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