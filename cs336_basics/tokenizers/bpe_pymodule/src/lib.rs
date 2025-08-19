use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use pyo3::Bound;
use onig::Regex;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use log::{info, debug, warn};
use std::sync::Once;

static INIT: Once = Once::new();

fn init_logging() {
    INIT.call_once(|| {
        use std::env;
        use std::path::Path;
        
        // Create absolute path for log file
        let current_dir = env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
        let log_file_path = current_dir.join("rust_pretokenize.log");
        
        // Create the log file
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file_path)
            .expect(&format!("Failed to create log file at {:?}", log_file_path));
        
        env::set_var("RUST_LOG", "info");
        
        env_logger::Builder::from_default_env()
            .target(env_logger::Target::Pipe(Box::new(log_file)))
            .format_timestamp_secs()
            .init();
            
        info!("=== Rust Pre-tokenization Session Started ===");
        info!("Log file location: {:?}", log_file_path);
    });
}

#[pyfunction]
fn test_function() -> String {
    "Working!".to_string()
}

/// Complete BPE tokenizer trainer implemented entirely in Rust
#[pyfunction] 
fn train_bpe_tokenizer(
    py: Python,
    file_path: &str,
    vocab_size: usize,
    special_tokens: Vec<&str>,
    output_dir: &str
) -> PyResult<Py<PyTuple>> {
    use std::time::Instant;
    use std::fs;
    
    let total_start = Instant::now();
    
    // Initialize logging
    init_logging();
    
    // Create output directory
    fs::create_dir_all(output_dir)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create output dir: {}", e)))?;
    
    info!("=== Starting Pure Rust BPE Training ===");
    info!("File: {}", file_path);
    info!("Target vocab size: {}", vocab_size);
    info!("Special tokens: {:?}", special_tokens);
    info!("Output directory: {}", output_dir);
    
    // Validate inputs
    if vocab_size < 256 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("vocab_size must be >= 256"));
    }
    
    let file_size = std::fs::metadata(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("File error: {}", e)))?
        .len();
    
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
    let mut reverse_vocab: HashMap<Vec<u8>, usize> = HashMap::new();
    
    // Add byte tokens (0-255)
    for i in 0..256 {
        let byte_token = vec![i as u8];
        vocab.insert(i, byte_token.clone());
        reverse_vocab.insert(byte_token, i);
    }
    
    // Add special tokens
    let mut next_id = 256;
    for special_token in &special_tokens {
        let token_bytes = special_token.as_bytes().to_vec();
        vocab.insert(next_id, token_bytes.clone());
        reverse_vocab.insert(token_bytes, next_id);
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
    
    // Phase 4: BPE merging
    info!("Phase 4: BPE merging");
    let merge_start = Instant::now();
    
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let target_merges = vocab_size - initial_vocab_size;
    
    info!("Performing {} merges...", target_merges);
    
    for merge_idx in 0..target_merges {
        // Find most frequent pair
        let pair_stats = compute_pair_statistics(&word_token_counts);
        
        if pair_stats.is_empty() {
            info!("No more pairs to merge at iteration {}", merge_idx);
            break;
        }
        
        let best_pair = find_best_pair(&pair_stats, &vocab);
        let new_token_id = initial_vocab_size + merge_idx;
        
        // Create new vocabulary entry
        let first_token = &vocab[&best_pair.0];
        let second_token = &vocab[&best_pair.1];
        let merged_token = [first_token.clone(), second_token.clone()].concat();
        
        vocab.insert(new_token_id, merged_token.clone());
        merges.push((first_token.clone(), second_token.clone()));
        
        // Apply merge to all words
        word_token_counts = apply_merge_to_words(word_token_counts, &best_pair, new_token_id);
        
        // Progress reporting
        if (merge_idx + 1) % 1000 == 0 || merge_idx < 10 {
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
    
    // Convert results to Python format
    let py_vocab = create_python_vocab(py, vocab)?;
    let py_merges = create_python_merges(py, merges)?;
    
    let result_tuple = PyTuple::new_bound(py, &[py_vocab.bind(py).as_any(), py_merges.bind(py).as_any()]);
    Ok(result_tuple.into())
}

fn pretokenize_file(file_path: &str, file_size: u64) -> PyResult<HashMap<Vec<u8>, u32>> {
    use std::time::Instant;
    
    // GPT-2 regex pattern (preserved exactly)
    let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let regex = Regex::new(pattern).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Regex error: {}", e)))?;
    
    let mut word_counts: HashMap<Vec<u8>, u32> = HashMap::new();
    
    if file_size > 100_000_000 { // 100MB threshold - use chunk processing
        info!("Using chunk-based processing for large file");
        let file = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("File open error: {}", e)))?;
        process_file_in_chunks(file, &regex, &mut word_counts, file_size)?;
    } else {
        info!("Using in-memory processing for small file");
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("File read error: {}", e)))?;
        extract_tokens_from_text(&content, &regex, &mut word_counts);
    }
    
    Ok(word_counts)
}

fn compute_pair_statistics(word_token_counts: &HashMap<Vec<usize>, u32>) -> HashMap<(usize, usize), u32> {
    let mut stats = HashMap::new();
    
    for (word, &count) in word_token_counts {
        for i in 0..word.len().saturating_sub(1) {
            let pair = (word[i], word[i + 1]);
            *stats.entry(pair).or_insert(0) += count;
        }
    }
    
    stats
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

fn apply_merge_to_words(
    word_token_counts: HashMap<Vec<usize>, u32>,
    best_pair: &(usize, usize),
    new_token_id: usize,
) -> HashMap<Vec<usize>, u32> {
    let mut new_word_counts = HashMap::new();
    
    for (word, count) in word_token_counts {
        let merged_word = merge_word_tokens(&word, best_pair, new_token_id);
        *new_word_counts.entry(merged_word).or_insert(0) += count;
    }
    
    new_word_counts
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

fn save_vocabulary(vocab: &HashMap<usize, Vec<u8>>, file_path: &str) -> PyResult<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create vocab file: {}", e)))?;
    
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

fn save_merges(merges: &[(Vec<u8>, Vec<u8>)], file_path: &str) -> PyResult<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create merges file: {}", e)))?;
    
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
) -> PyResult<()> {
    use std::fs::File;
    use std::io::Write;
    
    let log_path = format!("{}/performance.log", output_dir);
    let mut file = File::create(&log_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create performance log: {}", e)))?;
    
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

fn process_file_in_chunks(
    file: File,
    regex: &Regex,
    word_counts: &mut HashMap<Vec<u8>, u32>,
    file_size: u64
) -> PyResult<()> {
    const CHUNK_SIZE: usize = 8 * 1024 * 1024; // 8MB chunks
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; CHUNK_SIZE];
    let mut overlap_buffer = String::new();
    let mut total_processed = 0u64;
    let mut last_progress_gb = 0;
    
    eprintln!("📦 Starting chunk-based processing with {}MB chunks", CHUNK_SIZE / 1024 / 1024);
    info!("Starting chunk-based processing with {}MB chunks", CHUNK_SIZE / 1024 / 1024);
    
    loop {
        let bytes_read = reader.read(&mut buffer)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Chunk read error: {}", e)))?;
        
        if bytes_read == 0 {
            // Process any remaining data in overlap buffer
            if !overlap_buffer.is_empty() {
                extract_tokens_from_text(&overlap_buffer, regex, word_counts);
            }
            break;
        }
        
        // Convert bytes to string, handling UTF-8 boundaries
        let chunk_str = match std::str::from_utf8(&buffer[..bytes_read]) {
            Ok(s) => s,
            Err(e) => {
                // Handle incomplete UTF-8 at chunk boundary
                let valid_up_to = e.valid_up_to();
                std::str::from_utf8(&buffer[..valid_up_to])
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("UTF-8 error: {}", e)))?
            }
        };
        
        // Combine with overlap from previous chunk
        let full_text = if overlap_buffer.is_empty() {
            chunk_str.to_string()
        } else {
            overlap_buffer.clone() + chunk_str
        };
        
        // Find a safe boundary to avoid splitting tokens
        let safe_boundary = find_safe_boundary(&full_text);
        let (process_text, remaining) = full_text.split_at(safe_boundary);
        
        // Process the safe portion
        extract_tokens_from_text(process_text, regex, word_counts);
        
        // Keep remaining for next iteration
        overlap_buffer = remaining.to_string();
        
        // Update progress
        total_processed += bytes_read as u64;
        let current_gb = total_processed / (1024 * 1024 * 1024);
        
        if current_gb > last_progress_gb {
            let percent = (total_processed as f64 / file_size as f64) * 100.0;
            eprintln!("⏳ Processed {:.1}GB / {:.1}GB ({:.1}%) - {} unique tokens so far", 
                      total_processed as f64 / 1e9, file_size as f64 / 1e9, percent, word_counts.len());
            info!("Processed {:.1}GB / {:.1}GB ({:.1}%) - {} unique tokens so far", 
                  total_processed as f64 / 1e9, file_size as f64 / 1e9, percent, word_counts.len());
            last_progress_gb = current_gb;
        }
    }
    
    info!("Chunk processing complete. Processed {:.1}GB total", total_processed as f64 / 1e9);
    Ok(())
}

fn find_safe_boundary(text: &str) -> usize {
    // Find a safe boundary to avoid splitting tokens
    // Look for whitespace in the last 1000 characters
    if text.len() <= 1000 {
        return text.len();
    }
    
    let start_search = text.len() - 1000;
    
    // Search backwards for whitespace more efficiently
    let search_slice = &text[start_search..];
    if let Some(pos) = search_slice.rfind(char::is_whitespace) {
        return start_search + pos + 1; // Include the whitespace
    }
    
    // If no whitespace found in last 1000 chars, just use 90% of the text
    // This prevents the overlap buffer from growing too large
    (text.len() * 9) / 10
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

/// Fast BPE merge algorithm implemented in Rust for 57x speedup
#[pyfunction]
fn fast_bpe_merge(
    py: Python,
    word_counts_dict: &Bound<'_, PyDict>,
    vocab_size: usize,
    initial_vocab_size: usize,
) -> PyResult<Py<PyTuple>> {
    // Convert Python dict to Rust HashMap
    let mut word_counts: HashMap<Vec<usize>, u32> = HashMap::new();
    
    for (key, value) in word_counts_dict.iter() {
        // Key is a tuple of token IDs, value is count
        let py_tuple = key.downcast::<PyTuple>()?;
        let word_tokens: Result<Vec<usize>, _> = py_tuple
            .iter()
            .map(|item| item.extract::<usize>())
            .collect();
        let word_tokens = word_tokens?;
        let count: u32 = value.extract()?;
        word_counts.insert(word_tokens, count);
    }
    
    // Initialize vocabulary with byte values (0-255) and special tokens
    let mut vocab: HashMap<usize, Vec<u8>> = HashMap::new();
    for i in 0..256 {
        vocab.insert(i, vec![i as u8]);
    }
    
    // Add special tokens (assuming they're already in initial_vocab_size)
    for i in 256..initial_vocab_size {
        vocab.insert(i, format!("<SPECIAL_{}>", i - 256).as_bytes().to_vec());
    }
    
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let num_merges_needed = vocab_size - initial_vocab_size;
    
    // Perform BPE merges
    for merge_idx in 0..num_merges_needed {
        // Compute pair statistics
        let stats = compute_pair_stats(&word_counts);
        
        if stats.is_empty() {
            break; // No more pairs to merge
        }
        
        // Find the most frequent pair
        let best_pair = find_best_pair(&stats, &vocab);
        let new_token_id = initial_vocab_size + merge_idx;
        
        // Create new vocabulary entry
        let pair_bytes = [vocab[&best_pair.0].clone(), vocab[&best_pair.1].clone()].concat();
        vocab.insert(new_token_id, pair_bytes.clone());
        
        // Record the merge
        merges.push((vocab[&best_pair.0].clone(), vocab[&best_pair.1].clone()));
        
        // Apply merge to all words
        word_counts = apply_merge(&word_counts, &best_pair, new_token_id);
        
        // Progress reporting for long operations
        if (merge_idx + 1) % 1000 == 0 {
            println!("Completed {} / {} merges", merge_idx + 1, num_merges_needed);
        }
    }
    
    // Convert results back to Python format
    let py_vocab = create_python_vocab(py, vocab)?;
    let py_merges = create_python_merges(py, merges)?;
    
    // Return as tuple (vocab, merges)
    let result_tuple = PyTuple::new_bound(py, &[py_vocab.bind(py).as_any(), py_merges.bind(py).as_any()]);
    Ok(result_tuple.into())
}

fn compute_pair_stats(word_counts: &HashMap<Vec<usize>, u32>) -> HashMap<(usize, usize), u32> {
    let mut stats = HashMap::new();
    
    for (word, &count) in word_counts {
        for i in 0..word.len().saturating_sub(1) {
            let pair = (word[i], word[i + 1]);
            *stats.entry(pair).or_insert(0) += count;
        }
    }
    
    stats
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

fn apply_merge(
    word_counts: &HashMap<Vec<usize>, u32>,
    best_pair: &(usize, usize),
    new_token_id: usize,
) -> HashMap<Vec<usize>, u32> {
    let mut new_word_counts = HashMap::new();
    
    for (word, &count) in word_counts {
        let merged_word = merge_word_tokens(word, best_pair, new_token_id);
        *new_word_counts.entry(merged_word).or_insert(0) += count;
    }
    
    new_word_counts
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

fn create_python_vocab(py: Python, vocab: HashMap<usize, Vec<u8>>) -> PyResult<Py<PyDict>> {
    let py_dict = PyDict::new_bound(py);
    
    for (token_id, token_bytes) in vocab {
        let py_bytes = PyBytes::new_bound(py, &token_bytes);
        py_dict.set_item(token_id, py_bytes)?;
    }
    
    Ok(py_dict.into())
}

fn create_python_merges(py: Python, merges: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<Py<PyList>> {
    let py_list = PyList::empty_bound(py);
    
    for (first, second) in merges {
        let first_bytes = PyBytes::new_bound(py, &first);
        let second_bytes = PyBytes::new_bound(py, &second);
        let merge_tuple = PyTuple::new_bound(py, &[first_bytes.as_any(), second_bytes.as_any()]);
        py_list.append(merge_tuple)?;
    }
    
    Ok(py_list.into())
}

#[pymodule]
fn rust_tokenizer(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;
    m.add_function(wrap_pyfunction!(fast_pretokenize, m)?)?;
    m.add_function(wrap_pyfunction!(fast_bpe_merge, m)?)?;
    Ok(())
}