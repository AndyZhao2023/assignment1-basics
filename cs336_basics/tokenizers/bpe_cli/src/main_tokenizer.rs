mod tokenizer_optimized;
mod save_optimized;

use clap::{Parser, Subcommand};
use std::fs;
use std::path::Path;
use std::time::Instant;
use tokenizer_optimized::OptimizedBpeTokenizer;
use save_optimized::save_tokens_as_npy_optimized;

#[derive(Parser)]
#[command(name = "bpe_trainer_optimized")]
#[command(version = "2.0")]
#[command(about = "Optimized BPE tokenizer with tiktoken-inspired performance improvements")]
struct Args {
    #[command(subcommand)]
    command: Commands,
    
    #[arg(short, long, help = "Enable verbose output")]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    Tokenize {
        #[arg(short, long, help = "Input text file to tokenize")]
        input: String,
        
        #[arg(long, help = "Path to vocabulary JSON file")]
        vocab_file: String,
        
        #[arg(short, long, help = "Path to merges text file")]
        merges_file: String,
        
        #[arg(short, long, help = "Output directory for tokenized arrays")]
        output_dir: String,
        
        #[arg(short, long, default_value = "0.9", help = "Train/val split ratio")]
        split_ratio: f64,
        
        #[arg(long, default_value = "npy", help = "Output format (npy)")]
        output_format: String,
    },
}

fn split_dataset(text: &str, split_ratio: f64) -> (String, String) {
    log::info!("Splitting dataset with {:.1}% train ratio...", split_ratio * 100.0);
    
    // Split by document delimiter for clean boundaries
    let documents: Vec<&str> = text.split("<|endoftext|>").collect();
    let num_train_docs = (documents.len() as f64 * split_ratio) as usize;
    
    let train_docs = &documents[..num_train_docs];
    let val_docs = &documents[num_train_docs..];
    
    let train_text = train_docs.join("<|endoftext|>");
    let val_text = val_docs.join("<|endoftext|>");
    
    log::info!("✓ Split into {} train docs, {} val docs", train_docs.len(), val_docs.len());
    log::info!("✓ Train size: {} bytes", train_text.len());
    log::info!("✓ Val size: {} bytes", val_text.len());
    
    (train_text, val_text)
}


fn tokenize_dataset(
    tokenizer: &OptimizedBpeTokenizer,
    text: &str,
    name: &str,
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    log::info!("Tokenizing {} set ({:.1} MB)...", name, text.len() as f64 / 1024.0 / 1024.0);
    
    let start = Instant::now();
    let tokens = if text.len() > 100_000_000 { // 100MB threshold
        tokenizer.encode_with_progress(text, true)
    } else {
        tokenizer.encode(text)
    };
    let duration = start.elapsed();
    
    let throughput = text.len() as f64 / duration.as_secs_f64() / 1024.0 / 1024.0;
    log::info!("✓ {} tokenization: {} tokens in {:.2}s ({:.2} MB/s)", 
              name, tokens.len(), duration.as_secs_f64(), throughput);
    
    Ok(tokens)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(if args.verbose { log::LevelFilter::Info } else { log::LevelFilter::Warn })
        .format_timestamp_secs()
        .init();
    
    match args.command {
        Commands::Tokenize { input, vocab_file, merges_file, output_dir, split_ratio, output_format } => {
            // Validate arguments
            if !Path::new(&input).exists() {
                log::error!("Input file does not exist: {}", input);
                std::process::exit(1);
            }
            
            if !Path::new(&vocab_file).exists() {
                log::error!("Vocabulary file does not exist: {}", vocab_file);
                std::process::exit(1);
            }
            
            if !Path::new(&merges_file).exists() {
                log::error!("Merges file does not exist: {}", merges_file);
                std::process::exit(1);
            }
            
            if split_ratio <= 0.0 || split_ratio >= 1.0 {
                log::error!("split_ratio must be between 0 and 1, got {}", split_ratio);
                std::process::exit(1);
            }
            
            println!("🦀 Optimized BPE Tokenizer v2.0");
            println!("📁 Input: {}", input);
            println!("📚 Vocab: {}", vocab_file);
            println!("🔀 Merges: {}", merges_file);
            println!("📂 Output: {}", output_dir);
            println!("📊 Split ratio: {:.1}%", split_ratio * 100.0);
            println!("💾 Output format: {}", output_format);
            
            // Load tokenizer
            let tokenizer = OptimizedBpeTokenizer::from_files(&vocab_file, &merges_file)?;
            
            // Load and process dataset
            log::info!("=== Starting Dataset Tokenization ===");
            log::info!("Input: {}", input);
            log::info!("Output dir: {}", output_dir);
            log::info!("Split ratio: {:.1}%", split_ratio * 100.0);
            log::info!("Output format: {}", output_format);
            
            log::info!("Loading dataset...");
            let start = Instant::now();
            let text = fs::read_to_string(&input)?;
            log::info!("✓ Loaded {:.1} MB of text in {:.2}s", 
                     text.len() as f64 / 1024.0 / 1024.0,
                     start.elapsed().as_secs_f64());
            
            // Split dataset
            let (train_text, val_text) = split_dataset(&text, split_ratio);
            
            // Tokenize training set
            let train_tokens = tokenize_dataset(&tokenizer, &train_text, "Train")?;
            
            // Tokenize validation set
            let val_tokens = tokenize_dataset(&tokenizer, &val_text, "Val")?;
            
            // Save as numpy arrays
            let input_filename = Path::new(&input)
                .file_stem()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("dataset");
            
            let train_output = format!("{}/{}_train.npy", output_dir, input_filename);
            let val_output = format!("{}/{}_val.npy", output_dir, input_filename);
            
            save_tokens_as_npy_optimized(&train_tokens, &train_output)?;
            save_tokens_as_npy_optimized(&val_tokens, &val_output)?;
            
            println!("✅ Tokenization complete!");
            println!("📊 Train: {} tokens → {}", train_tokens.len(), train_output);
            println!("📊 Val: {} tokens → {}", val_tokens.len(), val_output);
        }
    }
    
    Ok(())
}