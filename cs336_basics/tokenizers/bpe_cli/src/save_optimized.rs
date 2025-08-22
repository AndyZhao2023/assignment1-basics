use std::fs::{self, File};
use std::io::{Write, BufWriter};
use std::path::Path;

pub fn save_tokens_as_npy_optimized(tokens: &[u16], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Saving {} tokens to {} (optimized)", tokens.len(), output_path);
    let start = std::time::Instant::now();
    
    // Create output directory
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }
    
    // Use BufWriter with large buffer (8MB)
    let file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    
    // NPY magic number
    writer.write_all(b"\x93NUMPY")?;
    
    // Version 1.0
    writer.write_all(&[0x01, 0x00])?;
    
    // Create header
    let shape_str = format!("({},)", tokens.len());
    let header = format!(
        "{{'descr': '<u2', 'fortran_order': False, 'shape': {}}}", 
        shape_str
    );
    
    // Pad header to 64-byte boundary
    let header_len = header.len();
    let padding_len = (64 - (10 + header_len) % 64) % 64;
    let padded_header = format!("{}{}", header, " ".repeat(padding_len));
    
    // Write header length (little-endian u16)
    let total_header_len = padded_header.len() as u16;
    writer.write_all(&total_header_len.to_le_bytes())?;
    
    // Write padded header
    writer.write_all(padded_header.as_bytes())?;
    
    // OPTIMIZATION: Write in chunks instead of one at a time
    const CHUNK_SIZE: usize = 1_000_000; // 1M tokens = 2MB per chunk
    let mut bytes_written = 0;
    let total_bytes = tokens.len() * 2;
    let mut last_log_percent = 0;
    
    for chunk in tokens.chunks(CHUNK_SIZE) {
        // Convert chunk to bytes in one go
        let mut byte_buffer = Vec::with_capacity(chunk.len() * 2);
        for &token in chunk {
            byte_buffer.extend_from_slice(&token.to_le_bytes());
        }
        
        // Write entire chunk at once
        writer.write_all(&byte_buffer)?;
        bytes_written += byte_buffer.len();
        
        // Progress logging
        let percent = (bytes_written * 100) / total_bytes;
        if percent >= last_log_percent + 5 {
            let elapsed = start.elapsed().as_secs_f64();
            let mb_written = bytes_written as f64 / 1024.0 / 1024.0;
            let throughput = mb_written / elapsed;
            log::info!("Writing progress: {}% ({:.1} MB @ {:.1} MB/s)", 
                     percent, mb_written, throughput);
            last_log_percent = percent;
        }
    }
    
    // Ensure all data is flushed to disk
    writer.flush()?;
    
    let duration = start.elapsed();
    let mb_written = total_bytes as f64 / 1024.0 / 1024.0;
    log::info!("✓ Saved {} tokens ({:.1} MB) in {:.2}s ({:.1} MB/s)", 
              tokens.len(), mb_written, duration.as_secs_f64(),
              mb_written / duration.as_secs_f64());
    Ok(())
}