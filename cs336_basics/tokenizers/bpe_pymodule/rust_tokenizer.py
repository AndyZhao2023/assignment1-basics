"""
Rust-optimized BPE tokenizer implementation.
Uses Rust for performance-critical operations while maintaining Python interface compatibility.
"""

from typing import List, Tuple, Dict, Optional
import regex as re

# Try to import Rust optimizations, fall back to Python if not available
try:
    import rust_tokenizer
    RUST_AVAILABLE = True
    print("✅ Rust optimizations loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"Warning: Rust optimizations not available, falling back to Python: {e}")

from ..tokenizer import (
    get_pre_tokenizer, _get_stats, _merge_word, 
    Tokenizer as PythonTokenizer
)

def train_bpe_rust_optimized(
    input_path: str, 
    vocab_size: int, 
    special_tokens: List[str], 
    verbose: bool = True, 
    num_workers: Optional[int] = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Rust-optimized BPE training function.
    Falls back to Python implementation if Rust is not available.
    """
    assert vocab_size >= 256
    
    if not RUST_AVAILABLE:
        # Fall back to Python implementation
        from ..tokenizer import train_bpe
        return train_bpe(input_path, vocab_size, special_tokens, verbose, num_workers)
    
    if verbose:
        print("Using Rust-optimized BPE training...")
        import os
        file_size = os.path.getsize(input_path)
        print(f"File size: {file_size / 1e9:.2f} GB")
    
    # 1. Initialize vocab and merges
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        idx = len(vocab)
        vocab[idx] = token.encode('utf-8')
    
    merges = []
    
    # 2. Use Rust for fast pre-tokenization
    if verbose:
        print("\nPhase 1: Fast pre-tokenization with Rust...")
        import time
        start_time = time.time()
    
    try:
        # Rust pre-tokenization returns word counts as bytes -> count dict
        # Check if we have a log file path to pass
        log_file_path = None
        if hasattr(rust_tokenizer.fast_pretokenize, '__code__') and 'log_file_path' in rust_tokenizer.fast_pretokenize.__code__.co_varnames:
            log_file_path = f"profiling_logs/rust_profile_{int(time.time())}.jsonl"
        
        if log_file_path:
            word_counts_raw = rust_tokenizer.fast_pretokenize(
                input_path, 
                special_tokens, 
                num_workers,
                log_file_path
            )
        else:
            word_counts_raw = rust_tokenizer.fast_pretokenize(
                input_path, 
                special_tokens, 
                num_workers
            )
        
        # Convert to format expected by Python BPE code
        reverse_vocab = {v: k for k, v in vocab.items()}
        special_tokens_bytes = {token.encode('utf-8') for token in special_tokens}
        
        word_counts = {}
        for py_bytes, count in word_counts_raw.items():
            # Convert PyBytes to bytes
            word_bytes = bytes(py_bytes)
            
            # Handle special tokens
            if word_bytes in special_tokens_bytes:
                word_tuple = (reverse_vocab[word_bytes],)
            else:
                word_tuple = tuple(word_bytes)
            word_counts[word_tuple] = count
            
    except Exception as e:
        if verbose:
            print(f"Rust pre-tokenization failed: {e}")
            print("Falling back to Python pre-tokenization...")
        
        # Fall back to Python pre-tokenization
        from ..tokenizer import train_bpe
        return train_bpe(input_path, vocab_size, special_tokens, verbose, num_workers)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Pre-tokenization complete in {elapsed:.1f} seconds")
        print(f"Found {len(word_counts)} unique words")
        print(f"Processing speed: {file_size / elapsed / 1e6:.1f} MB/s")
    
    # 3. Compute initial pair statistics (use optimized Python version)
    if verbose:
        print("Phase 2: Computing initial pair statistics...")
    
    stats = _get_stats(word_counts)
    
    # 4. Iteratively merge the most frequent pair
    if verbose:
        print("\nPhase 3: Performing BPE merges...")
        print(f"Target merges: {vocab_size - len(vocab)}")
        merge_start = time.time()
    
    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        if not stats:
            if verbose:
                print("No more pairs to merge.")
            break
        
        # Find the most frequent pair
        best_pair = max(stats.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))[0]
        
        pair_bytes = (vocab[best_pair[0]], vocab[best_pair[1]])
        merges.append(pair_bytes)
        
        new_token_id = len(vocab)
        
        # Use optimized merge logic
        new_word_counts = {}
        stats_delta = {}

        for word, count in word_counts.items():
            new_word = _merge_word(word, best_pair, new_token_id)
            
            if new_word != word:
                # Calculate stats delta
                for k in range(len(word) - 1):
                    pair = (word[k], word[k+1])
                    stats_delta[pair] = stats_delta.get(pair, 0) - count
                for k in range(len(new_word) - 1):
                    pair = (new_word[k], new_word[k+1])
                    stats_delta[pair] = stats_delta.get(pair, 0) + count
            
            new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count

        # Apply stats updates
        for pair, delta in stats_delta.items():
            stats[pair] = stats.get(pair, 0) + delta
            if stats[pair] <= 0:
                del stats[pair]
        
        word_counts = new_word_counts
        vocab[new_token_id] = pair_bytes[0] + pair_bytes[1]
        
        # Progress reporting
        if verbose:
            if (i + 1) % 100 == 0:
                elapsed = time.time() - merge_start
                rate = (i + 1) / elapsed
                remaining = (num_merges - i - 1) / rate if rate > 0 else 0
                print(f"Merge {i+1}/{num_merges} ({(i+1)*100/num_merges:.1f}%) - "
                      f"Speed: {rate:.1f} merges/sec - "
                      f"ETA: {remaining/60:.1f} min")
            elif i == num_merges - 1:
                print(f"Merge {i+1}/{num_merges} complete!")

    if verbose:
        total_time = time.time() - start_time
        merge_time = time.time() - merge_start
        print(f"\nBPE training complete!")
        print(f"  Final vocab size: {len(vocab)}")
        print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Pre-tokenization: {elapsed:.1f} seconds")
        print(f"  BPE merges: {merge_time:.1f} seconds")

    return vocab, merges


class RustOptimizedTokenizer(PythonTokenizer):
    """
    Tokenizer class that uses Rust optimizations when available.
    Maintains full compatibility with the Python tokenizer interface.
    """
    
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        super().__init__(vocab, merges, special_tokens)
        self.rust_available = RUST_AVAILABLE
    
    @classmethod
    def train_from_file(
        cls, 
        input_path: str, 
        vocab_size: int, 
        special_tokens: List[str], 
        verbose: bool = True,
        num_workers: Optional[int] = None
    ) -> 'RustOptimizedTokenizer':
        """
        Class method to train a BPE tokenizer using Rust optimizations.
        """
        vocab, merges = train_bpe_rust_optimized(
            input_path, vocab_size, special_tokens, verbose, num_workers
        )
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


# For backward compatibility
def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], verbose: bool = True, num_workers: Optional[int] = None):
    """Wrapper function for backward compatibility."""
    return train_bpe_rust_optimized(input_path, vocab_size, special_tokens, verbose, num_workers)