import regex as re
from typing import List, Optional, Iterator, Dict, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import Counter

from cs336_basics.pretokenization_example import find_chunk_boundaries

# Regex pattern from GPT-2, see page 6 of the assignment PDF
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# BPE Training functions

def get_pre_tokenizer(special_tokens: Optional[List[str]] = None) -> 'Tokenizer':
    """Returns a Tokenizer instance that only does pre-tokenization."""
    return Tokenizer(special_tokens=special_tokens)

def _get_stats(word_counts: Dict[tuple, int]) -> Dict[tuple, int]:
    """Helper function to compute initial pair frequencies."""
    stats = {}
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            stats[pair] = stats.get(pair, 0) + count
    return stats

def _merge_word(word: tuple, pair: tuple, new_token_id: int) -> tuple:
    """Helper function to merge a pair in a single word."""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i+1]) == pair:
            new_word.append(new_token_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def _process_chunk(chunk_boundary: Tuple[int, int], input_path: str, special_tokens: List[str], reverse_vocab: Dict[bytes, int]) -> Counter:
    """Processes a chunk of the input file and returns word counts."""
    start, end = chunk_boundary
    word_counts = Counter()
    pre_tokenizer = get_pre_tokenizer(special_tokens)
    special_tokens_set = set(token.encode('utf-8') for token in special_tokens)

    with open(input_path, 'r', encoding='utf-8') as f:
        f.seek(start)
        text = f.read(end - start)

    for pre_token_str in pre_tokenizer.pre_tokenize(text):
        encoded_token = pre_token_str.encode('utf-8')
        if encoded_token in special_tokens_set:
            word_tuple = (reverse_vocab[encoded_token],)
        else:
            word_tuple = tuple(encoded_token)
        word_counts[word_tuple] += 1
    return word_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer from a text file, using an optimized merging strategy and parallel pre-tokenization.
    """
    assert vocab_size >= 256
    
    # 1. Initialize vocab and merges
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        idx = len(vocab)
        vocab[idx] = token.encode('utf-8')
    
    merges = []
    
    # 2. Pre-tokenize the text and count initial word frequencies
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    # Check file size and use multiprocessing only for larger files (>1MB)
    import os
    file_size = os.path.getsize(input_path)
    use_multiprocessing = file_size > 1024 * 1024  # 1MB threshold
    
    if use_multiprocessing:
        num_processes = cpu_count()
        with open(input_path, 'rb') as f:
            # Use the first special token as the split token, or a default if none are provided
            split_token = special_tokens[0].encode('utf-8') if special_tokens else b'\n'
            chunk_boundaries = find_chunk_boundaries(f, num_processes, split_token)

        pool = Pool(num_processes)
        process_chunk_partial = partial(_process_chunk, input_path=input_path, special_tokens=special_tokens, reverse_vocab=reverse_vocab)
        chunk_word_counts = pool.map(process_chunk_partial, zip(chunk_boundaries[:-1], chunk_boundaries[1:]))
        pool.close()
        pool.join()

        # Aggregate word counts from all processes
        word_counts = Counter()
        for counts in chunk_word_counts:
            word_counts.update(counts)
    else:
        # Single-threaded processing for smaller files
        word_counts = Counter()
        pre_tokenizer = get_pre_tokenizer(special_tokens)
        special_tokens_set = set(token.encode('utf-8') for token in special_tokens)

        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        for pre_token_str in pre_tokenizer.pre_tokenize(text):
            encoded_token = pre_token_str.encode('utf-8')
            if encoded_token in special_tokens_set:
                word_tuple = (reverse_vocab[encoded_token],)
            else:
                word_tuple = tuple(encoded_token)
            word_counts[word_tuple] += 1

    # 3. Compute initial pair statistics
    stats = _get_stats(word_counts)

    # 4. Iteratively merge the most frequent pair
    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        if not stats:
            print("No more pairs to merge.")
            break
        
        # Find the most frequent pair, breaking ties by lexicographically greater pair
        # According to the assignment PDF: "deterministically break ties in pair frequency 
        # by preferring the lexicographically greater pair"
        max_count = max(stats.values())
        best_pair = max(
            (p for p, count in stats.items() if count == max_count),
            key=lambda p: (vocab[p[0]], vocab[p[1]])
        )
        
        pair_bytes = (vocab[best_pair[0]], vocab[best_pair[1]])
        merges.append(pair_bytes)
        
        new_token_id = len(vocab)
        
        # --- OPTIMIZED MERGE LOGIC ---
        new_word_counts = {}
        stats_delta = {}

        for word, count in word_counts.items():
            new_word = _merge_word(word, best_pair, new_token_id)
            
            if new_word != word:
                # If the word changed, calculate the delta for stats
                # Subtract all pairs from the old word
                for k in range(len(word) - 1):
                    pair = (word[k], word[k+1])
                    stats_delta[pair] = stats_delta.get(pair, 0) - count
                # Add all pairs from the new word
                for k in range(len(new_word) - 1):
                    pair = (new_word[k], new_word[k+1])
                    stats_delta[pair] = stats_delta.get(pair, 0) + count
            
            new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count

        # Apply the collected updates to stats
        for pair, delta in stats_delta.items():
            stats[pair] = stats.get(pair, 0) + delta
        
        word_counts = new_word_counts
        # --- END OPTIMIZED MERGE LOGIC ---

        # Clean up stats
        stats = {k: v for k, v in stats.items() if v > 0}

        # Add the new merged token to the vocabulary
        vocab[new_token_id] = pair_bytes[0] + pair_bytes[1]
        
        if (i + 1) % 10 == 0 or i == num_merges - 1:
             print(f"Merge {i+1}/{num_merges}: {pair_bytes} -> {vocab[new_token_id]} (ID: {new_token_id})")

    return vocab, merges


# -----------------------------------------------------------------------------
# Tokenizer class (for encoding/decoding)

class Tokenizer:
    """
    A BPE Tokenizer. It can pre-tokenize text and later will encode/decode.
    """
    def __init__(self, special_tokens: Optional[List[str]] = None):
        """
        Initializes the tokenizer for pre-tokenization purposes.
        """
        self.pat = re.compile(PAT)
        self.special_tokens_set = set(special_tokens) if special_tokens else set()
        
        if self.special_tokens_set:
            escaped_tokens = [re.escape(token) for token in special_tokens]
            self.special_splitter = re.compile(f"({'|'.join(escaped_tokens)})")
        else:
            self.special_splitter = None

    def pre_tokenize(self, text: str) -> Iterator[str]:
        """
        Splits the text into pre-tokens using a memory-efficient generator.
        """
        if not self.special_splitter:
            for match in self.pat.finditer(text):
                yield match.group(0)
            return

        parts = self.special_splitter.split(text)
        for part in parts:
            if not part: continue
            if part in self.special_tokens_set:
                yield part
            else:
                for match in self.pat.finditer(part):
                    yield match.group(0)

if __name__ == '__main__':
    # Example usage of the BPE training function
    
    # Create a dummy corpus file based on the PDF example
    dummy_corpus = "low low low low low\nlower lower\nwidest widest widest\nnewest newest newest newest newest newest"
    dummy_file = "dummy_corpus.txt"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(dummy_corpus)

    # Define desired vocab size and special tokens
    vocab_size = 270 # 256 base + special + merges
    special_tokens = ["<|endoftext|>"]

    print("Training BPE tokenizer...")
    # This function call now matches the problem description
    final_vocab, final_merges = train_bpe(dummy_file, vocab_size, special_tokens)
    
    print("\n--- Training Complete ---")
    print(f"Final Vocab Size: {len(final_vocab)}")
    print(f"Number of Merges: {len(final_merges)}")
    
    print("\nVocabulary (new tokens):")
    new_tokens = {k: v for k, v in final_vocab.items() if k >= 256}
    for k, v in new_tokens.items():
        print(f"{k}: {v.decode('utf-8', errors='replace')}")

    print("\nMerges:")
    for m in final_merges:
        print(f"({m[0].decode('utf-8', 'replace')}, {m[1].decode('utf-8', 'replace')})")