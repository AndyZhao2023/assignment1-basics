import regex as re
from typing import List, Optional, Iterator, Dict, Tuple
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import mmap
import os

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
    # Quick check: if the pair doesn't exist, return the original word
    found_pair = False
    for i in range(len(word) - 1):
        if (word[i], word[i+1]) == pair:
            found_pair = True
            break
    
    if not found_pair:
        return word
    
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


def _process_chunk(args):
    """Worker function to process a chunk of text for BPE training."""
    file_path, special_tokens, start_pos, end_pos = args
    
    # Create pre-tokenizer
    pre_tokenizer = get_pre_tokenizer(special_tokens)
    special_tokens_set = set(token.encode('utf-8') for token in special_tokens)
    
    # Count words in this chunk
    word_counts = Counter()
    
    # Read and process the chunk
    with open(file_path, 'rb') as f:
        f.seek(start_pos)
        chunk_data = f.read(end_pos - start_pos)
    
    text = chunk_data.decode('utf-8', errors='ignore')
    
    for pre_token_str in pre_tokenizer.pre_tokenize(text):
        encoded_token = pre_token_str.encode('utf-8')
        if encoded_token in special_tokens_set:
            # Special tokens are not split
            word_tuple = (encoded_token,)
        else:
            word_tuple = tuple(encoded_token)
        word_counts[word_tuple] += 1
    
    return word_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], verbose: bool = True, num_workers: Optional[int] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer from a text file, using an optimized merging strategy and parallel pre-tokenization.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size (must be >= 256)
        special_tokens: List of special tokens to include
        verbose: Whether to print progress
        num_workers: Number of worker processes (defaults to CPU count)
    """
    assert vocab_size >= 256
    
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 workers to avoid memory issues
    
    # 1. Initialize vocab and merges
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        idx = len(vocab)
        vocab[idx] = token.encode('utf-8')
    
    merges = []
    
    # 2. Pre-tokenize the text and count initial word frequencies using multiprocessing
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    if verbose:
        print(f"Pre-tokenizing with {num_workers} workers...")
    
    file_size = os.path.getsize(input_path)
    
    # For small files, use single-threaded processing
    if file_size < 10 * 1024 * 1024:  # 10MB threshold
        # Single-threaded processing for small files
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
    else:
        # Multiprocessing for large files
        # Calculate chunk boundaries
        chunk_size = file_size // num_workers
        chunks = []
        delimiter = b'<|endoftext|>'
        
        # Find chunk boundaries at document delimiters
        with open(input_path, 'rb') as f:
            for i in range(num_workers):
                start = i * chunk_size
                if i == num_workers - 1:
                    end = file_size
                else:
                    # Find next delimiter after the chunk boundary
                    end = min((i + 1) * chunk_size, file_size)
                    # Look for delimiter to avoid splitting documents
                    f.seek(max(0, end - 100))
                    search_data = f.read(min(1100, file_size - f.tell()))
                    pos = search_data.find(delimiter)
                    if pos != -1:
                        end = max(0, end - 100) + pos + len(delimiter)
                
                chunks.append((input_path, special_tokens, start, end))
        
        # Process chunks in parallel
        with Pool(num_workers) as pool:
            chunk_results = pool.map(_process_chunk, chunks)
        
        # Merge word counts from all workers
        word_counts = Counter()
        for chunk_count in chunk_results:
            word_counts.update(chunk_count)
        
        # Convert special token bytes back to their token IDs
        special_tokens_bytes = {token.encode('utf-8') for token in special_tokens}
        new_word_counts = Counter()
        for word, count in word_counts.items():
            if len(word) == 1 and word[0] in special_tokens_bytes:
                # Convert special token back to its ID
                new_word = (reverse_vocab[word[0]],)
            else:
                new_word = word
            new_word_counts[new_word] = count
        word_counts = new_word_counts
    
    if verbose:
        print(f"Pre-tokenization complete. Found {len(word_counts)} unique words.")

    # 3. Compute initial pair statistics
    stats = _get_stats(word_counts)

    # 4. Iteratively merge the most frequent pair
    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        if not stats:
            if verbose:
                print("No more pairs to merge.")
            break
        
        # Find the most frequent pair, breaking ties by lexicographically greater pair
        # According to the assignment PDF: "deterministically break ties in pair frequency 
        # by preferring the lexicographically greater pair"
        best_pair = max(stats.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))[0]
        
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

        # Add the new merged token to the vocabulary
        vocab[new_token_id] = pair_bytes[0] + pair_bytes[1]
        
        # Reduced printing frequency for speed
        if verbose and ((i + 1) % 50 == 0 or i == num_merges - 1):
             print(f"Merge {i+1}/{num_merges}: {pair_bytes} -> {vocab[new_token_id]} (ID: {new_token_id})")

    return vocab, merges


# -----------------------------------------------------------------------------
# Tokenizer class (for encoding/decoding)

class Tokenizer:
    """
    A BPE Tokenizer. It can pre-tokenize text and encode/decode using trained vocab and merges.
    """
    def __init__(self, vocab: Optional[Dict[int, bytes]] = None, merges: Optional[List[Tuple[bytes, bytes]]] = None, special_tokens: Optional[List[str]] = None):
        """
        Initializes the tokenizer with vocab, merges, and special tokens.
        """
        # Pre-tokenization setup
        self.pat = re.compile(PAT)
        self.special_tokens_set = set(special_tokens) if special_tokens else set()
        
        if self.special_tokens_set:
            # Sort special tokens by length (descending) to handle overlapping tokens correctly
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_tokens]
            self.special_splitter = re.compile(f"({'|'.join(escaped_tokens)})")
        else:
            self.special_splitter = None
        
        # BPE encoding/decoding setup
        if vocab is not None and merges is not None:
            self.vocab = vocab.copy()
            self.merges = merges.copy()
            
            # Add special tokens to vocab if not already present
            if special_tokens:
                for token in special_tokens:
                    token_bytes = token.encode('utf-8')
                    if token_bytes not in self.vocab.values():
                        new_id = max(self.vocab.keys()) + 1
                        self.vocab[new_id] = token_bytes
            
            # Create reverse vocab mapping for encoding
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            # Create merge priority mapping (lower index = higher priority)
            self.merge_priorities = {}
            for i, (a, b) in enumerate(self.merges):
                merged = a + b
                self.merge_priorities[merged] = i
        else:
            self.vocab = None
            self.merges = None
            self.reverse_vocab = None
            self.merge_priorities = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None) -> 'Tokenizer':
        """
        Class method to construct a Tokenizer from serialized vocab and merges files.
        """
        import json
        
        # Load vocabulary
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Convert vocab to int -> bytes mapping
        vocab = {}
        for token_id_str, token_str in vocab_data.items():
            # Convert string representation back to bytes
            try:
                # Handle byte strings that may be encoded as escape sequences
                token_bytes = token_str.encode('utf-8').decode('unicode_escape').encode('latin-1')
            except:
                # Fallback: encode as UTF-8
                token_bytes = token_str.encode('utf-8')
            vocab[int(token_id_str)] = token_bytes
        
        # Load merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        try:
                            # Handle byte strings that may be encoded as escape sequences
                            token1 = parts[0].encode('utf-8').decode('unicode_escape').encode('latin-1')
                            token2 = parts[1].encode('utf-8').decode('unicode_escape').encode('latin-1')
                        except:
                            # Fallback: encode as UTF-8
                            token1 = parts[0].encode('utf-8')
                            token2 = parts[1].encode('utf-8')
                        merges.append((token1, token2))
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
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
    
    def _encode_word(self, word_bytes: bytes) -> List[int]:
        """
        Apply BPE merges to a single word (sequence of bytes).
        """
        if not self.vocab or not self.merges:
            raise ValueError("Tokenizer not initialized with vocab and merges")
        
        # Start with individual bytes
        word = [bytes([b]) for b in word_bytes]
        
        # Apply merges in order
        for merge_a, merge_b in self.merges:
            if len(word) < 2:
                break
                
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == merge_a and word[i + 1] == merge_b:
                    # Apply merge
                    new_word.append(merge_a + merge_b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        # Convert to token IDs
        token_ids = []
        for token_bytes in word:
            if token_bytes in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[token_bytes])
            else:
                # Fallback: encode as individual bytes
                for b in token_bytes:
                    byte_token = bytes([b])
                    if byte_token in self.reverse_vocab:
                        token_ids.append(self.reverse_vocab[byte_token])
        
        return token_ids
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into a sequence of token IDs.
        """
        if not self.vocab or not self.merges:
            raise ValueError("Tokenizer not initialized with vocab and merges")
        
        token_ids = []
        special_tokens_bytes = {token.encode('utf-8') for token in self.special_tokens_set}
        
        for pre_token_str in self.pre_tokenize(text):
            pre_token_bytes = pre_token_str.encode('utf-8')
            
            # Handle special tokens
            if pre_token_bytes in special_tokens_bytes:
                if pre_token_bytes in self.reverse_vocab:
                    token_ids.append(self.reverse_vocab[pre_token_bytes])
                continue
            
            # Apply BPE encoding (use optimized version)
            word_token_ids = self._encode_word_fast(pre_token_bytes)
            token_ids.extend(word_token_ids)
        
        return token_ids
    
    def _encode_word_fast(self, word_bytes: bytes) -> List[int]:
        """
        Optimized BPE encoding that avoids scanning all merges but preserves original behavior.
        Uses merge_priorities lookup instead of iterating through all merges.
        """
        if not self.vocab or not self.merges:
            raise ValueError("Tokenizer not initialized with vocab and merges")
        
        # Start with individual bytes
        word = [bytes([b]) for b in word_bytes]
        
        # Process merges in the same order as original (by merge file order/priority)
        # But only check if the merge is applicable instead of scanning all merges
        for merge_a, merge_b in self.merges:
            if len(word) < 2:
                break
            
            # Quick check: is this merge even possible in current word?
            merge_possible = False
            for i in range(len(word) - 1):
                if word[i] == merge_a and word[i + 1] == merge_b:
                    merge_possible = True
                    break
            
            if not merge_possible:
                continue  # Skip to next merge (this is the optimization!)
            
            # Apply the merge (same as original)
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == merge_a and word[i + 1] == merge_b:
                    # Apply merge
                    new_word.append(merge_a + merge_b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        # Convert to token IDs
        token_ids = []
        for token_bytes in word:
            if token_bytes in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[token_bytes])
            else:
                # Fallback: encode as individual bytes
                for b in token_bytes:
                    byte_token = bytes([b])
                    if byte_token in self.reverse_vocab:
                        token_ids.append(self.reverse_vocab[byte_token])
        
        return token_ids

    def encode_iterable(self, iterable) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        Memory-efficient tokenization for large files.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back into text.
        """
        if not self.vocab:
            raise ValueError("Tokenizer not initialized with vocab")
        
        # Convert token IDs to bytes
        byte_sequence = b''
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]
        
        # Decode bytes to string, replacing malformed bytes with Unicode replacement character
        try:
            return byte_sequence.decode('utf-8', errors='replace')
        except Exception:
            return byte_sequence.decode('utf-8', errors='replace')

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