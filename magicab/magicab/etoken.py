from .base_tok import get_valid_stats, merge, save_byte_vocab, load_byte_vocab
import json, re, os
from typing import Union
from tqdm import tqdm  # Add this import at the top of the file
from rust_tokenizer import PyETokenizer
from copy import deepcopy
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

def encode_char(text, special_tokens, special2idx, char2idx): 
    pattern = f"({'|'.join(re.escape(token) for token in special_tokens)})"
    segments = re.split(pattern, text)
    ids = []
    for seg in segments:
        if not seg:  # Skip empty segments
            continue
        if seg in special2idx:  # Handle special tokens
            ids.append(special2idx[seg])
        else:  # Handle regular characters
            ids.extend(char2idx[c] for c in seg)
    return ids 


def encode_bytes(text, special_tokens, special2idx, byte2idx):
    """Encode text as byte tokens."""
    pattern = f"({'|'.join(re.escape(token) for token in special_tokens)})"
    segments = re.split(pattern, text)
    ids = []
    for seg in segments:
        if not seg:  # Skip empty segments
            continue
        if seg in special2idx:  # Handle special tokens
            ids.append(special2idx[seg])
        else:  # Handle regular text as bytes
            # Convert text to bytes and map to token IDs
            byte_data = seg.encode('utf-8')
            ids.extend(byte2idx[b] for b in byte_data)
    return ids


def decode_bytes(vocab, ids): 
    result = []
    byte_buffer = []
    
    # Process tokens by grouping consecutive byte tokens
    for idx in ids:
        token = vocab[idx]
        if isinstance(token, bytes):
            # Accumulate byte tokens
            byte_buffer.append(token)
        else:
            # We hit a special token - decode any accumulated bytes first
            if byte_buffer:
                # Decode the accumulated bytes to a string
                bytes_segment = b"".join(byte_buffer)
                result.append(bytes_segment.decode("utf-8", errors="replace"))
                byte_buffer = []
            # Add the special token directly (it's already a string)
            result.append(token)
    
    # Don't forget to process any remaining bytes in the buffer
    if byte_buffer:
        bytes_segment = b"".join(byte_buffer)
        result.append(bytes_segment.decode("utf-8", errors="replace"))
    
    # Join all string segments
    return "".join(result)



class TokenTrie:
    class Node:
        def __init__(self):
            self.children = {}  # char -> Node
            self.token_id = None
            self.is_end = False

    def __init__(self):
        self.root = self.Node()
        self.id2token = {}  # id -> token string
        self.token2id = {}  # token string -> id
        self.merges = {}    # (id1, id2) -> merged_id
        self.next_id = 0

    def add_token(self, token: str, token_id: int = None): # -> token_id, is_new
        if token in self.token2id:
            return token_id, False

        if token_id is None:
            token_id = self.next_id
            self.next_id += 1

        # Add to bidirectional mappings
        self.id2token[token_id] = token
        self.token2id[token] = token_id

        # Add to trie
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = self.Node()
            node = node.children[char]
        node.is_end = True
        node.token_id = token_id

        return token_id, True

    def add_merge(self, id1: int, id2: int):
        token1 = self.id2token[id1]
        token2 = self.id2token[id2]
        merged = token1 + token2
        merged_id, is_new = self.add_token(merged)
        if is_new: 
            self.merges[(id1, id2)] = merged_id
        return merged_id, is_new
    
    def remove_merge(self, token_id: int):
        """Remove all merges that produce or use the given token_id"""
        # Remove merges where this token_id is the result
        self.merges = {
            pair: merged_id for pair, merged_id in self.merges.items()
            if merged_id != token_id
        }
        
        # Remove merges where this token_id is part of the pair
        self.merges = {
            (id1, id2): merged_id for (id1, id2), merged_id in self.merges.items()
            if id1 != token_id and id2 != token_id
        }

    def remove_token(self, token_id: int): 
        """Remove a token and its trie nodes if they're not used by other tokens"""
        if token_id not in self.id2token:
            return
            
        token = self.id2token[token_id]
        
        # Remove from mappings
        del self.id2token[token_id]
        del self.token2id[token]
        
        # Remove from trie structure
        node = self.root
        current_node = None
        for char in token:
            if char not in node.children:
                break
            current_node = node.children[char]
            node = current_node
        
        if current_node and current_node.token_id == token_id:
            current_node.is_end = False
            current_node.token_id = None
            
        # Remove associated merges
        self.remove_merge(token_id)
        
    def truncate_vocab(self, target_vocab_size: int): 
        """Truncate vocabulary to target size"""
        self.id2token = {k: v for k, v in self.id2token.items() if k < target_vocab_size}
        self.token2id = {v: k for k, v in self.id2token.items()}
        self.merges = {k: v for k, v in self.merges.items() if v < target_vocab_size}

    def find_token(self, token: str) -> int:
        """Find token ID using trie structure for faster lookups.
        
        Returns:
            int: Token ID if found, None if not found
        """
        node = self.root
        for char in token:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.token_id if node.is_end else None

    def get_merge_result(self, id1: int, id2: int) -> int:
        return self.merges.get((id1, id2))
    
    def save(self, path, mode="char"):
        """Save TokenTrie state to a JSON file."""
        if mode == "char": 
            id2token = {str(k): v for k, v in self.id2token.items()}
        elif mode == "byte": 
            byte_to_str_dict = {bytes([i]): str(i) for i in range(256)}
            byte_to_str = lambda v: byte_to_str_dict[v] if v in byte_to_str_dict else v
            id2token = {str(k): byte_to_str(v) for k, v in self.id2token.items()}
            
        data = {
            'id2token': id2token,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            'next_id': self.next_id,
            "mode": mode
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path, mode='char'):
        """Load TokenTrie from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trie = cls()
        
        # Restore tokens
        for k, v in data['id2token'].items():
            if ('mode' in data and data['mode'] == "char") or mode == "char": 
                trie.add_token(v, int(k))
            elif ('mode' in data and data['mode'] == "byte") or mode == "byte": 
                str_to_byte_dict = {str(i): bytes([i]) for i in range(256)}
                str_to_byte = lambda v: str_to_byte_dict[v] if v in str_to_byte_dict else v
                trie.add_token(str_to_byte(v), int(k))
        
        # Restore merges
        for k, v in data['merges'].items():
            id1, id2 = map(int, k.split(','))
            trie.merges[(id1, id2)] = v
            
        # Restore next_id
        trie.next_id = data['next_id']
        
        return trie

class ETokenizer: 
    """ 
    ETokenizer is a tokenizer that could be updated on-the-fly 
    """
    def __init__(self, char_vocab=False, byte_vocab=False, mode="char"):

        self.eos_token = "<|endoftext|>"
        self.pad_token = "<pad>"
        self.user_token = "<USER> "
        self.assistant_token = "<ASSISTANT> "
        self.special_tokens = [self.eos_token, self.pad_token, self.user_token, self.assistant_token]
        
        self.mode = mode
        if mode == "char" or char_vocab:
            self.char_vocab = char_vocab
            self.char2idx = {c:i for i, c in char_vocab.items()} if char_vocab else {}
            self._init_token_trie(char_vocab=self.char_vocab)
            self.special_ids = list(self.special2idx.values())
        elif mode == "byte" or byte_vocab:
            self.byte_vocab = byte_vocab or {i: bytes([i]) for i in range(256)}
            self.byte2idx = {i: i for i in range(256)} if not byte_vocab else {b: i for i, b in byte_vocab.items()}
            self._init_token_trie(byte_vocab=self.byte_vocab)
            self.special_ids = list(self.special2idx.values())
        else:
            raise ValueError(f"Invalid tokenizer mode: {mode}. Use 'char' or 'byte'")
                
        self.template = {"user": self.user_token + "{user}" + self.eos_token, 
                        "assistant": self.assistant_token + "{assistant}" + self.eos_token}
        
    @property 
    def pad_token_id(self): 
        return self.token_trie.token2id[self.pad_token]
    
    @property 
    def eos_token_id(self): 
        return self.token_trie.token2id[self.eos_token] 
    
    
    def _init_token_trie(self, char_vocab=None, byte_vocab=None):
        # Initialize TokenTrie with basic vocabulary
        self.token_trie = TokenTrie()
        self.special2idx = {}
    
        # Use provided character vocabulary
        if char_vocab:
            for idx, token in char_vocab.items():
                token_idx = idx
                self.token_trie.add_token(token, token_idx)
                
        if byte_vocab:
            for idx, token in byte_vocab.items():
                token_idx = idx
                self.token_trie.add_token(token, token_idx)
                
        # Add special tokens first
        vocab_size = len(self.token_trie.id2token)
        for idx, special in enumerate(self.special_tokens):
            special_idx = vocab_size + idx
            self.token_trie.add_token(special, special_idx)
            self.special2idx[special] = special_idx
        
        return self.token_trie.id2token

    @property
    def vocab(self):
        return self.token_trie.id2token

    @property
    def merges(self):
        return self.token_trie.merges

    @property 
    def rust_tokenizer(self): 
        return PyETokenizer(self.special_ids, self.merges)
    
    @property 
    def inverse_vocab(self): 
        return {v:k for k, v in self.vocab.items()}
    
    @property 
    def vocab_size(self): 
        return len(self.token_trie.id2token)
    
    @property 
    def leaf_merges(self): 
        # 1. can't be in char_vocab
        # 2. should have no dependent tokens
        leaf_merges = {} 
        for k, v in self.merges.items(): 
            if all(v not in pair for pair in self.merges.keys() if pair != k):
                leaf_merges[k] = v 
        return leaf_merges
    
    @property 
    def leaf_token_ids(self): 
        return list(self.leaf_merges.values())

    def decode(self, ids):
        # given ids (list of integers), return Python string
        if self.mode == "char": 
            return "".join(self.vocab[idx] for idx in ids)
        elif self.mode=="byte": 
            return decode_bytes(self.vocab, ids)
        else: 
            raise ValueError(f"Invalid tokenizer mode: {self.mode}. Use 'char' or 'byte'")
        
    
    def _encode_python_exhaustive(self, ids): 
        """
        Exhaustive Encoding | Python ver. 
        """
        while len(ids) >= 2:
            # find pairs and their stats given the current ids
            stats = get_valid_stats(ids, self.special_ids)
            
            # Instead of picking the minimum pair from stats, first filter out any invalid pairs.
            valid_pairs = [p for p in stats if p in self.merges]
            if not valid_pairs:
                break  # no valid merge available
            
            # Pick the valid pair with the lowest merge index
            pair = min(valid_pairs, key=lambda p: self.merges[p])
            
            # Proceed to merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids
    
    
    def _encode_python(self, ids):
        """
        Optimized Exhaustive Encoding using TokenTrie for prefix matching.
        
        Instead of computing all valid pair statistics and filtering,
        we use the TokenTrie to scan for the longest merge candidate starting
        at each token position. This leverages the structure of the Trie to
        quickly check concatenated token strings in sequence.
        """
        def find_longest_merge(start):
            # We need at least two tokens to form a merge candidate.
            if start >= len(ids) - 1:
                return None
            # Initialize the candidate using the first two tokens
            try: 
                current_str = self.token_trie.id2token[ids[start]] + self.token_trie.id2token[ids[start + 1]] # buggy: when special_id is found, it's not a 'bytes' object
            except: 
                return None
            
            merged_token_id = self.token_trie.find_token(current_str)
            if merged_token_id is None:
                return None

            # Record the current candidate: (span length, resulting token id)
            best_candidate = (2, merged_token_id)
            # Try to extend the match by adding subsequent tokens in the sequence.
            for i in range(start + 2, len(ids)):
                current_str += self.token_trie.id2token[ids[i]]
                token_id = self.token_trie.find_token(current_str)
                if token_id is None:
                    break
                # Found a longer valid merge candidate.
                best_candidate = (i - start + 1, token_id)
            return best_candidate

        pos = 0
        # Continue scanning until no more merges can be done.
        while pos < len(ids):
            candidate = find_longest_merge(pos)
            if candidate is not None:
                span_length, merged_token_id = candidate
                # Do the merge by replacing the token span with the merged token id.
                ids = ids[:pos] + [merged_token_id] + ids[pos + span_length:]
                # After a merge, it's safe to step back one index (if pos > 0)
                # so that we can catch any new merge opportunities that might span
                # tokens across the previous merge.
                pos = max(0, pos - 1)
            else:
                pos += 1
        return ids
    
    
    
    def _encode(self, ids, mode="trie"): 
        """ 
        Rust speed-up encoding
        """
        assert mode in ["trie", "exhaustive", "rust"]
        if mode == "trie": 
            return self._encode_python(ids)
        elif mode == "exhaustive": 
            return self._encode_python_exhaustive(ids)
        else: 
            return self.rust_tokenizer.encode(ids)
    
    def encode(self, text, mode="trie"):
        """ 
        Exhaustive Encoding | Byte level vocabulary base --- we need character-level vocabulary
        """
        start_time = time.time()
        ids = self.encode_id(text)
        char_time = time.time() - start_time
        # print(" - Char encode time: ", char_time)

        start_time = time.time()
        encoded = self._encode(ids, mode)
        encode_time = time.time() - start_time
        # print(" - Merge encode time: ", encode_time)
        
        # print("Total encode time: ", char_time + encode_time)
        return encoded
    
    def _process_token_pair(self, prefix_token_idx, curr_token_idx, *args):
        # Check if merge already exists
        merged_id = self.token_trie.get_merge_result(prefix_token_idx, curr_token_idx)
        if merged_id is not None:
            return merged_id, None, None  # Remove unused next_id return value

        # Create new merge
        merged_id, is_new = self.token_trie.add_merge(prefix_token_idx, curr_token_idx)
        if not is_new: 
            return merged_id, None, None
        
        return merged_id, self.token_trie.next_id, (merged_id, curr_token_idx)

    def _add_tokens(self, tokens_to_group, group_positions=None):
        
        vocab = dict(self.token_trie.id2token)
        merges = dict(self.token_trie.merges)
        self.token_trie.next_id = next_idx = max(vocab.keys()) + 1 # next_id is not updated in place before
        
        eom_tokens = []
        pair_token_groups = []
        pair_token_positions = [] if group_positions is None else []
        
        for group_idx, token_group in enumerate(tokens_to_group):
            if len(token_group) <= 1:
                continue
                
            # Process tokens in a sliding window
            prefix_idx = token_group[0]
            for i in range(1, len(token_group)):
                curr_idx = token_group[i]
                
                # Process the token pair
                merged_id, next_id, pair_info = self._process_token_pair(
                    prefix_idx, curr_idx
                )
                
                # Record new pair if created
                if pair_info:
                    eom_tokens.append(curr_idx)
                    pair_token_groups.append(pair_info)
                    if group_positions is not None:
                        pair_token_positions.append((
                            group_positions[group_idx][i-1],
                            group_positions[group_idx][i]
                        ))
                    
                
                prefix_idx = merged_id

        return vocab, merges, eom_tokens, pair_token_groups, pair_token_positions

    def add_tokens(self, tokens_to_group, group_positions=None, in_place=False):
        """
        Add tokens to the tokenizer
        """
        vocab, merges, eom_tokens, pair_token_groups, pair_token_positions = self._add_tokens(tokens_to_group, group_positions)
                
        if not in_place: 
            self.token_trie.id2token = vocab
            self.token_trie.token2id = {token: idx for idx, token in vocab.items()}  # Update token2id mapping
            self.token_trie.merges = merges
            self.token_trie.next_id = max(vocab.keys()) + 1  # Reset next_id counter
        
        if group_positions is not None:
            return eom_tokens, pair_token_groups, pair_token_positions
        else:
            return eom_tokens, pair_token_groups
        
    def truncate_vocab(self, target_vocab_size: int):
        """Truncate vocabulary to target size while preserving special tokens"""
        # Calculate minimum allowed vocabulary size
        min_vocab_size = len(self.char_vocab) + len(self.special_tokens)
        
        if target_vocab_size < min_vocab_size:
            raise ValueError(
                f"Target vocabulary size ({target_vocab_size}) cannot be smaller than "
                f"base vocabulary size ({min_vocab_size}: {len(self.char_vocab)} chars + "
                f"{len(self.special_tokens)} special tokens)"
            )
            
        # Preserve special tokens
        special_ids = set(self.special_ids)
        
        # Filter vocabulary while preserving special tokens
        self.token_trie.id2token = {
            k: v for k, v in self.token_trie.id2token.items() 
            if k < target_vocab_size or k in special_ids
        }
        self.token_trie.token2id = {v: k for k, v in self.token_trie.id2token.items()}
        
        # Filter merges
        self.token_trie.merges = {
            k: v for k, v in self.token_trie.merges.items() 
            if v < target_vocab_size or v in special_ids
        }
        
        # Rebuild trie structure
        new_trie = TokenTrie()
        for token_id, token in self.token_trie.id2token.items():
            new_trie.add_token(token, token_id)
        new_trie.merges = self.token_trie.merges
        new_trie.next_id = target_vocab_size
        
        self.token_trie = new_trie
        
    def identify_splittable_tokens(self, tokens_to_split): 
        
        tokens_removed = []
        for token_id in tokens_to_split:
            if token_id not in self.vocab or len(self.vocab[token_id]) <= 1:
                continue
            
            # Remove from vocabulary
            del self.vocab[token_id]
            tokens_removed.append(token_id)
            
            # Remove any merges that created this token
            self.merges = {pair: idx for pair, idx in self.merges.items() 
                          if idx != token_id}
        return tokens_removed
    
    def remove_tokens(self, tokens_to_remove, in_place=True):
        if not in_place:
            new_tokenizer = self.copy()
            new_tokenizer.remove_tokens(tokens_to_remove)
            return new_tokenizer.token_trie.id2token, new_tokenizer.token_trie.merges
            
        # Remove from tokenizer
        for token_id in tokens_to_remove:
            self.token_trie.remove_token(token_id)                

    def split_tokens(self, tokens_to_split):
        """
        Remove tokens from vocabulary if their byte length is > 1
        Maintains consecutive token indices by remapping remaining tokens
        """
        # Decide tokens to remove : can't be base vocabulary tokens
        tokens_to_remove = self.identify_splittable_tokens(tokens_to_split)
        if not tokens_to_remove:
            return []
        # Remove tokens from vocabulary
        self.remove_tokens(tokens_to_remove)
        
        
    def encode_char(self, text): 
        return encode_char(text, self.special_tokens, self.special2idx, self.char2idx)
    
    
    def encode_byte(self, text): 
        return encode_bytes(text, self.special_tokens, self.special2idx, self.byte2idx)
    
    def encode_id(self, text): 
        if self.mode == "char": 
            return self.encode_char(text)
        elif self.mode == "byte": 
            return self.encode_byte(text)
        else: 
            raise ValueError(f"Invalid tokenizer mode: {self.mode}. Use 'char' or 'byte'")
    
    def save(self, path):
        """Save ETokenizer state to a JSON file."""
        if self.mode == "char": 
            vocab_dict = self.char_vocab
            data = {
                'char_vocab': vocab_dict,
                'special_tokens': self.special_tokens,
                'mode': self.mode
            }
        else: 
            byte_to_str_dict = {bytes([i]): str(i) for i in range(256)}
            vocab_dict = {str(k): byte_to_str_dict[v] for k, v in self.byte_vocab.items()}
            data = {
                'byte_vocab': vocab_dict,
                'special_tokens': self.special_tokens,
                'mode': self.mode
            }
        
        # Save main data
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save TokenTrie to a separate file
        trie_path = path.replace('.json', '_trie.json')
        self.token_trie.save(trie_path, mode=self.mode)

    @classmethod
    def load(cls, path):
        """Load ETokenizer from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert char_vocab keys back to integers
        char_vocab, byte_vocab = None, None
        if ("mode" in data and data['mode'] == "char") or "char_vocab" in data: 
            char_vocab = {int(k): v for k, v in data['char_vocab'].items()}
            mode = "char"
        elif ("mode" in data and data['mode'] == "byte") or "byte_vocab" in data: 
            str_to_byte_dict = {str(i): bytes([i]) for i in range(256)}
            byte_vocab = {int(k): str_to_byte_dict[v] for k, v in data['byte_vocab'].items()}
            mode = "byte"
        # Create instance (this already sets up use_char, char_vocab, char2idx)
        inst = cls(char_vocab=char_vocab, byte_vocab=byte_vocab, mode=mode)
        inst.special_tokens = data['special_tokens']
        
        # Load TokenTrie from separate file
        trie_path = path.replace('.json', '_trie.json')
        inst.token_trie = TokenTrie.load(trie_path, mode=mode)
        
        # Only need to rebuild special token mappings
        inst.special2idx = {token: inst.token_trie.token2id[token] 
                           for token in inst.special_tokens}
        inst.special_ids = list(inst.special2idx.values())
        
        return inst
    
    def sanity_check(self, text = "Hello, world!"):
        assert self.decode(self.encode(text)) == text, "encode & decode mismatch"
        print("Tokenizer has matching encoding & decoding")
        
        
    def copy(self):
        c = ETokenizer(deepcopy(self.char_vocab))
        c.token_trie = deepcopy(self.token_trie)
        return c
    
    def encode_with_chunking(self, text: Union[str, list], chunk_size=256*8, batch_size=50, max_workers=32, mode='sequential'): # can we make this async, and parallelize across chunks?
        if isinstance(text, str): 
           chunks = chunk_text(text, chunk_size)
           return _encode_chunks(chunks, self, chunk_size)
        elif isinstance(text, list): 
            if mode == 'sequential':
                ids = []
                for t in text: 
                    chunks = chunk_text(t, chunk_size)
                    ids.extend(_encode_chunks(chunks, self, chunk_size))
                return ids
            elif mode == 'parallel': 
                return _encode_chunks_parallel(text, self, chunk_size, batch_size=batch_size, max_workers=max_workers)
            elif mode == 'multiprocessing': 
                return multiprocessing_encoding(text, chunk_size, max_workers)
        else: 
            raise ValueError(f"Invalid input type: {type(text)}. Expected str or list.")    
    
    def apply_chat_template(self, conversation,
                            add_generation_prompt=True, block_size=512): 
        
        formatted_text = ""
        for i, conv in enumerate(conversation): 
            role = "user" if "user" in conv else "assistant"
            turn_text = self.template[role].format(**conv)
                    
            if role == "assistant":
                tokens = self.encode(turn_text)
                formatted_text += turn_text
            else:
                tokens = self.encode(turn_text)
                formatted_text += turn_text
                
        if add_generation_prompt:
            formatted_text += self.assistant_token
            
        return formatted_text[-block_size:] # backward slice
    
    
    def prepare_sft_data(self, conversation, add_generation_prompt=True, 
                            return_dict=False, block_size=512):

        formatted_text = ""
        loss_mask = []
        
        # Process conversation turns
        n_turns = len(conversation)
        
        conv_texts = []
        conv_tokens = [] 
        conv_loss_masks = []
        user_indices = [] 
        assistant_indices = [] 
        
        for i, conv in enumerate(conversation): 
            role = "user" if "user" in conv else "assistant"
            turn_text = self.template[role].format(**conv)
                    
            if role == "assistant":
                conv_texts.append(turn_text)
                tokens = self.encode(turn_text)
                conv_loss_masks.append([1] * len(tokens))
                conv_tokens.append(tokens)
                assistant_indices.append(i)
            else:
                conv_texts.append(turn_text)
                tokens = self.encode(turn_text)
                conv_loss_masks.append([0] * len(tokens))
                conv_tokens.append(tokens)
                user_indices.append(i)
                
        formatted_text, loss_mask = self.random_slice_conversation(user_indices, assistant_indices, conv_texts, conv_tokens, conv_loss_masks, block_size)
        
        if return_dict:
            return {
                "text": formatted_text,
                "loss_mask": loss_mask
            }
        return formatted_text.strip()
    
    def _prepare_pt_conversation_data(self, conversation): 
        conv_text = ""
        for i, conv in enumerate(conversation): 
            role = "user" if "user" in conv else "assistant"
            turn_text = self.template[role].format(**conv)
            conv_text += turn_text
        return conv_text
    
    def prepare_pt_conversation_data(self, conversation): 
        # assumption is each conversation has same rounds
        
        # calculate number of conversations (batch-size) | every conversation has first round 
        for i, conv in enumerate(conversation): 
            role = "user" if "user" in conv else "assistant"
            n_conv = len(conv[role])
            break 
        
        # split each conversation and prepare list
        conv_texts = [[] for _ in range(n_conv)]
        conversations = [[] for _ in range(n_conv)]
        
        for i, conv in enumerate(conversation): 
            role = "user" if "user" in conv else "assistant"
            for j in range(n_conv): 
                conversations[j].append({role: conv[role][j]})        
        
        for i in range(n_conv): 
            conv_texts[i] = self._prepare_pt_conversation_data(conversations[i])
            
        return conv_texts
        
        
        

    def random_slice_conversation(self, user_indices, assistant_indices, conv_texts, conv_tokens, conv_loss_masks, block_size):
        """
        Extract a coherent conversation slice within token limit constraints.
        
        Returns:
            tuple: (formatted_text, loss_mask)
        """
        # Find valid starting points (user messages with at least one following assistant message)
        valid_start_indices = [idx for idx in user_indices if any(a_idx > idx for a_idx in assistant_indices)]
        
        if not valid_start_indices:
            raise ValueError("No valid conversation slice possible")
        
        # Select random starting point
        start_idx = random.choice(valid_start_indices)
        
        # Build the maximum consecutive sequence that fits in the block_size
        sequence = []
        total_tokens = 0
        
        for i in range(start_idx, len(conv_tokens)):
            # Stop if adding this segment would exceed token limit
            if total_tokens + len(conv_tokens[i]) > block_size:
                break
                
            # Stop if we encounter a non-consecutive turn
            if sequence and i != sequence[-1] + 1:
                break
                
            sequence.append(i)
            total_tokens += len(conv_tokens[i])
        
        # Ensure we end with an assistant message
        while sequence and sequence[-1] not in assistant_indices:
            sequence.pop()
        
        if not sequence:
            raise ValueError("No valid conversation slice possible")
        
        # Construct the formatted text and loss mask
        formatted_text = ""
        loss_mask = []
        
        for i, idx in enumerate(sequence):
            # Handle the final segment (strip trailing whitespace)
            text = conv_texts[idx]
            tokens = conv_tokens[idx]
            mask = conv_loss_masks[idx]
                
            formatted_text += text
            loss_mask.extend(mask)
        
        return formatted_text, loss_mask
    
def _encode_chunks(chunks, tok, chunk_size=256*8): 
    max_merge_len = max([len(v) for v in tok.vocab.values()])
    ids = []
    prev_chunk_ids = []
    for chunk in chunks: 
        chunk_ids = tok.encode(chunk)
        
        if prev_chunk_ids: 
            connect_start, connect_end = chunk_size-1, max_merge_len-1
            connect_ids = tok._encode(prev_chunk_ids[connect_start:] + chunk_ids[:connect_end])
            ids.extend(connect_ids + chunk_ids[connect_end:-1])
        else:
            ids.extend(chunk_ids[:-1])    
        prev_chunk_ids = chunk_ids    
        
    if chunks and max_merge_len < len(prev_chunk_ids):
        ids.extend(prev_chunk_ids[-1:])
        
    return ids


def chunk_text(text, chunk_size=256*8):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for the space
        
        if current_length + word_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks



# async def _encode_chunks_parallel(texts, tokenizer, chunk_size, batch_size=50, max_workers=32):
#     """
#     Optimized async function for encoding text chunks.
#     Uses a thread pool for CPU-bound tokenization operations.
#     """
#     import asyncio
#     from concurrent.futures import ThreadPoolExecutor
    
#     # Create a shared thread pool with an appropriate number of workers
#     # For CPU-bound tasks, typically use min(32, os.cpu_count() + 4)
#     import os
#     max_workers = min(max_workers, (os.cpu_count() or 4) + 4)
    
#     # Pre-chunk all texts to reduce overhead within the async loop
#     chunked_texts = [(i, chunk_text(t, chunk_size)) for i, t in enumerate(texts)]
#     results = [None] * len(texts)
    
#     async def process_batch(batch):
#         loop = asyncio.get_running_loop()
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 loop.run_in_executor(
#                     executor, 
#                     lambda x: (x[0], _encode_chunks(x[1], tokenizer, chunk_size)),
#                     item
#                 )
#                 for item in batch
#             ]
#             batch_results = await asyncio.gather(*futures)
#             for idx, ids in batch_results:
#                 results[idx] = ids
    
#     # Process in batches
#     batches = [chunked_texts[i:i+batch_size] for i in range(0, len(chunked_texts), batch_size)]
#     await asyncio.gather(*[process_batch(batch) for batch in batches])
    
#     return results


async def _encode_chunks_parallel(texts, tokenizer, chunk_size, batch_size=20, max_workers=8):
    """
    Optimized async function for encoding text chunks.
    Uses a thread pool for CPU-bound tokenization operations with controlled concurrency.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import os
    
    # Use a more conservative number of workers
    # For servers, a smaller fixed number is often better than scaling with CPU count
    cpu_count = os.cpu_count() or 4
    max_workers = min(max_workers, cpu_count)
    
    # Create results container
    results = [None] * len(texts)
    
    # Process texts in batches to control memory usage
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # Single thread pool for all processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_idx, batch in enumerate(batches):
            # Pre-chunk just this batch to reduce memory pressure
            chunked_batch = [(texts.index(t), chunk_text(t, chunk_size)) for t in batch]
            
            # Process each text in the current batch with controlled concurrency
            futures = []
            for i, chunks in chunked_batch:
                future = executor.submit(_encode_chunks, chunks, tokenizer, chunk_size)
                futures.append((i, future))
            
            # Wait for this batch to complete before moving to next batch
            for i, future in futures:
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Error processing text {i}: {str(e)}")
    
    return results


from multiprocessing import Pool, cpu_count

def _single_encoding(args): 
    text, chunk_size = args
    # Create tokenizer inside the process to avoid sharing across processes
    process_tokenizer = ETokenizer(mode="byte")
    chunks = chunk_text(text, chunk_size)
    return _encode_chunks(chunks, process_tokenizer, chunk_size)

def multiprocessing_encoding(texts, chunk_size=512, max_workers=8):
    num_workers = min(max_workers, cpu_count())
    args = [(text, chunk_size) for text in texts]
    with Pool(num_workers) as p:
        return p.map(_single_encoding, args)
