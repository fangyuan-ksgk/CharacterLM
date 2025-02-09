from .base_tok import get_valid_stats, merge, Tokenizer
from .base_tok import _remove_tokens 
import json, re
from tqdm import tqdm  # Add this import at the top of the file
from rust_tokenizer import PyETokenizer
from copy import deepcopy
import time

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

    def add_token(self, token: str, token_id: int = None):
        if token in self.token2id:
            return self.token2id[token]

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

        return token_id

    def add_merge(self, id1: int, id2: int):
        token1 = self.id2token[id1]
        token2 = self.id2token[id2]
        merged = token1 + token2
        merged_id = self.add_token(merged)
        self.merges[(id1, id2)] = merged_id
        return merged_id

    def find_token(self, token: str) -> int:
        return self.token2id.get(token)

    def get_merge_result(self, id1: int, id2: int) -> int:
        return self.merges.get((id1, id2))
    
    
  
class ETokenizer: 
    """ 
    ETokenizer is a tokenizer that could be updated on-the-fly 
    """
    def __init__(self, char_vocab=False):

        self.special_tokens = ["<|endoftext|>", "<pad>"]
        self._init_token_trie(char_vocab)
        
        self.use_char = bool(char_vocab)
        self.char_vocab = char_vocab
        self.char2idx = {c:i for i, c in char_vocab.items()}
        self.special_ids = list(self.special2idx.values())
        self.token_trie = TokenTrie()
        
        # Initialize with char_vocab
        for idx, char in char_vocab.items():
            self.token_trie.add_token(char, idx)
            
    def _init_token_trie(self, char_vocab=False):
        # Initialize TokenTrie with basic vocabulary
        self.token_trie = TokenTrie()
        self.special2idx = {}
        
        # Add byte-level or character vocabulary
        if not char_vocab:
            # Basic byte-level vocabulary (size 256)
            for idx in range(256):
                self.token_trie.add_token(bytes([idx]), idx)
        else:
            # Use provided character vocabulary
            for idx, token in char_vocab.items():
                self.token_trie.add_token(token, idx)
        
        # Add special tokens
        vocab_size = 256 if not char_vocab else len(char_vocab)
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
        return len(self.vocab)
    
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
        if self.use_char:
            return "".join(self.vocab[idx] for idx in ids)
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text 
    
    def _encode_python(self, ids): 
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
    
    def _encode(self, ids): 
        """ 
        Rust speed-up encoding
        """
        return self.rust_tokenizer.encode(ids)
    
    
    def encode(self, text):
        """ 
        Exhaustive Encoding | Byte level vocabulary base --- we need character-level vocabulary
        """
        ids = self.encode_char(text)

        return self._encode(ids)
            
            
    def _process_token_pair(self, prefix_token_idx, curr_token_idx, *args):
        # Check if merge already exists
        merged_id = self.token_trie.get_merge_result(prefix_token_idx, curr_token_idx)
        if merged_id is not None:
            return merged_id, self.token_trie.next_id, None

        # Create new merge
        merged_id = self.token_trie.add_merge(prefix_token_idx, curr_token_idx)
        return merged_id, self.token_trie.next_id, (merged_id, curr_token_idx)

    def _add_tokens(self, tokens_to_group, group_positions=None):
        vocab = dict(self.vocab)
        merges = dict(self.merges)
        next_idx = max(vocab.keys()) + 1
        
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
                merged_id, next_idx, pair_info = self._process_token_pair(
                    prefix_idx, curr_idx, vocab, merges, next_idx
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
    
        return vocab, merges, eom_tokens, pair_token_groups, pair_token_positions

    def add_tokens(self, tokens_to_group, group_positions=None, in_place=False):
     
        vocab, merges, eom_tokens, pair_token_groups, pair_token_positions = self._add_tokens(tokens_to_group, group_positions)
                
        if in_place: 
            self.token_trie.id2token = vocab
            self.token_trie.merges = merges
        
        if group_positions is not None:
            return eom_tokens, pair_token_groups, pair_token_positions
        else:
            return eom_tokens, pair_token_groups
        
        
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
            
        # Remove from id2token and token2id
        for token_id in tokens_to_remove:
            if token_id in self.token_trie.id2token:
                token = self.token_trie.id2token[token_id]
                del self.token_trie.id2token[token_id]
                del self.token_trie.token2id[token]
        
        # Remove from merges
        self.token_trie.merges = {
            pair: idx for pair, idx in self.token_trie.merges.items()
            if idx not in tokens_to_remove
        }

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
    
    def save(self, path):
        # Convert data to JSON-serializable format
        data = {
            'char_vocab': {str(k): v for k, v in self.char_vocab.items()},
            'id2token': {str(k): v for k, v in self.token_trie.id2token.items()},
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.token_trie.merges.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        char_vocab = {int(k): v for k, v in data['char_vocab'].items()}
        inst = cls(char_vocab)
        
        # Restore tokens and merges
        for k, v in data['id2token'].items():
            inst.token_trie.add_token(v, int(k))
        
        for k, v in data['merges'].items():
            id1, id2 = map(int, k.split(','))
            inst.token_trie.merges[(id1, id2)] = v
            
        return inst
    
    def sanity_check(self, text = "Hello, world!"):
        assert self.decode(self.encode(text)) == text, "encode & decode mismatch"
        print("Tokenizer has matching encoding & decoding")
        
        
    def copy(self):
        c = ETokenizer(deepcopy(self.char_vocab))
        c.token_trie = deepcopy(self.token_trie)
        return c