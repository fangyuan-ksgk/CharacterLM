from .base_tok import get_valid_stats, merge, Tokenizer
from .base_tok import _remove_tokens 
import json, re
from tqdm import tqdm  # Add this import at the top of the file
from rust_tokenizer import PyETokenizer


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

  
class ETokenizer(Tokenizer): 
    """ 
    ETokenizer is a tokenizer that could be updated on-the-fly 
    """
    def __init__(self, char_vocab=False):
        super().__init__(char_vocab)
        
        self.use_char = bool(char_vocab)
        self.char_vocab = char_vocab
        self.char2idx = {c:i for i, c in char_vocab.items()}
        self.special_ids = list(self.special2idx.values())

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
            

    def add_tokens(self, tokens_to_group, group_positions=None, in_place=False):
        """
        Add new tokens by grouping existing tokens together
        tokens_to_group[idx] = [token_1, token_2, ..., token_n]
        - progressively combine 2 tokens at a time (1-2, 12-3, 123-4, ...)
        - check for pre-existing merges when adding new tokens
        - return end-of-merge token idx for wte/lm-head update (for each merges)
        - need to return group_idx & eom_position for initializing wte ...
        """
        from copy import deepcopy
        if not in_place: 
            orig_vocab = deepcopy(self.vocab)
            orig_merges = deepcopy(self.merges)
            
        eom_tokens = [] 
        pair_token_groups = []
        pair_token_positions = []
        
        for group_idx, token_group in enumerate(tokens_to_group):

            if not all(t in self.vocab for t in token_group):
                raise ValueError(f"All tokens in group must exist in vocabulary: {token_group}")
            
            if len(token_group) == 1:
                continue
            
            # The Correct Logic should be a two-pointer approach
            length = len(token_group) - 1
            l, r = 0, 1
            prefix_token_idx = token_group[l]
            prefix_token = self.vocab[prefix_token_idx]
            
            while l < r: 
                
                curr_token_idx = token_group[r]                
                curr_token = self.vocab[curr_token_idx]
                
                new_token = prefix_token + curr_token
                
                if new_token in self.inverse_vocab:
                    prefix_token_idx = self.inverse_vocab[new_token]
                    
                else:
                    new_idx = max(self.vocab.keys()) + 1  # maximum token idx plus one | assume consecutive token ids
                    self.vocab[new_idx] = new_token
                    self.merges[tuple([prefix_token_idx, curr_token_idx])] = new_idx 
                    if in_place:                    
                        print(f" :: Add new token {new_token}  Id: {new_idx}")
                    prefix_token_idx = new_idx
                    
                    eom_tokens.append(curr_token_idx) # end-of-merge token idx
                    pair_token_groups.append(tuple([prefix_token_idx, curr_token_idx]))
                    
                    if group_positions is not None:
                        pair_token_positions.append(tuple([group_positions[group_idx][l], group_positions[group_idx][r]]))
                    
                # update pointers 
                l += 1
                r = min(r+1, length)
                
                # update prefix token
                prefix_token = new_token
                
        if not in_place: 
            self.vocab = orig_vocab
            self.merges = orig_merges
        
        # in-place change on tokenizer should be avoided
        # eom_tokens : list of end-of-merge token indices
        # pair_token_groups : list of pairs of token indices
        # pair_token_positions : list of positions of pairs of tokens
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
        
        new_vocab, new_merges = _remove_tokens(self, tokens_to_remove)
        
        if in_place: 
            self.vocab = new_vocab
            self.merges = new_merges
        else: 
            return new_vocab, new_merges
        
                 
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
            'vocab': {str(k): v for k, v in self.vocab.items()},
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        # Save to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        # Load from file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert data back to original format
        char_vocab = {int(k): v for k, v in data['char_vocab'].items()}
        vocab = {int(k): v for k, v in data['vocab'].items()}
        merges = {tuple(map(int, k.split(','))): v 
                 for k, v in data['merges'].items()}
        
        # Create instance and set attributes
        inst = cls(char_vocab)
        inst.vocab = vocab
        inst.merges = merges
        return inst
    
    def sanity_check(self, text = "Hello, world!"):
        assert self.decode(self.encode(text)) == text, "encode & decode mismatch"
        print("Tokenizer has matching encoding & decoding")
        
        
    def copy(self):
        from copy import deepcopy
        c = ETokenizer(deepcopy(self.char_vocab))
        c.vocab = deepcopy(self.vocab)
        c.merges = deepcopy(self.merges)
        return c