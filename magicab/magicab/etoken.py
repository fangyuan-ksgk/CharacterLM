from .base_tok import * 
import json 


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
    def inverse_vocab(self): 
        return {v:k for k, v in self.vocab.items()}
    
    @property 
    def vocab_size(self): 
        return len(self.vocab)

    def decode(self, ids):
        # given ids (list of integers), return Python string
        if self.use_char:
            return "".join(self.vocab[idx] for idx in ids)
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text 
    
    def encode(self, text):
        """ 
        Exhaustive Encoding | Byte level vocabulary base --- we need character-level vocabulary
        """
        ids = self.encode_char(text)

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


    def add_tokens(self, tokens_to_group, group_positions=None, return_eom=False):
        """
        Add new tokens by grouping existing tokens together
        tokens_to_group[idx] = [token_1, token_2, ..., token_n]
        - progressively combine 2 tokens at a time (1-2, 12-3, 123-4, ...)
        - check for pre-existing merges when adding new tokens
        - return end-of-merge token idx for wte/lm-head update (for each merges)
        - need to return group_idx & eom_position for initializing wte ...
        """
        eom_tokens = [] 
        group_indices = []
        in_group_positions = []
        
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
                    prefix_token_idx = new_idx
                    eom_tokens.append(curr_token_idx) # end-of-merge token idx
                    group_indices.append(group_idx)
                    in_group_positions.append(r)
                    
                # update pointers 
                l += 1
                r = min(r+1, length)
                
                # update prefix token
                prefix_token = new_token
                
        if return_eom and group_positions is not None:
            eom_positions = [group_positions[group_idx][in_group_position] for in_group_position, group_idx in zip(in_group_positions, group_indices)]
            return eom_tokens, eom_positions
        elif return_eom and group_positions is None:
            return eom_tokens
        
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
    
    def remove_tokens(self, tokens_to_remove): 
        # Create a new vocabulary with consecutive indices
        old_to_new = {}
        new_vocab = {}
        next_idx = 0
        
        # Sort keys to ensure deterministic remapping
        for old_idx in sorted(self.vocab.keys()):
            new_vocab[next_idx] = self.vocab[old_idx]
            old_to_new[old_idx] = next_idx
            next_idx += 1
        
        # Update merges with new token indices
        new_merges = {}
        for (t1, t2), idx in self.merges.items():
            if idx not in tokens_to_remove:
                new_merges[(old_to_new.get(t1, t1), old_to_new.get(t2, t2))] = old_to_new[idx]
        
        self.vocab = new_vocab
        self.merges = new_merges
                
    
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
        
        
        