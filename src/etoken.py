from .base import * 

  
class ETokenizer(Tokenizer): 
    """ 
    ETokenizer is a tokenizer that could be updated on-the-fly 
    """
    def __init__(self, char_vocab=False):
        super().__init__(char_vocab)
        self.use_char = bool(char_vocab)
        self.byte2idx = {b.encode("utf-8")[0]:i for i, b in self.vocab.items()}
        
    def train(self, text, vocab_size, verbose=False):
        """ 
        I would need to change this :: it already assume 'co-occurance based token merging'
        KeyPoint for 'merges' dictionary: 
        - pair of token ids : new token id 
        - order of pair in merges: 'better/frequent' merge first
        - during encoding we also favor earlier merges
        """
        raise NotImplementedError


    def decode(self, ids):
        # given ids (list of integers), return Python string
        if self.use_char:
            return "".join(self.vocab[idx] for idx in ids)
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """ 
        Might need to change this :: 
        - The vocab is no longer constructued via co-occurance pattern but rather joint perplexity level 
        - The encoding should also assume a different mechanism than checking 'co-occurance' statistics
        """
        text_bytes = text.encode("utf-8")
        ids = [self.byte2idx[b] for b in text_bytes]
            
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # respect order of 'merges' (trained tokenization)
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


    def add_tokens(self, tokens_to_group):
        """
        Add new tokens by grouping existing tokens together
        tokens_to_group[idx] = [token_1, token_2, ..., token_n]
        - progressively combine 2 tokens at a time (1-2, 12-3, 123-4, ...)
        """
        for token_group in tokens_to_group:

            if not all(t in self.vocab for t in token_group):
                raise ValueError(f"All tokens in group must exist in vocabulary: {token_group}")
            
            if len(token_group) == 1:
                continue
            
            prefix_token_idx = token_group[0]
            prefix_token = self.vocab[prefix_token_idx]
            
            for curr_token_idx in token_group[1:]: 
                
                new_token = prefix_token + self.vocab[curr_token_idx]
                new_idx = max(self.vocab.keys()) + 1  # maximum token idx plus one | not continuous
                print(" :: Adding new token: ", new_token, " with idx: ", new_idx)
                self.vocab[new_idx] = new_token
                self.merges[tuple([prefix_token_idx, curr_token_idx])] = new_idx

                prefix_token = new_token
                
                
    def remove_tokens(self, tokens_to_split):
        """
        Remove tokens from vocabulary if their byte length is > 1
        Maintains consecutive token indices by remapping remaining tokens
        """
        # First, remove the tokens and associated merges
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
        
        if not tokens_removed:
            return

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
            if idx not in tokens_removed:
                new_merges[(old_to_new.get(t1, t1), old_to_new.get(t2, t2))] = old_to_new[idx]
        
        self.vocab = new_vocab
        self.merges = new_merges
        
        print(" :: Removed tokens: ", tokens_removed)