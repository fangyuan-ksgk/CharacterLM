from .base import * 

  
class ETokenizer(Tokenizer): 
    """ 
    ETokenizer is a tokenizer that could be updated on-the-fly 
    """
    def __init__(self):
        super().__init__()
        

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
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # def encode(self, text):
    #     """ 
    #     Might need to change this :: 
    #     - The vocab is no longer constructued via co-occurance pattern but rather joint perplexity level 
    #     - The encoding should also assume a different mechanism than checking 'co-occurance' statistics
    #     """
    #     # given a string text, return the token ids
    #     text_bytes = text.encode("utf-8") # raw bytes
    #     ids = list(text_bytes) # list of integers in range 0..255
    #     while len(ids) >= 2:
    #         # find the pair with the lowest merge index
    #         stats = get_stats(ids)
    #         pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # favor more frequent co-occurance
    #         if pair not in self.merges:
    #             break # nothing else can be merged anymore
    #         # otherwise let's merge the best pair (lowest merge index)
    #         idx = self.merges[pair]
    #         ids = merge(ids, pair, idx)
    #     return ids
    
    def encode(self, text): # To be tested
        """
        Efficient encoding using pre-sorted merges list.
        Single pass through the sequence with a sliding window,
        always applying the earliest/best merge when possible.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        i = 0
        while i < len(ids) - 1:
            pair = (ids[i], ids[i + 1])
            if pair in self.merges:
                # Apply merge and back up one step if possible
                ids[i:i+2] = [self.merges[pair]]
                i = max(0, i - 1)  # back up to check for new merge opportunities
            else:
                i += 1
                
        return ids

    def add_tokens(self, tokens_to_group):
        """
        Add new tokens by grouping existing tokens together
        tokens_to_group[idx] = [token_1, token_2] | Note: restrict to 2 token merges in here 
        """
        for token_group in tokens_to_group:
            # Verify all tokens exist in vocabulary
            if not all(t in self.vocab for t in token_group):
                raise ValueError(f"All tokens in group must exist in vocabulary: {token_group}")
            
            new_bytes = b"".join(self.vocab[t] for t in token_group)
            new_idx = len(self.vocab) 
            self.vocab[new_idx] = new_bytes
            self.merges[tuple(token_group)] = new_idx
            
    def remove_tokens(self, tokens_to_split):
        """
        Remove tokens from vocabulary if their byte length is > 1
        """
        for token_id in tokens_to_split:
            if token_id not in self.vocab:
                continue
            
            # Only remove tokens that represent more than one byte
            if len(self.vocab[token_id]) > 1:
                # Remove from vocabulary
                del self.vocab[token_id]
                
                # Remove any merges that created this token
                self.merges = {pair: idx for pair, idx in self.merges.items() 
                              if idx != token_id}