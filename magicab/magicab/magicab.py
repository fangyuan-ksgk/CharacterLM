# appears to be a better name than 'update'
import os 
import torch 
from typing import Optional
from .grouping import detect_spike_token, detect_group_token, get_spike_token_mask, get_group_token_mask, get_remove_token_mask
from .utils import calculate_bits_per_char, shift_token_loss, short_one, inference
from .vis import visualize_text_multiline
from .grouping import detect_spike_token_batch, detect_remove_token_batch, detect_group_token_batch
from .utils import prep_char_perplexity_batch, get_char_perplexity_batch
from .vocab_update import _cache_vocabulary_change, add_to_vocab, remove_from_vocab

class Magicab:
    """Manages joint updates to both model vocabulary and tokenizer vocabulary"""
    
    def __init__(self, model, tokenizer, spike_quantile_threshold=0.8, group_quantile_threshold=0.6, 
                 checkpoint_dir: str = "checkpoint/base"):
        self.model = model
        self.tokenizer = tokenizer
        self.spike_quantile_threshold = spike_quantile_threshold
        self.group_quantile_threshold = group_quantile_threshold
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = os.path.join(checkpoint_dir, "log")
        self.reset_update_info()
        os.makedirs(self.log_dir, exist_ok=True)
        
    def reset_update_info(self): 
        self.embed_cache = {}
        self.project_cache = {}
        self.token_addition = {}
        self.token_removal = {}
        self.tokenizer_copy = self.tokenizer.copy()
        
    def inference(self, text = None, input_ids = None, target_ids = None, pad: bool = False): 
        return inference(self.model, self.tokenizer, text, input_ids, target_ids, pad)
    
    def cache_vocab_change(self, text = None, input_ids = None, target_ids = None, pad: bool = False):         
        _cache_vocabulary_change(self, text, input_ids, target_ids)
        
    def update_vocab(self):
        """Updates both model and tokenizer vocabularies based on perplexity patterns"""
        add_to_vocab(self)
        remove_from_vocab(self)
        self.reset_update_info()

    def visualize_changes(self, texts = None, input_ids = None, target_ids = None, file_name: str = "demo"): 
        
        """Visualizes the changes in perplexity before and after updating the vocabulary"""
        
        res = self.inference(texts, input_ids, target_ids)
        
        texts, token_ids, token_perplexity, char_token_mask = res['texts'], res['token_ids'], res['token_perplexity'], res['char_token_mask']
        decode = lambda x: self.tokenizer.decode(x)

        # (a). Spiking token visualization
        spike_color = 'pink'
        spike_token_indices, spike_token_mask, spike_token_groups = detect_spike_token_batch(token_perplexity, quantile_threshold=self.spike_quantile_threshold, color=spike_color, char_token_mask=char_token_mask)
        char_perplexity, char_colors, char_groups = prep_char_perplexity_batch(texts, token_ids, token_perplexity, spike_token_mask, spike_token_groups, char_token_mask, decode, mask_color=spike_color)

        file_name = "spike"

        for text, char_color, group in zip(texts, char_colors, char_groups): 
            visualize_text_multiline(text, char_color, group, max_chars_per_row=60, title='Spiking Token', output_path=os.path.join(self.log_dir, f"{file_name}_spike_{text}.png"))
            
            
        # (b). Remove token visualization 
        remove_color = 'orange'
        tokens_to_remove, remove_token_indices, remove_token_mask, remove_token_groups = detect_remove_token_batch(token_ids, token_perplexity, self.tokenizer, quantile_threshold=self.spike_quantile_threshold, color=remove_color, char_token_mask=char_token_mask)
        char_perplexity, char_colors, char_groups = prep_char_perplexity_batch(texts, token_ids, token_perplexity, remove_token_mask, remove_token_groups, char_token_mask, decode, mask_color=remove_color)

        file_name = "remove"
        for text, char_color, group in zip(texts, char_colors, char_groups): 
            visualize_text_multiline(text, char_color, group, max_chars_per_row=60, title='Remove Token', output_path=os.path.join(self.log_dir, f"{file_name}_remove_{text}.png"))
            
            
        # (c). Group token visualization 
        group_color = 'lightgreen'
        tokens_to_group, group_token_mask, token_groups, group_token_positions = detect_group_token_batch(token_ids, token_perplexity, quantile_threshold=self.group_quantile_threshold, color=group_color, char_token_mask=char_token_mask)
        char_perplexity, char_colors, char_groups = prep_char_perplexity_batch(texts, token_ids, token_perplexity, group_token_mask, token_groups, char_token_mask, decode, mask_color=group_color)

        file_name = "group"
        for text, char_color, group in zip(texts, char_colors, char_groups): 
            visualize_text_multiline(text, char_color, group, max_chars_per_row=60, title='Group Token', output_path=os.path.join(self.log_dir, f"{file_name}_group_{text}.png"))
            
    def _detect_spike_tokens(self, token_ids, token_perplexity, char_token_mask):
        """Identifies tokens with unusually high perplexity"""
        return detect_spike_token_batch(
            token_ids, 
            token_perplexity,
            quantile_threshold=self.spike_quantile_threshold,
            char_token_mask=char_token_mask
        )
        
    def _detect_remove_tokens(self, token_ids, token_perplexity, char_token_mask):
        """Identifies tokens with unusually high perplexity"""
        tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups =  detect_remove_token_batch(
            token_ids, 
            token_perplexity,
            self.tokenizer,
            quantile_threshold=self.spike_quantile_threshold,
            char_token_mask=char_token_mask
        )
            
        return tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups

    def _detect_group_tokens(self, token_ids, token_perplexity, char_token_mask):
        """Identifies sequences of tokens that should be merged"""
        return detect_group_token_batch(
            token_ids, 
            token_perplexity, 
            quantile_threshold=self.group_quantile_threshold,
            char_token_mask=char_token_mask
        )

    def _update_word_embeddings(self, tokens_to_remove, tokens_to_add):
        """Updates the token embedding matrix"""
        temp_wte = add_token_wte(self.model.transformer.wte, tokens_to_add)
        return remove_token_wte(temp_wte, tokens_to_remove)

    def _update_language_model_head(self, tokens_to_remove, eom_tokens):
        """Updates the language model output layer using end-of-merge tokens"""
        temp_lm_head = add_token_lm_head(self.model.lm_head, init_indices=eom_tokens)
        return remove_token_lm_head(temp_lm_head, tokens_to_remove)  
    
    def sanity_check(self, texts = ["Hello, world!"]):
        """Performs sanity checks on vocabulary sizes across tokenizer and model components"""
        tokenizer_size_match = (len(self.tokenizer.merges) + 
                              len(self.tokenizer.char_vocab) + 
                              len(self.tokenizer.special_ids) == len(self.tokenizer.vocab))
        
        print("Tokenizer vocab_size matching:", tokenizer_size_match)
        print("Tokenizer vocab_size:", len(self.tokenizer.vocab))
        print("LM Head vocab_size:", self.model.lm_head.weight.shape[0])
        print("WTE vocab_size:", self.model.transformer.wte.weight.shape[0])
        
        all_sizes_match = (self.model.transformer.wte.weight.shape[0] == 
                          self.model.lm_head.weight.shape[0] == 
                          len(self.tokenizer.vocab))
        print("ALL Vocab Size Matching Sanity Check:", all_sizes_match)
        
        
        res = self.inference(text=texts)
        token_ids = res['token_ids']
        token_perplexity = res['token_perplexity']
        char_token_mask = res['char_token_mask']
        
        # Attentoin: we obtain 'token positions' not token ids for spike, remove and group tokens (!)
        for row_idx in range(len(texts)): 
            self.tokenizer.sanity_check(texts[row_idx])
        print(":: Tokenizer encode & decode equivalaence check passed")
        
        # spike
        spike_color = 'pink'
        spike_token_positions, spike_token_mask, spike_token_groups = detect_spike_token_batch(token_perplexity, quantile_threshold=self.spike_quantile_threshold, color=spike_color, char_token_mask=char_token_mask)

        for token_id, token_loss, char_mask, spike_token_pos in zip(token_ids, token_perplexity, char_token_mask, spike_token_positions): 
            threshold = torch.quantile(token_loss, self.spike_quantile_threshold)
            assert (token_loss[spike_token_pos] > threshold).all().item(), "Spike token should have perplexity above threshold"
            assert char_mask[spike_token_pos].all().item(), "Spike token should not be a special token"
            assert spike_token_pos.max() < len(token_id), "Spike token position is out of bound"
        print(":: Spike Token Sanity Check Passed")


        # remove 
        remove_color = 'orange'
        tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups = detect_remove_token_batch(token_ids, token_perplexity, self.tokenizer, quantile_threshold=self.spike_quantile_threshold, color=remove_color, char_token_mask=char_token_mask)
        
        char_ids = torch.tensor(list(self.tokenizer.char_vocab.keys()))
        base_char_mask = torch.isin(token_ids, char_ids)
        leaf_token_mask = torch.isin(token_ids, torch.tensor(list(self.tokenizer.leaf_token_ids)))
        
        for (tokens, remove_mask, base_mask, char_mask, token_loss) in zip(token_ids,
                                                                   remove_token_mask, 
                                                                   base_char_mask, 
                                                                   char_token_mask,
                                                                   token_perplexity):
            threshold = torch.quantile(token_loss, self.spike_quantile_threshold)
            assert (token_loss[remove_mask] > threshold).all().item(), "token_loss is not greater than threshold"
            assert (base_mask & remove_mask).any().item() == False, "base character tokens are removed"
            assert (remove_mask & ~char_mask).any().item() == False, "special tokens are removed"
            assert remove_mask.max() < len(token_id), "remove token position is out of bound"
            for row_idx in range(len(tokens_to_remove)): 
                assert all([i in self.tokenizer.leaf_token_ids for i in tokens_to_remove[row_idx]]), "leaf tokens are removed"            
            assert all([i in self.tokenizer.leaf_token_ids for i in self.token_removal]), "leaf tokens are removed"

        print(":: Remove Token Sanity Check Passed")
                

        group_color = 'lightgreen'
        tokens_to_group, group_token_mask, token_groups, group_token_positions = detect_group_token_batch(token_ids, token_perplexity, quantile_threshold=self.group_quantile_threshold, color=group_color, char_token_mask=char_token_mask)

        # group
        for token_id, token_loss, char_mask, group_token_ids in zip(token_ids, token_perplexity, char_token_mask, group_token_positions): 
            for group in group_token_ids: 
                assert (token_loss[group[1:]] < threshold).all().item(), "Group token should have perplexity below threshold"
                assert (token_loss[group[1:]] < token_loss[group[:-1]]).all().item(), "Group token should have decreasing perplexity"        
                assert torch.tensor(group).max() < len(token_id), "group token position is out of bound"
                
        print(":: Group Token Sanity Check Passed")
        
        
        return tokenizer_size_match and all_sizes_match