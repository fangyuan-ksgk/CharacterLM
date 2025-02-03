# appears to be a better name than 'update'
import os 
import torch 
from typing import Optional
from .grouping import detect_spike_token, detect_group_token, get_spike_token_mask, get_group_token_mask, get_remove_token_mask
from .embed import add_token_wte, remove_token_wte, add_token_lm_head, remove_token_lm_head
from .utils import calculate_bits_per_char, shift_token_loss, map_token_to_char_perplexity, short_one, inference
from .vis import visualize_text_multiline
from .grouping import detect_spike_token_batch, detect_remove_token_batch, detect_group_token_batch
from .utils import map_batch_token_to_char_perplexity

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
        os.makedirs(self.log_dir, exist_ok=True)
        
    def inference(self, text = None, input_ids = None, target_ids = None, pad: bool = False): 
        return inference(self.model, self.tokenizer, text, input_ids, target_ids, pad)
        
    def update_vocabulary(self, text = None, input_ids = None, target_ids = None, pad: bool = False):
        """Updates both model and tokenizer vocabularies based on perplexity patterns"""
        
        res = self.inference(text, input_ids, target_ids, pad)
        input_ids, token_ids, token_perplexity = res['input_ids'], res['token_ids'], res['token_perplexity']
        
        tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups = self._detect_remove_tokens(token_ids, token_perplexity)      
        tokens_to_group, group_token_masks, token_groups, group_token_positions = self._detect_group_tokens(token_ids, token_perplexity)

        for input_ids_row, tokens_to_remove_row, tokens_to_group_row, group_token_positions_row in zip(input_ids, tokens_to_remove, tokens_to_group, group_token_positions): 
            
            input_ids_row = input_ids_row.unsqueeze(0)
            
            self.tokenizer.remove_tokens(tokens_to_remove_row)

            eom_tokens, eom_positions = self.tokenizer.add_tokens(
                tokens_to_group_row, 
                group_positions=group_token_positions_row,
                return_eom=True
            )
        
            representations = self.model.get_representation(input_ids_row)
            eom_input_positions = short_one(eom_positions)
            group_token_embeddings = representations[-1][0, eom_input_positions]
            
            new_wte = self._update_word_embeddings(
                tokens_to_remove=tokens_to_remove_row,
                tokens_to_add=group_token_embeddings
            )
            
            new_lm_head = self._update_language_model_head(
                tokens_to_remove=tokens_to_remove_row,
                eom_tokens=eom_tokens
            )
        
            self.model.transformer.wte = new_wte
            self.model.lm_head = new_lm_head
        
        return self.model, self.tokenizer

    def visualize_changes(self, texts = None, input_ids = None, target_ids = None, file_name: str = "demo"): 
        
        """Visualizes the changes in perplexity before and after updating the vocabulary"""
        
        res = self.inference(texts, input_ids, target_ids)
        
        texts, token_ids, token_perplexity = res['texts'], res['token_ids'], res['token_perplexity']
        decode = lambda x: self.tokenizer.decode(x)
        
        # (a). Spiking token visualization
        spike_color = 'pink'
        spike_token_indices, spike_token_mask, spike_token_groups = detect_spike_token_batch(token_perplexity, quantile_threshold=self.spike_quantile_threshold, color=spike_color)
        char_perplexity, char_colors, groups = map_batch_token_to_char_perplexity(texts, token_ids, token_perplexity, decode, spike_token_mask, spike_token_groups, mask_color=spike_color)
        file_name = "spike"

        for text, char_color, group in zip(texts, char_colors, groups): 
            visualize_text_multiline(text, char_color, group, max_chars_per_row=60, title='Spiking Token', output_path=os.path.join(self.log_dir, f"{file_name}_spike_{text}.png"))
            
            
        # (b). Remove token visualization 
        remove_quantile_threshold = 0.8
        remove_color = 'orange'
        remove_token_indices, remove_token_mask, remove_token_groups = detect_remove_token_batch(token_ids, token_perplexity, self.tokenizer, quantile_threshold=remove_quantile_threshold, color=remove_color)
        char_perplexity, char_colors, groups = map_batch_token_to_char_perplexity(texts, token_ids, token_perplexity, decode, remove_token_mask, remove_token_groups, mask_color=remove_color)

        file_name = "remove"
        for text, char_color, group in zip(texts, char_colors, groups): 
            visualize_text_multiline(text, char_color, group, max_chars_per_row=60, title='Remove Token', output_path=os.path.join(self.log_dir, f"{file_name}_remove_{text}.png"))
            
            
        # (c). Group token visualization 
        group_quantile_threshold = 0.8
        group_color = 'lightgreen'
        tokens_to_group, group_token_mask, token_groups, group_token_positions = detect_group_token_batch(token_ids, token_perplexity, quantile_threshold=group_quantile_threshold, color=group_color)
        # take in token groups and convert it into char_groups for visualization 
        char_perplexity, char_colors, groups = map_batch_token_to_char_perplexity(texts, token_ids, token_perplexity, decode, group_token_mask, token_groups, mask_color=group_color)

        file_name = "group"
        for text, char_color, group in zip(texts, char_colors, groups): 
            visualize_text_multiline(text, char_color, group, max_chars_per_row=60, title='Group Token', output_path=os.path.join(self.log_dir, f"{file_name}_group_{text}.png"))
            
    
    def _detect_spike_tokens(self, token_ids, token_perplexity):
        """Identifies tokens with unusually high perplexity"""
        return detect_spike_token(
            token_ids, 
            token_perplexity,
            quantile_threshold=self.spike_quantile_threshold
        )
        
    def _detect_remove_tokens(self, token_ids, token_perplexity):
        """Identifies tokens with unusually high perplexity"""
        remove_token_positions, remove_token_mask, remove_token_groups =  detect_remove_token_batch(
            token_ids, 
            token_perplexity,
            self.tokenizer,
            quantile_threshold=self.spike_quantile_threshold
        )
        
        tokens_to_remove = []
        for remove_token_mask_row, token_ids_row in zip(remove_token_mask, token_ids): 
            tokens_to_remove.append(token_ids_row[remove_token_mask_row])
            
        return tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups

    def _detect_group_tokens(self, token_ids, token_perplexity):
        """Identifies sequences of tokens that should be merged"""
        return detect_group_token_batch(
            token_ids, 
            token_perplexity, 
            quantile_threshold=self.group_quantile_threshold
        )

    def _update_word_embeddings(self, tokens_to_remove, tokens_to_add):
        """Updates the token embedding matrix"""
        temp_wte = add_token_wte(self.model.transformer.wte, tokens_to_add)
        return remove_token_wte(temp_wte, tokens_to_remove)

    def _update_language_model_head(self, tokens_to_remove, eom_tokens):
        """Updates the language model output layer using end-of-merge tokens"""
        temp_lm_head = add_token_lm_head(self.model.lm_head, init_indices=eom_tokens)
        return remove_token_lm_head(temp_lm_head, tokens_to_remove)  
    
    
    def sanity_check(self):
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
        
        return tokenizer_size_match and all_sizes_match