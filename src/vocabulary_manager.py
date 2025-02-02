# appears to be a better name tha 'update'
import torch 
from .grouping import detect_spike_token, detect_group_token, get_spike_token_mask, get_group_token_mask, get_remove_token_mask
from .embed import add_token_wte, remove_token_wte, add_token_lm_head, remove_token_lm_head
from .utils import calculate_bits_per_char, shift_token_loss, map_token_to_char_perplexity
from .vis import visualize_text_multiline

class VocabularyManager:
    """Manages joint updates to both model vocabulary and tokenizer vocabulary"""
    
    def __init__(self, model, tokenizer, spike_quantile_threshold=0.8, group_quantile_threshold=0.6):
        self.model = model
        self.tokenizer = tokenizer
        self.spike_quantile_threshold = spike_quantile_threshold
        self.group_quantile_threshold = group_quantile_threshold
        
        
    def inference(self, text): 
        """ 
        Miscellaneous results from model inference
        """
        token_ids = torch.tensor(self.tokenizer.encode(text)).view(1, -1)
        input_ids, target_ids = token_ids[:, :-1], token_ids[:, 1:]
        logits, token_loss = self.model(input_ids, targets=target_ids, reduction='none') # loss is provided as an 'average' loss per token --- I want singular loss per token 
        
        decode = lambda x: self.tokenizer.decode(x)
        bpc_loss = calculate_bits_per_char(token_loss, target_ids, decode)
        token_perplexity = shift_token_loss(token_loss)
        char_perplexity = map_token_to_char_perplexity(text, token_ids, token_perplexity, decode)
        return {"input_ids": input_ids, "token_ids": token_ids, "token_perplexity": token_perplexity, "bpc_loss": bpc_loss, "char_perplexity": char_perplexity}

        
    def update_vocabulary(self, text):
        """Updates both model and tokenizer vocabularies based on perplexity patterns"""
        
        res = self.inference(text)
        input_ids, token_ids, token_perplexity, bpc_loss, char_perplexity = res.values()
        
        # 1. Detect tokens to add/remove
        tokens_to_split = self._detect_spike_tokens(token_ids, token_perplexity)
        tokens_to_remove = self.tokenizer.identify_splittable_tokens(tokens_to_split)
        
        tokens_to_group, group_positions = self._detect_group_tokens(token_ids, token_perplexity)
        
        # 2. Update tokenizer vocabulary and get end-of-merge positions
        self.tokenizer.remove_tokens(tokens_to_remove)
        eom_tokens, eom_positions = self.tokenizer.add_tokens(
            tokens_to_group, 
            group_positions=group_positions,
            return_eom=True
        )
        
        # 3. Get embeddings for new tokens using end-of-merge positions
        representations = self.model.get_representation(input_ids).detach()
        group_token_embeddings = representations[-1][0, eom_positions]
        
        # 4. Update model weights
        new_wte = self._update_word_embeddings(
            tokens_to_remove=tokens_to_remove,
            tokens_to_add=group_token_embeddings
        )
        
        new_lm_head = self._update_language_model_head(
            tokens_to_remove=tokens_to_remove,
            eom_tokens=eom_tokens
        )
        
        # 5. Update model layers
        self.model.transformer.wte = new_wte
        self.model.lm_head = new_lm_head
        
        return self.model, self.tokenizer


    def visualize_changes(self, text): 
        """Visualizes the changes in perplexity before and after updating the vocabulary"""
        res = self.inference(text)
        token_ids, token_perplexity = res['token_ids'], res['token_perplexity']
        decode = lambda x: self.tokenizer.decode(x)
        
        # (a). Spiking token visualization | TBD: in the future there will be large tokens spanning a few characters to be splitted (!)
        spike_color = 'pink'
        spike_token_mask, spike_token_groups = get_spike_token_mask(token_perplexity, quantile_threshold=self.spike_quantile_threshold, color=spike_color)
        char_perplexity, char_colors, groups = map_token_to_char_perplexity(text, token_ids, token_perplexity, decode, spike_token_mask, spike_token_groups, mask_color=spike_color)
        visualize_text_multiline(text, char_colors, groups=groups, max_chars_per_row=60, title='Spiking Token')

        # (b). Remove token visualization 
        remove_quantile_threshold = 0.8
        remove_color = 'orange'
        remove_token_mask, remove_token_groups = get_remove_token_mask(token_ids, token_perplexity, self.tokenizer, quantile_threshold=remove_quantile_threshold, color=remove_color)
        char_perplexity, char_colors, groups = map_token_to_char_perplexity(text, token_ids, token_perplexity, decode, remove_token_mask, remove_token_groups, mask_color=remove_color)
        visualize_text_multiline(text, char_colors, groups=groups, max_chars_per_row=60, title='Tokens to Remove')


        # (c). Group token visualization 
        group_quantile_threshold = 0.4
        group_color = 'lightgreen'
        group_token_mask, token_groups = get_group_token_mask(token_perplexity, quantile_threshold=group_quantile_threshold, color=group_color)
        # take in token groups and convert it into char_groups for visualization 
        char_perplexity, char_colors, groups = map_token_to_char_perplexity(text, token_ids, token_perplexity, decode, group_token_mask, token_groups, mask_color=group_color)
        visualize_text_multiline(text, char_colors, groups=groups, max_chars_per_row=60, title='Group Token')

    
    def _detect_spike_tokens(self, token_ids, token_perplexity):
        """Identifies tokens with unusually high perplexity"""
        return detect_spike_token(
            token_ids, 
            token_perplexity,
            quantile_threshold=self.spike_quantile_threshold
        )

    def _detect_group_tokens(self, token_ids, token_perplexity):
        """Identifies sequences of tokens that should be merged"""
        return detect_group_token(
            token_ids,
            token_perplexity, 
            quantile_threshold=self.group_quantile_threshold,
            return_indices=True
        )

    def _update_word_embeddings(self, tokens_to_remove, tokens_to_add):
        """Updates the token embedding matrix"""
        temp_wte = add_token_wte(self.model.transformer.wte, tokens_to_add)
        return remove_token_wte(temp_wte, tokens_to_remove)

    def _update_language_model_head(self, tokens_to_remove, eom_tokens):
        """Updates the language model output layer using end-of-merge tokens"""
        temp_lm_head = add_token_lm_head(self.model.lm_head, init_indices=eom_tokens)
        return remove_token_lm_head(temp_lm_head, tokens_to_remove)  