# appears to be a better name tha 'update'
from .grouping import detect_spike_token, detect_group_token
from .embed import add_token_wte, remove_token_wte, add_token_lm_head, remove_token_lm_head

class VocabularyManager:
    """Manages joint updates to both model vocabulary and tokenizer vocabulary"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def update_vocabulary(self, input_ids, token_ids, token_perplexity):
        """Updates both model and tokenizer vocabularies based on perplexity patterns"""
        
        # 1. Detect tokens to add/remove
        tokens_to_split = self._detect_spike_tokens(token_ids, token_perplexity)
        tokens_to_remove = self.tokenizer.identify_splittable_tokens(tokens_to_split)
        
        tokens_to_group, group_positions = self._detect_group_tokens(token_ids, token_perplexity)
        
        # 2. Get embeddings for new tokens
        eog_positions = [group[-1] for group in group_positions]
        representations = self.model.get_representation(input_ids)
        group_token_embeddings = representations[-1][0, eog_positions]
        
        # 3. Update model weights
        new_wte = self._update_word_embeddings(
            tokens_to_remove=tokens_to_remove,
            tokens_to_add=group_token_embeddings
        )
        
        new_lm_head = self._update_language_model_head(
            tokens_to_remove=tokens_to_remove,
            tokens_to_add=tokens_to_group
        )
        
        # 4. Update tokenizer vocabulary
        self.tokenizer.remove_tokens(tokens_to_remove)
        self.tokenizer.add_tokens(tokens_to_group)
        
        # 5. Update model layers
        self.model.transformer.wte = new_wte
        self.model.lm_head = new_lm_head
        
        return self.model, self.tokenizer

    def _detect_spike_tokens(self, token_ids, token_perplexity):
        """Identifies tokens with unusually high perplexity"""
        return detect_spike_token(
            token_ids, 
            token_perplexity,
            quantile_threshold=self.config.spike_quantile_threshold
        )

    def _detect_group_tokens(self, token_ids, token_perplexity):
        """Identifies sequences of tokens that should be merged"""
        return detect_group_token(
            token_ids,
            token_perplexity, 
            quantile_threshold=self.config.group_quantile_threshold,
            return_indices=True
        )

    def _update_word_embeddings(self, tokens_to_remove, tokens_to_add):
        """Updates the token embedding matrix"""
        temp_wte = add_token_wte(self.model.transformer.wte, tokens_to_add)
        return remove_token_wte(temp_wte, tokens_to_remove)

    def _update_language_model_head(self, tokens_to_remove, tokens_to_add):
        """Updates the language model output layer"""
        eog_indices = [tokens[-1] for tokens in tokens_to_add]
        temp_lm_head = add_token_lm_head(self.model.lm_head, init_indices=eog_indices)
        return remove_token_lm_head(temp_lm_head, tokens_to_remove) 