from src.grouping import detect_group_token, detect_spike_token
from src.embed import remove_token_lm_head, add_token_lm_head, remove_token_wte, add_token_wte

# TBD: simplfy and verify
def update_token(model, tok, input_ids, token_ids, token_perplexity, spike_quantile_threshold, group_quantile_threshold):
    
    # (a). Spike token | Split 'spike tokens' -- removing these tokens and use sub-token units instead 
    tokens_to_split = detect_spike_token(token_ids, token_perplexity, quantile_threshold=spike_quantile_threshold)
    tokens_to_remove = tok.identify_splittable_tokens(tokens_to_split)
    tok.remove_tokens(tokens_to_remove)

    # (b). Group token | Merge 'group tokens' -- adding these tokens to the tokenizer
    tokens_to_group, group_positions = detect_group_token(token_ids, token_perplexity, quantile_threshold=group_quantile_threshold, return_indices=True)
    tok.add_tokens(tokens_to_group)

    # (c). Update LLM 'wte' | Use end-of-group representations to initialize new token embeddings
    eog_positions = [group[-1] for group in group_positions] # end-of-group indices
    representations = model.get_representation(input_ids) 
    group_token_embeddings = representations[-1][0, eog_positions] # end-of-group representations 
    add_wte = add_token_wte(model.transformer.wte, group_token_embeddings)
    new_wte = remove_token_wte(add_wte, tokens_to_remove) # Assumption: weights are ordered by token-idx

    # (d). Update LLM 'lm_head' | Use end-of-group token projection to initialize new token projection (should be average of group projection instead?)
    eog_indices = [tokens[-1] for tokens in tokens_to_group] # end-of-group indices
    add_lm_head = add_token_lm_head(model.lm_head, init_indices=eog_indices)
    new_lm_head = remove_token_lm_head(add_lm_head, tokens_to_remove)
    
    model.transformer.wte = new_wte
    model.lm_head = new_lm_head
    return model, tok 