from collections import defaultdict
from typing import Optional
import torch
import time
from functools import wraps

def run_avg_dict_update(d, key, new_value): 
    if key not in d: 
        d[key] = new_value 
    else: 
        d[key] = (new_value + d[key]) / 2
    return d 

def run_avg_dict_merge(d1, d2): 
    for key in d2: 
        if key not in d1: 
            d1[key] = d2[key] 
        else: 
            d1[key] = (d2[key] + d1[key]) / 2
    return d1

def add_dict_merge(d1, d2):
    for key in d2:
        if key not in d1: 
            d1[key] = d2[key]
        else:
            d1[key] += d2[key]
    return d1

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def _prep_vocabulary_addition(self, input_ids, tokens_to_group, group_token_positions):
    
    # Vocabulary Addition
    embed_cache = {} # key: token ids  values: tensor of embedding vectors 
    project_cache = {} # key: token ids  values: tensor of projection vectors 
    token_addition = defaultdict(int) # key: tuple of group token ids : counts of group tokens 

    # Representations to initialize wte embedding 
    full_reps = self.model.get_representation(input_ids)
    rep_layer_idx = -1 
    reps = full_reps[rep_layer_idx]
    
    for row_idx in range(len(input_ids)):
        input_ids_row = input_ids[row_idx]
        tokens_to_group_row = tokens_to_group[row_idx]
        group_positions = group_token_positions[row_idx]
        
        eom_token_ids, pair_token_groups, pair_token_positions = self.tokenizer_copy.add_tokens(tokens_to_group_row, group_positions, in_place=True)
        eom_positions = [p[-1] for p in pair_token_positions]

        reps_row = reps[row_idx] # wte row initialization vectors 
        
        embeddings_row = self.model.transformer.wte.weight[eom_token_ids].detach()
        projects_row = self.model.lm_head.weight[eom_token_ids].detach() # lm_head row initialization vectors
        
        # tokens_to_group_row is shorter than eom_token_ids, which represent index of pairwise merge token
        # e.g. tokens_to_group_row = [(1, 2, 3)], eom_token_ids = [2, 3]
        
        for token_tuple in tokens_to_group_row: 
            token_tuple = tuple(token_tuple)
            token_addition[token_tuple] += 1
            
        for eom_token_id, embed_vec, project_vec in zip(eom_token_ids, embeddings_row, projects_row):            
            project_cache = run_avg_dict_update(project_cache, eom_token_id, project_vec)
            embed_cache = run_avg_dict_update(embed_cache, eom_token_id, embed_vec)
            
    return token_addition, embed_cache, project_cache


def _prep_vocabulary_removal(tokens_to_remove):     
    token_removal = defaultdict(int)  # key: tuple of group token ids : counts of group tokens 
    for row_idx in range(len(tokens_to_remove)):
        for token_id in tokens_to_remove[row_idx]:
            token_removal[token_id.item()] += 1
    return token_removal


@timing_decorator
def _cache_vocabulary_change(self, texts=None, input_ids=None, target_ids=None):
    """Prepares vocabulary change for text batch"""
    
    t0 = time.time()
    # Get all required data in one inference pass
    res = self.inference(text=texts, input_ids=input_ids, target_ids=target_ids)
    input_ids, token_ids, token_perplexity, char_token_mask = (
        res['input_ids'], res['token_ids'], 
        res['token_perplexity'], res['char_token_mask']
    )
    print(f"Inference took: {time.time() - t0:.4f} seconds")

    t1 = time.time()
    # Process removals and groupings in parallel if possible
    tokens_to_remove, remove_positions, remove_mask, remove_groups = self._detect_remove_tokens(
        token_ids, token_perplexity, char_token_mask
    )
    print(f"Remove token detection took: {time.time() - t1:.4f} seconds")
    
    t2 = time.time()
    tokens_to_group, group_masks, token_groups, group_positions = self._detect_group_tokens(
        token_ids, token_perplexity, char_token_mask
    )
    print(f"Group token detection took: {time.time() - t2:.4f} seconds")

    t3 = time.time()
    # Process vocabulary additions
    token_addition, embed_cache, project_cache = _prep_vocabulary_addition(
        self, input_ids, tokens_to_group, group_positions
    )
    print(f"Vocabulary addition prep took: {time.time() - t3:.4f} seconds")

    t4 = time.time()
    # Filter tokens efficiently using sets
    leaf_token_set = set(self.tokenizer_copy.leaf_token_ids)
    filtered_tokens_to_remove = [
        [i for i in row if i in leaf_token_set] 
        for row in tokens_to_remove
    ]
    token_removal = _prep_vocabulary_removal(filtered_tokens_to_remove)
    print(f"Token removal prep took: {time.time() - t4:.4f} seconds")
    
    t5 = time.time()
    # Update caches efficiently using dict operations
    self.embed_cache = run_avg_dict_merge(self.embed_cache, embed_cache)
    self.project_cache = run_avg_dict_merge(self.project_cache, project_cache)
    self.token_addition = add_dict_merge(self.token_addition, token_addition)
    self._token_removal = add_dict_merge(self._token_removal, token_removal)
    print(f"Cache updates took: {time.time() - t5:.4f} seconds")

def remove_token_wte(wte, token_ids):
    # Get current weights
    weights = wte.weight.data
    # Remove the specified row
    mask = torch.ones(weights.size(0), dtype=torch.bool)
    mask[token_ids] = False
    new_weights = weights[mask]
    
    # Create new embedding layer with updated size
    new_embedding = torch.nn.Embedding(new_weights.size(0), weights.size(1))
    new_embedding.weight.data = new_weights
    
    return new_embedding

def remove_token_lm_head(lm_head, token_ids):
    # Get current weights
    weights = lm_head.weight.data
    # Remove the specified row
    mask = torch.ones(weights.size(0), dtype=torch.bool)
    mask[token_ids] = False
    new_weights = weights[mask]
    
    # Create new linear layer with updated size
    new_lm_head = torch.nn.Linear(weights.size(1), new_weights.size(0), bias=lm_head.bias is not None)
    new_lm_head.weight.data = new_weights
    if lm_head.bias is not None:
        new_bias = lm_head.bias.data[mask]
        new_lm_head.bias.data = new_bias
    
    return new_lm_head

def add_token_wte(wte, new_embedding_vectors):
    # Get current weights
    weights = wte.weight.data
    
    # Ensure new_embedding_vectors has shape (x, hidden_dim)
    if len(new_embedding_vectors.shape) == 1:
        new_embedding_vectors = new_embedding_vectors.unsqueeze(0)
    
    # Add new rows
    new_weights = torch.cat([weights, new_embedding_vectors], dim=0)
    
    # Create new embedding layer with updated size
    new_embedding = torch.nn.Embedding(new_weights.size(0), weights.size(1))
    new_embedding.weight.data = new_weights
        
    return new_embedding

def add_token_lm_head(lm_head, new_projection_vector: Optional[torch.Tensor]):
    # Get current weights
    weights = lm_head.weight.data
    
    if len(new_projection_vector.shape) == 1: 
        new_projection_vector = new_projection_vector.unsqueeze(0)
        
    new_weights = torch.cat([weights, new_projection_vector], dim=0)

    # Create new linear layer with updated size
    new_lm_head = torch.nn.Linear(weights.size(1), new_weights.size(0), bias=lm_head.bias is not None)
    new_lm_head.weight.data = new_weights
    if lm_head.bias is not None:
        new_bias = torch.cat([lm_head.bias.data, torch.zeros(1, device=lm_head.bias.device)], dim=0)
        new_lm_head.bias.data = new_bias
    
    return new_lm_head

def update_model(self, new_wte, new_lm_head):
    self.model.transformer.wte = new_wte
    self.model.lm_head = new_lm_head

@timing_decorator
def add_to_vocab(self, max_size_change: int = 500):
    # tokenizer addition 
    # sort by counts, prioritize frequent groups 
    tokens_to_group = sorted(self.token_addition.keys(), key=lambda x: self.token_addition[x], reverse=True)[:max_size_change]
    eom_tokens, pair_token_groups = self.tokenizer.add_tokens(tokens_to_group, in_place=True)

    print(f":: Total {len(tokens_to_group)} token groups, added {len(pair_token_groups)} pairwise merges")
    print(f":: Total {len(eom_tokens)} new tokens added")
    
    # wte addition 
    embed_vecs = torch.stack([self.embed_cache[id] for id in eom_tokens])
    new_wte = add_token_wte(self.model.transformer.wte, embed_vecs)

    # project addition 
    project_vecs = torch.stack([self.project_cache[id] for id in eom_tokens])
    new_lm_head = add_token_lm_head(self.model.lm_head, project_vecs)
    
    update_model(self, new_wte, new_lm_head)
    
    
@timing_decorator
def remove_from_vocab(self, max_size_change: int = 500): 
    tokens_to_remove = sorted(self.token_removal.keys(), key=lambda x: self.token_removal[x], reverse=True)[:max_size_change]
    print(f":: Total {len(tokens_to_remove)} tokens to remove")
    self.tokenizer.remove_tokens(tokens_to_remove)
    new_wte = remove_token_wte(self.model.transformer.wte, tokens_to_remove)
    new_lm_head = remove_token_lm_head(self.model.lm_head, tokens_to_remove)
    update_model(self, new_wte, new_lm_head)