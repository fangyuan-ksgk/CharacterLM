from collections import defaultdict
from typing import Optional
import torch

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

def _prep_vocabulary_addition(self, input_ids, tokens_to_group, group_token_positions):
    
    # Vocabulary Addition
    wte_addition = {} # key: tuple of group token ids : tensor of initialization row vector
    head_addition = {} # key: tuple of group token ids : tensor of initialization row vector
    token_addition = defaultdict(int) # key: tuple of group token ids : counts of group tokens 

    # Representations to initialize wte embedding 
    full_reps = self.model.get_representation(input_ids)
    rep_layer_idx = -1 
    reps = full_reps[rep_layer_idx]

    for row_idx in range(len(input_ids)):
        input_ids_row = input_ids[row_idx]
        tokens_to_group_row = tokens_to_group[row_idx]
        eog_token_ids = [p[-1] for p in tokens_to_group_row] 
        group_positions = group_token_positions[row_idx]
        eog_positions = [p[-1] for p in group_positions]

        reps_row = reps[row_idx] # wte row initialization vectors 
        projects_row = self.model.lm_head.weight[eog_token_ids].detach() # lm_head row initialization vectors 

        for token_tuple, rep_row, project_row in zip(tokens_to_group_row, reps_row, projects_row):
            token_tuple = tuple(token_tuple)
            
            wte_addition = run_avg_dict_update(wte_addition, token_tuple, rep_row)
            head_addition = run_avg_dict_update(head_addition, token_tuple, project_row)
            token_addition[token_tuple] += 1
            
    return wte_addition, head_addition, token_addition

def _prep_vocabulary_removal(tokens_to_remove): 
    token_removal = defaultdict(int)  # key: tuple of group token ids : counts of group tokens 
    for row_idx in range(len(tokens_to_remove)):
        for token_id in tokens_to_remove[row_idx]:
            token_removal[token_id.item()] += 1
    return token_removal


def _prep_vocabulary_change(self, texts = None, input_ids = None, target_ids = None): 
    """Prepares vocabulary change for text batch"""
    
    res = self.inference(text = texts, input_ids = input_ids, target_ids = target_ids)
    input_ids, token_ids, token_perplexity, char_token_mask = res['input_ids'], res['token_ids'], res['token_perplexity'], res['char_token_mask']

    tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups = self._detect_remove_tokens(token_ids, token_perplexity, char_token_mask)      
    tokens_to_group, group_token_masks, token_groups, group_token_positions = self._detect_group_tokens(token_ids, token_perplexity, char_token_mask)

    wte_addition, head_addition, token_addition = _prep_vocabulary_addition(self, input_ids, tokens_to_group, group_token_positions)

    token_removal = _prep_vocabulary_removal(tokens_to_remove)
    
    self.wte_addition = run_avg_dict_merge(self.wte_addition, wte_addition)
    self.head_addition = run_avg_dict_merge(self.head_addition, head_addition)
    self.token_addition = add_dict_merge(self.token_addition, token_addition)
    self.token_removal = add_dict_merge(self.token_removal, token_removal)
    
    

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

def add_token_lm_head(lm_head, init_indices: Optional[torch.Tensor] = None,
                      new_projection_vector: Optional[torch.Tensor] = None):
    # Get current weights
    weights = lm_head.weight.data
    assert new_projection_vector is not None or init_indices is not None, "Either new_projection_vector or init_indices must be provided"

    # Add new row
    if new_projection_vector is not None: 
        if len(new_projection_vector.shape) == 1: 
            new_projection_vectors = new_projection_vector.unsqueeze(0)
        new_weights = torch.cat([weights, new_projection_vectors], dim=0)
    else: 
        new_weights = torch.cat([weights, weights[init_indices]], dim=0)
    
    # Create new linear layer with updated size
    new_lm_head = torch.nn.Linear(weights.size(1), new_weights.size(0), bias=lm_head.bias is not None)
    new_lm_head.weight.data = new_weights
    if lm_head.bias is not None:
        new_bias = torch.cat([lm_head.bias.data, torch.zeros(1, device=lm_head.bias.device)], dim=0)
        new_lm_head.bias.data = new_bias
    
    return new_lm_head