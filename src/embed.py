import torch 
from typing import Optional

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

