import torch 

# Spike token: Sudden Jump in perplexity above threshold
def _detect_spike_token(token_loss, quantile_threshold=0.80, perplexity_threshold=None): 
    """ 
    Spike token cross perplexity threshold, and shows a sudden increase in perplexity
    :: Need to consider 'multiple-character tokens' 
    """
    quantile_threshold = torch.quantile(token_loss, quantile_threshold).item()
    loss_threshold = max(quantile_threshold, perplexity_threshold if perplexity_threshold is not None else 0.)
    spike_token_positions = []
    for i in range(len(token_loss)):
        last_token_loss = token_loss[max(i-1, 0)]
        if token_loss[i] > loss_threshold and token_loss[i] > last_token_loss: 
            spike_token_positions.append(i)
    return spike_token_positions

def detect_spike_token(token_ids, token_loss, quantile_threshold=0.80, perplexity_threshold=None): 
    spike_token_positions = _detect_spike_token(token_loss, quantile_threshold=quantile_threshold, perplexity_threshold=perplexity_threshold)
    tokens_to_spike = []
    for position in spike_token_positions: 
        tokens_to_spike.append(token_ids[0, position].item())
    return tokens_to_spike

def get_spike_token_mask(token_loss, quantile_threshold=0.80, perplexity_threshold=None, color='red'): 
    spike_token_positions = _detect_spike_token(token_loss, quantile_threshold=quantile_threshold, perplexity_threshold=perplexity_threshold)
    spike_token_mask = torch.zeros_like(token_loss, dtype=torch.bool)
    spike_token_mask[spike_token_positions] = True
    
    spike_token_groups = []
    for position in spike_token_positions: 
        spike_token_groups.append((position, position + 1, str(len(spike_token_groups) + 1), color))
    
    return spike_token_mask, spike_token_groups


def detect_spike_token_batch(token_perplexity, quantile_threshold=0.80, perplexity_threshold=None, color='red', return_groups=True, char_token_mask=None):
    """ 
    Detect spike token in batch data 
    """
    quantile_threshold = torch.quantile(token_perplexity, quantile_threshold, dim=1)
    loss_threshold = torch.maximum(quantile_threshold, torch.tensor(perplexity_threshold if perplexity_threshold is not None else 0.))
    
    prev_perplexity = torch.cat([token_perplexity[:, :1], token_perplexity[:, :-1]], dim=-1)
    spike_token_mask = (token_perplexity > prev_perplexity) & (token_perplexity > loss_threshold.unsqueeze(-1))
    
    if char_token_mask is not None: 
        spike_token_mask = spike_token_mask & char_token_mask
        
    spike_token_positions = [torch.nonzero(row)[:, 0] for row in spike_token_mask]
    
    if return_groups: 
        spike_token_groups = []
        for row_positions in spike_token_positions: 
            spike_token_gs = []
            for position in row_positions: 
                spike_token_gs.append((position.item(), position.item() + 1, str(len(spike_token_gs) + 1), color))
            spike_token_groups.append(spike_token_gs)
        return spike_token_positions, spike_token_mask, spike_token_groups
    
    return spike_token_positions, spike_token_mask


import time 

def detect_remove_token_batch(token_ids, token_perplexity, tokenizer, quantile_threshold=0.80, perplexity_threshold=None, color='red', return_groups=True, char_token_mask=None,
                              cal_mask_device: str = "cpu"):
    """ 
    Detect remove token in batch data 
    """
    t0 = time.time()
    quantile_threshold = torch.quantile(token_perplexity, quantile_threshold, dim=1)
    loss_threshold = torch.maximum(quantile_threshold, torch.tensor(perplexity_threshold if perplexity_threshold is not None else 0.))
    spike_token_mask = token_perplexity > loss_threshold.unsqueeze(-1)
    base_char_mask = torch.isin(token_ids, torch.tensor(list(tokenizer.char_vocab.keys())).to(cal_mask_device))
    leaf_token_mask = torch.isin(token_ids, torch.tensor(list(tokenizer.leaf_token_ids)).to(cal_mask_device))
    remove_token_mask = spike_token_mask & ~base_char_mask & leaf_token_mask # only remove leaf-tokens to avoid collapsing tokenization
    
    if char_token_mask is not None: 
        remove_token_mask = remove_token_mask & char_token_mask & leaf_token_mask
    
    remove_token_positions = [torch.nonzero(row)[:, 0].tolist() for row in remove_token_mask]
    print(f"   :: Remove token mask calculation: {time.time() - t0:.4f} seconds")
    
    t1 = time.time()
    tokens_to_remove = [] 
    for remove_token_mask_row, token_ids_row in zip(remove_token_mask, token_ids): 
        tokens_to_remove.append(token_ids_row[remove_token_mask_row].tolist())
    print(f"   :: Remove token list appending loop: {time.time() - t1:.4f} seconds")
    
    t2 = time.time()
    if return_groups: 
        remove_token_groups = []
        for remove_token_positions_row in remove_token_positions: 
            remove_token_groups_row = []
            for position in remove_token_positions_row: 
                try: 
                    curr_group = position.item(), position.item() + 1, str(len(remove_token_groups_row) + 1), color
                except: 
                    curr_group = position, position + 1, str(len(remove_token_groups_row) + 1), color
                remove_token_groups_row.append(curr_group)
            remove_token_groups.append(remove_token_groups_row)
        print(f"   :: Remove token group appending loop: {time.time() - t2:.4f} seconds")
        return tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups
    
    return tokens_to_remove, remove_token_positions, remove_token_mask, remove_token_groups


def _detect_remove_token_positions(token_ids, token_loss, tok, quantile_threshold=0.80, perplexity_threshold=None): 
    spike_token_indices = _detect_spike_token(token_loss, quantile_threshold=quantile_threshold, perplexity_threshold=perplexity_threshold)
    positions_to_remove = []
    for index in spike_token_indices:
        token_id = token_ids[0, index].item()
        if token_id in tok.merges.values(): 
            positions_to_remove.append(index)
    return positions_to_remove

def get_remove_token_mask(token_ids, token_loss, tok, quantile_threshold=0.80, perplexity_threshold=None, color='red'): 
    positions_to_remove = _detect_remove_token_positions(token_ids, token_loss, tok, quantile_threshold=quantile_threshold, perplexity_threshold=perplexity_threshold)
    remove_token_mask = torch.zeros_like(token_loss, dtype=torch.bool)
    remove_token_mask[positions_to_remove] = True
    
    remove_token_groups = []
    for index in positions_to_remove: 
        remove_token_groups.append((index.item(), index.item() + 1, str(len(remove_token_groups) + 1), color))
    
    return remove_token_mask, remove_token_groups


from rust_tokenizer import detect_group_token as detect_group_token_rust

def detect_group_token_batch(token_ids, token_perplexity, cache_token_addition, quantile_threshold=0.7, 
                             perplexity_threshold=None, 
                             color='green', 
                             char_token_mask=None,
                             cal_mask_device: str = "cpu"): 
    """ 
    Detect group token in batch data 
    """
    
    if token_perplexity.ndim == 1: 
        token_perplexity = token_perplexity.unsqueeze(0)

    quantile_threshold = torch.quantile(token_perplexity, quantile_threshold, dim=1)
    perplexity_threshold = torch.minimum(quantile_threshold, torch.tensor(perplexity_threshold if perplexity_threshold is not None else 999.))
    
    token_perplexity = [t.tolist() for t in token_perplexity]
    token_ids = [t.tolist() for t in token_ids]
    char_token_mask = [t.tolist() for t in char_token_mask]
    perplexity_threshold = perplexity_threshold.tolist()
    
    tokens_to_group, group_token_positions, group_token_masks, groups,  = detect_group_token_rust(token_perplexity, token_ids,
                                                                                                 perplexity_threshold=perplexity_threshold,
                                                                                                 char_token_mask=char_token_mask)
    
    return tokens_to_group, group_token_masks, groups, group_token_positions