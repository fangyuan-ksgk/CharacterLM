import torch 

# Spike token: Sudden Jump in perplexity above threshold
def _detect_spike_token(token_loss, quantile_threshold=0.80): 
    """ 
    Spike token cross perplexity threshold, and shows a sudden increase in perplexity
    :: Need to consider 'multiple-character tokens' 
    """
    loss_threshold = torch.quantile(token_loss, quantile_threshold)
    spike_token_positions = []
    for i in range(len(token_loss)):
        last_token_loss = token_loss[max(i-1, 0)]
        if token_loss[i] > loss_threshold and token_loss[i] > last_token_loss: 
            spike_token_positions.append(i)
    return spike_token_positions

def detect_spike_token(token_ids, token_loss, quantile_threshold=0.80): 
    spike_token_positions = _detect_spike_token(token_loss, quantile_threshold=quantile_threshold)
    tokens_to_spike = []
    for position in spike_token_positions: 
        tokens_to_spike.append(token_ids[0, position].item())
    return tokens_to_spike

def get_spike_token_mask(token_loss, quantile_threshold=0.80, color='red'): 
    spike_token_positions = _detect_spike_token(token_loss, quantile_threshold=quantile_threshold)
    spike_token_mask = torch.zeros_like(token_loss, dtype=torch.bool)
    spike_token_mask[spike_token_positions] = True
    
    spike_token_groups = []
    for position in spike_token_positions: 
        spike_token_groups.append((position, position + 1, str(len(spike_token_groups) + 1), color))
    
    return spike_token_mask, spike_token_groups


def detect_spike_token_batch(token_perplexity, quantile_threshold=0.80, color='red', return_groups=True, char_token_mask=None):
    """ 
    Detect spike token in batch data 
    """
    loss_threshold = torch.quantile(token_perplexity, quantile_threshold, axis=-1)
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


def detect_remove_token_batch(token_ids, token_perplexity, tokenizer, quantile_threshold=0.80, color='red', return_groups=True, char_token_mask=None):
    """ 
    Detect remove token in batch data 
    """
    loss_threshold = torch.quantile(token_perplexity, quantile_threshold, axis=-1)
    spike_token_mask = token_perplexity > loss_threshold.unsqueeze(-1)
    base_char_mask = torch.isin(token_ids, torch.tensor(list(tokenizer.char_vocab.keys())))
    remove_token_mask = spike_token_mask & base_char_mask
    
    if char_token_mask is not None: 
        remove_token_mask = remove_token_mask & char_token_mask
    
    remove_token_positions = [torch.nonzero(row)[:, 0] for row in remove_token_mask]
    
    if return_groups: 
        remove_token_groups = []
        for remove_token_positions_row in remove_token_positions: 
            remove_token_groups_row = []
            for position in remove_token_positions_row: 
                curr_group = position.item(), position.item() + 1, str(len(remove_token_groups_row) + 1), color
                remove_token_groups_row.append(curr_group)
            remove_token_groups.append(remove_token_groups_row)
    
        return remove_token_positions, remove_token_mask, remove_token_groups
    
    return remove_token_positions, remove_token_mask


def _detect_remove_token_positions(token_ids, token_loss, tok, quantile_threshold=0.80): 
    spike_token_indices = _detect_spike_token(token_loss, quantile_threshold=quantile_threshold)
    positions_to_remove = []
    for index in spike_token_indices:
        token_id = token_ids[0, index].item()
        if token_id in tok.merges.values(): 
            positions_to_remove.append(index)
    return positions_to_remove

def get_remove_token_mask(token_ids, token_loss, tok, quantile_threshold=0.80, color='red'): 
    positions_to_remove = _detect_remove_token_positions(token_ids, token_loss, tok, quantile_threshold=quantile_threshold)
    remove_token_mask = torch.zeros_like(token_loss, dtype=torch.bool)
    remove_token_mask[positions_to_remove] = True
    
    remove_token_groups = []
    for index in positions_to_remove: 
        remove_token_groups.append((index.item(), index.item() + 1, str(len(remove_token_groups) + 1), color))
    
    return remove_token_mask, remove_token_groups

# Natural token group: consecutive decrease in perplexity below threshold
def _detect_group_token(token_loss, quantile_threshold=0.7, char_token_mask=None): 
    loss_threshold = torch.quantile(token_loss, quantile_threshold)
    natural_group = []
    curr_group = []

    i = 0
    while i < len(token_loss): 
        
        valid_item = True
        if char_token_mask is not None: 
            if not char_token_mask[i]: 
                valid_item = False
        
        if not valid_item: 
            if len(curr_group) > 1: 
                natural_group.append(curr_group)
            curr_group = []
            i += 1
            continue
        
        if not curr_group: 
            curr_group.append(i) # first token in group can be hard to guess, point is the continuation of the group should be simple
        elif token_loss[i] <= token_loss[i-1] and token_loss[i] < loss_threshold:  # Continue group if decreasing
            curr_group.append(i)
        else: 
            if len(curr_group) > 1: 
                natural_group.append(curr_group)
            curr_group = []
        i += 1
            
    if len(curr_group) > 1:
        natural_group.append(curr_group)    
        
    return natural_group



def detect_group_token(token_ids, token_loss, quantile_threshold=0.7, return_indices=False): 
    group_token_positions = _detect_group_token(token_loss, quantile_threshold=quantile_threshold)
    tokens_to_group = []
    for group in group_token_positions: 
        tokens_to_group.append(token_ids[0, group].tolist())
    if return_indices: 
        return tokens_to_group, group_token_positions
    else: 
        return tokens_to_group
    
    

def get_group_token_mask(token_loss, quantile_threshold=0.7, color='green', char_token_mask=None): 
    # group token position includes special tokens 
    group_token_positions = _detect_group_token(token_loss, quantile_threshold=quantile_threshold, char_token_mask=char_token_mask)
    group_token_mask = torch.zeros_like(token_loss, dtype=torch.bool)
    groups = []
    for group in group_token_positions:
        g_start, g_end = group[0], group[-1]+1
        group_token_mask[g_start: g_end] = True
        curr_group = g_start, g_end, str(len(groups) + 1), color
        groups.append(curr_group)
    
    return group_token_positions, group_token_mask, groups


def detect_group_token_batch(token_ids, token_perplexity, quantile_threshold=0.7, color='green', char_token_mask=None): 
    """ 
    Detect group token in batch data 
    """
    if token_perplexity.ndim == 1: 
        token_perplexity = token_perplexity.unsqueeze(0)
        
    tokens_to_group = []
    group_token_positions = []
    group_token_masks = []
    groups = []
    for token_ids_row, token_perp, char_token_mask_row in zip(token_ids, token_perplexity, char_token_mask): 
        group_token_positions_row, group_token_masks_row, group_row = get_group_token_mask(token_loss=token_perp, quantile_threshold=quantile_threshold, color=color, char_token_mask=char_token_mask_row)
        
        group_token_positions.append(group_token_positions_row)
        group_token_masks.append(group_token_masks_row)
        groups.append(group_row)
        
        tokens_to_group_row = []
        for g in group_token_positions_row: 
            tokens_to_group_row.append(token_ids_row[g].tolist())
        tokens_to_group.append(tokens_to_group_row)
        
    group_token_masks = torch.stack(group_token_masks, axis=0)
    
    return tokens_to_group, group_token_masks, groups, group_token_positions