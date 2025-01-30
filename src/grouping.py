import torch 

# Spike token: Sudden Jump in perplexity above threshold
def detect_spike_token(token_loss, quantile_threshold=0.80): 
    """ 
    Spike token cross perplexity threshold, and shows a sudden increase in perplexity
    """
    loss_threshold = torch.quantile(token_loss, quantile_threshold)
    spike_tokens = []
    for i in range(len(token_loss)):
        last_token_loss = token_loss[max(i-1, 0)]
        if token_loss[i] > loss_threshold and token_loss[i] > last_token_loss: 
            spike_tokens.append(i)
    return spike_tokens

def get_spike_token_mask(token_loss, quantile_threshold=0.80): 
    spike_token_indices = detect_spike_token(token_loss, quantile_threshold=quantile_threshold)
    spike_token_mask = torch.zeros_like(token_loss, dtype=torch.bool)
    spike_token_mask[spike_token_indices] = True
    return spike_token_mask

# Natural token group: consecutive decrease in perplexity below threshold
def detect_group_token(token_loss, quantile_threshold=0.7): 
    loss_threshold = torch.quantile(token_loss, quantile_threshold)
    natural_group = []
    curr_group = []

    for i in range(len(token_loss)):
        if not curr_group:  # Start new group
            curr_group.append(i)
        elif token_loss[i] <= token_loss[i-1] and token_loss[i] < loss_threshold:  # Continue group if decreasing
            curr_group.append(i)
        else:  # End group if increasing
            if len(curr_group) > 1:  # Only keep groups of 2 or more
                natural_group.append(curr_group)
            curr_group = [i]  # Start new group with current index
                
    if len(curr_group) > 1:
        natural_group.append(curr_group)    
    return natural_group

def get_group_token_mask(token_loss, quantile_threshold=0.7): 
    group_token = detect_group_token(token_loss, quantile_threshold=quantile_threshold)
    group_token_mask = torch.zeros_like(token_loss, dtype=torch.bool)
    for group in group_token: 
        group_token_mask[group] = True
    return group_token_mask
