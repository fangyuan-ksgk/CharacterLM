import numpy as np 
import torch 
import math 

def shift_token_loss(token_loss, return_tensor=True): 
    token_perplexity = token_loss.detach().numpy()
    token_perplexity = np.pad(token_perplexity, (1, 0), mode='constant', constant_values=0.)
    if return_tensor: 
        return torch.tensor(token_perplexity)
    else: 
        return token_perplexity 

def map_token_to_char_perplexity(text, token_ids, token_perplexity, decode_fn, token_mask=None, mask_color='red'):
    """
    Map token loss to character-level perplexity
    """
    char_perplexity = np.zeros(len(text))
    char_colors = ['white'] * len(text)
    # Decode each token and track character positions
    curr_char_idx = 0
    for token_idx, token_id in enumerate(token_ids[0]):
        token_text = decode_fn([token_id.item()])
        token_len = len(token_text)
        
        # Assign the token's perplexity to all characters in this token
        char_perplexity[curr_char_idx:curr_char_idx + token_len] = token_perplexity[token_idx]
        
        if token_mask is not None:
            token_color = mask_color if token_mask[token_idx] else 'white'
            char_colors[curr_char_idx:curr_char_idx + token_len] = [token_color] * token_len
    
        curr_char_idx += token_len
        
    if token_mask is not None:
        return char_perplexity, char_colors
    else:
        return char_perplexity

def get_naive_char_color(char_perplexity):
    p80 = np.quantile(char_perplexity, 0.80)
    p60 = np.quantile(char_perplexity, 0.60)
    # char color array: red for per>p80, pink for per>p60, None for per<p60
    char_colors = ['white'] * len(char_perplexity)
    for i in range(len(char_perplexity)):
        if char_perplexity[i] > p80:
            char_colors[i] = 'red'
        elif char_perplexity[i] > p60:
            char_colors[i] = 'pink'
    return char_colors


def calculate_bits_per_char(token_loss, target_ids, decode_fn):
    # Convert from nats to bits
    bits_per_token = token_loss * math.log2(math.e)
    
    # Calculate character length for each token
    token_lens = torch.tensor([len(decode_fn([token])) for token in target_ids[0].tolist()])
    
    # Ensure no division by zero
    total_chars = token_lens.sum()
    if total_chars == 0:
        return torch.tensor(0.0)
        
    # Calculate weighted average of bits per character
    bits_per_char = (bits_per_token.detach() * token_lens).sum() / total_chars
    return bits_per_char