import numpy as np 

def shift_token_loss(token_loss): 
    token_perplexity = token_loss.detach().numpy()
    token_perplexity = np.pad(token_perplexity, (1, 0), mode='constant', constant_values=0.)
    return token_perplexity 

def map_token_to_char_perplexity(text, token_ids, token_perplexity, decode_fn):
    """
    Map token loss to character-level perplexity
    """
    char_perplexity = np.zeros(len(text))
    
    # Decode each token and track character positions
    curr_char_idx = 0
    for token_idx, token_id in enumerate(token_ids[0]):
        token_text = decode_fn([token_id.item()])
        token_len = len(token_text)
        
        # Assign the token's perplexity to all characters in this token
        char_perplexity[curr_char_idx:curr_char_idx + token_len] = token_perplexity[token_idx]
        curr_char_idx += token_len
    
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