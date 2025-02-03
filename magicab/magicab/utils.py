import numpy as np 
import torch 
import math 
from torch.nn import functional as F



def shift_token_loss(token_loss, return_tensor=True): 
    if len(token_loss.shape) == 1: 
        token_perplexity = token_loss.detach().numpy()
        token_perplexity = np.pad(token_perplexity, (1, 0), mode='constant', constant_values=0.)
    else: 
        token_perplexity = token_loss.detach().numpy()
        token_perplexity = np.pad(token_perplexity, [(0, 0), (1, 0)], mode='constant', constant_values=0.)

    if return_tensor: 
        return torch.tensor(token_perplexity)
    else: 
        return token_perplexity 

def map_token_to_char_group(text, token_ids, decode_fn, token_groups): 
    char_groups = [] 
    for token_group in token_groups: 
        start_token_position, end_token_position, group_name, group_color = token_group
        prefix_test = decode_fn(token_ids[:start_token_position].tolist())
        token_text = decode_fn(token_ids[start_token_position:end_token_position].tolist())
        start_text_position = len(prefix_test)
        end_text_position = start_text_position + len(token_text)
        char_groups.append((start_text_position, end_text_position, group_name, group_color))
    return char_groups

def map_token_to_char_perplexity(text, token_ids, token_perplexity, decode_fn, token_mask=None, token_groups=None, mask_color='red'):
    """
    Map token loss to character-level perplexity
    """
    if token_ids.ndim > 1: 
        assert token_ids.shape[0] == 1, "token_ids should be 1D tensor"
        token_ids = token_ids[0]
    if token_perplexity.ndim > 1: 
        assert token_perplexity.shape[0] == 1, "token_perplexity should be 1D tensor"
        token_perplexity = token_perplexity[0]
    if token_mask is not None and token_mask.ndim > 1: 
        assert token_mask.shape[0] == 1, "token_mask should be 1D tensor"
        token_mask = token_mask[0]
    assert isinstance(text, str), "text should be a string"
        
    char_perplexity = np.zeros(len(text))
    char_colors = ['white'] * len(text)
    # Decode each token and track character positions
    curr_char_idx = 0

    for token_idx, token_id in enumerate(token_ids):
        token_text = decode_fn([token_id.item()])
        token_len = len(token_text)
        
        # Assign the token's perplexity to all characters in this token
        char_perplexity[curr_char_idx:curr_char_idx + token_len] = token_perplexity[token_idx]
        
        if token_mask is not None:
            token_color = mask_color if token_mask[token_idx] else 'white'
            char_colors[curr_char_idx:curr_char_idx + token_len] = [token_color] * token_len

        curr_char_idx += token_len
        
    if token_mask is not None and token_groups is not None:
        char_groups = map_token_to_char_group(text, token_ids, decode_fn, token_groups)
        return char_perplexity, char_colors, char_groups
    elif token_mask is not None:
        return char_perplexity, char_colors
    else:
        return char_perplexity
    
    
    
    
def map_batch_token_to_char_perplexity(texts, token_ids, token_perplexity, decode_fn, token_mask=None, token_groups=None, mask_color='red'):
    
    results = []
    batch_size = len(texts)
    for i in range(batch_size):
        sample_token_ids = token_ids[i:i+1]

        sample_token_perplexity = token_perplexity[i] if hasattr(token_perplexity, 'ndim') and token_perplexity.ndim > 1 else token_perplexity
        
        sample_token_mask = token_mask[i] if token_mask is not None else None
        sample_token_groups = token_groups[i] if token_groups is not None else None
        
        result = map_token_to_char_perplexity(
            texts[i],
            sample_token_ids,
            sample_token_perplexity,
            decode_fn,
            token_mask=sample_token_mask,
            token_groups=sample_token_groups,
            mask_color=mask_color
        )
        results.append(result)
        
    # Handle different return types based on input arguments
    if isinstance(results[0], tuple):
        # Unzip the tuples into separate lists
        return tuple(x if isinstance(x[0], list) else np.stack(x, axis=0) for x in zip(*results))
    else:
        return np.stack(results, axis=0)

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
    bits_per_token = token_loss * math.log2(math.e)
    
    token_ids_flat = target_ids.reshape(-1).tolist()
    token_lens = [len(decode_fn([token])) for token in token_ids_flat]
    
    token_lens = torch.tensor(token_lens, dtype=bits_per_token.dtype, device=bits_per_token.device)
    token_lens = token_lens.reshape(target_ids.size())
    
    total_chars = token_lens.sum()
    if total_chars == 0:
        return torch.tensor(0.0, dtype=bits_per_token.dtype, device=bits_per_token.device)
        
    bits_per_char = (bits_per_token.detach() * token_lens).sum() / total_chars
    return bits_per_char


def batch_inference(model, tokenizer, input_ids, target_ids): 
    """ 
    Miscellaneous results from model inference
    """
    decode = lambda x: tokenizer.decode(x)

    token_ids = torch.cat([input_ids, target_ids[:, -1:]], dim=1)
    texts = [decode(token_ids[i].tolist()) for i in range(token_ids.size(0))]

    logits, token_loss = model(input_ids, targets=target_ids, reduction='none') # loss is provided as an 'average' loss per token --- I want singular loss per token 
    
    bpc_loss = calculate_bits_per_char(token_loss, target_ids, decode)
    token_perplexity = shift_token_loss(token_loss)
    char_perplexity = map_batch_token_to_char_perplexity(texts, token_ids, token_perplexity, decode)

    return {"input_ids": input_ids, "token_ids": token_ids, "token_perplexity": token_perplexity, 
            "bpc_loss": bpc_loss, "char_perplexity": char_perplexity}
    
    
def inference(model, tokenizer,
              text = None, 
              input_ids = None, 
              target_ids = None,
              pad = False):
    
    valid_text = text is not None and (isinstance(text, str) or isinstance(text, list))
    valid_batch = (input_ids is not None and target_ids is not None) and (input_ids.shape == target_ids.shape)
    assert valid_text or valid_batch, "Either text or input_ids and target_ids must be provided"
    
    if valid_text: 
        texts = text if isinstance(text, list) else [text]
        token_ids_list = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        
        max_token_len = max([token_ids.shape[0] for token_ids in token_ids_list])
        assert pad or all([token_ids.shape[0] == max_token_len for token_ids in token_ids_list]), "All token ids must be of the same length, or we need to pad them"
        if pad: 
            pad_token_id = tokenizer.special2idx["<pad>"]
            token_ids_list = [F.pad(token_ids, pad=(max_token_len - token_ids.shape[0], 0), mode='constant', value=pad_token_id) for token_ids in token_ids_list]
                    
        token_ids = torch.stack(token_ids_list, dim=0)
        
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
    
    res = batch_inference(model, tokenizer, input_ids, target_ids)
    
    if valid_text: 
        res['texts'] = texts
        
    return res

    
def short_one(positions): 
    return [p-1 for p in positions]