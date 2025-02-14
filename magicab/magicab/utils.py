import numpy as np 
import torch 
import math 
from torch.nn import functional as F



def shift_token_loss(token_loss, return_tensor=False): 
    """ 
    return_tensor: whether to return a tensor or a numpy array
    """
    # F.pad preserves the input tensor's device automatically
    token_perplexity = F.pad(token_loss, (1, 0), mode='constant', value=0.)

    if return_tensor: 
        return token_perplexity
    else: 
        return token_perplexity.cpu().numpy()
    
    
def map_token_to_char_group(text, token_ids, decode_fn, token_groups, char_token_mask = None): 
    char_groups = [] 
    for token_group in token_groups: 
        start_token_position, end_token_position, group_name, group_color = token_group
        
        if char_token_mask is not None: 
            prefix_token_ids = token_ids[:start_token_position][char_token_mask[:start_token_position]]
        else: 
            prefix_token_ids = token_ids[:start_token_position]
            
        prefix_text = decode_fn(prefix_token_ids.tolist())
        token_text = decode_fn(token_ids[start_token_position:end_token_position].tolist())
        token_text = decode_fn(token_ids[start_token_position:end_token_position].tolist())
        start_text_position = len(prefix_text)
        end_text_position = start_text_position + len(token_text)
        char_groups.append((start_text_position, end_text_position, group_name, group_color))
        
    return char_groups


def prep_char_perplexity(text, # pure text 
                         token_ids, # can contain special tokens
                         token_perplexity, 
                         decode_fn, 
                         token_mask, 
                         token_groups, 
                         char_token_mask,
                         mask_color='red'):
    """
    Map token loss to character-level perplexity
    """
        
    char_perplexity = np.zeros(len(text))
    char_colors = ['white'] * len(text)
    # Decode each token and track character positions
    curr_char_idx = 0

    for token_idx, token_id in enumerate(token_ids):
        
        if not char_token_mask[token_idx]: # skip special tokens (they are not present in text) 
            continue 
        
        token_text = decode_fn([token_id.item()])
        
        token_len = len(token_text)
        
        # Assign the token's perplexity to all characters in this token
        char_perplexity[curr_char_idx:curr_char_idx + token_len] = token_perplexity[token_idx]
        
        if token_mask is not None:
            token_color = mask_color if token_mask[token_idx] else 'white'
            char_colors[curr_char_idx:curr_char_idx + token_len] = [token_color] * token_len

        curr_char_idx += token_len
        
    char_groups = map_token_to_char_group(text, token_ids, decode_fn, token_groups, char_token_mask)
        
    return char_perplexity, char_colors, char_groups


def prep_char_perplexity_batch(texts, token_ids, token_perplexity, spike_token_mask, spike_token_groups, char_token_mask, decode, mask_color='red'): 
    char_perplexity, char_colors, char_groups = [], [], []
    for texts_, token_ids_, token_perplexity_, spike_token_mask_, spike_token_groups_, char_token_mask_ in zip(texts, token_ids, token_perplexity, spike_token_mask, spike_token_groups, char_token_mask): 
        char_perplexity_, char_colors_, char_groups_ = prep_char_perplexity(texts_, token_ids_, token_perplexity_, decode, spike_token_mask_, spike_token_groups_, char_token_mask_, mask_color=mask_color)
        char_perplexity.append(char_perplexity_)
        char_colors.append(char_colors_)
        char_groups.append(char_groups_)
    
    return np.stack(char_perplexity, axis=0), char_colors, char_groups 


def get_char_perplexity(token_ids, token_perplexity, decode_fn):
    decoded_text = ''
    char_perplexity = []
    
    for token_idx, token_id in enumerate(token_ids):
        token_text = decode_fn([token_id.item()])
        decoded_text += token_text
        char_perplexity.extend([token_perplexity[token_idx]] * len(token_text))
            
    return torch.tensor(char_perplexity)


def get_char_perplexity_batch(token_ids, token_perplexity, decode_fn): 
    char_perplexity = []
    for token_ids_, token_perplexity_ in zip(token_ids, token_perplexity): 
        char_perplexity.append(get_char_perplexity(token_ids_, token_perplexity_, decode_fn))
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


def calculate_bits_per_char(token_loss, target_ids, decode_fn, special_token_mask=None):
    
    if special_token_mask is not None: 
        token_loss = token_loss[special_token_mask]
        target_ids = target_ids[special_token_mask]
    
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


def _pad_batch_inference(model, tokenizer, input_ids, target_ids,
                         return_char_perplexity: bool = False,
                         return_representation: bool = False,
                         device: str = "cuda", # computation device
                         return_device: str = "cpu"): 
    """ 
    Helper function to pad batch inference
    For processing, we do it on CPU
    """
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    res = {"input_ids": input_ids.to(return_device)}
    with torch.no_grad():
        if return_representation: 
            logits, token_loss, reps = model(input_ids, targets=target_ids, reduction='none', return_representation=True) # loss is provided as an 'average' loss per token --- I want singular loss per token 
            res["reps"] = reps.to(return_device)
        else: 
            logits, token_loss = model(input_ids, targets=target_ids, reduction='none') # loss is provided as an 'average' loss per token --- I want singular loss per token 

    input_ids = input_ids.to(return_device)
    target_ids = target_ids.to(return_device)
    token_loss = token_loss.to(return_device)

    token_ids = torch.cat([input_ids, target_ids[:, -1:]], dim=1)
    res["token_ids"] = token_ids
    special_token_mask = token_ids.ne(tokenizer.special2idx["<pad>"])

    token_loss = token_loss * special_token_mask[:, 1:] # zero-out loss for pad tokens
    token_perplexity = shift_token_loss(token_loss, return_tensor=True)
    res["token_perplexity"] = token_perplexity
    
    if return_char_perplexity: 
        
        # This might not be correct with more composite tokens
        char_token_ids = [token_ids_row[special_token_mask_row] for token_ids_row, special_token_mask_row in zip(token_ids, special_token_mask)]

        # # These functional need to take care of 'special tokens' -- it need to remove those, especially 'decode' tokens
        decode = lambda x: tokenizer.decode(x)
        bpc_loss = calculate_bits_per_char(token_loss, target_ids, decode, special_token_mask[:, 1:])

        char_token_loss = [token_loss_row[special_token_mask_row[1:]] for token_loss_row, special_token_mask_row in zip(token_loss, special_token_mask)]
        char_token_perplexity = [shift_token_loss(char_token_loss_row, return_tensor=True) for char_token_loss_row in char_token_loss]
        
        char_perplexity = get_char_perplexity_batch(char_token_ids, char_token_perplexity, decode)
        
        res["char_perplexity"] = char_perplexity
        res["bpc_loss"] = bpc_loss

    return res 


def batch_inference(model, 
                    tokenizer, 
                    input_ids, 
                    target_ids, 
                    return_char_perplexity: bool = False,
                    return_representation: bool = False,
                    device: str = "cuda",
                    return_device: str = "cpu"): 
    """ 
    Miscellaneous results from model inference
    """
    res = {}
    res["input_ids"] = input_ids.to(return_device)
    
    # Move tensors to device
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    # Use torch.no_grad() to disable gradient tracking
    with torch.no_grad():
        if return_representation:
            logits, token_loss, reps = model(input_ids, targets=target_ids, reduction='none', return_representation=True)
            res["reps"] = reps.to(return_device)
        else: 
            logits, token_loss = model(input_ids, targets=target_ids, reduction='none')
    
    input_ids = input_ids.to(return_device)
    target_ids = target_ids.to(return_device)
    token_loss = token_loss.to(return_device)
    
    token_ids = torch.cat([input_ids, target_ids[:, -1:]], dim=1)
    res["token_ids"] = token_ids
    
    token_perplexity = shift_token_loss(token_loss, return_tensor=True) # numpy if return_device is cpu
    res["token_perplexity"] = token_perplexity
    
    if return_char_perplexity: 
        decode = lambda x: tokenizer.decode(x)
        bpc_loss = calculate_bits_per_char(token_loss, target_ids, decode)
        res["bpc_loss"] = bpc_loss
        
        char_perplexity = get_char_perplexity_batch(token_ids.to("cpu"), token_perplexity.to("cpu"), decode) # cpu
        res["char_perplexity"] = char_perplexity
        
    return res 
    
    
def inference(model, 
              tokenizer,
              output_tokenizer = None,
              text = None, 
              input_ids = None, 
              target_ids = None,
              pad: bool = False,
              return_representation: bool = False,
              return_char_perplexity: bool = False,
              device: str = "cuda",
              return_device: str = "cpu"): # Issue: why should we assume 'texts' to have same length?
    
    valid_text = text is not None and (isinstance(text, str) or isinstance(text, list))
    valid_batch = (input_ids is not None and target_ids is not None) and (input_ids.shape == target_ids.shape)
    assert valid_text or valid_batch, "Either text or input_ids and target_ids must be provided"
    
    if output_tokenizer is None:  # same tokenizer for input and output
        output_tokenizer = tokenizer
        
    if valid_text: 
        texts = text if isinstance(text, list) else [text]
        token_ids_list = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        
        max_token_len = max([token_ids.shape[0] for token_ids in token_ids_list])
        if not all([token_ids.shape[0] == max_token_len for token_ids in token_ids_list]): 
            pad = True
            
        if pad: 
            pad_token_id = tokenizer.special2idx["<pad>"]
            token_ids_list = [F.pad(token_ids, pad=(max_token_len - token_ids.shape[0], 0), mode='constant', value=pad_token_id) for token_ids in token_ids_list]
                    
        token_ids = torch.stack(token_ids_list, dim=0).to(device)
        
        input_ids = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]
        
    else: 
        token_ids = torch.cat([input_ids, target_ids[:, -1:]], dim=1)
        texts = [output_tokenizer.decode(token_ids[i].tolist()) for i in range(token_ids.size(0))]
    
    if pad: 
        res = _pad_batch_inference(model, output_tokenizer, input_ids, target_ids, return_char_perplexity, return_representation, device=device, return_device=return_device)
    else: 
        res = batch_inference(model, output_tokenizer, input_ids, target_ids, return_char_perplexity, return_representation, device=device, return_device=return_device)
    
    res['texts'] = texts
    res["char_token_mask"] = ~torch.isin(res["token_ids"], torch.tensor(tokenizer.special_ids).to(return_device)) # character & merge tokens
        
    return res

    
def short_one(positions): 
    return [p-1 for p in positions]