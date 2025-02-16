from collections import defaultdict
from typing import Optional
import torch
import time
import random
from functools import wraps
import numpy as np 

def run_avg_dict_update(d, key, new_value): 
    if key not in d: 
        d[key] = new_value 
    else: 
        d[key] = (new_value + d[key]) / 2
    return d 


def update_caches(project_cache, embed_cache, eom_token_ids, embeddings_row, projects_row, device="mps"):
    
    for key, value in zip(eom_token_ids, projects_row): 
        project_cache[key] = value 
        
    for key, value in zip(eom_token_ids, embeddings_row): 
        embed_cache[key] = value 
        
    return project_cache, embed_cache

def update_input_caches(embed_cache, eom_token_ids, embeddings_row, device="mps"):
    for key, value in zip(eom_token_ids, embeddings_row): 
        embed_cache[key] = value 
    return embed_cache

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


def _prep_vocabulary_addition(self, input_ids, tokens_to_group, group_token_positions, reps):
    """ 
    This ignores cached merges in magicab and re-process everything, redundant
    - should add a filter gadget to skip already processed tokens
    - (modified on model.py | use representations from one forward pass)
    """
    
    # Initialize timing dictionaries
    timings = {
        'add_tokens': 0,
        'slice_embeddings': 0,
        'update_token_addition': 0,
        'update_caches': 0
    }
    
    # Vocabulary Addition
    embed_cache = {}  
    project_cache = {}  
    token_addition = defaultdict(int)
    
    for row_idx in range(len(input_ids)):
        input_ids_row = input_ids[row_idx]
        tokens_to_group_row = tokens_to_group[row_idx]
        group_positions = group_token_positions[row_idx]
        
        # Time add_tokens
        t0 = time.time()
        eom_token_ids, pair_token_groups, pair_token_positions = self.tokenizer_copy.add_tokens(
            tokens_to_group_row, group_positions, in_place=True
        )
        timings['add_tokens'] += time.time() - t0
        
        eom_positions = [p[-1] for p in pair_token_positions]
        reps_row = reps[row_idx]
        
        # Time slice_embeddings
        t0 = time.time()
        embeddings_row = self.model.transformer.wte.weight[eom_token_ids].detach()
        projects_row = self.model.lm_head.weight[eom_token_ids].detach()
        timings['slice_embeddings'] += time.time() - t0
        
        # Time update_token_addition
        t0 = time.time()
        for token_tuple in tokens_to_group_row:
            token_tuple = tuple(token_tuple)
            token_addition[token_tuple] += 1
        timings['update_token_addition'] += time.time() - t0
        
        # Time update_caches
        t0 = time.time()
        project_cache, embed_cache = update_caches(project_cache, embed_cache, eom_token_ids, embeddings_row, projects_row,
                                                   device=self.device)
      
        timings['update_caches'] += time.time() - t0
    
    # Print timing summary
    for component, elapsed in timings.items():
        print(f"   :: {component}: {elapsed:.4f} seconds")
        print(f"   :: {component}: per input row  {elapsed / len(input_ids):.4f} seconds")
    
    return token_addition, embed_cache, project_cache


def _prep_input_vocabulary_addition(self, input_ids, tokens_to_group, group_token_positions, reps):
    """ 
    This ignores cached merges in magicab and re-process everything, redundant
    - should add a filter gadget to skip already processed tokens
    - (modified on model.py | use representations from one forward pass)
    """
    
    # Initialize timing dictionaries
    timings = {
        'add_tokens': 0,
        'slice_embeddings': 0,
        'update_token_addition': 0,
        'update_caches': 0
    }
    
    # Vocabulary Addition
    embed_cache = {}  
    token_addition = defaultdict(int)
    
    for row_idx in range(len(input_ids)):
        input_ids_row = input_ids[row_idx]
        tokens_to_group_row = tokens_to_group[row_idx]
        group_positions = group_token_positions[row_idx]
        
        # Time add_tokens
        t0 = time.time()
        eom_token_ids, pair_token_groups, pair_token_positions = self.tokenizer_copy.add_tokens(
            tokens_to_group_row, group_positions, in_place=True
        )
        timings['add_tokens'] += time.time() - t0
        
        eom_positions = [p[-1] for p in pair_token_positions]
        reps_row = reps[row_idx]
        
        # Time slice_embeddings
        t0 = time.time()
        embeddings_row = self.model.transformer.wte.weight[eom_token_ids].detach()
        timings['slice_embeddings'] += time.time() - t0
        
        # Time update_token_addition
        t0 = time.time()
        for token_tuple in tokens_to_group_row:
            token_tuple = tuple(token_tuple)
            token_addition[token_tuple] += 1
        timings['update_token_addition'] += time.time() - t0
        
        # Time update_caches
        t0 = time.time()
        embed_cache = update_input_caches(embed_cache, eom_token_ids, embeddings_row,
                                                   device=self.device)
      
        timings['update_caches'] += time.time() - t0
    
    # Print timing summary
    for component, elapsed in timings.items():
        print(f"   :: {component}: {elapsed:.4f} seconds")
        print(f"   :: {component}: per input row  {elapsed / len(input_ids):.4f} seconds")
    
    return token_addition, embed_cache


def _prep_vocabulary_removal(tokens_to_remove):     
    token_removal = defaultdict(int)  # key: tuple of group token ids : counts of group tokens 
    for row_idx in range(len(tokens_to_remove)):
        for token_id in tokens_to_remove[row_idx]:
            token_removal[token_id] += 1
    return token_removal

def _cache_input_vocabulary_change(self, texts=None, input_ids=None, target_ids=None, avoid_duplicate: bool = False, cal_mask_device: str = "cpu"):
    """Prepares vocabulary change for input batch"""
    
    # Get all required data in one inference pass
    res = self.inference(text=texts, input_ids=input_ids, target_ids=target_ids, return_representation=True, return_device=cal_mask_device)
    input_ids, token_ids, token_perplexity, char_token_mask, reps = (
        res['input_ids'], res['token_ids'], 
        res['token_perplexity'], res['char_token_mask'], res['reps']
    )
    
    # load back to cpu for speedups 
    token_ids = token_ids.to("cpu")
    token_perplexity = token_perplexity.to("cpu")
    char_token_mask = char_token_mask.to("cpu")
    
    tokens_to_group, group_masks, token_groups, group_positions = self._detect_group_tokens(
        token_ids, token_perplexity, char_token_mask, avoid_duplicate=avoid_duplicate, cal_mask_device=cal_mask_device
    ) # consider duplicate here

    # Process vocabulary additions
    token_addition, embed_cache = _prep_input_vocabulary_addition(
        self, input_ids, tokens_to_group, group_positions, reps
    )
    
    # Update caches efficiently using dict operations : switch to batch operations 
    self.embed_cache = run_avg_dict_merge(self.embed_cache, embed_cache)
    self.token_addition = add_dict_merge(self.token_addition, token_addition)
    

@timing_decorator
def _cache_vocabulary_change(self, texts=None, input_ids=None, target_ids=None, avoid_duplicate: bool = False, 
                             cal_mask_device: str = "cpu"):
    """Prepares vocabulary change for text batch"""
    
    # print("Begin vocabulary change caching ...")
    t0 = time.time()
    # Get all required data in one inference pass
    res = self.inference(text=texts, input_ids=input_ids, target_ids=target_ids, return_representation=True, return_device=cal_mask_device)
    input_ids, token_ids, token_perplexity, char_token_mask, reps = (
        res['input_ids'], res['token_ids'], 
        res['token_perplexity'], res['char_token_mask'], res['reps']
    )
    # print(f" - Inference took: {time.time() - t0:.4f} seconds")

    # t1 = time.time()
    # # Process removals and groupings in parallel if possible
    # tokens_to_remove, remove_positions, remove_mask, remove_groups = self._detect_remove_tokens(
    #     token_ids, token_perplexity, char_token_mask, cal_mask_device=cal_mask_device
    # )
    
    # print(f" - Remove token detection took: {time.time() - t1:.4f} seconds")
    
    t2 = time.time()
    
    # load back to cpu for speedups 
    token_ids = token_ids.to("cpu")
    token_perplexity = token_perplexity.to("cpu")
    char_token_mask = char_token_mask.to("cpu")
    
    tokens_to_group, group_masks, token_groups, group_positions = self._detect_group_tokens(
        token_ids, token_perplexity, char_token_mask, avoid_duplicate=avoid_duplicate, cal_mask_device=cal_mask_device
    ) # consider duplicate here
    print(f" - Group token detection took: {time.time() - t2:.4f} seconds")

    t3 = time.time()
    # Process vocabulary additions
    token_addition, embed_cache, project_cache = _prep_vocabulary_addition(
        self, input_ids, tokens_to_group, group_positions, reps
    )
    print(f" - Vocabulary addition prep took: {time.time() - t3:.4f} seconds")
    
    
    # t4 = time.time()
    # token_removal = _prep_vocabulary_removal(tokens_to_remove)
    # print(f" - Token removal prep took: {time.time() - t4:.4f} seconds")
    
    t5 = time.time()
    # Update caches efficiently using dict operations : switch to batch operations
    
    self.embed_cache = run_avg_dict_merge(self.embed_cache, embed_cache)
    self.project_cache = run_avg_dict_merge(self.project_cache, project_cache)
    self.token_addition = add_dict_merge(self.token_addition, token_addition)
    # self._token_removal = add_dict_merge(self._token_removal, token_removal)
    
    print(f" - Cache updates took: {time.time() - t5:.4f} seconds")

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
    # I've disabled repetitive token addition, we might as well randomize the order here
    tokens_to_group = list(self.token_addition.keys())
    random.shuffle(tokens_to_group)
    tokens_to_group = tokens_to_group[:max_size_change]
    eom_tokens, pair_token_groups = self.tokenizer.add_tokens(tokens_to_group, in_place=True)

    print(f":: Total {len(tokens_to_group)} token groups, added {len(pair_token_groups)} pairwise merges")
    print(f":: Total {len(eom_tokens)} new tokens added")
    print(f":: Total {len(eom_tokens)} new embeddings to be added into model")
    
    # wte addition 
    if len(eom_tokens) > 0: 
        print(":: Original WTE vocab size: ", self.model.transformer.wte.weight.shape[0])
        embed_vecs = torch.stack([self.embed_cache[id] for id in eom_tokens])
        new_wte = add_token_wte(self.model.transformer.wte, embed_vecs)
        print(":: New WTE vocab size: ", new_wte.weight.shape[0])
    else: 
        new_wte = self.model.transformer.wte

    # project addition 
    if len(eom_tokens) > 0: 
        print(":: Original LM head vocab size: ", self.model.lm_head.weight.shape[0])
        project_vecs = torch.stack([self.project_cache[id] for id in eom_tokens])
        new_lm_head = add_token_lm_head(self.model.lm_head, project_vecs)
        print(":: New LM head vocab size: ", new_lm_head.weight.shape[0])
    else: 
        new_lm_head = self.model.lm_head
        
    assert new_wte.weight.shape[0].item() == new_lm_head.weight.shape[0].item(), "vocab size mismatch between wte and lm_head"
    assert self.tokenizer.vocab_size == new_wte.weight.shape[0].item(), "vocab size mismatch between tokenizer and wte"
    print(":: Vocab size matches between updated tokenizer and model in line 375.")
    
    update_model(self, new_wte, new_lm_head)
    
def add_to_input_vocab(self, max_size_change: int = 5000): 
    
    # tokenizer addition 
    tokens_to_group = list(self.token_addition.keys())
    random.shuffle(tokens_to_group)
    tokens_to_group = tokens_to_group[:max_size_change]
    eom_tokens, pair_token_groups = self.tokenizer.add_tokens(tokens_to_group, in_place=True) # self.tokenizer is the input vocab tokenizer

    print(f":: Total {len(tokens_to_group)} token groups, added {len(pair_token_groups)} pairwise merges")
    print(f":: Total {len(eom_tokens)} new tokens added")
    
    # wte addition 
    if len(eom_tokens) > 0: 
        embed_vecs = torch.stack([self.embed_cache[id] for id in eom_tokens])
        new_wte = add_token_wte(self.model.transformer.wte, embed_vecs)
    else: 
        new_wte = self.model.transformer.wte

    update_model(self, new_wte, self.model.lm_head) # only update wte but keep lm_head as is 
    
    
@timing_decorator
def remove_from_vocab(self, max_size_change: int = 500): 
    
    print(f":: Removing from vocabulary, selecting {max_size_change} from {len(self.token_removal)} candidates ...")
    
    tokens_to_remove = list(self.token_removal.keys())[:max_size_change]
    # tokens_to_remove = sorted(self.token_removal.keys(), key=lambda x: self.token_removal[x], reverse=True)[:max_size_change]
    
    print(f":: Picked total {len(tokens_to_remove)} tokens to remove")
    self.tokenizer.remove_tokens(tokens_to_remove)
    new_wte = remove_token_wte(self.model.transformer.wte, tokens_to_remove)
    new_lm_head = remove_token_lm_head(self.model.lm_head, tokens_to_remove)
    update_model(self, new_wte, new_lm_head)
    

# truncate vocab size of model
def truncate_model(model, target_vocab_size: int):
    # Get current dimensions
    hidden_dim = model.transformer.wte.weight.size(1)
    
    # Create new embedding layer with truncated size
    new_wte = torch.nn.Embedding(target_vocab_size, hidden_dim)
    new_wte.weight.data = model.transformer.wte.weight.data[:target_vocab_size]
    
    # Create new linear layer with truncated size
    new_lm_head = torch.nn.Linear(hidden_dim, target_vocab_size, bias=model.lm_head.bias is not None)
    new_lm_head.weight.data = model.lm_head.weight.data[:target_vocab_size]
    if model.lm_head.bias is not None:
        new_lm_head.bias.data = model.lm_head.bias.data[:target_vocab_size]
    
    # Update the model's layers
    model.transformer.wte = new_wte
    model.lm_head = new_lm_head
    
    return model