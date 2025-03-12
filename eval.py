import os
import pickle, json
import torch
import numpy as np 
from functools import partial
from model import GPTConfig, GPT
from spline_model import SplineGPTConfig, SplineGPT
from magicab import ETokenizer
from magicab import evaluate_bpc, evaluate_token_stat

# -----------------------------------------------------------------------------
out_dir = 'checkpoint/base' # ignored if init_from is not 'resume'
dataset = "composio"
data_subfolder = ""
model_type="GPT"
run_idx=1
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
compile = True # use PyTorch 2.0 to compile the model to be faster
batch_size=256 # batch size of evaluation data
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


# load model checkpoint from 'out_dir'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print("Checkpoint path: ", ckpt_path)

checkpoint = torch.load(ckpt_path, map_location=device)
model = GPT.load_model(checkpoint, device)
model.eval()
model = model.to(device)
if compile: 
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
block_size = model.config.block_size

# Load tokenizer from checkpoint 
tokenizer = ETokenizer.load(checkpoint['tokenizer_path'])
print("Loaded tokenizer with vocab size: ", tokenizer.vocab_size)

data_dir = os.path.join('data', dataset) if data_subfolder == "" else os.path.join('data', dataset, data_subfolder)

if 'enwiki' in data_dir:
    from magicab.data import get_batch 
    get_batch = get_batch 
else: 
    from magicab.data import get_batch_slice # require pad_token_id
    get_batch = lambda data_dir, split, block_size, batch_size, device: get_batch_slice(data_dir, split, tokenizer.pad_token_id, block_size, batch_size, device)
    
bpc = evaluate_bpc(model, tokenizer, data_dir, 256, batch_size, "cpu", device, get_batch, num_batches=10)
print(f"BPC of loaded checkpoint: {bpc}")

token_count_dict, token_bpc_dict = evaluate_token_stat(model, tokenizer, data_dir, 256, 256, device, get_batch, num_batches=10)
print(f"Token count dict: {token_count_dict}")
print(f"Token BPC dict: {token_bpc_dict}")

# Save info 
info = {"run_idx": run_idx, "bpc": bpc, "model_type": model_type, "config": checkpoint['model_args'], "token_count_dict": token_count_dict, "token_bpc_dict": token_bpc_dict}
with open(os.path.join(out_dir, f"info_{run_idx}.pkl"), "wb") as f:
    pickle.dump(info, f)