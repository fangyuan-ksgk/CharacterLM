import os
import pickle, json
from contextlib import nullcontext
import torch
import tiktoken
import numpy as np 
from model import GPTConfig, GPT
from spline_model import SplineGPTConfig, SplineGPT
from magicab import ETokenizer, Magicab, update_magicab, save_magicab
from data.enwiki.util import prepare_enwiki_data

# -----------------------------------------------------------------------------
out_dir = 'checkpoint/base' # ignored if init_from is not 'resume'
new_dir = "checkpoint/new"
model_type = "GPT"
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
block_size=256 # context length of model 
batch_size=256 # batch size of training data
max_size_change = 2000 # max number of tokens to add
thres = 0.6 # below this threshold, tokens will be grouped together
run_idx = 0 
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# load model checkpoint from 'out_dir'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
if model_type == "GPT": 
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
elif model_type == "SplineGPT": 
    splineconf = SplineGPTConfig(**checkpoint['model_args'])
    model = SplineGPT(splineconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Load tokenizer from checkpoint 
tokenizer = ETokenizer.load(checkpoint['tokenizer_path'])


# Update Magicab Vocabulary & Training Data 
data_dir = os.path.join('data', 'enwiki')


from magicab.magicab import evaluate_bpc, evaluate_token_stat

bpc = evaluate_bpc(model, tokenizer, "data/enwiki", 256, 256, "cpu", device, num_batches=10)
print(f"BPC of loaded checkpoint: {bpc}")

# save token statistics, too, will come handy for experiments
token_count_dict, token_bpc_dict = evaluate_token_stat(model, tokenizer, "data/enwiki", 256, 256, "cpu", device, num_batches=10)
print(f"Token count dict: {token_count_dict}")
print(f"Token BPC dict: {token_bpc_dict}")

# Save info 
info = {"run_idx": run_idx, "bpc": bpc, "model_type": model_type, "config": checkpoint['model_args']}
with open(os.path.join(out_dir, f"info_{run_idx}.pkl"), "wb") as f:
    pickle.dump(info, f)