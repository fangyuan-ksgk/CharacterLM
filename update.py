import os
import pickle
from contextlib import nullcontext
import torch
import numpy as np 
from model import GPTConfig, GPT
from magicab import ETokenizer, Magicab, update_magicab, save_magicab
from data.enwiki.util import prepare_enwiki_data

# -----------------------------------------------------------------------------
out_dir = 'checkpoint/base' # ignored if init_from is not 'resume'
new_dir = "checkpoint/new"
dataset = "enwiki"
data_subfolder = ""
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
block_size=256 # context length of model 
batch_size=256 # batch size of training data
max_size_change = 2000 # max number of tokens to add
thres = 0.6 # below this threshold, tokens will be grouped together
target_vocab_size = 92 # directly truncate to base vocabulary size
truncate_vocab = False # whether to truncate vocabulary or not

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
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
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

# Initialize Magicab object 
magicab = Magicab(tokenizer=tokenizer, model=model, checkpoint_dir=out_dir,
                  group_perplexity_threshold=thres)

# Update Magicab Vocabulary & Training Data 
if data_subfolder == "": 
    data_dir = os.path.join('data', dataset)
else: 
    data_dir = os.path.join('data', dataset, data_subfolder)

# Update Magicab Vocabulary 
if truncate_vocab: 
    print("Truncating vocabulary to target size: ", target_vocab_size)
    magicab.truncate_vocab(target_vocab_size)
else: 
    print("Growing vocabulary with maximum size change: ", max_size_change)
    update_magicab(magicab, 
                data_dir, 
                block_size=block_size, 
                batch_size=batch_size, 
                device_type=device,
                max_size_change=max_size_change)

print("After Update Tokenizer vocab size: ", magicab.tokenizer.vocab_size)

# Update Training Data 
prepare_enwiki_data(clean=True, tokenizer=magicab.tokenizer, checkpoint_dir=new_dir)

# Save model checkpoint & tokenizer | checkpoint is updated inside save_magicab
save_magicab(checkpoint, magicab, new_dir)