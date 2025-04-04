import os
import pickle
from contextlib import nullcontext
import torch
import numpy as np 
from model import GPTConfig, GPT
from magicab import ETokenizer, Magicab, update_magicab, save_magicab
from data.enwiki.util import prepare_enwiki_data

# -----------------------------------------------------------------------------
load_dir = "checkpoint/base"
new_dir = "checkpoint/update"
dataset = "enwiki" # "composio/datasets"
data_subfolder = ""
device = 'cuda'
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

if device == 'cuda': 
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# load model checkpoint from 'load_dir'
ckpt_path = os.path.join(load_dir, 'ckpt.pt')
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
magicab = Magicab(tokenizer=tokenizer, model=model, checkpoint_dir=load_dir,
                  group_perplexity_threshold=thres)

# Update Magicab Vocabulary & Training Data 
if data_subfolder == "": 
    data_dir = os.path.join('data', dataset)
else: 
    data_dir = os.path.join('data', dataset, data_subfolder)
    
if 'enwiki' in data_dir:
    from magicab.data import get_batch 
    get_batch = get_batch 
else: 
    from magicab.data import get_batch_slice 
    get_batch = lambda data_dir, split, block_size, batch_size, device: get_batch_slice(data_dir, split, tokenizer.pad_token_id, block_size, batch_size, device)

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
                get_batch_fn=get_batch,
                max_size_change=max_size_change)

print("After Update Tokenizer vocab size: ", magicab.tokenizer.vocab_size)

# Save model checkpoint & tokenizer | checkpoint is updated inside save_magicab
save_magicab(checkpoint, magicab, new_dir)


# Encode training data (TBD: unify all data encoding functionals)
if 'enwiki' in data_dir:
    prepare_enwiki_data(clean=True, tokenizer=magicab.tokenizer, checkpoint_dir=new_dir, data_subfolder=data_subfolder)
elif 'composio' in data_dir: 
    print("Composio dataset encoding functional here ...")
    from data.composio.util import process_composio_pt_data
    process_composio_pt_data(
        datasets_dir=os.path.join('data', dataset),
        save_dir=os.path.join('data', 'composio', data_subfolder), # little twist
        tokenizer_path= new_dir + "/tokenizer.json",
        mode="byte",
        init_vocab=False,
        tokenizer=magicab.tokenizer
    )
elif 'openweb' in data_dir: 
    import subprocess 
    command_str = f"python data/{dataset}/process_pt_data.py --tokenizer_path={new_dir} --init_vocab=False"
    subprocess.run(command_str, shell=True, check=True)