import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from magicab import ETokenizer, Magicab, update_magicab
from data.enwiki.util import prepare_enwiki_data

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
new_dir = "new_out"
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
block_size=256 # context length of model 
batch_size=256 # batch size of training data 
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
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
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Initialize Magicab Object
    if 'tokenizer_path' in meta: 
        tokenizer = ETokenizer.load(meta['tokenizer_path'])
    else: 
        tokenizer = ETokenizer(char_vocab=meta['itos'])
        
    magicab = Magicab(tokenizer=tokenizer, model=model, checkpoint_dir=out_dir)

# Update Magicab Vocabulary & Training Data 
from magicab import update_magicab
from data.enwiki.util import prepare_enwiki_data

data_dir = os.path.join('data', 'enwiki')


# Update Magicab Vocabulary 
update_magicab(magicab, 
               data_dir, 
               block_size=block_size, 
               batch_size=batch_size, 
               device_type=device,
               max_size_change=2000)

# Both Tokenizer and Model will be updated, so we'd need to save them 

# Update Training Data 
prepare_enwiki_data(clean=True, tokenizer=magicab.tokenizer) # relabel training data with updated vocabulary

# Save model checkpoint & tokenizer 
from magicab import save_magicab
save_magicab(checkpoint, magicab, out_dir)