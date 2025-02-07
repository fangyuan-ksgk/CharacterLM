import os, torch, pickle 
from model import GPT, GPTConfig
from magicab import ETokenizer 
from magicab import Magicab
from data.enwiki.util import prepare_enwiki_data
from magicab import update_magicab
import time 

device = "cuda"
out_dir = "out-enwiki-char"
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

# Load model 
model = GPT.load_model(checkpoint, device)

# Load tokenizer 
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
tok = ETokenizer(char_vocab=meta['itos'])


magicab = Magicab(model, tok, checkpoint_dir="checkpoint/base", spike_perplexity_threshold=5.0, group_perplexity_threshold=0.2)

# Update data with tokenizer (11min --> 40s with rust tokenization)
t0 = time.time()
prepare_enwiki_data(clean=True, tokenizer=magicab.tokenizer) # in-place update on trianing data 
t1 = time.time()
print("Time taken to update data: ", t1 - t0)

data_dir = "data/enwiki/"
block_size = 256
batch_size = 256
device_type = "cpu"

# Update vocabulary on training data
# vocabulary grows huge very soon, need to cap on allowed size change
update_magicab(magicab, data_dir, block_size, batch_size, device_type, max_size_change=200) 
t2 = time.time() 
print("Time taken to update vocabulary: ", t2 - t1)