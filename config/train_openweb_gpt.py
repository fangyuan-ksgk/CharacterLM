# train byte-level GPT on openweb pre-train dataset

# load_dir = 'checkpoint/smol_byte/gpt_tiny'
init_from = 'scratch'
mode = 'byte'
out_dir = 'checkpoint/openweb_gpt'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'openweb_gpt'
wandb_run_name = 'gpt_tiny'

dataset = 'openweb'
data_subfolder="gpt_tiny"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512 # context of up to 512 previous characters

# baby GPT model :) | GPT-2 Small Scale Model 
model_type = "GPT"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 5e-4 # gpt small 124M learning rate
max_iters = 50000
magicab_interval = 1000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially