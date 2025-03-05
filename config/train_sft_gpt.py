# Parameter for SFT
model_dir = "checkpoint/gpt_tiny/base"
out_dir = 'checkpoint/gpt_tiny/sft_base'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'sft_gpt'
wandb_run_name = 'gpt'

dataset = "alpaca"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512 # context of up to 512 previous characters

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
magicab_interval = 1000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially