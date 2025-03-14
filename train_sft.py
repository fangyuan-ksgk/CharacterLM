"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import glob
import time
import math
import shutil 
import pickle
from tqdm import tqdm
from contextlib import nullcontext

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from spline_model import SplineGPTConfig, SplineGPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
model_dir = "checkpoint/base"
out_dir = "checkpoint/sft"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
model_type = "GPT"
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'alpaca-sft'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'alpaca' # hard-coded for now 

gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512

# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 5000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 20000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


def setup_training_environment():
    """Set up distributed training if needed and return environment details."""
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device_name = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device_name)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        local_gradient_accumulation_steps = gradient_accumulation_steps // ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device_name = device
        local_gradient_accumulation_steps = gradient_accumulation_steps
    
    # Create output directory
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    # Set up random seed
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set up mixed precision
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    return {
        'ddp': ddp,
        'device': device_name,
        'master_process': master_process,
        'ddp_world_size': ddp_world_size,
        'ctx': ctx,
        'device_type': device_type,
        'gradient_accumulation_steps': local_gradient_accumulation_steps
    }
    
    
from data.alpaca.util import get_batch


def get_lr(iter_num):
    """Calculate learning rate based on warmup and decay schedule."""
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / (warmup_iters + 1)
    if iter_num > lr_decay_iters:
        return min_lr
    
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def init_model():
    
    """Initialize model based on configuration."""
    if init_from == 'resume':
        print(f"Resuming training from {model_dir}")
        ckpt_path = os.path.join(model_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Extract model configuration from checkpoint
        checkpoint_model_args = checkpoint['model_args']
        model_args = {}
        
        # Common parameters for all model types
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
            
        # Create appropriate model type
        if model_type == "GPT":
            config = GPTConfig(**model_args)
            model = GPT(config)
        elif model_type == "SplineGPT":
            # Add SplineGPT specific parameters
            if 'spline_control_layers' in checkpoint_model_args:
                model_args['spline_control_layers'] = checkpoint_model_args['spline_control_layers']
            config = SplineGPTConfig(**model_args)
            model = SplineGPT(config)
            
        # Load weights
        state_dict = checkpoint['model']
        # Remove unwanted prefix if present
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
    elif init_from.startswith('gpt2'):
        assert model_type == "GPT", "Only GPT is supported for loading from GPT-2 weights"
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        model = GPT.from_pretrained(init_from, dict(dropout=0.1))
        # Extract model configuration
        model_args = {}
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    
    # Adjust block size if needed
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size
    
    return model, model_args


@torch.no_grad()
def estimate_loss(model, ctx):
    """Evaluate model on training and validation sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, loss_mask = get_batch(batch_size, split, device)
            with ctx:
                logits, loss = model(X, Y, reduction='none')
            losses[k] = (loss * loss_mask).mean().item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(model, optimizer, model_args, iter_num, best_val_loss, out_dir, is_ddp=False):
    """Save model checkpoint."""
    raw_model = model.module if is_ddp else model
    tokenizer_path = os.path.join(out_dir, 'tokenizer.json')
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': {k: globals()[k] for k in config_keys},
        "tokenizer_path": tokenizer_path
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

def save_tokenizer(model_dir, out_dir): 
    tokenizer_files = glob.glob(os.path.join(model_dir, "*.json"))
    for tokenizer_file in tokenizer_files:
        shutil.copy(tokenizer_file, os.path.join(out_dir, os.path.basename(tokenizer_file)))
    
def train_step(model, X, Y, loss_mask, optimizer, scaler, ctx, grad_accum_steps, is_ddp=False):
    """Execute a single training step with gradient accumulation."""
    # Forward backward update with gradient accumulation
    for micro_step in range(grad_accum_steps):
        if is_ddp:
            # Only sync gradients at the last micro step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        
        with ctx:
            logits, loss = model(X, Y, reduction='none')
            loss = (loss * loss_mask).mean()
            loss = loss / grad_accum_steps  # Scale for gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    return loss.item() * grad_accum_steps


def train():
    # Setup training environment
    global device  # Update the global device variable
    env = setup_training_environment()
    device = env['device']
    
    # Initialize model and optimizer
    model, model_args = init_model()
    model.to(device)
    
    # Compile model if requested
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    # Wrap model in DDP if needed
    if env['ddp']:
        model = DDP(model, device_ids=[int(device.split(':')[1])])
    
    # Initialize optimizer and scaler
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), env['device_type']
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Initialize training state
    iter_num = 0
    best_val_loss = float('inf')
    
    # Initialize wandb if enabled
    if wandb_log and env['master_process']:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name)
    
    # Add progress bar
    pbar = tqdm(total=max_iters, initial=iter_num, dynamic_ncols=True)
    
    # Main training loop
    X, Y, loss_mask = get_batch(batch_size, 'train', device)  # Get first batch
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while iter_num < max_iters:
        # Set learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation and checkpointing
        if iter_num % eval_interval == 0 and env['master_process']:
            losses = estimate_loss(model, env['ctx'])
            
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to wandb if enabled
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            
            # Save checkpoint if needed
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(
                        model, optimizer, model_args, iter_num, best_val_loss, out_dir, is_ddp=env['ddp']
                    )
        
        # Training step
        loss = train_step(
            model, X, Y, loss_mask, optimizer, scaler, env['ctx'], 
            env['gradient_accumulation_steps'], is_ddp=env['ddp']
        )
        
        # Get next batch
        X, Y, loss_mask = get_batch(batch_size, 'train', device)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0 and env['master_process']:
            # Calculate and log metrics
            lossf = loss  # This is already scaled properly in train_step
            if local_iter_num >= 5:  # Let training settle
                raw_model = model.module if env['ddp'] else model
                mfu = raw_model.estimate_mfu(batch_size * env['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # Update counters
        iter_num += 1
        local_iter_num += 1
        if env['master_process']:
            pbar.update(1)
    
    # Clean up
    if env['ddp']:
        destroy_process_group()
    pbar.close()
    
    # save tokenizer
    save_tokenizer(model_dir, out_dir)
    
    
if __name__ == "__main__":
    # Get all configuration keys for saving
    config_keys = [k for k,v in globals().items() 
                    if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

    # Load configuration from command line or config file
    exec(open('configurator.py').read())

    train()