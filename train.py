"""
Training script using Hugging Face Accelerate.

To run on a single GPU:
$ accelerate launch train.py --config config/default_config.json

To run with multiple GPUs:
$ accelerate launch --multi_gpu train.py --config config/default_config.json
"""

import os
import math
import json
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from model import GPTConfig, GPT

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('--config', type=str, default='config/default_config.json',
                      help='path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_batch(data_dir, split, block_size, batch_size, device):
    filename = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_lr(iter_num, config):
    if not config['lr_schedule']['decay_lr']:
        return config['optimizer']['learning_rate']
    
    warmup_iters = config['lr_schedule']['warmup_iters']
    lr_decay_iters = config['lr_schedule']['lr_decay_iters']
    min_lr = config['lr_schedule']['min_lr']
    learning_rate = config['optimizer']['learning_rate']

    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(model, data_dir, config, device):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        losses[split] = torch.zeros(config['io']['eval_iters']).to(device)
        for k in range(config['io']['eval_iters']):
            X, Y = get_batch(data_dir, split, config['data']['block_size'], 
                           config['data']['batch_size'], device)
            logits, loss = model(X, Y)
            losses[split][k] = loss.item()
    model.train()
    return {k: v.mean().item() for k, v in losses.items()}

def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")

    # Set up data directory
    data_dir = os.path.join('data', config['data']['dataset'])
    if accelerator.is_main_process:
        os.makedirs(config['io']['out_dir'], exist_ok=True)
    
    # Set seeds for reproducibility
    set_seed(1337)

    # Initialize model
    if config['io']['init_from'] == 'scratch':
        print("Initializing a new model from scratch")
        meta_vocab_size = None
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                meta_vocab_size = meta['vocab_size']
        
        vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
        model_args = dict(
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            n_embd=config['model']['n_embd'],
            block_size=config['data']['block_size'],
            bias=config['model']['bias'],
            vocab_size=vocab_size,
            dropout=config['model']['dropout']
        )
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    else:
        raise NotImplementedError(f"init_from={config['io']['init_from']} not implemented yet")

    # Initialize optimizer
    optimizer = model.configure_optimizers(
        config['optimizer']['weight_decay'],
        config['optimizer']['learning_rate'],
        (config['optimizer']['beta1'], config['optimizer']['beta2']),
        device_type='cuda'
    )

    # Prepare model and optimizer with accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Initialize wandb if enabled
    if config['wandb']['enabled'] and accelerator.is_main_process:
        import wandb
        wandb.init(project=config['wandb']['project'], 
                  name=config['wandb']['run_name'])

    # Training loop
    iter_num = 0
    best_val_loss = float('inf')
    
    progress_bar = tqdm(
        range(config['optimizer']['max_iters']),
        disable=not accelerator.is_local_main_process
    )

    X, Y = get_batch(data_dir, 'train', config['data']['block_size'], 
                     config['data']['batch_size'], device)

    while iter_num < config['optimizer']['max_iters']:
        # determine learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets
        if iter_num % config['io']['eval_interval'] == 0 and accelerator.is_main_process:
            losses = estimate_loss(model, data_dir, config, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if config['wandb']['enabled']:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })

            if losses['val'] < best_val_loss or config['io']['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {config['io']['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['io']['out_dir'], 'ckpt.pt'))

        # forward backward update
        with accelerator.accumulate(model):
            logits, loss = model(X, Y)
            loss = loss / config['data']['gradient_accumulation_steps']
            accelerator.backward(loss)
            
            if config['optimizer']['grad_clip'] != 0.0:
                accelerator.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            
            optimizer.step()
            optimizer.zero_grad()

        # fetch next batch
        X, Y = get_batch(data_dir, 'train', config['data']['block_size'], 
                        config['data']['batch_size'], device)
        
        # update progress
        progress_bar.update(1)
        iter_num += 1

    accelerator.wait_for_everyone()
    progress_bar.close()

if __name__ == '__main__':
    main()