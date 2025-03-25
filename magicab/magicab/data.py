import torch 
import numpy as np 
import random 
import os 
from tqdm import tqdm
import glob


def save_sequences_for_memmap(sequences, file_path):
    """save sequences (list of list) with meta data encoded"""        
    lengths = [len(seq) for seq in sequences]
    offsets = [0]
    for length in lengths[:-1]:
        offsets.append(offsets[-1] + length)
    
    with open(file_path, 'wb') as f:
        np.array([len(sequences)], dtype=np.int32).tofile(f)
        np.array(offsets, dtype=np.int64).tofile(f)
        np.array(lengths, dtype=np.int32).tofile(f)
        for seq in sequences:
            np.array(seq, dtype=np.int32).tofile(f)
            
 
def get_split_path(data_dir, split): 
    """Multiple train files handler"""
    file_paths = glob.glob(os.path.join(data_dir, f'*{split}*.bin'))
    assert len(file_paths) > 0, f"No {split} files found in {data_dir}"
    random.shuffle(file_paths)
    return file_paths[0]


def get_batch_slice(data_dir, split, pad_token_id, block_size=512, batch_size=2, device='cpu'):
    
    file_path = get_split_path(data_dir, split)

    with open(file_path, 'rb') as f:
        n_seq = np.fromfile(f, dtype=np.int32, count=1)[0]
        offsets = np.fromfile(f, dtype=np.int64, count=n_seq)
        lengths = np.fromfile(f, dtype=np.int32, count=n_seq)
        
        # Calculate where the data section starts
        data_start = 4 + n_seq * 8 + n_seq * 4  # bytes
        
    # Randomly choose a sequence
    seq_indices = random.sample(range(n_seq), batch_size)
    seq_lengths = [lengths[i] for i in seq_indices]
    seq_offsets = [offsets[i] for i in seq_indices]
    
    # Choose start position for the block
    x_list = [] 
    y_list = []
    for seq_idx, seq_length, seq_offset in zip(seq_indices, seq_lengths, seq_offsets):
        if seq_length <= block_size:
            start_pos = 0
            actual_block_size = seq_length
        else:
            max_start = seq_length - block_size
            start_pos = random.randint(0, max_start)
            actual_block_size = block_size
    
        # Calculate how much to read - need an extra token for target if available
        read_size = min(actual_block_size + 1, seq_length - start_pos)
    
        # Create memmap and load the slice plus one extra token if available
        data_offset = data_start + seq_offset * 4  # 4 bytes per int32
        mmap = np.memmap(file_path, dtype=np.int32, mode='r', 
                        offset=data_offset + start_pos * 4, 
                        shape=(read_size,))
        
        # Convert to regular numpy array to detach from memmap
        data = mmap.copy()
        
        # Create input sequence x (properly padded to block_size)
        x_data = data[:actual_block_size]
        if len(x_data) < block_size:
            x_data = np.concatenate([x_data, np.full((block_size - len(x_data),), pad_token_id, dtype=np.int32)])
        
        # Create target sequence y (shifted by one position)
        if read_size > actual_block_size:  # We have an extra token
            y_data = data[1:read_size]
            if len(y_data) < block_size:  # Need padding
                y_data = np.concatenate([y_data, np.full((block_size - len(y_data),), pad_token_id, dtype=np.int32)])
        else:  # At the end of sequence, need to pad with one more token than x
            y_data = np.concatenate([data[1:], np.full((block_size - len(data) + 1,), pad_token_id, dtype=np.int32)])
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(x_data.astype(np.int64))
        y = torch.from_numpy(y_data.astype(np.int64))
        
        x_list.append(x)
        y_list.append(y)
        
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    # Move to device
    if 'cuda' in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


def get_batch(data_dir, split, block_size, batch_size, device):
    """Load a batch of data from disk."""
    # We recreate np.memmap every batch to avoid a memory leak
    file_path = get_split_path(data_dir, split)
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if 'cuda' in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


def compute_bpc(x, y, model, tokenizer): # corrected version
    per_token_nll = model(x, y, reduction='none')[1]  # shape: [batch_size, seq_len]
    per_token_char_count = torch.tensor([[len(tokenizer.vocab[id]) for id in tokens.tolist()] for tokens in y])  # shape: [batch_size, seq_len]
    print("  - average token entropy: ", per_token_nll.mean().item())
    print("  - average token char count: ", per_token_char_count.float().mean().item())

    # Sum total NLL and total character count
    total_nll = per_token_nll.sum()
    total_chars = per_token_char_count.sum()
    
    # Calculate true BPC across all characters
    bpc = (total_nll.to("cpu").detach() / total_chars) / torch.log(torch.tensor(2.0))
    return bpc


def evaluate_bpc(model, tokenizer, data_dir, block_size, batch_size, device, get_batch_fn, num_batches=10):
    total_bpc = 0 
    for _ in tqdm(range(num_batches), desc="Evaluating BPC"): 
        x, y = get_batch_fn(data_dir, 'val', block_size, batch_size, device)
        bpc_loss = compute_bpc(x, y, model, tokenizer)
        total_bpc += bpc_loss.mean()
    return total_bpc / num_batches