import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import torch 


def alpaca_to_conversation(row):
    # Basic conversation with instruction as user input
    conv = [{"user": row["instruction"]}]
    
    # Add input if it exists and isn't empty
    if "input" in row and row["input"] and not pd.isna(row["input"]):
        # Append input to instruction or create a separate turn based on your preference
        conv[0]["user"] += "\n" + row["input"]
    
    # Add output as assistant response
    if "output" in row and row["output"] and not pd.isna(row["output"]):
        conv.append({"assistant": row["output"]})
    
    return conv

def preprocess_alpaca_data(tok, max_samples=None):
    
    df = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    
    # Process a sample to test
    sample_idx = 0
    sample_row = df.iloc[sample_idx]
    sample_conv = alpaca_to_conversation(sample_row)
    print(f"Sample conversation format:\n{sample_conv}")

    # Process the sample with your tokenizer
    sample_res_dict = tok.prepare_sft_data(sample_conv, return_dict=True)
    sample_formatted_text = sample_res_dict["text"]
    sample_loss_mask = sample_res_dict["loss_mask"]

    print(f"\nFormatted text length: {len(sample_formatted_text)}")
    print(f"Loss mask length: {len(sample_loss_mask)}")

    # Now process the entire dataset (or a subset for testing)
    processed_data = []
    num_samples = len(df) if max_samples is None else min(max_samples, len(df))

    for i in range(num_samples):
        row = df.iloc[i]
        conv = alpaca_to_conversation(row)
        
        try:
            res_dict = tok.prepare_sft_data(conv, return_dict=True)
            
            # Store the processed data
            processed_data.append({
                "text": res_dict["text"],
                "loss_mask": res_dict["loss_mask"],
                "original_idx": i
            })
            
        except Exception as e:
            # print(f"Error processing row {i}: {e}")
            continue
            
    print(f"\nSuccessfully processed {len(processed_data)} samples")
    
    return processed_data


# For memory-mapped binary storage
def prepare_for_memmap(data_list, tok):
    # Extract all texts and convert to token IDs
    all_texts = [item["text"] for item in data_list]
    all_tokens = [tok.encode(text) for text in all_texts]
    
    # Find maximum sequence length for padding
    max_len = max(len(tokens) for tokens in all_tokens)
    
    # Create arrays for tokens and loss masks
    token_array = np.zeros((len(data_list), max_len), dtype=np.int32)
    mask_array = np.zeros((len(data_list), max_len), dtype=np.int8)
    
    # Fill arrays
    for i, item in enumerate(data_list):
        tokens = tok.encode(item["text"])
        token_array[i, :len(tokens)] = tokens
        mask = item["loss_mask"]
        mask_array[i, :len(mask)] = mask
    
    # Save metadata (just original indices and sequence lengths)
    metadata = {
        "original_indices": [item["original_idx"] for item in data_list],
        "sequence_lengths": [len(tok.encode(item["text"])) for item in data_list]
    }
    
    return token_array, mask_array, metadata


def prepare_alpaca_data(tokenizer, block_size=512): 
    
    processed_data = preprocess_alpaca_data(tokenizer)

    # split into train/val, store as .bin files
    train_data, val_data = train_test_split(processed_data, test_size=0.1, random_state=42)

    # Create memory-mapped arrays
    train_tokens, train_masks, train_metadata = prepare_for_memmap(train_data, tokenizer)
    val_tokens, val_masks, val_metadata = prepare_for_memmap(val_data, tokenizer)

    # Save only numpy arrays and minimal metadata
    np.save("data/alpaca/train_tokens.npy", train_tokens)
    np.save("data/alpaca/train_masks.npy", train_masks)
    np.save("data/alpaca/val_tokens.npy", val_tokens)
    np.save("data/alpaca/val_masks.npy", val_masks)

    # Save minimal metadata
    np.savez("data/alpaca/train_metadata.npz", **train_metadata)
    np.savez("data/alpaca/val_metadata.npz", **val_metadata)
    
    print(f"- Total Training samples: {len(train_tokens)}")
    print(f"- Total Validation samples: {len(val_tokens)}")

    
def truncate_batch(input_ids, target_ids, loss_mask):
    
    # Find the last non-zero index for each sequence
    non_zero_indices = (input_ids != 0).cumsum(dim=1).argmax(dim=1)
    
    # Get the maximum index (plus 1 for safety margin)
    max_len = max(non_zero_indices.max().item() + 1, 1)
    
    # Truncate all tensors to max_len
    trunc_input_ids = input_ids[:, :max_len]
    
    trunc_target_ids = target_ids[:, :max_len]
    trunc_loss_mask = loss_mask[:, :max_len]
    return trunc_input_ids, trunc_target_ids, trunc_loss_mask


def get_batch(batch_size, split='train', device='mps'):
     
    if split == 'train': 
        tokens = np.load("data/alpaca/train_tokens.npy", mmap_mode='r')
        masks = np.load("data/alpaca/train_masks.npy", mmap_mode='r')
    else: 
        tokens = np.load("data/alpaca/val_tokens.npy", mmap_mode='r')
        masks = np.load("data/alpaca/val_masks.npy", mmap_mode='r')
        
    ix = torch.randint(tokens.shape[0], (batch_size,))
    
    input_ids = torch.stack([torch.from_numpy((tokens[i][:-1]).astype(np.int64)) for i in ix])
    target_ids = torch.stack([torch.from_numpy((tokens[i][1:]).astype(np.int64)) for i in ix])
    loss_mask = torch.stack([torch.from_numpy((masks[i][1:]).astype(np.int64)) for i in ix])
    
    input_ids, target_ids, loss_mask = truncate_batch(input_ids, target_ids, loss_mask)
    
    if 'cuda' in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously
        input_ids, target_ids, loss_mask = input_ids.pin_memory().to(device, non_blocking=True), target_ids.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        input_ids, target_ids, loss_mask = input_ids.to(device), target_ids.to(device), loss_mask.to(device)
    
    return input_ids, target_ids, loss_mask