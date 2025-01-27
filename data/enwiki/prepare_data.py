"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# Adjust paths for running from parent directory
data_dir = os.path.join('data', 'enwiki')
input_file_path = os.path.join(data_dir, 'filtered_enwiki9.txt')
print("Input file path: ", input_file_path)
# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(input_file_path):
    # Change to the script's directory first, then execute download_data.sh
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.system('bash download_data.sh')
    os.system('bash filter_data.sh')
    
# Read the enwiki9 data
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train, validation and test splits
n = len(data)
train_data = data[:90_000_000]  # first 90M characters
val_data = data[90_000_000:95_000_000]  # next 5M characters
test_data = data[95_000_000:100_000_000]  # final 5M characters

# encode all three splits to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"test has {len(test_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
test_ids.tofile(os.path.join(data_dir, 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)