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
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Prepare enwiki dataset for character-level language modeling')
parser.add_argument('--clean', action='store_true', help='Use cleaned version of the dataset')
args = parser.parse_args()

# Replace clean_txt variable with args.clean
data_dir = os.path.join('data', 'enwiki')
input_file_path = os.path.join(data_dir, 'enwik8')
if not os.path.exists(input_file_path):
    os.system("bash data/enwiki/download_data.sh")

if args.clean:
    clean_file_path = os.path.join(data_dir, 'enwik8_clean.txt')
    if not os.path.exists(clean_file_path):
        os.system("python data/enwiki/filter_data.py")
else: 
    clean_file_path = input_file_path
    
print("Input file path: ", clean_file_path)
# Create the data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)
    
# Read the enwiki9 data
with open(clean_file_path, 'r', encoding='utf-8') as f:
    data = f.read()   

# get all the unique characters that occur in this text
data_bytes = data.encode('utf-8')
sorted_bytes = sorted(list(set(data_bytes)))
byte_size = len(sorted_bytes)

chars = sorted(list(set(data)))
vocab_size = len(chars)

print("all the unique characters:", ''.join(chars))
print(f"char vocab size: {vocab_size:,}")
print(f"byte vocab size: {byte_size:,}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train, validation and test splits
n = len(data)
if args.clean:
    train_bytes = data_bytes[:50_000_000]
    val_bytes = data_bytes[50_000_000:52_000_000]
    test_bytes = data_bytes[52_000_000:]
else:
    train_bytes = data_bytes[:90_000_000]
    val_bytes = data_bytes[90_000_000:95_000_000]
    test_bytes = data_bytes[95_000_000:100_000_000]
    
train_data = bytes(train_bytes).decode('utf-8')
val_data = bytes(val_bytes).decode('utf-8')
test_data = bytes(test_bytes).decode('utf-8')

# encode all three splits to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)
print(f"Total tokens: {n}")
print(f"Total bytes: {len(data_bytes)}")
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