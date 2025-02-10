import os
import pickle
import requests
import numpy as np
import argparse
import regex as re
from tqdm import tqdm
from magicab import ETokenizer


def clean_wiki_text(content):
    """Clean Wikipedia text content following wikifil.pl conventions"""
    # Remove redirect text content
    txt = re.sub('^#REDIRECT.*$', '', content)
    
    # Remove references with their content
    txt = re.sub(r'<ref[^<]*<\/ref>', '', txt)
    
    # Remove various wiki markup
    replacements = [
        (r'\{\{.*?\}\}', ''),  # Remove templates
        (r'\{\|.*?\|\}', ''),  # Remove tables
        (r"'{2,5}", ''),       # Remove bold/italic markers
        (r'\[\[(Category|Image|Media|File):.*?\]\]', ''),  # Remove media/category links
        (r'\|thumb\b|\|left\b|\|right\b|\|\d+px\b', ''),  # Remove image options
        (r'\[\[[a-z\-]*:[^\]]*\]\]', ''),  # Remove interlanguage links
        (r'\[\[[^\|\]]*\|', '[['),  # Clean wiki links, preserve visible text
        (r'\[\[|\]\]', ''),    # Remove remaining [[ and ]]
        (r'\[http:[^] ]*', '['),  # Clean URLs, preserve visible text
        (r'\[|\]', ''),        # Remove remaining [ and ]
        (r'={2,5}.*?={2,5}', ''),  # Remove headers
        (r'&lt;|&gt;|&quot;|&amp;|&nbsp;', ' '),  # Convert HTML entities
        (r'<[^>]*>', ''),      # Remove remaining HTML tags
        (r'[*#:;]+\s.*?\n', '\n'),  # Remove list items and indents
        (r'(\n\s*){3,}', '\n\n'),  # Normalize multiple newlines
        (r'^\s+|\s+$', '')     # Trim whitespace
    ]
    
    for pattern, replacement in replacements:
        txt = re.sub(pattern, replacement, txt, flags=re.MULTILINE|re.DOTALL)
    
    # Add this new section to filter characters
    allowed_chars = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ⏎'''
    txt = ''.join(c for c in txt if c in allowed_chars)
    
    return txt



def prepare_enwiki_data(clean=False, tokenizer=None, checkpoint_dir="checkpoint/play"):
    """
    Prepare the enwiki dataset for language modeling.
    Args:
        clean (bool): Whether to use cleaned version of the dataset
        tokenizer: Custom tokenizer object. If None, uses character-level tokenization
    Returns:
        dict: Meta information including vocabulary size and encoders/decoders
    """
    data_dir = os.path.join('data', 'enwiki')
    input_file_path = os.path.join(data_dir, 'enwik8')
    if not os.path.exists(input_file_path):
        os.system("bash data/enwiki/download_data.sh")

    if clean:
        clean_file_path = os.path.join(data_dir, 'enwik8_clean.txt')
        if not os.path.exists(clean_file_path):
            os.system("python data/enwiki/filter_data.py")
    else: 
        clean_file_path = input_file_path
        
    print("Input file path: ", clean_file_path)
    os.makedirs(data_dir, exist_ok=True)
        
    with open(clean_file_path, 'r', encoding='utf-8') as f:
        data = f.read()   

    # Use custom tokenizer if provided, otherwise use char-level tokenization
    if tokenizer is None:
        # Character-level tokenization
        chars = sorted(list(set(data)))
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        tokenizer = ETokenizer(char_vocab=itos)
           
    vocab_size = tokenizer.vocab_size

    # create the train, validation and test splits
    n = len(data)
    if clean:
        train_data = data[:50_000_000]
        val_data = data[50_000_000:52_000_000]
        test_data = data[52_000_000:]
    else:
        train_data = data[:90_000_000]
        val_data = data[90_000_000:95_000_000]
        test_data = data[95_000_000:100_000_000]

    # encode all splits
    print(f"Encoding data with tokenizer ...")
    train_ids = tokenizer.encode_with_chunking(train_data)
    val_ids = tokenizer.encode_with_chunking(val_data)
    test_ids = tokenizer.encode_with_chunking(test_data)
    
    print(f"Total tokens: {n}")
    print(f"Vocab size: {vocab_size}")
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

    # save meta information
    meta = {
        "vocab_size": vocab_size, 
        "tokenizer_path": os.path.join(checkpoint_dir, 'tokenizer.json'),
        "itos": itos
    }
    
    tokenizer.save(os.path.join(checkpoint_dir, 'tokenizer.json'))
        
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
        
    return meta