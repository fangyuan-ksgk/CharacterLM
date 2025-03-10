from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from argparse import ArgumentParser
import numpy as np
from magicab import ETokenizer, save_sequences_for_memmap
import torch
import os


def process_fineweb_edu(example, tokenizer, block_size=512, batch_size=20, max_workers=8):
    text = example['text']
    ids = tokenizer.encode_with_chunking(text, batch_size=batch_size, max_workers=max_workers, mode='multiprocessing')
    return {"ids": ids}

def process_cosmopedia(example, tokenizer, block_size=512, batch_size=20, max_workers=8):
    prompt = example['prompt']
    text = example['text']
    conversation = [
        {"user": prompt},
        {"assistant": text},
    ]
    conv_text = tokenizer.prepare_sft_data(conversation, block_size=block_size)
    ids = tokenizer.encode_with_chunking(conv_text, batch_size=batch_size, max_workers=max_workers, mode='multiprocessing')
    return {"ids": ids}

def process_python_edu(example, tokenizer, block_size=512, batch_size=20, max_workers=8):
    text = example['text']
    ids = tokenizer.encode_with_chunking(text, batch_size=batch_size, max_workers=max_workers, mode='multiprocessing')
    return {"ids": ids}

def process_fine_math(example, tokenizer, block_size=512, batch_size=20, max_workers=8):
    text = example['text']
    ids = tokenizer.encode_with_chunking(text, batch_size=batch_size, max_workers=max_workers, mode='multiprocessing')
    return {"ids": ids}

def process_dataset(dataset, processor_fn, tokenizer, block_size, num_proc, train_size, val_size, desc_prefix, batch_size=20, max_workers=8):
    """Generic function to process any dataset with the appropriate processor function"""
    column_names = dataset.column_names
    
    train_dataset = dataset.select(range(train_size)).map(
        processor_fn, 
        fn_kwargs={'tokenizer': tokenizer, 'block_size': block_size, 'batch_size': batch_size, 'max_workers': max_workers},
        num_proc=num_proc,
        remove_columns=column_names,
        batched=True,
        desc=f"Processing {desc_prefix} train"
    )
    
    val_dataset = dataset.select(range(train_size, train_size + val_size)).map(
        processor_fn, 
        fn_kwargs={'tokenizer': tokenizer, 'block_size': block_size, 'batch_size': batch_size, 'max_workers': max_workers},
        num_proc=num_proc,
        remove_columns=column_names,
        batched=True,
        desc=f"Processing {desc_prefix} val"
    )
    
    print(f"Processed {desc_prefix}: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset
            
            
def main(args):
    # Load Etokenizer
    if args.init_vocab:
        tokenizer = ETokenizer(mode=args.mode)
    else:
        tokenizer = ETokenizer.load(args.tokenizer_name_or_path)

    # Initialize lists to collect all processed examples
    all_train_ids = []
    all_val_ids = []
    
    # Define dataset configs - each entry contains:
    # (dataset_name, processor_function, train_size, val_size)
    dataset_configs = [
        ('fineweb-edu', process_fineweb_edu),
        ('cosmopedia-v2', process_cosmopedia),
        ('finemath', process_fine_math),
    ]

    # Process all datasets
    for dataset_name, processor_fn in dataset_configs:
        try:
            dataset = load_from_disk(os.path.join(args.datasets_dir, dataset_name))
            val_size = 1000
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = process_dataset(
                dataset=dataset,
                processor_fn=processor_fn,
                tokenizer=tokenizer,
                block_size=args.block_size,
                num_proc=args.num_proc,
                train_size=train_size,
                val_size=val_size,
                desc_prefix=dataset_name,
                batch_size=args.batch_size
            )
            
            all_train_ids.extend(train_dataset["ids"])
            all_val_ids.extend(val_dataset["ids"])
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    # Save combined datasets as .bin files
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_path = os.path.join(args.save_dir, "train.bin")
    val_path = os.path.join(args.save_dir, "val.bin")
    
    print(f"Saving {len(all_train_ids)} train examples to {train_path}")
    save_sequences_for_memmap(all_train_ids, train_path)
    
    print(f"Saving {len(all_val_ids)} val examples to {val_path}")
    save_sequences_for_memmap(all_val_ids, val_path)
    
    if args.init_vocab:
        os.makedirs(os.path.dirname(args.tokenizer_name_or_path), exist_ok=True)
        tokenizer.save(args.tokenizer_name_or_path + "/tokenizer.json")

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_name_or_path", type=str, default="checkpoint/base/tokenizer.json")
    argparser.add_argument("--mode", type=str, default="char")
    argparser.add_argument("--init_vocab", type=bool, default=False)
    argparser.add_argument("--block_size", type=int, default=512)
    argparser.add_argument("--num_proc", type=int, default=1)
    argparser.add_argument("--batch_size", type=int, default=20)
    argparser.add_argument("--max_workers", type=int, default=8)
    args = argparser.parse_args()

    main(args)