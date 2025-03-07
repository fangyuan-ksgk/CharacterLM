from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from argparse import ArgumentParser
from magicab import Etokenizer
import torch
import os


def process_fineweb_edu(example, tokenizer, max_length=2048):
    text = example['text']
    ids = tokenizer.encode_with_chunking(text)
    return {"ids": ids}

def process_cosmopedia(example, tokenizer, max_length=2048):
    prompt = example['prompt']
    text = example['text']
    conversation = [
        {"user": prompt},
        {"assistant": text},
    ]
    conv_text = tokenizer.prepare_sft_data(conversation)
    ids = tokenizer.encode_with_chunking(conv_text)
    return {"ids": ids}

def process_python_edu(example, tokenizer, max_length=2048):
    text = example['text']
    ids = tokenizer.encode_with_chunking(text)
    return {"ids": ids}

def process_fine_math(example, tokenizer, max_length=2048):
    text = example['text']
    ids = tokenizer.encode_with_chunking(text)
    return {"ids": ids}


def process_dataset(dataset, processor_fn, tokenizer, max_length, num_proc, train_size, val_size, desc_prefix):
    """Generic function to process any dataset with the appropriate processor function"""
    column_names = dataset.column_names
    
    train_dataset = dataset.select(range(train_size)).map(
        processor_fn, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length},
        num_proc=num_proc,
        remove_columns=column_names,
        batched=True,
        desc=f"Processing {desc_prefix} train"
    )
    
    val_dataset = dataset.select(range(train_size, train_size + val_size)).map(
        processor_fn, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length},
        num_proc=num_proc,
        remove_columns=column_names,
        batched=True,
        desc=f"Processing {desc_prefix} val"
    )
    
    print(f"Processed {desc_prefix}: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset

def main(args):
    # Load Etokenizer
    tokenizer = Etokenizer.load(args.tokenizer_name_or_path)

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
    for dataset_name, processor_fn, train_size, val_size in dataset_configs:
        try:
            dataset = load_from_disk(os.path.join(args.datasets_dir, dataset_name))
            val_size = 1000
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = process_dataset(
                dataset=dataset,
                processor_fn=processor_fn,
                tokenizer=tokenizer,
                max_length=args.max_length,
                num_proc=args.num_proc,
                train_size=train_size,
                val_size=val_size,
                desc_prefix=dataset_name
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
    torch.save(all_train_ids, train_path)
    
    print(f"Saving {len(all_val_ids)} val examples to {val_path}")
    torch.save(all_val_ids, val_path)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_name_or_path", type=str, default="SmallDoge/Doge-tokenizer")
    argparser.add_argument("--train_examples", type=int, default=128_000_000)
    argparser.add_argument("--test_examples", type=int, default=1_000)
    argparser.add_argument("--max_length", type=int, default=2048)
    argparser.add_argument("--num_proc", type=int, default=1)
    args = argparser.parse_args()

    main(args)