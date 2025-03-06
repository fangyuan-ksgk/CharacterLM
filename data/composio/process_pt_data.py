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


def main(args):

    # Calculate the size of fineweb-edu, cosmopedia-v2, python-edu, fine-math
    fineweb_edu_ratio, cosmopedia_v2_ratio, python_edu_ratio, fine_math_ratio = 0.7, 0.2, 0.05, 0.05

    fineweb_edu_train_size = int(args.train_examples * fineweb_edu_ratio)
    cosmopedia_v2_train_size = int(args.train_examples * cosmopedia_v2_ratio)
    python_edu_train_size = int(args.train_examples * python_edu_ratio)
    fine_math_train_size = int(args.train_examples * fine_math_ratio)

    fineweb_edu_test_size = int(args.test_examples * fineweb_edu_ratio)
    cosmopedia_v2_test_size = int(args.test_examples * cosmopedia_v2_ratio)
    python_edu_test_size = int(args.test_examples * python_edu_ratio)
    fine_math_test_size = int(args.test_examples * fine_math_ratio)


    # Load Etokenizer
    tokenizer = Etokenizer.load(args.tokenizer_name_or_path)

    # Initialize lists to collect all processed examples
    all_train_ids = []
    all_val_ids = []
    
    # Process fineweb-edu
    dataset = load_from_disk(args.datasets_dir + '/fineweb-edu')
    column_names = dataset.column_names
    
    # Split into train and val
    train_size = fineweb_edu_train_size
    val_size = fineweb_edu_test_size
    
    train_dataset = dataset.select(range(train_size)).map(
        process_fineweb_edu, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing fineweb-edu train"
    )
    val_dataset = dataset.select(range(train_size, train_size + val_size)).map(
        process_fineweb_edu, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing fineweb-edu val"
    )
    
    all_train_ids.extend(train_dataset["ids"])
    all_val_ids.extend(val_dataset["ids"])
    print(f"Processed fineweb-edu: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Process Cosmopedia-v2
    dataset = load_from_disk(args.datasets_dir + '/cosmopedia-v2')
    column_names = dataset.column_names
    
    # Split into train and val
    train_size = cosmopedia_v2_train_size
    val_size = cosmopedia_v2_test_size
    
    train_dataset = dataset.select(range(train_size)).map(
        process_cosmopedia, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Processing cosmopedia-v2 train"
    )
    val_dataset = dataset.select(range(train_size, train_size + val_size)).map(
        process_cosmopedia, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        desc="Processing cosmopedia-v2 val"
    )
    
    all_train_ids.extend(train_dataset["ids"])
    all_val_ids.extend(val_dataset["ids"])
    print(f"Processed cosmopedia-v2: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Process Python Education
    dataset = load_from_disk(args.datasets_dir + '/python-edu')
    column_names = dataset.column_names
    
    # Split into train and val
    train_size = python_edu_train_size
    val_size = python_edu_test_size
    
    train_dataset = dataset.select(range(train_size)).map(
        process_python_edu, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing python-edu train"
    )
    val_dataset = dataset.select(range(train_size, train_size + val_size)).map(
        process_python_edu, 
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing python-edu val"
    )
    
    all_train_ids.extend(train_dataset["ids"])
    all_val_ids.extend(val_dataset["ids"])
    print(f"Processed python-edu: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Process FineMath
    dataset = load_from_disk(args.datasets_dir + '/finemath')
    column_names = dataset.column_names
    
    # Split into train and val
    train_size = fine_math_train_size
    val_size = fine_math_test_size
    
    train_dataset = dataset.select(range(train_size)).map(
        process_fine_math,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing finemath train"
    )
    val_dataset = dataset.select(range(train_size, train_size + val_size)).map(
        process_fine_math,
        fn_kwargs={'tokenizer': tokenizer, 'max_length': args.max_length},
        num_proc=args.num_proc,
        remove_columns=column_names,
        batched=True,
        desc="Processing finemath val"
    )
    
    all_train_ids.extend(train_dataset["ids"])
    all_val_ids.extend(val_dataset["ids"])
    print(f"Processed finemath: {len(train_dataset)} train, {len(val_dataset)} val")

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