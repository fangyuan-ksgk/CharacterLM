import os 
from magicab import ETokenizer
from datasets import load_from_disk
from magicab.data import save_sequences_for_memmap
import pickle 

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
    conv_text = tokenizer.prepare_pt_conversation_data(conversation)
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

def process_composio_pt_data(
    datasets_dir,
    save_dir, # places to store processed data (not tokenizer)
    tokenizer_path, # places to store tokenizer
    block_size=512,
    init_vocab=False,
    tokenizer = None, 
    mode="byte",
    num_proc=1,
    batch_size=20,
    max_workers=8,
    val_size=1000,
): 
    if init_vocab: 
        tokenizer = ETokenizer(mode=mode)
    else: 
        assert tokenizer is not None or tokenizer_path is not None, "Tokenizer must be provided if init_vocab is False"
        if not tokenizer: 
            print(f" Loading tokenizer from {tokenizer_path}")
            tokenizer = ETokenizer.load(tokenizer_path + "/tokenizer.json") 
    
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
            dataset = load_from_disk(os.path.join(datasets_dir, dataset_name)) # raw data loader
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = process_dataset(
                dataset=dataset,
                processor_fn=processor_fn,
                tokenizer=tokenizer,
                block_size=block_size,
                num_proc=num_proc,
                train_size=train_size,
                val_size=val_size,
                desc_prefix=dataset_name,
                batch_size=batch_size,
                max_workers=max_workers
            )
            
            all_train_ids.extend(train_dataset["ids"])
            all_val_ids.extend(val_dataset["ids"])
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    # Save combined datasets as .bin files
    os.makedirs(save_dir, exist_ok=True)
    
    train_path = os.path.join(save_dir, "train.bin")
    val_path = os.path.join(save_dir, "val.bin")
    
    print(f"Saving {len(all_train_ids)} train examples to {train_path}")
    save_sequences_for_memmap(all_train_ids, train_path)
    
    print(f"Saving {len(all_val_ids)} val examples to {val_path}")
    save_sequences_for_memmap(all_val_ids, val_path)
    
    if init_vocab: 
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save(tokenizer_path + "/tokenizer.json")  
        
    # save meta file 
    meta_path = os.path.join(save_dir, 'meta.pkl')
    meta = {
        "vocab_size": tokenizer.vocab_size, 
        "tokenizer_path": os.path.join(save_dir, 'tokenizer.json')
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)