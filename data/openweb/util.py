import os 
from magicab import ETokenizer
from datasets import load_from_disk
from magicab.data import save_sequences_for_memmap
import pickle 
import multiprocessing as mp


def process_fineweb(example, tokenizer, batch_size=20, max_workers=8):
    text = example['text']
    def tokenize(text): 
        return tokenizer.encode_with_chunking(text, 
                                              chunk_size=256*8,
                                              batch_size=batch_size, 
                                              max_workers=max_workers, 
                                              mode='multiprocessing')
    return {"ids": tokenize(text)}


def process_fineweb_data(
    datasets_dir,
    save_dir, # places to store processed data (not tokenizer)
    tokenizer_path, # places to store tokenizer
    init_vocab=False,
    tokenizer = None, 
    mode="byte",
    num_proc=1,
    batch_size=20,
    max_workers=8,
    val_size=1000,
    shard_size=1000000,
): 
    
    if init_vocab: 
        tokenizer = ETokenizer(mode=mode)
    else: 
        assert tokenizer is not None or tokenizer_path is not None, "Tokenizer must be provided if init_vocab is False"
        if not tokenizer: 
            print(f" Loading tokenizer from {tokenizer_path}")
            if os.path.exists(tokenizer_path + "/tokenizer.json"):
                tokenizer = ETokenizer.load(tokenizer_path + "/tokenizer.json") 
            else: 
                tokenizer = ETokenizer.load(tokenizer_path)


    import numpy as np 

    # magicab object already supports multi-processing ...   
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    fw = load_from_disk(os.path.join(datasets_dir, "fineweb-10B"))

    # multi-processing based tokenization 
    tokens = process_fineweb(fw, tokenizer, batch_size=batch_size, max_workers=max_workers)['ids']

    # data sharding for storage 
    from tqdm import tqdm
    shard_index = 0 

    # is there enough space in the current shard for the new tokens?
    if token_count + len(tokens) < shard_size:
        # simply append tokens to current shard
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        # update progress bar | initialize & update or just update
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))

    else:
        # write to the current shard and start a new one
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(save_dir, f"fineweb_{split}_{shard_index:06d}.bin")
        # split the document into whatever fits in this shard; the remainder goes to next one
        remainder = shard_size - token_count
        progress_bar.update(remainder) # force complete the progress bar to 100% when writing data shard
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
        save_sequences_for_memmap(all_tokens_np, filename)
        shard_index += 1
        progress_bar = None
        # populate the next shard with the leftovers of the current doc
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder
   
    
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