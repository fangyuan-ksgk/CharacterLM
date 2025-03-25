import argparse
import os
from magicab import ETokenizer
from datasets import load_from_disk, load_dataset
import numpy as np 
import multiprocessing as mp 
from tqdm import tqdm 
import pickle 

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    - Limits token count and vocab size
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def download_fineweb_edu(save_dir, max_samples=None):
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb", 
            name="sample-10BT", 
            split="train",
            streaming=False,
        )
        if max_samples is not None and max_samples > 0:
            dataset = dataset.take(max_samples)
        
        print(dataset)
        dataset.save_to_disk(save_dir + "/fineweb-10B")
    except Exception as e:
        print(f"fineweb download error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--save_dir", type=str, default="./datasets/base")
    parser.add_argument("--tokenizer_path", type=str, default="./datasets/tokenizer")
    parser.add_argument("--init_vocab", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="byte")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--shard_size", type=int, default=10**8)
    parser.add_argument("--max_iter", type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tokenizer_path, exist_ok=True)

    # download dataset if it doesn't exist
    if not os.path.exists(os.path.join(args.datasets_dir, "fineweb-10B")):
        download_fineweb_edu(args.datasets_dir)

    # load tokenizer  
    if args.init_vocab: 
        tokenizer = ETokenizer(mode=args.mode)
    else: 
        assert args.tokenizer_path is not None, "Tokenizer must be provided if init_vocab is False"
        if os.path.exists(args.tokenizer_path + "/tokenizer.json"):
            tokenizer = ETokenizer.load(args.tokenizer_path + "/tokenizer.json") 
        else: 
            tokenizer = ETokenizer.load(args.tokenizer_path)

    # load dataset
    fw = load_from_disk(os.path.join(args.datasets_dir, "fineweb-10B"))

    def tokenize(doc): 
        tokens =[] 
        tokens.extend(tokenizer.encode_with_chunking(doc["text"]))
        if isinstance(tokens[0], list): 
            tokens_np = np.concatenate(tokens)
        else:
            tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    
    # temp: cap iteration
    MAX_ITER = args.max_iter
    
    # Tokenize dataset with multiprocessing
    nprocs = max(1, os.cpu_count() - 2)
    with mp.Pool(nprocs) as pool: 
        shard_index = 0
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0 
        progress_bar = None 
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < args.shard_size: 
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None: 
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else: 
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(args.save_dir, f"fineweb_{split}_{shard_index:06d}.bin")
                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                # slice tokens to fill-in remainder positions
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
    
                shard_index += 1
                progress_bar = None 
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder 

            if MAX_ITER and shard_index >= MAX_ITER: 
                break
    
        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(args.save_dir, f"fineweb_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count]) 
   
    
    if args.init_vocab: 
        os.makedirs(args.tokenizer_path, exist_ok=True)
        tokenizer.save(args.tokenizer_path + "/tokenizer.json")  
        
    # save meta file 
    meta_path = os.path.join(args.save_dir, 'meta.pkl')
    meta = {
        "vocab_size": tokenizer.vocab_size, 
        "tokenizer_path": os.path.join(args.save_dir, 'tokenizer.json')
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)