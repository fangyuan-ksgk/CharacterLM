# util function to get vocab size
from magicab import ETokenizer 
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing the tokenizer.json file')
    parser.add_argument("--num_iterations", type=int, required=True,
                        help="Number of iterations")
    parser.add_argument("--base_vocab_size", type=int, required=True,
                        help="Minimum vocabulary size")
    args = parser.parse_args()

    # Get the tokenizer from the checkpoint directory
    tokenizer = ETokenizer.load(args.checkpoint_dir + "/tokenizer.json")

    # Get the vocab size from the tokenizer
    vocab_size = tokenizer.vocab_size
    
    # create log-linear interpolation between vocab_size and base_vocab_size
    vocab_sizes = np.logspace(np.log10(vocab_size), np.log10(args.base_vocab_size), args.num_iterations + 1)
    vocab_sizes = np.round(vocab_sizes).astype(int).tolist()
    vocab_sizes = vocab_sizes[1:]
    
    print(vocab_sizes)

if __name__ == "__main__":
    main()