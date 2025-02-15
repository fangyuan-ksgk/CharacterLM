"""
Prepare enwiki dataset for character-level language modeling. 
Save tokenizer along with the data bins. 
"""

import argparse
from util import prepare_enwiki_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare enwiki dataset for language modeling')
    parser.add_argument('--clean', action='store_true', help='Use cleaned version of the dataset')
    parser.add_argument("--out_dir", type=str, default="checkpoint/base")
    parser.add_argument("--tokenizer_path", type=str, default="")

    args = parser.parse_args()
    
    prepare_enwiki_data(clean=args.clean, checkpoint_dir=args.out_dir, tokenizer_path=args.tokenizer_path)