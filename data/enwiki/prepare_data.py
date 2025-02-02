"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import argparse
from util import prepare_enwiki_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare enwiki dataset for language modeling')
    parser.add_argument('--clean', action='store_true', help='Use cleaned version of the dataset')
    args = parser.parse_args()
    
    prepare_enwiki_data(clean=args.clean)