from util import process_fineweb_data
import argparse


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default="./datasets")
    parser.add_argument("--save_dir", type=str, default="./datasets")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer")
    parser.add_argument("--init_vocab", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="byte")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=1000)

    args = parser.parse_args()
    process_fineweb_data(**vars(args))

if __name__ == "__main__":
    main()