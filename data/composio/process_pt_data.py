from argparse import ArgumentParser
from util import process_composio_pt_data

def main(args):
    
    print("Loaded tokenizer from:  ***", args.tokenizer_path,"***")
    args.tokenizer_path = args.tokenizer_path.strip()
    args.datasets_dir = args.datasets_dir.strip()
    args.save_dir = args.save_dir.strip()
    
    process_composio_pt_data(
        datasets_dir=args.datasets_dir,
        save_dir=args.save_dir,
        tokenizer_path=args.tokenizer_path,
        mode=args.mode,
        tokenizer=None,
        init_vocab=args.init_vocab,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        val_size=args.val_size
    )

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--datasets_dir", type=str, default="./datasets")
    argparser.add_argument("--save_dir", type=str, default="./datasets")
    argparser.add_argument("--tokenizer_path", type=str, default="checkpoint/base")
    argparser.add_argument("--mode", type=str, default="char")
    argparser.add_argument("--init_vocab", type=bool, default=False)
    argparser.add_argument("--num_proc", type=int, default=1)
    argparser.add_argument("--batch_size", type=int, default=20)
    argparser.add_argument("--max_workers", type=int, default=8)
    argparser.add_argument("--val_size", type=int, default=1000)
    args = argparser.parse_args()

    main(args)