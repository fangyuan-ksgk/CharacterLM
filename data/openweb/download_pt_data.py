from datasets import load_dataset, Dataset
import argparse


def download_fineweb_edu(save_dir, cache_dir, max_samples=None):
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb", 
            name="sample-10BT", 
            split="train",
            streaming=False,
            cache_dir=cache_dir,
        )
        if max_samples is not None and max_samples > 0:
            dataset = dataset.take(max_samples)
        
        print(dataset)
        dataset.save_to_disk(save_dir + "/fineweb-10B")
    except Exception as e:
        print(f"fineweb download error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    download_fineweb_edu(save_dir=args.save_dir, cache_dir=args.cache_dir, max_samples=args.max_samples)