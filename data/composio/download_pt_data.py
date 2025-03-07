from datasets import load_dataset, Dataset
from argparse import ArgumentParser
import multiprocessing

# Calculate desired number of samples from each dataset
def calculate_count(target_total_tlen=2 * 1e9):
    sample_count_prior = [5e3, 5e3, 5e3]
    ratio_prior = [0.7, 0.2, 0.1] # following doge-llm (likely following smolLM)
    target_tlen = [target_total_tlen * r for r in ratio_prior]
    current_tlen = [18661951, 24733823, 23052633] # collected from one run 
    sample_count = [int(_target_tlen / _curr_tlen * _curr_sample_count) for _target_tlen, _curr_tlen, _curr_sample_count in zip(target_tlen, current_tlen, sample_count_prior)]
    return sample_count

# Download Fineweb-Edu dataset
def download_fineweb_edu(save_dir, cache_dir, num_proc, max_samples=None):
    try:
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "fineweb-edu-dedup",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )
        if max_samples is not None and max_samples > 0:
            dataset = dataset.take(max_samples)
        
        # Convert to standard dataset for processing
        dataset_list = list(dataset)
        dataset = Dataset.from_list(dataset_list)
        
        print(dataset)
        dataset.save_to_disk(save_dir + "/fineweb-edu")
    except Exception as e:
        print(f"fineweb download error: {e}")


# Download Cosmopedia-v2 dataset
def download_cosmopedia_v2(save_dir, cache_dir, num_proc, max_samples=None):
    try:
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )
        if max_samples is not None and max_samples > 0:
            dataset = dataset.take(max_samples)
            
        # Convert to standard dataset for processing
        dataset_list = list(dataset)
        dataset = Dataset.from_list(dataset_list)
        
        print(dataset)
        dataset.save_to_disk(save_dir + "/cosmopedia-v2")
    except Exception as e:
        print(f"cosmopedia download error: {e}")


# Download FineMath dataset
def download_fine_math(save_dir, cache_dir, num_proc, max_samples=None):
    try:
        dataset = load_dataset(
            "HuggingFaceTB/finemath",
            "finemath-4plus",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )
        if max_samples is not None and max_samples > 0:
            dataset = dataset.take(max_samples)
            
        # Convert to standard dataset for processing
        dataset_list = list(dataset)
        dataset = Dataset.from_list(dataset_list)
        
        print(dataset)
        dataset.save_to_disk(save_dir + "/finemath")
    except Exception as e:
        print(f"fine math download error: {e}")


def main(args):
    max_samples_per_dataset = calculate_count(args.total_chars)
    
    if args.is_parallel is True:
        processes = []
        download_funcs = [
            download_fineweb_edu,
            download_cosmopedia_v2,
            download_fine_math,
        ]
        for i, func in enumerate(download_funcs):
            p = multiprocessing.Process(
                target=func, args=(args.save_dir, args.cache_dir, args.num_proc, max_samples_per_dataset[i])
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    else:
        download_fineweb_edu(args.save_dir, args.cache_dir, args.num_proc, max_samples_per_dataset[0])
        download_cosmopedia_v2(args.save_dir, args.cache_dir, args.num_proc, max_samples_per_dataset[1])
        download_fine_math(args.save_dir, args.cache_dir, args.num_proc, max_samples_per_dataset[2])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./datasets")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--is_parallel", action="store_true", help="Whether to download all datasets in parallel.")
    parser.add_argument("--total_chars", type=int, default=2 * 1e9, help="Target total length of the dataset. Default is 2 billion.")
    args = parser.parse_args()
    print(args)
    main(args)