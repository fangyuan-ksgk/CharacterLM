import os
import boto3
import gzip
from datasets import load_dataset, Dataset
from botocore.exceptions import ClientError
from argparse import ArgumentParser
import multiprocessing
import platform

s3 = boto3.client("s3")
bucket_name = "softwareheritage"


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


# Download content from blob_id in Python Education dataset
def download_python_edu_contents(blob_id):
    # Initialize resources inside function instead of globally
    s3 = boto3.client("s3")
    bucket_name = "softwareheritage"
    
    key = f"content/{blob_id}"
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=obj["Body"]) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
        return {"text": content, "download_success": True}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"File not found: {key}")
            return {"text": "", "download_success": False}
        else:
            raise


# Download Python Education dataset
def download_python_edu(save_dir, cache_dir, num_proc, max_samples=None):
    try:
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "python-edu",
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )
        
        # Process and filter in memory with streaming data
        processed_examples = []
        for example in dataset:
            result = download_python_edu_contents(example["blob_id"])
            if result["download_success"]:
                example.update(result)
                processed_examples.append(example)
            if max_samples is not None and max_samples > 0 and len(processed_examples) >= max_samples:
                break
                
        # Convert to standard dataset
        dataset = Dataset.from_list(processed_examples)
        
        print(dataset)
        dataset.save_to_disk(save_dir + "/python-edu")
    except Exception as e:
        print(f"python edu download error: {e}")


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
    if args.is_parallel is True:
        processes = []
        download_funcs = [
            download_fineweb_edu,
            download_cosmopedia_v2,
            download_python_edu,
            download_fine_math,
        ]
        for func in download_funcs:
            p = multiprocessing.Process(
                target=func, args=(args.save_dir, args.cache_dir, args.num_proc, args.max_samples)
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    else:
        download_fineweb_edu(args.save_dir, args.cache_dir, args.num_proc, args.max_samples)
        download_cosmopedia_v2(args.save_dir, args.cache_dir, args.num_proc, args.max_samples)
        download_python_edu(args.save_dir, args.cache_dir, args.num_proc, args.max_samples)
        download_fine_math(args.save_dir, args.cache_dir, args.num_proc, args.max_samples)


if __name__ == "__main__":
    # Set multiprocessing start method based on platform
    if platform.system() != "Windows":
        # Use 'spawn' on Linux/Unix for better compatibility
        multiprocessing.set_start_method('spawn', force=True)
    else:
        # Windows needs freeze_support
        multiprocessing.freeze_support()
        
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./datasets")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--is_parallel", action="store_true", help="Whether to download all datasets in parallel.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to keep per dataset. None means keep all.")
    args = parser.parse_args()
    print(args)
    main(args)