python data/composio/download_pt_data.py --save_dir="data/composio/datasets" --cache_dir="data/composio/cache" --total_chars=2000000000

python data/composio/process_pt_data.py --datasets_dir="data/composio/datasets" --save_dir="data/composio/processed" --tokenizer_name_or_path="checkpoint/base/tokenizer.json" --block_size=512