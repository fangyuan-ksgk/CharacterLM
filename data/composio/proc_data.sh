python data/composio/download_pt_data.py --save_dir="data/composio/datasets" --cache_dir="data/composio/cache" --total_chars=2000000000

python data/composio/process_pt_data.py --datasets_dir="data/composio/datasets"\
                                        --save_dir="data/composio/processed1"\
                                        --tokenizer_name_or_path="checkpoint/base_byte"\
                                        --block_size=512\
                                        --mode="byte"\
                                        --init_vocab=True\
                                        --batch_size=1\
                                        --max_workers=1