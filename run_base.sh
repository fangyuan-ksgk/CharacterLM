# run_dir="checkpoint/base"
# python data/enwiki/prepare_data.py --clean --out_dir="$run_dir"
# python train.py config/train_enwiki_char.py --out_dir="$run_dir"
# python eval.py --model_type="GPT" --out_dir="$run_dir" --run_idx=0


run_name = "base_byte"
run_dir="checkpoint/base_byte"
python data/composio/process_pt_data.py --datasets_dir="data/composio/datasets"\
                                        --save_dir="data/composio/$run_name"\
                                        --tokenizer_name_or_path="checkpoint/$run_name"\
                                        --block_size=512\
                                        --mode="byte"\
                                        --init_vocab=True\
                                        --batch_size=20\
                                        --max_workers=8
python train.py config/train_smol_byte.py --data_subfolder="$run_name" --out_dir="checkpoint/smol_byte/base_byte"
python eval.py --model_type="GPT" --data_subfolder="$run_name" --out_dir="checkpoint/smol_byte/base_byte" --run_idx=0