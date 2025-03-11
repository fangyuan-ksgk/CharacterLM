# run_dir="checkpoint/base"
# python data/enwiki/prepare_data.py --clean --out_dir="$run_dir"
# python train.py config/train_enwiki_char.py --out_dir="$run_dir"
# python eval.py --model_type="GPT" --out_dir="$run_dir" --run_idx=0


run_name="base_byte"
run_dir="checkpoint/$run_name"
dataset_name="composio"

# Create directory if it doesn't exist
mkdir -p "$run_dir"

python data/$dataset_name/process_pt_data.py --datasets_dir="data/$dataset_name/datasets"\
                                        --save_dir="data/$dataset_name/$run_name"\
                                        --tokenizer_name_or_path="$run_dir"\ # no .json really? 
                                        --block_size=512\
                                        --mode="byte"\
                                        --init_vocab=True\
                                        --batch_size=20\
                                        --max_workers=8

python train.py config/train_smol_byte.py --dataset="$dataset_name" --data_subfolder="$run_name" --out_dir="$run_dir"
python eval.py --model_type="GPT" --dataset="$dataset_name" --data_subfolder="$run_name" --out_dir="$run_dir" --run_idx=0