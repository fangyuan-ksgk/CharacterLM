# run_dir="checkpoint/base"
# python data/enwiki/prepare_data.py --clean --out_dir="$run_dir"
# python train.py config/train_enwiki_char.py --out_dir="$run_dir"
# python eval.py --model_type="GPT" --out_dir="$run_dir" --run_idx=0


run_name="base_byte2"
run_dir="checkpoint/$run_name"
# dataset_name="composio"
dataset_name="openweb"

# Create directory if it doesn't exist
mkdir -p "$run_dir"

# encoding pt data 
# python data/$dataset_name/process_pt_data.py --datasets_dir="data/$dataset_name/datasets"\
#                                         --save_dir="data/$dataset_name/$run_name"\
#                                         --tokenizer_path="$run_dir"\
#                                         --mode="byte"\
#                                         --init_vocab=True\
#                                         --max_workers=8\
#                                         --max_shard=20

# pt model with specific config
python train.py config/train_openweb_gpt.py --dataset="$dataset_name"\
                                            --data_subfolder="$run_name"\
                                            --out_dir="$run_dir"\
                                            --max_epochs=1

# evaluate pt model
python eval.py --model_type="GPT"\
               --dataset="$dataset_name"\
               --data_subfolder="$run_name"\
               --out_dir="$run_dir"\
               --run_idx=0