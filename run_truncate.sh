# Curriculum base vocabulary truncation 

run_dir="checkpoint/run2"

python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/base" --tokenizer_path="experiment/run2/big_vocab/tokenizer.json"

python train.py config/train_enwiki_char.py --out_dir="$run_dir/base"
python eval.py --model_type="GPT" --out_dir="$run_dir/base" --run_idx=0
python update.py --out_dir="$run_dir/base" --new_dir="$run_dir/iter1_raw" --truncate_vocab=True --target_vocab_size=92
mkdir -p $run_dir/iter1
cp -r $run_dir/iter1_raw/* $run_dir/iter1

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/iter1"
python eval.py --model_type="GPT" --out_dir="$run_dir/iter1" --run_idx=1


python train.py config/train_enwiki_char.py --out_dir="checkpoint/run2/base" --wandb_log=False
