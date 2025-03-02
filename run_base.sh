run_dir="checkpoint/base"
python data/enwiki/prepare_data.py --clean --out_dir="$run_dir"
python train.py config/train_enwiki_char.py --out_dir="$run_dir" --max_iters=5000
python eval.py --model_type="GPT" --out_dir="$run_dir" --run_idx=0