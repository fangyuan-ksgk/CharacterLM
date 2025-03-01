run_dir="checkpoint/run4"
python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/max_vocab" --tokenizer_path="checkpoint/run3/increase_iter5/tokenizer.json"
python train.py config/train_enwiki_char.py --out_dir="$run_dir/max_vocab" --max_iters=30000
python eval.py --model_type="GPT" --out_dir="$run_dir/max_vocab" --run_idx=0

python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/min_vocab" --tokenizer_path="checkpoint/run3/base/tokenizer.json"
python train.py config/train_enwiki_char.py --out_dir="$run_dir/min_vocab" --max_iters=30000
python eval.py --model_type="GPT" --out_dir="$run_dir/min_vocab" --run_idx=0




python train_sft.py config/train_sft_gpt.py --load_dir="checkpoint/base" --out_dir="checkpoint/sft_gpt" --max_iters=5000 --device="mps"