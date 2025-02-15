# Curriculum base vocabulary truncation 

run_dir="checkpoint/run2"

python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/base" --tokenizer_path="experiment/run2/big_vocab/tokenizer.json"

python train.py config/train_enwiki_char.py --out_dir="$run_dir/base"
python eval.py --model_type="GPT" --out_dir="$run_dir/base" --run_idx=0

python eval.py --model_type="GPT" --out_dir="checkpoint/run2/base" --run_idx=0


vocab_sizes=(14000 7000 3500 1750 875 440 220 110 92)

# First iteration remains the same
python update.py --out_dir="$run_dir/base" --new_dir="$run_dir/iter1_raw" --truncate_vocab=True --target_vocab_size=${vocab_sizes[0]}
mkdir -p $run_dir/iter1
cp -r $run_dir/iter1_raw/* $run_dir/iter1

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/iter1"
python eval.py --model_type="GPT" --out_dir="$run_dir/iter1" --run_idx=1

# Additional iterations with gradually decreasing vocabulary sizes
for i in {2..9}; do
    prev_iter=$((i-1))
    
    # Update vocabulary and create new checkpoint
    python update.py --out_dir="$run_dir/iter${prev_iter}" --new_dir="$run_dir/iter${i}_raw" --truncate_vocab=True --target_vocab_size=${vocab_sizes[$((i-1))]}
    mkdir -p "$run_dir/iter${i}"
    cp -r "$run_dir/iter${i}_raw"/* "$run_dir/iter${i}"
    
    # Train and evaluate with new vocabulary
    python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/iter${i}"
    python eval.py --model_type="GPT" --out_dir="$run_dir/iter${i}" --run_idx=$i
done



python eval.py --model_type="GPT" --out_dir="checkpoint/run2/base" --run_idx=0
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter1" --run_idx=1
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter2" --run_idx=2
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter3" --run_idx=3
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter4" --run_idx=4
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter5" --run_idx=5
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter6" --run_idx=6
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter7" --run_idx=7
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter8" --run_idx=8
python eval.py --model_type="GPT" --out_dir="checkpoint/run2/iter9" --run_idx=9