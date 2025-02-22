run_dir="checkpoint/run5"
mkdir -p $run_dir
python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/base"
python train.py config/train_enwiki_char.py --out_dir="$run_dir/base"
python eval.py --model_type="GPT" --out_dir="$run_dir/base" --run_idx=0

num_iterations=5  # Adjust this number as needed
for iter in $(seq 1 $num_iterations); do
    # prev_iter=$((iter - 1))
    # prev_dir="$run_dir/$([ $prev_iter -eq 0 ] && echo 'base' || echo "increase_iter${prev_iter}")"
    
    # # Update vocabulary | this is buggy, failed vocabulary increase here (!) - TBD - debug this
    # python update.py --out_dir="$prev_dir" --new_dir="$run_dir/increase_iter${iter}_raw" \
    #     --thres=0.3 --max_size_change=3000
    
    # # Create new directory and copy files
    # mkdir -p "$run_dir/increase_iter${iter}"
    # cp -r "$run_dir/increase_iter${iter}_raw"/* "$run_dir/increase_iter${iter}"
    
    # # Train and evaluate
    # python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/increase_iter${iter}"
    python eval.py --model_type="GPT" --out_dir="$run_dir/increase_iter${iter}" --run_idx=$iter
done

# decreasing vocabulary training 

# load checkpoint -- that's the key here 
num_iterations=10 

# Get vocabulary sizes and convert from Python list to bash array
vocab_sizes_str=$(python get_vocab_size.py --checkpoint_dir="$run_dir/increase_iter5" --num_iterations=$num_iterations --base_vocab_size=92)
# Remove brackets and split into array
vocab_sizes=($(echo $vocab_sizes_str | tr -d '[]' | tr ',' ' '))

for iter in $(seq 1 $num_iterations); do
    # prev_iter=$((iter - 1))
    # prev_dir="$run_dir/$([ $prev_iter -eq 0 ] && echo 'increase_iter5' || echo "decrease_iter${prev_iter}")"
    
    # # Update vocabulary
    # python update.py --out_dir="$prev_dir" --new_dir="$run_dir/decrease_iter${iter}_raw" \
    #     --truncate_vocab=True --target_vocab_size=${vocab_sizes[$((iter-1))]}
    
    # # Create new directory and copy files
    # mkdir -p "$run_dir/decrease_iter${iter}"
    # cp -r "$run_dir/decrease_iter${iter}_raw"/* "$run_dir/decrease_iter${iter}"

    # # Train and evaluate
    # python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/decrease_iter${iter}"
    python eval.py --model_type="GPT" --out_dir="$run_dir/decrease_iter${iter}" --run_idx=$iter
done
