run_dir="checkpoint/gpt_small"
config_file="config/train_enwiki_gpt_small.py"
data_subfolder="gpt_small"

mkdir -p $run_dir
python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/base" --data_subfolder="$data_subfolder"
python train.py $config_file --out_dir="$run_dir/base" --data_subfolder="$data_subfolder"
python eval.py --model_type="GPT" --out_dir="$run_dir/base" --run_idx=0 --data_subfolder="$data_subfolder"

num_iterations=8  
for iter in $(seq 1 $num_iterations); do
    prev_iter=$((iter - 1))
    prev_dir="$run_dir/$([ $prev_iter -eq 0 ] && echo 'base' || echo "increase_iter${prev_iter}")"
    
    # Update Vocabulary
    python update.py --out_dir="$prev_dir" --new_dir="$run_dir/increase_iter${iter}_raw" --data_subfolder="$data_subfolder" \
        --thres=0.3 --max_size_change=3000
    
    # Create new directory and copy files
    mkdir -p "$run_dir/increase_iter${iter}"
    cp -r "$run_dir/increase_iter${iter}_raw"/* "$run_dir/increase_iter${iter}"
    
    # Train and evaluate
    python train.py $config_file --init_from="resume" --out_dir="$run_dir/increase_iter${iter}" --data_subfolder="$data_subfolder"
    python eval.py --model_type="GPT" --out_dir="$run_dir/increase_iter${iter}" --run_idx=$iter --data_subfolder="$data_subfolder"
done


run_dir="checkpoint/gpt_medium"
config_file="config/train_enwiki_gpt_medium.py"
data_subfolder="gpt_medium"

mkdir -p $run_dir
python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/base" --data_subfolder="$data_subfolder"
python train.py $config_file --out_dir="$run_dir/base" --data_subfolder="$data_subfolder"
python eval.py --model_type="GPT" --out_dir="$run_dir/base" --run_idx=0 --data_subfolder="$data_subfolder"

num_iterations=8  
for iter in $(seq 1 $num_iterations); do
    prev_iter=$((iter - 1))
    prev_dir="$run_dir/$([ $prev_iter -eq 0 ] && echo 'base' || echo "increase_iter${prev_iter}")"
    
    # Update Vocabulary
    python update.py --out_dir="$prev_dir" --new_dir="$run_dir/increase_iter${iter}_raw" --data_subfolder="$data_subfolder" \
        --thres=0.3 --max_size_change=3000
    
    # Create new directory and copy files
    mkdir -p "$run_dir/increase_iter${iter}"
    cp -r "$run_dir/increase_iter${iter}_raw"/* "$run_dir/increase_iter${iter}"
    
    # Train and evaluate
    python train.py $config_file --init_from="resume" --out_dir="$run_dir/increase_iter${iter}" --data_subfolder="$data_subfolder"
    python eval.py --model_type="GPT" --out_dir="$run_dir/increase_iter${iter}" --run_idx=$iter --data_subfolder="$data_subfolder"
done


run_dir="checkpoint/gpt_large"
config_file="config/train_enwiki_gpt_large.py"
data_subfolder="gpt_large"

mkdir -p $run_dir
python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/base" --data_subfolder="$data_subfolder"
python train.py $config_file --out_dir="$run_dir/base" --data_subfolder="$data_subfolder"
python eval.py --model_type="GPT" --out_dir="$run_dir/base" --run_idx=0 --data_subfolder="$data_subfolder"

num_iterations=8  
for iter in $(seq 1 $num_iterations); do
    prev_iter=$((iter - 1))
    prev_dir="$run_dir/$([ $prev_iter -eq 0 ] && echo 'base' || echo "increase_iter${prev_iter}")"
    
    # Update Vocabulary
    python update.py --out_dir="$prev_dir" --new_dir="$run_dir/increase_iter${iter}_raw" --data_subfolder="$data_subfolder" \
        --thres=0.3 --max_size_change=3000
    
    # Create new directory and copy files
    mkdir -p "$run_dir/increase_iter${iter}"
    cp -r "$run_dir/increase_iter${iter}_raw"/* "$run_dir/increase_iter${iter}"
    
    # Train and evaluate
    python train.py $config_file --init_from="resume" --out_dir="$run_dir/increase_iter${iter}" --data_subfolder="$data_subfolder"
    python eval.py --model_type="GPT" --out_dir="$run_dir/increase_iter${iter}" --run_idx=$iter --data_subfolder="$data_subfolder"
done