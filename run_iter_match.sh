# Iteration Match Experiment

orig_run_dir="checkpoint/gpt_tiny"
run_dir="checkpoint/gpt_tiny_iter_match"
config_file="config/train_enwiki_gpt_tiny.py"
data_subfolder="gpt_tiny_iter_match"
mkdir -p $run_dir

num_iterations=8  # Adjust this number as needed
accumulated_iter=5000 # count base model's training iterations (5K)
for iter in $(seq 1 $num_iterations); do

    # accumulating training iterations
    accumulated_iter=$((accumulated_iter+5000))
    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
    # Train and evaluate
    python train.py "${config_file}" --init_from="retrain_iter_match" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --max_iters=${accumulated_iter} --data_subfolder="${data_subfolder}"
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter} --data_subfolder="${data_subfolder}"

done


orig_run_dir="checkpoint/gpt_small"
run_dir="checkpoint/gpt_small_iter_match"
config_file="config/train_enwiki_gpt_small.py"
data_subfolder="gpt_small_iter_match"
mkdir -p $run_dir

num_iterations=8  # Adjust this number as needed
accumulated_iter=5000 # count base model's training iterations (5K)
for iter in $(seq 1 $num_iterations); do

    # accumulating training iterations
    accumulated_iter=$((accumulated_iter+5000))
    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
    # Train and evaluate
    python train.py "${config_file}" --init_from="retrain_iter_match" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --max_iters=${accumulated_iter} --data_subfolder="${data_subfolder}"
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter} --data_subfolder="${data_subfolder}"

done