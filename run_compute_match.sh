# Compute Matching Experiment (v.s. run5)
orig_run_dir="checkpoint/gpt_small"
run_dir="checkpoint/gpt_small_cm"
data_subfolder="gpt_small"
mkdir -p $run_dir

num_iterations=8  # Adjust this number as needed
accumulated_iter=5000 # base model gets trained for 5k iter

for iter in $(seq 1 $num_iterations); do

    accumulated_iter=$((accumulated_iter+5000))
    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
    # Train and evaluate
    python train.py config/train_enwiki_gpt_small.py --init_from="retrain" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --max_iters=${accumulated_iter} --data_subfolder="${data_subfolder}"
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter}

done

orig_run_dir="checkpoint/gpt_medium"
run_dir="checkpoint/gpt_medium_cm"
data_subfolder="gpt_medium"
mkdir -p $run_dir

num_iterations=8  # Adjust this number as needed
accumulated_iter=5000 # base model gets trained for 5k iter

for iter in $(seq 1 $num_iterations); do

    accumulated_iter=$((accumulated_iter+5000))
    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
    # Train and evaluate
    python train.py config/train_enwiki_gpt_medium.py --init_from="retrain" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --max_iters=${accumulated_iter} --data_subfolder="${data_subfolder}"
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter}

done