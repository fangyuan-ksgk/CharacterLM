# Compute Matching Experiment (v.s. run5)
orig_run_dir="checkpoint/run5"
run_dir="checkpoint/run5_cm"
mkdir -p $run_dir


num_iterations=10  # Adjust this number as needed
accumulated_iter=5000 # base model gets trained for 5k iter

for iter in $(seq 1 $num_iterations); do

    accumulated_iter=accumulated_iter+5000
    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_run_dir}/tokenizer.json"
    
    # Train and evaluate
    python train.py config/train_enwiki_char.py --out_dir="${curr_dir}" --max_iters=${accumulated_iter}
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter}

done

# decreasing vocabulary training 
num_iterations=20 

for iter in $(seq 1 $num_iterations); do

    accumulated_iter=accumulated_iter+5000
    orig_dir="${orig_run_dir}/decrease_iter${iter}"
    curr_dir="${run_dir}/decrease_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_run_dir}/tokenizer.json"

    # Train and evaluate
    python train.py config/train_enwiki_char.py --out_dir="${curr_dir}" --max_iters=${accumulated_iter}
    python eval.py --model_type="GPT" --out_dir="${run_dir}/decrease_iter${iter}" --run_idx=${iter}
done