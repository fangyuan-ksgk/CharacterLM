# Compute Matching Experiment (v.s. run5)
orig_run_dir="checkpoint/run5"
run_dir="checkpoint/run5_cm"
mkdir -p $run_dir


num_iterations=10  # Adjust this number as needed
accumulated_iter=5000 # base model gets trained for 5k iter
increase_vocab_sizes=(92 4359 7941 11382 14819 18276 21700 25103 28527 31909 35321)
for iter in $(seq 1 $num_iterations); do

    accumulated_iter=$((accumulated_iter+5000))
    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json"
    
    # Train and evaluate
    python train.py config/train_enwiki_char.py --out_dir="${curr_dir}" --max_iters=${accumulated_iter} --meta_vocab_size=${increase_vocab_sizes[$((iter-1))]}
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter}

done

# decreasing vocabulary training 
num_iterations=20 
decrease_vocab_sizes=(18276 18276 14468 10744 7979 5926 4401 3268 2427 1803 1339 994 738 548 407 302 225 167 124 92)

for iter in $(seq 1 $num_iterations); do

    accumulated_iter=$((accumulated_iter+5000))
    orig_dir="${orig_run_dir}/decrease_iter${iter}"
    curr_dir="${run_dir}/decrease_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json"

    # Train and evaluate
    python train.py config/train_enwiki_char.py --out_dir="${curr_dir}" --max_iters=${accumulated_iter} --meta_vocab_size=${decrease_vocab_sizes[$((iter-1))]}
    
    python eval.py --model_type="GPT" --out_dir="${run_dir}/decrease_iter${iter}" --run_idx=${iter}
done