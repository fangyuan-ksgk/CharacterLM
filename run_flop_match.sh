# Compute Matching Experiment (v.s. run5)
# orig_run_dir="checkpoint/gpt_tiny"
# run_dir="checkpoint/gpt_tiny_cm"
# config_file="config/train_enwiki_gpt_tiny.py"
# data_subfolder="gpt_tiny_cm"
# mkdir -p $run_dir

# num_iterations=8  # Adjust this number as needed

# for iter in $(seq 1 $num_iterations); do

#     orig_dir="${orig_run_dir}/increase_iter${iter}"
#     curr_dir="${run_dir}/increase_iter${iter}"
    
#     # prepare encoding data 
#     python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
#     # Train and evaluate
#     python train.py "${config_file}" --init_from="retrain" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --data_subfolder="${data_subfolder}"
#     python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter} --data_subfolder="${data_subfolder}"

# done


orig_run_dir="checkpoint/gpt_tiny2"
run_dir="checkpoint/gpt_tiny2_cm"
config_file="config/train_enwiki_gpt_tiny2.py"
data_subfolder="gpt_tiny2_cm"
mkdir -p $run_dir

num_iterations=8  # Adjust this number as needed

for iter in $(seq 1 $num_iterations); do

    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
    # Train and evaluate
    python train.py "${config_file}" --init_from="retrain" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --data_subfolder="${data_subfolder}"
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter} --data_subfolder="${data_subfolder}"

done


orig_run_dir="checkpoint/gpt_tiny3"
run_dir="checkpoint/gpt_tiny3_cm"
config_file="config/train_enwiki_gpt_tiny3.py"
data_subfolder="gpt_tiny3_cm"
mkdir -p $run_dir

num_iterations=8  # Adjust this number as needed

for iter in $(seq 1 $num_iterations); do

    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
    # Train and evaluate
    python train.py "${config_file}" --init_from="retrain" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --data_subfolder="${data_subfolder}"
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter} --data_subfolder="${data_subfolder}"

done


orig_run_dir="checkpoint/gpt_small"
run_dir="checkpoint/gpt_small_cm"
config_file="config/train_enwiki_gpt_small.py"
data_subfolder="gpt_small_cm"
mkdir -p $run_dir

num_iterations=8  # Adjust this number as needed

for iter in $(seq 1 $num_iterations); do

    orig_dir="${orig_run_dir}/increase_iter${iter}"
    curr_dir="${run_dir}/increase_iter${iter}"
    
    # prepare encoding data 
    python data/enwiki/prepare_data.py --clean --out_dir="${curr_dir}" --tokenizer_path="${orig_dir}/tokenizer.json" --data_subfolder="${data_subfolder}"
    
    # Train and evaluate
    python train.py config/train_enwiki_gpt_small.py --init_from="retrain" --load_dir="${orig_dir}" --out_dir="${curr_dir}" --data_subfolder="${data_subfolder}"
    python eval.py --model_type="GPT" --out_dir="${run_dir}/increase_iter${iter}" --run_idx=${iter} --data_subfolder="${data_subfolder}"

done