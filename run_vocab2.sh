run_dir="checkpoint/gpt_tiny"
dataset_name="composio"
config_file="config/train_smol_byte.py"
data_subfolder="smol_byte"

mkdir -p $run_dir
python data/composio/process_pt_data.py --datasets_dir="data/$dataset_name/datasets"\
                                        --save_dir="data/$dataset_name/$data_subfolder"\
                                        --tokenizer_name_or_path="$run_dir"\ # no .json really? 
                                        --block_size=512\
                                        --mode="byte"\
                                        --init_vocab=True\
                                        --batch_size=20\
                                        --max_workers=8

python train.py $config_file --dataset="$dataset_name" --data_subfolder="$data_subfolder" --out_dir="$run_dir/base"
python eval.py --model_type="GPT" --dataset="$dataset_name" --data_subfolder="$data_subfolder" --out_dir="$run_dir/base" --run_idx=0

# rest to be fixed ...
num_iterations=6
for iter in $(seq 1 $num_iterations); do
    prev_iter=$((iter - 1))
    prev_dir="$run_dir/$([ $prev_iter -eq 0 ] && echo 'base' || echo "increase_iter${prev_iter}")"
    
    # Update Vocabulary
    python update.py --load_dir="$prev_dir" \
                     --new_dir="$run_dir/increase_iter${iter}_raw" \
                     --dataset="$dataset_name" \
                     --data_subfolder="$data_subfolder" \
                     --thres=0.3 \
                     --max_size_change=3000
    
    # Create new directory and copy files
    mkdir -p "$run_dir/increase_iter${iter}"
    cp -r "$run_dir/increase_iter${iter}_raw"/* "$run_dir/increase_iter${iter}"
    
    # Train and evaluate
    python train.py $config_file --init_from="resume" --out_dir="$run_dir/increase_iter${iter}" --dataset="$dataset_name" --data_subfolder="$data_subfolder"
    python eval.py --model_type="GPT" --out_dir="$run_dir/increase_iter${iter}" --run_idx=$iter --dataset="$dataset_name" --data_subfolder="$data_subfolder"
    
done