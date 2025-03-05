# run supervised finetuning on pre-trained model checkpoint 

base_model_dir="checkpoint/gpt_tiny/increase_iter6"

# prepare alpaca data
python data/alpaca/prepare_data.py --model_dir="${base_model_dir}"

# run supervised finetuning
python train_sft.py --model_dir="${base_model_dir}" --dataset="alpaca" --max_iters=5000