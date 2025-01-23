api install zip
pip install torch numpy transformers datasets tiktoken wandb tqdm
pip install accelerate
python data/enwiki/prepare_data.py

wandb login
accelerate config
# python train.py config/train_enwiki_char.py
