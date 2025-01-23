api install zip
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/enwiki/prepare_data.py

wandb login
# python train.py config/train_enwiki_char.py
