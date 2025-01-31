apt update
apt install zip
pip install torch numpy transformers datasets tiktoken wandb tqdm
pip install accelerate
pip install SoMaJo
python data/enwiki/prepare_data.py

wandb login
accelerate config
# python train_baseline.py config/train_enwiki_char.py
# accelerate launch train.py --config config/enwiki_char.json