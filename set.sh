apt update
apt install zip
pip install torch numpy transformers datasets tiktoken wandb tqdm
pip install accelerate
pip install SoMaJo
pip install maturin
cd magicab && pip install -e .
cd ..
python data/enwiki/prepare_data.py

wandb login
accelerate config
# python train.py config/train_enwiki_char.py
# accelerate launch train.py --config config/enwiki_char.json
