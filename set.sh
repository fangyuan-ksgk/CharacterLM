apt update
apt install zip
pip install torch numpy transformers datasets tiktoken wandb tqdm
pip install accelerate
pip install SoMaJo
pip install maturin
cd magicab/rust_tokenizer && maturin build --release
pip install target/wheels/rust_tokenizer-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl --force-reinstall
cd .. && pip install -e .
cd ..
python data/enwiki/prepare_data.py

wandb login
accelerate config
# python train.py config/train_enwiki_char.py
# accelerate launch train.py --config config/enwiki_char.json
