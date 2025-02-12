apt update
apt install zip
pip install torch numpy transformers datasets tiktoken wandb tqdm
pip install accelerate
pip install SoMaJo
pip install maturin
# in case rust is not installed already
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install setuptools-rust
source $HOME/.cargo/env
# build rust tokenizer
cd magicab/rust_tokenizer && maturin build --release
pip install target/wheels/rust_tokenizer-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl --force-reinstall
cd .. && pip install -e .
cd ..
python data/enwiki/prepare_data.py --clean

wandb login
accelerate config
# Iteration Scripts | Passed through till the 3rd steps 
python train.py config/train_enwiki_char.py --out_dir="checkpoint/base"
python update.py --out_dir="checkpoint/base" --new_dir="checkpoint/iter1_raw"

mkdir -p checkpoint/iter1
cp -r checkpoint/iter1_raw/* checkpoint/iter1

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="checkpoint/iter1"
python update.py --out_dir="checkpoint/iter1" --new_dir="checkpoint/iter2_raw"
cp -r checkpoint/iter2_raw checkpoint/iter2

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="checkpoint/iter2"
accelerate launch train.py --config config/enwiki_char.json
