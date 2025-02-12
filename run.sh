python data/enwiki/prepare_data.py --clean

python train.py config/train_enwiki_char.py --out_dir="checkpoint/base"
python update.py --out_dir="checkpoint/base" --new_dir="checkpoint/iter1_raw"

mkdir -p checkpoint/iter1
cp -r checkpoint/iter1_raw/* checkpoint/iter1

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="checkpoint/iter1"
python update.py --out_dir="checkpoint/iter1" --new_dir="checkpoint/iter2_raw"

mkdir -p checkpoint/iter2
cp -r checkpoint/iter2_raw/* checkpoint/iter2

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="checkpoint/iter2"
python update.py --out_dir="checkpoint/iter2" --new_dir="checkpoint/iter3_raw"

mkdir -p checkpoint/iter3
cp -r checkpoint/iter3_raw/* checkpoint/iter3

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="checkpoint/iter3"