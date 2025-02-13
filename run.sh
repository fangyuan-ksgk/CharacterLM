run_dir="checkpoint/run1"
mkdir -p $run_dir

python data/enwiki/prepare_data.py --clean --out_dir="$run_dir/base"

python train.py config/train_enwiki_char.py --out_dir="$run_dir/base"
python update.py --out_dir="$run_dir/base" --new_dir="$run_dir/iter1_raw" --thres=0.3 --max_size_change=3000
mkdir -p $run_dir/iter1
cp -r $run_dir/iter1_raw/* $run_dir/iter1

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/iter1"
python update.py --out_dir="$run_dir/iter1" --new_dir="$run_dir/iter2_raw" --thres=0.3 --max_size_change=3000
mkdir -p $run_dir/iter2
cp -r $run_dir/iter2_raw/* $run_dir/iter2

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/iter2"
python update.py --out_dir="$run_dir/iter2" --new_dir="$run_dir/iter3_raw" --thres=0.3 --max_size_change=3000
mkdir -p $run_dir/iter3
cp -r $run_dir/iter3_raw/* $run_dir/iter3

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/iter3"
python update.py --out_dir="$run_dir/iter3" --new_dir="$run_dir/iter4_raw" --thres=0.3 --max_size_change=3000
mkdir -p $run_dir/iter4
cp -r $run_dir/iter4_raw/* $run_dir/iter4

python train.py config/train_enwiki_char.py --init_from="resume" --out_dir="$run_dir/iter4"