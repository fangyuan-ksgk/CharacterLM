run_dir="checkpoint/run_spline"
mkdir -p $run_dir

python data/enwiki/prepare_data.py --clean --out_dir="$run_dir"

# train spline-gpt (0th layer token embedding enhancement)
python train.py config/train_enwiki_char_spline.py --out_dir="$run_dir/base"
python eval.py --model_type="SplineGPT" --out_dir="$run_dir/base" --run_idx=0

# train spline-gpt (1st layer token embedding enhancement)
python train.py config/train_enwiki_char_spline.py --out_dir="$run_dir/layer1"
python eval.py --model_type="SplineGPT" --out_dir="$run_dir/layer1" --run_idx=1

# train spline-gpt (2nd layer token embedding enhancement)
python train.py config/train_enwiki_char_spline.py --out_dir="$run_dir/layer2"
python eval.py --model_type="SplineGPT" --out_dir="$run_dir/layer2" --run_idx=2

# train spline-gpt (3rd layer token embedding enhancement)
python train.py config/train_enwiki_char_spline.py --out_dir="$run_dir/layer3"
python eval.py --model_type="SplineGPT" --out_dir="$run_dir/layer3" --run_idx=3

# train spline-gpt (4th layer token embedding enhancement)
python train.py config/train_enwiki_char_spline.py --out_dir="$run_dir/layer4"
python eval.py --model_type="SplineGPT" --out_dir="$run_dir/layer4" --run_idx=4