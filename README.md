Attempts at training character-level language models

1. Prepare enwiki9 dataset 
```
bash data/enwiki/download_data.sh
python data/enwiki/prepare_data.py
```

2. Train character-level language model 
```
python train.py config/train_enwiki_char.py
```

