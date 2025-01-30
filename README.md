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

3. Visualize token perplexity pattern 
(a). Spike tokens : sudden jump in perplexity above certain threshold 
![Spike tokens](spiking_tokens.png)

(b). Natural tokens : perplexity below certain threshold 
![Natural tokens](natural_tokens.png)