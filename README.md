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
![Spike tokens](spike_tokens.png)

(b). Natural token groups : consecutive tokens with decreasing perplexity below certain threshold
![Natural token groups](group_token.png)

4. Using 'Group tokens' and 'Spike tokens' to progressively update a 'E-Tokenizer', no more co-occurance based merging, we directly let model decide what it wants to merge. 
