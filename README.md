Building LLM with vocabulary curriculums.

Set up 
```
bash set.sh
```

arXiv: https://arxiv.org/abs/2502.17910

Pre-train base model on enwiki8 dataset (with vocabulary curriculum) 
```
bash run_vocab_curriculum.sh
```

In comparison, pre-training base model with same FLOPs without vocabulary curriculum: 
```
bash run_flop_match.sh
```

Supervised finetuning on Alpaca dataset 
```
bash run_sft.sh
```

