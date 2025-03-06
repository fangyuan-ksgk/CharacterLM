<div align="center">

# 🐳 Building LLM with vocabulary curriculums

[![arxiv](https://img.shields.io/badge/arXiv-2502.17910-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2502.17910)

</div>

Set up 
```
bash set.sh
```

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

