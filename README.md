# LoNER
The repository to the paper "LoNER".

LoNER is a pretraining corpus for named entity recognition with over 100k fine-defined labels. The dataset is particularly
designed for NER approaches utilizing label verbalizers to capture the semantics of the labels.

You can find the dataset on huggingface: https://huggingface.co/datasets/loner

Or download it here directly: [CoNLL-format](to do), [JSONL-format](to do), [JSONL-format with offsets](to do)

## Requirements
```
conda create -name loner python=3.8
pip install -r requirements.txt
```

## How to create a dataset from scratch
```
python create_dataset.py --download_data --output_format ["jsonl", "conll", "jsonl_with_offsets"]
```

### Citation
If you use this dataset, please cite our paper.