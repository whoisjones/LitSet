# LitSet - Dataset Generation
The repository to generete the [LitSet](https://github.com/flairNLP/label-interpretation-learning) dataset for our EACL'24 paper.

LitSet uses entity disambiguation datasets for label interpretation training. The resulting model can be used for few-shot named entity recognition using verbalized label descriptions.

## Requirements
```
conda create -name litset_dataset python=3.10
conda activate litset_dataset
pip install -r requirements.txt
```

## How to create the dataset
```
python create_dataset.py --download_data --output_format jsonl
```