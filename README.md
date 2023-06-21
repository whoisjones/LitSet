# NER4ALL
The repository to the paper "NER4ALL".

NER4ALL uses entity disambiguation datasets for pre-training few-shot named entity recognition models using labels in natural language.

## Requirements
```
conda create -name ner4all python=3.10
pip install -r requirements.txt
```

## How to use
```
python create_dataset.py --download_data --output_format jsonl_bio
```