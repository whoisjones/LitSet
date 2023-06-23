# NER4ALL
The repository to the paper "NER4ALL".

NER4ALL uses entity disambiguation datasets for pre-training few-shot named entity recognition models using labels in natural language.

## Requirements
```
conda create -name ner4all python=3.10
pip install -r requirements.txt
```

## How to create the dataset
```
python create_dataset.py --download_data --output_format jsonl_bio
```

## How to run experiments
Pre-training on NER4ALL+ZELDA:
```
python experiments/train_ner4all.py --pretrain --dataset_path PATH_TO_CREATED_DATASET --labelID2label_file PATH_TO_LABEL_FILE --corpus_size 100k/500k/1M 
```

Low-resource experiments on NER4ALL+ZELDA:
```
python experiments/train_ner4all.py --low_resource --dataset fewnerd --fewnerd_granularity fine --pretrained_hf_encoder PATH_TO_PRETRAINED_NER4ALL_TOKEN_ENCODER --pretrained_hf_decoder PATH_TO_PRETRAINED_NER4ALL_LABEL_DECODER --lr 5e-6 --epochs 200 --k 1 2 4 8 16 0 -1
```
k = -1 means full-finetuning on the target dataset.

Tag-set extension on NER4ALL+ZELDA:
```
python experiments/train_ner4all.py --tagset_extension --dataset fewnerd --fewnerd_granularity fine --pretrained_hf_encoder PATH_TO_PRETRAINED_NER4ALL_TOKEN_ENCODER --pretrained_hf_decoder PATH_TO_PRETRAINED_NER4ALL_LABEL_DECODER --lr 5e-6 --epochs 200 --k 1 2 4 8 16 0 --fewshot_seeds 10 20 30 50
```
Few-shot seeds are used to generate different label splits on FewNERD.

Tag-set adaption on NER4ALL+ZELDA:
```
python experiments/train_ner4all.py --tagset_adaption --dataset fewnerd --fewnerd_granularity fine --pretrained_hf_encoder PATH_TO_PRETRAINED_NER4ALL_TOKEN_ENCODER --pretrained_hf_decoder PATH_TO_PRETRAINED_NER4ALL_LABEL_DECODER --lr 5e-6 --epochs 200 --k 1 2 4 8 16 0
```