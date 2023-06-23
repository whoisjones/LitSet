import json
import copy
import glob
import random
import argparse
from pathlib import Path
from typing import List, Union, Dict, Any

import numpy as np
import numpy.random
import torch
import torch.cuda
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, PreTrainedTokenizer
from datasets import load_dataset

import flair
from flair.data import Sentence, Dictionary, Token, Span
from flair.embeddings import (
    TransformerWordEmbeddings, TokenEmbeddings,
    TransformerDocumentEmbeddings, Embeddings, SentenceTransformerDocumentEmbeddings
)
from flair.trainers import ModelTrainer
from flair.training_utils import store_embeddings
from torch.utils.data.dataset import Subset

from dataset_loader import get_masked_fewnerd_corpus, get_corpus


class TokenClassifier(flair.nn.DefaultClassifier[Sentence, Token]):
    """This is a simple class of models that tags individual words in text."""

    def __init__(
        self,
        embeddings: TokenEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        span_encoding: str = "BIOES",
        **classifierargs,
    ) -> None:
        """Initializes a TokenClassifier.

        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        """
        # if the classifier predicts BIO/BIOES span labels, the internal label dictionary must be computed
        if label_dictionary.span_labels:
            internal_label_dictionary = self._create_internal_label_dictionary(label_dictionary, span_encoding)
        else:
            internal_label_dictionary = label_dictionary

        super().__init__(
            embeddings=embeddings,
            label_dictionary=internal_label_dictionary,
            final_embedding_size=embeddings.embedding_length,
            **classifierargs,
        )

        # fields in case this is a span-prediction problem
        self.span_prediction_problem = self._determine_if_span_prediction_problem(internal_label_dictionary)
        self.span_encoding = span_encoding

        # the label type
        self._label_type: str = label_type

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    @staticmethod
    def _create_internal_label_dictionary(label_dictionary, span_encoding):
        internal_label_dictionary = Dictionary(add_unk=False)
        for label in label_dictionary.get_items():
            if label == "<unk>":
                continue
            internal_label_dictionary.add_item("O")
            if span_encoding == "BIOES":
                internal_label_dictionary.add_item("S-" + label)
                internal_label_dictionary.add_item("B-" + label)
                internal_label_dictionary.add_item("E-" + label)
                internal_label_dictionary.add_item("I-" + label)
            if span_encoding == "BIO":
                internal_label_dictionary.add_item("B-" + label)
                internal_label_dictionary.add_item("I-" + label)

        return internal_label_dictionary

    def _determine_if_span_prediction_problem(self, dictionary: Dictionary) -> bool:
        return any(item.startswith(("B-", "S-", "I-")) for item in dictionary.get_items())

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "embeddings": self.embeddings.save_embeddings(use_state_dict=False),
            "label_dictionary": self.label_dictionary,
            "label_type": self.label_type,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            embeddings=state.get("embeddings"),
            label_dictionary=state.get("label_dictionary"),
            label_type=state.get("label_type"),
            **kwargs,
        )

    def _get_embedding_for_data_point(self, prediction_data_point: Token) -> torch.Tensor:
        names = self.embeddings.get_names()
        return prediction_data_point.get_embedding(names)

    def _get_data_points_from_sentence(self, sentence: Sentence) -> List[Token]:
        # special handling during training if this is a span prediction problem
        if self.training and self.span_prediction_problem:
            for token in sentence.tokens:
                token.set_label(self.label_type, "O")
                for span in sentence.get_spans(self.label_type):
                    span_label = span.get_label(self.label_type).value
                    if len(span) == 1:
                        if self.span_encoding == "BIOES":
                            span.tokens[0].set_label(self.label_type, "S-" + span_label)
                        elif self.span_encoding == "BIO":
                            span.tokens[0].set_label(self.label_type, "B-" + span_label)
                    else:
                        for token in span.tokens:
                            token.set_label(self.label_type, "I-" + span_label)
                        span.tokens[0].set_label(self.label_type, "B-" + span_label)
                        if self.span_encoding == "BIOES":
                            span.tokens[-1].set_label(self.label_type, "E-" + span_label)

        return sentence.tokens

    def _post_process_batch_after_prediction(self, batch, label_name):
        if self.span_prediction_problem:
            for sentence in batch:
                # internal variables
                previous_tag = "O-"
                current_span: List[Token] = []

                for token in sentence:
                    bioes_tag = token.get_label(label_name).value

                    # non-set tags are OUT tags
                    if bioes_tag == "" or bioes_tag == "O" or bioes_tag == "_":
                        bioes_tag = "O-"

                    # anything that is not OUT is IN
                    in_span = bioes_tag != "O-"

                    # does this prediction start a new span?
                    starts_new_span = False

                    if bioes_tag[:2] in {"B-", "S-"} or (
                        in_span
                        and previous_tag[2:] != bioes_tag[2:]
                        and (bioes_tag[:2] == "I-" or previous_tag[2:] == "S-")
                    ):
                        # B- and S- always start new spans
                        # if the predicted class changes, I- starts a new span
                        # if the predicted class changes and S- was previous tag, start a new span
                        starts_new_span = True

                    # if an existing span is ended (either by reaching O or starting a new span)
                    if (starts_new_span or not in_span) and len(current_span) > 0:
                        sentence[current_span[0].idx - 1 : current_span[-1].idx].set_label(label_name, previous_tag[2:])
                        # reset for-loop variables for new span
                        current_span = []

                    if in_span:
                        current_span.append(token)

                    # remember previous tag
                    previous_tag = bioes_tag

                    token.remove_labels(label_name)
                    token.remove_labels(self.label_type)

                # if there is a span at end of sentence, add it
                if len(current_span) > 0:
                    sentence[current_span[0].idx - 1 : current_span[-1].idx].set_label(label_name, previous_tag[2:])

    @property
    def label_type(self):
        return self._label_type

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        if self.span_prediction_problem:
            for datapoint in batch:
                # all labels default to "O"
                for token in datapoint:
                    token.set_label("gold_bio", "O")
                    token.set_label("predicted_bio", "O")

                # set gold token-level
                for gold_label in datapoint.get_labels(gold_label_type):
                    gold_span: Span = gold_label.data_point
                    prefix = "B-"
                    for token in gold_span:
                        token.set_label("gold_bio", prefix + gold_label.value)
                        prefix = "I-"

                # set predicted token-level
                for predicted_label in datapoint.get_labels("predicted"):
                    predicted_span: Span = predicted_label.data_point
                    prefix = "B-"
                    for token in predicted_span:
                        token.set_label("predicted_bio", prefix + predicted_label.value)
                        prefix = "I-"

                # now print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label('gold_bio').value} "
                        f"{token.get_label('predicted_bio').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")

        else:
            for datapoint in batch:
                # print labels in CoNLL format
                for token in datapoint:
                    eval_line = (
                        f"{token.text} "
                        f"{token.get_label(gold_label_type).value} "
                        f"{token.get_label('predicted').value}\n"
                    )
                    lines.append(eval_line)
                lines.append("\n")
        return lines

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> "TokenClassifier":
        from typing import cast

        return cast("TokenClassifier", super().load(model_path=model_path))


class NER4ALLDecoderFlair(torch.nn.Module):
    def __init__(self, label_embedding: Embeddings, label_dictionary: Dictionary, requires_masking: bool, num_negatives: int = 128):
        super().__init__()
        self.label_embedding = label_embedding
        self.verbalized_labels: List[Sentence] = self.verbalize_labels(label_dictionary)
        self.requires_masking = requires_masking
        self.num_negatives = num_negatives
        self.to(flair.device)

    @staticmethod
    def verbalize_labels(label_dictionary) -> List[Sentence]:
        verbalized_labels = []
        for byte_label, idx in label_dictionary.item2idx.items():
            str_label = byte_label.decode("utf-8")
            if label_dictionary.span_labels:
                if str_label == "O":
                    verbalized_labels.append("outside")
                elif str_label.startswith("B-"):
                    verbalized_labels.append("begin " + str_label.split("-")[1])
                elif str_label.startswith("I-"):
                    verbalized_labels.append("inside " + str_label.split("-")[1])
                elif str_label.startswith("E-"):
                    verbalized_labels.append("ending " + str_label.split("-")[1])
                elif str_label.startswith("S-"):
                    verbalized_labels.append("single " + str_label.split("-")[1])
            else:
                verbalized_labels.append(str_label)
        return list(map(Sentence, verbalized_labels))

    def embedding_sublist(self, labels) -> List[Sentence]:
        unique_entries = set(labels)

        # Randomly sample entries from the larger list
        while len(unique_entries) < self.num_negatives:
            entry = random.choice(range(len(self.verbalized_labels)))
            unique_entries.add(entry)

        return [self.verbalized_labels[idx] for idx in unique_entries], unique_entries

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:

        if self.training and self.requires_masking:
            labels_to_include = labels.cpu().numpy().tolist()
            labels, indices = self.embedding_sublist(labels_to_include)
            self.label_embedding.embed(labels)
        elif not self.training or not self.requires_masking:
            labels = self.label_embedding.embed(self.verbalized_labels)

        label_tensor = torch.stack([label.get_embedding() for label in labels])

        if self.training:
            store_embeddings(labels, "none")

        scores = torch.mm(inputs, label_tensor.T)

        if self.training and self.requires_masking:
            all_scores = torch.zeros(scores.shape[0], len(self.verbalized_labels), device=flair.device)
            all_scores[:, torch.LongTensor(list(indices))] = scores
        elif not self.training or not self.requires_masking:
            all_scores = scores

        return all_scores


class NER4ALLModel(torch.nn.Module):
    def __init__(self, labels: dict, encoder_model: str, decoder_model: str, tokenizer: PreTrainedTokenizer, uniform_p: list, num_negatives: int = 128, geometric_p: float = 0.5):
        super(NER4ALLModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        if decoder_model == "all-mpnet-base-v2":
            self.decoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        else:
            self.decoder = AutoModel.from_pretrained(decoder_model)
        labels = {int(k): v for k, v in labels.items()}
        self.labels = labels
        self.num_labels = len(labels)
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives
        self.uniform_p = uniform_p
        self.geometric_p = geometric_p
        self.loss = torch.nn.CrossEntropyLoss()

    def _verbalize_labels(self, selected_labels):
        #label_descriptions = [self.labels[i] for i in selected_labels]
        label_descriptions = []
        label_granularities = ["description", "labels"]
        for i in selected_labels:
            if i == 0:
                label_description = "outside"
            else:
                label_granularity = np.random.choice(label_granularities, p=self.uniform_p)
                fallback_option = [x for x in label_granularities if x != label_granularity][0]
                if self.labels.get(i).get(label_granularity) is None and self.labels.get(i).get(fallback_option) is not None:
                    label_granularity = fallback_option
                elif self.labels.get(i).get(label_granularity) is None and self.labels.get(i).get(fallback_option) is None:
                    label_description = "miscellaneous"
                    label_descriptions.append(label_description)
                    continue

                if label_granularity == "description":
                    label_description = f"{'begin' if self.labels.get(i)['bio_tag'] == 'B-' else 'inside'} {self.labels.get(i)[label_granularity]}"
                elif label_granularity == "labels":
                    num_labels = np.random.geometric(self.geometric_p, 1)
                    num_labels = num_labels if num_labels <= len(self.labels.get(i).get("labels")) else len(self.labels.get(i).get("labels"))
                    sampled_labels = np.random.choice(self.labels.get(i).get("labels"), num_labels, replace=False).tolist()
                    label_description = f"{'begin' if self.labels.get(i)['bio_tag'] == 'B-' else 'inside'} {', '.join(sampled_labels)}"
                else:
                    raise ValueError(f"Unknown label granularity {label_granularity}")
            label_descriptions.append(label_description)
        return label_descriptions

    def _prepare_labels(self, labels):
        positive_labels = torch.unique(labels)
        positive_labels = positive_labels[(positive_labels != -100)]
        number_negatives_needed = self.num_negatives - positive_labels.size(0)
        if number_negatives_needed > 0:
            negative_labels = numpy.random.choice(np.arange(0, len(self.labels)), size=number_negatives_needed, replace=False)
            labels_for_batch = np.unique(np.concatenate([positive_labels.detach().cpu().numpy(), negative_labels]))
        else:
            labels_for_batch = positive_labels.detach().cpu().numpy()
        labels = self.adjust_batched_labels(labels_for_batch, labels)
        label_descriptions = self._verbalize_labels(labels_for_batch)
        encoded_labels = self.tokenizer(label_descriptions, padding=True, truncation=True, max_length=64, return_tensors="pt").to(labels.device)
        return encoded_labels, labels

    def adjust_batched_labels(self, labels_for_batch, original_label_tensor):
        batch_size = original_label_tensor.size(0)
        adjusted_label_tensor = torch.zeros_like(original_label_tensor)

        labels_for_batch = labels_for_batch[(labels_for_batch != 0)]

        label_mapping = {label.item(): idx + 1 for idx, label in enumerate(labels_for_batch)}
        label_mapping[-100] = -100
        label_mapping[0] = 0

        for i in range(batch_size):
            adjusted_label_tensor[i] = torch.tensor([label_mapping.get(label.item(), -1) for label in original_label_tensor[i]])

        return adjusted_label_tensor

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, labels):
        token_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_labels, labels = self._prepare_labels(labels)
        label_hidden_states = self.decoder(**encoded_labels)
        if "sentence-transformers" in self.decoder.name_or_path:
            label_embeddings = self.mean_pooling(label_hidden_states, encoded_labels['attention_mask'])
            label_embeddings = torch.nn.functional.normalize(label_embeddings, p=2, dim=1)
        else: # take cls token
            label_embeddings = label_hidden_states.last_hidden_state[:, 0, :]
        logits = torch.matmul(token_hidden_states.last_hidden_state, label_embeddings.T)
        return (self.loss(logits.transpose(1, 2), labels),)


def get_save_base_path(args, task_name):
    is_pretraining = True if "pretrain" in task_name else False
    if args.dataset_path:
        is_zelda = True if "zelda" in args.dataset_path.lower() else False
        dataset = f"NER4ALL{'-ZELDA' if is_zelda else ''}"
    else:
        dataset = f"{args.dataset}{args.fewnerd_granularity if args.dataset == 'fewnerd' else ''}"

    if is_pretraining:
        sampling = "-".join([str(x) for x in args.uniform_p])
        training_arguments = f"_{args.lr}_seed-{args.seed}_mask-{args.num_negatives}_size-{args.corpus_size}{f'_sampling-{sampling}' if sampling != '0.5-0.5' else ''}"

        if args.encoder_transformer == args.decoder_transformer:
            model_arguments = f"{args.encoder_transformer}"
        else:
            model_arguments = f"{args.encoder_transformer}_{args.decoder_transformer}"
    else:
        pretraining_model = args.pretrained_hf_encoder.split('/')[-2]
        training_arguments = f"-{args.lr}_pretrained-on-{pretraining_model}"

    return Path(
        f"{args.cache_path}/{task_name}/"
        f"{model_arguments + '_' if is_pretraining else ''}"
        f"{dataset}"
        f"{training_arguments}"
    )


def get_corpus_size(args):
    if args.corpus_size == "100k":
        num_samples = 100000
    elif args.corpus_size == "500k":
        num_samples = 500000
    elif args.corpus_size == "1M":
        num_samples = 1000000
    else:
        raise ValueError("Invalid corpus size")
    return num_samples


def pretrain(args):
    pl.seed_everything(args.seed)

    save_base_path = get_save_base_path(args, task_name="pretrained-ner4all")

    dataset = load_dataset("json", data_files=glob.glob(f'{args.dataset_path}/*'))
    num_samples = get_corpus_size(args)
    random_numbers = random.sample(range(0, len(dataset["train"]) + 1), num_samples)
    small_dataset = dataset["train"].select(random_numbers)

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_transformer)
    if args.decoder_transformer == "all-mpnet-base-v2":
        decoder_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_transformer)

    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_and_align_labels(examples):
        tokenized_inputs = encoder_tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    train_dataset = small_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=small_dataset.column_names,
    )

    data_collator = DataCollatorForTokenClassification(encoder_tokenizer)

    training_args = TrainingArguments(
        output_dir=str(save_base_path),
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        save_strategy="epoch",
        save_total_limit=3,
        seed=args.seed,
    )

    with open(args.labelID2label_file, 'r') as f:
        labels = json.load(f)

    model = NER4ALLModel(labels=labels, encoder_model=args.encoder_transformer, decoder_model=args.decoder_transformer, tokenizer=decoder_tokenizer, num_negatives=args.num_negatives, uniform_p=args.uniform_p, geometric_p=args.geometric_p)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=encoder_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.encoder.save_pretrained(save_base_path / "encoder")
    encoder_tokenizer.save_pretrained(save_base_path / "encoder")
    model.decoder.save_pretrained(save_base_path / "decoder")
    decoder_tokenizer.save_pretrained(save_base_path / "decoder")


def tagset_adaption(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name="fewshot-ner4all-tagset-adaption")

    with open(f"data/fewshot_fewnerdfine.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}

    # every pretraining seed masks out different examples in the dataset
    base_corpus = get_corpus(args.dataset, args.fewnerd_granularity)

    # iterate over k-shots
    for k in args.k:

        # average k-shot scores over 3 seeds for pretraining seed
        results[f"{k}"] = {"results": []}

        for seed in range(0, 5):

            if seed > 0 and k == 0:
                continue

            # ensure same sampling strategy for each seed
            flair.set_seed(seed)
            corpus = copy.copy(base_corpus)
            if k != 0:
                if k == -1:
                    pass
                else:
                    corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{seed}"])
                    corpus._dev = Subset(base_corpus._train, [])
            else:
                pass

            # mandatory for flair to work
            tag_type = "ner"
            label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
            decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary, span_encoding="BIO")
            decoder_dict.span_labels = True

            encoder = TransformerWordEmbeddings(args.pretrained_hf_encoder, use_context_separator=False, use_context=False)
            if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                label_embeddings = SentenceTransformerDocumentEmbeddings(args.pretrained_hf_decoder)
            else:
                label_embeddings = TransformerDocumentEmbeddings(args.pretrained_hf_decoder, use_context_separator=False, use_context=False)
            decoder = NER4ALLDecoderFlair(
                label_embedding=label_embeddings, label_dictionary=decoder_dict,
                requires_masking=False, num_negatives=args.num_negatives)
            model = TokenClassifier(embeddings=encoder, decoder=decoder, label_dictionary=label_dictionary,
                                    label_type=tag_type, span_encoding="BIO")

            if k != 0:
                trainer = ModelTrainer(model, corpus)

                save_path = save_base_path / f"{k}shot_{seed}"

                # 7. run fine-tuning
                result = trainer.train(
                    save_path,
                    learning_rate=args.lr,
                    mini_batch_size=args.bs,
                    mini_batch_chunk_size=args.mbs,
                    max_epochs=args.epochs if k != -1 else 3,
                    optimizer=torch.optim.AdamW,
                    train_with_dev=True,
                    min_learning_rate=args.lr * 1e-2,
                    save_final_model=False,
                )

                results[f"{k}"]["results"].append(result["test_score"])

                for sentence in corpus.train:
                    for token in sentence:
                        token.remove_labels(tag_type)
            else:
                save_path = save_base_path / f"{k}shot_{seed}"
                import os

                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                result = model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                results[f"{k}"]["results"].append(result.main_score)
                with open(save_path / "result.txt", "w") as f:
                    f.write(result.detailed_results)


def low_resource(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name="fewshot-ner4all-low-resource")

    with open(f"data/fewshot_{args.dataset}{args.fewnerd_granularity if args.dataset == 'fewnerd' else ''}.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}

    # every pretraining seed masks out different examples in the dataset
    base_corpus = get_corpus(args.dataset, args.fewnerd_granularity)

    # iterate over k-shots
    for k in args.k:

        # average k-shot scores over 5 seeds for pretraining seed
        results[f"{k}"] = {"results": []}

        for seed in range(0, 5):

            if seed > 0 and k == 0:
                continue

            # ensure same sampling strategy for each seed
            flair.set_seed(seed)
            corpus = copy.copy(base_corpus)
            if k != 0:
                if k == -1:
                    pass
                else:
                    corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{seed}"])
                    corpus._dev = Subset(base_corpus._train, [])
            else:
                pass

            # mandatory for flair to work
            tag_type = "ner"
            label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
            decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary, span_encoding="BIO")
            decoder_dict.span_labels = True

            # Create model
            encoder = TransformerWordEmbeddings(args.pretrained_hf_encoder, use_context_separator=False, use_context=False)
            if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                label_embeddings = SentenceTransformerDocumentEmbeddings(args.pretrained_hf_decoder)
            else:
                label_embeddings = TransformerDocumentEmbeddings(args.pretrained_hf_decoder, use_context_separator=False, use_context=False)
            decoder = NER4ALLDecoderFlair(
                label_embedding=label_embeddings, label_dictionary=decoder_dict,
                requires_masking=False, num_negatives=args.num_negatives)
            model = TokenClassifier(embeddings=encoder, decoder=decoder, label_dictionary=label_dictionary,
                                    label_type=tag_type, span_encoding="BIO")

            if k != 0:
                trainer = ModelTrainer(model, corpus)

                save_path = save_base_path / f"{k}shot_{seed}"

                # 7. run fine-tuning
                result = trainer.train(
                    save_path,
                    learning_rate=args.lr,
                    mini_batch_size=args.bs,
                    mini_batch_chunk_size=args.mbs,
                    max_epochs=args.epochs if k != -1 else 3,
                    optimizer=torch.optim.AdamW,
                    train_with_dev=True,
                    min_learning_rate=args.lr * 1e-2,
                    save_final_model=False,
                )

                results[f"{k}"]["results"].append(result["test_score"])

                for sentence in corpus.train:
                    for token in sentence:
                        token.remove_labels(tag_type)
            else:
                save_path = save_base_path / f"{k}shot_{seed}"
                import os

                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                result = model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                results[f"{k}"]["results"].append(result.main_score)
                with open(save_path / "result.txt", "w") as f:
                    f.write(result.detailed_results)


def tagset_extension(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name="fewshot-ner4all-tagset-extension")

    with open(f"data/fewshot_masked-fewnerd-{args.fewnerd_granularity}.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}
    for fewshot_seed in args.fewshot_seeds:

        # every pretraining seed masks out different examples in the dataset
        base_corpus, kept_labels = get_masked_fewnerd_corpus(
            fewshot_seed, args.fewnerd_granularity, inverse_mask=True
        )

        # iterate over k-shots
        for k in args.k:

            # average k-shot scores over 3 seeds for pretraining seed
            results[f"{k}-{fewshot_seed}"] = {"results": []}

            for seed in range(0, 5):
                # ensure same sampling strategy for each seed
                flair.set_seed(seed)
                corpus = copy.copy(base_corpus)
                if k != 0:
                    corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{fewshot_seed}-{seed}"])
                else:
                    pass
                corpus._dev = Subset(base_corpus._train, [])

                # mandatory for flair to work
                tag_type = "ner"
                label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
                decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary, span_encoding="BIO")
                decoder_dict.span_labels = True

                encoder = TransformerWordEmbeddings(args.pretrained_hf_encoder)
                if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                    label_embeddings = SentenceTransformerDocumentEmbeddings(args.pretrained_hf_decoder)
                else:
                    label_embeddings = TransformerDocumentEmbeddings(args.pretrained_hf_decoder)
                decoder = NER4ALLDecoderFlair(
                    label_embedding=label_embeddings, label_dictionary=decoder_dict,
                    requires_masking=False, num_negatives=args.num_negatives)
                model = TokenClassifier(embeddings=encoder, decoder=decoder, label_dictionary=label_dictionary,
                                        label_type=tag_type, span_encoding="BIO")

                if k > 0:
                    trainer = ModelTrainer(model, corpus)

                    save_path = save_base_path / f"{k}shot_{fewshot_seed}_{seed}"

                    # 7. run fine-tuning
                    result = trainer.train(
                        save_path,
                        learning_rate=args.lr,
                        mini_batch_size=args.bs,
                        mini_batch_chunk_size=args.mbs,
                        max_epochs=args.epochs,
                        optimizer=torch.optim.AdamW,
                        train_with_dev=True,
                        min_learning_rate=args.lr * 1e-2,
                        save_final_model=False,
                    )

                    results[f"{k}-{fewshot_seed}"]["results"].append(result["test_score"])

                    for sentence in corpus.train:
                        for token in sentence:
                            token.remove_labels(tag_type)
                else:
                    save_path = save_base_path / f"{k}shot_{fewshot_seed}_{seed}"
                    import os

                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                    result = model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                    results[f"{k}-{fewshot_seed}"]["results"].append(result.main_score)
                    with open(save_path / "result.txt", "w") as f:
                        f.write(result.detailed_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--low_resource", action="store_true")
    parser.add_argument("--tagset_extension", action="store_true")
    parser.add_argument("--tagset_adaption", action="store_true")

    # Pretraining arguments
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fewshot_seeds", type=int, nargs="+", default=[10])
    parser.add_argument("--cache_path", type=str, default="") #TODO: insert path to working directory here
    # Few datasets loaded from flair or huggingface
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--fewnerd_granularity", type=str, default="") # only required when dataset == "fewnerd"
    # NER4ALL needs to be loaded from disk, use dataset_path argument
    parser.add_argument("--dataset_path", type=str, default="") #TODO: insert path to dataset here
    parser.add_argument("--labelID2label_file", type=str, default="") #TODO: insert path to labelID2labl.json file here
    parser.add_argument("--corpus_size", type=str, default="100k")
    parser.add_argument("--encoder_transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--decoder_transformer", type=str, default="bert-base-uncased")

    # Training arguments
    parser.add_argument("--num_negatives", type=int, default=0)
    parser.add_argument("--uniform_p", type=float, nargs="+", default=[0.5, 0.5])
    parser.add_argument("--geometric_p", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--mbs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="+", default=[1])
    parser.add_argument("--pretrained_hf_encoder", type=str, default="")
    parser.add_argument("--pretrained_hf_decoder", type=str, default="")
    args = parser.parse_args()

    if not any([args.dataset, args.dataset_path]):
        raise ValueError("no dataset provided.")

    if args.pretrain:
        pretrain(args)
    if args.low_resource:
        low_resource(args)
    if args.tagset_extension:
        tagset_extension(args)
    if args.tagset_adaption:
        tagset_adaption(args)
