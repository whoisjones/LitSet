import json
import glob

from tqdm import tqdm
from syntok.tokenizer import Tokenizer
import syntok.segmenter as segmenter

import src
from src.store import write_to_file, counter
from src.utils import to_range, is_range_within

tok = Tokenizer()

def tokenize(text):
    return tok.tokenize(text)


def get_entities_from_trex_sample(sample, entityID2labelID, labelID2label):
    entities = {}

    for entity in sample["entities"]:
        entity_id = entity["uri"].split("/")[-1]
        annotator = entity["annotator"]
        start, end = entity["boundaries"][0], entity["boundaries"][1]
        if annotator == "Wikidata_Spotlight_Entity_Linker":
            if labelID2label.get(str(entityID2labelID.get(entity_id))) is not None:
                entities[range(start, end)] = labelID2label.get(str(entityID2labelID.get(entity_id)))

    return entities


def get_entities_from_zelda_sample(sample, entityID2labelID, labelID2label, wikipageID2wikidataID):
    entities = {}

    for entity, boundaries in zip(sample["wikipedia_ids"], sample["index"]):
        entity_id = wikipageID2wikidataID.get(str(entity))
        start, end = boundaries[0], boundaries[1]
        if labelID2label.get(str(entityID2labelID.get(entity_id))) is not None:
            entities[range(start, end)] = labelID2label.get(str(entityID2labelID.get(entity_id)))

    return entities

def sample_to_conll(sample, entities):

    for boundary in sample["sentences_boundaries"]:
        conll_sentence = ""

        sentence = sample["text"][boundary[0]:boundary[1]]

        for token in tok.tokenize(sentence):
            matches = [(entity_range, labels) for entity_range, labels in entities.items() if token.offset + boundary[0] in entity_range]

            if matches:
                ranges, label = zip(*matches)
                beginning_offsets = [r.start - boundary[0] for r in ranges]
                if token.offset in beginning_offsets:
                    conll_sentence += f"{token.value.strip()}\t{f'B-{label[0]}'}\n"
                else:
                    conll_sentence += f"{token.value.strip()}\t{f'I-{label[0]}'}\n"
            else:
                conll_sentence += f"{token.value.strip()}\t{'O'}\n"

        yield conll_sentence


def sample_to_jsonl(sample, entities):

    for boundary in sample["sentences_boundaries"]:
        text = sample["text"][boundary[0]:boundary[1]]
        labels = []
        offsets = []
        for entity_range in entities.keys():
            if is_range_within(to_range(boundary), entity_range):
                labels.append(entities.get(entity_range))
                offsets.append([entity_range.start - boundary[0], entity_range.stop - boundary[0]])

        yield {
            "id": counter(),
            "text": text,
            "offsets": offsets,
            "labels": labels
        }


def create_labeled_datapoint(dataset, sample, output_format, entityID2labelID, labelID2label, wikipageID2wikidataID = None):
    if dataset == "trex":
        entities = get_entities_from_trex_sample(sample, entityID2labelID, labelID2label)
    elif dataset == "zelda":
        entities = get_entities_from_zelda_sample(sample, entityID2labelID, labelID2label, wikipageID2wikidataID)
        for paragraph in segmenter.process(sample["text"]):
            boundaries = []
            for sentence in paragraph:
                boundaries.append([sentence[0].offset, sentence[-1].offset + len(sentence[-1].value)])
            sample["sentences_boundaries"] = boundaries
    else:
        raise ValueError("Invalid dataset")

    if output_format == "conll":
        return sample_to_conll(sample, entities)
    elif output_format == "jsonl":
        return sample_to_jsonl(sample, entities)
    else:
        raise ValueError("Invalid output format")


def build_NER(output_format: list):
    # Quality checks
    valid_output_formats = {"conll", "jsonl"}
    if not all(item in valid_output_formats for item in output_format) and len(set(output_format)) <= 2:
        raise ValueError("Invalid output format")

    # Load all relevant mappings
    with open(src.ENTITY_DIR / "entityID2labelID.json") as f:
        entityID2labelID = json.load(f)

    with open(src.ENTITY_DIR / "labelID2label.json") as f:
        labelID2label = json.load(f)

    with open(src.ENTITY_DIR / "wikipageID2wikidataID.json", "r") as f:
        wikipageID2wikidataID = json.load(f)

    # Generate Zelda data points
    zelda_files = glob.glob(str(src.DATA_DIR / "zelda" / "zelda" / "train_data" / "*.jsonl")) \
                + glob.glob(str(src.DATA_DIR / "zelda" / "zelda" / "test_data" / "jsonl" / "*.jsonl"))

    for zelda_file in tqdm(zelda_files, desc="Processing Zelda files"):
        with open(zelda_file) as f:
            input_file = f.readlines()

        for sample in tqdm(input_file, desc="Processing Zelda samples"):
            sample = json.loads(sample)
            for _format in output_format:
                sample_generator = create_labeled_datapoint("zelda", sample, _format, entityID2labelID, labelID2label, wikipageID2wikidataID)
                for data_point in sample_generator:
                    write_to_file(data_point, _format)

    # Generate T-REx data points
    trex_files = glob.glob(str(src.DATA_DIR / "trex" / "*.json"))

    for trex_file in tqdm(trex_files, desc="Processing T-REx files"):

        with open(trex_file) as f:
            input_file = json.load(f)

        for sample in input_file:

            for _format in output_format:
                sample_generator = create_labeled_datapoint("trex", sample, _format, entityID2labelID, labelID2label)
                for data_point in sample_generator:
                    write_to_file(data_point, _format)
