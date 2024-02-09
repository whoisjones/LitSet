import json
import glob
from itertools import chain

from tqdm import tqdm
from syntok.tokenizer import Tokenizer
import syntok.segmenter as segmenter

import src
from src.store import write_to_file, jsonl_counter, jsonl_bio_counter

tok = Tokenizer()


def tokenize(text):
    return tok.tokenize(text)


def get_entities_from_trex_sample(sample, entityID2labelID):
    entities = {}
    for entity in sample["entities"]:
        entity_id = entity["uri"].split("/")[-1]
        annotator = entity["annotator"]
        start, end = entity["boundaries"][0], entity["boundaries"][1]
        if annotator == "Wikidata_Spotlight_Entity_Linker":
            if entityID2labelID.get(entity_id) is not None:
                entities[range(start, end)] = entityID2labelID.get(entity_id)

    return entities


def get_entities_from_zelda_sample(sample, entityID2labelID, wikipageID2wikidataID):
    entities = {}

    for entity, boundaries in zip(sample["wikipedia_ids"], sample["index"]):
        entity_id = wikipageID2wikidataID.get(str(entity))
        start, end = boundaries[0], boundaries[1]
        if entityID2labelID.get(entity_id) is not None:
            entities[range(start, end)] = entityID2labelID.get(entity_id)

    return entities


def create_datapoint_from_zelda(sample, entityID2labelID, wikipageID2wikidataID):
    entities = get_entities_from_zelda_sample(sample, entityID2labelID, wikipageID2wikidataID)
    for paragraph in segmenter.process(sample["text"]):
        boundaries = []
        for sentence in paragraph:
            boundaries.append([sentence[0].offset, sentence[-1].offset + len(sentence[-1].value)])
        sample["sentences_boundaries"] = boundaries
    return entities, sample


def sample_to_jsonl(sample, entities):

    for boundary in sample["sentences_boundaries"]:

        sentence = sample["text"][boundary[0]:boundary[1]]
        tokens = []
        ner_tags = []
        for token in tok.tokenize(sentence):
            tokens.append(token.value.strip())

            matches = [(entity_range, labels) for entity_range, labels in entities.items() if token.offset + boundary[0] in entity_range]
            if matches:
                ranges, label = zip(*matches)
                ner_tags.append(label[0])
            else:
                ner_tags.append(0)

        assert len(tokens) == len(ner_tags)

        yield {
            "id": jsonl_counter(),
            "tokens": tokens,
            "ner_tags": ner_tags
        }


def sample_to_jsonl_bio(sample, entities):

    for boundary in sample["sentences_boundaries"]:

        sentence = sample["text"][boundary[0]:boundary[1]]
        tokens = []
        ner_tags = []
        prev_io_tag = None
        for token in tok.tokenize(sentence):
            tokens.append(token.value.strip())

            matches = [(entity_range, labels) for entity_range, labels in entities.items() if token.offset + boundary[0] in entity_range]
            if matches:
                ranges, label = zip(*matches)
                if prev_io_tag is None or prev_io_tag != label[0]:
                    ner_tags.append(label[0])
                else:
                    ner_tags.append(label[0] + 1)
                prev_io_tag = label[0]
            else:
                ner_tags.append(0)
                prev_io_tag = 0

        assert len(tokens) == len(ner_tags)

        if len(tokens) == 0:
            continue

        yield {
            "id": jsonl_bio_counter(),
            "tokens": tokens,
            "ner_tags": ner_tags
        }


def sample_to_format_generator(output_format: str, entities: dict, sample: dict):
    if output_format == "jsonl":
        return sample_to_jsonl(sample, entities)
    elif output_format == "jsonl_bio":
        return sample_to_jsonl_bio(sample, entities)
    else:
        raise ValueError("Invalid output format")


def build_NER(output_format: list):
    # Quality checks
    valid_output_formats = ["jsonl", "jsonl_bio"]
    if not all(item in valid_output_formats for item in output_format) and len(set(output_format)) <= 2:
        raise ValueError("Invalid output format")

    # Load all relevant mappings
    with open(src.ENTITY_DIR / "entityID2labelID.json") as f:
        entityID2labelID = json.load(f)

    with open(src.ENTITY_DIR / "wikipageID2wikidataID.json", "r") as f:
        wikipageID2wikidataID = json.load(f)

    # Generate Zelda data points
    zelda_files = glob.glob(str(src.DATA_DIR / "zelda" / "zelda" / "train_data" / "*.jsonl"))

    for zelda_file in tqdm(zelda_files, desc="Processing Zelda files"):

        with open(zelda_file, encoding="utf-8") as f:
            input_file = f.readlines()

        format_generators = {_format: [] for _format in output_format}

        test = 10
        for idx, sample in tqdm(enumerate(input_file), desc="Processing Zelda samples"):
            sample = json.loads(sample)
            entities, sample = create_datapoint_from_zelda(sample, entityID2labelID, wikipageID2wikidataID)

            for _format in output_format:
                format_generators[_format].append(sample_to_format_generator(_format, entities, sample))

            if idx > test:
                break

        for _format, generators in format_generators.items():
            write_to_file(chain(*generators), _format)
