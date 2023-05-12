import json
import logging
from tqdm import tqdm

import src

logger = logging.getLogger("logger")


def filter_wrong_value_format_in_mapping(dictionary):
    filtered_dictionary = {}
    for key, values in tqdm(dictionary.items(), desc="Filtering"):
        if not isinstance(key, str):
            logger.info("Invalid key: not a string. Removing it.")
            continue

        if not key.startswith("Q"):
            logger.info("Invalid key: does not start with 'Q'. Removing it.")
            continue

        if not isinstance(values, list):
            logger.info("Invalid value: not a list")
            values = []
        else:
            if not all([isinstance(_, str) for _ in values]):
                logger.info("Invalid value: not a string. Removing it.")
                values = [_ for _ in values if isinstance(_, str)]

            if not all([_.startswith("Q") for _ in values]):
                logger.info(f"Invalid value: does not start with 'Q'. Removing it.")
                values = [_ for _ in values if _.startswith("Q")]

        if len(values) == 0:
            continue
        else:
            filtered_dictionary[key] = values

    return filtered_dictionary


def filter_wrong_value_format_in_label_mapping(dictionary):
    filtered_dictionary = {}
    for key, values in dictionary.items():
        if not isinstance(key, str):
            logger.info("Invalid key: not a string. Removing it.")
            continue

        if not key.startswith("Q"):
            logger.info("Invalid key: does not start with 'Q'. Removing it.")
            continue

        if not isinstance(values, str):
            logger.info("Invalid value: not a string. Removing it.")
            continue

        filtered_dictionary[key] = values

    return filtered_dictionary


def clean_crawled_entities():

    with open(src.ENTITY_DIR / "entityID2instanceID.json", "r") as f:
        entityID2instanceID = json.load(f)

    filtered_entityID2instanceID = filter_wrong_value_format_in_mapping(entityID2instanceID)

    if not entityID2instanceID == filtered_entityID2instanceID:
        with open(src.ENTITY_DIR / "entityID2instanceID.json", "w") as f:
            json.dump(filtered_entityID2instanceID, f)

    with open(src.ENTITY_DIR / "instanceID2subclassID.json", "r") as f:
        instanceID2subclassID = json.load(f)

    filtered_instanceID2subclassID = filter_wrong_value_format_in_mapping(instanceID2subclassID)

    if not instanceID2subclassID == filtered_instanceID2subclassID:
        with open(src.ENTITY_DIR / "instanceID2subclassID.json", "w") as f:
            json.dump(filtered_instanceID2subclassID, f)

    with open(src.ENTITY_DIR / "instanceID2label.json", "r") as f:
        instanceID2label = json.load(f)

    filtered_instanceID2label = filter_wrong_value_format_in_label_mapping(instanceID2label)

    if not instanceID2label == filtered_instanceID2label:
        with open(src.ENTITY_DIR / "instanceID2label.json", "w") as f:
            json.dump(filtered_instanceID2label, f)

    with open(src.ENTITY_DIR / "subclassID2label.json", "r") as f:
        subclassID2label = json.load(f)

    filtered_subclassID2label = filter_wrong_value_format_in_label_mapping(subclassID2label)

    if not subclassID2label == filtered_subclassID2label:
        with open(src.ENTITY_DIR / "subclassID2label.json", "w") as f:
            json.dump(filtered_subclassID2label, f)
