import json
import os
import re
import logging
from tqdm import tqdm

import src

logger = logging.getLogger("logger")


def extract_entities():
    trex_entities = extract_trex_entities()
    zelda_entities = extract_zelda_entities()

    wikipageID2wikidataID = {wikipage: wikidata for wikipage, wikidata in zelda_entities}
    with open(src.ENTITY_DIR / "wikipageID2wikidataID.json", "w") as f:
        json.dump(wikipageID2wikidataID, f)

    zelda_wikidata_entities = set([entity[1] for entity in zelda_entities])

    wikidata_entities = trex_entities.union(zelda_wikidata_entities)

    save_path = src.ENTITY_DIR / "wikidata_entities.json"

    logger.info(f"Saving it to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(list(wikidata_entities), f)


def extract_trex_entities() -> set:
    entity_id_pattern = re.compile(r'Q\d+')
    entity_ids = set()

    for file_name in tqdm(os.listdir(src.DATA_DIR / "trex"), desc="Extracting T-REx entities"):
        if file_name.endswith(".zip"):
            continue
        with open(src.DATA_DIR / "trex" / file_name) as f:
            for line in f:
                entities = entity_id_pattern.findall(line)
                entity_ids.update(entities)

    return entity_ids


def extract_zelda_entities() -> set:
    with open(src.DATA_DIR / "zelda" / "zelda" / "other" / "entity_descriptions.jsonl", "r", encoding="utf-8") as lines:
        wikipage_ids = [json.loads(line)["wikipedia_id"] for line in lines]

    entities = src.api.dbpedia.convert_wikipage_to_wikidata(wikipage_ids)

    return set(entities)
