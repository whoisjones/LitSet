import json
import logging

import src

logger = logging.getLogger("logger")


def extract_entities():
    with open(src.DATA_DIR / "zelda" / "zelda" / "other" / "entity_descriptions.jsonl", "r", encoding="utf-8") as lines:
        wikipage_ids = [json.loads(line)["wikipedia_id"] for line in lines]

    entities = src.api.dbpedia.convert_wikipage_to_wikidata(wikipage_ids)
    entities = set(entities)

    logger.info(f"Saving it ZELDA wikipage2wikidata mapping...")
    wikipageID2wikidataID = {wikipage: wikidata for wikipage, wikidata in entities}
    with open(src.ENTITY_DIR / "wikipageID2wikidataID.json", "w") as f:
        json.dump(wikipageID2wikidataID, f)

    zelda_wikidata_entities = set([entity[1] for entity in entities])
    logger.info(f"Saving it ZELDA wikidata entities...")
    with open(src.ENTITY_DIR / "entities.json", "w") as f:
        json.dump(list(zelda_wikidata_entities), f)

