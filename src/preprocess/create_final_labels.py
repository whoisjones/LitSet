import json
from tqdm import tqdm

import src


def create_final_annotations():
    with open(src.ENTITY_DIR / "entities.json") as f:
        entities = json.load(f)

    with open(src.ENTITY_DIR / "entityID2instanceID.json") as f:
        entityID2instanceID = json.load(f)

    with open(src.ENTITY_DIR / "instanceID2subclassID.json") as f:
        instanceID2subclassID = json.load(f)

    with open(src.ENTITY_DIR /"instanceID2label.json") as f:
        instanceID2label = json.load(f)

    with open(src.ENTITY_DIR /"subclassID2label.json") as f:
        subclassID2label = json.load(f)

    with open(src.ENTITY_DIR /"entityID2shortdescriptions.json") as f:
        entityID2shortdescriptions = json.load(f)

    entityID2labelID = {}
    labelID2label = {"0": "O"}
    for entity_id in tqdm(entities, desc="Computing entity labels"):
        instance_ids = entityID2instanceID.get(entity_id)
        if instance_ids is not None:
            instance_labels = [instanceID2label.get(instance_id) for instance_id in instance_ids]
            subclass_ids = [item for sublist in
                            [instanceID2subclassID.get(instance_id) for instance_id in instance_ids if
                             instanceID2subclassID.get(instance_id) is not None] for
                            item in sublist]
            subclass_labels = [subclassID2label.get(subclass_id) for subclass_id in subclass_ids]

            possible_labels = instance_labels + subclass_labels
            weak_exclusions = ["list"]
            for excluded_term in weak_exclusions:
                possible_labels = [possible_label for possible_label in possible_labels if excluded_term != possible_label]

            strong_exclusions = ["Wikimedia", "MediaWiki"]
            for excluded_term in strong_exclusions:
                possible_labels = [possible_label for possible_label in possible_labels if
                                   excluded_term not in possible_label]

        possible_description = entityID2shortdescriptions.get(entity_id)

        annotation = {}
        if possible_labels:
            annotation["labels"] = list(set(possible_labels))

        if possible_description:
            annotation["description"] = possible_description

        if annotation:
            label_id = len(labelID2label)
            labelID2label[label_id] = {**annotation, **{"bio_tag": "B-"}}
            labelID2label[label_id + 1] = {**annotation, **{"bio_tag": "I-"}}
            entityID2labelID[entity_id] = label_id

    with open(src.ENTITY_DIR / "entityID2labelID.json", "w") as f:
        json.dump(entityID2labelID, f)

    with open(src.ENTITY_DIR / "labelID2label.json", "w") as f:
        json.dump(labelID2label, f)
