import json
from tqdm import tqdm

import src


def create_final_annotations():
    with open("entities/entityID2instanceID.json") as f:
        entityID2instanceID = json.load(f)

    with open("entities/instanceID2subclassID.json") as f:
        instanceID2subclassID = json.load(f)

    with open("entities/instanceID2label.json") as f:
        instanceID2label = json.load(f)

    with open("entities/subclassID2label.json") as f:
        subclassID2label = json.load(f)

    entityID2labelID = {}
    label2labelID = {}
    for entity_id, instance_ids in tqdm(entityID2instanceID.items(), desc="Computing entity labels"):
        instance_labels = [instanceID2label.get(instance_id) for instance_id in instance_ids]
        subclass_ids = [item for sublist in
                        [instanceID2subclassID.get(instance_id) for instance_id in instance_ids if instanceID2subclassID.get(instance_id) is not None] for
                        item in sublist]
        subclass_labels = [subclassID2label.get(subclass_id) for subclass_id in subclass_ids]

        possible_labels = instance_labels + subclass_labels

        weak_exclusions = ["list"]
        for excluded_term in weak_exclusions:
            possible_labels = [possible_label for possible_label in possible_labels if excluded_term != possible_label]

        strong_exclusions = ["Wikimedia", "MediaWiki"]
        for excluded_term in strong_exclusions:
            possible_labels = [possible_label for possible_label in possible_labels if excluded_term not in possible_label]

        if not possible_labels:
            continue
        else:
            label = " | ".join(tuple(set(possible_labels)))

            if label not in label2labelID:
                label2labelID[label] = len(label2labelID)
                entityID2labelID[entity_id] = len(label2labelID) - 1
            else:
                entityID2labelID[entity_id] = label2labelID.get(label)

    with open(src.ENTITY_DIR / "entityID2labelID.json", "w") as f:
        json.dump(entityID2labelID, f)

    labelID2label = {value: key for key, value in label2labelID.items()}
    with open(src.ENTITY_DIR / "labelID2label.json", "w") as f:
        json.dump(labelID2label, f)
