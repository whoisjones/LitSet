import random
from typing import Union
from pathlib import Path
import glob
import json

from torch.utils.data.dataset import Subset

from flair.datasets import CONLL_03, FEWNERD, ONTONOTES, WNUT_17
from flair.datasets.sequence_labeling import MultiFileColumnCorpus


class NER4ALL_ZELDA(MultiFileColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path],
        in_memory: bool = False,
        **corpusargs,
    ) -> None:
        if type(base_path) is str:
            base_path = Path(base_path)

        dataset_path = Path(base_path) / "conll"

        train_files = glob.glob(str(dataset_path / "*.conll"))

        with open(base_path / "labelID2label.json", "r") as f:
            label_mapping = json.load(f)

        super().__init__(
            train_files=train_files,
            column_format={0: "text", 1: "ner"},
            in_memory=in_memory,
            sample_missing_splits=False,
            label_name_map=label_mapping,
            **corpusargs,
        )


def get_corpus(corpus: str, fewnerd_granularity: str = "", base_path: str = "data"):
    """
    corpus: str = either 'wnut_17', 'conll_03', 'ontonotes'
    fewnerd_granularity: str = if corpus == 'fewnerd', fewnerd_granularity can either bei 'fine', 'coarse' or 'coarse-fine'
    """
    if corpus == "wnut_17":
        return WNUT_17(
            label_name_map={
                "corporation": "corporation",
                "creative-work": "creative work",
                "group": "group",
                "location": "location",
                "person": "person",
                "product": "product",
            }
        )
    elif corpus == "conll_03":
        dataset = CONLL_03(
            base_path=base_path,
            column_format={0: "text", 1: "pos", 2: "chunk", 3: "ner"},
            label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
        )
        valid_indices = []
        # We need to filter them out in order to ensure we use the same IDs from the corpora as on HF datasets
        for idx, sentence in enumerate(dataset.train):
            if "DOCSTART" not in sentence.text:
                valid_indices.append(idx)
        dataset._train = Subset(dataset._train, valid_indices)
        return dataset
    elif corpus == "ontonotes":
        return ONTONOTES(
            label_name_map={
                "CARDINAL": "cardinal",
                "DATE": "date",
                "EVENT": "event",
                "FAC": "facility",
                "GPE": "geographical social political entity",
                "LANGUAGE": "language",
                "LAW": "law",
                "LOC": "location",
                "MONEY": "money",
                "NORP": "nationality religion political",
                "ORDINAL": "ordinal",
                "ORG": "organization",
                "PERCENT": "percent",
                "PERSON": "person",
                "PRODUCT": "product",
                "QUANTITY": "quantity",
                "TIME": "time",
                "WORK_OF_ART": "work of art",
            }
        )
    elif corpus == "fewnerd":
        return FEWNERD(label_name_map=get_fewnerd_label_map(fewnerd_granularity))
    elif corpus == "ner4all":
        return NER4ALL_ZELDA(base_path=base_path)
    else:
        raise Exception("no valid corpus.")


def get_fewnerd_label_map(fewnerd_granularity: str):
    if fewnerd_granularity == "fine":
        return {
            "location-GPE": "geographical social political entity",
            "person-other": "other person",
            "organization-other": "other organization",
            "organization-company": "company",
            "person-artist/author": "author artist",
            "person-athlete": "athlete",
            "person-politician": "politician",
            "building-other": "other building",
            "organization-sportsteam": "sportsteam",
            "organization-education": "eduction",
            "location-other": "other location",
            "other-biologything": "biology",
            "location-road/railway/highway/transit": "road railway highway transit",
            "person-actor": "actor",
            "product-other": "other product",
            "event-sportsevent": "sportsevent",
            "organization-government/governmentagency": "government agency",
            "location-bodiesofwater": "bodies of water",
            "organization-media/newspaper": "media newspaper",
            "art-music": "music",
            "other-chemicalthing": "chemical",
            "event-attack/battle/war/militaryconflict": "attack war battle military conflict",
            "organization-politicalparty": "political party",
            "art-writtenart": "written art",
            "other-award": "award",
            "other-livingthing": "living thing",
            "event-other": "other event",
            "art-film": "film",
            "product-software": "software",
            "organization-sportsleague": "sportsleague",
            "other-language": "language",
            "other-disease": "disease",
            "organization-showorganization": "show organization",
            "product-airplane": "airplane",
            "other-astronomything": "astronomy",
            "organization-religion": "religion",
            "product-car": "car",
            "person-scholar": "scholar",
            "other-currency": "currency",
            "person-soldier": "soldier",
            "location-mountain": "mountain",
            "art-broadcastprogram": "broadcastprogram",
            "location-island": "island",
            "art-other": "other art",
            "person-director": "director",
            "product-weapon": "weapon",
            "other-god": "god",
            "building-theater": "theater",
            "other-law": "law",
            "product-food": "food",
            "other-medical": "medical",
            "product-game": "game",
            "location-park": "park",
            "product-ship": "ship",
            "building-sportsfacility": "sportsfacility",
            "other-educationaldegree": "educational degree",
            "building-airport": "airport",
            "building-hospital": "hospital",
            "product-train": "train",
            "building-library": "library",
            "building-hotel": "hotel",
            "building-restaurant": "restaurant",
            "event-disaster": "disaster",
            "event-election": "election",
            "event-protest": "protest",
            "art-painting": "painting",
        }
    elif fewnerd_granularity == "coarse":
        return {
            "location-GPE": "location",
            "person-other": "person",
            "organization-other": "organization",
            "organization-company": "organization",
            "person-artist/author": "person",
            "person-athlete": "person",
            "person-politician": "person",
            "building-other": "building",
            "organization-sportsteam": "organization",
            "organization-education": "organization",
            "location-other": "location",
            "other-biologything": "other",
            "location-road/railway/highway/transit": "location",
            "person-actor": "person",
            "product-other": "product",
            "event-sportsevent": "event",
            "organization-government/governmentagency": "organization",
            "location-bodiesofwater": "location",
            "organization-media/newspaper": "organization",
            "art-music": "art",
            "other-chemicalthing": "other",
            "event-attack/battle/war/militaryconflict": "event",
            "organization-politicalparty": "organization",
            "art-writtenart": "art",
            "other-award": "other",
            "other-livingthing": "other",
            "event-other": "event",
            "art-film": "art",
            "product-software": "product",
            "organization-sportsleague": "organization",
            "other-language": "other",
            "other-disease": "other",
            "organization-showorganization": "organization",
            "product-airplane": "product",
            "other-astronomything": "other",
            "organization-religion": "organization",
            "product-car": "product",
            "person-scholar": "person",
            "other-currency": "other",
            "person-soldier": "person",
            "location-mountain": "location",
            "art-broadcastprogram": "art",
            "location-island": "location",
            "art-other": "art",
            "person-director": "person",
            "product-weapon": "product",
            "other-god": "other",
            "building-theater": "building",
            "other-law": "other",
            "product-food": "product",
            "other-medical": "other",
            "product-game": "product",
            "location-park": "location",
            "product-ship": "product",
            "building-sportsfacility": "building",
            "other-educationaldegree": "other",
            "building-airport": "building",
            "building-hospital": "building",
            "product-train": "product",
            "building-library": "building",
            "building-hotel": "building",
            "building-restaurant": "building",
            "event-disaster": "event",
            "event-election": "event",
            "event-protest": "event",
            "art-painting": "art",
        }
    elif fewnerd_granularity == "coarse-without-misc":
        return {
            "location-GPE": "location",
            "person-other": "person",
            "organization-other": "organization",
            "organization-company": "organization",
            "person-artist/author": "person",
            "person-athlete": "person",
            "person-politician": "person",
            "building-other": "building",
            "organization-sportsteam": "organization",
            "organization-education": "organization",
            "location-other": "location",
            "other-biologything": "biology",
            "location-road/railway/highway/transit": "location",
            "person-actor": "person",
            "product-other": "product",
            "event-sportsevent": "event",
            "organization-government/governmentagency": "organization",
            "location-bodiesofwater": "location",
            "organization-media/newspaper": "organization",
            "art-music": "art",
            "other-chemicalthing": "chemical",
            "event-attack/battle/war/militaryconflict": "event",
            "organization-politicalparty": "organization",
            "art-writtenart": "art",
            "other-award": "award",
            "other-livingthing": "living thing",
            "event-other": "event",
            "art-film": "art",
            "product-software": "product",
            "organization-sportsleague": "organization",
            "other-language": "language",
            "other-disease": "disease",
            "organization-showorganization": "organization",
            "product-airplane": "product",
            "other-astronomything": "astronomy",
            "organization-religion": "organization",
            "product-car": "product",
            "person-scholar": "person",
            "other-currency": "currency",
            "person-soldier": "person",
            "location-mountain": "location",
            "art-broadcastprogram": "art",
            "location-island": "location",
            "art-other": "art",
            "person-director": "person",
            "product-weapon": "product",
            "other-god": "god",
            "building-theater": "building",
            "other-law": "law",
            "product-food": "product",
            "other-medical": "medical",
            "product-game": "product",
            "location-park": "location",
            "product-ship": "product",
            "building-sportsfacility": "building",
            "other-educationaldegree": "educational degree",
            "building-airport": "building",
            "building-hospital": "building",
            "product-train": "product",
            "building-library": "building",
            "building-hotel": "building",
            "building-restaurant": "building",
            "event-disaster": "event",
            "event-election": "event",
            "event-protest": "event",
            "art-painting": "art",
        }
    elif fewnerd_granularity == "coarse-fine":
        return {
            "location-GPE": "location geographical social political entity",
            "person-other": "person other",
            "organization-other": "organization other",
            "organization-company": "organization company",
            "person-artist/author": "person artist author",
            "person-athlete": "person athlete",
            "person-politician": "person politician",
            "building-other": "building other",
            "organization-sportsteam": "organization sportsteam",
            "organization-education": "organization education",
            "location-other": "location other",
            "other-biologything": "other biology",
            "location-road/railway/highway/transit": "location road railway highway transit",
            "person-actor": "person actor",
            "product-other": "product other",
            "event-sportsevent": "event sportsevent",
            "organization-government/governmentagency": "organization government governmentagency",
            "location-bodiesofwater": "location bodies of water",
            "organization-media/newspaper": "organization media newspaper",
            "art-music": "art music",
            "other-chemicalthing": "other chemical",
            "event-attack/battle/war/militaryconflict": "event attack battle war militaryconflict",
            "organization-politicalparty": "organization political party",
            "art-writtenart": "art written art",
            "other-award": "other award",
            "other-livingthing": "other living thing",
            "event-other": "event other",
            "art-film": "art film",
            "product-software": "product software",
            "organization-sportsleague": "organization sportsleague",
            "other-language": "other language",
            "other-disease": "other disease",
            "organization-showorganization": "organization showorganization",
            "product-airplane": "product airplane",
            "other-astronomything": "other astronomy",
            "organization-religion": "organization religion",
            "product-car": "product car",
            "person-scholar": "person scholar",
            "other-currency": "other currency",
            "person-soldier": "person soldier",
            "location-mountain": "location mountain",
            "art-broadcastprogram": "art broadcast program",
            "location-island": "location island",
            "art-other": "art other",
            "person-director": "person director",
            "product-weapon": "product weapon",
            "other-god": "other god",
            "building-theater": "building theater",
            "other-law": "other law",
            "product-food": "product food",
            "other-medical": "other medical",
            "product-game": "product game",
            "location-park": "location park",
            "product-ship": "product ship",
            "building-sportsfacility": "building sportsfacility",
            "other-educationaldegree": "other educational degree",
            "building-airport": "building airport",
            "building-hospital": "building hospital",
            "product-train": "product train",
            "building-library": "building library",
            "building-hotel": "building hotel",
            "building-restaurant": "building restaurant",
            "event-disaster": "event disaster",
            "event-election": "event election",
            "event-protest": "event protest",
            "art-painting": "art painting",
        }


def get_masked_fewnerd_corpus(seed: int = None, fewnerd_granularity: str = "", inverse_mask: bool = False):
    coarse_labels = ["person", "location", "organization", "product", "art", "event", "building", "other"]
    random.seed(seed)
    selected_labels = random.sample(coarse_labels, int(0.5 * len(coarse_labels)))
    not_selected_labels = list(set(coarse_labels) - set(selected_labels))
    if not inverse_mask:
        labels_to_keep = selected_labels
    else:
        labels_to_keep = not_selected_labels
    label_map = get_fewnerd_label_map(fewnerd_granularity)
    masked_label_map = {k: v if k.startswith(tuple(labels_to_keep)) else "O" for k, v in label_map.items()}
    return FEWNERD(label_name_map=masked_label_map), labels_to_keep