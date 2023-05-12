from pathlib import Path

from . import (
    preprocess,
    api,
    store,
    utils,
    dataset_builder,
)

import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"logger": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    }
)

logger = logging.getLogger("logger")

DATA_DIR = Path("data")
ENTITY_DIR = Path("entities")

NER_DATASET_DIR = Path("ner_dataset")

__all__ = [
    "DATA_DIR",
    "ENTITY_DIR",
    "logger",
    "preprocess",
    "api",
    "store",
    "utils",
    "dataset_builder",
]
