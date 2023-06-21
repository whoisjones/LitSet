import os
import src
import logging
import argparse

logger = logging.getLogger("logger")


def create_ner_dataset(args):
    if not os.path.exists(src.DATA_DIR):
        os.mkdir(src.DATA_DIR)

    if args.download_data:
        if not os.path.exists(src.DATA_DIR / "zelda"):
            os.mkdir(src.DATA_DIR / "zelda")
            logger.info("Download ZELDA dataset...")
            src.api.download_zelda()
            logger.info("Done.")

            logger.info("Extracing ZELDA dataset...")
            src.api.extract_zelda()
            logger.info("Done.")

    elif not os.path.exists(src.DATA_DIR / "zelda"):
        raise Exception("ZELDA dataset missing. Please set --download_data to True.")

    if not os.path.exists(src.ENTITY_DIR):
        os.mkdir(src.ENTITY_DIR)

    if "entities.json" not in os.listdir(src.ENTITY_DIR):
        logger.info("Extracting all entites from datasets...")
        src.preprocess.extract_entities()
        logger.info("Done.")

    qa_required = False
    if "entityID2instanceID.json" not in os.listdir(src.ENTITY_DIR):
        logger.info("Query instance of for entities with SPARQL...")
        src.api.retrieve_entity2instance()
        logger.info("Done.")
        qa_required = True

    if "entityID2shortdescriptions.json" not in os.listdir(src.ENTITY_DIR):
        logger.info("Query instance of for entities with SPARQL...")
        src.api.retrieve_short_descriptions()
        logger.info("Done.")
        qa_required = True

    if qa_required:
        logger.info("Quality checks for queried entities...")
        src.preprocess.clean_crawled_entities()
        logger.info("Done.")

    if not os.path.exists(src.ENTITY_DIR / "entityID2labelID.json") and not os.path.exists(src.ENTITY_DIR / "labelID2label.json"):
        logger.info("Creating final labels...")
        src.preprocess.create_final_annotations()
        logger.info("Done.")

    if not os.path.exists(src.NER_DATASET_DIR):
        os.mkdir(src.NER_DATASET_DIR)

    for output_format in args.output_format:
        if not os.path.exists(src.NER_DATASET_DIR / output_format):
            os.mkdir(src.NER_DATASET_DIR / output_format)

    logger.info(f"Create NER dataset in {args.output_format}-format...")
    src.dataset_builder.build_NER(args.output_format)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--output_format", nargs="*", default=["jsonl_bio"])
    args = parser.parse_args()
    create_ner_dataset(args)
