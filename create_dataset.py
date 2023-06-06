import os
import src
import logging
import argparse

logger = logging.getLogger("logger")


def create_ner_dataset(args):
    if not os.path.exists(src.DATA_DIR):
        os.mkdir(src.DATA_DIR)

    if args.download_data:
        if args.use_datasets not in ["trex", "zelda", "all"]:
            raise Exception("Please set --use_datasets to either 'trex', 'zelda' or 'all'.")

        if args.use_datasets in ["trex", "all"] and not os.path.exists(src.DATA_DIR / "trex"):
            os.mkdir(src.DATA_DIR / "trex")
            logger.info("Download T-REx dataset...")
            src.api.download_trex()
            logger.info("Done.")

            logger.info("Extracing T-REx dataset...")
            src.api.extract_trex()
            logger.info("Done.")

        if args.use_datasets in ["zelda", "all"] and not os.path.exists(src.DATA_DIR / "zelda"):
            os.mkdir(src.DATA_DIR / "zelda")
            logger.info("Download ZELDA dataset...")
            src.api.download_zelda()
            logger.info("Done.")

            logger.info("Extracing ZELDA dataset...")
            src.api.extract_zelda()
            logger.info("Done.")

    elif not any([os.path.exists(src.DATA_DIR / "trex"), os.path.exists(src.DATA_DIR / "zelda")]):
        raise Exception("Datasets missing. Please set --download_data to True.")

    if not os.path.exists(src.ENTITY_DIR):
        os.mkdir(src.ENTITY_DIR)

    if ("trex_entities.json" not in os.listdir(src.ENTITY_DIR) and args.use_datasets in ["trex", "all"]) or \
        ("zelda_entities.json" not in os.listdir(src.ENTITY_DIR) and args.use_datasets in ["zelda", "all"]):
        logger.info("Extracting all entites from datasets...")
        src.preprocess.extract_entities(args)
        logger.info("Done.")

        logger.info("Query instance of for entities with SPARQL...")
        src.api.retrieve_entity2instance(args)
        logger.info("Done.")

        logger.info("Query instance of for entities with SPARQL...")
        src.api.retrieve_short_descriptions(args)
        logger.info("Done.")

        logger.info("Quality checks for queried entities...")
        src.preprocess.clean_crawled_entities()
        logger.info("Done.")

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
    parser.add_argument("--output_format", nargs="*", default=["conll", "jsonl", "jsonl_with_offsets"])
    parser.add_argument("--use_datasets", default="all")
    args = parser.parse_args()
    create_ner_dataset(args)
