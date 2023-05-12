from .wikidata import retrieve_entity2instance
from .trex import download_trex, extract_trex
from .zelda import download_zelda, extract_zelda
from .dbpedia import convert_wikipage_to_wikidata

__all__ = ["retrieve_entity2instance", "download_trex", "extract_trex", "download_zelda", "extract_zelda", "convert_wikipage_to_wikidata"]