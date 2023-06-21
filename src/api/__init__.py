from .wikidata import retrieve_entity2instance, retrieve_short_descriptions
from .zelda import download_zelda, extract_zelda
from .dbpedia import convert_wikipage_to_wikidata

__all__ = ["retrieve_entity2instance", "retrieve_short_descriptions", "download_zelda", "extract_zelda", "convert_wikipage_to_wikidata"]