import requests
import json
from tqdm import tqdm


def convert_wikipage_to_wikidata(wikipage_ids: list):
    url = 'https://dbpedia.org/sparql'
    headers = {'Accept': 'application/json'}
    results = []

    chunk_size = 500

    # Iterate over wikipage IDs and retrieve wikidata ID for each
    for i in tqdm(range(0, len(wikipage_ids), chunk_size), desc="Converting wikipage IDs to wikidata IDs"):
        chunk = wikipage_ids[i:i + chunk_size]

        query = f"""
        SELECT ?wikipageID (MIN(xsd:integer(REPLACE(STR(?wikidata), "http://www.wikidata.org/entity/Q", ""))) AS ?minEntity) (CONCAT("Q", MIN(xsd:integer(REPLACE(STR(?wikidata), "http://www.wikidata.org/entity/Q", "")))) AS ?wikidataID)
        WHERE {{
            VALUES ?wikipageID {{{" ".join([str(_) for _ in chunk])}}}
            ?id dbo:wikiPageID ?wikipageID ;
            owl:sameAs ?wikidata .
        FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
        }}
        GROUP BY ?wikipageID
        """
        r = requests.get(url, headers=headers, params={'query': query})
        data = json.loads(r.text)
        _wikipage_ids = [_['wikipageID']['value'] for _ in data['results']['bindings']]
        wikidata_ids = [_['wikidataID']['value'].split("/")[-1] for _ in data['results']['bindings']]
        results.extend(zip(_wikipage_ids, wikidata_ids))

    return results
