import json
import requests
from tqdm import tqdm

import src


def retrieve_entity2instance():
    with open(src.ENTITY_DIR / "entities.json", "r") as f:
        wikidata_entities = json.load(f)

    # define the SPARQL endpoint
    endpoint_url = "https://query.wikidata.org/sparql"

    # max header size
    chunk_size = 200

    headers = {
        "User-Agent": "ner4all/1.0",
        "Accept": "application/sparql-results+json"
    }

    entityID2instanceID = {}
    instanceID2subclassID = {}
    instanceID2label = {}
    subclassID2label = {}

    for i in tqdm(range(0, len(wikidata_entities), chunk_size)):
        chunk = wikidata_entities[i:i + chunk_size]

        query = f"""
        SELECT DISTINCT ?entity ?instance ?instanceLabel ?subclass ?subclassLabel WHERE {{
          VALUES ?entity {{{" ".join(["wd:" + entity for entity in chunk])}}}
          ?entity wdt:P31 ?instance.
          OPTIONAL {{?instance wdt:P279 ?subclass.}}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        # set the query string
        response = requests.get(endpoint_url, params={"query": query}, headers=headers)

        # save the results
        if response.status_code == 200:
            data = json.loads(response.text)
            temp_e2i = {}
            temp_i2s = {}
            for result in data["results"]["bindings"]:
                entity_id = result["entity"]["value"].split("/")[-1]
                instance_id = result["instance"]["value"].split("/")[-1]
                instance_label = result["instanceLabel"]["value"]
                subclass_id = result["subclass"]["value"].split("/")[-1] if "subclass" in result else None
                subclass_label = result["subclassLabel"]["value"] if "subclassLabel" in result else None

                temp_e2i.setdefault(entity_id, set()).add(instance_id)
                temp_i2s.setdefault(instance_id, set()).add(subclass_id)

                if instance_id not in instanceID2label:
                    instanceID2label[instance_id] = instance_label

                if subclass_id not in subclassID2label and subclass_id is not None:
                    subclassID2label[subclass_id] = subclass_label

            entityID2instanceID = src.utils.data.merge_dicts(entityID2instanceID, temp_e2i)
            instanceID2subclassID = src.utils.data.merge_dicts(instanceID2subclassID, temp_i2s)

    entityID2instanceID = {key: list(value) for key, value in entityID2instanceID.items()}
    with open(src.ENTITY_DIR / f"entityID2instanceID.json", "w") as f:
        json.dump(entityID2instanceID, f)

    instanceID2subclassID = {key: list(value) for key, value in instanceID2subclassID.items()}
    with open(src.ENTITY_DIR / f"instanceID2subclassID.json", "w") as f:
        json.dump(instanceID2subclassID, f)

    with open(src.ENTITY_DIR / f"instanceID2label.json", "w") as f:
        json.dump(instanceID2label, f)

    with open(src.ENTITY_DIR / f"subclassID2label.json", "w") as f:
        json.dump(subclassID2label, f)


def retrieve_short_descriptions():
    with open(src.ENTITY_DIR / "entities.json", "r") as f:
        wikidata_entities = json.load(f)

    # Set the endpoint URL for the Wikidata SPARQL service
    url = 'https://query.wikidata.org/sparql'
    # Set the request headers (optional)
    headers = {
        "User-Agent": "ner4all/1.0",
        "Accept": "application/sparql-results+json"
    }

    chunk_size = 200
    short_descriptions = {}

    for i in tqdm(range(0, len(wikidata_entities), chunk_size)):
        chunk = wikidata_entities[i:i + chunk_size]

        # Construct the SPARQL query using an f-string
        query = f'''
        SELECT ?entity ?shortDescription WHERE {{
          VALUES ?entity {{ {" ".join(["wd:" + entity for entity in chunk])} }}
          ?entity schema:description ?shortDescription .
          FILTER(LANG(?shortDescription) = "en")
        }}
        '''
        # Set the request parameters
        params = {
            'query': query
        }
        # Send the HTTP GET request
        response = requests.get(url, headers=headers, params=params)
        # Parse the JSON response
        data = response.json()
        # Extract the short descriptions from the response
        bindings = data['results']['bindings']
        for binding in bindings:
            entity = binding['entity']['value']
            short_description = binding['shortDescription']['value']
            if short_description not in ["Wikimedia disambiguation page", "Wikimedia list article"]:
                short_descriptions[entity.split("/")[-1]] = short_description

    with open(src.ENTITY_DIR / f"entityID2shortdescription.json", "w") as f:
        json.dump(short_descriptions, f)
