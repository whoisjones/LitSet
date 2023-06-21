import os
import json

max_file_size = 1e8  # maximum file size in bytes
jsonl_bio_file_index = 0

jsonl_bio_data_point_index = 0

def jsonl_bio_counter():
    global jsonl_bio_data_point_index
    jsonl_bio_data_point_index += 1
    return jsonl_bio_data_point_index


def write_to_file(data_generator, output_format):
    global jsonl_bio_file_index

    # create new file if file size exceeds maximum file size
    if output_format == "jsonl_bio":
        if os.path.isfile(f"ner_dataset/jsonl_bio/part_{jsonl_bio_file_index}.jsonl") and os.path.getsize(
                f"ner_dataset/jsonl_bio/part_{jsonl_bio_file_index}.jsonl") >= max_file_size:
            jsonl_bio_file_index += 1

        with open(f"ner_dataset/jsonl_bio/part_{jsonl_bio_file_index}.jsonl", "a") as f:
            for data_point in data_generator:
                f.write(json.dumps(data_point) + "\n")
