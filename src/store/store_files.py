import os
import json

max_file_size = 1e8  # maximum file size in bytes
jsonl_file_index = 0
offset_file_index = 0
conll_file_index = 0
jsonl_bio_file_index = 0

jsonl_data_point_index = 0
jsonl_offset_data_point_index = 0
jsonl_bio_data_point_index = 0


def jsonl_counter():
    global jsonl_data_point_index
    jsonl_data_point_index += 1
    return jsonl_data_point_index


def jsonl_offset_counter():
    global jsonl_offset_data_point_index
    jsonl_offset_data_point_index += 1
    return jsonl_offset_data_point_index


def jsonl_bio_counter():
    global jsonl_bio_data_point_index
    jsonl_bio_data_point_index += 1
    return jsonl_bio_data_point_index


def write_to_file(data_generator, output_format):
    global jsonl_file_index
    global conll_file_index
    global offset_file_index
    global jsonl_bio_file_index

    # create new file if file size exceeds maximum
    if output_format == "conll":
        if os.path.isfile(f"ner_dataset/conll/part_{conll_file_index}.conll") and os.path.getsize(f"ner_dataset/conll/part_{conll_file_index}.conll") >= max_file_size:
            conll_file_index += 1

        with open(f"ner_dataset/conll/part_{conll_file_index}.conll", "a") as f:
            for data_point in data_generator:
                f.write(data_point + "\n")

    elif output_format == "jsonl":
        if os.path.isfile(f"ner_dataset/jsonl/part_{jsonl_file_index}.jsonl") and os.path.getsize(
                f"ner_dataset/jsonl/part_{jsonl_file_index}.jsonl") >= max_file_size:
            jsonl_file_index += 1

        with open(f"ner_dataset/jsonl/part_{jsonl_file_index}.jsonl", "a") as f:
            for data_point in data_generator:
                f.write(json.dumps(data_point) + "\n")

    elif output_format == "jsonl_bio":
        if os.path.isfile(f"ner_dataset/jsonl_bio/part_{jsonl_bio_file_index}.jsonl") and os.path.getsize(
                f"ner_dataset/jsonl_bio/part_{jsonl_bio_file_index}.jsonl") >= max_file_size:
            jsonl_bio_file_index += 1

        with open(f"ner_dataset/jsonl_bio/part_{jsonl_bio_file_index}.jsonl", "a") as f:
            for data_point in data_generator:
                f.write(json.dumps(data_point) + "\n")

    elif output_format == "jsonl_with_offsets":
        if os.path.isfile(f"ner_dataset/jsonl_with_offsets/part_{offset_file_index}.jsonl") and os.path.getsize(
                f"ner_dataset/jsonl_with_offsets/part_{offset_file_index}.jsonl") >= max_file_size:
            offset_file_index += 1

        with open(f"ner_dataset/jsonl_with_offsets/part_{offset_file_index}.jsonl", "a") as f:
            for data_point in data_generator:
                f.write(json.dumps(data_point) + "\n")
