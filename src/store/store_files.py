import os
import json

max_file_size = 1e8  # maximum file size in bytes
json_file_index = 0
conll_file_index = 0

data_point_index = 0


def counter():
    global data_point_index
    data_point_index += 1
    return data_point_index


def write_to_file(data, output_format):
    global json_file_index
    global conll_file_index

    # create new file if file size exceeds maximum
    if output_format == "conll":
        if os.path.isfile(f"ner_dataset/{output_format}/part_{conll_file_index}.{output_format}") and os.path.getsize(f"ner_dataset/{output_format}/part_{conll_file_index}.{output_format}") >= max_file_size:
            conll_file_index += 1

        with open(f"ner_dataset/{output_format}/part_{conll_file_index}.{output_format}", "a") as f:
            f.write(data + "\n")

    elif output_format == "jsonl":
        if os.path.isfile(f"ner_dataset/{output_format}/part_{json_file_index}.{output_format}") and os.path.getsize(
                f"ner_dataset/{output_format}/part_{json_file_index}.{output_format}") >= max_file_size:
            json_file_index += 1

        with open(f"ner_dataset/{output_format}/part_{json_file_index}.{output_format}", "a") as f:
            f.write(json.dumps(data) + "\n")
