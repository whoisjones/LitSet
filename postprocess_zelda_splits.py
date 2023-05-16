import json
from tqdm import tqdm


def main():
    # Define the maximum size limit for each split in bytes (e.g., 100MB)
    split_size_limit = 100 * 1024 * 1024
    # input_lines = open("ner_dataset/conll/part_34.conll", "r", encoding="utf-8")
    with open("ner_dataset/conll/part_34.conll", "r") as f:
        input_lines = f.readlines()

    split_counter = 1
    current_size = 0
    idx = 0
    splits = []
    for record in tqdm(input_lines):
        record_size = len(record.encode("utf-8"))
        # Check if adding the current record exceeds the size limit
        if current_size + record_size > split_size_limit:
            if record == "\n":
                current_size = 0
                split_counter += 1
                idx += 1
                splits.append(idx)
                continue
        current_size += record_size

        idx += 1

    for idx, i in enumerate(splits):
        if idx == 0:
            # Write the current split to a separate file
            with open(f'ner_dataset/conll/output_34_{idx + 1}.conll', 'a') as outfile:
                outfile.write("".join(input_lines[:i]))

        elif idx == len(splits):
            with open(f'ner_dataset/conll/output_34_{idx + 1}.conll', 'a') as outfile:
                outfile.write("".join(input_lines[i:]))

        else:
            with open(f'ner_dataset/conll/output_34_{idx + 1}.conll', 'a') as outfile:
                outfile.write("".join(input_lines[splits[idx - 1]:i]))

    return idx + 1


def append(idx):
    input_lines = open("ner_dataset/conll/part_35.conll", "r", encoding="utf-8")
    for record in tqdm(input_lines):
        with open(f'ner_dataset/conll/output_34_{idx}.conll', 'a') as outfile:
            outfile.write(record)


def test(idx):
    input_lines = open("ner_dataset/conll/part_34.conll", "r", encoding="utf-8")
    l1 = 0
    for record in input_lines:
        if record == "\n":
            l1 += 1

    input_lines = open("ner_dataset/conll/part_35.conll", "r", encoding="utf-8")
    for record in input_lines:
        if record == "\n":
            l1 += 1

    l2 = 0
    for i in range(1, idx + 1):
        input_lines = open(f"ner_dataset/conll/output_34_{i}.conll", "r", encoding="utf-8")
        for record in input_lines:
            if record == "\n":
                l2 += 1

    print(l1, l2)


if __name__ == "__main__":
    idx = main()
    append(idx)
    test(idx)
