def merge_dicts(dict1, dict2):
    merged_dict = dict(dict1)  # create a copy of dict1 to avoid modifying it
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key].update(value)
        else:
            merged_dict[key] = value
    return merged_dict
