def merge_dicts(dict1, dict2):
    merged_dict = dict(dict1)  # create a copy of dict1 to avoid modifying it
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key].update(value)
        else:
            merged_dict[key] = value
    return merged_dict


def to_range(list1: list):
    assert len(list1) == 2
    return range(list1[0], list1[1]+1)


def is_range_within(sentence_range, span_range):
    return sentence_range.start <= span_range.start and sentence_range.stop >= span_range.stop
