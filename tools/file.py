import json
from pprint import pprint
from typing import Dict, List, Union


def load_json(path: str) -> Union[Dict, List[Dict]]:
    """Load a json file from path

    Args:
        path (str): relative or absolute path to json file

    Returns:
        Union[Dict, List[Dict]]: metadata loaded from file
    """
    data = json.load(open(path, "r", encoding="UTF-8"))

    print("Number of key-value pairs:", len(data))
    print("Type of data:", type(data))

    if isinstance(data, dict):
        first_pair = next(iter(data.items()))
        print(f"First Pair:")
        key, val = first_pair
        print("Key:", key)
        print("Value:")
        pprint(val)
    elif isinstance(data, list):
        print("First element:")
        pprint(data[0])

    return data
