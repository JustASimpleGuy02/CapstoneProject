import json
import pickle5 as pickle
from pprint import pprint
from typing import Dict, List, Union

import pandas as pd


def load_json(path: str, verbose: bool = False) -> Union[Dict, List[Dict]]:
    """Load a json file from path

    Args:
        path (str): relative or absolute path to json file

    Returns:
        Union[Dict, List[Dict]]: metadata loaded from file
    """
    data = json.load(open(path, "r", encoding="UTF-8"))

    if verbose:
        print("Number of key-value pairs:", len(data))
        print("Type of data:", type(data))

        if isinstance(data, dict):
            print("List of keys:", data.keys())
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


def load_csv(path: str):
    """Load data from a csv file

    Args:
    path (str): relative or absolute path to a csv file

    Returns:
    pd.DataFrame
    """
    df = pd.read_csv(path, index_col=None)
    return df


def load_pkl(path: str):
    """Load content from a pickle file

    Args:
        path (str): path to the pickle file

    Returns:
        content (Union[Dict, List[Dict]]): content to save to pickle file
    """
    with open(path, "rb") as f:
        content = pickle.load(path)
    f.close()

    return content


def to_csv(path: str, df: pd.DataFrame):
    """Save a csv file

    Args:
    path (str): relative or absolute path to a csv file to save
    df (pd.DataFrame): dataframe to save
    """
    df.to_csv(path, index=False)
    return df


def to_pkl(out_path: str, content: Union[Dict, List[Dict]]):
    """Save content data to a pickle file

    Args:
        content (Union[Dict, List[Dict]]): content to save to pickle file

    Returns:
        path (str): output file path
    """
    with open(out_path, "wb") as f:
        pickle.dump(content, f)
    f.close()
