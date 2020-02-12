import json
import pickle
from pathlib import Path


def load_json_file(filepath: Path, logger) -> dict:
    """loads json file"""

    try:
        with open(filepath, 'r') as fp:
            all_docs_kb = json.load(fp)
    except FileNotFoundError as err:
        logger.error(err)
        raise
    return all_docs_kb


def load_pickle_file(filepath: Path) -> dict:
    """loads pkl file from filepath"""
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    return data


def save_pickle_dict(new_dict: list, filepath: Path) -> int:
    """
    Saves a list of dictionaries as a pickle file
    https://stackoverflow.com/questions/7100125/storing-python-dictionaries
    """

    with open(filepath, 'wb') as fp:
        pickle.dump(new_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return 1
