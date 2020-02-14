import json
import pickle
import numpy as np
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


def save_json_file(new_dict: list, filepath: Path) -> dict:
    """loads json file"""

    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(new_dict, fp)
    return 



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


def build_qid_class_dicts(all_docs_kb: list, logger) -> list:
    """Builds two dictionaries making a reference questionId <-> Class label

    Parameters
    ----------
    all_docs_kb: list
        list of dictionaries containing the KB
    logger: logging.getLogger
        logger on DEBUG level

    Returns
    -------
    qid_to_class, class_to_qid: dict, dict
         dictionaries to look up correpondance qid <-> class
    """

    # Class labels go from 0 to number of entries in the database
    qid_to_class = {entry['questionId']: count for count,
                    entry in enumerate(all_docs_kb)}
    class_to_qid = {value: key for key, value in qid_to_class.items()}

    return qid_to_class, class_to_qid

def initialize_use_model():
    """
    Initializes universal sentence encoder via tf_hub

    hub_module
    needs to download the hub_module at first
    is saved in cache:
    /var/folders/4n/qmjbytzd0pbc0n7bpp9wh3dc0000gs/T/tfhub_modules

    https://www.tensorflow.org/hub/api_docs/python/hub/load

    Returns
    -------
    A trackable object (see tf.saved_model.load() documentation for details)
    """

    embed = hub.load(HUB_MODEL)
    return embed


def initialize_nlp_model():
    """
    Initializes nlp object (fr_core_news_md)
    https://spacy.io/models/fr

    The nlp object is a processing pipeline
    Model must be downloaded and stored at first
    Consists of (Tokenizer, tagger, parser )

    Returns
    -------
    nlp_model
    """

    nlp = spacy.load(SPACY_MODEL)
    return nlp


def encode_ids_and_scores(row, num_of_classes):
    scores = np.zeros(num_of_classes)
    if row[0]:
        for idx, score in zip(row[0], row[1]):
            scores[idx] = score
    return scores


def encode_responses(bm25_response, use_response, num_of_classes, qid_to_class):
    scores_bm25 = np.zeros(num_of_classes)
    scores_use = np.zeros(num_of_classes)

    for entry in bm25_response:
        pos = qid_to_class[entry["questionId"]]
        score = entry["score"]
        scores_bm25[pos] = score

    for entry in use_response:
        pos = qid_to_class[entry["questionId"]]
        score = entry["score"]
        scores_use[pos] = score
    return np.concatenate((scores_bm25, scores_use))
