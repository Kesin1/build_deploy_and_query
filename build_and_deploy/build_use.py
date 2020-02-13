"""Extract title and body, process phrases, add vectors, save as pickle

This script is doing the following. It first initializes a spacy nlp model
in french and a hub_module, notably the multilingual Sentence Encoder
Then it loads a json file containing the KB of a client in json format.
For each json entry it requires to have the following keys:
{
"question_id": int;
"title": str;
"responses": list of dict,
...
}

The dictionary in the list of "responses" needs to contain the key "body": str

The script then builds a new dictionary with the above mentioned keys and
items and adds vectors infered from m-USE.
It then saves this dictionary as a pickle file in the file folder.

This script requires to have installed
spacy==2.2
tensorflow>=2.0.0
tensorflow_hub>=0.6.0
tensorflow_text>=2.0.0rc0
"""

import argparse
import logging
import json
import pickle
import copy
import spacy
# import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from argparse import RawTextHelpFormatter
from pathlib import Path
from definitions import SPACY_MODEL, HUB_MODEL, DATA_KB_WITH_VECTORS_FILE

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


def load_json_file(filepath: Path, logger) -> dict:
    """loads json file"""

    try:
        with open(filepath, 'r') as fp:
            all_docs_kb = json.load(fp)
    except FileNotFoundError as err:
        logger.error(err)
        raise
    return all_docs_kb


def save_dict(new_dict: list, filepath: Path) -> int:
    """
    Saves a dictionary as a pickle file
    https://stackoverflow.com/questions/7100125/storing-python-dictionaries
    """
    
    with open(filepath, 'wb') as fp:
        pickle.dump(new_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return 1


def remove_stopwords_punct(phrase: str, nlp) -> str:
    """Removes stopwords and punctuation"""

    doc = nlp(phrase)
    return " ".join([token.text for token in doc
                     if not (token.is_stop or token.is_punct)])


def nlp_parsing(all_docs_kb: list, nlp: spacy.load) -> list:
    """
    Parses the entries in all_docs_kb and saves them in a new list

    Parameters
    ----------
    all_docs_kb: list
        list of dictionaries where each dictionary entry
        represents an entry in the kb
    nlp: spacy processing pipeline
        nlp model loaded via spacy.load()

    Returns
    -------
    list
        list of dictionaries where each dictionary entry represents an entry
        in the kb
        each entry has the keys
        questionId(int), title(str), title_wo_stopwords(str),
        body(list of str), body_wo_stopwords(list of str)
    """

    new_docs = []
    for entry in all_docs_kb:
        new_dict = {}
        new_dict["questionId"] = entry["questionId"]
        new_dict["title"] = entry["title"]
        new_dict["title_wo_stopwords"] = remove_stopwords_punct(
            new_dict["title"], nlp)

        # process body entirely, since we need to seperate the sentences anyhow
        body_text = entry["responses"][0]["body"]
        new_dict["body"] = [sent.text for sent in nlp(body_text).sents]
        new_dict["body_wo_stopwords"] = [remove_stopwords_punct(sent, nlp)
                                         for sent in new_dict["body"]]

        new_docs.append(copy.deepcopy(new_dict))
    return new_docs


def adding_vectors(new_docs: list, embed: hub.module) -> list:
    """
    Adds vectors to the entries in new_docs using m-USE

    Parameters
    ----------
    new_docs: list
        list of dictionaries where each dictionary entry
        represents an entry in the kb
    nlp: spacy processing pipeline
        nlp model loaded via spacy.load()

    Returns
    -------
    list
        list of dictionaries where each dictionary entry represents an entry
        in the kb
        each entry has the keys
        questionId: int,
        title: 
            dict {'text': , 'vector': },
        title_wo_stopwords: 
            dict {'text': , 'vector': },
        body: 
            list of dict, 
        body_wo_stopwords: 
            list of dict
        dict in body: 
            {'text': , 'vector': }
    """

    new_docs_with_vectors = []
    for entry in new_docs:
        new_dict = {}
        new_dict["questionId"] = entry["questionId"]

        new_dict["title"] = {
            "text": entry["title"],
            "vector": embed(entry["title"]).numpy()
        }
        new_dict["title_wo_stopwords"] = {
            "text": entry["title_wo_stopwords"],
            "vector": embed(entry["title_wo_stopwords"]).numpy()
        }

        new_dict["body"] = [
            {
                "text": text,
                "vector": embed(text).numpy()
            }
            for text in entry["body"]
        ]

        new_dict["body_wo_stopwords"] = [
            {
                "text": text,
                "vector": embed(text).numpy()
            }
            for text in entry["body_wo_stopwords"]
        ]

        new_docs_with_vectors.append(copy.deepcopy(new_dict))
    return new_docs_with_vectors


def build_use(embed: hub.Module, nlp: spacy.load, args: argparse.ArgumentParser(), logger: logging.getLogger()) -> int:
    """
    Initializes the models (spacy and USE) and takes the steps mentioned in 
    the docstring of this script

    Parameters
    ----------
    args: argparse.ArgumentParser
        The command line arguments given
    logger: logging.getLogger
        logger on DEBUG level

    Returns
    -------
    None
    """

    # load the data
    logger.info("# load the data")
    all_docs_kb = load_json_file(args.filepath_json, logger)

    # check for right format and types
    # logger.info("# check for right format and types")
    # check_entries(all_docs_kb, logger)

    # nlp parsing
    logger.info("# nlp parsing")
    new_docs = nlp_parsing(all_docs_kb, nlp)

    # adding vectors
    logger.info("# adding vectors")
    new_docs_with_vectors = adding_vectors(new_docs, embed)

    # # saves new dictionary in a pickle file
    # logger.info("# saves new dictionary in a pickle file")
    # save_dict(new_docs_with_vectors, Path(DATA_KB_WITH_VECTORS_FILE))

    return new_docs_with_vectors


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    # Argument parsing
    logger.info("# Argument parsing")
    parser = argparse.ArgumentParser(
        description=(
            "Input: .json file containing the KB\n" \
            "Output: None\n" \
            "Sideeffects: Creates data_kb_with_vectors with name set in settings"
        ), formatter_class=RawTextHelpFormatter)
    args = parser.add_argument(
        "filepath_json", type=Path,
        help="File containing the KB export in json format")
    args = parser.parse_args()

    main(args, logger)
