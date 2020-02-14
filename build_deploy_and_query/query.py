import argparse
import logging
import configparser
import pickle
import copy
import shutil
import operator
import os
import numpy as np
import pandas as pd
import pathlib

from argparse import RawTextHelpFormatter
from pathlib import Path
import time
from datetime import datetime
from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from elasticsearch6 import Elasticsearch

from definitions import (
    ROOT,
    CLIENT,
    TRAINING_DATA_FILE,
    INDEX_NAME,
    SPACY_MODEL,
    HUB_MODEL,
    DATA_KB_WITH_VECTORS_FILE,
    MAPPING_JSON_FILE,
    BULK_REQUEST_JSON_MODEL,
)

from utils import load_pickle_file, load_json_file, encode_responses

from index_elasticsearch import index_kb
from make_query_bm25 import make_query_bm25

from build_use import initialize_use_model
from make_query_use import make_query_use

def load_model(KB_id):
    """loads all necessary Data"""

    KB_id = 1
    path_live = Path(ROOT) / "models" / f"KB_id_{KB_id}" / "live"

    # get the configuration
    config = configparser.ConfigParser()
    config.read(path_live / "config.cfg")
    config_set = "DEFAULT"
    without_stopwords = config[config_set].getboolean("without_stopwords")
    num_of_sentences = config[config_set].getint("num_of_sentences")
    all_docs_kb_filepath = path_live / config[config_set]["all_docs_kb_filename"]
    
    # load model and scaler
    model = load(path_live / "logreg_model.joblib")
    scaler = load_pickle_file(path_live / "std_scaler.pkl")

    # load the kb that the model was built on
    logger.info("# load the kb that the model was built on")
    all_docs_kb = load_json_file(all_docs_kb_filepath, logger)

    # load kb with vectors
    data_kb_with_vectors = load_pickle_file(path_live / DATA_KB_WITH_VECTORS_FILE)
    
    # load the two dictionaries
    qid_to_class, class_to_qid = (
        load_pickle_file(path_live / "qid_to_class.pkl"),
        load_pickle_file(path_live / "class_to_qid.pkl"),
    )

    return (
        model,
        scaler,
        all_docs_kb,
        data_kb_with_vectors,
        qid_to_class,
        class_to_qid,
        without_stopwords,
        num_of_sentences,
    )

def main(args: argparse.ArgumentParser(), logger: logging.getLogger()):
    """asdf und nochmal asdf"""

    # load_live_model
    (
        model,
        scaler,
        all_docs_kb,
        data_kb_with_vectors,
        qid_to_class,
        class_to_qid,
        without_stopwords,
        num_of_sentences,
    ) = load_model(args.KB_id)

    # set args parameters
    args.without_stopwords = without_stopwords
    args.num_of_sentences = num_of_sentences
    
    # load use and bm25-elasticsearch
    logger.info("# load use and bm25-elasticsearch")
    embed = initialize_use_model()
    client = Elasticsearch()

    # execute main index_elasticsearch
    logger.info("# execute main index_elasticsearch")
    index_kb(all_docs_kb, client, args, logger)

    
    while True:
        query = input("enter your query or q: ")
        if query == "q":
            print("Exiting")
            break
        args.query = query

        start = time.time()
        use_response = make_query_use(data_kb_with_vectors, embed, args, logger)
        end = time.time()
        print(end - start)
        
        bm25_response = make_query_bm25(client, args)


        response_vector_level_1 = encode_responses(bm25_response, use_response, len(model.classes_), qid_to_class).reshape(1, -1)

        scaler.transform(response_vector_level_1)
        response_vector_level_2 = model.predict_log_proba(response_vector_level_1)


        response_vector_level_2 = np.argsort(response_vector_level_2)[::-1]
        response = [class_to_qid[pos] for pos in response_vector_level_2[0]]


        
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger()

    # Argument parsing
    logger.info("# Argument parsing")
    parser = argparse.ArgumentParser(
        description=(
            "Input: query string"
            "Output: json of the form {questionId: ..., score: ...}"
        ),
        formatter_class=RawTextHelpFormatter,
    )

    args = parser.add_argument(
        "KB_id", type=int, help="KB_id integer identifying the KB"
    )

    args = parser.parse_args()
    main(args, logger)
