import argparse
import logging
import configparser
import json
import pickle
import copy
import shutil
import operator
import os
import pandas as pd
import pathlib

from argparse import RawTextHelpFormatter
from pathlib import Path
from datetime import datetime
from joblib import dump, load

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

from utils import (
    load_json_file,
    save_json_file,
    load_pickle_file,
    save_pickle_dict,
    build_qid_class_dicts,
)

from index_elasticsearch import index_kb
from make_query_bm25 import make_query_bm25

from build_use import build_use, initialize_use_model, initialize_nlp_model
from make_query_use import make_query_use

from train_and_save_model import train_and_save_model


def get_data_points(
    all_docs_kb: list, data_kb_with_vectors: list, qid_to_class: dict, embed, client
):
    """
    Iterates over the KB in all_docs_kb

    Parameters
    ----------

    all_docs_kb: list
        list of dictionaries containing the KB entries
    data_kb_with_vector: list
        list of dictionaries containing the KB entries with vectors
    qid_to_class: dict
        dictionary containing reference between questionId - class_label
    client: Elasticsearch()
        Elasticsearch client
    embed: USE hub_module


    Returns
    -------
    ordered list of dictionaries with keys
        ["query", "class_label", "bm25_class_labels", "bm25_scores", 
        "use_class_labels", "use_scores", "questionId", "title", "body"]
    """

    # do the cool stuff
    data_points = []
    for entry in all_docs_kb:
        new_data_point = {}
        new_data_point["questionId"] = entry["questionId"]
        new_data_point["title"] = entry["title"]
        new_data_point["body"] = entry["responses"][0]["body"]
        new_data_point["class_label"] = qid_to_class[entry["questionId"]]

        # iterate over queries inside the entry
        for query in entry["queries"]:
            args.query = query
            new_data_point["query"] = args.query

            # get a response from USE
            use_response = make_query_use(data_kb_with_vectors, embed, args, logger)
            # convert to use_ids, use_class and use_scores
            use_response = pd.DataFrame(use_response)
            new_data_point["use_class_labels"] = [
                qid_to_class[id_] for id_ in use_response.questionId.tolist()
            ]
            new_data_point["use_scores"] = use_response.score.tolist()

            # get response from BM25
            bm25_response = make_query_bm25(client, args)
            # convert to bm25_ids, bm25_class and bm25_scores
            bm25_response = pd.DataFrame(bm25_response)
            # check if there are answers
            if bm25_response.empty:
                logger.info("No responses found for this query")
                new_data_point["bm25_class_labels"] = []
                new_data_point["bm25_scores"] = []
            else:
                new_data_point["bm25_class_labels"] = [
                    qid_to_class[id_] for id_ in bm25_response.questionId.tolist()
                ]
                new_data_point["bm25_scores"] = bm25_response.score.tolist()

            data_points.append(copy.deepcopy(new_data_point))

    return data_points


def save_data(
    logreg_model,
    std_scaler,
    df_training_data,
    qid_to_class,
    class_to_qid,
    all_docs_kb,
    data_kb_with_vectors,
    args
):
    """Saves the new model in models/KB_id_{KB_id} with the scaler, 
       the training data, the ref-tables and the data_kb_with_vectors
    """

    path = Path(ROOT)
    KB_id = args.KB_id
    
    path_live = path / "models" / f"KB_id_{KB_id}" / "live"
    path_live.mkdir(parents=True, exist_ok=True)

    path_archive = path / "models" / f"KB_id_{KB_id}" / "archive"
    path_archive.mkdir(parents=True, exist_ok=True)

    # check if path_live is empty
    files = os.listdir(path_live)

    if files:
        # move file to path_archive
        for f in files:
            path_archive_new = path_archive/datetime.now().strftime('%d_%m_%Y_time_%H_%M_%S')
            path_archive_new.mkdir(parents=True, exist_ok=True)
            
            shutil.move(src=str((path_live / f)), dst=str(path_archive_new))
        # append the time being archived
        with open(path_archive_new / "logs", "a") as fp:
            fp.write(
                f"\nArchived: {datetime.now().strftime('%d_%m_%Y_time_%H_%M_%S')}"
            )

    # save training data
    df_training_data.to_csv(path_live / TRAINING_DATA_FILE, sep=";")

    # save all_docs_kb
    save_json_file(all_docs_kb, path_live / args.filepath_json)
    # save data_kb_with_vector
    save_pickle_dict(data_kb_with_vectors, path_live / DATA_KB_WITH_VECTORS_FILE)

    # save reference dictionaries
    save_pickle_dict(qid_to_class, path_live / "qid_to_class.pkl")
    save_pickle_dict(class_to_qid, path_live / "class_to_qid.pkl")

    # save scaler
    save_pickle_dict(std_scaler, path_live / "std_scaler.pkl")
    dump(logreg_model, open(path_live / "logreg_model.joblib", "wb"))

    # save a logs file
    with open(path_live / "logs", "a") as fp:
        fp.write(
            f"Went live at: {datetime.now().strftime('%d_%m_%Y_time_%H_%M_%S')}"
        )

    # saves config file of how the model was created
    configfile_name = path_live/"config.cfg"
    # Check if there is already a configurtion file
    if not os.path.isfile(configfile_name):
        # Create the configuration file as it doesn't exist yet
        cfgfile = open(configfile_name, 'w')

        # Add content to the file
        Config = configparser.ConfigParser()
        Config.set(configparser.DEFAULTSECT, 'without_stopwords', str(args.without_stopwords))
        Config.set(configparser.DEFAULTSECT, 'num_of_sentences', str(args.num_of_sentences))
        Config.set(configparser.DEFAULTSECT, 'all_docs_kb_filename', str(args.filepath_json))
        Config.write(cfgfile)
        cfgfile.close()

        
    return


def main(args: argparse.ArgumentParser(), logger: logging.getLogger()):
    """
    Creates the training data from args.filepath_json

    1. Builds the data necessary to use Universal Sentence encoder
       (runs main from build_use.py)
    2. Indexes the data in ES
       (runs main from index_elasticsearch.py)
    3. loads the KB in form of the json given as input
    4. Builds questionId - Class_label reference dicts and saves them
    5. loads the KB with vectors, created with main_build_use
    6. initialises hub_module USE
    7. sets up Elasticsearch Client
    8. Query USE and ES (BM25) and save the responses list of dicts
    9. Create pd.DataFrame from this list of dicts and saves

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

    # load the knowledgebase
    logger.info("# load the knowledgebase")
    all_docs_kb = load_json_file(args.filepath_json, logger)

    # setup Client
    logger.info("# setup Client")
    client = Elasticsearch()

    # Initialize the use_model as tensorflow_hub module
    logger.info("# Initialize the use_model as tensorflow_hub module")
    embed = initialize_use_model()

    # initialize_nlp_model
    logger.info("# initialize_nlp_model")
    nlp = initialize_nlp_model()
    
    # execute main_build_use
    logger.info("# execute main_build_use")
    data_kb_with_vectors = build_use(embed, nlp, args, logger)

    # execute main index_elasticsearch
    logger.info("# execute main index_elasticsearch")
    index_kb(all_docs_kb, client, args, logger)

    # Build questionId - Class-label dictionaries
    logger.info("# Build questionId - Class-label dictionaries")
    qid_to_class, class_to_qid = build_qid_class_dicts(all_docs_kb, logger)

    # create training data points
    logger.info("# create training data points")
    data_points = get_data_points(
        all_docs_kb, data_kb_with_vectors, qid_to_class, embed, client
    )

    # save the data_points in csv format
    df_training_data = pd.DataFrame(
        data_points,
        columns=[
            "query",
            "class_label",
            "bm25_class_labels",
            "bm25_scores",
            "use_class_labels",
            "use_scores",
            "questionId",
            "title",
            "body",
        ],
    )


    # train and get the model
    model, scaler = train_and_save_model(df_training_data, args, logger)

    # save
    save_data(
        model,
        scaler,
        df_training_data,
        qid_to_class,
        class_to_qid,
        all_docs_kb,
        data_kb_with_vectors,
        args
    )
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger()

    # Argument parsing
    logger.info("# Argument parsing")
    parser = argparse.ArgumentParser(
        description=(
            "Input: .json file containing the KB with Queries\n"
            "Output: None\n"
            "Sideeffects: \n \t Creates data_kb_with_vectors with name set in settings\n"
            "\t Creates Index in ElasticSearch with the given json file\n"
            "\t Creates table as training_data with name set in in settings"
        ),
        formatter_class=RawTextHelpFormatter,
    )

    # filepath of the KB json
    args = parser.add_argument(
        "filepath_json", type=Path, help="File containing the KB export in json format"
    )

    args = parser.add_argument(
        "KB_id", type=int, help="KB_id integer identifying the KB"
    )

    # argument to indicate if evaluate string with or without stopwords
    parser.add_argument(
        "--without_stopwords",
        "--wo",
        help="Flag, "
        "if set then calculate_scores without stopwords in a documents body",
        action="store_true",
    )

    # How many sentences of the body shall be included in calculation
    parser.add_argument(
        "--num_of_sentences",
        type=int,
        choices=range(6),
        help="Up to num_of_sentences sentences "
        "of the body shall be included inside the calculation for the scores.\n"
        "Maximum: 5\n"
        "if nothing or 0 then only question title",
        default=1,
    )

    args = parser.parse_args()
    main(args, logger)
