"""Indexes KB that is stored in data.json

This script is doing the following. It cleans up the index. Then it loads the 
data und bulk-uploads the data

This script requires to have installed
elasticsearch6>=6.0.0,<7.0.0
"""
import json
from elasticsearch6 import Elasticsearch
from elasticsearch6.helpers import bulk

import argparse
import logging

from argparse import RawTextHelpFormatter
from pathlib import Path
from definitions import CLIENT, INDEX_NAME, MAPPING_JSON_FILE, BULK_REQUEST_JSON_MODEL


def load_json_file(filepath: Path, logger) -> dict:
    """loads json file"""
    try:
        with open(filepath, 'r') as fp:
            all_docs_kb = json.load(fp)
    except FileNotFoundError as err:
        logger.error(err)
        raise
    return all_docs_kb

def index_kb(args: argparse.ArgumentParser(), logger: logging.getLogger()):
    """
    Indexes the KB
    
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

    # clean the index
    logger.info("# clean the Index")
    client = Elasticsearch()
    # Delete Index
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    # Create Index
    with open(MAPPING_JSON_FILE) as mapping_json_file:
        source = mapping_json_file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)

    # load the data
    logger.info("# load the data")
    all_docs_kb = load_json_file(args.filepath_json, logger)

    # upload the KB
    logger.info("# upload the KB")
    bulk(client, all_docs_kb)
    return


def main(args: argparse.ArgumentParser(), logger: logging.getLogger()):
    """
    Indexes the KB
    
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

    # indexing KB
    index_kb(args, logger)

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    # Argument parsing
    logger.info("# Argument parsing")
    parser = argparse.ArgumentParser(
        description=(
            "Input: .json file containing the KB\n" \
            "Output: None\n" \
            "Sideeffects: Creates Index in ElasticSearch with the given json file"
        ), formatter_class=RawTextHelpFormatter)
    args = parser.add_argument(
        "filepath_json", type=Path,
        help="File containing the KB export in json format")
    args = parser.parse_args()

    main(args, logger)
