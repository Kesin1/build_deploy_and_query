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

def index_kb(all_docs_kb: dict, client: Elasticsearch(), args: argparse.ArgumentParser(), logger: logging.getLogger()):
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

    # Delete Index
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    # Create Index
    with open(MAPPING_JSON_FILE) as mapping_json_file:
        source = mapping_json_file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)

    # upload the KB
    logger.info("# upload the KB")
    bulk(client, all_docs_kb)
    return
