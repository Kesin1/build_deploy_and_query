"""Query against the KB that is stored in the ElasticSearch Index

This script is doing the following. 
1. setup Client
2. setup and make query request
3. filter out questionId and score

This script requires to have the all_docs_with_vectors.pkl in the 
file folder
"""

import argparse
import logging
import json
from elasticsearch6 import Elasticsearch
from argparse import RawTextHelpFormatter
from pathlib import Path
from definitions import CLIENT, INDEX_NAME, QUERY_JSON_MODEL

def setup_query(args: argparse.ArgumentParser()) -> dict:
    """
    Builds a request that can be forwarded to an ES-Client containing
    the query 

    Parameters
    ----------
    args: argparse.ArgumentParser
        The command line arguments given

    Returns
    -------
    dict
        dictionary containing a request
    """

    # setup query request
    query = args.query
    query_request = json.loads(QUERY_JSON_MODEL)
    query_request["query"]["bool"]["should"][0]["multi_match"]["query"] = query
    query_request["query"]["bool"]["should"][0]["multi_match"]["query"] = query
    query_request["query"]["bool"]["should"][1]["nested"]["query"]["multi_match"]["query"] = query
    query_request["suggest"]["text"] = query

    return query_request


def make_request(query_request: dict, client: Elasticsearch()) -> list:
    """
    Makes a Call to an ES-Client with the query_request
    and returns a sorted list of dicitonaries containing the questionId and Score

    Parameters
    ----------
    query_request: dict
        dictionary containing the request to make at the client
    client: Elasticsearch()
        Elasticsearch client
    Returns
    -------
    ordered list of dictionaries {'questionId: ..., 'score: }
        
    """

    # make query request
    response = client.search(
        index=INDEX_NAME,       # index names to search
        body=query_request
    )

    # filter out questionId and score
    hits = response["hits"]["hits"]
    bm25_scores = [{'questionId': hit["_source"]["questionId"], 'score': hit["_score"]} for hit in hits]

    return bm25_scores


def make_query_bm25(client, args):
    """
    Makes a Call to an ES-Client with the query_request
    and returns a sorted list of dicitonaries containing the questionId and Score

    Parameters
    ----------
    client: Elasticsearch()
        Elasticsearch client
    args: argparse.ArgumentParser
        The command line arguments given - needs to contain argument query

    Returns
    -------
    ordered list of dictionaries {'questionId: ..., 'score: }

    """

    # setup query
    # logger.info("# setup query")
    query_request = setup_query(args)

    # make request
    # logger.info("# make request")
    bm25_scores = make_request(query_request, client)

    return bm25_scores


def main(args: argparse.ArgumentParser(), logger: logging.getLogger()) -> list:
    """
    Setups Client, Makes the call at Elasticsearch with the query
    and returns a sorted list of dicitonaries 

    Parameters
    ----------
    args: argparse.ArgumentParser
        The command line arguments given
    logger: logging.getLogger
        logger on DEBUG level

    Returns
    -------
    ordered list of dictionaries {'questionId: ..., 'score: }
    """

    # setup Client
    logger.info("# setup Client")
    client = Elasticsearch()

    # setup query
    logger.info("# setup query")
    query_request = setup_query(args)

    # make request
    logger.info("# make request")    
    bm25_scores = make_request(query_request, client)

    return bm25_scores
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    # Argument parsing
    logger.info("# Argument parsing")
    parser = argparse.ArgumentParser(description="Input: query_string\nOutput: list of {questionId:... ;score:... }",
        formatter_class=RawTextHelpFormatter)

    # query
    args = parser.add_argument("query", type=str, help="a query string")

    args = parser.parse_args()

    main(args, logger)
