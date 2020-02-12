import configparser
import os
from os import environ
__all__ = ["ROOT", "CLIENT", "TRAINING_DATA_FILE", "FILTER_TOP_N", "SPACY_MODEL", "HUB_MODEL", "DATA_KB_WITH_VECTORS_FILE",
           "INDEX_NAME", "BULK_REQUEST_JSON_MODEL", "MAPPING_JSON_FILE", "SEARCH_SIZE", "QUERY_JSON_MODEL", "NUM_OF_SCORES"]

if 'ROOT_build_and_deploy' not in environ:
    environ["ROOT_build_and_deploy"] = os.path.dirname(
        os.path.abspath(__file__))

ROOT = os.getenv('ROOT_build_and_deploy', '')

config = configparser.ConfigParser()
config.read('settings.cfg')
config_set = 'DEFAULT'

CLIENT = config[config_set]["ClientName"]
TRAINING_DATA_FILE = config[config_set]["training_data_file"]
FILTER_DATA_TOP_N = int(config[config_set]["filter_data_top_n"])

SPACY_MODEL = config[config_set]["spacy_model"]
HUB_MODEL = config[config_set]["hub_Model"]
DATA_KB_WITH_VECTORS_FILE = config[config_set]["data_kb_with_vectors_file"]

INDEX_NAME = CLIENT + "_posts"
BULK_REQUEST_JSON_MODEL = config[config_set]["bulk_request_json_model"]
MAPPING_JSON_FILE = config[config_set]["mapping_json_file"]
QUERY_JSON_MODEL = config[config_set]["Query_Json_model"]
