import argparse
import time
import json
import sys
from tqdm import tqdm
import numpy as np

from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
from elasticsearch.helpers import parallel_bulk

from transformers import AutoConfig
from FlagEmbedding import FlagModel

from utils import IndexWrapper

def build(
    index_name: str,
    config_path: str,
    data_path: str = None,
    override: bool = False,
    server: str = "http://localhost:9200"
):
    if not index_name:
        raise RuntimeError("Lacking index_name.")
    if not config_path:
        raise RuntimeError("Lacking config_path.")
    index = IndexWrapper(
        index_name=index_name, 
        config_path=config_path,
        server=server,
        retry_on_timeout=True
    )
    if not override and index.check_if_index_exists():
        raise RuntimeError("An index with same name exists. If y?ou want to override it, please set --override.")
    index.init()
    if data_path:
        index.add_docs_from_file(data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_path", type=str, default=None, 
                        help="If you set this, arguments in the corresponding file will be used.")
    parser.add_argument("--index_name", type=str, default=None, help="Name of the new-built index.")
    parser.add_argument("--config_path", type=str, default=None, help="Config about how to build index.")
    parser.add_argument("--data_path", type=str, default=None, help="File of the corpus to load. \
                        If you don't set data_path, you would create an empty index.")
    parser.add_argument("--server", type=str, default="http://localhost:9200", 
                        help="Url of Elasticsearch server. Default to 'http://localhost:9200'.")
    parser.add_argument("--override", action="store_true", default=False, 
                        help="Override if an index with same name exists. Default to False.")
    args = vars(parser.parse_args()) # 转成熟悉的dict类型
    if "args_path" in args:
        with open(args["args_path"], "r") as fin:
            args = json.load(fin)
    build(**args)
