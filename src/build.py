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

# from utils import check_if_index_exists

# def build(
#     data_path: str,
#     index_name: str,
#     config_path: str,
#     override: bool = False,
#     server: str = "http://localhost:9200"
# ):
#     # link to server
#     es = Elasticsearch(server, retry_on_timeout=True)
#     if not es.ping():
#         raise ConnectionError(f"Failed to connect {server}.")
#     if check_if_index_exists(es, index_name):
#         if not override:
#             raise RuntimeError("An index with same name exists. If you want to override it, please set --override.")
#         es.indices.delete(index=index_name)
#         time.sleep(5)
#     # load config
#     with open(config_path, "r") as fin:
#         config = json.load(fin)
#     # load embedding model (dense retrieval)
#     if config["use_dense"]:
#         model_config = AutoConfig.from_pretrained(config["embedding_model_name_or_path"])
#         model = FlagModel(config["embedding_model_name_or_path"],
#                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
#                   use_fp16=True) # 配置纯抄README

#     # bulid index
#     body = {
#         # 下面这块关于中文的设置丢进config文件中了
#         # "settings":{
#         #     "analysis":{
#         #         "analyzer":{
#         #             "chinese":{ # Elasticsearch中没有内置的中文分词器，要安装插件
#         #                 "type": "ik_max_word" # ik分词器
#         #             }
#         #         }
#         #     }
#         # },
#         "mappings":{
#             "properties":{
#                 "id": {
#                     "type": "keyword"
#                 },
#                 "text": {
#                     "type": "text",
#                     "analyzer": config["language"]
#                 }
#             }
#         }
#     }
#     if "settings" in config:
#         body["settings"] = config["settings"]
#     if config["use_dense"]:
#         body["mappings"]["properties"]["text_embedding"] = {
#             "type": "dense_vector",
#             "dims": model_config.hidden_size,
#             "similarity": config.get("similarity", "cosine") 
#             # 可选l2_norm、dot_product、cosine、max_inner_product。具体见https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-similarity
#             # dot_product要求向量已经归一化，而且还会强制检查。
#             # 听说BGE的编码已经默认归一化了：https://huggingface.co/BAAI/bge-large-zh-v1.5/discussions/10
#         }
#     es.indices.create(index=index_name, body=body)
#     # load_data
#     chunk_size = config.get("load_chunk_size", 10000)
#     batch_size = config.get("embed_batch_size", 256)
#     thread_count = config.get("load_thread_count", 16)
#     def submit_docs(docs):
#         if config["use_dense"]:
#             texts = [doc["_source"]["text"] for doc in docs]
#             embeddings = model.encode_corpus(corpus=texts, batch_size=batch_size,) # 这个函数内自带进度条，我不会在不改库函数的前提下关掉
#             for i, doc in enumerate(docs):
#                 doc["_source"]["text_embedding"] = embeddings[i]
#         for ok, action in tqdm(parallel_bulk(es, docs, thread_count=thread_count, raise_on_error=False), desc="Adding Docs"):
#             if not ok:
#                 print(action, file=sys.stderr)
#                 print("Halt.", file=sys.stderr)
#                 raise RuntimeError("Some doc failed to index.")
#             # print(ok)
#             # print(action)
#             pass
#     docs = []
#     with open(data_path, "r") as fin:
#         for line in fin:
#             data = json.loads(line)
#             doc = {
#                 "_index": index_name,
#                 "_source":{
#                     "id": data["id"],
#                     "text": data["text"]
#                 }
#             }
#             docs.append(doc)
#             if len(docs) >= chunk_size:
#                 submit_docs(docs); docs = []
#     submit_docs(docs); docs = []
#     # def generate_actions():
#     #     with open(data_path, "r") as fin:
#     #         for line in fin:
#     #             data = json.loads(line)
#     #             doc = {
#     #                 "_index": index_name,
#     #                 "_source":{
#     #                     "id": data["id"],
#     #                     "text": data["text"]
#     #                 }
#     #             }
#     #             if config["use_dense"]:
#     #                 doc["_source"]["text_embedding"] = model.encode(data["text"]).tolist()
#     #             yield doc
#     # for ok, action in tqdm(parallel_bulk(es, generate_actions(), thread_count=thread_count, chunk_size=batch_size)):
#     #     pass

#     from utils import check_if_index_exists

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
