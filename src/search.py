from typing import List, Dict, Tuple
import argparse
import time
import json
from tqdm import tqdm
from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
from FlagEmbedding import FlagModel, FlagReranker

from utils import check_if_index_exists

# def search(
#     index_name: str,
#     method: str,
#     config_path: str,
#     query_path: str | List[Dict],
#     server: str = "http://localhost:9200",
#     rerank: bool = True,
#     filtered_count: int = 5,
#     coarse_filtered_count: int = None,
#     kNN_num_candidates: int = None
# ) -> List[List[Dict]]:
#     # link to server
#     es = Elasticsearch(server, retry_on_timeout=True)
#     if not es.ping():
#         raise ConnectionError(f"Cannot ping {server}.")
#     if not check_if_index_exists(es, index_name):
#         raise RuntimeError("An index with same name exists. If you want to override it, please set --override.")
#     # load config
#     with open(config_path, "r") as fin:
#         config = json.load(fin)
#     # load queries:
#     if type(query_path) is str:
#         with open(query_path, "r") as fin:
#             qrys = [json.loads(line) for line in fin]
#     else:
#         qrys = query_path
#     # start to search
#     if not (method == "BM25" or method == "dense"):
#         raise RuntimeError(f"Method not supported!")
#     if not coarse_filtered_count:
#         coarse_filtered_count = filtered_count
#     if not kNN_num_candidates:
#         kNN_num_candidates = coarse_filtered_count * 100
#     results = []
#     if method == "BM25":
#         for qry in qrys:
#             body = {
#                 "size": coarse_filtered_count,
#                 "query":{
#                     "match":{
#                         "text": qry["text"]
#                     }
#                 },
#                 "sort":[
#                     {"_score": {"order": "desc"}}
#                 ]
#             }
#             response = es.search(index=index_name, body=body)
#             results.append([{
#                 "id": hit["_source"]["id"], 
#                 "text": hit["_source"]["text"],
#                 "score": hit["_score"]
#                 } for hit in response["hits"]["hits"]
#             ])
#     else:
#         if not config["use_dense"]:
#             raise RuntimeError("use_dense==False in your config!")
#         encoder = FlagModel(config["embedding_model_name_or_path"],
#                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
#                   use_fp16=True) # 配置纯抄README
#         texts = [qry["text"] for qry in qrys]
#         embeddings = encoder.encode_queries(queries=texts, batch_size=config.get("queries_embed_batch_size", 64))
#         for i, qry in enumerate(qrys):
#             body = {
#                 "query":{
#                     "knn":{
#                         "field": "text_embedding",
#                         "query_vector": embeddings[i].tolist(),
#                         "k": coarse_filtered_count,
#                         "num_candidates": kNN_num_candidates
#                     }
#                 }
#             }
#             response = es.search(index=index_name, body=body)
#             results.append([{
#                 "id": hit["_source"]["id"], 
#                 "text": hit["_source"]["text"],
#                 "score": hit["_score"]
#                 } for hit in response["hits"]["hits"]
#             ])
#         if rerank:
#             reranker = FlagReranker(config["reranker_model_name_or_path"], use_fp16=True)
#             for qry, docs in zip(qrys, results):
#                 for doc in docs: 
#                     doc["score"] = reranker.compute_score([qry["text"], doc["text"]])[0] # 覆盖retriever的分数
#                 docs.sort(key=lambda doc: -doc["score"])
                
#     for i, result in enumerate(results):
#         results[i] = result[:filtered_count]
#     return results
    
from utils import IndexWrapper

def search(
    index_name: str,
    config_path: str,
    query_path: str | List[Dict],
    method: str,
    server: str = "http://localhost:9200",
    rerank: bool = True,
    filtered_count: int = 5,
    coarse_filtered_count: int = None,
    kNN_num_candidates: int = None
) -> List[List[Dict]]:
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
    if not index.check_if_index_exists():
        raise RuntimeError(f"The index {index_name} does not exists.")
    return index.search_with_file_of_queries(
        data_path=query_path,
        method=method,
        rerank=rerank,
        filtered_count=filtered_count,
        coarse_filtered_count=coarse_filtered_count,
        kNN_num_candidates=kNN_num_candidates
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_path", type=str, default=None, 
                        help="If you set this, arguments in the corresponding file will be used.")
    parser.add_argument("--index_name", type=str, default="stard", help="Name of the new-built index.")
    parser.add_argument("--config_path", type=str, default="../config/chinese.json", help="Config about how to build index.")
    parser.add_argument("--server", type=str, default="http://localhost:9200", 
                        help="Url of Elasticsearch server. Default to 'http://localhost:9200'.")
    parser.add_argument("--method", type=str, default="BM25", 
                        help="Method for retrieval. Only support 'BM25' and 'dense' (excluding quotation marks). \
                            If you choose 'dense', choose whether to use reranker.")
    parser.add_argument("--rerank", action="store_true", default=False,
                        help="Whether to use reranker (if method=='dense'). Default to False.")
    parser.add_argument("--filtered_count", type=int, default=5, 
                        help="Number of documents to be returned. Default to 5.")
    parser.add_argument("--coarse_filtered_count", type=int, default=None,
                        help="If you used two stage retrieval (i.e. retrieve and rerank in dense retrieval), \
                            you can choose the number of documents filtered in the first stage, which might be larger than that in the second stage. \
                            Default to be equal to filtered_count.")
    parser.add_argument("--kNN_num_candidates", type=int, default=None,
                        help="We use approximate kNN in dense retrieval. The algorithm would find k nearest neighbors of a query vector among c document vectors, \
                            where k = coarse_filtered_count, c = kNN_num_candidates.\
                            Default to be equal to 100 * coarse_filtered_count.")
    parser.add_argument("--query_path", type=str, default=None, 
                        help="The path to jsonl file where queries are. Each line in the jsonl file must has attribute 'text'.\
                            You can also use List[Dict] to directly represent it if you use --search_args_path")
    args = vars(parser.parse_args()) # 转成熟悉的dict类型    
    if "args_path" in args:
        with open(args["args_path"], "r") as fin:
            args = json.load(fin)
    else:
        args["coarse_filtered_count"] = args["filtered_count"]

    results = search(**args)
    # print(results)
    print(json.dumps(results, ensure_ascii=False))

    # for docs in results:
    #     for doc in docs:
    #         print(json.dumps(doc, ensure_ascii=False))
    #     print()