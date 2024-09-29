from typing import List, Dict, Tuple
import sys
import json
import logging
from tqdm import tqdm

from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk

from transformers import AutoConfig
from FlagEmbedding import FlagModel, FlagReranker

# logging.basicConfig(level=logging.INFO) 
# logger = logging.getLogger(__name__)

# def check_if_index_exists(es: Elasticsearch, index_name: str):
#     return index_name in es.indices.get_alias(index="_all").keys()

class IndexWrapper:
    def __init__(
        self,
        index_name: str,
        config_path: str,
        server: str = "http://localhost:9200",
        **Elasticsearch_kwargs
    ):
        # link to server
        self.es = Elasticsearch(server, **Elasticsearch_kwargs)
        if not self.es.ping():
            raise ConnectionError(f"Failed to connect {server}.")
        self.index_name = index_name
        with open(config_path, "r") as fin:
            self.config = json.load(fin)
        # load model if use_dense
        if self.config["use_dense"]:
            self.encoder_config = AutoConfig.from_pretrained(self.config["embedding_model_name_or_path"])
            self.encoder = FlagModel(self.config["embedding_model_name_or_path"],
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                    use_fp16=True) # 配置纯抄README
            if self.config["reranker_model_name_or_path"]:
                # self.reranker_config = AutoConfig.from_pretrained(self.config["reranker_model_name_or_path"])
                self.reranker = FlagReranker(self.config["reranker_model_name_or_path"], use_fp16=True)
            else:
                self.reranker = None
        # Some default values
        if "embed_batch_size" not in self.config:
            self.config["embed_batch_size"] = 256
        if "chunk_size" not in self.config:
            self.config["chunk_size"] = 8192
        if "thread_count" not in self.config:
            self.config["thread_count"] = 16

    def check_if_index_exists(self):
        '''
        Always warning here. But it doesn't matter.
        '''
        return self.index_name in self.es.indices.get_alias(index="_all").keys()

    def init(self):
        '''
        (re)build index according to config.
        '''
        if self.check_if_index_exists():
            self.es.indices.delete(index=self.index_name)
        body = {
            # 下面这块关于中文的设置丢进config文件中了
            # "settings":{
            #     "analysis":{
            #         "analyzer":{
            #             "chinese":{ # Elasticsearch中没有内置的中文分词器，要安装插件
            #                 "type": "ik_max_word" # ik分词器
            #             }
            #         }
            #     }
            # },
            "mappings":{
                "properties":{ # 关于各单位数据存什么
                    "id": {
                        "type": "keyword" # keyword要求有唯一性
                    },
                    "text": {
                        "type": "text", # text类型会用BM25分析
                        "analyzer": self.config["language"]
                    }
                    # 如果开启稠密向量检索，还要存放文本的稠密向量编码
                }
            }
        }
        if "settings" in self.config:
            body["settings"] = self.config["settings"]
        if self.config["use_dense"]:
            body["mappings"]["properties"]["text_embedding"] = {
                "type": "dense_vector",
                "dims": self.encoder_config.hidden_size,
                "similarity": self.config.get("similarity", "max_inner_product") 
                # 可选l2_norm、dot_product、cosine、max_inner_product。具体见https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-similarity
                # max_inner_product是内积的基础上做了手脚使得结果非负，结果和直接内积在保序的意义下是等价的。关于dot_product：
                # dot_product要求向量已经归一化，而且还会强制检查。
                # 听说BGE的编码已经默认归一化了：https://huggingface.co/BAAI/bge-large-zh-v1.5/discussions/10
                # 尴尬的是可能由于舍入误差，不能通过归一化的检查。
            }
        self.es.indices.create(index=self.index_name, body=body)

    def submit_operations(self, operations, show_tqdm=False, tqdm_desc=False):
        # parallel_bulk的封装罢了
        for ok, action in tqdm(parallel_bulk(self.es, operations, thread_count=self.config["thread_count"], raise_on_error=False), disable=not show_tqdm, desc=tqdm_desc):
            if not ok:
                raise RuntimeError(f"action={action}")
            pass

    def data_grouper(self, data: List[Dict[str, int]]):
        # 将数据分组的目的是不要一下子在内存中加入大量的数据却屯着没有送到Elasticsearch服务端
        group = []
        for d in data:
            group.append(d)
            if len(group) >= self.config["chunk_size"]:
                yield group
                group = []
        if group:
            yield group

    def add_docs(self, corpus: List[Dict[str, int]]):
        '''
        Each item in corpus: {"id": ..., "text": ...}
        Recommended to use generator.
        '''
        for group in self.data_grouper(corpus):
            docs = [{
                    "_index": self.index_name,
                    "_source":{
                        "id": d["id"],
                        "text": d["text"]
                    }
                } for d in group
            ]
            if self.config["use_dense"]:
                embeddings = self.encoder.encode_corpus(
                    corpus=[d["text"] for d in group], 
                    batch_size=self.config["embed_batch_size"]
                ) # 这个函数内自带进度条，我不知道怎么在不改库函数的前提下关掉
                for i, doc in enumerate(docs):
                    doc["_source"]["text_embedding"] = embeddings[i].tolist() # 因为是网络操作所以要转成特定形式
                self.submit_operations(docs, show_tqdm=True, tqdm_desc="Add docs")
        
    def add_docs_from_file(self, data_path: str):
        '''
        jsonl file. Each line:
        {"id": ..., "text": ...}
        '''
        def generator():
            with open(data_path, "r") as fin:
                for line in fin:
                    d = json.loads(line)
                    yield {
                        "id": d["id"],
                        "text": d["text"]
                    }
        self.add_docs(generator())

    def search(
        self, 
        queries: List[Dict[str, int]], 
        method,  # "BM25" or "dense"
        rerank: bool = True, # 是否启用reranker。仅有method=="dense"时才会有影响
        filtered_count: int = 5, # 最终对各询问分别需要返回的文档数目
        coarse_filtered_count: int = None, # 仅有method=="dense"且rerank=True时才会有影响。
        # 设定之后，检索阶段将先检索出coarse_filtered_count个结果，这些结果重排后保留前filtered_count个结果。
        kNN_num_candidates: int = None # 仅有method=="dense"时才会有影响。
        # 在近似k近邻居算法中，部分向量会被筛选出，然后在这些向量中选取相关度最大的k个。kNN_num_candidates就是一开始被筛选出的部分向量的个数。
    ):
        '''
        Each item in queries: {"text": ...}
        Recommended to use generator.
        '''
        results = []
        if method == "BM25":
            for qry in queries: # 逐个请求
                body = {
                    "size": filtered_count,
                    "query":{
                        "match":{
                            "text": qry["text"]
                        }
                    },
                    "sort":[
                        {"_score": {"order": "desc"}}
                    ]
                }
                response = self.es.search(index=self.index_name, body=body)
                results.append([{
                    "id": hit["_source"]["id"], 
                    "text": hit["_source"]["text"],
                    "score": hit["_score"]
                    } for hit in response["hits"]["hits"]
                ])
        else:
            if not method == "dense":
                raise NotImplementedError(f"Method {method} not supported! Use BM25 or dense instead.")
            if not coarse_filtered_count or not rerank:
                coarse_filtered_count = filtered_count
            if not kNN_num_candidates:
                kNN_num_candidates = coarse_filtered_count * 100
            if not self.config["use_dense"]:
                raise RuntimeError("use_dense==False in your config!")
            for group in self.data_grouper(queries): # 分组操作，因为希望能用到些并行。
                embeddings = self.encoder.encode_queries(
                    queries=[qry["text"] for qry in group], 
                    batch_size=self.config["batch_size"]
                )
                results_group = []
                for i, qry in enumerate(group):
                    body = {
                        "query":{
                            "knn":{
                                "field": "text_embedding",
                                "query_vector": embeddings[i].tolist(),
                                "k": coarse_filtered_count,
                                "num_candidates": kNN_num_candidates
                            }
                        }
                    }
                    response = self.es.search(index=self.index_name, body=body)
                    results_group.append([{
                        "id": hit["_source"]["id"], 
                        "text": hit["_source"]["text"],
                        "score": hit["_score"]
                        } for hit in response["hits"]["hits"]
                    ])
                if rerank:
                    if not self.reranker:
                        raise AttributeError("You haven't set reranker.")
                    for qry, docs in zip(group, results_group):
                        scores = self.reranker.compute_score(
                            [[qry["text"], doc["text"]] for doc in docs],
                            batch_size=self.config["batch_size"]
                        )
                        for i, doc in enumerate(docs): 
                            doc["score"] = scores[i] # 覆盖retriever的分数
                        docs.sort(key=lambda doc: -doc["score"])
                    for i, docs in enumerate(results_group):
                        results_group[i] = docs[:filtered_count]
                results.extend(results_group)
        return results

    def search_with_file_of_queries(
        self, 
        data_path: str,
        **kwargs
    ):
        '''
        jsonl file. Each line:
        {"text": ...}
        '''
        def generator():
            with open(data_path, "r") as fin:
                for line in fin:
                    q = json.loads(line)
                    yield {
                        "text": q["text"]
                    }
        return self.search(queries=generator(), **kwargs)

    def count_docs(self):
        return self.es.count(index=self.index_name)["count"]

        

        
