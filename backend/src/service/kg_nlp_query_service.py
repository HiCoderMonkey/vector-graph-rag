import json
import logging
import time
import uuid

import concurrent.futures
from datetime import timedelta

import requests
import torch

from base.meta_singeton import MetaSingleton
from common.embedding.corom_embedding import CoromEmbedding
from common.raner.nlp_raner import NlpRaNer
from common.rerank.ranking_chinese_base.ranking_chinese import RankingChinese
from common.split_word.ba_struct_bert.ba_struct_bert import BaStructBert
from dao.neo4j_dao import Neo4jDao
from db.my_redis.redis_client import RedisClient
from db.vdb.milvus.milvus_vector import MilvusConfig, MilvusVector
from dto.document import Document
from resource import AppConfig
from dto.request_dto import InputData, SaveAnswerData
from env import Instance as env_import
from common.utils.neo4j_utils import build_tree, get_paths

logger = logging.getLogger(__name__)


def chunk_list(lst, n):
    """将列表分成每组n个的小列表"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def re_rank_chunk(ranking_chinese, chunk, data_text_input,**kwargs):
    """对一个chunk进行重排序"""
    return ranking_chinese.do_re_rank(src_s=data_text_input, re_rank_list=chunk,**kwargs)


def re_rank_with_multithreading(ranking_chinese, data_text_input, re_rank_list, chunks_size=30,**kwargs):
    # 将re_rank_list分成每组5个的小列表
    chunks = list(chunk_list(re_rank_list, chunks_size))

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 开线程执行
        futures = []
        for chunk in chunks:
            future = executor.submit(re_rank_chunk, ranking_chinese, chunk, data_text_input,**kwargs)
            futures.append(future)
            # time.sleep(0.01)  # 间隔10ms
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    # 根据分数降序排序
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    return sorted_results


class KgNlpQueryService(metaclass=MetaSingleton):
    neo4j_dao: Neo4jDao = None
    redis_client: RedisClient = None
    corom_embedding: CoromEmbedding = None
    nlp_raner: NlpRaNer = None
    ba_struct_bert: BaStructBert = None
    ranking_chinese: RankingChinese = None

    def __init__(self):
        self.neo4j_dao = Neo4jDao()
        self.redis_client = RedisClient()
        # 初始化向量数据库
        self.milvus_config = AppConfig().config["MILVUS"]
        milvusConfig = MilvusConfig(host=self.milvus_config["HOST"],
                                    port=self.milvus_config["PORT"],
                                    user=self.milvus_config["USER"],
                                    password=self.milvus_config["PASSWORD"],
                                    secure=self.milvus_config["SECURE"],
                                    database=self.milvus_config["DATABASE"])
        self.milvus_vector = MilvusVector(collection_name=self.milvus_config["DEFAULT_COLLECTION"], config=milvusConfig)
        # 初始化embedding
        self.corom_embedding = CoromEmbedding()
        # 初始化 ner模型
        # self.nlp_raner = NlpRaNer()
        # 分词模型
        self.ba_struct_bert = BaStructBert()
        # ranking模型
        self.ranking_chinese = RankingChinese()

    def kg_nlp_query_prompt(self, data: InputData):
        # 1、获取文本实体
        # result = self.nlp_raner.get_ner_result(data.text_input)
        # word = self.ba_struct_bert.do_split_word(data.text_input)
        last_answer = """"""
        # if not result:
        #     result = [' '.join(f'"{s}"' for s in word)]
        relations = self.get_same_relation(data.text_input)
        result = [data.text_input]
        # 2、查询最相似的实体
        docs = []
        # 示例代码片段 1
        start_time = time.time()
        # 代码片段 1 的代码
        # 判断是不是追问：是追问提取上一次的实体
        # is_ask_question = self.is_ask_question(data.text_input, data.session_id)
        is_ask_question = False
        if is_ask_question:
            docs = self.get_last_node_entity(data.session_id)
            embeddings = [self.corom_embedding.get_embedding(f"{t}") for t in result]
            # 最后的答案差一次加到 docs里
            qa_list = self.get_session_qa_list(data.session_id)
            last_a = ""
            if qa_list:
                last_a = qa_list[-1].llm_answer
                embeddings.append(self.corom_embedding.get_embedding(f"{last_a}"))
            logger.info(f"embeddings的长度{len(embeddings)}")
            for emb in embeddings:
                docs.extend(
                    self.milvus_vector.search_by_vector(query_vector=emb,
                                                        collection_name=self.milvus_config["NODE_COLL"]))
            # 对docs再 rank一下
            # docs = self.ranking_chinese.do_re_rank(src_s=f'{last_a}{" ".join(result)}', re_rank_list=docs)
            # docs = sorted(docs, key=lambda x: x.metadata["rank_score"] + x.metadata["score"], reverse=True)
            docs = docs[:5]
            logger.info(f"向量匹配的：{docs}")
        else:
            embeddings = [self.corom_embedding.get_embedding(f"{t}") for t in result]
            # 如果有历史问答，把最后一个回答也加到docs向量里
            qa_list = self.get_session_qa_list(data.session_id)
            if qa_list:
                last_a = qa_list[-1].llm_answer
                embeddings.append(self.corom_embedding.get_embedding(f"{last_a}"))
            for emb in embeddings:
                docs.extend(
                    self.milvus_vector.search_by_vector(query_vector=emb,
                                                        collection_name=self.milvus_config["NODE_COLL"],
                                                        score_threshold=45))
            logger.info(f"向量匹配的：{docs}")
            # 对docs再 rank一下
            docs = self.ranking_chinese.do_re_rank(src_s=" ".join(result), re_rank_list=docs)
            docs = sorted(docs, key=lambda x: x.metadata["rank_score"] + x.metadata["score"], reverse=True)
            docs = docs[:5]
            # 保存最后一次查询的实体
            self.save_last_node_entity(docs, data.session_id)
            logger.info(f"reRank后：{docs}")
        end_time = time.time()
        logger.info(f"向量匹配耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()
        docs_type_dict = {}
        logger.info(f"相似节点的长度{len(docs)}")
        for doc in docs:
            if doc.metadata["node_label"] not in docs_type_dict:
                docs_type_dict[doc.metadata["node_label"]] = []
            docs_type_dict[doc.metadata["node_label"]].append(doc)
        # 3、获取所有的图路径描述
        paths = []
        for dk, dv in docs_type_dict.items():
            ids = [d.metadata["id"] for d in dv]
            type = dk
            ps = self._neo4j_path_ids(ids, type, relations=relations)
            paths.extend(ps)
        # 4、使用ReRank获取和用户问题最相似的语义
        re_rank_list = paths
        if not re_rank_list:
            return []
        end_time = time.time()
        logger.info(f"查询图数据库耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()

        qa_list = self.get_session_qa_list(data.session_id)
        last_a = ""
        if qa_list and is_ask_question:
            last_a = qa_list[-1].llm_answer

        # 使用方法
        if torch.cuda.is_available():
            logger.info("使用gpu推理")

            rank = self.ranking_chinese.do_re_rank(src_s=f'{last_a} {data.text_input}', re_rank_list=re_rank_list)
            # rank = re_rank_with_multithreading(ranking_chinese=self.ranking_chinese,
            #                                    data_text_input=f'{last_a} {data.text_input}',
            #                                    re_rank_list=re_rank_list,chunks_size=100)
        else:
            logger.info("使用cpu推理")
            rank = re_rank_with_multithreading(ranking_chinese=self.ranking_chinese,
                                               data_text_input=f'{last_a} {data.text_input}',
                                               re_rank_list=re_rank_list)
        end_time = time.time()
        logger.info(f"rerank耗时: {end_time - start_time:.4f} 秒")

        # 获取分数 > 0.45 的
        # 获取最高分
        highest_score = max(r.score for r in rank)
        line_limit = highest_score * 0.7
        down_limit = 0.35
        rank_top = [r.text for r in rank if r.score > down_limit and r.score > line_limit]
        print(rank)
        return {
            "line_limit": line_limit,
            "down_limit": down_limit,
            "top_texts": rank_top,
            "all_ranks": [r.dict() for r in rank],
            "is_ask_question": is_ask_question
        }

    def _neo4j_path_ids(self, ids: list, doc_type: str, start_node_type: list = ["Tour"],
                        end_node_type: list = ["Destination"], database: str = None, relations=None) -> list[str]:
        with self.neo4j_dao.get_session(database=database) as tx:
            if doc_type in start_node_type:
                path_str = "(n)-[r*..4]->(o)"
            elif doc_type in end_node_type:
                path_str = "(n)<-[r*..4]-(o)"
            else:
                path_str = "(n)-[r*..4]-(o)"
            result = self.neo4j_dao.get_path(tx, ids, path_str, relations=relations)
            ns, rs = self._build_tree_arr(result)
        root_nodes = build_tree(ns, rs)
        # 获取所有路径描述的字符串数组
        paths = get_paths(root_nodes)
        # 打印路径
        return paths

    def get_same_relation(self, query_str):
        query_embedding = self.corom_embedding.get_embedding(query_str)
        dos = self.milvus_vector.search_by_vector(query_embedding, collection_name="kg_relationship_coll", top_k=4)
        res = []
        for d in dos:
            res.append(d.page_content)
        return res

    def _neo4j_path(self, doc: Document, start_node_type: list = ["Tour"], end_node_type: list = ["Destination"],
                    database: str = None) -> \
            list[str]:
        with self.neo4j_dao.get_session(database=database) as tx:
            id = doc.metadata["id"]
            if doc.metadata['node_label'] in start_node_type:
                path_str = "(n)-[r*..3]->(o)"
            elif doc.metadata['node_label'] in end_node_type:
                path_str = "(n)<-[r*..3]-(o)"
            else:
                path_str = "(n)-[r*..3]-(o)"
            result = self.neo4j_dao.get_path(tx, [id], path_str)
            ns, rs = self._build_tree_arr(result)
        root_nodes = build_tree(ns, rs)
        # 获取所有路径描述的字符串数组
        paths = get_paths(root_nodes)
        # 打印路径
        return paths

    def _build_tree_arr(self, result):
        # Process the result
        ns = []
        rs = []
        n_id_set = set()
        rs_id_set = set()
        for record in result:
            n = record["n"]
            if n.id not in n_id_set:
                n_id_set.add(n.id)
                ns.append({"id": n.id, "name": n.get("name")})
            for r in record["r"]:
                if r.id not in rs_id_set:
                    rs_id_set.add(r.id)
                    rs.append({"start_id": r.start_node.id, "end_id": r.end_node.id, "description": r.type})
            o = record["o"]
            if o.id not in n_id_set:
                n_id_set.add(o.id)
                ns.append({"id": o.id, "name": o.get("name")})
        return ns, rs

    def get_redis_user_session_key(self, session_id: str):
        app_info = AppConfig().config["APP"]
        session_key = f'{app_info["APP_NAME"]}:{env_import().get_env()}:session:{session_id}'
        return session_key

    def get_session_qa_list(self, session_id: str) -> list[SaveAnswerData]:
        session_key = self.get_redis_user_session_key(session_id)
        # 获取现有的会话数据
        session_data = self.redis_client.redis.lrange(session_key, 0, -1)
        session_qa_list = [SaveAnswerData(**json.loads(entry)) for entry in session_data]
        return session_qa_list

    def save_answer(self, data: SaveAnswerData):
        """
        保存答案
        :param data:
        :return:
        """
        session_key = self.get_redis_user_session_key(data.session_id)

        # 获取现有的会话数据
        session_data = self.redis_client.redis.lrange(session_key, 0, -1)
        session_data = [json.loads(entry) for entry in session_data]

        # 添加新的数据
        session_data.append(data.dict())

        # 只保留最新的 20 条记录
        if len(session_data) > 20:
            session_data = session_data[-20:]

        # 将更新后的会话数据保存回 Redis
        self.redis_client.redis.delete(session_key)  # 先删除旧的列表
        for entry in session_data:
            self.redis_client.redis.rpush(session_key, json.dumps(entry, ensure_ascii=False))  # 将每条记录推入列表尾部

        # 设置过期时间为 24 小时
        self.redis_client.redis.expire(session_key, timedelta(hours=24))

    def save_last_node_entity(self, docs, session_id):
        docs = [d.dict() for d in docs]
        self.redis_client.redis.set(self.get_last_node_entity_key(session_id), json.dumps(docs, ensure_ascii=False),
                                    ex=timedelta(hours=24))

    def get_last_node_entity(self, session_id) -> list[Document]:
        docs = []
        last_node_entity_key = self.get_last_node_entity_key(session_id)
        last_node_entity = self.redis_client.redis.get(last_node_entity_key)
        if last_node_entity:
            last_node_entity = json.loads(last_node_entity)
            if isinstance(last_node_entity, list):
                docs = [Document(**d) for d in last_node_entity]
        return docs

    def get_last_node_entity_key(self, session_id: str):
        app_info = AppConfig().config["APP"]
        last_node_entity_key = f'{app_info["APP_NAME"]}:{env_import().get_env()}:last_node_entity:{session_id}'
        return last_node_entity_key

    def is_ask_question(self, text_input, session_id) -> bool:
        dify_config = AppConfig().config["DIFY"]
        # API URL
        url = f'http://{dify_config["HOST"]}:{dify_config["PORT"]}{dify_config["WORKFLOWS_PATH"]}'

        # Headers
        headers = {
            'Authorization': f'Bearer {dify_config["IS_ASK_QUESTION_KEY"]}',
            'Content-Type': 'application/json'
        }

        user_list = self.get_session_qa_list(session_id)
        user_list = "\n".join([f"user：{u.user_question}\nanswer：{u.llm_answer}" for u in user_list])
        # Data
        data = {
            "inputs": {
                "user_list": user_list,
                "question": text_input
            },
            "response_mode": "blocking",
            "user": "abc-123"
        }

        # Send the POST request
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            logger.info(f'判断是否是追问返回的：{response_data}')
            text = response_data.get('data', {}).get('outputs', {}).get('text')

            # Parse the text
            if text is not None:
                result = True if text in '是' else False
                return result
            else:
                logger.error("No 'text' field in the response.")
        else:
            logger.error("Request failed with status code:", response.status_code)
            logger.error("Response:", response.text)

        return False

    def kg_nlp_query_prompt_fast(self, data):
        start_time = time.time()
        query_str = data.text_input
        query_embedding = self.corom_embedding.get_embedding(query_str)
        end_time = time.time()
        logger.info(f"文字向量化耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()
        dos = self.milvus_vector.search_by_vector(query_embedding, collection_name="kg_graph_path", top_k=200)
        end_time = time.time()
        logger.info(f"查询图耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()
        paths = []
        doc_log_info = []
        for d in dos:
            doc_log_info.append((d.page_content,d.metadata['score']))
            paths.append(d.page_content)
        logger.info(f"{doc_log_info}")
        dos = None
        end_time = time.time()
        logger.info(f"构建路径耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()
        re_rank_list = paths
        if not re_rank_list:
            return []
        # 使用方法
        if torch.cuda.is_available():
            logger.info("使用gpu推理")

            # rank = self.ranking_chinese.do_re_rank(src_s=f'{data.text_input}', re_rank_list=re_rank_list,is_onxx=True)
            rank = re_rank_with_multithreading(ranking_chinese=self.ranking_chinese,
                                               data_text_input=f'{data.text_input}',
                                               re_rank_list=re_rank_list,
                                               is_onxx=True)
        else:
            logger.info("使用cpu推理")
            rank = re_rank_with_multithreading(ranking_chinese=self.ranking_chinese,
                                               data_text_input=f'{data.text_input}',
                                               re_rank_list=re_rank_list)
        re_rank_list = None
        end_time = time.time()
        logger.info(f"rerank耗时: {end_time - start_time:.4f} 秒")

        # 获取分数 > 0.45 的
        # 获取最高分
        highest_score = max(r.score for r in rank)
        if highest_score > 1:
            highest_score = 1
        line_limit = highest_score * 0.85
        down_limit = 0.35
        rank_top = [r.text for r in rank if r.score > down_limit and r.score > line_limit]
        rank_top = rank_top[:20]
        rank_top = [ran.replace("，", "->") for ran in rank_top]
        # logger.info(f"top_texts:{rank_top}")
        # print(rank)
        return {
            "line_limit": line_limit,
            "down_limit": down_limit,
            "top_texts": rank_top,
            # "all_ranks": [r.dict() for r in rank],
            "is_ask_question": False
        }

    def kg_nlp_query_prompt_fast_v2(self, data):
        start_time = time.time()
        query_str = data.text_input
        query_embedding = self.corom_embedding.get_embedding(query_str)
        end_time = time.time()
        logger.info(f"文字向量化耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()
        dos = self.milvus_vector.search_by_vector(query_embedding, collection_name="kg_graph_path_v2", top_k=200)
        end_time = time.time()
        logger.info(f"查询图耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()
        paths = []
        chunk_ids = []
        for d in dos:
            # logger.info(f"{d.page_content}:{d.metadata['doc_id']} : {d.metadata['score']}")
            paths.append(d.page_content)
            chunk_ids.extend(d.metadata['chunk_id_list'])
        # TODO 获取文档的 chunk
        chunk_texts = []
        with self.neo4j_dao.get_session() as tx:
            chunk_noe4j_texts = self.neo4j_dao.get_chunk_by_chunk_ids(tx=tx, chunk_ids=chunk_ids)
            for chunk in chunk_noe4j_texts:
                chunk_texts.append(chunk['chunkText'])

        end_time = time.time()
        logger.info(f"构建路径耗时: {end_time - start_time:.4f} 秒")
        start_time = time.time()
        re_rank_list = paths
        if not re_rank_list:
            return []
        # 使用方法
        if torch.cuda.is_available():
            logger.info("使用gpu推理")

            rank = self.ranking_chinese.do_re_rank(src_s=f'{data.text_input}', re_rank_list=re_rank_list)
        else:
            logger.info("使用cpu推理")
            rank = re_rank_with_multithreading(ranking_chinese=self.ranking_chinese,
                                               data_text_input=f'{data.text_input}',
                                               re_rank_list=re_rank_list)
        end_time = time.time()
        logger.info(f"rerank耗时: {end_time - start_time:.4f} 秒")

        # 获取分数 > 0.45 的
        # 获取最高分
        highest_score = max(r.score for r in rank)
        line_limit = highest_score * 0.85
        down_limit = 0.35
        rank_top = [r.text for r in rank if r.score > down_limit and r.score > line_limit]
        rank_top = rank_top[:20]
        rank_top = [ran.replace("，", "->") for ran in rank_top]
        print(rank)
        return {
            "line_limit": line_limit,
            "down_limit": down_limit,
            "top_texts": rank_top,
            "all_ranks": [r.dict() for r in rank],
            "is_ask_question": False
        }
