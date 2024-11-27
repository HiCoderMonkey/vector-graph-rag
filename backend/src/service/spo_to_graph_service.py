import json
import logging
import re
import uuid

import requests
import sseclient
from pymilvus import MilvusException

from base.meta_singeton import MetaSingleton
from common.embedding.corom_embedding import CoromEmbedding
from dao.neo4j_dao import Neo4jDao
from db.my_redis.redis_client import RedisClient
from db.vdb.milvus.milvus_vector import MilvusConfig, MilvusVector
from dto.db_model import DocumentKG
from dto.document import Document
from env import Instance as env_import
from resource import AppConfig
from tqdm import tqdm
from dto.request_dto import LlmToSpoRequest, SpoToGraphReuqest
from common.http_request import http_request
from dao.kg_llm_dao import KgLLMDao
from service.spo_return_handle import tour_return_handle, common_return_handle, museum_return_handle
from common.utils import text_utils

logger = logging.getLogger(__name__)

spo_handle = {
    "common": common_return_handle,
    "tour": tour_return_handle,
    "museum": museum_return_handle
}


class SopToGraphService(metaclass=MetaSingleton):
    neo4j_dao: Neo4jDao = None
    redisClient: RedisClient = None
    corom_embedding: CoromEmbedding = None
    kg_llm_dao: KgLLMDao = None

    def __init__(self):
        self.neo4j_dao = Neo4jDao()
        self.redisClient = RedisClient()
        self.kg_llm_dao = KgLLMDao()
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

    def add_spo_to_graph(self, data, title: str = "", database: str = None):

        # 初始化进度条
        progress_bar = tqdm(data["spo"], desc="Adding SPO to graph", unit="record")
        total_size = len(data["spo"])
        percent_int = 0
        for record in progress_bar:
            subject = record["subject"]
            subject["name"] = subject["name"].replace("+", "_").replace("、", ",").replace(" ", "")
            subject["type"] = subject["type"].replace(" ", "")
            relation = record["relation"].replace("-", "_").replace("(", "").replace(")", "").replace("+",
                                                                                                      "_").replace(
                "、", ",").replace("%", "")
            obj = record["object"]
            obj["name"] = obj["name"].replace("+", "_").replace("、", ",").replace(" ", "")
            obj["type"] = obj["type"].replace(" ", "")

            if text_utils.is_blank_string(obj["name"]) or text_utils.is_blank_string(
                    subject["name"]) or text_utils.is_blank_string(relation) or text_utils.is_blank_string(
                subject["type"]) or text_utils.is_blank_string(obj["type"]):
                # 空白的跳过
                continue

            if not subject["type"] in ["Tour", "Destination"]:
                s_name = f"{title}:{subject['name']}"
            else:
                s_name = subject['name']
            subject_id = self.get_or_create_id(subject["type"], s_name)
            s_same = self.get_similarity_node(name=subject["name"])
            if s_same:
                subject["id"] = s_same.metadata["doc_id"]
                subject["type"] = s_same.metadata["node_label"]
            else:
                subject["id"] = subject_id
            if not obj["type"] in ["Tour", "Destination"]:
                s_obj = f"{title}:{obj['name']}"
            else:
                s_obj = obj['name']
            object_id = self.get_or_create_id(obj["type"], s_obj)
            obj["id"] = object_id
            o_smae = self.get_similarity_node(name=obj["name"])
            if o_smae:
                obj["id"] = o_smae.metadata["doc_id"]
                obj["type"] = o_smae.metadata["node_label"]
            else:
                obj["id"] = object_id
            with self.neo4j_dao.get_session(database=database) as session:
                session.write_transaction(self.neo4j_dao.create_entity_if_not_exists, subject["type"], subject["name"],
                                          subject_id)
                session.write_transaction(self.neo4j_dao.create_entity_if_not_exists, obj["type"], obj["name"],
                                          object_id)
                session.write_transaction(self.neo4j_dao.create_relationship_by_id, subject, relation, obj)

            # 节点关系数据保存到向量数据库
            docs = []
            if not subject["type"] in ["Day"]:
                docs.append(Document(page_content=subject["name"],
                                     metadata={"doc_id": subject["id"], "id": subject_id,
                                               "node_label": subject["type"]}))
            if not obj["type"] in ["Day"]:
                docs.append(
                    Document(page_content=obj["name"],
                             metadata={"doc_id": obj["id"], "id": object_id, "node_label": obj["type"]}))
            if docs:
                embeddings = [self.corom_embedding.get_embedding(d.page_content) for d in docs]
                self.milvus_vector.create(texts=docs, embeddings=embeddings,
                                          collection_name=self.milvus_config["NODE_COLL"], database=database)
            # 关系保存到向量数据库
            docs = [Document(page_content=relation, metadata={"doc_id": relation, "relationship_type": relation})]
            embeddings = [self.corom_embedding.get_embedding(d.page_content) for d in docs]
            self.milvus_vector.create(texts=docs, embeddings=embeddings,
                                      collection_name=self.milvus_config["RELATIONSHIP_COLL"], database=database)

            # 更新进度条
            progress_bar.set_postfix({"subject": subject["name"], "relation": relation, "object": obj["name"]})
            percent_int += 1
            yield round((percent_int / total_size) * 100, 0)

    def get_or_create_id(self, type, name, similarity_node: bool = False) -> str:
        # 以前是redis的
        app_info = AppConfig().config["APP"]
        key = f'{app_info["APP_NAME"]}:{env_import().get_env()}:node_id:{type}:{name}'
        uid = str(uuid.uuid4()).replace("-", "")
        if not self.redisClient.redis.exists(key):
            self.redisClient.redis.set(key, uid)
        return self.redisClient.redis.get(key)

    def get_similarity_node(self, name):
        try:
            query_embedding = self.corom_embedding.get_embedding(name)
            dos = self.milvus_vector.search_by_vector(query_embedding, collection_name="kg_node_coll", top_k=1,
                                                      score_threshold=87)
            if dos:
                d = dos[0]
                logger.info(f"相似节点:{d.page_content}-{d.metadata['doc_id']}")
                return d
            else:
                return None
        except MilvusException as e:
            if  e.code == 4:
                logger.error("还没有向量集合～～")
                return None
            else:
                raise e
        except Exception as e:
            raise e

    def llm_to_spo(self, data: LlmToSpoRequest):
        # 查询用的是什么spo机器人
        robot_info = self.kg_llm_dao.get_spo_llm_robot_config_by_id(id=data.spo_llm_robot_id)
        # 查询文档所有分片
        dify_config = AppConfig().config["DIFY"]
        api_key = dify_config["DATASETS_KEY"]  # 替换为你的 API 密钥
        url = f'{dify_config["HTTP_HOST"]}/v1/datasets/{data.datasets_id}/documents/{data.document_id}/segments'
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        data_spo = http_request.http_get(url, headers)
        spos = []
        for d in data_spo["data"]:
            answer = ""
            for s in self.llm_to_spo_item(d["content"], robot_info.key):
                if s["type"] == "item":
                    yield s
                if s["type"] == "all":
                    answer = s["answer"]
            handle_method = spo_handle.get(robot_info.return_handle_method, common_return_handle)
            logger.info(f"question:{d['content']}")
            logger.info(f"answer:{answer}")
            spo = handle_method(json_text=answer, root_obj=data.doc_name)
            spos.extend([{"subject": s.subject.dict(), "relation": s.relation, "object": s.object.dict()} for s in spo])
        spo_json = {
            "spo": spos
        }
        documentKG = DocumentKG(
            id=data.document_id,
            database_id=data.datasets_id,
            spo_json=spo_json,
            is_saved_to_graph_db=False
        )
        self.kg_llm_dao.save_or_update_spo_json(documentKG)
        yield {"type": "end", "answer": ""}

    def llm_to_spo_item(self, text: str, api_key: str):
        dify_config = AppConfig().config["DIFY"]
        url = f'{dify_config["HTTP_HOST"]}/v1/chat-messages'

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            "inputs": {},
            "query": text,
            "response_mode": "streaming",
            "conversation_id": "",
            "user": "abc-123"
        }

        response = requests.post(url, headers=headers, json=data, stream=True)

        client = sseclient.SSEClient(response)
        all_answer = ""
        for event in client.events():
            print(f"Received event: {event.event}")
            if event.event == 'message':
                # print(event.data)
                try:
                    message_data = json.loads(event.data)
                    answer = message_data.get('answer', '')
                    if message_data.get("event", "message_end") == "message_end":
                        break
                    all_answer += answer  # 将每个部分回答连接起来
                    yield {"type": "item", "answer": answer}
                    print(f"Received partial answer: {answer}")
                except Exception as e:
                    logger.error(f'error:{e}, event.data:{event.data}')

            elif event.event == 'message_end':
                print("Received final message.")
                message_data = json.loads(event.data)
                # Optionally, you can process metadata here if needed
                metadata = message_data.get('metadata', {})
                print(f"Metadata: {metadata}")
            else:
                print(f"Received event: {event.event}")

        response.close()
        logger.info(f"all_answer:{all_answer}")
        # 使用正则表达式提取内容
        try:
            all_answer = re.search(r'\`\`\`json\n(.*?)\n\`\`\`', all_answer, re.DOTALL).group(1)
        except Exception as e:
            logger.error(f"error:{e}, all_answer:{all_answer}")

        yield {"type": "all", "answer": all_answer}

    def spo_to_graph(self, data: SpoToGraphReuqest):
        # 查询spojson
        document_kg = self.kg_llm_dao.get_document_kg_by_id(data.document_id)
        if not document_kg or not document_kg.spo_json:
            return
        for data_percent in self.add_spo_to_graph(document_kg.spo_json, title=data.doc_name):
            yield data_percent
        # 修改状态为已入库
        self.kg_llm_dao.update_document_kg(
            document_kg=DocumentKG(**{"id": data.document_id, "is_saved_to_graph_db": True}))
        yield 100
