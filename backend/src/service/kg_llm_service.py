import json
import logging
from itertools import chain

import requests
from tqdm import tqdm

from base.meta_singeton import MetaSingleton
from common.embedding.corom_embedding import CoromEmbedding
from dao.kg_llm_dao import KgLLMDao
from dao.neo4j_dao import Neo4jDao
from common.utils.neo4j_utils import build_tree, get_paths
from db.vdb.milvus.milvus_vector import MilvusConfig, MilvusVector
from dto.document import Document
from resource import AppConfig

logger = logging.getLogger(__name__)


class PathEntity:
    def __init__(self, path: str, start_id: int, all_node_ids: list):
        self.path = path
        self.start_id = start_id
        self.all_node_ids = all_node_ids

    def set_chunk_list(self, chunkck_list):
        self.chunk_list = chunkck_list


def build_list_path(res):
    paths = []
    for r in res:
        path = PathEntity(path=r['path'], start_id=r['start_ids'], all_node_ids=r['all_node_ids'])
        paths.append(path)
    return paths


def group_chunk(result_chunk_list):
    group_chunk_by_nodeid = {}
    for r in result_chunk_list:
        if not group_chunk_by_nodeid.get(r['nodeId']):
            group_chunk_by_nodeid[r['nodeId']] = [r['chunkId']]
        else:
            group_chunk_by_nodeid[r['nodeId']].append(r['chunkId'])
    return group_chunk_by_nodeid


class KgLmmService(metaclass=MetaSingleton):
    kg_llm_dao: KgLLMDao = None
    neo4j_dao: Neo4jDao = None

    def __init__(self):
        self.kg_llm_dao = KgLLMDao()
        self.neo4j_dao = Neo4jDao()
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

    def get_spo_robots(self):
        robots = self.kg_llm_dao.get_all_spo_llm_robot()
        robots = [{"id": robot["id"], "name": robot["name"]} for robot in robots]
        return robots

    def get_all_path_to_vdb(self):
        """
        保存路径所有的path到vdb
        """
        limit = 100
        page = 0
        with tqdm(total=None, desc="保存路径所有的path到vdb") as bar:
            while True:
                with self.neo4j_dao.get_session() as tx:
                    result = self.neo4j_dao.get_node_list(tx=tx, skip=page * limit, limit=limit)
                    ns, rs = self._build_tree_arr(result)
                    if not ns or not rs:
                        break
                    page += 1
                root_nodes = build_tree(ns, rs)
                # 获取所有路径描述的字符串数组
                paths = get_paths(root_nodes)
                # logger.info(f'路径{paths}')
                # 保存路径到向量数据库
                docs = [Document(page_content=path, metadata={"doc_id": path}) for path in paths]
                embeddings = [self.corom_embedding.get_embedding(d.page_content) for d in docs]
                try:
                    index_params = {
                        'metric_type': 'IP',
                        'index_type': "IVF_FLAT",
                        # 'params': {"M": 8, "efConstruction": 64}
                        'params': {"nlist": 512}
                    }
                    self.milvus_vector.create(texts=docs, embeddings=embeddings, index_params=index_params,
                                              collection_name=f'{self.milvus_config["GRAPH_PATH_COLL"]}_new_2')
                except Exception as e:
                    logger.error(f"{e}")
                bar.update(1)

    def get_all_path_to_vdb_new(self, in_start_id=-1, limit=1000):
        """
        保存路径所有的path到vdb
        """
        page = 0
        start_id = in_start_id
        with tqdm(total=None, desc="保存路径所有的path到vdb") as bar:
            while True:
                with self.neo4j_dao.get_session() as tx:
                    result = self.neo4j_dao.get_node_list_new(tx=tx, start_id=start_id, limit=limit)
                    # TODO 获取最大ID，获取所有节点id，后面in查询关联的分片
                    paths = build_list_path(result)
                    all_node_id_set = set(chain.from_iterable(p.all_node_ids for p in paths))
                    result_chunk = self.neo4j_dao.get_chunk_by_node_ids(tx=tx, node_ids=list(all_node_id_set))
                    group_chunk_by_nodeid = group_chunk(result_chunk_list=result_chunk)
                    if not paths:
                        break
                    start_id = paths[-1].start_id
                # 获取所有路径描述的字符串数组
                # 保存路径到向量数据库
                print(f"当前id：{start_id}")
                docs = []
                for path in paths:
                    try:
                        if not path.path:
                            continue
                        chunk_list = []
                        for node_id in path.all_node_ids:
                            if group_chunk_by_nodeid.get(node_id):
                                chunk_list.extend(group_chunk_by_nodeid[node_id])
                        path.set_chunk_list(chunk_list)
                        docs.append(
                            Document(page_content=path.path, metadata={"doc_id": path.path, "chunk_id_list": chunk_list}))
                    except Exception as e:
                        print(path)
                        raise e
                if not docs:
                    continue
                embeddings = self.corom_embedding.get_embedding_list([d.page_content for d in docs])
                try:
                    index_params = {
                        'metric_type': 'IP',
                        'index_type': "IVF_FLAT",
                        # 'params': {"M": 8, "efConstruction": 64}
                        'params': {"nlist": 512}
                    }
                    self.milvus_vector.create(texts=docs, embeddings=embeddings, index_params=index_params,
                                              collection_name=f'{self.milvus_config["GRAPH_PATH_COLL"]}_new_3')
                except Exception as e:
                    logger.error(f"{e}")
                bar.update(1)

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
                name = n.get("name", n.get("id"))
                if not name:
                    name = f"{n.id}"
                ns.append({"id": n.id, "name": name})
            if isinstance(record["r"], list):
                for r in record["r"]:
                    if r.id not in rs_id_set:
                        rs_id_set.add(r.id)
                        rs.append({"start_id": r.start_node.id, "end_id": r.end_node.id, "description": r.type})
            else:
                r = record["r"]
                if r.id not in rs_id_set:
                    rs_id_set.add(r.id)
                    rs.append({"start_id": r.start_node.id, "end_id": r.end_node.id, "description": r.type})

            o = record["o"]
            if o.id not in n_id_set:
                n_id_set.add(o.id)
                ns.append({"id": o.id, "name": o.get("name")})
        return ns, rs
