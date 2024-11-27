import json

import requests

from base.meta_singeton import MetaSingleton
from dto.request_dto import DocsQueryParams
from resource import AppConfig
from dao.kg_llm_dao import KgLLMDao
from dao.dify_dao import DifyDao


class DocsService(metaclass=MetaSingleton):
    kg_llm_dao: KgLLMDao = None
    dify_dao: DifyDao = None

    def __init__(self):
        self.kg_llm_dao = KgLLMDao()
        self.dify_dao = DifyDao()

    def query_docs(self, dataset_id: str, params: DocsQueryParams):
        dify_config = AppConfig().config["DIFY"]
        # 定义请求的 URL 和 API 密钥
        dataset_id = dataset_id  # 替换为你的 dataset_id
        api_key = dify_config["DATASETS_KEY"]  # 替换为你的 API 密钥

        url = f'{dify_config["HTTP_HOST"]}/v1/datasets/{dataset_id}/documents'
        headers = {
            'Authorization': f'Bearer {api_key}'
        }

        # 发送 GET 请求
        response = requests.get(url, headers=headers)
        res = {
            "data": [],
            "has_more": False,
            "limit": 20,
            "total": 0,
            "page": 0
        }

        # 检查请求是否成功
        if response.status_code == 200:
            # 解析 JSON 响应
            data = response.json()

            # 查询入库状态
            doc_ids = [doc['id'] for doc in data['data']]
            docs_results = self.kg_llm_dao.get_kg_docs_by_ids(doc_ids)
            docs_results = {d["id"]: d for d in docs_results}

            # 提取名称字段
            res["data"] = [{
                "id": doc.get("id"),
                "name": doc.get("name"),
                "is_saved_to_graph_db": docs_results.get(doc.get("id"))["is_saved_to_graph_db"] if docs_results.get(
                    doc.get("id")) else False,
                "is_spo": True if docs_results.get(doc["id"]) and docs_results[doc["id"]]["spo_json"] else False
            } for doc in data['data']]
            res["has_more"] = data['has_more']
            res["limit"] = data['limit']
            res["total"] = data['total']
            res["page"] = data['page']

        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(response.text)

        return res

    def doc_detail(self, doc_id):
        doc = self.kg_llm_dao.get_document_kg_by_id(doc_id)
        dif_doc = self.dify_dao.get_dif_doc_by_id(doc_id)
        doc_name = dif_doc.name if dif_doc else ""
        last_dot_index = doc_name.rfind(".")
        if last_dot_index != -1:
            doc_name = doc_name[:last_dot_index]
        return {
            "id": dif_doc.id,
            "name": doc_name,
            "dataset_id": dif_doc.dataset_id,
            "spo_json": doc.spo_json if doc else None ,
            "is_saved_to_graph_db": doc.is_saved_to_graph_db if doc else False
        }
