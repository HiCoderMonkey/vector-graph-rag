import logging

from sqlalchemy.orm import joinedload

from base.meta_singeton import MetaSingleton
from db.postgre_sql.pg_client import PgClient, sqlalchemy_to_dict, to_plain_object
from dto.db_model import DocumentKG, SpoLlmRobotConfig
from resource import AppConfig

logger = logging.getLogger(__name__)


class KgLLMDao(metaclass=MetaSingleton):
    pg_client: PgClient

    def __init__(self):
        pg_config = AppConfig().config["POSTGRESQL"]
        self.pg_client = PgClient(dbname=pg_config["KG_DB"], user=pg_config["USER"], password=pg_config["PASSWORD"],
                                  host=pg_config["HOST"], port=pg_config["PORT"])

    def get_kg_docs_by_ids(self, ids: list[str]) -> list[DocumentKG]:
        with self.pg_client.get_session() as session:
            res_data = []
            res = session.query(DocumentKG).filter(DocumentKG.id.in_(ids)).all()
            for r in res:
                res_data.append(sqlalchemy_to_dict(r))
            return res_data

    def get_spo_llm_robot_config_by_id(self, id: str) -> SpoLlmRobotConfig:
        with self.pg_client.get_session() as session:
            spoLlmRobotConfig = session.query(SpoLlmRobotConfig).filter_by(id=id).first()
            if spoLlmRobotConfig:
                spoLlmRobotConfig = to_plain_object(spoLlmRobotConfig)
                return spoLlmRobotConfig

    def save_or_update_spo_json(self, documentKG: DocumentKG):
        with self.pg_client.get_session() as session:
            instance = session.query(DocumentKG).filter_by(id=documentKG.id).first()
            if instance:
                instance.spo_json = documentKG.spo_json  # 只更新 spo_json 字段
            else:
                instance = documentKG
                session.add(instance)

    def update_document_kg(self, document_kg: DocumentKG):
        with self.pg_client.get_session() as session:
            instance = session.query(DocumentKG).filter_by(id=document_kg.id).first()
            if instance:
                instance.is_saved_to_graph_db = document_kg.is_saved_to_graph_db  # 更新入库字段

    def get_document_kg_by_id(self, id: str) -> DocumentKG:
        with self.pg_client.get_session() as session:
            instance = session.query(DocumentKG).filter_by(id=id).first()
            if instance:
                instance = to_plain_object(instance)
            return instance

    def get_all_spo_llm_robot(self):
        with self.pg_client.get_session() as session:
            res = []
            spoLlmRobotConfigs = session.query(SpoLlmRobotConfig).all()
            if spoLlmRobotConfigs:
                for  d in spoLlmRobotConfigs:
                    res.append(sqlalchemy_to_dict(d))
            return res
