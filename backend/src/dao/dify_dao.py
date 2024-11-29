import logging

from base.meta_singeton import MetaSingleton
from db.postgre_sql.pg_client import PgClient, to_plain_object
from resource import AppConfig
from dto.db_model import Document

logger = logging.getLogger(__name__)


class DifyDao(metaclass=MetaSingleton):
    pg_client: PgClient

    def __init__(self):
        pg_config = AppConfig().config["POSTGRESQL"]
        self.pg_client = PgClient(dbname=pg_config["DIFY_DB"], user=pg_config["USER"], password=pg_config["PASSWORD"],
                                  host=pg_config["HOST"], port=pg_config["PORT"])

    def get_dif_doc_by_id(self, id: str)->Document:
        with self.pg_client.get_session() as session:
            instance = session.query(Document).filter_by(id=id).first()
            if instance:
                instance = to_plain_object(instance)
            return instance
