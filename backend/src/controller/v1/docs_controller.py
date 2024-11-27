import logging

from base.meta_singeton import MetaSingleton
from dto.request_dto import DocsQueryParams
from service.docs_service import DocsService

logger = logging.getLogger(__name__)


class DocsController(metaclass=MetaSingleton):
    service: DocsService = None

    def __init__(self):
        self.service = DocsService()

    def query_docs(self, dataset_id: str, params: DocsQueryParams):
        return self.service.query_docs(dataset_id=dataset_id, params=params)

    def doc_detail(self, doc_id):
        return self.service.doc_detail(doc_id=doc_id)
