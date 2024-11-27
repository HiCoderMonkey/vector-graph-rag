import json
import logging

from base.meta_singeton import MetaSingleton
from service.spo_to_graph_service import SopToGraphService
from dto.request_dto import LlmToSpoRequest, SpoToGraphReuqest

logger = logging.getLogger(__name__)


class SpoToGraphController(metaclass=MetaSingleton):
    service: SopToGraphService = None
    def __init__(self):
        self.service = SopToGraphService()

    def llm_to_spo(self, data:LlmToSpoRequest):
        for data in self.service.llm_to_spo(data):
            yield f'data: {json.dumps(data, ensure_ascii=False)}\n\n'

    def spo_to_graph(self, data:SpoToGraphReuqest):
        for data in self.service.spo_to_graph(data):
            yield f'data: {json.dumps({"percent": data}, ensure_ascii=False)}\n\n'

