import logging

from base.meta_singeton import MetaSingleton
from dto.request_dto import InputData, SaveAnswerData
from service.kg_nlp_query_service import KgNlpQueryService

logger = logging.getLogger(__name__)


class KgNlpQueryController(metaclass=MetaSingleton):
    service: KgNlpQueryService = None
    def __init__(self):
        self.service = KgNlpQueryService()

    def kg_nlp_query_prompt(self,data: InputData):
        prompt = self.service.kg_nlp_query_prompt(data)
        return prompt


    def save_answer(self,data: SaveAnswerData):
        logger.info(f"保存回答:{data.dict()}")
        self.service.save_answer(data)
        return {"msg":"ok"}

    def kg_nlp_query_prompt_fast(self, data):
        prompt = self.service.kg_nlp_query_prompt_fast(data)
        return prompt

    def kg_nlp_query_prompt_fast_v2(self, data):
        prompt = self.service.kg_nlp_query_prompt_fast_v2(data)
        return prompt
