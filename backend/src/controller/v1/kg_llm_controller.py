import logging

from base.meta_singeton import MetaSingleton
from service.kg_llm_service import KgLmmService

logger = logging.getLogger(__name__)


class KgLlmController(metaclass=MetaSingleton):
    service: KgLmmService = None

    def __init__(self):
        self.service = KgLmmService()

    def get_spo_robots(self):
        return self.service.get_spo_robots()
