import os
from abc import ABC, abstractmethod
from common.ai_models.ai_model_base import AiModelBase


class EmbeddingBase(ABC):
    _ai_model_base: AiModelBase = None

    def __init__(self, **kwargs):
        self.embedding_dim = kwargs["dim"]
        self._init_ai_model_base()

    @abstractmethod
    def _init_ai_model_base(self):
        raise NotImplementedError


    def get_dim(self):
        return self.embedding_dim

    def get_embeddings(self,text:str)->list[float]:
        pass