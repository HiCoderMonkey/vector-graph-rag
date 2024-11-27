import logging
from typing import List

from langchain_core.embeddings import Embeddings
from modelscope import pipeline, Tasks

from common import ai_models
from common.ai_models import ai_model_factory
from common.embedding.embedding_base import EmbeddingBase

logger = logging.getLogger(__name__)

class CoromEmbedding(EmbeddingBase,Embeddings):
    def __init__(self):
        super().__init__(dim=768)
        self.pipeline_se = pipeline(Tasks.sentence_embedding, model=self._ai_model_base.load_model())
        # 先推理一下，后面接口就会快
        self.get_embeddings("哈哈哈")
        logger.info("embedding推理一下，后面接口就会快！！！")

    def _init_ai_model_base(self):
        self._ai_model_base = ai_model_factory.BuildAiModel(model_platform=ai_models.ModelPlatform.MODELSCOPE,
                                                            model_id='iic/nlp_corom_sentence-embedding_chinese-base')

    def get_embedding(self, text: str) -> list[float]:
        # 当输入仅含有soure_sentence时，会输出source_sentence中每个句子的向量表示以及首个句子与其他句子的相似度。
        result = []
        inputs = {
            "source_sentence": [text]
        }
        result_se = self.pipeline_se(input=inputs)
        return result_se['text_embedding'][0].tolist()

    def embed_documents(
        self, texts: List[str], batch_size: int = 0
    ) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List[str] The list of texts to embed.
            batch_size: [int] The batch size of embeddings to send to the model.
                If zero, then the largest batch size will be detected dynamically
                at the first request, starting from 250, down to 5.

        Returns:
            List of embeddings, one for each text.
        """
        result = []
        inputs = {
            "source_sentence": texts
        }
        result_se = self.pipeline_se(input=inputs)
        result = [item.tolist() for item in result_se['text_embedding']]
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.get_embedding(text)



