import os

from common.ai_models.ai_model_base import AiModelBase


class AiModelModelScope(AiModelBase):

    def __init__(self, model_id: str, model_revision: str = None):
        self._model_revision = model_revision
        super().__init__(model_id=model_id, model_revision=model_revision)

    def _download_embedding_model_by_modelscope(self):
        """
            Download embedding model by modelscope.
            """
        model_cache_dir = os.path.expanduser(f'{self._model_path}/{self._model_id}')

        if os.path.exists(model_cache_dir):
            print(f'Model {self._model_id} has already been downloaded.')
            return
        # 模型下载
        from modelscope import snapshot_download
        snapshot_download(self._model_id, cache_dir=f'{self._model_path}', revision=self._model_revision)

