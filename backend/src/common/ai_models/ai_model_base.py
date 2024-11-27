import os
from abc import ABC, abstractmethod
from env import get_root_dir

class AiModelBase(ABC):
    _model_id: str = None
    _model_path: str = None
    _model_revision: str = None

    def __init__(self, model_id: str, model_revision: str = None, **kwargs):
        self._model_id = model_id
        self._model_revision = model_revision
        # 获取项目根路径
        current_file_path = get_root_dir()
        # 获取当前文件所在的目录
        current_dir = os.path.dirname(current_file_path)
        self._model_path = kwargs.get("model_path", current_dir + "/models")

        # 下载模型
        self._download_embedding_model_by_modelscope()

    @abstractmethod
    def _download_embedding_model_by_modelscope(self):
        raise NotImplementedError

    def load_model(self) -> str:
        """
        获取模型的路径
        """
        return f'{self._model_path}/{self._model_id}'
