from common.ai_models import ModelPlatform
from common.ai_models.ai_model_base import AiModelBase
from common.ai_models.mdoelscope.modelscope import AiModelModelScope


def BuildAiModel(model_platform: ModelPlatform, model_id: str, model_revision: str = None) -> AiModelBase:
    if model_platform == ModelPlatform.MODELSCOPE:
        # model_scope
        return AiModelModelScope(model_id=model_id, model_revision=model_revision)
