from typing import Optional

from pydantic import BaseModel, Field


class InputData(BaseModel):
    text_input: str = Field(..., description="text_input是必须的")  # 假设你接收一个包含输入文本的参数
    session_id: str = Field(..., description="session_id是必须的")  #

class InputRanerData(BaseModel):
    text_input: str = Field(..., description="text_input是必须的")  # 假设你接收一个包含输入文本的参数
    model_path: str = Field(..., description="model_path是必须的")  #

class SaveAnswerData(BaseModel):
    llm_answer: str = Field(..., description="llm_answer是必须的")
    user_question: str = Field(..., description="user_question是必须的")
    session_id: str = Field(..., description="session_id是必须的")

class DocsQueryParams(BaseModel):
    keyword: Optional[str] = Field(None, description="搜索关键词，可选，目前仅搜索文档名称")
    page: Optional[int] = Field(1, description="页码，可选", gt=0)
    limit: Optional[int] = Field(20, description="返回条数，可选，默认 20，范围 1-100", ge=1, le=100)

class LlmToSpoRequest(BaseModel):
    document_id: str = Field(..., description="document_id是必须的")
    datasets_id: str = Field(..., description="datasets_id是必须的")
    spo_llm_robot_id: str = Field(..., description="spo_llm_robot_id是必须的")
    doc_name: str = Field(..., description="doc_name是必须的")

class SpoToGraphReuqest(BaseModel):
    document_id: str = Field(..., description="document_id是必须的")
    doc_name: str = Field(..., description="doc_name是必须的")

class DoRerankItemRequest(BaseModel):
    id: str = Field(..., description="id是必须的")
    source: str = Field(..., description="source是必须的")
    comparison: str = Field(..., description="comparison是必须的")

class DoRerankRequest(BaseModel):
    datas: list[DoRerankItemRequest] = Field(..., description="datas是必须的")
