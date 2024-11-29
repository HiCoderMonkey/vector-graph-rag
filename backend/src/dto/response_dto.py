from typing import Optional

from pydantic import BaseModel, Field

class RerankItemResponse(BaseModel):
    id: str
    score: float

class RerankResponse(BaseModel):
    datas : list[RerankItemResponse]
