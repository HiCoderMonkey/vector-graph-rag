from pydantic import BaseModel

class EntityData(BaseModel):
    type: str
    name: str

class SpoItemData(BaseModel):
    subject: EntityData
    relation: str
    object: EntityData

