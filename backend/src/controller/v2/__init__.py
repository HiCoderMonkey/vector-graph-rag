import asyncio
import logging

from fastapi import APIRouter

from dto.request_dto import InputData

from controller.v1.kg_nlp_query_controller import KgNlpQueryController

logger = logging.getLogger(__name__)

router = APIRouter()

kg_nlp_query_controller = KgNlpQueryController()

@router.post("/kg-nlp-query/prompt-fast")
async def kg_nlp_query_prompt_fast(data: InputData):
    try:
        logger.info(data.dict())
        result = await asyncio.to_thread(kg_nlp_query_controller.kg_nlp_query_prompt_fast_v2,data=data)
        logger.info(result)
        return result
    except Exception as e:
        logger.error(e)
        raise
