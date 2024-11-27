import asyncio
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from modelscope import pipeline, Tasks

from common.raner.nlp_raner import NlpRaNer
from controller.v1.rerank_controller import RerankController
from controller.v1.spo_to_graph_controller import SpoToGraphController
from dto.request_dto import InputData, SaveAnswerData, DocsQueryParams, LlmToSpoRequest, SpoToGraphReuqest, \
    InputRanerData, DoRerankRequest
from controller.v1.kg_nlp_query_controller import KgNlpQueryController
from controller.v1.docs_controller import DocsController
from controller.v1.kg_llm_controller import KgLlmController
from common.utils import cuda_utils
from dto.response_dto import RerankResponse

logger = logging.getLogger(__name__)

router = APIRouter()

kg_nlp_query_controller = KgNlpQueryController()
docs_controller = DocsController()
spo_to_graph_controller = SpoToGraphController()
kg_llm_controller = KgLlmController()
rerank_controller = RerankController()

# 创建一个信号量，最大值为 10，并且每个任务执行完之后释放一个信号量即同时只能有 10 个请求进入
semaphore = asyncio.Semaphore(10)


@router.post("/kg-nlp-query/prompt")
def kg_nlp_query_prompt(data: InputData):
    return kg_nlp_query_controller.kg_nlp_query_prompt(data=data)


@router.post("/kg-nlp-query/prompt-fast")
async def kg_nlp_query_prompt_fast(data: InputData):
    try:
        async with semaphore:  # 限流
            logger.info(data.dict())
            result = await asyncio.to_thread(kg_nlp_query_controller.kg_nlp_query_prompt_fast, data=data)
            logger.info(result)
            return result
    except Exception as e:
        logger.error(e)
        raise e  # 使用 HTTPException 处理错误


@router.post("/rerank/do_reranks")
async def get_kg_nlp_query_prompt_fast(data: DoRerankRequest) -> RerankResponse:
    return rerank_controller.do_rerank_rank(data)


@router.post("/kg-nlp-query/save-answer")
def save_answer(data: SaveAnswerData):
    return kg_nlp_query_controller.save_answer(data=data)


@router.get("/datasets/{dataset_id}/documents")
def get_documents(dataset_id: str, params: DocsQueryParams = Depends()):
    return docs_controller.query_docs(dataset_id, params)


@router.post("/doc/llm-to-spo")
def llm_to_spo(data: LlmToSpoRequest):
    event_stream = spo_to_graph_controller.llm_to_spo(data)
    return StreamingResponse(event_stream, media_type="text/event-stream")


@router.post("/doc/spo-to-graph")
def spo_to_graph(data: SpoToGraphReuqest):
    event_stream = spo_to_graph_controller.spo_to_graph(data)
    return StreamingResponse(event_stream, media_type="text/event-stream")


@router.get("/doc/{doc_id}/detail")
def doc_detail(doc_id: str):
    return docs_controller.doc_detail(doc_id)


@router.get("/doc/get-spo-robots")
def get_spo_robots():
    return kg_llm_controller.get_spo_robots()


nlp_raner_path = None
nlp_raner = None


@router.post("/test-raner")
def test_raner(data: InputRanerData):
    global nlp_raner_path
    global nlp_raner
    if nlp_raner_path == None or nlp_raner_path != data.model_path:
        nlp_raner_path = data.model_path
        nlp_raner = pipeline(Tasks.named_entity_recognition,
                             model=data.model_path)
    result = nlp_raner(data.text_input)
    if result and result.get('output', []):
        result['output'] = [{
            "span": o['span'],
            "typpe": o['type'],
            "prob": float(o['prob']),
            "start": int(o['start']),
            "end": int(o['end'])
        } for o in result['output']]
    return result
