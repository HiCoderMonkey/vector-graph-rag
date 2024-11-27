import logging

from base.meta_singeton import MetaSingleton
from common.rerank.ranking_chinese_base.ranking_chinese import RankingChinese
from dto.request_dto import DoRerankRequest
from dto.response_dto import RerankResponse, RerankItemResponse

logger = logging.getLogger(__name__)


class RerankController(metaclass=MetaSingleton):
    ranking_chinese: RankingChinese = None

    def __init__(self):
        # ranking模型
        self.ranking_chinese = RankingChinese()

    def do_rerank_rank(self, data: DoRerankRequest) -> RerankResponse:
        datas = []
        for item in data.datas:
            rank = self.ranking_chinese.do_re_rank(item.source,[item.comparison])
            score = round(rank[0].score,2)
            datas.append(RerankItemResponse(id=item.id, score=score))
        return RerankResponse(datas=datas)
