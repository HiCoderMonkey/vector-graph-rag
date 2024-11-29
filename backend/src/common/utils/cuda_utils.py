import gc
import logging

import torch

logger = logging.getLogger(__name__)

def do_gc():
    logger.info("清理缓存")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("清理GPU缓存GPU内存")