# 初始化log
import logging
import subprocess
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager

import pynvml
import torch
import uvicorn

import env
from log.app_logging import init_log
from middlewares.trace_id_middlewares import TraceIDMiddleware

init_log()

from db.neo4j.neo4j_client import Neo4jClient
from env import Instance as EnvInstance
from resource import load_config, AppConfig
from db.my_redis.redis_client import RedisClient
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse


def init_app():
    # 初始化环境
    EnvInstance()
    # 初始化配置文件
    load_config()
    # 初始化redis
    redis_config = AppConfig().config["REDIS"]
    RedisClient(host=redis_config["HOST"], port=redis_config["PORT"], db=redis_config["DB"],
                password=redis_config["PASSWORD"])
    # 初始化neo4j
    neo4j_config = AppConfig().config["NEO4J"]
    Neo4jClient(uri=neo4j_config["URI"], user=neo4j_config["USER_NAME"], password=neo4j_config["PASSWORD"])


# 初始化配置什么的
init_app()

from fastapi import FastAPI
from controller.v1 import router as v1_router
from controller.v2 import router as v2_router
from common.utils import cuda_utils

logger = logging.getLogger(__name__)

def _gc():
    cuda_utils.do_gc()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc()


app = FastAPI(lifespan=lifespan)

IS_HEALTH = True

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"An exception occurred: {exc}")
    # 获取异常类型
    exc_type = type(exc).__name__

    # 获取堆栈跟踪
    stack_trace = traceback.format_exc()
    # 记录异常信息，包括类型和堆栈跟踪
    logger.error(exc,exc_info=True)
    # 如果 BackgroundTasks 存在，则添加任务
    _gc()

    return JSONResponse(
        status_code=500,
        content={"message": "An internal error occurred. Please try again later."}
    )

def check_gpu_memory(threshold_mb, check_interval=10):
    # 示例用法：每 10 秒检查一次 GPU 显存是否超过 3000 MB
    # 这个函数应该在一个单独的线程或进程中启动
    global IS_HEALTH
    logger.info("启动显存检查～～")
    pynvml.nvmlInit()  # 初始化 NVML
    try:
        while True:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 获取 GPU 0 的句柄
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / (1024 ** 2)  # 将字节转换为 MB

            if mem_used_mb > threshold_mb:
                logger.info(f"显存使用超过阈值 {threshold_mb} MB，正在退出...")
                IS_HEALTH = False

            time.sleep(check_interval)  # 等待一段时间后再检查
    except Exception as e:
        logger.error(f"显存检查发生错误：{e}")
    finally:
        pynvml.nvmlShutdown()  # 结束 NVML

def start_memory_monitor():
    monitor_thread = threading.Thread(target=lambda: check_gpu_memory(threshold_mb=24000))
    monitor_thread.daemon = True
    monitor_thread.start()


def init_router():
    # 添加中间件,加入traceid
    app.add_middleware(TraceIDMiddleware)

    #显存检测
    start_memory_monitor()

    app.include_router(v1_router, prefix="/v1", tags=["v1"])


# 初始化路由
init_router()

@app.get("/gc")
async def root(background_tasks: BackgroundTasks):
    background_tasks.add_task(_gc)
    return {"message": "gc ok！"}

@app.get("/health")
async def health():
    if IS_HEALTH:
        return {"message": "health"}
    else:
        # 如果健康检查失败，返回 500 状态码（或其他适当的状态码）
        raise HTTPException(status_code=500, detail="no health")

if __name__ == '__main__':
    if env.Instance().get_env() in ['dev','test-my']:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        uvicorn.run("main:app", host="0.0.0.0", workers=40, reload=True, port=8080)
