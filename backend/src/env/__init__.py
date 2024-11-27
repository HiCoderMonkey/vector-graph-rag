# -*- coding=utf-8
import logging

from base.meta_singeton import MetaSingleton as meta_singeton_base
import os
import pkg_resources

PROD = 'prod'
TEST = 'test'
DEV = 'dev'
ENV_LIST = [PROD, TEST, DEV]
APP_ENV = 'app_env'

logger = logging.getLogger(__name__)

# 跟目录
ROOT_DIR = pkg_resources.resource_filename('env', '') + '/../../'
logger.info(f"ROOT_DIR:{ROOT_DIR}")


class Instance(metaclass=meta_singeton_base):
    app_env: str

    def __init__(self):
        app_env = os.environ.get(APP_ENV,"dev")
        logger.info(f"app_env:{app_env}")
        if app_env not in ENV_LIST:
            raise ValueError(f"app_env 配置错误！！！")
        self.app_env = app_env

    def is_prod(self):
        return self.app_env == PROD

    def is_test(self):
        return self.app_env == TEST

    def is_dev(self):
        return self.app_env == DEV

    def get_env(self):
        return self.app_env

    def get_env_config_file(self):
        return f'{ROOT_DIR}/resource/{self.app_env}.yml'

    def get_application_common_file(self):
        return f'{ROOT_DIR}/resource/application.yml'


# 获取根目录的文件夹
def get_root_dir() -> str:
    return ROOT_DIR
