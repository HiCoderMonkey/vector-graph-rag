from base.meta_singeton import MetaSingleton

import env
import yaml


class AppConfig(metaclass=MetaSingleton):
    config:dict = None

    def __init__(self, config):
        self.config = config


def load_config() -> AppConfig:
    config = {}
    with open(env.Instance().get_env_config_file(), "r", encoding="utf-8") as f:
        config_env = yaml.safe_load(f)
        for k,v in config_env.items():
            config[k] = v
    return AppConfig(config)
