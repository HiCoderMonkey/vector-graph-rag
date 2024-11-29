import logging

from base.meta_singeton import MetaSingleton
import redis

logger = logging.getLogger(__name__)

class RedisClient(metaclass=MetaSingleton):
    def __init__(self,host, port, db,password):
        try:
            self.redis = redis.StrictRedis(host=host, port=port, db=db, password=password, decode_responses=True)
            # Perform a ping to check the connection
            if self.redis.ping():
                logger.info("Redis connection successful")
        except redis.ConnectionError as e:
            raise e