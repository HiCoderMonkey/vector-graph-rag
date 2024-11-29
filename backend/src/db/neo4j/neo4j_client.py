from neo4j import GraphDatabase

from base.meta_singeton import MetaSingleton


class Neo4jClient(metaclass=MetaSingleton):
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()