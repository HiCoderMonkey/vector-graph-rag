import logging

from neo4j import Session

from base.meta_singeton import MetaSingleton
from db.neo4j.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class Neo4jDao(metaclass=MetaSingleton):
    neo4j_client: Neo4jClient = None

    def __init__(self):
        self.neo4j_client = Neo4jClient()

    def get_session(self, database=None) -> Session:
        if database:
            return self.neo4j_client.driver.session(database=database)
        else:
            return self.neo4j_client.driver.session()

    def create_entity_if_not_exists(self, tx, entity_type, name, id):
        result = tx.run(f"""
            MERGE (e:{entity_type} {{id: $id}})
            ON CREATE SET e.name = $name
            RETURN e.id
        """, name=name, id=id)
        return result.single()[0]

    def create_node(self,tx, uri, labels, properties):
        cypher = "MERGE (n { uri: $uri }) SET n += $properties"
        tx.run(cypher, uri=uri, properties=properties)
        for label in labels:
            tx.run("MATCH (n { uri: $uri }) SET n :`" + label + "`", uri=uri)

    def create_relationship(self,tx, start_uri, end_uri, rel_type):
        cypher = """
        MATCH (a { uri: $start_uri })
        MATCH (b { uri: $end_uri })
        MERGE (a)-[r:`""" + rel_type + """`]->(b)
        """
        tx.run(cypher, start_uri=start_uri, end_uri=end_uri)

    def create_relationship_by_id(self, tx, subject, relation, obj):
        query = f"""
            MATCH (a {{id: $subject_id}}), (b {{id: $object_id}})
            MERGE (a)-[r:{relation}]->(b)
            RETURN type(r)
        """
        tx.run(query, subject_id=subject['id'], object_id=obj['id'])

    def get_path(self, tx, ids, path_str: str, relations):
        if relations:
            query = f"""
            MATCH {path_str}
            WHERE n.id in $ids
            AND ANY(rel IN r WHERE type(rel) IN $relations)
            RETURN n,r,o
            """
            logger.info(f"query: {query},params: {ids}, {relations}")
            result = tx.run(query, ids=ids, relations=relations)
        else:
            query = f"""
            MATCH {path_str}
            WHERE n.id in $ids
            RETURN n,r,o
            """
            logger.info(f"query: {query},params: {ids}")
            result = tx.run(query, ids=ids)
        return result

    def get_node_list(self,tx, skip: int = 0, limit: int = 500):
        """
        todo 记录一下这个查询语句，可以把路径文本频出来的
        MATCH p=(n1)-[r*]->(n2)
        WITH [x IN nodes(p) | x.name] AS nodeNames, [y IN relationships(p) | type(y)] AS relTypes
        RETURN reduce(output = nodeNames[0], i IN range(0, size(relTypes) - 1) | output + "," + relTypes[i] + "," + nodeNames[i + 1]) AS path
        SKIP 500 LIMIT 500
        """
        query = f"""
                MATCH (n)-[r]-(o)
                RETURN n,r,o
                skip $skip limit $limit
                """
        result = tx.run(query, skip=skip, limit=limit)
        return result

    def get_node_list_new(self, tx, start_id=-1, limit=500):
        query = f"""
            MATCH p=(n1)-[r*..3]->(n2)
            WHERE NOT (n1:Chunk OR n1:Document) AND NOT (n2:Chunk OR n2:Document) AND id(n1)>$start_id
            WITH [x IN nodes(p) | CASE WHEN x.name IS NULL OR x.name = "" THEN x.id ELSE x.name END] AS nodeNames,
                 id(n1) AS nodeIds,
                 [x IN nodes(p) | id(x)] AS all_node_ids,
                 [y IN relationships(p) | type(y)] AS relTypes,
                 [x IN nodes(p) | x] AS allNodes
            RETURN reduce(output = nodeNames[0], i IN range(0, size(relTypes) - 1) | output + "," + relTypes[i] + "," + nodeNames[i + 1]) AS path,
                   nodeIds AS start_ids,
                   all_node_ids AS all_node_ids
            LIMIT $limit
        """
        result = tx.run(query, start_id=start_id, limit=limit)
        return result

    def get_chunk_by_node_ids(self,tx,node_ids:list):
        query = f"""
            MATCH (node)<-[:HAS_ENTITY]-(chunk:Chunk)
            where id(node) IN $node_ids
            WITH id(node) AS nodeId,
                 id(chunk) as chunkId
            return nodeId,chunkId
        """
        result = tx.run(query,node_ids=node_ids)
        return result

    def get_chunk_by_chunk_ids(self,tx,chunk_ids:list):
        query = f"""
            MATCH (chunk:Chunk)
            where id(chunk) IN $chunk_ids
            WITH chunk.text as chunkText
            return chunkText
        """
        result = tx.run(query,chunk_ids=chunk_ids)
        return result
