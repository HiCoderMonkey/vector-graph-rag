import uuid

from sqlalchemy import Column, String, Boolean, JSON, Integer, UUID, Text, DateTime, Float, VARCHAR, func

from db.postgre_sql.pg_client import Base

# 定义模型
class DocumentKG(Base):
    __tablename__ = 'document_kg'

    id = Column(String(255), primary_key=True, nullable=False)
    database_id = Column(String(255), nullable=False)
    spo_json = Column(JSON, nullable=False)
    is_saved_to_graph_db = Column(Boolean, default=False)

    def __repr__(self):
        return f"<DocumentKG(id={self.id}, database_id={self.database_id}, is_saved_to_graph_db={self.is_saved_to_graph_db})>"

# 定义模型类
class SpoLlmRobotConfig(Base):
    __tablename__ = 'spo_llm_robot_config'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    key = Column(String(255), nullable=False)
    return_handle_method = Column(String(255), nullable=False)

class Document(Base):
    __tablename__ = 'documents'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    position = Column(Integer, nullable=False)
    data_source_type = Column(String(255), nullable=False)
    data_source_info = Column(Text)
    dataset_process_rule_id = Column(UUID(as_uuid=True))
    batch = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    created_from = Column(String(255), nullable=False)
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_api_request_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime(timezone=False), server_default=func.now(), nullable=False)
    processing_started_at = Column(DateTime(timezone=False))
    file_id = Column(Text)
    word_count = Column(Integer)
    parsing_completed_at = Column(DateTime(timezone=False))
    cleaning_completed_at = Column(DateTime(timezone=False))
    splitting_completed_at = Column(DateTime(timezone=False))
    tokens = Column(Integer)
    indexing_latency = Column(Float)
    completed_at = Column(DateTime(timezone=False))
    is_paused = Column(Boolean, default=False)
    paused_by = Column(UUID(as_uuid=True))
    paused_at = Column(DateTime(timezone=False))
    error = Column(Text)
    stopped_at = Column(DateTime(timezone=False))
    indexing_status = Column(VARCHAR(255), default='waiting', nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    disabled_at = Column(DateTime(timezone=False))
    disabled_by = Column(UUID(as_uuid=True))
    archived = Column(Boolean, default=False, nullable=False)
    archived_reason = Column(String(255))
    archived_by = Column(UUID(as_uuid=True))
    archived_at = Column(DateTime(timezone=False))
    updated_at = Column(DateTime(timezone=False), server_default=func.now(), onupdate=func.now(), nullable=False)
    doc_type = Column(String(40))
    doc_metadata = Column(JSON)
    doc_form = Column(VARCHAR(255), default='text_model', nullable=False)
    doc_language = Column(String(255))
