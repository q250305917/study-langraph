"""
向量存储模块

提供多种向量数据库的集成，包括嵌入模型管理、向量存储和检索功能。
支持Chroma、FAISS、Pinecone等多种向量数据库。
"""

from .manager import VectorStoreManager
from .embeddings import EmbeddingManager, get_embedding_model
from .stores import (
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    PineconeVectorStore,
    get_vector_store
)
from .indexing import (
    IndexManager,
    IndexConfig,
    create_index,
    update_index
)

__all__ = [
    # 核心管理器
    "VectorStoreManager",
    "EmbeddingManager",
    
    # 嵌入模型
    "get_embedding_model",
    
    # 向量存储
    "BaseVectorStore",
    "ChromaVectorStore", 
    "FAISSVectorStore",
    "PineconeVectorStore",
    "get_vector_store",
    
    # 索引管理
    "IndexManager",
    "IndexConfig",
    "create_index",
    "update_index"
]