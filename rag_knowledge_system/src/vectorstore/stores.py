"""
向量存储实现模块

提供多种向量数据库的具体实现，包括Chroma、FAISS、Pinecone等。
每个存储实现都遵循统一的接口，支持文档添加、搜索和管理功能。
"""

import os
import json
import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np

# LangChain导入
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

# 可选依赖
try:
    from langchain.vectorstores import Chroma
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from langchain.vectorstores import FAISS
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from langchain.vectorstores import Pinecone
    import pinecone
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

try:
    from langchain.vectorstores import ElasticsearchStore
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """向量存储基类"""
    
    def __init__(self, 
                 collection_name: str,
                 embedding_manager,
                 **kwargs):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            embedding_manager: 嵌入管理器
            **kwargs: 额外参数
        """
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager
        self.store = None
        
        # 统计信息
        self.stats = {
            "total_documents": 0,
            "total_searches": 0,
            "total_additions": 0,
            "last_updated": None
        }
    
    @abstractmethod
    def _initialize_store(self, **kwargs):
        """初始化存储（子类实现）"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """添加文档"""
        pass
    
    @abstractmethod
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         **kwargs) -> List[Document]:
        """相似度搜索"""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self,
                                   query: str,
                                   k: int = 5,
                                   **kwargs) -> List[Tuple[Document, float]]:
        """带分数的相似度搜索"""
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """获取文档数量"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["document_count"] = self.get_document_count()
        return stats
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_documents": 0,
            "total_searches": 0,
            "total_additions": 0,
            "last_updated": None
        }


class ChromaVectorStore(BaseVectorStore):
    """Chroma向量存储实现"""
    
    def __init__(self,
                 collection_name: str,
                 embedding_manager,
                 persist_directory: str = "./chroma_db",
                 **kwargs):
        """
        初始化Chroma存储
        
        Args:
            collection_name: 集合名称
            embedding_manager: 嵌入管理器
            persist_directory: 持久化目录
            **kwargs: 额外参数
        """
        if not HAS_CHROMA:
            raise ImportError("需要安装Chroma: pip install chromadb")
        
        super().__init__(collection_name, embedding_manager)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._initialize_store(**kwargs)
        
        logger.info(f"初始化Chroma向量存储: {collection_name}")
    
    def _initialize_store(self, **kwargs):
        """初始化Chroma存储"""
        try:
            # 创建嵌入函数适配器
            class EmbeddingAdapter:
                def __init__(self, embedding_manager):
                    self.embedding_manager = embedding_manager
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    result = self.embedding_manager.embed_documents(texts)
                    return result.embeddings
                
                def embed_query(self, text: str) -> List[float]:
                    return self.embedding_manager.embed_query(text)
            
            embedding_adapter = EmbeddingAdapter(self.embedding_manager)
            
            # 初始化Chroma
            self.store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_adapter,
                persist_directory=str(self.persist_directory),
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"初始化Chroma失败: {e}")
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """添加文档到Chroma"""
        try:
            # 生成文档ID
            ids = [f"{self.collection_name}_{i}_{hash(doc.page_content)}" 
                  for i, doc in enumerate(documents)]
            
            # 添加文档
            self.store.add_documents(documents, ids=ids, **kwargs)
            
            # 持久化
            self.store.persist()
            
            # 更新统计
            self.stats["total_additions"] += len(documents)
            self.stats["total_documents"] = self.get_document_count()
            self.stats["last_updated"] = os.path.getmtime(self.persist_directory)
            
            logger.info(f"向Chroma添加 {len(documents)} 个文档")
            return ids
        
        except Exception as e:
            logger.error(f"添加文档到Chroma失败: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         **kwargs) -> List[Document]:
        """Chroma相似度搜索"""
        try:
            results = self.store.similarity_search(query, k=k, **kwargs)
            
            self.stats["total_searches"] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"Chroma搜索失败: {e}")
            raise
    
    def similarity_search_with_score(self,
                                   query: str,
                                   k: int = 5,
                                   **kwargs) -> List[Tuple[Document, float]]:
        """Chroma带分数的相似度搜索"""
        try:
            results = self.store.similarity_search_with_score(query, k=k, **kwargs)
            
            self.stats["total_searches"] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"Chroma带分数搜索失败: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> bool:
        """从Chroma删除文档"""
        try:
            self.store.delete(ids)
            self.store.persist()
            
            logger.info(f"从Chroma删除 {len(ids)} 个文档")
            return True
        
        except Exception as e:
            logger.error(f"从Chroma删除文档失败: {e}")
            return False
    
    def get_document_count(self) -> int:
        """获取Chroma文档数量"""
        try:
            # Chroma特定的计数方法
            collection = self.store._collection
            return collection.count()
        except:
            return 0


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量存储实现"""
    
    def __init__(self,
                 collection_name: str,
                 embedding_manager,
                 index_path: Optional[str] = None,
                 **kwargs):
        """
        初始化FAISS存储
        
        Args:
            collection_name: 集合名称
            embedding_manager: 嵌入管理器
            index_path: 索引文件路径
            **kwargs: 额外参数
        """
        if not HAS_FAISS:
            raise ImportError("需要安装FAISS: pip install faiss-cpu")
        
        super().__init__(collection_name, embedding_manager)
        self.index_path = index_path or f"./faiss_indexes/{collection_name}"
        self.metadata_path = f"{self.index_path}_metadata.pkl"
        
        # 确保目录存在
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_store(**kwargs)
        
        logger.info(f"初始化FAISS向量存储: {collection_name}")
    
    def _initialize_store(self, **kwargs):
        """初始化FAISS存储"""
        try:
            # 创建嵌入函数适配器
            class EmbeddingAdapter:
                def __init__(self, embedding_manager):
                    self.embedding_manager = embedding_manager
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    result = self.embedding_manager.embed_documents(texts)
                    return result.embeddings
                
                def embed_query(self, text: str) -> List[float]:
                    return self.embedding_manager.embed_query(text)
            
            embedding_adapter = EmbeddingAdapter(self.embedding_manager)
            
            # 尝试加载现有索引
            if os.path.exists(f"{self.index_path}.faiss") and os.path.exists(self.metadata_path):
                self.store = FAISS.load_local(
                    self.index_path,
                    embedding_adapter,
                    allow_dangerous_deserialization=True
                )
                logger.info("加载现有FAISS索引")
            else:
                # 创建空的FAISS存储
                # 需要先添加一个虚拟文档来初始化
                dummy_doc = Document(page_content="初始化文档", metadata={"dummy": True})
                self.store = FAISS.from_documents([dummy_doc], embedding_adapter)
                
                # 删除虚拟文档
                self.store.delete([0])
                
                logger.info("创建新的FAISS索引")
        
        except Exception as e:
            logger.error(f"初始化FAISS失败: {e}")
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """添加文档到FAISS"""
        try:
            # 生成文档ID
            start_id = len(self.store.docstore._dict)
            ids = [str(start_id + i) for i in range(len(documents))]
            
            # 添加文档
            self.store.add_documents(documents, ids=ids, **kwargs)
            
            # 保存索引
            self.save()
            
            # 更新统计
            self.stats["total_additions"] += len(documents)
            self.stats["total_documents"] = self.get_document_count()
            self.stats["last_updated"] = os.path.getmtime(f"{self.index_path}.faiss")
            
            logger.info(f"向FAISS添加 {len(documents)} 个文档")
            return ids
        
        except Exception as e:
            logger.error(f"添加文档到FAISS失败: {e}")
            raise
    
    def similarity_search(self,
                         query: str,
                         k: int = 5,
                         **kwargs) -> List[Document]:
        """FAISS相似度搜索"""
        try:
            results = self.store.similarity_search(query, k=k, **kwargs)
            
            self.stats["total_searches"] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"FAISS搜索失败: {e}")
            raise
    
    def similarity_search_with_score(self,
                                   query: str,
                                   k: int = 5,
                                   **kwargs) -> List[Tuple[Document, float]]:
        """FAISS带分数的相似度搜索"""
        try:
            results = self.store.similarity_search_with_score(query, k=k, **kwargs)
            
            self.stats["total_searches"] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"FAISS带分数搜索失败: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> bool:
        """从FAISS删除文档"""
        try:
            # FAISS不直接支持删除，需要重新构建索引
            # 这里实现一个简化版本
            logger.warning("FAISS不支持直接删除，建议重新构建索引")
            return False
        
        except Exception as e:
            logger.error(f"从FAISS删除文档失败: {e}")
            return False
    
    def get_document_count(self) -> int:
        """获取FAISS文档数量"""
        try:
            return len(self.store.docstore._dict)
        except:
            return 0
    
    def save(self):
        """保存FAISS索引"""
        try:
            self.store.save_local(self.index_path)
            logger.info(f"FAISS索引已保存: {self.index_path}")
        except Exception as e:
            logger.error(f"保存FAISS索引失败: {e}")


class PineconeVectorStore(BaseVectorStore):
    """Pinecone向量存储实现"""
    
    def __init__(self,
                 collection_name: str,
                 embedding_manager,
                 api_key: str,
                 environment: str,
                 **kwargs):
        """
        初始化Pinecone存储
        
        Args:
            collection_name: 索引名称
            embedding_manager: 嵌入管理器
            api_key: Pinecone API密钥
            environment: Pinecone环境
            **kwargs: 额外参数
        """
        if not HAS_PINECONE:
            raise ImportError("需要安装Pinecone: pip install pinecone-client")
        
        super().__init__(collection_name, embedding_manager)
        self.api_key = api_key
        self.environment = environment
        
        self._initialize_store(**kwargs)
        
        logger.info(f"初始化Pinecone向量存储: {collection_name}")
    
    def _initialize_store(self, **kwargs):
        """初始化Pinecone存储"""
        try:
            # 初始化Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # 检查索引是否存在
            if self.collection_name not in pinecone.list_indexes():
                # 创建索引
                pinecone.create_index(
                    name=self.collection_name,
                    dimension=self.embedding_manager.config.dimension,
                    metric="cosine",
                    **kwargs
                )
                logger.info(f"创建Pinecone索引: {self.collection_name}")
            
            # 创建嵌入函数适配器
            class EmbeddingAdapter:
                def __init__(self, embedding_manager):
                    self.embedding_manager = embedding_manager
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    result = self.embedding_manager.embed_documents(texts)
                    return result.embeddings
                
                def embed_query(self, text: str) -> List[float]:
                    return self.embedding_manager.embed_query(text)
            
            embedding_adapter = EmbeddingAdapter(self.embedding_manager)
            
            # 初始化Pinecone存储
            self.store = Pinecone.from_existing_index(
                index_name=self.collection_name,
                embedding=embedding_adapter
            )
        
        except Exception as e:
            logger.error(f"初始化Pinecone失败: {e}")
            raise
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """添加文档到Pinecone"""
        try:
            # 生成文档ID
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
            
            # 添加文档
            self.store.add_documents(documents, ids=ids, **kwargs)
            
            # 更新统计
            self.stats["total_additions"] += len(documents)
            self.stats["total_documents"] = self.get_document_count()
            
            logger.info(f"向Pinecone添加 {len(documents)} 个文档")
            return ids
        
        except Exception as e:
            logger.error(f"添加文档到Pinecone失败: {e}")
            raise
    
    def similarity_search(self,
                         query: str,
                         k: int = 5,
                         **kwargs) -> List[Document]:
        """Pinecone相似度搜索"""
        try:
            results = self.store.similarity_search(query, k=k, **kwargs)
            
            self.stats["total_searches"] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"Pinecone搜索失败: {e}")
            raise
    
    def similarity_search_with_score(self,
                                   query: str,
                                   k: int = 5,
                                   **kwargs) -> List[Tuple[Document, float]]:
        """Pinecone带分数的相似度搜索"""
        try:
            results = self.store.similarity_search_with_score(query, k=k, **kwargs)
            
            self.stats["total_searches"] += 1
            
            return results
        
        except Exception as e:
            logger.error(f"Pinecone带分数搜索失败: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> bool:
        """从Pinecone删除文档"""
        try:
            # 获取Pinecone索引
            index = pinecone.Index(self.collection_name)
            index.delete(ids=ids)
            
            logger.info(f"从Pinecone删除 {len(ids)} 个文档")
            return True
        
        except Exception as e:
            logger.error(f"从Pinecone删除文档失败: {e}")
            return False
    
    def get_document_count(self) -> int:
        """获取Pinecone文档数量"""
        try:
            index = pinecone.Index(self.collection_name)
            stats = index.describe_index_stats()
            return stats.total_vector_count
        except:
            return 0


# 工厂函数
def get_vector_store(store_type: str,
                    collection_name: str,
                    embedding_manager,
                    **kwargs) -> BaseVectorStore:
    """
    创建向量存储的工厂函数
    
    Args:
        store_type: 存储类型
        collection_name: 集合名称
        embedding_manager: 嵌入管理器
        **kwargs: 额外参数
        
    Returns:
        向量存储实例
    """
    store_map = {
        "chroma": ChromaVectorStore,
        "faiss": FAISSVectorStore,
        "pinecone": PineconeVectorStore
    }
    
    if store_type.lower() not in store_map:
        raise ValueError(f"不支持的向量存储类型: {store_type}")
    
    store_class = store_map[store_type.lower()]
    return store_class(collection_name, embedding_manager, **kwargs)


# 便捷函数
def create_vector_store(documents: List[Document],
                       store_type: str = "chroma",
                       collection_name: str = "default",
                       embedding_provider: str = "openai",
                       embedding_model: str = "text-embedding-ada-002",
                       **kwargs) -> BaseVectorStore:
    """
    创建并初始化向量存储的便捷函数
    
    Args:
        documents: 初始文档列表
        store_type: 存储类型
        collection_name: 集合名称
        embedding_provider: 嵌入提供商
        embedding_model: 嵌入模型
        **kwargs: 额外参数
        
    Returns:
        初始化后的向量存储
    """
    from .embeddings import get_embedding_model
    
    # 创建嵌入管理器
    embedding_manager = get_embedding_model(
        embedding_provider, 
        embedding_model,
        **kwargs.get('embedding_kwargs', {})
    )
    
    # 创建向量存储
    store_kwargs = {k: v for k, v in kwargs.items() if k != 'embedding_kwargs'}
    vector_store = get_vector_store(
        store_type,
        collection_name,
        embedding_manager,
        **store_kwargs
    )
    
    # 添加初始文档
    if documents:
        vector_store.add_documents(documents)
        logger.info(f"向量存储创建完成，包含 {len(documents)} 个文档")
    
    return vector_store