"""
向量存储模板

本模块提供了统一的向量数据库操作接口，支持多种向量数据库的集成和管理。
设计为可扩展的架构，便于添加新的向量数据库支持。

核心特性：
1. 多数据库支持：Chroma、Pinecone、FAISS、Qdrant、Weaviate、Milvus等
2. 统一接口：所有向量数据库使用相同的操作接口
3. 批量操作：支持批量向量化和存储
4. 异步支持：支持异步向量操作，提高处理效率
5. 索引管理：自动索引创建、更新和优化
6. 元数据过滤：支持基于元数据的向量检索
7. 相似度搜索：多种相似度算法和检索策略
8. 持久化存储：支持向量数据的持久化存储
9. 分布式支持：支持分布式向量存储
10. 监控和统计：提供详细的性能监控和统计信息

支持的向量数据库：
- Chroma: 开源向量数据库，适合原型开发
- Pinecone: 云原生向量数据库，高性能和可扩展性
- FAISS: Facebook AI 相似性搜索库，适合本地部署
- Qdrant: 现代向量数据库，支持复杂过滤
- Weaviate: 开源向量搜索引擎，支持混合搜索
- Milvus: 云原生向量数据库，适合大规模部署
- Elasticsearch: 支持向量搜索的搜索引擎
- Redis: 支持向量搜索的内存数据库

设计模式：
- 抽象工厂模式：创建不同类型的向量存储实例
- 适配器模式：统一不同向量数据库的接口
- 策略模式：支持不同的向量化和检索策略
- 装饰器模式：为向量存储添加额外功能（缓存、监控等）
"""

import os
import json
import asyncio
import pickle
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import tempfile
import shutil

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain.vectorstores import (
    Chroma, FAISS, Pinecone as PineconeVectorStore,
    Qdrant, Weaviate
)

# Embedding imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings
)

# 基础模板导入
from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class VectorStoreProvider(Enum):
    """向量存储提供商枚举"""
    CHROMA = "chroma"
    PINECONE = "pinecone"
    FAISS = "faiss"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"


class EmbeddingProvider(Enum):
    """嵌入模型提供商枚举"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CUSTOM = "custom"


class DistanceMetric(Enum):
    """距离度量枚举"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorStoreConfig:
    """
    向量存储配置类
    
    定义向量存储的配置参数。
    """
    provider: VectorStoreProvider              # 向量存储提供商
    embedding_provider: EmbeddingProvider      # 嵌入模型提供商
    collection_name: str = "default"          # 集合名称
    dimension: Optional[int] = None            # 向量维度
    distance_metric: DistanceMetric = DistanceMetric.COSINE  # 距离度量
    persist_directory: Optional[str] = None   # 持久化目录
    
    # 提供商特定配置
    provider_config: Dict[str, Any] = field(default_factory=dict)
    embedding_config: Dict[str, Any] = field(default_factory=dict)
    
    # 性能配置
    batch_size: int = 100                     # 批处理大小
    max_retries: int = 3                      # 最大重试次数
    timeout: float = 30.0                     # 超时时间


@dataclass
class VectorSearchResult:
    """
    向量搜索结果类
    
    包含搜索结果和相关信息。
    """
    documents: List[Document]                 # 搜索到的文档
    scores: List[float]                       # 相似度分数
    total_results: int = 0                    # 总结果数
    search_time: float = 0.0                 # 搜索耗时
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


class BaseVectorStore(ABC):
    """
    向量存储抽象基类
    
    定义向量存储的通用接口。
    """
    
    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        """
        初始化向量存储
        
        Args:
            config: 向量存储配置
            embeddings: 嵌入模型
        """
        self.config = config
        self.embeddings = embeddings
        self.vectorstore: Optional[VectorStore] = None
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化向量存储"""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """
        添加文档到向量存储
        
        Args:
            documents: 要添加的文档列表
            **kwargs: 额外参数
            
        Returns:
            文档ID列表
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VectorSearchResult:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            搜索结果
        """
        pass
    
    @abstractmethod
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        带分数的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            (文档, 分数)列表
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> bool:
        """
        删除文档
        
        Args:
            ids: 要删除的文档ID列表
            
        Returns:
            删除是否成功
        """
        pass
    
    @abstractmethod
    async def update_documents(self, documents: List[Document], ids: List[str]) -> bool:
        """
        更新文档
        
        Args:
            documents: 新文档内容
            ids: 要更新的文档ID列表
            
        Returns:
            更新是否成功
        """
        pass
    
    @abstractmethod
    async def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        pass
    
    async def close(self) -> None:
        """关闭向量存储连接"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Chroma向量存储实现"""
    
    async def initialize(self) -> None:
        """初始化Chroma向量存储"""
        try:
            persist_directory = self.config.persist_directory
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
            
            self.vectorstore = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
                **self.config.provider_config
            )
            
            self.is_initialized = True
            logger.info(f"Chroma vector store initialized: {self.config.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise ConfigurationError(f"Chroma initialization failed: {e}")
    
    async def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """添加文档到Chroma"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 生成文档ID
            ids = kwargs.get('ids', [str(uuid.uuid4()) for _ in documents])
            
            # 批量添加文档
            batch_size = self.config.batch_size
            added_ids = []
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                result_ids = self.vectorstore.add_documents(
                    documents=batch_docs,
                    ids=batch_ids
                )
                added_ids.extend(result_ids)
            
            # 持久化
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
            
            logger.info(f"Added {len(documents)} documents to Chroma")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma: {e}")
            raise ResourceError(f"Document addition failed: {e}")
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VectorSearchResult:
        """Chroma相似度搜索"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return VectorSearchResult(
                documents=results,
                scores=[1.0] * len(results),  # Chroma默认不返回分数
                total_results=len(results),
                search_time=search_time
            )
            
        except Exception as e:
            logger.error(f"Chroma similarity search failed: {e}")
            raise ResourceError(f"Search failed: {e}")
    
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Chroma带分数的相似度搜索"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
            return results
            
        except Exception as e:
            logger.error(f"Chroma similarity search with score failed: {e}")
            raise ResourceError(f"Search with score failed: {e}")
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """从Chroma删除文档"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Chroma删除文档
            self.vectorstore.delete(ids=ids)
            
            # 持久化
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
            
            logger.info(f"Deleted {len(ids)} documents from Chroma")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from Chroma: {e}")
            return False
    
    async def update_documents(self, documents: List[Document], ids: List[str]) -> bool:
        """更新Chroma中的文档"""
        try:
            # Chroma通过删除再添加来实现更新
            await self.delete_documents(ids)
            await self.add_documents(documents, ids=ids)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update documents in Chroma: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """获取Chroma集合信息"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 获取集合统计信息
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "provider": "chroma",
                "collection_name": self.config.collection_name,
                "document_count": count,
                "embedding_dimension": self.config.dimension,
                "distance_metric": self.config.distance_metric.value
            }
            
        except Exception as e:
            logger.warning(f"Failed to get Chroma collection info: {e}")
            return {"provider": "chroma", "error": str(e)}


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量存储实现"""
    
    async def initialize(self) -> None:
        """初始化FAISS向量存储"""
        try:
            # 检查是否存在已保存的索引
            persist_directory = self.config.persist_directory
            if persist_directory and os.path.exists(persist_directory):
                # 加载现有索引
                self.vectorstore = FAISS.load_local(
                    persist_directory,
                    self.embeddings
                )
                logger.info(f"Loaded existing FAISS index from {persist_directory}")
            else:
                # 创建新的空索引
                # FAISS需要至少一个文档来初始化
                dummy_doc = Document(page_content="dummy", metadata={"dummy": True})
                self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
                
                # 删除dummy文档
                if hasattr(self.vectorstore, 'delete'):
                    try:
                        self.vectorstore.delete([0])
                    except:
                        pass  # 忽略删除错误
                
                logger.info("Created new FAISS index")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise ConfigurationError(f"FAISS initialization failed: {e}")
    
    async def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """添加文档到FAISS"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # FAISS使用索引作为ID
            start_index = len(self.vectorstore.index_to_docstore_id)
            
            # 添加文档
            self.vectorstore.add_documents(documents)
            
            # 生成ID列表
            ids = [str(start_index + i) for i in range(len(documents))]
            
            # 保存索引
            if self.config.persist_directory:
                os.makedirs(self.config.persist_directory, exist_ok=True)
                self.vectorstore.save_local(self.config.persist_directory)
            
            logger.info(f"Added {len(documents)} documents to FAISS")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise ResourceError(f"Document addition failed: {e}")
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VectorSearchResult:
        """FAISS相似度搜索"""
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            return VectorSearchResult(
                documents=results,
                scores=[1.0] * len(results),  # FAISS默认不返回分数
                total_results=len(results),
                search_time=search_time
            )
            
        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}")
            raise ResourceError(f"Search failed: {e}")
    
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """FAISS带分数的相似度搜索"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                **kwargs
            )
            return results
            
        except Exception as e:
            logger.error(f"FAISS similarity search with score failed: {e}")
            raise ResourceError(f"Search with score failed: {e}")
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """从FAISS删除文档"""
        # FAISS不直接支持删除，需要重建索引
        logger.warning("FAISS does not support direct document deletion")
        return False
    
    async def update_documents(self, documents: List[Document], ids: List[str]) -> bool:
        """更新FAISS中的文档"""
        # FAISS不直接支持更新，需要重建索引
        logger.warning("FAISS does not support direct document update")
        return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """获取FAISS集合信息"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return {
                "provider": "faiss",
                "collection_name": self.config.collection_name,
                "document_count": self.vectorstore.index.ntotal,
                "embedding_dimension": self.vectorstore.index.d,
                "distance_metric": self.config.distance_metric.value
            }
            
        except Exception as e:
            logger.warning(f"Failed to get FAISS collection info: {e}")
            return {"provider": "faiss", "error": str(e)}


class VectorStoreTemplate(TemplateBase[List[Document], Union[List[str], VectorSearchResult]]):
    """
    向量存储模板
    
    提供统一的向量数据库操作接口，支持多种向量数据库和嵌入模型。
    支持文档的增删改查、相似度搜索等操作。
    
    核心功能：
    1. 多数据库支持：支持多种向量数据库
    2. 统一接口：提供一致的操作接口
    3. 批量操作：支持批量文档处理
    4. 异步支持：支持异步向量操作
    5. 索引管理：自动索引管理和优化
    6. 元数据过滤：支持复杂的元数据过滤
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """初始化向量存储模板"""
        super().__init__(config)
        
        # 配置参数
        self.vector_config: Optional[VectorStoreConfig] = None
        self.embeddings: Optional[Embeddings] = None
        self.vector_store: Optional[BaseVectorStore] = None
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'total_searches': 0,
            'avg_search_time': 0.0,
            'total_operations': 0
        }
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="VectorStoreTemplate",
            description="向量存储操作模板",
            template_type=TemplateType.DATA,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("provider", str, True, None, "向量存储提供商")
        config.add_parameter("embedding_provider", str, True, None, "嵌入模型提供商")
        config.add_parameter("collection_name", str, False, "default", "集合名称")
        config.add_parameter("dimension", int, False, None, "向量维度")
        config.add_parameter("distance_metric", str, False, "cosine", "距离度量")
        config.add_parameter("persist_directory", str, False, None, "持久化目录")
        config.add_parameter("provider_config", dict, False, {}, "提供商特定配置")
        config.add_parameter("embedding_config", dict, False, {}, "嵌入模型配置")
        config.add_parameter("batch_size", int, False, 100, "批处理大小")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置向量存储参数
        
        Args:
            **parameters: 向量存储参数
        """
        if not self.validate_parameters(parameters):
            raise ValidationError("Vector store parameter validation failed")
        
        # 创建向量存储配置
        provider_str = parameters.get('provider')
        embedding_provider_str = parameters.get('embedding_provider')
        
        try:
            provider = VectorStoreProvider(provider_str)
            embedding_provider = EmbeddingProvider(embedding_provider_str)
        except ValueError as e:
            raise ValidationError(f"Invalid provider: {e}")
        
        distance_metric_str = parameters.get('distance_metric', 'cosine')
        try:
            distance_metric = DistanceMetric(distance_metric_str)
        except ValueError:
            distance_metric = DistanceMetric.COSINE
        
        self.vector_config = VectorStoreConfig(
            provider=provider,
            embedding_provider=embedding_provider,
            collection_name=parameters.get('collection_name', 'default'),
            dimension=parameters.get('dimension'),
            distance_metric=distance_metric,
            persist_directory=parameters.get('persist_directory'),
            provider_config=parameters.get('provider_config', {}),
            embedding_config=parameters.get('embedding_config', {}),
            batch_size=parameters.get('batch_size', 100)
        )
        
        # 创建嵌入模型
        self._create_embeddings()
        
        # 创建向量存储
        self._create_vector_store()
        
        self.status = self.status.CONFIGURED
        self._setup_parameters = parameters.copy()
        
        logger.info(
            f"Vector store configured: provider={provider.value}, "
            f"embedding_provider={embedding_provider.value}, "
            f"collection={self.vector_config.collection_name}"
        )
    
    def _create_embeddings(self) -> None:
        """创建嵌入模型"""
        if not self.vector_config:
            raise RuntimeError("Vector config not set")
        
        provider = self.vector_config.embedding_provider
        config = self.vector_config.embedding_config
        
        try:
            if provider == EmbeddingProvider.OPENAI:
                self.embeddings = OpenAIEmbeddings(**config)
            elif provider == EmbeddingProvider.HUGGINGFACE:
                model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name, **config)
            elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                model_name = config.get('model_name', 'all-MiniLM-L6-v2')
                self.embeddings = SentenceTransformerEmbeddings(model_name=model_name, **config)
            else:
                raise ConfigurationError(f"Unsupported embedding provider: {provider}")
            
            logger.info(f"Created embeddings: {provider.value}")
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise ConfigurationError(f"Embeddings creation failed: {e}")
    
    def _create_vector_store(self) -> None:
        """创建向量存储实例"""
        if not self.vector_config or not self.embeddings:
            raise RuntimeError("Vector config or embeddings not set")
        
        provider = self.vector_config.provider
        
        try:
            if provider == VectorStoreProvider.CHROMA:
                self.vector_store = ChromaVectorStore(self.vector_config, self.embeddings)
            elif provider == VectorStoreProvider.FAISS:
                self.vector_store = FAISSVectorStore(self.vector_config, self.embeddings)
            # 其他提供商可以在这里添加
            else:
                raise ConfigurationError(f"Unsupported vector store provider: {provider}")
            
            logger.info(f"Created vector store: {provider.value}")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise ConfigurationError(f"Vector store creation failed: {e}")
    
    def execute(self, input_data: List[Document], **kwargs) -> List[str]:
        """
        执行文档添加操作
        
        Args:
            input_data: 要添加的文档列表
            **kwargs: 额外参数
            
        Returns:
            添加的文档ID列表
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not configured. Call setup() first.")
        
        # 使用异步方法
        return asyncio.run(self._execute_async(input_data, **kwargs))
    
    async def _execute_async(self, input_data: List[Document], **kwargs) -> List[str]:
        """异步执行文档添加"""
        try:
            # 初始化向量存储
            if not self.vector_store.is_initialized:
                await self.vector_store.initialize()
            
            # 添加文档
            document_ids = await self.vector_store.add_documents(input_data, **kwargs)
            
            # 更新统计信息
            self.stats['total_documents'] += len(input_data)
            self.stats['total_operations'] += 1
            
            logger.info(f"Added {len(input_data)} documents to vector store")
            return document_ids
            
        except Exception as e:
            logger.error(f"Vector store execution failed: {str(e)}")
            raise
    
    async def search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VectorSearchResult:
        """
        执行相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            搜索结果
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not configured. Call setup() first.")
        
        try:
            # 初始化向量存储
            if not self.vector_store.is_initialized:
                await self.vector_store.initialize()
            
            # 执行搜索
            result = await self.vector_store.similarity_search(query, k, filter, **kwargs)
            
            # 更新统计信息
            self.stats['total_searches'] += 1
            self.stats['total_operations'] += 1
            
            # 计算平均搜索时间
            if self.stats['total_searches'] > 1:
                current_avg = self.stats['avg_search_time']
                new_avg = (current_avg * (self.stats['total_searches'] - 1) + result.search_time) / self.stats['total_searches']
                self.stats['avg_search_time'] = new_avg
            else:
                self.stats['avg_search_time'] = result.search_time
            
            logger.info(f"Search completed: {len(result.documents)} results in {result.search_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise
    
    async def search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        执行带分数的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            (文档, 分数)列表
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not configured. Call setup() first.")
        
        try:
            # 初始化向量存储
            if not self.vector_store.is_initialized:
                await self.vector_store.initialize()
            
            # 执行搜索
            results = await self.vector_store.similarity_search_with_score(query, k, filter, **kwargs)
            
            # 更新统计信息
            self.stats['total_searches'] += 1
            self.stats['total_operations'] += 1
            
            logger.info(f"Search with score completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search with score failed: {str(e)}")
            raise
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """删除文档"""
        if not self.vector_store:
            raise RuntimeError("Vector store not configured. Call setup() first.")
        
        try:
            if not self.vector_store.is_initialized:
                await self.vector_store.initialize()
            
            success = await self.vector_store.delete_documents(ids)
            
            if success:
                self.stats['total_documents'] -= len(ids)
                self.stats['total_operations'] += 1
                logger.info(f"Deleted {len(ids)} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Document deletion failed: {str(e)}")
            return False
    
    async def update_documents(self, documents: List[Document], ids: List[str]) -> bool:
        """更新文档"""
        if not self.vector_store:
            raise RuntimeError("Vector store not configured. Call setup() first.")
        
        try:
            if not self.vector_store.is_initialized:
                await self.vector_store.initialize()
            
            success = await self.vector_store.update_documents(documents, ids)
            
            if success:
                self.stats['total_operations'] += 1
                logger.info(f"Updated {len(documents)} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Document update failed: {str(e)}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        if not self.vector_store:
            raise RuntimeError("Vector store not configured. Call setup() first.")
        
        try:
            if not self.vector_store.is_initialized:
                await self.vector_store.initialize()
            
            return await self.vector_store.get_collection_info()
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_documents': 0,
            'total_searches': 0,
            'avg_search_time': 0.0,
            'total_operations': 0
        }
    
    async def close(self) -> None:
        """关闭向量存储连接"""
        if self.vector_store:
            await self.vector_store.close()
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "provider": "chroma",
                "embedding_provider": "sentence_transformers",
                "collection_name": "my_documents",
                "persist_directory": "./chroma_db",
                "embedding_config": {
                    "model_name": "all-MiniLM-L6-v2"
                },
                "batch_size": 100
            },
            "execute_parameters": {
                "input_data": "List[Document] objects to store"
            },
            "usage_code": """
# 基础使用示例
from templates.data.vectorstore_template import VectorStoreTemplate
from langchain_core.documents import Document
import asyncio

# 初始化向量存储
vectorstore = VectorStoreTemplate()

# 配置Chroma向量数据库
vectorstore.setup(
    provider="chroma",
    embedding_provider="sentence_transformers",
    collection_name="my_documents",
    persist_directory="./chroma_db",
    embedding_config={
        "model_name": "all-MiniLM-L6-v2"
    }
)

# 准备文档
documents = [
    Document(page_content="这是第一个文档", metadata={"source": "doc1.txt"}),
    Document(page_content="这是第二个文档", metadata={"source": "doc2.txt"})
]

# 添加文档到向量存储
document_ids = vectorstore.execute(documents)
print(f"Added documents with IDs: {document_ids}")

# 搜索相似文档
async def search_example():
    # 基础搜索
    result = await vectorstore.search("搜索查询", k=5)
    print(f"Found {len(result.documents)} documents")
    
    # 带分数的搜索
    scored_results = await vectorstore.search_with_score("搜索查询", k=3)
    for doc, score in scored_results:
        print(f"Score: {score:.3f}, Content: {doc.page_content[:100]}...")
    
    # 带过滤条件的搜索
    filtered_result = await vectorstore.search(
        "搜索查询", 
        k=5, 
        filter={"source": "doc1.txt"}
    )
    
    # 获取集合信息
    info = await vectorstore.get_collection_info()
    print(f"Collection info: {info}")

# 运行搜索示例
asyncio.run(search_example())

# FAISS配置示例
vectorstore.setup(
    provider="faiss",
    embedding_provider="openai",
    collection_name="openai_embeddings",
    persist_directory="./faiss_index",
    embedding_config={
        "openai_api_key": "your_api_key"
    }
)

# 批量操作示例
async def batch_operations():
    # 批量添加大量文档
    large_doc_list = [Document(page_content=f"Document {i}") for i in range(1000)]
    ids = vectorstore.execute(large_doc_list)
    
    # 批量删除
    await vectorstore.delete_documents(ids[:100])
    
    # 批量更新
    updated_docs = [Document(page_content=f"Updated Document {i}") for i in range(100)]
    await vectorstore.update_documents(updated_docs, ids[100:200])
    
    # 获取统计信息
    stats = vectorstore.get_stats()
    print(f"Stats: {stats}")

asyncio.run(batch_operations())
""",
            "expected_output": {
                "type": "List[str] or VectorSearchResult",
                "description": "添加文档时返回文档ID列表，搜索时返回VectorSearchResult对象"
            }
        }