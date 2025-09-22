"""
向量存储管理器

整合嵌入模型、向量存储和索引管理功能，提供统一的向量存储管理接口。
支持多种存储后端、自动索引管理和性能监控。
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# LangChain导入
from langchain.schema import Document

# 本地导入
from .embeddings import EmbeddingManager, EmbeddingConfig, get_embedding_model
from .stores import BaseVectorStore, get_vector_store
from .indexing import IndexManager, IndexConfig, IndexStats

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器
    
    提供完整的向量存储管理功能，包括：
    1. 嵌入模型管理
    2. 向量存储操作
    3. 索引管理和优化
    4. 性能监控和统计
    """
    
    def __init__(self,
                 collection_name: str,
                 store_type: str = "chroma",
                 embedding_config: Optional[EmbeddingConfig] = None,
                 index_config: Optional[IndexConfig] = None,
                 **store_kwargs):
        """
        初始化向量存储管理器
        
        Args:
            collection_name: 集合名称
            store_type: 存储类型
            embedding_config: 嵌入配置
            index_config: 索引配置
            **store_kwargs: 存储特定参数
        """
        self.collection_name = collection_name
        self.store_type = store_type
        
        # 初始化嵌入管理器
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.embedding_manager = EmbeddingManager(self.embedding_config)
        
        # 初始化向量存储
        self.vector_store = get_vector_store(
            store_type,
            collection_name, 
            self.embedding_manager,
            **store_kwargs
        )
        
        # 初始化索引管理器
        self.index_config = index_config or IndexConfig(name=collection_name)
        self.index_manager = IndexManager(
            self.vector_store,
            self.index_config
        )
        
        # 管理器统计
        self.manager_stats = {
            "created_time": datetime.now(),
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "last_operation_time": None
        }
        
        logger.info(f"初始化向量存储管理器: {collection_name} ({store_type})")
    
    def add_documents(self, 
                     documents: List[Document],
                     batch_size: Optional[int] = None,
                     **kwargs) -> List[str]:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
            **kwargs: 额外参数
            
        Returns:
            文档ID列表
        """
        try:
            self._start_operation("add_documents")
            
            # 使用配置的批处理大小
            if batch_size is None:
                batch_size = self.embedding_config.batch_size
            
            all_ids = []
            
            # 批量处理文档
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                logger.info(f"处理文档批次 {i//batch_size + 1}: {len(batch)} 个文档")
                
                # 添加到向量存储
                batch_ids = self.vector_store.add_documents(batch, **kwargs)
                all_ids.extend(batch_ids)
            
            # 更新索引
            self.index_manager.update_index(documents)
            
            self._complete_operation("add_documents", True)
            
            logger.info(f"成功添加 {len(documents)} 个文档")
            return all_ids
        
        except Exception as e:
            self._complete_operation("add_documents", False)
            logger.error(f"添加文档失败: {e}")
            raise
    
    def search(self,
               query: str,
               k: int = 5,
               search_type: str = "similarity",
               filters: Optional[Dict[str, Any]] = None,
               **kwargs) -> List[Document]:
        """
        搜索文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            search_type: 搜索类型 (similarity, mmr)
            filters: 过滤条件
            **kwargs: 额外参数
            
        Returns:
            搜索结果文档列表
        """
        try:
            self._start_operation("search")
            
            # 使用索引管理器进行搜索
            if search_type == "similarity":
                results = self.index_manager.search(query, k=k, **kwargs)
            elif search_type == "mmr":
                # 最大边际相关性搜索
                if hasattr(self.vector_store.store, 'max_marginal_relevance_search'):
                    results = self.vector_store.store.max_marginal_relevance_search(
                        query, k=k, **kwargs
                    )
                else:
                    logger.warning("当前存储不支持MMR搜索，回退到相似度搜索")
                    results = self.index_manager.search(query, k=k, **kwargs)
            else:
                raise ValueError(f"不支持的搜索类型: {search_type}")
            
            # 应用过滤器
            if filters:
                results = self._apply_filters(results, filters)
            
            self._complete_operation("search", True)
            
            return results
        
        except Exception as e:
            self._complete_operation("search", False)
            logger.error(f"搜索失败: {e}")
            raise
    
    def search_with_score(self,
                         query: str,
                         k: int = 5,
                         score_threshold: Optional[float] = None,
                         **kwargs) -> List[Tuple[Document, float]]:
        """
        带分数的搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 分数阈值
            **kwargs: 额外参数
            
        Returns:
            (文档, 分数) 元组列表
        """
        try:
            self._start_operation("search_with_score")
            
            # 执行搜索
            results = self.index_manager.search(
                query, k=k, with_score=True, **kwargs
            )
            
            # 应用分数阈值
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results 
                          if score >= score_threshold]
            
            self._complete_operation("search_with_score", True)
            
            return results
        
        except Exception as e:
            self._complete_operation("search_with_score", False)
            logger.error(f"带分数搜索失败: {e}")
            raise
    
    def delete_documents(self, 
                        doc_ids: Optional[List[str]] = None,
                        filter_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        删除文档
        
        Args:
            doc_ids: 文档ID列表
            filter_dict: 过滤条件
            
        Returns:
            是否成功
        """
        try:
            self._start_operation("delete_documents")
            
            if doc_ids:
                success = self.index_manager.delete_documents(doc_ids)
            elif filter_dict:
                # 基于过滤条件删除（如果支持）
                if hasattr(self.vector_store, 'delete_by_filter'):
                    success = self.vector_store.delete_by_filter(filter_dict)
                else:
                    logger.warning("当前存储不支持基于过滤器的删除")
                    success = False
            else:
                raise ValueError("必须提供doc_ids或filter_dict")
            
            self._complete_operation("delete_documents", success)
            
            return success
        
        except Exception as e:
            self._complete_operation("delete_documents", False)
            logger.error(f"删除文档失败: {e}")
            return False
    
    def update_document(self, 
                       doc_id: str, 
                       document: Document) -> bool:
        """
        更新单个文档
        
        Args:
            doc_id: 文档ID
            document: 新文档内容
            
        Returns:
            是否成功
        """
        try:
            # 删除旧文档
            delete_success = self.delete_documents([doc_id])
            
            if not delete_success:
                return False
            
            # 添加新文档
            new_ids = self.add_documents([document])
            
            return len(new_ids) > 0
        
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False
    
    def get_document_count(self) -> int:
        """获取文档总数"""
        try:
            return self.vector_store.get_document_count()
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        return {
            "collection_name": self.collection_name,
            "store_type": self.store_type,
            "document_count": self.get_document_count(),
            "embedding_model": self.embedding_config.model_name,
            "embedding_provider": self.embedding_config.provider,
            "dimension": self.embedding_config.dimension,
            "index_info": self.index_manager.get_index_info(),
            "created_time": self.manager_stats["created_time"].isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计信息"""
        return {
            "manager_stats": self._get_manager_stats(),
            "embedding_stats": self.embedding_manager.get_stats(),
            "vector_store_stats": self.vector_store.get_stats(),
            "index_stats": self.index_manager.stats.to_dict()
        }
    
    def optimize(self) -> bool:
        """优化向量存储和索引"""
        try:
            logger.info("开始优化向量存储")
            
            # 优化索引
            index_success = self.index_manager.optimize_index()
            
            # 清理嵌入缓存
            self.embedding_manager.model.clear_cache()
            
            logger.info(f"优化完成，索引优化: {index_success}")
            return index_success
        
        except Exception as e:
            logger.error(f"优化失败: {e}")
            return False
    
    def backup(self, backup_path: str) -> bool:
        """
        备份向量存储
        
        Args:
            backup_path: 备份路径
            
        Returns:
            是否成功
        """
        try:
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 备份配置
            config_data = {
                "collection_name": self.collection_name,
                "store_type": self.store_type,
                "embedding_config": self.embedding_config.__dict__,
                "index_config": self.index_config.__dict__,
                "stats": self.get_stats()
            }
            
            with open(backup_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
            
            # 备份向量存储数据
            if hasattr(self.vector_store, 'save'):
                self.vector_store.save(str(backup_path / "vector_store"))
            
            logger.info(f"备份完成: {backup_path}")
            return True
        
        except Exception as e:
            logger.error(f"备份失败: {e}")
            return False
    
    def _apply_filters(self, 
                      results: List[Document], 
                      filters: Dict[str, Any]) -> List[Document]:
        """应用过滤器"""
        filtered_results = []
        
        for doc in results:
            match = True
            
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                elif doc.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(doc)
        
        return filtered_results
    
    def _start_operation(self, operation_type: str):
        """开始操作"""
        self.manager_stats["total_operations"] += 1
        self.manager_stats["last_operation_time"] = datetime.now()
        logger.debug(f"开始操作: {operation_type}")
    
    def _complete_operation(self, operation_type: str, success: bool):
        """完成操作"""
        if success:
            self.manager_stats["successful_operations"] += 1
        else:
            self.manager_stats["failed_operations"] += 1
        
        logger.debug(f"操作完成: {operation_type}, 成功: {success}")
    
    def _get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计"""
        stats = self.manager_stats.copy()
        
        # 计算成功率
        total_ops = stats["total_operations"]
        if total_ops > 0:
            stats["success_rate"] = stats["successful_operations"] / total_ops
        else:
            stats["success_rate"] = 0.0
        
        # 转换datetime为字符串
        if stats["created_time"]:
            stats["created_time"] = stats["created_time"].isoformat()
        if stats["last_operation_time"]:
            stats["last_operation_time"] = stats["last_operation_time"].isoformat()
        
        return stats


# 便捷函数
def create_vector_store_manager(collection_name: str,
                              documents: Optional[List[Document]] = None,
                              store_type: str = "chroma",
                              embedding_provider: str = "openai",
                              embedding_model: str = "text-embedding-ada-002",
                              **kwargs) -> VectorStoreManager:
    """
    创建向量存储管理器的便捷函数
    
    Args:
        collection_name: 集合名称
        documents: 初始文档列表
        store_type: 存储类型
        embedding_provider: 嵌入提供商
        embedding_model: 嵌入模型
        **kwargs: 额外参数
        
    Returns:
        VectorStoreManager实例
    """
    # 创建嵌入配置
    embedding_config = EmbeddingConfig(
        provider=embedding_provider,
        model_name=embedding_model,
        **kwargs.get('embedding_kwargs', {})
    )
    
    # 创建索引配置
    index_config = IndexConfig(
        name=collection_name,
        **kwargs.get('index_kwargs', {})
    )
    
    # 过滤存储参数
    store_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['embedding_kwargs', 'index_kwargs']}
    
    # 创建管理器
    manager = VectorStoreManager(
        collection_name=collection_name,
        store_type=store_type,
        embedding_config=embedding_config,
        index_config=index_config,
        **store_kwargs
    )
    
    # 添加初始文档
    if documents:
        manager.add_documents(documents)
        logger.info(f"初始化向量存储管理器，包含 {len(documents)} 个文档")
    
    return manager


def load_vector_store_manager(backup_path: str) -> VectorStoreManager:
    """
    从备份加载向量存储管理器
    
    Args:
        backup_path: 备份路径
        
    Returns:
        VectorStoreManager实例
    """
    backup_path = Path(backup_path)
    
    # 加载配置
    with open(backup_path / "config.json", 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # 重建配置对象
    embedding_config = EmbeddingConfig(**config_data["embedding_config"])
    index_config = IndexConfig(**config_data["index_config"])
    
    # 创建管理器
    manager = VectorStoreManager(
        collection_name=config_data["collection_name"],
        store_type=config_data["store_type"],
        embedding_config=embedding_config,
        index_config=index_config
    )
    
    logger.info(f"从备份加载向量存储管理器: {backup_path}")
    return manager