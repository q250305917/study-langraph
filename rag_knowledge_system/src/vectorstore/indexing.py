"""
向量索引管理模块

提供向量索引的创建、更新、优化和管理功能。
支持增量更新、索引优化和性能监控。
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

# LangChain导入
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """索引配置"""
    
    # 基本配置
    name: str = "default_index"
    description: str = ""
    version: str = "1.0.0"
    
    # 索引策略
    index_type: str = "flat"  # flat, ivf, hnsw
    metric: str = "cosine"  # cosine, euclidean, inner_product
    
    # 性能参数
    nlist: int = 1024  # IVF参数
    nprobe: int = 10   # 搜索时的probe数量
    efConstruction: int = 200  # HNSW构建参数
    efSearch: int = 100  # HNSW搜索参数
    M: int = 16  # HNSW连接数
    
    # 更新策略
    update_threshold: int = 1000  # 触发重建的文档数量
    update_interval: int = 3600   # 更新间隔（秒）
    enable_auto_optimize: bool = True
    
    # 缓存配置
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl: int = 3600  # 缓存TTL（秒）


@dataclass
class IndexStats:
    """索引统计信息"""
    
    # 基本信息
    name: str = ""
    created_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # 数据统计
    total_documents: int = 0
    total_vectors: int = 0
    dimension: int = 0
    
    # 性能统计
    index_size_mb: float = 0.0
    build_time_seconds: float = 0.0
    avg_search_time_ms: float = 0.0
    
    # 操作统计
    total_searches: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    
    # 质量指标
    recall_at_1: float = 0.0
    recall_at_10: float = 0.0
    precision_at_10: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        
        # 处理datetime对象
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data


class IndexManager:
    """索引管理器"""
    
    def __init__(self,
                 vector_store,
                 config: Optional[IndexConfig] = None,
                 stats_file: Optional[str] = None):
        """
        初始化索引管理器
        
        Args:
            vector_store: 向量存储实例
            config: 索引配置
            stats_file: 统计文件路径
        """
        self.vector_store = vector_store
        self.config = config or IndexConfig()
        self.stats_file = stats_file or f"./stats/{self.config.name}_stats.json"
        
        # 创建统计目录
        Path(self.stats_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 加载或初始化统计
        self.stats = self._load_stats()
        
        # 性能监控
        self.search_times = []
        self.cache = {} if self.config.enable_cache else None
        
        logger.info(f"初始化索引管理器: {self.config.name}")
    
    def create_index(self, documents: List[Document]) -> bool:
        """
        创建索引
        
        Args:
            documents: 文档列表
            
        Returns:
            是否成功
        """
        try:
            start_time = time.time()
            
            logger.info(f"开始创建索引: {len(documents)} 个文档")
            
            # 添加文档到向量存储
            doc_ids = self.vector_store.add_documents(documents)
            
            # 更新统计
            build_time = time.time() - start_time
            self._update_stats_after_creation(documents, build_time)
            
            # 保存统计
            self._save_stats()
            
            logger.info(f"索引创建完成，耗时 {build_time:.2f} 秒")
            return True
        
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False
    
    def update_index(self, documents: List[Document]) -> bool:
        """
        更新索引
        
        Args:
            documents: 新文档列表
            
        Returns:
            是否成功
        """
        try:
            start_time = time.time()
            
            logger.info(f"更新索引: {len(documents)} 个新文档")
            
            # 检查是否需要重建
            if self._should_rebuild_index(len(documents)):
                logger.info("触发索引重建")
                return self._rebuild_index(documents)
            
            # 增量更新
            doc_ids = self.vector_store.add_documents(documents)
            
            # 更新统计
            update_time = time.time() - start_time
            self._update_stats_after_update(documents, update_time)
            
            # 保存统计
            self._save_stats()
            
            logger.info(f"索引更新完成，耗时 {update_time:.2f} 秒")
            return True
        
        except Exception as e:
            logger.error(f"更新索引失败: {e}")
            return False
    
    def search(self,
               query: str,
               k: int = 5,
               with_score: bool = False,
               **kwargs) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        搜索索引
        
        Args:
            query: 查询文本
            k: 返回结果数量
            with_score: 是否返回分数
            **kwargs: 额外参数
            
        Returns:
            搜索结果
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(query, k, kwargs)
            if self.cache and cache_key in self.cache:
                cached_result, cache_time = self.cache[cache_key]
                if time.time() - cache_time < self.config.cache_ttl:
                    logger.debug(f"缓存命中: {query[:50]}")
                    return cached_result
            
            # 执行搜索
            if with_score:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, **kwargs
                )
            else:
                results = self.vector_store.similarity_search(
                    query, k=k, **kwargs
                )
            
            # 更新缓存
            if self.cache:
                self.cache[cache_key] = (results, time.time())
                
                # 清理过期缓存
                if len(self.cache) > self.config.cache_size:
                    self._cleanup_cache()
            
            # 记录性能
            search_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.search_times.append(search_time)
            
            # 更新统计
            self.stats.total_searches += 1
            if len(self.search_times) > 100:
                self.stats.avg_search_time_ms = sum(self.search_times[-100:]) / 100
            
            return results
        
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def optimize_index(self) -> bool:
        """
        优化索引
        
        Returns:
            是否成功
        """
        try:
            logger.info("开始优化索引")
            
            # 检查是否支持优化
            if hasattr(self.vector_store, 'optimize'):
                self.vector_store.optimize()
                logger.info("索引优化完成")
                return True
            else:
                logger.warning("当前向量存储不支持索引优化")
                return False
        
        except Exception as e:
            logger.error(f"索引优化失败: {e}")
            return False
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        删除文档
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            是否成功
        """
        try:
            success = self.vector_store.delete_documents(doc_ids)
            
            if success:
                self.stats.total_deletions += len(doc_ids)
                self.stats.total_documents = max(0, self.stats.total_documents - len(doc_ids))
                self._save_stats()
                
                # 清空相关缓存
                if self.cache:
                    self.cache.clear()
                
                logger.info(f"删除 {len(doc_ids)} 个文档")
            
            return success
        
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        info = {
            "config": asdict(self.config),
            "stats": self.stats.to_dict(),
            "store_type": type(self.vector_store).__name__,
            "cache_size": len(self.cache) if self.cache else 0
        }
        
        return info
    
    def _should_rebuild_index(self, new_doc_count: int) -> bool:
        """检查是否应该重建索引"""
        # 基于文档数量阈值
        if new_doc_count >= self.config.update_threshold:
            return True
        
        # 基于时间间隔
        if self.stats.last_updated:
            time_since_update = (datetime.now() - self.stats.last_updated).total_seconds()
            if time_since_update >= self.config.update_interval:
                return True
        
        return False
    
    def _rebuild_index(self, new_documents: List[Document]) -> bool:
        """重建索引"""
        try:
            logger.info("开始重建索引")
            
            # 获取现有文档
            existing_count = self.vector_store.get_document_count()
            
            # 这里简化处理，直接添加新文档
            # 在实际实现中，可能需要重新创建整个索引
            doc_ids = self.vector_store.add_documents(new_documents)
            
            # 优化索引
            if self.config.enable_auto_optimize:
                self.optimize_index()
            
            logger.info("索引重建完成")
            return True
        
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            return False
    
    def _update_stats_after_creation(self, documents: List[Document], build_time: float):
        """创建后更新统计"""
        self.stats.name = self.config.name
        self.stats.created_time = datetime.now()
        self.stats.last_updated = datetime.now()
        self.stats.total_documents = len(documents)
        self.stats.total_vectors = len(documents)
        self.stats.build_time_seconds = build_time
        self.stats.total_additions += len(documents)
        
        # 计算索引大小（估算）
        if hasattr(self.vector_store.embedding_manager, 'config'):
            dimension = self.vector_store.embedding_manager.config.dimension
            self.stats.dimension = dimension
            # 估算：每个向量占用dimension * 4字节（float32）
            self.stats.index_size_mb = (len(documents) * dimension * 4) / (1024 * 1024)
    
    def _update_stats_after_update(self, documents: List[Document], update_time: float):
        """更新后更新统计"""
        self.stats.last_updated = datetime.now()
        self.stats.total_documents += len(documents)
        self.stats.total_vectors += len(documents)
        self.stats.total_additions += len(documents)
        
        # 更新索引大小
        if self.stats.dimension > 0:
            additional_size = (len(documents) * self.stats.dimension * 4) / (1024 * 1024)
            self.stats.index_size_mb += additional_size
    
    def _get_cache_key(self, query: str, k: int, kwargs: Dict) -> str:
        """生成缓存键"""
        content = f"{query}:{k}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, cache_time) in self.cache.items():
            if current_time - cache_time >= self.config.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        # 如果仍然太大，删除最旧的条目
        while len(self.cache) > self.config.cache_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
    
    def _load_stats(self) -> IndexStats:
        """加载统计信息"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理datetime字段
                datetime_fields = ['created_time', 'last_updated']
                for field in datetime_fields:
                    if field in data and isinstance(data[field], str):
                        try:
                            data[field] = datetime.fromisoformat(data[field])
                        except:
                            data[field] = None
                
                return IndexStats(**data)
        
        except Exception as e:
            logger.warning(f"加载统计文件失败: {e}")
        
        return IndexStats(name=self.config.name)
    
    def _save_stats(self):
        """保存统计信息"""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats.to_dict(), f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logger.error(f"保存统计文件失败: {e}")


# 便捷函数
def create_index(vector_store,
                documents: List[Document],
                config: Optional[IndexConfig] = None) -> IndexManager:
    """
    创建索引的便捷函数
    
    Args:
        vector_store: 向量存储实例
        documents: 文档列表
        config: 索引配置
        
    Returns:
        IndexManager实例
    """
    manager = IndexManager(vector_store, config)
    success = manager.create_index(documents)
    
    if not success:
        raise Exception("索引创建失败")
    
    return manager


def update_index(manager: IndexManager,
                documents: List[Document]) -> bool:
    """
    更新索引的便捷函数
    
    Args:
        manager: 索引管理器
        documents: 新文档列表
        
    Returns:
        是否成功
    """
    return manager.update_index(documents)


def benchmark_index(manager: IndexManager,
                   test_queries: List[str],
                   k: int = 10) -> Dict[str, Any]:
    """
    对索引进行基准测试
    
    Args:
        manager: 索引管理器
        test_queries: 测试查询列表
        k: 返回结果数量
        
    Returns:
        基准测试结果
    """
    results = {
        "total_queries": len(test_queries),
        "search_times": [],
        "result_counts": [],
        "avg_search_time": 0.0,
        "min_search_time": float('inf'),
        "max_search_time": 0.0
    }
    
    logger.info(f"开始基准测试，{len(test_queries)} 个查询")
    
    for i, query in enumerate(test_queries):
        start_time = time.time()
        
        try:
            search_results = manager.search(query, k=k)
            search_time = (time.time() - start_time) * 1000  # 毫秒
            
            results["search_times"].append(search_time)
            results["result_counts"].append(len(search_results))
            
            results["min_search_time"] = min(results["min_search_time"], search_time)
            results["max_search_time"] = max(results["max_search_time"], search_time)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已完成 {i + 1}/{len(test_queries)} 个查询")
        
        except Exception as e:
            logger.error(f"查询失败 {i}: {e}")
            continue
    
    # 计算统计
    if results["search_times"]:
        results["avg_search_time"] = sum(results["search_times"]) / len(results["search_times"])
        results["p50_search_time"] = sorted(results["search_times"])[len(results["search_times"]) // 2]
        results["p95_search_time"] = sorted(results["search_times"])[int(len(results["search_times"]) * 0.95)]
        results["p99_search_time"] = sorted(results["search_times"])[int(len(results["search_times"]) * 0.99)]
    
    logger.info(f"基准测试完成，平均搜索时间: {results['avg_search_time']:.2f}ms")
    
    return results