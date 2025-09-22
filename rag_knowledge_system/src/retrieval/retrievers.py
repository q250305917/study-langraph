"""
检索器实现模块

提供多种检索策略的具体实现，包括向量检索、关键词检索、混合检索等。
每个检索器都支持灵活的配置和性能优化。
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

# LangChain导入
from langchain.schema import Document
from langchain.retrievers.base import BaseRetriever as LangChainBaseRetriever

# 可选依赖
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """检索配置"""
    
    # 基本参数
    k: int = 5
    max_k: int = 20
    similarity_threshold: float = 0.0
    
    # 性能参数
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # 多样性参数
    diversity_threshold: float = 0.8
    enable_mmr: bool = False
    lambda_mult: float = 0.5
    
    # 过滤参数
    enable_filters: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass 
class RetrievalResult:
    """检索结果"""
    documents: List[Document]
    scores: Optional[List[float]] = None
    query: str = ""
    k: int = 0
    retrieval_time: float = 0.0
    retriever_type: str = ""
    metadata: Optional[Dict[str, Any]] = None


class BaseRetriever(ABC):
    """检索器基类"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        初始化检索器
        
        Args:
            config: 检索配置
        """
        self.config = config or RetrievalConfig()
        self.cache = {} if self.config.enable_cache else None
        
        # 性能统计
        self.stats = {
            "total_queries": 0,
            "total_documents_retrieved": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    @abstractmethod
    def _retrieve(self, query: str, k: int, **kwargs) -> RetrievalResult:
        """执行检索（子类实现）"""
        pass
    
    def retrieve(self, query: str, k: Optional[int] = None, **kwargs) -> RetrievalResult:
        """
        检索文档
        
        Args:
            query: 查询文本
            k: 返回文档数量
            **kwargs: 额外参数
            
        Returns:
            RetrievalResult对象
        """
        start_time = time.time()
        
        # 使用默认k值
        if k is None:
            k = self.config.k
        
        # 限制k值
        k = min(k, self.config.max_k)
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key(query, k, kwargs)
            if self.cache and cache_key in self.cache:
                cached_result, cache_time = self.cache[cache_key]
                if time.time() - cache_time < self.config.cache_ttl:
                    self.stats["cache_hits"] += 1
                    logger.debug(f"缓存命中: {query[:50]}")
                    return cached_result
            
            # 执行检索
            result = self._retrieve(query, k, **kwargs)
            result.query = query
            result.k = k
            result.retrieval_time = time.time() - start_time
            result.retriever_type = self.__class__.__name__
            
            # 应用过滤器
            if self.config.enable_filters:
                result = self._apply_filters(result)
            
            # 更新缓存
            if self.cache:
                self.cache[cache_key] = (result, time.time())
                self._cleanup_cache()
                self.stats["cache_misses"] += 1
            
            # 更新统计
            self._update_stats(result)
            
            return result
        
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise
    
    def _get_cache_key(self, query: str, k: int, kwargs: Dict) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{query}:{k}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """清理缓存"""
        if not self.cache or len(self.cache) <= self.config.cache_size:
            return
        
        # 删除最旧的条目
        current_time = time.time()
        sorted_cache = sorted(
            self.cache.items(),
            key=lambda x: x[1][1]  # 按时间排序
        )
        
        # 保留最新的cache_size个条目
        new_cache = {}
        for key, (result, cache_time) in sorted_cache[-self.config.cache_size:]:
            if current_time - cache_time < self.config.cache_ttl:
                new_cache[key] = (result, cache_time)
        
        self.cache = new_cache
    
    def _apply_filters(self, result: RetrievalResult) -> RetrievalResult:
        """应用过滤器"""
        if not self.config.metadata_filters:
            return result
        
        filtered_docs = []
        filtered_scores = []
        
        for i, doc in enumerate(result.documents):
            if self._match_filters(doc, self.config.metadata_filters):
                filtered_docs.append(doc)
                if result.scores:
                    filtered_scores.append(result.scores[i])
        
        result.documents = filtered_docs
        result.scores = filtered_scores if result.scores else None
        
        return result
    
    def _match_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """检查文档是否匹配过滤器"""
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            
            if isinstance(value, list):
                if doc.metadata[key] not in value:
                    return False
            elif doc.metadata[key] != value:
                return False
        
        return True
    
    def _update_stats(self, result: RetrievalResult):
        """更新统计信息"""
        self.stats["total_queries"] += 1
        self.stats["total_documents_retrieved"] += len(result.documents)
        self.stats["total_time"] += result.retrieval_time
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算平均值
        if stats["total_queries"] > 0:
            stats["avg_time_per_query"] = stats["total_time"] / stats["total_queries"]
            stats["avg_docs_per_query"] = stats["total_documents_retrieved"] / stats["total_queries"]
        
        # 缓存统计
        total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        else:
            stats["cache_hit_rate"] = 0.0
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("检索器缓存已清空")


class VectorRetriever(BaseRetriever):
    """向量检索器"""
    
    def __init__(self, 
                 vector_store_manager,
                 config: Optional[RetrievalConfig] = None):
        """
        初始化向量检索器
        
        Args:
            vector_store_manager: 向量存储管理器
            config: 检索配置
        """
        super().__init__(config)
        self.vector_store_manager = vector_store_manager
        
        logger.info("初始化向量检索器")
    
    def _retrieve(self, query: str, k: int, **kwargs) -> RetrievalResult:
        """执行向量检索"""
        try:
            # 使用向量存储管理器进行搜索
            search_type = kwargs.get('search_type', 'similarity')
            
            if search_type == 'similarity_with_score':
                results_with_scores = self.vector_store_manager.search_with_score(
                    query, k=k, **kwargs
                )
                documents = [doc for doc, score in results_with_scores]
                scores = [score for doc, score in results_with_scores]
            else:
                documents = self.vector_store_manager.search(
                    query, k=k, search_type=search_type, **kwargs
                )
                scores = None
            
            # 应用相似度阈值
            if scores and self.config.similarity_threshold > 0:
                filtered_docs = []
                filtered_scores = []
                
                for doc, score in zip(documents, scores):
                    if score >= self.config.similarity_threshold:
                        filtered_docs.append(doc)
                        filtered_scores.append(score)
                
                documents = filtered_docs
                scores = filtered_scores
            
            return RetrievalResult(
                documents=documents,
                scores=scores
            )
        
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise


class KeywordRetriever(BaseRetriever):
    """关键词检索器（基于BM25）"""
    
    def __init__(self,
                 documents: List[Document],
                 config: Optional[RetrievalConfig] = None):
        """
        初始化关键词检索器
        
        Args:
            documents: 文档列表
            config: 检索配置
        """
        if not HAS_BM25:
            raise ImportError("需要安装rank-bm25: pip install rank-bm25")
        
        super().__init__(config)
        self.documents = documents
        
        # 构建BM25索引
        self._build_bm25_index()
        
        logger.info(f"初始化关键词检索器，文档数量: {len(documents)}")
    
    def _build_bm25_index(self):
        """构建BM25索引"""
        try:
            # 预处理文档文本
            corpus = []
            for doc in self.documents:
                # 简单的文本预处理
                text = doc.page_content.lower()
                tokens = text.split()  # 简单分词
                corpus.append(tokens)
            
            # 构建BM25索引
            self.bm25 = BM25Okapi(corpus)
            
            logger.info("BM25索引构建完成")
        
        except Exception as e:
            logger.error(f"构建BM25索引失败: {e}")
            raise
    
    def _retrieve(self, query: str, k: int, **kwargs) -> RetrievalResult:
        """执行关键词检索"""
        try:
            # 预处理查询
            query_tokens = query.lower().split()
            
            # 获取BM25分数
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top-k结果
            top_indices = np.argsort(scores)[::-1][:k]
            
            # 构建结果
            documents = [self.documents[i] for i in top_indices]
            doc_scores = [scores[i] for i in top_indices]
            
            return RetrievalResult(
                documents=documents,
                scores=doc_scores
            )
        
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            raise
    
    def add_documents(self, new_documents: List[Document]):
        """添加新文档"""
        self.documents.extend(new_documents)
        self._build_bm25_index()
        logger.info(f"添加 {len(new_documents)} 个文档，重建BM25索引")


class HybridRetriever(BaseRetriever):
    """混合检索器（结合向量检索和关键词检索）"""
    
    def __init__(self,
                 vector_retriever: VectorRetriever,
                 keyword_retriever: KeywordRetriever,
                 config: Optional[RetrievalConfig] = None,
                 vector_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器
            config: 检索配置
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        """
        super().__init__(config)
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # 权重归一化
        total_weight = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
        
        logger.info(f"初始化混合检索器，权重 - 向量: {self.vector_weight:.2f}, 关键词: {self.keyword_weight:.2f}")
    
    def _retrieve(self, query: str, k: int, **kwargs) -> RetrievalResult:
        """执行混合检索"""
        try:
            # 分别进行向量检索和关键词检索
            vector_k = min(k * 2, self.config.max_k)  # 获取更多候选
            keyword_k = min(k * 2, len(self.keyword_retriever.documents))
            
            # 向量检索
            vector_result = self.vector_retriever.retrieve(query, vector_k)
            
            # 关键词检索
            keyword_result = self.keyword_retriever.retrieve(query, keyword_k)
            
            # 融合结果
            fused_result = self._fuse_results(vector_result, keyword_result, k)
            
            return fused_result
        
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise
    
    def _fuse_results(self, 
                     vector_result: RetrievalResult,
                     keyword_result: RetrievalResult,
                     k: int) -> RetrievalResult:
        """融合检索结果"""
        # 创建文档到分数的映射
        doc_scores = {}
        
        # 处理向量检索结果
        vector_scores = vector_result.scores or [1.0] * len(vector_result.documents)
        for doc, score in zip(vector_result.documents, vector_scores):
            doc_id = self._get_doc_id(doc)
            normalized_score = self._normalize_score(score, 'vector')
            doc_scores[doc_id] = {
                'document': doc,
                'vector_score': normalized_score,
                'keyword_score': 0.0
            }
        
        # 处理关键词检索结果
        keyword_scores = keyword_result.scores or [1.0] * len(keyword_result.documents)
        for doc, score in zip(keyword_result.documents, keyword_scores):
            doc_id = self._get_doc_id(doc)
            normalized_score = self._normalize_score(score, 'keyword')
            
            if doc_id in doc_scores:
                doc_scores[doc_id]['keyword_score'] = normalized_score
            else:
                doc_scores[doc_id] = {
                    'document': doc,
                    'vector_score': 0.0,
                    'keyword_score': normalized_score
                }
        
        # 计算融合分数
        for doc_info in doc_scores.values():
            combined_score = (
                self.vector_weight * doc_info['vector_score'] +
                self.keyword_weight * doc_info['keyword_score']
            )
            doc_info['combined_score'] = combined_score
        
        # 排序并取top-k
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:k]
        
        # 构建最终结果
        final_documents = [item['document'] for item in sorted_docs]
        final_scores = [item['combined_score'] for item in sorted_docs]
        
        return RetrievalResult(
            documents=final_documents,
            scores=final_scores,
            metadata={
                'fusion_method': 'weighted_combination',
                'vector_weight': self.vector_weight,
                'keyword_weight': self.keyword_weight
            }
        )
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档ID"""
        # 使用文档内容的哈希作为ID
        import hashlib
        return hashlib.md5(doc.page_content.encode()).hexdigest()
    
    def _normalize_score(self, score: float, score_type: str) -> float:
        """归一化分数"""
        if score_type == 'vector':
            # 向量相似度通常在[0, 1]范围内
            return max(0.0, min(1.0, score))
        elif score_type == 'keyword':
            # BM25分数可能很大，使用sigmoid归一化
            import math
            return 1.0 / (1.0 + math.exp(-score / 10.0))
        else:
            return score


class MultiQueryRetriever(BaseRetriever):
    """多查询检索器（查询扩展）"""
    
    def __init__(self,
                 base_retriever: BaseRetriever,
                 llm,
                 config: Optional[RetrievalConfig] = None,
                 num_queries: int = 3):
        """
        初始化多查询检索器
        
        Args:
            base_retriever: 基础检索器
            llm: 用于查询扩展的LLM
            config: 检索配置
            num_queries: 生成的查询数量
        """
        super().__init__(config)
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_queries = num_queries
        
        logger.info(f"初始化多查询检索器，查询扩展数量: {num_queries}")
    
    def _retrieve(self, query: str, k: int, **kwargs) -> RetrievalResult:
        """执行多查询检索"""
        try:
            # 生成多个查询
            expanded_queries = self._expand_query(query)
            
            # 对每个查询执行检索
            all_results = []
            for expanded_query in expanded_queries:
                result = self.base_retriever.retrieve(expanded_query, k, **kwargs)
                all_results.append(result)
            
            # 融合所有结果
            fused_result = self._fuse_multi_results(all_results, k)
            
            return fused_result
        
        except Exception as e:
            logger.error(f"多查询检索失败: {e}")
            raise
    
    def _expand_query(self, query: str) -> List[str]:
        """扩展查询"""
        try:
            prompt = f"""
            基于以下查询，生成{self.num_queries}个相关但不同的查询，用于改善信息检索。
            确保每个查询都从不同角度探索相同的信息需求。
            
            原始查询: {query}
            
            请生成{self.num_queries}个扩展查询，每行一个：
            """
            
            response = self.llm.invoke(prompt)
            
            # 解析响应
            lines = response.content.strip().split('\n')
            expanded_queries = [query]  # 包含原始查询
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '-', '*')):
                    # 清理编号前缀
                    clean_query = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                        if clean_query.startswith(prefix):
                            clean_query = clean_query[len(prefix):].strip()
                    
                    if clean_query and clean_query != query:
                        expanded_queries.append(clean_query)
            
            # 限制查询数量
            return expanded_queries[:self.num_queries + 1]
        
        except Exception as e:
            logger.warning(f"查询扩展失败，使用原始查询: {e}")
            return [query]
    
    def _fuse_multi_results(self, 
                           results: List[RetrievalResult], 
                           k: int) -> RetrievalResult:
        """融合多个检索结果"""
        # 使用倒数排名融合（Reciprocal Rank Fusion）
        doc_scores = {}
        
        for result in results:
            for rank, doc in enumerate(result.documents):
                doc_id = self._get_doc_id(doc)
                
                # RRF分数：1 / (rank + 60)
                rrf_score = 1.0 / (rank + 60)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += rrf_score
                else:
                    doc_scores[doc_id] = {
                        'document': doc,
                        'score': rrf_score
                    }
        
        # 排序并取top-k
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        
        final_documents = [item['document'] for item in sorted_docs]
        final_scores = [item['score'] for item in sorted_docs]
        
        return RetrievalResult(
            documents=final_documents,
            scores=final_scores,
            metadata={
                'fusion_method': 'reciprocal_rank_fusion',
                'num_queries': len(results)
            }
        )
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档ID"""
        import hashlib
        return hashlib.md5(doc.page_content.encode()).hexdigest()


# 便捷函数
def create_hybrid_retriever(vector_store_manager,
                           documents: List[Document],
                           config: Optional[RetrievalConfig] = None,
                           vector_weight: float = 0.7) -> HybridRetriever:
    """
    创建混合检索器的便捷函数
    
    Args:
        vector_store_manager: 向量存储管理器
        documents: 文档列表（用于关键词检索）
        config: 检索配置
        vector_weight: 向量检索权重
        
    Returns:
        HybridRetriever实例
    """
    # 创建向量检索器
    vector_retriever = VectorRetriever(vector_store_manager, config)
    
    # 创建关键词检索器
    keyword_retriever = KeywordRetriever(documents, config)
    
    # 创建混合检索器
    return HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        config=config,
        vector_weight=vector_weight,
        keyword_weight=1.0 - vector_weight
    )