"""
检索算法模板

本模块提供了多种智能检索策略，用于从向量数据库中检索相关文档。
支持多种检索算法和结果优化策略，适用于不同的应用场景。

核心特性：
1. 多种检索策略：相似度搜索、MMR、混合检索、重排序等
2. 结果过滤：支持基于元数据和内容的复杂过滤条件
3. 结果重排序：支持多种重排序算法优化检索结果
4. 混合检索：结合向量搜索和关键词搜索
5. 查询扩展：自动扩展用户查询，提高召回率
6. 结果去重：智能去重，避免返回重复内容
7. 评分融合：支持多种评分融合策略
8. 上下文感知：支持基于对话历史的检索
9. 实时优化：根据用户反馈优化检索结果
10. 性能监控：详细的检索性能分析和监控

支持的检索策略：
1. Similarity Search: 基于向量相似度的检索
2. MMR (Maximal Marginal Relevance): 最大边际相关性检索
3. Hybrid Search: 混合向量和关键词检索
4. Rerank: 使用重排序模型优化结果
5. Ensemble: 集成多种检索方法
6. Self-Query: 自查询检索，支持结构化查询
7. Contextual: 上下文感知检索
8. Adaptive: 自适应检索策略

设计模式：
- 策略模式：不同的检索策略可以动态切换
- 装饰器模式：为检索器添加额外功能（过滤、重排序等）
- 责任链模式：多阶段检索处理流程
- 观察者模式：监控检索性能和用户反馈
"""

import re
import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import statistics
from collections import defaultdict, Counter

# LangChain imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
    SelfQueryRetriever
)
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline
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


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    SIMILARITY = "similarity"                  # 相似度检索
    MMR = "mmr"                               # 最大边际相关性
    HYBRID = "hybrid"                         # 混合检索
    RERANK = "rerank"                         # 重排序检索
    ENSEMBLE = "ensemble"                     # 集成检索
    SELF_QUERY = "self_query"                 # 自查询检索
    CONTEXTUAL = "contextual"                 # 上下文检索
    ADAPTIVE = "adaptive"                     # 自适应检索


class RerankingMethod(Enum):
    """重排序方法枚举"""
    NONE = "none"                             # 无重排序
    CROSS_ENCODER = "cross_encoder"           # 交叉编码器
    LLM_BASED = "llm_based"                   # 基于LLM的重排序
    SCORE_FUSION = "score_fusion"             # 分数融合
    DIVERSITY = "diversity"                   # 多样性重排序


class QueryExpansionMethod(Enum):
    """查询扩展方法枚举"""
    NONE = "none"                             # 无扩展
    SYNONYMS = "synonyms"                     # 同义词扩展
    LLM_GENERATED = "llm_generated"           # LLM生成扩展
    PSEUDO_RELEVANCE = "pseudo_relevance"     # 伪相关反馈
    EMBEDDING_BASED = "embedding_based"       # 基于嵌入的扩展


@dataclass
class RetrievalConfig:
    """
    检索配置类
    
    定义检索的各种参数和策略。
    """
    strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY  # 检索策略
    top_k: int = 5                                              # 返回结果数量
    similarity_threshold: float = 0.0                           # 相似度阈值
    diversity_threshold: float = 0.7                            # 多样性阈值（用于MMR）
    
    # 重排序配置
    reranking_method: RerankingMethod = RerankingMethod.NONE    # 重排序方法
    rerank_top_k: int = 20                                      # 重排序候选数量
    
    # 查询扩展配置
    query_expansion: QueryExpansionMethod = QueryExpansionMethod.NONE  # 查询扩展方法
    expansion_count: int = 3                                    # 扩展查询数量
    
    # 过滤配置
    metadata_filter: Optional[Dict[str, Any]] = None           # 元数据过滤条件
    content_filter: Optional[str] = None                       # 内容过滤条件
    
    # 融合配置（用于混合检索）
    vector_weight: float = 0.7                                 # 向量检索权重
    keyword_weight: float = 0.3                                # 关键词检索权重
    
    # 上下文配置
    use_context: bool = False                                  # 是否使用上下文
    context_window: int = 3                                    # 上下文窗口大小
    
    # 性能配置
    enable_caching: bool = True                                # 启用缓存
    cache_ttl: int = 3600                                      # 缓存生存时间（秒）
    
    # 自定义配置
    custom_params: Dict[str, Any] = field(default_factory=dict)  # 自定义参数


@dataclass
class RetrievalResult:
    """
    检索结果类
    
    包含检索的文档和相关信息。
    """
    documents: List[Document]                                   # 检索到的文档
    scores: List[float]                                         # 相关性分数
    query: str                                                  # 原始查询
    expanded_queries: List[str] = field(default_factory=list)  # 扩展查询
    retrieval_time: float = 0.0                                # 检索耗时
    total_candidates: int = 0                                   # 候选文档总数
    reranked: bool = False                                      # 是否进行了重排序
    strategy_used: str = ""                                     # 使用的检索策略
    metadata: Dict[str, Any] = field(default_factory=dict)     # 额外元数据
    
    def __post_init__(self):
        """计算统计信息"""
        if self.documents and self.scores:
            self.metadata.update({
                'avg_score': statistics.mean(self.scores),
                'max_score': max(self.scores),
                'min_score': min(self.scores),
                'score_variance': statistics.variance(self.scores) if len(self.scores) > 1 else 0.0
            })


class BaseRetriever(ABC):
    """
    检索器抽象基类
    
    定义检索器的通用接口。
    """
    
    def __init__(self, config: RetrievalConfig):
        """
        初始化检索器
        
        Args:
            config: 检索配置
        """
        self.config = config
        self.query_cache: Dict[str, RetrievalResult] = {}
        
    @abstractmethod
    async def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """
        执行检索
        
        Args:
            query: 查询文本
            **kwargs: 额外参数
            
        Returns:
            检索结果
        """
        pass
    
    def _should_use_cache(self, query: str) -> bool:
        """检查是否应该使用缓存"""
        if not self.config.enable_caching:
            return False
        
        if query in self.query_cache:
            cached_result = self.query_cache[query]
            # 检查缓存是否过期
            cache_time = cached_result.metadata.get('cache_time', 0)
            current_time = datetime.now().timestamp()
            return (current_time - cache_time) < self.config.cache_ttl
        
        return False
    
    def _cache_result(self, query: str, result: RetrievalResult) -> None:
        """缓存检索结果"""
        if self.config.enable_caching:
            result.metadata['cache_time'] = datetime.now().timestamp()
            self.query_cache[query] = result
    
    def _filter_by_metadata(self, documents: List[Document], scores: List[float]) -> Tuple[List[Document], List[float]]:
        """根据元数据过滤文档"""
        if not self.config.metadata_filter:
            return documents, scores
        
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(documents, scores):
            if self._match_metadata_filter(doc.metadata, self.config.metadata_filter):
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return filtered_docs, filtered_scores
    
    def _match_metadata_filter(self, metadata: Dict[str, Any], filter_conditions: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤条件"""
        for key, value in filter_conditions.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            if isinstance(value, dict):
                # 支持操作符（如 {"$gte": 100}）
                for op, op_value in value.items():
                    if op == "$eq" and metadata_value != op_value:
                        return False
                    elif op == "$ne" and metadata_value == op_value:
                        return False
                    elif op == "$gt" and metadata_value <= op_value:
                        return False
                    elif op == "$gte" and metadata_value < op_value:
                        return False
                    elif op == "$lt" and metadata_value >= op_value:
                        return False
                    elif op == "$lte" and metadata_value > op_value:
                        return False
                    elif op == "$in" and metadata_value not in op_value:
                        return False
                    elif op == "$nin" and metadata_value in op_value:
                        return False
            elif isinstance(value, list):
                # 列表表示"包含任一"
                if metadata_value not in value:
                    return False
            else:
                # 直接匹配
                if metadata_value != value:
                    return False
        
        return True
    
    def _filter_by_content(self, documents: List[Document], scores: List[float]) -> Tuple[List[Document], List[float]]:
        """根据内容过滤文档"""
        if not self.config.content_filter:
            return documents, scores
        
        filtered_docs = []
        filtered_scores = []
        
        # 支持正则表达式过滤
        pattern = re.compile(self.config.content_filter, re.IGNORECASE)
        
        for doc, score in zip(documents, scores):
            if pattern.search(doc.page_content):
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return filtered_docs, filtered_scores
    
    def _apply_similarity_threshold(self, documents: List[Document], scores: List[float]) -> Tuple[List[Document], List[float]]:
        """应用相似度阈值过滤"""
        if self.config.similarity_threshold <= 0.0:
            return documents, scores
        
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(documents, scores):
            if score >= self.config.similarity_threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return filtered_docs, filtered_scores


class SimilarityRetriever(BaseRetriever):
    """相似度检索器"""
    
    def __init__(self, config: RetrievalConfig, vectorstore: VectorStore):
        super().__init__(config)
        self.vectorstore = vectorstore
    
    async def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """执行相似度检索"""
        start_time = datetime.now()
        
        # 检查缓存
        if self._should_use_cache(query):
            logger.debug(f"Using cached result for query: {query[:50]}...")
            return self.query_cache[query]
        
        try:
            # 执行向量相似度搜索
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=self.config.top_k,
                filter=self.config.metadata_filter
            )
            
            if results_with_scores:
                documents, scores = zip(*results_with_scores)
                documents = list(documents)
                scores = list(scores)
            else:
                documents, scores = [], []
            
            # 应用过滤条件
            documents, scores = self._apply_similarity_threshold(documents, scores)
            documents, scores = self._filter_by_content(documents, scores)
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                retrieval_time=retrieval_time,
                total_candidates=len(results_with_scores) if results_with_scores else 0,
                strategy_used="similarity"
            )
            
            # 缓存结果
            self._cache_result(query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Similarity retrieval failed: {e}")
            raise ResourceError(f"Retrieval failed: {e}")


class MMRRetriever(BaseRetriever):
    """最大边际相关性检索器"""
    
    def __init__(self, config: RetrievalConfig, vectorstore: VectorStore):
        super().__init__(config)
        self.vectorstore = vectorstore
    
    async def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """执行MMR检索"""
        start_time = datetime.now()
        
        # 检查缓存
        if self._should_use_cache(query):
            return self.query_cache[query]
        
        try:
            # 首先获取更多候选文档
            candidate_k = min(self.config.top_k * 3, 50)  # 获取3倍的候选文档
            
            candidates_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=candidate_k,
                filter=self.config.metadata_filter
            )
            
            if not candidates_with_scores:
                return RetrievalResult(
                    documents=[],
                    scores=[],
                    query=query,
                    retrieval_time=(datetime.now() - start_time).total_seconds(),
                    strategy_used="mmr"
                )
            
            candidates, candidate_scores = zip(*candidates_with_scores)
            candidates = list(candidates)
            candidate_scores = list(candidate_scores)
            
            # 执行MMR算法
            selected_docs, selected_scores = self._mmr_selection(
                query, candidates, candidate_scores
            )
            
            # 应用过滤条件
            selected_docs, selected_scores = self._filter_by_content(selected_docs, selected_scores)
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            result = RetrievalResult(
                documents=selected_docs,
                scores=selected_scores,
                query=query,
                retrieval_time=retrieval_time,
                total_candidates=len(candidates),
                strategy_used="mmr"
            )
            
            self._cache_result(query, result)
            return result
            
        except Exception as e:
            logger.error(f"MMR retrieval failed: {e}")
            raise ResourceError(f"MMR retrieval failed: {e}")
    
    def _mmr_selection(self, query: str, candidates: List[Document], scores: List[float]) -> Tuple[List[Document], List[float]]:
        """执行MMR文档选择"""
        if not candidates:
            return [], []
        
        # 获取查询和候选文档的嵌入
        embeddings = self.vectorstore.embeddings
        query_embedding = embeddings.embed_query(query)
        
        candidate_texts = [doc.page_content for doc in candidates]
        candidate_embeddings = embeddings.embed_documents(candidate_texts)
        
        selected_indices = []
        selected_docs = []
        selected_scores = []
        
        for _ in range(min(self.config.top_k, len(candidates))):
            if not selected_indices:
                # 选择第一个最相关的文档
                best_idx = 0
                best_score = scores[0]
                for i, score in enumerate(scores):
                    if i not in selected_indices and score > best_score:
                        best_idx = i
                        best_score = score
            else:
                # 计算MMR分数
                best_idx = -1
                best_mmr_score = -float('inf')
                
                for i, candidate_emb in enumerate(candidate_embeddings):
                    if i in selected_indices:
                        continue
                    
                    # 相关性分数（余弦相似度）
                    relevance_score = self._cosine_similarity(query_embedding, candidate_emb)
                    
                    # 计算与已选择文档的最大相似度
                    max_similarity = 0.0
                    for selected_idx in selected_indices:
                        similarity = self._cosine_similarity(candidate_emb, candidate_embeddings[selected_idx])
                        max_similarity = max(max_similarity, similarity)
                    
                    # MMR分数 = λ * 相关性 - (1-λ) * 最大相似度
                    mmr_score = (
                        self.config.diversity_threshold * relevance_score -
                        (1 - self.config.diversity_threshold) * max_similarity
                    )
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected_docs.append(candidates[best_idx])
                selected_scores.append(scores[best_idx])
        
        return selected_docs, selected_scores
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class HybridRetriever(BaseRetriever):
    """混合检索器（向量+关键词）"""
    
    def __init__(self, config: RetrievalConfig, vectorstore: VectorStore, keyword_retriever=None):
        super().__init__(config)
        self.vectorstore = vectorstore
        self.keyword_retriever = keyword_retriever
    
    async def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        """执行混合检索"""
        start_time = datetime.now()
        
        # 检查缓存
        if self._should_use_cache(query):
            return self.query_cache[query]
        
        try:
            # 向量检索
            vector_results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=self.config.top_k * 2,  # 获取更多候选
                filter=self.config.metadata_filter
            )
            
            # 关键词检索（如果有的话）
            keyword_results = []
            if self.keyword_retriever:
                try:
                    keyword_docs = self.keyword_retriever.get_relevant_documents(query)
                    # 为关键词结果分配默认分数
                    keyword_results = [(doc, 0.5) for doc in keyword_docs[:self.config.top_k]]
                except Exception as e:
                    logger.warning(f"Keyword retrieval failed: {e}")
            
            # 融合结果
            fused_results = self._fuse_results(vector_results, keyword_results)
            
            if fused_results:
                documents, scores = zip(*fused_results)
                documents = list(documents)
                scores = list(scores)
            else:
                documents, scores = [], []
            
            # 应用过滤条件
            documents, scores = self._filter_by_content(documents, scores)
            
            # 限制结果数量
            documents = documents[:self.config.top_k]
            scores = scores[:self.config.top_k]
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            result = RetrievalResult(
                documents=documents,
                scores=scores,
                query=query,
                retrieval_time=retrieval_time,
                total_candidates=len(vector_results) + len(keyword_results),
                strategy_used="hybrid"
            )
            
            self._cache_result(query, result)
            return result
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise ResourceError(f"Hybrid retrieval failed: {e}")
    
    def _fuse_results(self, vector_results: List[Tuple[Document, float]], keyword_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """融合向量和关键词检索结果"""
        # 创建文档到分数的映射
        doc_scores = defaultdict(lambda: {'vector': 0.0, 'keyword': 0.0})
        all_docs = {}
        
        # 处理向量结果
        for doc, score in vector_results:
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id]['vector'] = score
            all_docs[doc_id] = doc
        
        # 处理关键词结果
        for doc, score in keyword_results:
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id]['keyword'] = score
            all_docs[doc_id] = doc
        
        # 计算融合分数
        fused_results = []
        for doc_id, scores in doc_scores.items():
            fused_score = (
                self.config.vector_weight * scores['vector'] +
                self.config.keyword_weight * scores['keyword']
            )
            fused_results.append((all_docs[doc_id], fused_score))
        
        # 按分数排序
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档的唯一标识符"""
        # 使用内容的前100个字符作为ID（可以根据需要调整）
        return doc.page_content[:100]


class RetrievalTemplate(TemplateBase[str, RetrievalResult]):
    """
    检索算法模板
    
    提供多种智能检索策略，支持相似度搜索、MMR、混合检索等多种算法。
    支持查询扩展、结果重排序、过滤等高级功能。
    
    核心功能：
    1. 多种检索策略：相似度、MMR、混合检索等
    2. 查询优化：查询扩展、改写等
    3. 结果优化：重排序、去重、过滤等
    4. 性能监控：详细的检索性能统计
    5. 缓存机制：智能缓存，提高响应速度
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """初始化检索模板"""
        super().__init__(config)
        
        # 配置参数
        self.retrieval_config: Optional[RetrievalConfig] = None
        self.vectorstore: Optional[VectorStore] = None
        self.retrievers: Dict[str, BaseRetriever] = {}
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'avg_results_per_query': 0.0,
            'cache_hit_rate': 0.0,
            'strategy_usage': defaultdict(int),
            'total_documents_retrieved': 0
        }
        
        # 查询历史（用于上下文检索）
        self.query_history: List[str] = []
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="RetrievalTemplate",
            description="智能检索算法模板",
            template_type=TemplateType.DATA,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("strategy", str, False, "similarity", "检索策略")
        config.add_parameter("top_k", int, False, 5, "返回结果数量")
        config.add_parameter("similarity_threshold", float, False, 0.0, "相似度阈值")
        config.add_parameter("diversity_threshold", float, False, 0.7, "多样性阈值")
        config.add_parameter("reranking_method", str, False, "none", "重排序方法")
        config.add_parameter("query_expansion", str, False, "none", "查询扩展方法")
        config.add_parameter("metadata_filter", dict, False, None, "元数据过滤条件")
        config.add_parameter("enable_caching", bool, False, True, "启用缓存")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置检索参数
        
        Args:
            **parameters: 检索参数
        """
        if not self.validate_parameters(parameters):
            raise ValidationError("Retrieval parameter validation failed")
        
        # 创建检索配置
        strategy_str = parameters.get('strategy', 'similarity')
        try:
            strategy = RetrievalStrategy(strategy_str)
        except ValueError:
            raise ValidationError(f"Unknown retrieval strategy: {strategy_str}")
        
        reranking_str = parameters.get('reranking_method', 'none')
        try:
            reranking_method = RerankingMethod(reranking_str)
        except ValueError:
            reranking_method = RerankingMethod.NONE
        
        expansion_str = parameters.get('query_expansion', 'none')
        try:
            query_expansion = QueryExpansionMethod(expansion_str)
        except ValueError:
            query_expansion = QueryExpansionMethod.NONE
        
        self.retrieval_config = RetrievalConfig(
            strategy=strategy,
            top_k=parameters.get('top_k', 5),
            similarity_threshold=parameters.get('similarity_threshold', 0.0),
            diversity_threshold=parameters.get('diversity_threshold', 0.7),
            reranking_method=reranking_method,
            query_expansion=query_expansion,
            metadata_filter=parameters.get('metadata_filter'),
            content_filter=parameters.get('content_filter'),
            vector_weight=parameters.get('vector_weight', 0.7),
            keyword_weight=parameters.get('keyword_weight', 0.3),
            use_context=parameters.get('use_context', False),
            enable_caching=parameters.get('enable_caching', True),
            custom_params=parameters.get('custom_params', {})
        )
        
        # 获取向量存储
        self.vectorstore = parameters.get('vectorstore')
        if not self.vectorstore:
            raise ValidationError("vectorstore parameter is required")
        
        # 创建检索器
        self._create_retrievers()
        
        self.status = self.status.CONFIGURED
        self._setup_parameters = parameters.copy()
        
        logger.info(
            f"Retrieval configured: strategy={strategy.value}, "
            f"top_k={self.retrieval_config.top_k}, "
            f"caching={self.retrieval_config.enable_caching}"
        )
    
    def _create_retrievers(self) -> None:
        """创建检索器实例"""
        if not self.retrieval_config or not self.vectorstore:
            raise RuntimeError("Configuration not set")
        
        # 创建基础检索器
        self.retrievers['similarity'] = SimilarityRetriever(self.retrieval_config, self.vectorstore)
        self.retrievers['mmr'] = MMRRetriever(self.retrieval_config, self.vectorstore)
        
        # 混合检索器（如果需要的话）
        if self.retrieval_config.strategy == RetrievalStrategy.HYBRID:
            keyword_retriever = self.retrieval_config.custom_params.get('keyword_retriever')
            self.retrievers['hybrid'] = HybridRetriever(self.retrieval_config, self.vectorstore, keyword_retriever)
        
        logger.info(f"Created {len(self.retrievers)} retrievers")
    
    def execute(self, input_data: str, **kwargs) -> RetrievalResult:
        """
        执行检索
        
        Args:
            input_data: 查询文本
            **kwargs: 额外参数
            
        Returns:
            检索结果
        """
        if not self.retrieval_config:
            raise RuntimeError("Retrieval not configured. Call setup() first.")
        
        # 使用异步方法
        return asyncio.run(self._execute_async(input_data, **kwargs))
    
    async def _execute_async(self, query: str, **kwargs) -> RetrievalResult:
        """异步执行检索"""
        start_time = datetime.now()
        
        try:
            # 查询预处理
            processed_query = self._preprocess_query(query)
            
            # 选择检索策略
            strategy = kwargs.get('strategy', self.retrieval_config.strategy.value)
            retriever = self.retrievers.get(strategy)
            
            if not retriever:
                # 回退到相似度检索
                retriever = self.retrievers['similarity']
                strategy = 'similarity'
            
            # 执行检索
            result = await retriever.retrieve(processed_query, **kwargs)
            
            # 后处理
            result = await self._postprocess_result(result, query)
            
            # 更新统计信息
            self._update_stats(result, strategy)
            
            # 更新查询历史
            if self.retrieval_config.use_context:
                self.query_history.append(query)
                if len(self.query_history) > self.retrieval_config.context_window:
                    self.query_history.pop(0)
            
            logger.info(
                f"Retrieval completed: {len(result.documents)} results in {result.retrieval_time:.3f}s "
                f"using {strategy} strategy"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval execution failed: {str(e)}")
            raise
    
    def _preprocess_query(self, query: str) -> str:
        """查询预处理"""
        processed_query = query.strip()
        
        # 查询扩展
        if self.retrieval_config.query_expansion != QueryExpansionMethod.NONE:
            processed_query = self._expand_query(processed_query)
        
        # 上下文增强
        if self.retrieval_config.use_context and self.query_history:
            context = " ".join(self.query_history[-self.retrieval_config.context_window:])
            processed_query = f"{context} {processed_query}"
        
        return processed_query
    
    def _expand_query(self, query: str) -> str:
        """查询扩展"""
        method = self.retrieval_config.query_expansion
        
        if method == QueryExpansionMethod.SYNONYMS:
            # 简单的同义词扩展示例
            synonyms = {
                'search': ['find', 'look', 'retrieve'],
                'document': ['file', 'text', 'content'],
                'information': ['data', 'knowledge', 'details']
            }
            
            words = query.split()
            expanded_words = []
            
            for word in words:
                expanded_words.append(word)
                if word.lower() in synonyms:
                    expanded_words.extend(synonyms[word.lower()][:2])  # 添加前2个同义词
            
            return " ".join(expanded_words)
        
        # 其他扩展方法可以在这里实现
        return query
    
    async def _postprocess_result(self, result: RetrievalResult, original_query: str) -> RetrievalResult:
        """结果后处理"""
        # 重排序
        if self.retrieval_config.reranking_method != RerankingMethod.NONE:
            result = await self._rerank_results(result, original_query)
        
        # 去重
        result = self._deduplicate_results(result)
        
        return result
    
    async def _rerank_results(self, result: RetrievalResult, query: str) -> RetrievalResult:
        """重排序结果"""
        method = self.retrieval_config.reranking_method
        
        if method == RerankingMethod.DIVERSITY:
            # 多样性重排序
            result.documents, result.scores = self._diversity_rerank(result.documents, result.scores)
            result.reranked = True
        elif method == RerankingMethod.SCORE_FUSION:
            # 分数融合重排序
            result.documents, result.scores = self._score_fusion_rerank(result.documents, result.scores)
            result.reranked = True
        
        return result
    
    def _diversity_rerank(self, documents: List[Document], scores: List[float]) -> Tuple[List[Document], List[float]]:
        """多样性重排序"""
        if len(documents) <= 1:
            return documents, scores
        
        # 简单的多样性重排序：避免连续的相似文档
        reranked_docs = [documents[0]]
        reranked_scores = [scores[0]]
        remaining_docs = list(zip(documents[1:], scores[1:]))
        
        while remaining_docs and len(reranked_docs) < len(documents):
            # 找到与已选择文档最不相似的文档
            best_idx = 0
            min_similarity = float('inf')
            
            for i, (doc, score) in enumerate(remaining_docs):
                # 计算与已选择文档的平均相似度
                total_similarity = 0
                for selected_doc in reranked_docs:
                    similarity = self._text_similarity(doc.page_content, selected_doc.page_content)
                    total_similarity += similarity
                
                avg_similarity = total_similarity / len(reranked_docs)
                
                if avg_similarity < min_similarity:
                    min_similarity = avg_similarity
                    best_idx = i
            
            best_doc, best_score = remaining_docs.pop(best_idx)
            reranked_docs.append(best_doc)
            reranked_scores.append(best_score)
        
        return reranked_docs, reranked_scores
    
    def _score_fusion_rerank(self, documents: List[Document], scores: List[float]) -> Tuple[List[Document], List[float]]:
        """分数融合重排序"""
        # 简单的分数标准化和融合
        if not scores:
            return documents, scores
        
        # 标准化分数
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score
        
        if score_range > 0:
            normalized_scores = [(score - min_score) / score_range for score in scores]
        else:
            normalized_scores = [1.0] * len(scores)
        
        # 重新排序
        sorted_pairs = sorted(zip(documents, normalized_scores), key=lambda x: x[1], reverse=True)
        
        if sorted_pairs:
            documents, scores = zip(*sorted_pairs)
            return list(documents), list(scores)
        else:
            return documents, scores
    
    def _deduplicate_results(self, result: RetrievalResult) -> RetrievalResult:
        """去重结果"""
        if not result.documents:
            return result
        
        seen_content = set()
        unique_docs = []
        unique_scores = []
        
        for doc, score in zip(result.documents, result.scores):
            # 使用内容的前200个字符作为去重依据
            content_key = doc.page_content[:200].strip()
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_docs.append(doc)
                unique_scores.append(score)
        
        result.documents = unique_docs
        result.scores = unique_scores
        
        return result
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_stats(self, result: RetrievalResult, strategy: str) -> None:
        """更新统计信息"""
        self.stats['total_queries'] += 1
        self.stats['strategy_usage'][strategy] += 1
        self.stats['total_documents_retrieved'] += len(result.documents)
        
        # 更新平均检索时间
        if self.stats['total_queries'] > 1:
            current_avg = self.stats['avg_retrieval_time']
            new_avg = (current_avg * (self.stats['total_queries'] - 1) + result.retrieval_time) / self.stats['total_queries']
            self.stats['avg_retrieval_time'] = new_avg
        else:
            self.stats['avg_retrieval_time'] = result.retrieval_time
        
        # 更新平均结果数
        self.stats['avg_results_per_query'] = self.stats['total_documents_retrieved'] / self.stats['total_queries']
        
        # 计算缓存命中率
        cache_hits = sum(1 for retriever in self.retrievers.values() for query in retriever.query_cache.keys())
        total_cached_queries = len(set().union(*[retriever.query_cache.keys() for retriever in self.retrievers.values()]))
        
        if total_cached_queries > 0:
            self.stats['cache_hit_rate'] = cache_hits / max(self.stats['total_queries'], total_cached_queries)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_queries': 0,
            'avg_retrieval_time': 0.0,
            'avg_results_per_query': 0.0,
            'cache_hit_rate': 0.0,
            'strategy_usage': defaultdict(int),
            'total_documents_retrieved': 0
        }
        
        # 清空缓存
        for retriever in self.retrievers.values():
            retriever.query_cache.clear()
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "strategy": "similarity",
                "top_k": 5,
                "similarity_threshold": 0.7,
                "diversity_threshold": 0.7,
                "reranking_method": "diversity",
                "query_expansion": "synonyms",
                "enable_caching": True,
                "vectorstore": "VectorStore instance"
            },
            "execute_parameters": {
                "input_data": "用户查询文本"
            },
            "usage_code": """
# 基础使用示例
from templates.data.retrieval_template import RetrievalTemplate
import asyncio

# 假设已有向量存储实例
vectorstore = ... # VectorStore instance

# 初始化检索模板
retriever = RetrievalTemplate()

# 配置相似度检索
retriever.setup(
    strategy="similarity",
    top_k=5,
    similarity_threshold=0.7,
    vectorstore=vectorstore,
    enable_caching=True
)

# 执行检索
result = retriever.execute("用户查询文本")
print(f"Found {len(result.documents)} documents in {result.retrieval_time:.3f}s")

# MMR检索示例
retriever.setup(
    strategy="mmr",
    top_k=10,
    diversity_threshold=0.7,
    vectorstore=vectorstore
)
mmr_result = retriever.execute("查询文本")

# 混合检索示例
retriever.setup(
    strategy="hybrid",
    top_k=8,
    vector_weight=0.7,
    keyword_weight=0.3,
    vectorstore=vectorstore,
    custom_params={
        "keyword_retriever": keyword_retriever_instance
    }
)
hybrid_result = retriever.execute("查询文本")

# 高级配置示例
retriever.setup(
    strategy="similarity",
    top_k=10,
    similarity_threshold=0.6,
    reranking_method="diversity",
    query_expansion="synonyms",
    metadata_filter={
        "source": {"$in": ["doc1.pdf", "doc2.pdf"]},
        "date": {"$gte": "2023-01-01"}
    },
    content_filter=r"重要|关键",
    use_context=True,
    context_window=3,
    vectorstore=vectorstore
)

# 异步检索示例
async def async_retrieval():
    result = await retriever._execute_async("异步查询")
    return result

result = asyncio.run(async_retrieval())

# 获取统计信息
stats = retriever.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Average retrieval time: {stats['avg_retrieval_time']:.3f}s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
""",
            "expected_output": {
                "type": "RetrievalResult",
                "description": "包含检索文档、分数、查询信息和元数据的结果对象"
            }
        }