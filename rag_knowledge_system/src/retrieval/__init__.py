"""
检索模块

提供多种检索策略的实现，包括向量检索、关键词检索、混合检索等。
支持重排序、多查询和上下文管理功能。
"""

from .retrievers import (
    BaseRetriever,
    VectorRetriever,
    KeywordRetriever,
    HybridRetriever,
    MultiQueryRetriever
)
from .rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    ScoreReranker,
    DiversityReranker
)
from .fusion import (
    RetrievalFusion,
    RankFusion,
    ScoreFusion
)
from .filters import (
    BaseFilter,
    MetadataFilter,
    SimilarityFilter,
    TimeFilter
)

__all__ = [
    # 检索器
    "BaseRetriever",
    "VectorRetriever", 
    "KeywordRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    
    # 重排序器
    "BaseReranker",
    "CrossEncoderReranker",
    "ScoreReranker", 
    "DiversityReranker",
    
    # 融合策略
    "RetrievalFusion",
    "RankFusion",
    "ScoreFusion",
    
    # 过滤器
    "BaseFilter",
    "MetadataFilter",
    "SimilarityFilter",
    "TimeFilter"
]