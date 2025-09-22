"""
生成模块

提供基于检索结果的智能回答生成功能，包括提示词管理、
上下文压缩、回答质量评估等功能。
"""

from .chains import RAGChain, ConversationalRAGChain
from .prompts import PromptManager, PromptTemplate, create_rag_prompt
from .context import ContextManager, ContextCompressor
from .evaluation import AnswerEvaluator, QualityMetrics

__all__ = [
    # 生成链
    "RAGChain",
    "ConversationalRAGChain",
    
    # 提示词管理
    "PromptManager",
    "PromptTemplate", 
    "create_rag_prompt",
    
    # 上下文管理
    "ContextManager",
    "ContextCompressor",
    
    # 评估
    "AnswerEvaluator",
    "QualityMetrics"
]