"""
RAG知识库系统

这是一个基于LangChain框架构建的完整的RAG（检索增强生成）知识库系统。
该系统实现了文档摄取、向量化存储、智能检索和回答生成的完整流程。

主要功能模块：
- 文档处理：支持多种格式文档的加载、分割和元数据提取
- 向量存储：集成多种嵌入模型和向量数据库
- 智能检索：混合检索策略，支持向量和关键词检索
- 生成模块：基于检索结果的智能回答生成
- API接口：完整的RESTful API和管理界面

技术栈：
- LangChain: 主要框架
- FastAPI: Web框架  
- Chroma/FAISS: 向量数据库
- OpenAI/HuggingFace: 嵌入和生成模型
"""

__version__ = "0.1.0"
__author__ = "LangChain Learning Team"

# 核心模块导入
from .src.document_processor import DocumentProcessor
from .src.vectorstore import VectorStoreManager
from .src.retrieval import HybridRetriever
from .src.generation import RAGChain

# 配置管理
from .config import RAGConfig

__all__ = [
    "DocumentProcessor",
    "VectorStoreManager", 
    "HybridRetriever",
    "RAGChain",
    "RAGConfig"
]