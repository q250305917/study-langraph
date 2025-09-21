"""
数据处理模板模块

本模块提供了完整的数据处理模板集合，涵盖文档加载、文本分割、向量化存储和检索等核心功能。
这些模板是构建RAG（检索增强生成）应用的基础组件。

核心组件：
1. DocumentLoaderTemplate: 多格式文档加载器
   - 支持PDF、Word、TXT、Markdown、HTML等多种格式
   - 统一的文档加载接口
   - 元数据提取和管理
   - 批量处理和异步加载

2. TextSplitterTemplate: 智能文本分割器
   - 多种分割策略：按字符、句子、段落、语义
   - 保持上下文连贯性
   - 自适应分割大小
   - 重叠窗口支持

3. VectorStoreTemplate: 向量数据库模板
   - 支持多种向量数据库：Chroma、Pinecone、FAISS、Qdrant等
   - 统一的向量操作接口
   - 批量向量化和存储
   - 索引管理和优化

4. RetrievalTemplate: 检索算法模板
   - 多种检索策略：相似度搜索、混合检索、重排序
   - 可配置的检索参数
   - 结果过滤和后处理
   - 性能监控和优化

使用示例：
```python
from templates.data import (
    DocumentLoaderTemplate, 
    TextSplitterTemplate,
    VectorStoreTemplate,
    RetrievalTemplate
)

# 文档加载和处理流水线
loader = DocumentLoaderTemplate()
loader.setup(file_types=['pdf', 'docx', 'txt'])

splitter = TextSplitterTemplate()
splitter.setup(strategy='semantic', chunk_size=1000, overlap=200)

vectorstore = VectorStoreTemplate()
vectorstore.setup(provider='chroma', embedding_model='openai')

retriever = RetrievalTemplate()
retriever.setup(strategy='hybrid', top_k=5)

# 处理文档
documents = loader.execute(file_path='./documents/')
chunks = splitter.execute(documents)
vectorstore.execute(chunks)
results = retriever.execute(query='用户查询')
```

设计原则：
- 模块化：每个组件专注于特定功能，可独立使用
- 可配置：通过参数配置适应不同场景需求
- 可扩展：易于添加新的格式、策略和提供商支持
- 高性能：支持批量处理和异步操作
- 统一接口：继承自TemplateBase，提供一致的使用体验
"""

from .document_loader import DocumentLoaderTemplate
from .text_splitter import TextSplitterTemplate  
from .vectorstore_template import VectorStoreTemplate
from .retrieval_template import RetrievalTemplate

__all__ = [
    'DocumentLoaderTemplate',
    'TextSplitterTemplate', 
    'VectorStoreTemplate',
    'RetrievalTemplate'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'LangChain Learning Team'

# 模块级配置
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# 支持的文件格式
SUPPORTED_DOCUMENT_FORMATS = [
    'pdf', 'docx', 'doc', 'txt', 'md', 'html', 'htm',
    'rtf', 'odt', 'epub', 'csv', 'json', 'xml'
]

# 支持的向量数据库
SUPPORTED_VECTOR_STORES = [
    'chroma', 'pinecone', 'faiss', 'qdrant', 'weaviate', 
    'milvus', 'elasticsearch', 'opensearch'
]

# 支持的文本分割策略
SUPPORTED_SPLIT_STRATEGIES = [
    'character', 'sentence', 'paragraph', 'semantic',
    'recursive_character', 'token_based', 'markdown_header'
]

# 支持的检索策略
SUPPORTED_RETRIEVAL_STRATEGIES = [
    'similarity', 'mmr', 'hybrid', 'rerank', 'ensemble'
]