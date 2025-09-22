"""
配置数据模型

定义各个模块的配置结构，使用Pydantic进行数据验证。
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class EmbeddingProvider(str, Enum):
    """嵌入模型提供商"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class VectorStoreType(str, Enum):
    """向量数据库类型"""
    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"
    ELASTICSEARCH = "elasticsearch"


class SplitterType(str, Enum):
    """文档分割器类型"""
    RECURSIVE_CHARACTER = "recursive_character"
    CHARACTER = "character"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


class DocumentProcessorConfig(BaseModel):
    """文档处理器配置"""
    
    # 支持的文件格式
    supported_formats: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md", ".html"],
        description="支持的文档格式"
    )
    
    # 分割器配置
    splitter_type: SplitterType = Field(
        default=SplitterType.RECURSIVE_CHARACTER,
        description="默认分割器类型"
    )
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="文档块大小"
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="文档块重叠大小"
    )
    
    # 元数据配置
    extract_metadata: bool = Field(
        default=True,
        description="是否提取元数据"
    )
    
    generate_summary: bool = Field(
        default=True,
        description="是否生成文档摘要"
    )
    
    # 并发配置
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="并发处理器数量"
    )
    
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="批处理大小"
    )

    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        """验证重叠不能超过块大小"""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class VectorStoreConfig(BaseModel):
    """向量存储配置"""
    
    # 嵌入模型配置
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="嵌入模型提供商"
    )
    
    embedding_model: str = Field(
        default="text-embedding-ada-002",
        description="嵌入模型名称"
    )
    
    embedding_dimension: int = Field(
        default=1536,
        ge=128,
        le=4096,
        description="嵌入向量维度"
    )
    
    # 向量数据库配置
    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.CHROMA,
        description="向量数据库类型"
    )
    
    collection_name: str = Field(
        default="rag_documents",
        description="集合名称"
    )
    
    persist_directory: str = Field(
        default="./data/vectorstore",
        description="持久化目录"
    )
    
    # 数据库连接配置
    database_url: Optional[str] = Field(
        default=None,
        description="数据库连接URL（用于Pinecone等云服务）"
    )
    
    api_key: Optional[str] = Field(
        default=None,
        description="API密钥"
    )
    
    # 索引配置
    index_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "metric": "cosine",
            "index_type": "IVF",
            "nlist": 1024
        },
        description="索引参数"
    )
    
    # 性能配置
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="批量插入大小"
    )


class RetrievalConfig(BaseModel):
    """检索配置"""
    
    # 基本检索参数
    default_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="默认检索文档数量"
    )
    
    max_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="最大检索文档数量"
    )
    
    # 混合检索配置
    enable_hybrid_search: bool = Field(
        default=True,
        description="启用混合检索"
    )
    
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="向量检索权重"
    )
    
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="关键词检索权重"
    )
    
    # 相似度阈值
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="相似度阈值"
    )
    
    # 重排序配置
    enable_reranking: bool = Field(
        default=True,
        description="启用重排序"
    )
    
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="重排序模型"
    )
    
    rerank_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="重排序候选数量"
    )
    
    # 多查询配置
    enable_multi_query: bool = Field(
        default=False,
        description="启用多查询检索"
    )
    
    query_expansion_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="查询扩展数量"
    )

    @validator('keyword_weight')
    def validate_weights(cls, v, values):
        """验证权重总和为1"""
        if 'vector_weight' in values:
            if abs(v + values['vector_weight'] - 1.0) > 0.001:
                raise ValueError("vector_weight + keyword_weight must equal 1.0")
        return v


class GenerationConfig(BaseModel):
    """生成配置"""
    
    # LLM配置
    llm_provider: str = Field(
        default="openai",
        description="LLM提供商"
    )
    
    llm_model: str = Field(
        default="gpt-3.5-turbo",
        description="LLM模型"
    )
    
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="生成温度"
    )
    
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="最大生成token数"
    )
    
    # 上下文配置
    max_context_length: int = Field(
        default=4000,
        ge=1000,
        le=16000,
        description="最大上下文长度"
    )
    
    context_compression: bool = Field(
        default=True,
        description="启用上下文压缩"
    )
    
    # Prompt配置
    system_prompt: str = Field(
        default="You are a helpful AI assistant that answers questions based on the provided context.",
        description="系统提示词"
    )
    
    prompt_template: str = Field(
        default="""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法找到相关信息。

上下文：
{context}

问题：{question}

回答：""",
        description="提示词模板"
    )
    
    # 引用配置
    include_sources: bool = Field(
        default=True,
        description="包含引用来源"
    )
    
    max_sources: int = Field(
        default=3,
        ge=1,
        le=10,
        description="最大引用来源数"
    )
    
    # 质量控制
    enable_fact_checking: bool = Field(
        default=False,
        description="启用事实检查"
    )
    
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )


class APIConfig(BaseModel):
    """API配置"""
    
    # 服务器配置
    host: str = Field(
        default="0.0.0.0",
        description="服务器地址"
    )
    
    port: int = Field(
        default=8000,
        ge=1000,
        le=65535,
        description="服务器端口"
    )
    
    # 安全配置
    enable_cors: bool = Field(
        default=True,
        description="启用CORS"
    )
    
    allowed_origins: List[str] = Field(
        default=["*"],
        description="允许的源"
    )
    
    api_key_required: bool = Field(
        default=False,
        description="需要API密钥"
    )
    
    api_keys: List[str] = Field(
        default_factory=list,
        description="有效的API密钥"
    )
    
    # 限流配置
    rate_limit_enabled: bool = Field(
        default=True,
        description="启用限流"
    )
    
    requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="每分钟请求限制"
    )
    
    # 文件上传配置
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        le=100 * 1024 * 1024,  # 100MB
        description="最大文件大小（字节）"
    )
    
    upload_path: str = Field(
        default="./data/uploads",
        description="文件上传路径"
    )
    
    # 监控配置
    enable_metrics: bool = Field(
        default=True,
        description="启用指标监控"
    )
    
    metrics_port: int = Field(
        default=9090,
        ge=1000,
        le=65535,
        description="指标监控端口"
    )
    
    # 日志配置
    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    
    log_file: Optional[str] = Field(
        default=None,
        description="日志文件路径"
    )


class DatabaseConfig(BaseModel):
    """数据库配置"""
    
    # PostgreSQL配置
    postgres_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL连接URL"
    )
    
    # Redis配置
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis连接URL"
    )
    
    # 连接池配置
    pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="连接池大小"
    )
    
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=50,
        description="最大溢出连接数"
    )
    
    # 缓存配置
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="缓存TTL（秒）"
    )


class MonitoringConfig(BaseModel):
    """监控配置"""
    
    # Prometheus配置
    enable_prometheus: bool = Field(
        default=True,
        description="启用Prometheus监控"
    )
    
    metrics_path: str = Field(
        default="/metrics",
        description="指标路径"
    )
    
    # 健康检查配置
    health_check_interval: int = Field(
        default=30,
        ge=10,
        le=300,
        description="健康检查间隔（秒）"
    )
    
    # 性能监控
    track_performance: bool = Field(
        default=True,
        description="跟踪性能指标"
    )
    
    slow_query_threshold: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="慢查询阈值（秒）"
    )