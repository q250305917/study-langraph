"""
RAG系统主配置

整合各个模块的配置，提供统一的配置管理接口。
"""

import os
from typing import Optional, Dict, Any
from functools import lru_cache
from pydantic import BaseSettings, Field

from .models import (
    DocumentProcessorConfig,
    VectorStoreConfig,
    RetrievalConfig,
    GenerationConfig,
    APIConfig,
    DatabaseConfig,
    MonitoringConfig
)


class RAGConfig(BaseSettings):
    """RAG系统完整配置"""
    
    # 基本信息
    app_name: str = Field(
        default="RAG Knowledge System",
        description="应用名称"
    )
    
    version: str = Field(
        default="0.1.0",
        description="应用版本"
    )
    
    environment: str = Field(
        default="development",
        description="运行环境"
    )
    
    debug: bool = Field(
        default=True,
        description="调试模式"
    )
    
    # 数据目录配置
    data_dir: str = Field(
        default="./data",
        description="数据根目录"
    )
    
    documents_dir: str = Field(
        default="./data/documents",
        description="文档存储目录"
    )
    
    processed_dir: str = Field(
        default="./data/processed",
        description="处理后文档目录"
    )
    
    vectorstore_dir: str = Field(
        default="./data/vectorstore",
        description="向量存储目录"
    )
    
    logs_dir: str = Field(
        default="./logs",
        description="日志目录"
    )
    
    # API密钥配置
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API密钥"
    )
    
    huggingface_api_key: Optional[str] = Field(
        default=None,
        description="HuggingFace API密钥"
    )
    
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API密钥"
    )
    
    elasticsearch_url: Optional[str] = Field(
        default=None,
        description="Elasticsearch URL"
    )
    
    # 子模块配置
    document_processor: DocumentProcessorConfig = Field(
        default_factory=DocumentProcessorConfig,
        description="文档处理器配置"
    )
    
    vectorstore: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig,
        description="向量存储配置"
    )
    
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="检索配置"
    )
    
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="生成配置"
    )
    
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API配置"
    )
    
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="数据库配置"
    )
    
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="监控配置"
    )
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        env_prefix = "RAG_"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        """初始化配置"""
        super().__init__(**kwargs)
        self._ensure_directories()
        self._validate_api_keys()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.data_dir,
            self.documents_dir,
            self.processed_dir,
            self.vectorstore_dir,
            self.logs_dir,
            self.api.upload_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_api_keys(self):
        """验证API密钥配置"""
        warnings = []
        
        # 检查必需的API密钥
        if not self.openai_api_key and self.vectorstore.embedding_provider.value == "openai":
            warnings.append("OpenAI API key is required for OpenAI embeddings")
        
        if not self.openai_api_key and self.generation.llm_provider == "openai":
            warnings.append("OpenAI API key is required for OpenAI LLM")
        
        if not self.pinecone_api_key and self.vectorstore.vector_store_type.value == "pinecone":
            warnings.append("Pinecone API key is required for Pinecone vector store")
        
        if warnings and not self.debug:
            raise ValueError(f"Missing required API keys: {'; '.join(warnings)}")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置字典"""
        config = {
            "model_name": self.generation.llm_model,
            "temperature": self.generation.temperature,
            "max_tokens": self.generation.max_tokens,
        }
        
        if self.generation.llm_provider == "openai":
            config["openai_api_key"] = self.openai_api_key
        
        return config
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取嵌入模型配置字典"""
        config = {
            "model": self.vectorstore.embedding_model,
            "dimension": self.vectorstore.embedding_dimension,
        }
        
        if self.vectorstore.embedding_provider.value == "openai":
            config["openai_api_key"] = self.openai_api_key
        elif self.vectorstore.embedding_provider.value == "huggingface":
            config["api_key"] = self.huggingface_api_key
        
        return config
    
    def get_vectorstore_config(self) -> Dict[str, Any]:
        """获取向量存储配置字典"""
        config = {
            "collection_name": self.vectorstore.collection_name,
            "persist_directory": self.vectorstore.persist_directory,
        }
        
        if self.vectorstore.vector_store_type.value == "pinecone":
            config.update({
                "api_key": self.pinecone_api_key,
                "index_name": self.vectorstore.collection_name,
            })
        elif self.vectorstore.vector_store_type.value == "elasticsearch":
            config["url"] = self.elasticsearch_url
        
        return config
    
    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        if self.database.postgres_url:
            return self.database.postgres_url
        
        # 默认SQLite数据库
        return f"sqlite:///{self.data_dir}/rag_system.db"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.dict(exclude={"openai_api_key", "huggingface_api_key", "pinecone_api_key"})
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'RAGConfig':
        """从文件加载配置"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


# 全局配置实例
_config_cache: Optional[RAGConfig] = None


@lru_cache()
def get_settings() -> RAGConfig:
    """获取全局配置实例（单例模式）"""
    global _config_cache
    if _config_cache is None:
        _config_cache = RAGConfig()
    return _config_cache


def update_settings(new_config: RAGConfig):
    """更新全局配置"""
    global _config_cache
    _config_cache = new_config
    # 清除缓存
    get_settings.cache_clear()


def get_config_value(key: str, default=None):
    """获取指定配置值"""
    config = get_settings()
    keys = key.split('.')
    
    value = config
    for k in keys:
        if hasattr(value, k):
            value = getattr(value, k)
        else:
            return default
    
    return value


# 配置验证器
class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_embedding_compatibility(config: RAGConfig) -> bool:
        """验证嵌入模型兼容性"""
        provider = config.vectorstore.embedding_provider
        model = config.vectorstore.embedding_model
        
        # OpenAI嵌入模型验证
        if provider.value == "openai":
            valid_models = [
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]
            return model in valid_models
        
        return True
    
    @staticmethod
    def validate_vectorstore_compatibility(config: RAGConfig) -> bool:
        """验证向量存储兼容性"""
        store_type = config.vectorstore.vector_store_type
        
        # 检查必需的依赖
        required_packages = {
            "chroma": ["chromadb"],
            "faiss": ["faiss-cpu"],
            "pinecone": ["pinecone-client"],
            "elasticsearch": ["elasticsearch"]
        }
        
        if store_type.value in required_packages:
            try:
                for package in required_packages[store_type.value]:
                    __import__(package.replace("-", "_"))
                return True
            except ImportError:
                return False
        
        return True
    
    @staticmethod
    def validate_all(config: RAGConfig) -> Dict[str, bool]:
        """验证所有配置"""
        return {
            "embedding_compatibility": ConfigValidator.validate_embedding_compatibility(config),
            "vectorstore_compatibility": ConfigValidator.validate_vectorstore_compatibility(config),
        }


# 环境特定配置
def get_development_config() -> RAGConfig:
    """获取开发环境配置"""
    return RAGConfig(
        environment="development",
        debug=True,
        api=APIConfig(
            enable_cors=True,
            api_key_required=False,
            rate_limit_enabled=False
        ),
        vectorstore=VectorStoreConfig(
            vector_store_type="chroma",
            persist_directory="./data/dev_vectorstore"
        )
    )


def get_production_config() -> RAGConfig:
    """获取生产环境配置"""
    return RAGConfig(
        environment="production",
        debug=False,
        api=APIConfig(
            api_key_required=True,
            rate_limit_enabled=True,
            requests_per_minute=100
        ),
        monitoring=MonitoringConfig(
            enable_prometheus=True,
            track_performance=True
        )
    )


def get_test_config() -> RAGConfig:
    """获取测试环境配置"""
    return RAGConfig(
        environment="test",
        debug=True,
        data_dir="./test_data",
        vectorstore=VectorStoreConfig(
            vector_store_type="chroma",
            persist_directory="./test_data/vectorstore"
        ),
        database=DatabaseConfig(
            postgres_url="sqlite:///test.db"
        )
    )