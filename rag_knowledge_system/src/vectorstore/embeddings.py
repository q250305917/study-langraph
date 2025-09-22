"""
嵌入模型管理模块

提供多种嵌入模型的统一接口，支持OpenAI、HuggingFace、SentenceTransformers等。
包含模型缓存、批量处理和性能优化功能。
"""

import os
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# LangChain导入
from langchain.embeddings.base import Embeddings

# 可选依赖
try:
    from langchain.embeddings import OpenAIEmbeddings
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    
    # 基本配置
    provider: str = "openai"
    model_name: str = "text-embedding-ada-002"
    dimension: int = 1536
    
    # API配置
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    
    # 性能配置
    batch_size: int = 100
    max_retries: int = 3
    request_timeout: float = 60.0
    
    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "./cache/embeddings"
    cache_size: int = 10000
    
    # HuggingFace特定配置
    device: str = "cpu"
    normalize_embeddings: bool = True
    
    # Token限制
    max_token_length: int = 8192


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    embeddings: List[List[float]]
    texts: List[str]
    model_name: str
    dimension: int
    processing_time: float
    token_count: int = 0
    cached_count: int = 0


class BaseEmbeddingModel(ABC):
    """嵌入模型基类"""
    
    def __init__(self, config: EmbeddingConfig):
        """初始化嵌入模型"""
        self.config = config
        self.cache = {} if config.enable_cache else None
        self._setup_cache_dir()
    
    def _setup_cache_dir(self):
        """设置缓存目录"""
        if self.config.enable_cache:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """嵌入文本（子类实现）"""
        pass
    
    def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """
        嵌入文档列表
        
        Args:
            texts: 文本列表
            
        Returns:
            EmbeddingResult对象
        """
        start_time = time.time()
        
        # 预处理文本
        processed_texts = self._preprocess_texts(texts)
        
        # 检查缓存
        embeddings, cache_hits = self._get_cached_embeddings(processed_texts)
        
        # 计算未缓存的嵌入
        uncached_indices = [i for i, emb in enumerate(embeddings) if emb is None]
        
        if uncached_indices:
            uncached_texts = [processed_texts[i] for i in uncached_indices]
            uncached_embeddings = self._embed_texts(uncached_texts)
            
            # 更新结果和缓存
            for i, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings[i] = embedding
                if self.cache is not None:
                    cache_key = self._get_cache_key(processed_texts[i])
                    self.cache[cache_key] = embedding
        
        # 计算token数量
        token_count = self._count_tokens(processed_texts)
        
        # 处理时间
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=processed_texts,
            model_name=self.config.model_name,
            dimension=self.config.dimension,
            processing_time=processing_time,
            token_count=token_count,
            cached_count=cache_hits
        )
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        result = self.embed_documents([text])
        return result.embeddings[0]
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """预处理文本"""
        processed = []
        
        for text in texts:
            # 清理文本
            text = text.strip()
            
            # 截断过长文本
            if self._count_tokens([text])[0] > self.config.max_token_length:
                text = self._truncate_text(text)
            
            processed.append(text)
        
        return processed
    
    def _truncate_text(self, text: str) -> str:
        """截断文本到token限制"""
        if not HAS_TIKTOKEN:
            # 简单截断：假设平均4个字符为1个token
            max_chars = self.config.max_token_length * 4
            return text[:max_chars]
        
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            if len(tokens) > self.config.max_token_length:
                truncated_tokens = tokens[:self.config.max_token_length]
                return encoding.decode(truncated_tokens)
        
        except Exception as e:
            logger.warning(f"Token截断失败: {e}")
            # 备用截断
            max_chars = self.config.max_token_length * 4
            return text[:max_chars]
        
        return text
    
    def _count_tokens(self, texts: List[str]) -> int:
        """计算token数量"""
        if not HAS_TIKTOKEN:
            # 简单估算
            return sum(len(text.split()) for text in texts)
        
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return sum(len(encoding.encode(text)) for text in texts)
        except:
            return sum(len(text.split()) for text in texts)
    
    def _get_cached_embeddings(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], int]:
        """获取缓存的嵌入"""
        if not self.cache:
            return [None] * len(texts), 0
        
        embeddings = []
        cache_hits = 0
        
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
                cache_hits += 1
            else:
                embeddings.append(None)
        
        return embeddings, cache_hits
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        content = f"{self.config.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("嵌入缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if not self.cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "max_size": self.config.cache_size
        }


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI嵌入模型"""
    
    def __init__(self, config: EmbeddingConfig):
        """初始化OpenAI嵌入模型"""
        if not HAS_OPENAI:
            raise ImportError("需要安装OpenAI: pip install openai langchain-openai")
        
        super().__init__(config)
        
        # 初始化OpenAI客户端
        self.client = OpenAIEmbeddings(
            model=config.model_name,
            openai_api_key=config.api_key,
            openai_api_base=config.api_base,
            max_retries=config.max_retries,
            request_timeout=config.request_timeout
        )
        
        logger.info(f"初始化OpenAI嵌入模型: {config.model_name}")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """使用OpenAI API嵌入文本"""
        try:
            # 批量处理
            all_embeddings = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                
                # 调用API
                batch_embeddings = self.client.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # 简单的速率限制
                if len(texts) > self.config.batch_size:
                    time.sleep(0.1)
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"OpenAI嵌入失败: {e}")
            raise


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    """HuggingFace嵌入模型"""
    
    def __init__(self, config: EmbeddingConfig):
        """初始化HuggingFace嵌入模型"""
        if not HAS_HUGGINGFACE:
            raise ImportError("需要安装HuggingFace: pip install sentence-transformers")
        
        super().__init__(config)
        
        # 初始化模型
        model_kwargs = {
            'device': config.device
        }
        
        encode_kwargs = {
            'normalize_embeddings': config.normalize_embeddings
        }
        
        self.client = HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        logger.info(f"初始化HuggingFace嵌入模型: {config.model_name}")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """使用HuggingFace模型嵌入文本"""
        try:
            return self.client.embed_documents(texts)
        
        except Exception as e:
            logger.error(f"HuggingFace嵌入失败: {e}")
            raise


class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    """SentenceTransformers嵌入模型"""
    
    def __init__(self, config: EmbeddingConfig):
        """初始化SentenceTransformers嵌入模型"""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("需要安装SentenceTransformers: pip install sentence-transformers")
        
        super().__init__(config)
        
        # 加载模型
        self.model = SentenceTransformer(
            config.model_name,
            device=config.device
        )
        
        # 获取实际维度
        self.config.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"初始化SentenceTransformers模型: {config.model_name}, 维度: {self.config.dimension}")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """使用SentenceTransformers嵌入文本"""
        try:
            # 批量编码
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(texts) > 100
            )
            
            # 转换为列表格式
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"SentenceTransformers嵌入失败: {e}")
            raise


class EmbeddingManager:
    """嵌入管理器"""
    
    def __init__(self, config: EmbeddingConfig):
        """初始化嵌入管理器"""
        self.config = config
        self.model = self._create_model()
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "total_texts": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "errors": 0
        }
    
    def _create_model(self) -> BaseEmbeddingModel:
        """创建嵌入模型"""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            return OpenAIEmbeddingModel(self.config)
        elif provider == "huggingface":
            return HuggingFaceEmbeddingModel(self.config)
        elif provider == "sentence_transformers":
            return SentenceTransformersEmbeddingModel(self.config)
        else:
            raise ValueError(f"不支持的嵌入提供商: {provider}")
    
    def embed_documents(self, texts: List[str]) -> EmbeddingResult:
        """嵌入文档"""
        try:
            result = self.model.embed_documents(texts)
            
            # 更新统计
            self._update_stats(result)
            
            return result
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"文档嵌入失败: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        try:
            embedding = self.model.embed_query(text)
            
            # 更新统计
            self.stats["total_requests"] += 1
            self.stats["total_texts"] += 1
            
            return embedding
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"查询嵌入失败: {e}")
            raise
    
    def _update_stats(self, result: EmbeddingResult):
        """更新统计信息"""
        self.stats["total_requests"] += 1
        self.stats["total_texts"] += len(result.texts)
        self.stats["total_tokens"] += result.token_count
        self.stats["total_time"] += result.processing_time
        self.stats["cache_hits"] += result.cached_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算平均值
        if stats["total_requests"] > 0:
            stats["avg_time_per_request"] = stats["total_time"] / stats["total_requests"]
            stats["avg_texts_per_request"] = stats["total_texts"] / stats["total_requests"]
        
        if stats["total_texts"] > 0:
            stats["avg_tokens_per_text"] = stats["total_tokens"] / stats["total_texts"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_texts"]
        
        return stats
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            "total_requests": 0,
            "total_texts": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "errors": 0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "provider": self.config.provider,
            "model_name": self.config.model_name,
            "dimension": self.config.dimension,
            "batch_size": self.config.batch_size,
            "cache_enabled": self.config.enable_cache,
            "cache_stats": self.model.get_cache_stats()
        }


# 工厂函数
def get_embedding_model(provider: str,
                       model_name: str,
                       **kwargs) -> EmbeddingManager:
    """
    创建嵌入模型的工厂函数
    
    Args:
        provider: 提供商名称
        model_name: 模型名称
        **kwargs: 额外配置参数
        
    Returns:
        EmbeddingManager实例
    """
    # 预设配置
    preset_configs = {
        "openai": {
            "text-embedding-ada-002": {"dimension": 1536},
            "text-embedding-3-small": {"dimension": 1536},
            "text-embedding-3-large": {"dimension": 3072}
        },
        "sentence_transformers": {
            "all-MiniLM-L6-v2": {"dimension": 384},
            "all-mpnet-base-v2": {"dimension": 768},
            "multi-qa-mpnet-base-dot-v1": {"dimension": 768}
        }
    }
    
    # 应用预设配置
    config_params = kwargs.copy()
    if provider in preset_configs and model_name in preset_configs[provider]:
        preset = preset_configs[provider][model_name]
        for key, value in preset.items():
            if key not in config_params:
                config_params[key] = value
    
    # 创建配置
    config = EmbeddingConfig(
        provider=provider,
        model_name=model_name,
        **config_params
    )
    
    return EmbeddingManager(config)


# 便捷函数
def embed_texts(texts: List[str],
               provider: str = "openai",
               model_name: str = "text-embedding-ada-002",
               **kwargs) -> List[List[float]]:
    """
    嵌入文本列表的便捷函数
    
    Args:
        texts: 文本列表
        provider: 提供商
        model_name: 模型名称
        **kwargs: 额外参数
        
    Returns:
        嵌入向量列表
    """
    manager = get_embedding_model(provider, model_name, **kwargs)
    result = manager.embed_documents(texts)
    return result.embeddings


def embed_query(text: str,
               provider: str = "openai", 
               model_name: str = "text-embedding-ada-002",
               **kwargs) -> List[float]:
    """
    嵌入查询文本的便捷函数
    
    Args:
        text: 查询文本
        provider: 提供商
        model_name: 模型名称
        **kwargs: 额外参数
        
    Returns:
        嵌入向量
    """
    manager = get_embedding_model(provider, model_name, **kwargs)
    return manager.embed_query(text)