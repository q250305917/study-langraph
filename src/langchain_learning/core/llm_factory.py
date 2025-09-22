"""
LangChain学习项目的LLM工厂模块

本模块实现了LLM（大语言模型）实例的统一创建和管理，支持：
- 多厂商LLM支持（OpenAI、Anthropic、Google等）
- 模型配置管理和验证
- 连接池和缓存机制
- 性能监控和统计
- 自动重试和错误处理
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, Protocol
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path

from pydantic import BaseModel, Field, validator
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

from .logger import get_logger
from .config import get_config_value
from .exceptions import (
    LLMError,
    ConfigurationError,
    ValidationError,
    TimeoutError,
    ErrorCodes,
    retry_on_exception
)

logger = get_logger(__name__)


class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """模型能力枚举"""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    EMBEDDING = "embedding"
    FINE_TUNING = "fine_tuning"


@dataclass
class ModelInfo:
    """
    模型信息
    
    描述模型的基本信息、能力和限制。
    """
    name: str
    provider: LLMProvider
    max_tokens: int
    capabilities: List[ModelCapability] = field(default_factory=list)
    context_window: int = 4096
    cost_per_token: float = 0.0
    supports_streaming: bool = True
    supports_async: bool = True
    version: str = "1.0"
    description: str = ""
    
    def has_capability(self, capability: ModelCapability) -> bool:
        """检查模型是否具有指定能力"""
        return capability in self.capabilities


class LLMConfig(BaseModel):
    """
    LLM配置模型
    
    定义创建LLM实例所需的配置参数。
    """
    
    # 基本配置
    provider: LLMProvider = Field(description="LLM提供商")
    model_name: str = Field(description="模型名称")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    api_base: Optional[str] = Field(default=None, description="API基础URL")
    
    # 生成参数
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="最大令牌数")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="核采样参数")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    
    # 高级配置
    timeout: float = Field(default=60.0, gt=0, description="请求超时时间")
    max_retries: int = Field(default=3, ge=0, description="最大重试次数")
    streaming: bool = Field(default=False, description="是否启用流式响应")
    
    # 特定提供商配置
    extra_config: Dict[str, Any] = Field(default_factory=dict, description="额外配置")
    
    class Config:
        extra = "allow"
    
    @validator('provider')
    def validate_provider(cls, v):
        """验证提供商"""
        if isinstance(v, str):
            try:
                return LLMProvider(v.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {v}")
        return v


@dataclass
class LLMMetrics:
    """LLM性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_request_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def update_request(self, success: bool, tokens: int, cost: float, response_time: float) -> None:
        """更新请求统计"""
        self.total_requests += 1
        self.last_request_time = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens_used += tokens
        self.total_cost += cost
        
        # 更新平均响应时间（指数移动平均）
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            alpha = 0.1
            self.average_response_time = (
                alpha * response_time + (1 - alpha) * self.average_response_time
            )


class LLMInstance:
    """
    LLM实例包装器
    
    包装LangChain的LLM实例，提供统一的接口和附加功能。
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        config: LLMConfig,
        model_info: ModelInfo
    ):
        """
        初始化LLM实例
        
        Args:
            llm: LangChain LLM实例
            config: LLM配置
            model_info: 模型信息
        """
        self.llm = llm
        self.config = config
        self.model_info = model_info
        self.metrics = LLMMetrics()
        self.created_at = time.time()
        
        logger.debug(f"Created LLM instance: {config.provider.value}/{config.model_name}")
    
    @retry_on_exception(max_retries=3, exceptions=(LLMError,))
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            **kwargs: 额外参数
            
        Returns:
            生成的文本
            
        Raises:
            LLMError: 生成失败
            TimeoutError: 请求超时
        """
        start_time = time.time()
        
        try:
            # 合并配置参数
            generation_kwargs = {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
            
            # 执行生成（支持超时）
            if hasattr(self.llm, 'ainvoke'):
                # 异步调用
                response = await asyncio.wait_for(
                    self.llm.ainvoke(prompt, **generation_kwargs),
                    timeout=self.config.timeout
                )
            else:
                # 同步调用
                import asyncio
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.llm.invoke, prompt, **generation_kwargs),
                    timeout=self.config.timeout
                )
            
            # 提取文本内容
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            # 更新统计信息
            response_time = time.time() - start_time
            tokens_used = self._estimate_tokens(prompt + result)
            cost = self._calculate_cost(tokens_used)
            
            self.metrics.update_request(True, tokens_used, cost, response_time)
            
            logger.debug(
                f"Generated response: {len(result)} chars, "
                f"{tokens_used} tokens, {response_time:.2f}s"
            )
            
            return result
            
        except asyncio.TimeoutError:
            # 超时处理
            response_time = time.time() - start_time
            self.metrics.update_request(False, 0, 0, response_time)
            
            raise TimeoutError(
                f"LLM request timeout after {self.config.timeout}s",
                error_code=ErrorCodes.API_TIMEOUT,
                context={
                    "provider": self.config.provider.value,
                    "model": self.config.model_name,
                    "timeout": self.config.timeout
                }
            )
            
        except Exception as e:
            # 其他错误处理
            response_time = time.time() - start_time
            self.metrics.update_request(False, 0, 0, response_time)
            
            raise LLMError(
                f"LLM generation failed: {str(e)}",
                error_code=ErrorCodes.LLM_API_ERROR,
                context={
                    "provider": self.config.provider.value,
                    "model": self.config.model_name
                },
                cause=e
            )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        聊天对话
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}, ...]
            **kwargs: 额外参数
            
        Returns:
            回复消息
        """
        # 将消息转换为提示文本（简化实现）
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nassistant:"
        
        return await self.generate(prompt, **kwargs)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的令牌数量
        
        使用简单的启发式方法估算，实际项目中应使用tokenizer。
        """
        # 粗略估算：英文约4个字符=1个token，中文约1.5个字符=1个token
        import re
        
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计其他字符
        other_chars = len(text) - chinese_chars
        
        # 估算令牌数
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        
        return max(estimated_tokens, 1)
    
    def _calculate_cost(self, tokens: int) -> float:
        """计算使用成本"""
        return tokens * self.model_info.cost_per_token
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "total_requests": self.metrics.total_requests,
            "success_rate": self.metrics.success_rate,
            "total_tokens": self.metrics.total_tokens_used,
            "total_cost": self.metrics.total_cost,
            "average_response_time": self.metrics.average_response_time,
            "uptime": time.time() - self.created_at
        }


class LLMFactory:
    """
    LLM工厂
    
    负责创建和管理LLM实例，提供统一的创建接口。
    """
    
    def __init__(self):
        """初始化LLM工厂"""
        self._model_registry: Dict[str, ModelInfo] = {}
        self._instance_cache: Dict[str, LLMInstance] = {}
        self._providers: Dict[LLMProvider, Callable] = {}
        
        # 注册内置提供商
        self._register_builtin_providers()
        
        # 注册内置模型
        self._register_builtin_models()
        
        logger.info("LLM Factory initialized")
    
    def register_model(self, model_info: ModelInfo) -> None:
        """
        注册模型信息
        
        Args:
            model_info: 模型信息
        """
        key = f"{model_info.provider.value}/{model_info.name}"
        self._model_registry[key] = model_info
        logger.debug(f"Registered model: {key}")
    
    def register_provider(self, provider: LLMProvider, factory_func: Callable) -> None:
        """
        注册LLM提供商
        
        Args:
            provider: 提供商类型
            factory_func: 创建LLM实例的工厂函数
        """
        self._providers[provider] = factory_func
        logger.debug(f"Registered provider: {provider.value}")
    
    def create_llm(
        self,
        config: Union[LLMConfig, Dict[str, Any]],
        use_cache: bool = True
    ) -> LLMInstance:
        """
        创建LLM实例
        
        Args:
            config: LLM配置
            use_cache: 是否使用缓存
            
        Returns:
            LLM实例
            
        Raises:
            LLMError: 创建失败
            ConfigurationError: 配置错误
        """
        # 标准化配置
        if isinstance(config, dict):
            llm_config = LLMConfig(**config)
        else:
            llm_config = config
        
        # 生成缓存键
        cache_key = self._generate_cache_key(llm_config)
        
        # 检查缓存
        if use_cache and cache_key in self._instance_cache:
            logger.debug(f"Using cached LLM instance: {cache_key}")
            return self._instance_cache[cache_key]
        
        try:
            # 获取模型信息
            model_info = self._get_model_info(llm_config)
            
            # 验证配置
            self._validate_config(llm_config, model_info)
            
            # 创建LLM实例
            llm = self._create_llm_instance(llm_config)
            
            # 包装为LLMInstance
            instance = LLMInstance(llm, llm_config, model_info)
            
            # 缓存实例
            if use_cache:
                self._instance_cache[cache_key] = instance
            
            logger.info(f"Created LLM instance: {llm_config.provider.value}/{llm_config.model_name}")
            return instance
            
        except Exception as e:
            if isinstance(e, (LLMError, ConfigurationError)):
                raise
            raise LLMError(
                f"Failed to create LLM instance: {str(e)}",
                error_code=ErrorCodes.LLM_MODEL_ERROR,
                context={
                    "provider": llm_config.provider.value,
                    "model": llm_config.model_name
                },
                cause=e
            )
    
    def create_llm_from_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMInstance:
        """
        从配置文件创建LLM实例
        
        Args:
            provider: 提供商名称，None则从配置读取
            model: 模型名称，None则从配置读取
            **kwargs: 额外配置参数
            
        Returns:
            LLM实例
        """
        # 从配置文件读取默认值
        config_dict = {
            "provider": provider or get_config_value("default_llm_provider", "openai"),
            "model_name": model or get_config_value("llm_model", "gpt-3.5-turbo"),
            "api_key": get_config_value("llm_api_key"),
            "temperature": get_config_value("llm_temperature", 0.7),
            "max_tokens": get_config_value("llm_max_tokens", 1000),
            **kwargs
        }
        
        return self.create_llm(config_dict)
    
    def list_models(self, provider: Optional[LLMProvider] = None) -> List[ModelInfo]:
        """
        列出可用模型
        
        Args:
            provider: 提供商筛选，None则返回所有模型
            
        Returns:
            模型信息列表
        """
        models = list(self._model_registry.values())
        
        if provider:
            models = [model for model in models if model.provider == provider]
        
        return models
    
    def get_model_info(self, provider: str, model_name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        key = f"{provider}/{model_name}"
        return self._model_registry.get(key)
    
    def get_cached_instances(self) -> List[LLMInstance]:
        """获取所有缓存的实例"""
        return list(self._instance_cache.values())
    
    def clear_cache(self) -> None:
        """清空实例缓存"""
        self._instance_cache.clear()
        logger.info("Cleared LLM instance cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取工厂统计信息"""
        total_requests = sum(
            instance.metrics.total_requests 
            for instance in self._instance_cache.values()
        )
        
        total_tokens = sum(
            instance.metrics.total_tokens_used 
            for instance in self._instance_cache.values()
        )
        
        total_cost = sum(
            instance.metrics.total_cost 
            for instance in self._instance_cache.values()
        )
        
        return {
            "registered_models": len(self._model_registry),
            "cached_instances": len(self._instance_cache),
            "supported_providers": [p.value for p in self._providers.keys()],
            "total_requests": total_requests,
            "total_tokens_used": total_tokens,
            "total_cost": total_cost
        }
    
    def _generate_cache_key(self, config: LLMConfig) -> str:
        """生成缓存键"""
        key_parts = [
            config.provider.value,
            config.model_name,
            str(config.temperature),
            str(config.max_tokens)
        ]
        return "/".join(key_parts)
    
    def _get_model_info(self, config: LLMConfig) -> ModelInfo:
        """获取模型信息"""
        key = f"{config.provider.value}/{config.model_name}"
        
        if key in self._model_registry:
            return self._model_registry[key]
        
        # 如果没有注册信息，创建默认信息
        logger.warning(f"Model info not found for {key}, using defaults")
        return ModelInfo(
            name=config.model_name,
            provider=config.provider,
            max_tokens=config.max_tokens or 4096,
            capabilities=[ModelCapability.TEXT_GENERATION]
        )
    
    def _validate_config(self, config: LLMConfig, model_info: ModelInfo) -> None:
        """验证配置"""
        # 检查API密钥
        if config.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC] and not config.api_key:
            # 尝试从环境变量获取
            import os
            env_key_map = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY"
            }
            
            env_key = env_key_map.get(config.provider)
            if env_key and os.getenv(env_key):
                config.api_key = os.getenv(env_key)
            else:
                raise ConfigurationError(
                    f"API key required for {config.provider.value}",
                    error_code=ErrorCodes.LLM_AUTH_ERROR,
                    context={"provider": config.provider.value}
                )
        
        # 检查令牌限制
        if config.max_tokens and config.max_tokens > model_info.max_tokens:
            logger.warning(
                f"max_tokens ({config.max_tokens}) exceeds model limit "
                f"({model_info.max_tokens}), will be capped"
            )
            config.max_tokens = model_info.max_tokens
    
    def _create_llm_instance(self, config: LLMConfig) -> BaseLanguageModel:
        """创建LangChain LLM实例"""
        provider = config.provider
        
        if provider not in self._providers:
            raise LLMError(
                f"Unsupported provider: {provider.value}",
                error_code=ErrorCodes.LLM_MODEL_ERROR,
                context={"provider": provider.value}
            )
        
        factory_func = self._providers[provider]
        return factory_func(config)
    
    def _register_builtin_providers(self) -> None:
        """注册内置提供商"""
        
        def create_openai_llm(config: LLMConfig) -> BaseLanguageModel:
            """创建OpenAI LLM"""
            return ChatOpenAI(
                model=config.model_name,
                api_key=config.api_key,
                base_url=config.api_base,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                streaming=config.streaming,
                **config.extra_config
            )
        
        def create_huggingface_llm(config: LLMConfig) -> BaseLanguageModel:
            """创建HuggingFace LLM"""
            return HuggingFacePipeline.from_model_id(
                model_id=config.model_name,
                task="text-generation",
                model_kwargs={
                    "temperature": config.temperature,
                    "max_length": config.max_tokens,
                    **config.extra_config
                }
            )
        
        # 注册提供商
        self.register_provider(LLMProvider.OPENAI, create_openai_llm)
        self.register_provider(LLMProvider.HUGGINGFACE, create_huggingface_llm)
    
    def _register_builtin_models(self) -> None:
        """注册内置模型"""
        
        # OpenAI模型
        openai_models = [
            ModelInfo(
                name="gpt-3.5-turbo",
                provider=LLMProvider.OPENAI,
                max_tokens=4096,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.FUNCTION_CALLING
                ],
                context_window=4096,
                cost_per_token=0.002 / 1000,
                description="OpenAI GPT-3.5 Turbo model"
            ),
            ModelInfo(
                name="gpt-4",
                provider=LLMProvider.OPENAI,
                max_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.CODE_GENERATION
                ],
                context_window=8192,
                cost_per_token=0.03 / 1000,
                description="OpenAI GPT-4 model"
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider=LLMProvider.OPENAI,
                max_tokens=4096,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.VISION
                ],
                context_window=128000,
                cost_per_token=0.01 / 1000,
                description="OpenAI GPT-4 Turbo model"
            )
        ]
        
        # 注册模型
        for model in openai_models:
            self.register_model(model)


# 全局LLM工厂实例
_global_factory: Optional[LLMFactory] = None


def get_llm_factory() -> LLMFactory:
    """
    获取全局LLM工厂实例
    
    Returns:
        LLM工厂实例
    """
    global _global_factory
    
    if _global_factory is None:
        _global_factory = LLMFactory()
    
    return _global_factory


def create_llm(
    config: Union[LLMConfig, Dict[str, Any]],
    use_cache: bool = True
) -> LLMInstance:
    """
    创建LLM实例的便捷函数
    
    Args:
        config: LLM配置
        use_cache: 是否使用缓存
        
    Returns:
        LLM实例
    """
    return get_llm_factory().create_llm(config, use_cache)


def create_llm_from_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> LLMInstance:
    """
    从配置文件创建LLM实例的便捷函数
    
    Args:
        provider: 提供商名称
        model: 模型名称
        **kwargs: 额外配置
        
    Returns:
        LLM实例
    """
    return get_llm_factory().create_llm_from_config(provider, model, **kwargs)