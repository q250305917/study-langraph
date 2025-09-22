"""
Anthropic模型使用模板

本模块提供了Anthropic Claude系列模型的统一接入模板，支持：
- Claude-3 系列模型 (Haiku, Sonnet, Opus)
- Claude-2 系列模型
- 流式和非流式输出
- 异步和同步调用
- 自动重试和错误处理
- 性能监控和优化
- 参数验证和配置管理

设计原理：
1. 统一接口：继承自TemplateBase，与OpenAI模板保持接口一致
2. Claude优化：针对Claude特性优化的参数设置和消息格式
3. 健壮性：完善的错误处理和重试机制
4. 性能监控：详细的使用统计和成本跟踪
5. 易用性：简洁的API设计和丰富的使用示例

使用示例：
    # 基础使用
    template = AnthropicTemplate()
    template.setup(
        api_key="your-api-key",
        model_name="claude-3-sonnet-20240229"
    )
    result = template.run("你好，Claude！")
    
    # 流式输出
    template.setup(stream=True)
    for chunk in template.stream("写一篇关于AI的文章"):
        print(chunk, end="")
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, Union, Iterator, AsyncIterator, List
from dataclasses import dataclass
import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message as AnthropicMessage, MessageStreamEvent

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    APIError,
    ErrorCodes
)

logger = get_logger(__name__)


@dataclass
class AnthropicResponse:
    """
    Anthropic响应数据类
    
    统一封装Anthropic API的响应结果，提供便于使用的数据结构。
    """
    content: str                              # 响应内容
    model: str                               # 使用的模型
    usage: Dict[str, int]                    # Token使用统计
    stop_reason: str                         # 停止原因
    response_time: float                     # 响应时间
    input_tokens: int                        # 输入token数
    output_tokens: int                       # 输出token数
    estimated_cost: float                    # 预估成本（美元）
    
    @classmethod
    def from_anthropic_response(
        cls, 
        response: AnthropicMessage, 
        response_time: float
    ) -> "AnthropicResponse":
        """从Anthropic原始响应创建实例"""
        # 获取文本内容
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
        
        # 预估成本（基于公开定价，实际可能有差异）
        model_pricing = {
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},   # 每1K tokens
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024},
            "claude-instant-1.2": {"input": 0.00163, "output": 0.00551}
        }
        
        model = response.model
        pricing = model_pricing.get(model, {"input": 0.008, "output": 0.024})
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        estimated_cost = (
            (input_tokens / 1000) * pricing["input"] +
            (output_tokens / 1000) * pricing["output"]
        )
        
        return cls(
            content=content,
            model=model,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            stop_reason=response.stop_reason or "unknown",
            response_time=response_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost
        )


class AnthropicTemplate(TemplateBase[str, AnthropicResponse]):
    """
    Anthropic Claude模型使用模板
    
    提供Claude系列模型的统一接入接口，支持丰富的配置选项和高级功能。
    
    核心特性：
    1. 模型支持：Claude-3 全系列模型和Claude-2系列
    2. 输出模式：支持同步、异步、流式输出
    3. 错误处理：自动重试、错误分类、降级策略
    4. 性能优化：连接复用、请求缓存、Token优化
    5. 监控统计：详细的使用统计和性能指标
    6. Claude特性：充分利用Claude的长上下文和安全特性
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化Anthropic模板
        
        Args:
            config: 模板配置，None时使用默认配置
        """
        super().__init__(config or self._create_default_config())
        
        # Anthropic客户端实例
        self.client: Optional[Anthropic] = None
        self.async_client: Optional[AsyncAnthropic] = None
        
        # 当前配置参数
        self.api_key: Optional[str] = None
        self.model_name: str = "claude-3-sonnet-20240229"
        self.max_tokens: int = 1000
        self.temperature: float = 0.7
        self.top_p: float = 1.0
        self.top_k: int = 5
        self.timeout: float = 30.0
        self.stream: bool = False
        self.system_prompt: str = ""
        
        # 性能和监控
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.error_count = 0
        self.last_request_time = 0.0
        
        logger.debug("Anthropic template initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="AnthropicTemplate",
            version="1.0.0",
            description="Anthropic Claude模型调用模板，支持Claude-3系列",
            template_type=TemplateType.LLM,
            author="LangChain Learning Project",
            async_enabled=True,
            cache_enabled=True,
            timeout=60.0,
            retry_count=3
        )
        
        # 添加参数定义
        config.add_parameter("api_key", str, True, None, "Anthropic API密钥")
        config.add_parameter("model_name", str, True, "claude-3-sonnet-20240229", "模型名称")
        config.add_parameter("max_tokens", int, False, 1000, "最大输出token数")
        config.add_parameter("temperature", float, False, 0.7, "输出随机性控制参数")
        config.add_parameter("top_p", float, False, 1.0, "核采样参数")
        config.add_parameter("top_k", int, False, 5, "top-k采样参数")
        config.add_parameter("timeout", float, False, 30.0, "请求超时时间")
        config.add_parameter("stream", bool, False, False, "是否使用流式输出")
        config.add_parameter("system_prompt", str, False, "", "系统提示词")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置模板参数
        
        Args:
            **parameters: 配置参数
            
        主要参数：
            api_key (str): Anthropic API密钥
            model_name (str): 模型名称，如'claude-3-sonnet-20240229'
            max_tokens (int): 最大输出token数
            temperature (float): 输出随机性，0.0-1.0
            top_p (float): 核采样参数，0.0-1.0
            top_k (int): top-k采样参数
            timeout (float): 请求超时时间（秒）
            stream (bool): 是否使用流式输出
            system_prompt (str): 系统提示词
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置参数
            self.api_key = parameters.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
            self.model_name = parameters.get("model_name", "claude-3-sonnet-20240229")
            self.max_tokens = parameters.get("max_tokens", 1000)
            self.temperature = parameters.get("temperature", 0.7)
            self.top_p = parameters.get("top_p", 1.0)
            self.top_k = parameters.get("top_k", 5)
            self.timeout = parameters.get("timeout", 30.0)
            self.stream = parameters.get("stream", False)
            self.system_prompt = parameters.get("system_prompt", "")
            
            # 验证API密钥
            if not self.api_key:
                raise ConfigurationError(
                    "Anthropic API key is required. Please set ANTHROPIC_API_KEY environment variable "
                    "or pass api_key parameter.",
                    error_code=ErrorCodes.CONFIG_VALIDATION_ERROR
                )
            
            # 初始化Anthropic客户端
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout
            }
            
            self.client = Anthropic(**client_kwargs)
            self.async_client = AsyncAnthropic(**client_kwargs)
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            # 更新状态
            self.status = self.status.CONFIGURED
            
            logger.info(f"Anthropic template configured with model: {self.model_name}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup Anthropic template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def execute(self, input_data: str, **kwargs) -> AnthropicResponse:
        """
        执行Anthropic模型调用（同步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
                - messages: 消息列表（高级用法）
                - system: 临时系统提示词
                - max_tokens: 临时最大token设置
                - temperature: 临时温度设置
                
        Returns:
            Anthropic响应对象
            
        Raises:
            APIError: API调用失败
            ValidationError: 输入验证失败
        """
        if not self.client:
            raise RuntimeError("Template not configured. Please call setup() first.")
        
        try:
            # 准备消息和参数
            messages, system = self._prepare_messages(input_data, kwargs)
            request_params = self._prepare_request_params(kwargs)
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 执行API调用（支持重试）
            response = self._call_anthropic_with_retry(messages, system, request_params)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 创建响应对象
            result = AnthropicResponse.from_anthropic_response(response, response_time)
            
            # 更新统计信息
            self._update_statistics(result)
            
            logger.debug(
                f"Anthropic call completed: {result.input_tokens + result.output_tokens} tokens, "
                f"{response_time:.3f}s, ${result.estimated_cost:.6f}"
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            if isinstance(e, anthropic.APIError):
                raise APIError(
                    f"Anthropic API error: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
            else:
                raise APIError(
                    f"Anthropic call failed: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
    
    async def execute_async(self, input_data: str, **kwargs) -> AnthropicResponse:
        """
        执行Anthropic模型调用（异步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
            
        Returns:
            Anthropic响应对象
        """
        if not self.async_client:
            raise RuntimeError("Template not configured. Please call setup() first.")
        
        try:
            # 准备消息和参数
            messages, system = self._prepare_messages(input_data, kwargs)
            request_params = self._prepare_request_params(kwargs)
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 执行异步API调用
            response = await self._call_anthropic_async_with_retry(messages, system, request_params)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 创建响应对象
            result = AnthropicResponse.from_anthropic_response(response, response_time)
            
            # 更新统计信息
            self._update_statistics(result)
            
            logger.debug(
                f"Async Anthropic call completed: {result.input_tokens + result.output_tokens} tokens, "
                f"{response_time:.3f}s, ${result.estimated_cost:.6f}"
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            if isinstance(e, anthropic.APIError):
                raise APIError(
                    f"Anthropic API error: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
            else:
                raise APIError(
                    f"Async Anthropic call failed: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
    
    def stream(self, input_data: str, **kwargs) -> Iterator[str]:
        """
        流式输出生成器（同步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
            
        Yields:
            str: 流式输出的文本片段
        """
        if not self.client:
            raise RuntimeError("Template not configured. Please call setup() first.")
        
        try:
            # 准备消息和参数
            messages, system = self._prepare_messages(input_data, kwargs)
            request_params = self._prepare_request_params(kwargs)
            
            # 执行流式API调用
            stream_kwargs = {"messages": messages, **request_params}
            if system:
                stream_kwargs["system"] = system
            
            with self.client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            yield event.delta.text
                    
        except Exception as e:
            self.error_count += 1
            logger.error(f"Anthropic stream failed: {str(e)}")
            raise APIError(
                f"Anthropic stream failed: {str(e)}",
                error_code=ErrorCodes.API_REQUEST_FAILED,
                cause=e
            )
    
    async def stream_async(self, input_data: str, **kwargs) -> AsyncIterator[str]:
        """
        流式输出生成器（异步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
            
        Yields:
            str: 流式输出的文本片段
        """
        if not self.async_client:
            raise RuntimeError("Template not configured. Please call setup() first.")
        
        try:
            # 准备消息和参数
            messages, system = self._prepare_messages(input_data, kwargs)
            request_params = self._prepare_request_params(kwargs)
            
            # 执行异步流式API调用
            stream_kwargs = {"messages": messages, **request_params}
            if system:
                stream_kwargs["system"] = system
            
            async with self.async_client.messages.stream(**stream_kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            yield event.delta.text
                    
        except Exception as e:
            self.error_count += 1
            logger.error(f"Async Anthropic stream failed: {str(e)}")
            raise APIError(
                f"Async Anthropic stream failed: {str(e)}",
                error_code=ErrorCodes.API_REQUEST_FAILED,
                cause=e
            )
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "Anthropic Claude模板使用示例",
            "setup_parameters": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "model_name": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "temperature": 0.7,
                "system_prompt": "你是一个有用的AI助手，专门帮助用户解答问题"
            },
            "execute_parameters": {
                "input_data": "介绍一下Python的主要特点和优势"
            },
            "expected_output": {
                "type": "AnthropicResponse",
                "fields": {
                    "content": "关于Python特点的详细介绍",
                    "model": "claude-3-sonnet-20240229",
                    "input_tokens": "约50-100",
                    "output_tokens": "约500-800",
                    "estimated_cost": "约$0.002-0.015"
                }
            },
            "usage_code": '''
# 基础使用
from templates.llm import AnthropicTemplate

template = AnthropicTemplate()
template.setup(
    api_key="your-api-key",
    model_name="claude-3-sonnet-20240229",
    max_tokens=1000
)

result = template.run("介绍Python特点")
print(result.content)

# 流式输出
template.setup(stream=True)
for chunk in template.stream("写一首关于编程的诗"):
    print(chunk, end="", flush=True)

# 异步调用
async def async_example():
    result = await template.run_async("异步处理示例")
    return result.content

# 高级用法 - 多轮对话
result = template.execute(
    "继续对话",
    messages=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        {"role": "user", "content": "介绍一下机器学习"}
    ]
)

# 使用系统提示词
result = template.execute(
    "写一个Python函数",
    system="你是一个Python编程专家，只提供高质量的代码和解释"
)
'''
        }
    
    def _prepare_messages(self, input_data: str, kwargs: Dict[str, Any]) -> tuple[List[Dict[str, str]], Optional[str]]:
        """
        准备API调用的消息列表和系统提示词
        
        Args:
            input_data: 输入文本
            kwargs: 额外参数
            
        Returns:
            (消息列表, 系统提示词)
        """
        messages = []
        
        # 获取系统提示词（Anthropic单独处理system参数）
        system = kwargs.get("system", self.system_prompt) or None
        
        # 处理自定义消息列表
        custom_messages = kwargs.get("messages", [])
        if custom_messages:
            messages.extend(custom_messages)
        
        # 添加用户输入
        if input_data:
            messages.append({"role": "user", "content": input_data})
        
        return messages, system
    
    def _prepare_request_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备API请求参数
        
        Args:
            kwargs: 额外参数
            
        Returns:
            请求参数字典
        """
        params = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
        }
        
        # 移除None值
        return {k: v for k, v in params.items() if v is not None}
    
    def _call_anthropic_with_retry(
        self, 
        messages: List[Dict], 
        system: Optional[str],
        params: Dict
    ) -> AnthropicMessage:
        """
        带重试机制的Anthropic API调用
        
        Args:
            messages: 消息列表
            system: 系统提示词
            params: 请求参数
            
        Returns:
            Anthropic响应对象
        """
        max_retries = self.config.retry_count
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                kwargs = {"messages": messages, **params}
                if system:
                    kwargs["system"] = system
                
                response = self.client.messages.create(**kwargs)
                self.request_count += 1
                return response
                
            except anthropic.RateLimitError as e:
                # 处理速率限制错误
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 60)  # 指数退避，最大60秒
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    last_exception = e
                else:
                    raise e
                    
            except anthropic.APITimeoutError as e:
                # 处理超时错误
                if attempt < max_retries:
                    logger.warning(f"Request timeout, retrying {attempt + 1}")
                    last_exception = e
                else:
                    raise e
                    
            except anthropic.APIConnectionError as e:
                # 处理连接错误
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Connection error, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    last_exception = e
                else:
                    raise e
                    
            except Exception as e:
                # 其他错误不重试
                raise e
        
        # 如果所有重试都失败，抛出最后一个异常
        if last_exception:
            raise last_exception
    
    async def _call_anthropic_async_with_retry(
        self, 
        messages: List[Dict], 
        system: Optional[str],
        params: Dict
    ) -> AnthropicMessage:
        """
        带重试机制的异步Anthropic API调用
        
        Args:
            messages: 消息列表
            system: 系统提示词
            params: 请求参数
            
        Returns:
            Anthropic响应对象
        """
        max_retries = self.config.retry_count
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                kwargs = {"messages": messages, **params}
                if system:
                    kwargs["system"] = system
                
                response = await self.async_client.messages.create(**kwargs)
                self.request_count += 1
                return response
                
            except anthropic.RateLimitError as e:
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 60)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    last_exception = e
                else:
                    raise e
                    
            except anthropic.APITimeoutError as e:
                if attempt < max_retries:
                    logger.warning(f"Request timeout, retrying {attempt + 1}")
                    last_exception = e
                else:
                    raise e
                    
            except anthropic.APIConnectionError as e:
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 30)
                    logger.warning(f"Connection error, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    last_exception = e
                else:
                    raise e
                    
            except Exception as e:
                raise e
        
        if last_exception:
            raise last_exception
    
    def _update_statistics(self, response: AnthropicResponse) -> None:
        """
        更新使用统计信息
        
        Args:
            response: Anthropic响应对象
        """
        total_tokens = response.input_tokens + response.output_tokens
        self.total_tokens_used += total_tokens
        self.total_cost += response.estimated_cost
        self.last_request_time = time.time()
        
        # 更新性能指标
        self.metrics.update({
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "error_count": self.error_count,
            "average_tokens_per_request": self.total_tokens_used / max(self.request_count, 1),
            "average_cost_per_request": self.total_cost / max(self.request_count, 1),
            "last_response_time": response.response_time
        })
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        估算API调用成本
        
        Args:
            input_tokens: 输入token数
            output_tokens: 输出token数
            
        Returns:
            预估成本（美元）
        """
        # Claude价格表（每1K tokens）
        pricing = {
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024},
            "claude-instant-1.2": {"input": 0.00163, "output": 0.00551}
        }
        
        model_pricing = pricing.get(self.model_name, {"input": 0.008, "output": 0.024})
        
        return (
            (input_tokens / 1000) * model_pricing["input"] +
            (output_tokens / 1000) * model_pricing["output"]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取详细的使用统计信息
        
        Returns:
            统计信息字典
        """
        base_stats = self.get_metrics()
        
        return {
            **base_stats,
            "model_name": self.model_name,
            "total_requests": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": round(self.total_cost, 6),
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_tokens_per_request": self.total_tokens_used / max(self.request_count, 1),
            "average_cost_per_request": self.total_cost / max(self.request_count, 1),
            "last_request_time": self.last_request_time
        }