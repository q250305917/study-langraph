"""
OpenAI模型使用模板

本模块提供了OpenAI系列模型的统一接入模板，支持：
- GPT-3.5/4 系列模型
- 流式和非流式输出
- 异步和同步调用
- 自动重试和错误处理
- 性能监控和优化
- 参数验证和配置管理

设计原理：
1. 统一接口：继承自TemplateBase，提供一致的使用体验
2. 灵活配置：支持丰富的参数配置和环境变量
3. 健壮性：完善的错误处理和重试机制
4. 性能优化：支持缓存、连接池等优化策略
5. 易用性：简洁的API设计和详细的使用示例

使用示例：
    # 基础使用
    template = OpenAITemplate()
    template.setup(
        api_key="your-api-key",
        model_name="gpt-3.5-turbo"
    )
    result = template.run("你好！")
    
    # 流式输出
    template.setup(stream=True)
    for chunk in template.stream("写一首诗"):
        print(chunk, end="")
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, Union, Iterator, AsyncIterator, List
from dataclasses import dataclass
import tiktoken
import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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
class OpenAIResponse:
    """
    OpenAI响应数据类
    
    统一封装OpenAI API的响应结果，提供便于使用的数据结构。
    """
    content: str                              # 响应内容
    model: str                               # 使用的模型
    usage: Dict[str, int]                    # Token使用统计
    finish_reason: str                       # 完成原因
    response_time: float                     # 响应时间
    total_tokens: int                        # 总token数
    prompt_tokens: int                       # 输入token数  
    completion_tokens: int                   # 输出token数
    estimated_cost: float                    # 预估成本（美元）
    
    @classmethod
    def from_openai_response(
        cls, 
        response: ChatCompletion, 
        response_time: float
    ) -> "OpenAIResponse":
        """从OpenAI原始响应创建实例"""
        choice = response.choices[0]
        usage = response.usage
        
        # 预估成本（基于公开定价，实际可能有差异）
        model_pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # 每1K tokens
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-32k": {"input": 0.06, "output": 0.12}
        }
        
        model = response.model
        pricing = model_pricing.get(model, {"input": 0.002, "output": 0.002})
        estimated_cost = (
            (usage.prompt_tokens / 1000) * pricing["input"] +
            (usage.completion_tokens / 1000) * pricing["output"]
        )
        
        return cls(
            content=choice.message.content or "",
            model=model,
            usage=usage.model_dump(),
            finish_reason=choice.finish_reason or "unknown",
            response_time=response_time,
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost=estimated_cost
        )


class OpenAITemplate(TemplateBase[str, OpenAIResponse]):
    """
    OpenAI模型使用模板
    
    提供OpenAI系列模型的统一接入接口，支持丰富的配置选项和高级功能。
    
    核心特性：
    1. 模型支持：GPT-3.5/4 全系列模型
    2. 输出模式：支持同步、异步、流式输出
    3. 错误处理：自动重试、错误分类、降级策略
    4. 性能优化：连接复用、请求缓存、Token优化
    5. 监控统计：详细的使用统计和性能指标
    6. 配置灵活：支持环境变量、配置文件、运行时配置
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化OpenAI模板
        
        Args:
            config: 模板配置，None时使用默认配置
        """
        super().__init__(config or self._create_default_config())
        
        # OpenAI客户端实例
        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        
        # 当前配置参数
        self.api_key: Optional[str] = None
        self.model_name: str = "gpt-3.5-turbo"
        self.temperature: float = 0.7
        self.max_tokens: int = 1000
        self.top_p: float = 1.0
        self.frequency_penalty: float = 0.0
        self.presence_penalty: float = 0.0
        self.timeout: float = 30.0
        self.stream: bool = False
        self.system_prompt: str = ""
        self.base_url: Optional[str] = None
        
        # 性能和监控
        self.token_encoder = None
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.error_count = 0
        self.last_request_time = 0.0
        
        logger.debug("OpenAI template initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="OpenAITemplate",
            version="1.0.0",
            description="OpenAI模型调用模板，支持GPT-3.5/4系列",
            template_type=TemplateType.LLM,
            author="LangChain Learning Project",
            async_enabled=True,
            cache_enabled=True,
            timeout=60.0,
            retry_count=3
        )
        
        # 添加参数定义
        config.add_parameter("api_key", str, True, None, "OpenAI API密钥")
        config.add_parameter("model_name", str, True, "gpt-3.5-turbo", "模型名称")
        config.add_parameter("temperature", float, False, 0.7, "输出随机性控制参数")
        config.add_parameter("max_tokens", int, False, 1000, "最大输出token数")
        config.add_parameter("top_p", float, False, 1.0, "核采样参数")
        config.add_parameter("frequency_penalty", float, False, 0.0, "频率惩罚")
        config.add_parameter("presence_penalty", float, False, 0.0, "存在惩罚")
        config.add_parameter("timeout", float, False, 30.0, "请求超时时间")
        config.add_parameter("stream", bool, False, False, "是否使用流式输出")
        config.add_parameter("system_prompt", str, False, "", "系统提示词")
        config.add_parameter("base_url", str, False, None, "自定义API基础URL")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置模板参数
        
        Args:
            **parameters: 配置参数
            
        主要参数：
            api_key (str): OpenAI API密钥
            model_name (str): 模型名称，如'gpt-3.5-turbo'
            temperature (float): 输出随机性，0.0-2.0
            max_tokens (int): 最大输出token数
            top_p (float): 核采样参数，0.0-1.0
            frequency_penalty (float): 频率惩罚，-2.0-2.0
            presence_penalty (float): 存在惩罚，-2.0-2.0
            timeout (float): 请求超时时间（秒）
            stream (bool): 是否使用流式输出
            system_prompt (str): 系统提示词
            base_url (str): 自定义API基础URL
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置参数
            self.api_key = parameters.get("api_key", os.getenv("OPENAI_API_KEY"))
            self.model_name = parameters.get("model_name", "gpt-3.5-turbo")
            self.temperature = parameters.get("temperature", 0.7)
            self.max_tokens = parameters.get("max_tokens", 1000)
            self.top_p = parameters.get("top_p", 1.0)
            self.frequency_penalty = parameters.get("frequency_penalty", 0.0)
            self.presence_penalty = parameters.get("presence_penalty", 0.0)
            self.timeout = parameters.get("timeout", 30.0)
            self.stream = parameters.get("stream", False)
            self.system_prompt = parameters.get("system_prompt", "")
            self.base_url = parameters.get("base_url")
            
            # 验证API密钥
            if not self.api_key:
                raise ConfigurationError(
                    "OpenAI API key is required. Please set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter.",
                    error_code=ErrorCodes.CONFIG_VALIDATION_ERROR
                )
            
            # 初始化OpenAI客户端
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            self.client = OpenAI(**client_kwargs)
            self.async_client = AsyncOpenAI(**client_kwargs)
            
            # 初始化token编码器（用于token计数）
            try:
                self.token_encoder = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # 如果模型不在tiktoken支持列表中，使用通用编码器
                self.token_encoder = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Model {self.model_name} not found in tiktoken, using cl100k_base encoding")
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            # 更新状态
            self.status = self.status.CONFIGURED
            
            logger.info(f"OpenAI template configured with model: {self.model_name}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup OpenAI template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def execute(self, input_data: str, **kwargs) -> OpenAIResponse:
        """
        执行OpenAI模型调用（同步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
                - messages: 消息列表（高级用法）
                - system_prompt: 临时系统提示词
                - temperature: 临时温度设置
                - max_tokens: 临时最大token设置
                
        Returns:
            OpenAI响应对象
            
        Raises:
            APIError: API调用失败
            ValidationError: 输入验证失败
        """
        if not self.client:
            raise RuntimeError("Template not configured. Please call setup() first.")
        
        try:
            # 准备消息
            messages = self._prepare_messages(input_data, kwargs)
            
            # 准备请求参数
            request_params = self._prepare_request_params(kwargs)
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 执行API调用（支持重试）
            response = self._call_openai_with_retry(messages, request_params)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 创建响应对象
            result = OpenAIResponse.from_openai_response(response, response_time)
            
            # 更新统计信息
            self._update_statistics(result)
            
            logger.debug(
                f"OpenAI call completed: {result.total_tokens} tokens, "
                f"{response_time:.3f}s, ${result.estimated_cost:.6f}"
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            if isinstance(e, openai.APIError):
                raise APIError(
                    f"OpenAI API error: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
            else:
                raise APIError(
                    f"OpenAI call failed: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
    
    async def execute_async(self, input_data: str, **kwargs) -> OpenAIResponse:
        """
        执行OpenAI模型调用（异步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
            
        Returns:
            OpenAI响应对象
        """
        if not self.async_client:
            raise RuntimeError("Template not configured. Please call setup() first.")
        
        try:
            # 准备消息和参数
            messages = self._prepare_messages(input_data, kwargs)
            request_params = self._prepare_request_params(kwargs)
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 执行异步API调用
            response = await self._call_openai_async_with_retry(messages, request_params)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 创建响应对象
            result = OpenAIResponse.from_openai_response(response, response_time)
            
            # 更新统计信息
            self._update_statistics(result)
            
            logger.debug(
                f"Async OpenAI call completed: {result.total_tokens} tokens, "
                f"{response_time:.3f}s, ${result.estimated_cost:.6f}"
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            if isinstance(e, openai.APIError):
                raise APIError(
                    f"OpenAI API error: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
            else:
                raise APIError(
                    f"Async OpenAI call failed: {str(e)}",
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
            messages = self._prepare_messages(input_data, kwargs)
            request_params = self._prepare_request_params(kwargs)
            request_params["stream"] = True
            
            # 执行流式API调用
            stream_response = self.client.chat.completions.create(
                messages=messages,
                **request_params
            )
            
            # 生成流式输出
            for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.error_count += 1
            logger.error(f"OpenAI stream failed: {str(e)}")
            raise APIError(
                f"OpenAI stream failed: {str(e)}",
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
            messages = self._prepare_messages(input_data, kwargs)
            request_params = self._prepare_request_params(kwargs)
            request_params["stream"] = True
            
            # 执行异步流式API调用
            stream_response = await self.async_client.chat.completions.create(
                messages=messages,
                **request_params
            )
            
            # 生成异步流式输出
            async for chunk in stream_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.error_count += 1
            logger.error(f"Async OpenAI stream failed: {str(e)}")
            raise APIError(
                f"Async OpenAI stream failed: {str(e)}",
                error_code=ErrorCodes.API_REQUEST_FAILED,
                cause=e
            )
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "OpenAI模板使用示例",
            "setup_parameters": {
                "api_key": "${OPENAI_API_KEY}",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "你是一个有用的AI助手"
            },
            "execute_parameters": {
                "input_data": "介绍一下Python的主要特点"
            },
            "expected_output": {
                "type": "OpenAIResponse",
                "fields": {
                    "content": "关于Python特点的详细介绍",
                    "model": "gpt-3.5-turbo",
                    "total_tokens": "约500-800",
                    "estimated_cost": "约$0.001-0.002"
                }
            },
            "usage_code": '''
# 基础使用
from templates.llm import OpenAITemplate

template = OpenAITemplate()
template.setup(
    api_key="your-api-key",
    model_name="gpt-3.5-turbo",
    temperature=0.7
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

# 高级用法 - 自定义消息
result = template.execute(
    "继续对话",
    messages=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        {"role": "user", "content": "介绍一下机器学习"}
    ]
)
'''
        }
    
    def _prepare_messages(self, input_data: str, kwargs: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        准备API调用的消息列表
        
        Args:
            input_data: 输入文本
            kwargs: 额外参数
            
        Returns:
            消息列表
        """
        messages = []
        
        # 添加系统提示词
        system_prompt = kwargs.get("system_prompt", self.system_prompt)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 处理自定义消息列表
        custom_messages = kwargs.get("messages", [])
        if custom_messages:
            messages.extend(custom_messages)
        
        # 添加用户输入
        if input_data:
            messages.append({"role": "user", "content": input_data})
        
        return messages
    
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
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
        }
        
        # 移除None值
        return {k: v for k, v in params.items() if v is not None}
    
    def _call_openai_with_retry(self, messages: List[Dict], params: Dict) -> ChatCompletion:
        """
        带重试机制的OpenAI API调用
        
        Args:
            messages: 消息列表
            params: 请求参数
            
        Returns:
            OpenAI响应对象
        """
        max_retries = self.config.retry_count
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    **params
                )
                self.request_count += 1
                return response
                
            except openai.RateLimitError as e:
                # 处理速率限制错误
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 60)  # 指数退避，最大60秒
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    last_exception = e
                else:
                    raise e
                    
            except openai.APITimeoutError as e:
                # 处理超时错误
                if attempt < max_retries:
                    logger.warning(f"Request timeout, retrying {attempt + 1}")
                    last_exception = e
                else:
                    raise e
                    
            except openai.APIConnectionError as e:
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
    
    async def _call_openai_async_with_retry(self, messages: List[Dict], params: Dict) -> ChatCompletion:
        """
        带重试机制的异步OpenAI API调用
        
        Args:
            messages: 消息列表
            params: 请求参数
            
        Returns:
            OpenAI响应对象
        """
        max_retries = self.config.retry_count
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.async_client.chat.completions.create(
                    messages=messages,
                    **params
                )
                self.request_count += 1
                return response
                
            except openai.RateLimitError as e:
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 60)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    last_exception = e
                else:
                    raise e
                    
            except openai.APITimeoutError as e:
                if attempt < max_retries:
                    logger.warning(f"Request timeout, retrying {attempt + 1}")
                    last_exception = e
                else:
                    raise e
                    
            except openai.APIConnectionError as e:
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
    
    def _update_statistics(self, response: OpenAIResponse) -> None:
        """
        更新使用统计信息
        
        Args:
            response: OpenAI响应对象
        """
        self.total_tokens_used += response.total_tokens
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
    
    def get_token_count(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
            
        Returns:
            token数量
        """
        if self.token_encoder:
            return len(self.token_encoder.encode(text))
        else:
            # 粗略估算：1 token ≈ 4 字符
            return len(text) // 4
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        估算API调用成本
        
        Args:
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            
        Returns:
            预估成本（美元）
        """
        # 价格表（每1K tokens）
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-32k": {"input": 0.06, "output": 0.12}
        }
        
        model_pricing = pricing.get(self.model_name, {"input": 0.002, "output": 0.002})
        
        return (
            (prompt_tokens / 1000) * model_pricing["input"] +
            (completion_tokens / 1000) * model_pricing["output"]
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