"""
本地LLM模型使用模板

本模块提供了本地部署LLM模型的统一接入模板，支持：
- Ollama 本地模型服务
- LlamaCpp Python bindings
- Transformers本地模型
- 自定义本地模型API
- 流式和非流式输出
- 异步和同步调用
- 模型管理和自动下载
- 性能监控和优化

设计原理：
1. 多后端支持：统一接口访问不同的本地模型后端
2. 资源管理：智能的模型加载和内存管理
3. 性能优化：批处理、缓存、量化等优化策略
4. 易于扩展：插件化架构支持新的模型后端
5. 零成本运行：无需API费用，完全本地化
6. 隐私保护：数据不离开本地环境

使用示例：
    # 使用Ollama后端
    template = LocalLLMTemplate()
    template.setup(
        backend="ollama",
        model_name="llama2",
        base_url="http://localhost:11434"
    )
    result = template.run("你好，本地AI！")
    
    # 使用LlamaCpp后端
    template.setup(
        backend="llamacpp",
        model_path="/path/to/model.gguf",
        n_ctx=2048
    )
"""

import os
import time
import asyncio
import subprocess
import json
import requests
from typing import Dict, Any, Optional, Union, Iterator, AsyncIterator, List
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    APIError,
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class LocalLLMBackend(Enum):
    """本地LLM后端类型"""
    OLLAMA = "ollama"              # Ollama服务
    LLAMACPP = "llamacpp"          # LlamaCpp Python
    TRANSFORMERS = "transformers"  # Hugging Face Transformers
    CUSTOM_API = "custom_api"      # 自定义API服务


@dataclass
class LocalLLMResponse:
    """
    本地LLM响应数据类
    
    统一封装本地模型的响应结果，提供便于使用的数据结构。
    """
    content: str                              # 响应内容
    model: str                               # 使用的模型
    backend: str                             # 使用的后端
    generation_time: float                   # 生成时间
    tokens_per_second: float                 # 生成速度（tokens/秒）
    total_tokens: int                        # 总token数（估算）
    prompt_tokens: int                       # 输入token数（估算）
    completion_tokens: int                   # 输出token数（估算）
    memory_usage: Optional[int] = None       # 内存使用（MB）
    
    @classmethod
    def create(
        cls,
        content: str,
        model: str,
        backend: str,
        generation_time: float,
        prompt_text: str = "",
        **kwargs
    ) -> "LocalLLMResponse":
        """创建响应实例"""
        # 估算token数量（粗略计算：1 token ≈ 4 字符）
        prompt_tokens = len(prompt_text) // 4 if prompt_text else 0
        completion_tokens = len(content) // 4
        total_tokens = prompt_tokens + completion_tokens
        tokens_per_second = completion_tokens / max(generation_time, 0.001)
        
        return cls(
            content=content,
            model=model,
            backend=backend,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            memory_usage=kwargs.get("memory_usage")
        )


class LocalLLMTemplate(TemplateBase[str, LocalLLMResponse]):
    """
    本地LLM模型使用模板
    
    提供本地模型的统一接入接口，支持多种本地部署方案。
    
    核心特性：
    1. 多后端支持：Ollama、LlamaCpp、Transformers等
    2. 模型管理：自动下载、加载、卸载模型
    3. 性能优化：批处理、缓存、内存管理
    4. 流式输出：支持实时流式生成
    5. 资源监控：CPU、内存、GPU使用情况
    6. 零成本：完全本地化，无API费用
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化本地LLM模板
        
        Args:
            config: 模板配置，None时使用默认配置
        """
        super().__init__(config or self._create_default_config())
        
        # 后端配置
        self.backend: LocalLLMBackend = LocalLLMBackend.OLLAMA
        self.model_name: str = "llama2"
        self.model_path: Optional[str] = None
        self.base_url: str = "http://localhost:11434"
        
        # 生成参数
        self.temperature: float = 0.7
        self.max_tokens: int = 1000
        self.top_p: float = 1.0
        self.top_k: int = 40
        self.repeat_penalty: float = 1.1
        self.system_prompt: str = ""
        
        # 后端特定参数
        self.n_ctx: int = 2048          # LlamaCpp context size
        self.n_gpu_layers: int = 0      # GPU layers for LlamaCpp
        self.device: str = "auto"       # Device for Transformers
        
        # 客户端和连接
        self.session = None
        self.model_instance = None
        
        # 性能和监控
        self.request_count = 0
        self.total_generation_time = 0.0
        self.total_tokens_generated = 0
        self.error_count = 0
        self.last_request_time = 0.0
        
        logger.debug("Local LLM template initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="LocalLLMTemplate",
            version="1.0.0",
            description="本地LLM模型调用模板，支持Ollama、LlamaCpp等",
            template_type=TemplateType.LLM,
            author="LangChain Learning Project",
            async_enabled=True,
            cache_enabled=True,
            timeout=120.0,  # 本地模型可能需要更长时间
            retry_count=2
        )
        
        # 添加参数定义
        config.add_parameter("backend", str, True, "ollama", "本地LLM后端类型")
        config.add_parameter("model_name", str, True, "llama2", "模型名称")
        config.add_parameter("model_path", str, False, None, "模型文件路径（LlamaCpp）")
        config.add_parameter("base_url", str, False, "http://localhost:11434", "Ollama服务URL")
        config.add_parameter("temperature", float, False, 0.7, "输出随机性控制参数")
        config.add_parameter("max_tokens", int, False, 1000, "最大输出token数")
        config.add_parameter("top_p", float, False, 1.0, "核采样参数")
        config.add_parameter("top_k", int, False, 40, "top-k采样参数")
        config.add_parameter("repeat_penalty", float, False, 1.1, "重复惩罚")
        config.add_parameter("system_prompt", str, False, "", "系统提示词")
        config.add_parameter("n_ctx", int, False, 2048, "上下文大小（LlamaCpp）")
        config.add_parameter("n_gpu_layers", int, False, 0, "GPU层数（LlamaCpp）")
        config.add_parameter("device", str, False, "auto", "设备类型（Transformers）")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置模板参数
        
        Args:
            **parameters: 配置参数
            
        主要参数：
            backend (str): 后端类型 ("ollama", "llamacpp", "transformers", "custom_api")
            model_name (str): 模型名称
            model_path (str): 模型文件路径（LlamaCpp专用）
            base_url (str): 服务URL（Ollama/自定义API）
            temperature (float): 输出随机性，0.0-2.0
            max_tokens (int): 最大输出token数
            top_p (float): 核采样参数
            top_k (int): top-k采样参数
            repeat_penalty (float): 重复惩罚
            system_prompt (str): 系统提示词
            n_ctx (int): 上下文大小（LlamaCpp）
            n_gpu_layers (int): GPU层数（LlamaCpp）
            device (str): 设备类型（Transformers）
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置基础参数
            backend_str = parameters.get("backend", "ollama")
            try:
                self.backend = LocalLLMBackend(backend_str)
            except ValueError:
                raise ValidationError(f"Unsupported backend: {backend_str}")
            
            self.model_name = parameters.get("model_name", "llama2")
            self.model_path = parameters.get("model_path")
            self.base_url = parameters.get("base_url", "http://localhost:11434")
            
            # 设置生成参数
            self.temperature = parameters.get("temperature", 0.7)
            self.max_tokens = parameters.get("max_tokens", 1000)
            self.top_p = parameters.get("top_p", 1.0)
            self.top_k = parameters.get("top_k", 40)
            self.repeat_penalty = parameters.get("repeat_penalty", 1.1)
            self.system_prompt = parameters.get("system_prompt", "")
            
            # 设置后端特定参数
            self.n_ctx = parameters.get("n_ctx", 2048)
            self.n_gpu_layers = parameters.get("n_gpu_layers", 0)
            self.device = parameters.get("device", "auto")
            
            # 初始化后端
            self._initialize_backend()
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            # 更新状态
            self.status = self.status.CONFIGURED
            
            logger.info(f"Local LLM template configured with backend: {self.backend.value}, model: {self.model_name}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup Local LLM template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def _initialize_backend(self) -> None:
        """初始化本地LLM后端"""
        try:
            if self.backend == LocalLLMBackend.OLLAMA:
                self._initialize_ollama()
            elif self.backend == LocalLLMBackend.LLAMACPP:
                self._initialize_llamacpp()
            elif self.backend == LocalLLMBackend.TRANSFORMERS:
                self._initialize_transformers()
            elif self.backend == LocalLLMBackend.CUSTOM_API:
                self._initialize_custom_api()
            else:
                raise ConfigurationError(f"Unsupported backend: {self.backend}")
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize {self.backend.value} backend: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def _initialize_ollama(self) -> None:
        """初始化Ollama后端"""
        import requests
        
        # 创建session
        self.session = requests.Session()
        
        # 检查Ollama服务是否可用
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama service is available")
            
            # 检查模型是否存在
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                # 尝试下载模型
                self._download_ollama_model()
                
        except requests.RequestException as e:
            raise ConfigurationError(
                f"Cannot connect to Ollama service at {self.base_url}. "
                f"Please ensure Ollama is running. Error: {str(e)}"
            )
    
    def _download_ollama_model(self) -> None:
        """下载Ollama模型"""
        logger.info(f"Attempting to download model: {self.model_name}")
        try:
            # 发起下载请求
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=300  # 下载可能需要较长时间
            )
            response.raise_for_status()
            
            # 处理流式响应以显示下载进度
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "status" in data:
                            logger.info(f"Download progress: {data['status']}")
                        if data.get("status") == "success":
                            logger.info(f"Model {self.model_name} downloaded successfully")
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            raise ResourceError(
                f"Failed to download model {self.model_name}: {str(e)}",
                error_code=ErrorCodes.RESOURCE_NOT_FOUND
            )
    
    def _initialize_llamacpp(self) -> None:
        """初始化LlamaCpp后端"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ConfigurationError(
                "llama-cpp-python not installed. Please install with: pip install llama-cpp-python",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR
            )
        
        if not self.model_path:
            raise ConfigurationError("model_path is required for LlamaCpp backend")
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise ResourceError(
                f"Model file not found: {self.model_path}",
                error_code=ErrorCodes.RESOURCE_NOT_FOUND
            )
        
        # 初始化Llama模型
        try:
            self.model_instance = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info(f"LlamaCpp model loaded: {self.model_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load LlamaCpp model: {str(e)}")
    
    def _initialize_transformers(self) -> None:
        """初始化Transformers后端"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ConfigurationError(
                "transformers and torch not installed. Please install with: pip install transformers torch",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR
            )
        
        try:
            # 确定设备
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model_instance = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            
            logger.info(f"Transformers model loaded: {self.model_name} on {device}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load Transformers model: {str(e)}")
    
    def _initialize_custom_api(self) -> None:
        """初始化自定义API后端"""
        import requests
        
        # 创建session
        self.session = requests.Session()
        
        # 检查自定义API服务是否可用
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            logger.info("Custom API service is available")
        except requests.RequestException as e:
            logger.warning(f"Cannot verify custom API service at {self.base_url}: {str(e)}")
    
    def execute(self, input_data: str, **kwargs) -> LocalLLMResponse:
        """
        执行本地LLM模型调用（同步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
                - system_prompt: 临时系统提示词
                - temperature: 临时温度设置
                - max_tokens: 临时最大token设置
                
        Returns:
            本地LLM响应对象
            
        Raises:
            APIError: 模型调用失败
            ValidationError: 输入验证失败
        """
        if not self._is_backend_ready():
            raise RuntimeError("Backend not configured. Please call setup() first.")
        
        try:
            # 准备提示词
            prompt = self._prepare_prompt(input_data, kwargs)
            
            # 记录开始时间
            start_time = time.time()
            
            # 根据后端执行调用
            if self.backend == LocalLLMBackend.OLLAMA:
                content = self._call_ollama(prompt, kwargs)
            elif self.backend == LocalLLMBackend.LLAMACPP:
                content = self._call_llamacpp(prompt, kwargs)
            elif self.backend == LocalLLMBackend.TRANSFORMERS:
                content = self._call_transformers(prompt, kwargs)
            elif self.backend == LocalLLMBackend.CUSTOM_API:
                content = self._call_custom_api(prompt, kwargs)
            else:
                raise APIError(f"Unsupported backend: {self.backend}")
            
            # 计算生成时间
            generation_time = time.time() - start_time
            
            # 创建响应对象
            result = LocalLLMResponse.create(
                content=content,
                model=self.model_name,
                backend=self.backend.value,
                generation_time=generation_time,
                prompt_text=prompt
            )
            
            # 更新统计信息
            self._update_statistics(result)
            
            logger.debug(
                f"Local LLM call completed: {result.completion_tokens} tokens, "
                f"{generation_time:.3f}s, {result.tokens_per_second:.1f} tokens/s"
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            if isinstance(e, APIError):
                raise
            else:
                raise APIError(
                    f"Local LLM call failed: {str(e)}",
                    error_code=ErrorCodes.API_REQUEST_FAILED,
                    cause=e
                )
    
    async def execute_async(self, input_data: str, **kwargs) -> LocalLLMResponse:
        """
        执行本地LLM模型调用（异步版本）
        
        对于本地模型，通常是CPU/GPU密集型操作，所以使用线程池执行同步调用
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data, **kwargs)
    
    def stream(self, input_data: str, **kwargs) -> Iterator[str]:
        """
        流式输出生成器（同步版本）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
            
        Yields:
            str: 流式输出的文本片段
        """
        if not self._is_backend_ready():
            raise RuntimeError("Backend not configured. Please call setup() first.")
        
        try:
            prompt = self._prepare_prompt(input_data, kwargs)
            
            if self.backend == LocalLLMBackend.OLLAMA:
                yield from self._stream_ollama(prompt, kwargs)
            elif self.backend == LocalLLMBackend.LLAMACPP:
                yield from self._stream_llamacpp(prompt, kwargs)
            elif self.backend == LocalLLMBackend.TRANSFORMERS:
                yield from self._stream_transformers(prompt, kwargs)
            elif self.backend == LocalLLMBackend.CUSTOM_API:
                yield from self._stream_custom_api(prompt, kwargs)
            else:
                raise APIError(f"Streaming not supported for backend: {self.backend}")
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Local LLM stream failed: {str(e)}")
            raise APIError(
                f"Local LLM stream failed: {str(e)}",
                error_code=ErrorCodes.API_REQUEST_FAILED,
                cause=e
            )
    
    async def stream_async(self, input_data: str, **kwargs) -> AsyncIterator[str]:
        """
        流式输出生成器（异步版本）
        
        将同步流式生成器包装为异步版本
        """
        loop = asyncio.get_event_loop()
        
        # 在线程池中运行同步流式生成器
        def sync_generator():
            return self.stream(input_data, **kwargs)
        
        generator = await loop.run_in_executor(None, sync_generator)
        
        for chunk in generator:
            yield chunk
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "本地LLM模板使用示例",
            "setup_parameters": {
                "backend": "ollama",
                "model_name": "llama2",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "你是一个运行在本地的AI助手"
            },
            "execute_parameters": {
                "input_data": "介绍一下本地LLM的优势"
            },
            "expected_output": {
                "type": "LocalLLMResponse",
                "fields": {
                    "content": "关于本地LLM优势的介绍",
                    "backend": "ollama",
                    "model": "llama2",
                    "generation_time": "约3-10秒",
                    "tokens_per_second": "约10-50 tokens/s"
                }
            },
            "usage_code": '''
# Ollama后端
from templates.llm import LocalLLMTemplate

template = LocalLLMTemplate()
template.setup(
    backend="ollama",
    model_name="llama2",
    temperature=0.7
)

result = template.run("你好，本地AI！")
print(result.content)

# LlamaCpp后端
template.setup(
    backend="llamacpp",
    model_path="/path/to/model.gguf",
    n_ctx=2048,
    n_gpu_layers=10
)

# 流式输出
for chunk in template.stream("写一篇关于本地AI的文章"):
    print(chunk, end="", flush=True)

# 检查模型状态
stats = template.get_statistics()
print(f"平均生成速度: {stats['average_tokens_per_second']:.1f} tokens/s")
'''
        }
    
    def _is_backend_ready(self) -> bool:
        """检查后端是否就绪"""
        if self.backend == LocalLLMBackend.OLLAMA:
            return self.session is not None
        elif self.backend == LocalLLMBackend.LLAMACPP:
            return self.model_instance is not None
        elif self.backend == LocalLLMBackend.TRANSFORMERS:
            return self.model_instance is not None and hasattr(self, 'tokenizer')
        elif self.backend == LocalLLMBackend.CUSTOM_API:
            return self.session is not None
        return False
    
    def _prepare_prompt(self, input_data: str, kwargs: Dict[str, Any]) -> str:
        """
        准备模型输入提示词
        
        Args:
            input_data: 用户输入
            kwargs: 额外参数
            
        Returns:
            格式化的提示词
        """
        system_prompt = kwargs.get("system_prompt", self.system_prompt)
        
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {input_data}\n\nAssistant:"
        else:
            return f"User: {input_data}\n\nAssistant:"
    
    def _call_ollama(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """调用Ollama模型"""
        params = self._get_generation_params(kwargs)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": params
        }
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _call_llamacpp(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """调用LlamaCpp模型"""
        params = self._get_generation_params(kwargs)
        
        output = self.model_instance(
            prompt,
            max_tokens=params.get("max_tokens", self.max_tokens),
            temperature=params.get("temperature", self.temperature),
            top_p=params.get("top_p", self.top_p),
            top_k=params.get("top_k", self.top_k),
            repeat_penalty=params.get("repeat_penalty", self.repeat_penalty),
            echo=False
        )
        
        return output["choices"][0]["text"]
    
    def _call_transformers(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """调用Transformers模型"""
        import torch
        
        params = self._get_generation_params(kwargs)
        
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # 生成参数
        generation_kwargs = {
            "max_new_tokens": params.get("max_tokens", self.max_tokens),
            "temperature": params.get("temperature", self.temperature),
            "top_p": params.get("top_p", self.top_p),
            "top_k": params.get("top_k", self.top_k),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model_instance.generate(inputs, **generation_kwargs)
        
        # 解码输出（移除输入部分）
        generated_tokens = outputs[0][len(inputs[0]):]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response
    
    def _call_custom_api(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """调用自定义API"""
        params = self._get_generation_params(kwargs)
        
        payload = {
            "prompt": prompt,
            "model": self.model_name,
            **params
        }
        
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("text", "")
    
    def _stream_ollama(self, prompt: str, kwargs: Dict[str, Any]) -> Iterator[str]:
        """Ollama流式生成"""
        params = self._get_generation_params(kwargs)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": params
        }
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    
    def _stream_llamacpp(self, prompt: str, kwargs: Dict[str, Any]) -> Iterator[str]:
        """LlamaCpp流式生成"""
        params = self._get_generation_params(kwargs)
        
        output_generator = self.model_instance(
            prompt,
            max_tokens=params.get("max_tokens", self.max_tokens),
            temperature=params.get("temperature", self.temperature),
            top_p=params.get("top_p", self.top_p),
            top_k=params.get("top_k", self.top_k),
            repeat_penalty=params.get("repeat_penalty", self.repeat_penalty),
            stream=True,
            echo=False
        )
        
        for output in output_generator:
            yield output["choices"][0]["text"]
    
    def _stream_transformers(self, prompt: str, kwargs: Dict[str, Any]) -> Iterator[str]:
        """Transformers流式生成（模拟）"""
        # Transformers原生不支持流式，我们模拟实现
        result = self._call_transformers(prompt, kwargs)
        
        # 按词分割，模拟流式输出
        words = result.split()
        for word in words:
            yield word + " "
            time.sleep(0.05)  # 模拟生成延迟
    
    def _stream_custom_api(self, prompt: str, kwargs: Dict[str, Any]) -> Iterator[str]:
        """自定义API流式生成"""
        params = self._get_generation_params(kwargs)
        
        payload = {
            "prompt": prompt,
            "model": self.model_name,
            "stream": True,
            **params
        }
        
        response = self.session.post(
            f"{self.base_url}/stream",
            json=payload,
            stream=True,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "text" in data:
                        yield data["text"]
                except json.JSONDecodeError:
                    continue
    
    def _get_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """获取生成参数"""
        return {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "repeat_penalty": kwargs.get("repeat_penalty", self.repeat_penalty)
        }
    
    def _update_statistics(self, response: LocalLLMResponse) -> None:
        """更新使用统计信息"""
        self.request_count += 1
        self.total_generation_time += response.generation_time
        self.total_tokens_generated += response.completion_tokens
        self.last_request_time = time.time()
        
        # 更新性能指标
        avg_generation_time = self.total_generation_time / self.request_count
        avg_tokens_per_second = self.total_tokens_generated / self.total_generation_time if self.total_generation_time > 0 else 0
        
        self.metrics.update({
            "total_requests": self.request_count,
            "total_generation_time": self.total_generation_time,
            "total_tokens_generated": self.total_tokens_generated,
            "error_count": self.error_count,
            "average_generation_time": avg_generation_time,
            "average_tokens_per_second": avg_tokens_per_second,
            "last_response_time": response.generation_time
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取详细的使用统计信息"""
        base_stats = self.get_metrics()
        
        avg_tokens_per_second = (
            self.total_tokens_generated / self.total_generation_time 
            if self.total_generation_time > 0 else 0
        )
        
        return {
            **base_stats,
            "backend": self.backend.value,
            "model_name": self.model_name,
            "total_requests": self.request_count,
            "total_generation_time": round(self.total_generation_time, 3),
            "total_tokens_generated": self.total_tokens_generated,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_generation_time": self.total_generation_time / max(self.request_count, 1),
            "average_tokens_per_second": round(avg_tokens_per_second, 1),
            "last_request_time": self.last_request_time
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "backend": self.backend.value,
            "model_name": self.model_name,
            "model_loaded": self._is_backend_ready()
        }
        
        if self.backend == LocalLLMBackend.OLLAMA:
            info["base_url"] = self.base_url
        elif self.backend == LocalLLMBackend.LLAMACPP:
            info["model_path"] = self.model_path
            info["n_ctx"] = self.n_ctx
            info["n_gpu_layers"] = self.n_gpu_layers
        elif self.backend == LocalLLMBackend.TRANSFORMERS:
            info["device"] = self.device
        
        return info
    
    def check_health(self) -> Dict[str, Any]:
        """检查模型健康状态"""
        health = {
            "backend_ready": self._is_backend_ready(),
            "backend": self.backend.value,
            "model": self.model_name,
            "status": "healthy" if self._is_backend_ready() else "not_ready"
        }
        
        try:
            if self.backend == LocalLLMBackend.OLLAMA and self.session:
                response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
                health["ollama_service"] = response.status_code == 200
            elif self.backend == LocalLLMBackend.LLAMACPP and self.model_instance:
                health["model_loaded"] = True
            elif self.backend == LocalLLMBackend.TRANSFORMERS and self.model_instance:
                health["model_loaded"] = True
                health["device"] = next(self.model_instance.parameters()).device.type
                
        except Exception as e:
            health["error"] = str(e)
            health["status"] = "error"
        
        return health