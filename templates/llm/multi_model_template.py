"""
多模型对比和切换模板

本模块提供了多个LLM模型的统一管理和对比功能，支持：
- 多个模型的同时配置和管理
- 模型性能对比和基准测试
- 智能负载均衡和故障转移
- 成本优化和模型路由
- A/B测试和实验管理
- 统一的接口和响应格式

设计原理：
1. 模型池管理：统一管理多个不同类型的模型实例
2. 路由策略：根据任务类型、成本、性能等因素智能路由
3. 性能监控：实时监控各模型的性能指标和健康状态
4. 故障转移：自动检测故障并切换到备用模型
5. 成本控制：根据预算和成本偏好选择最优模型
6. 实验框架：支持A/B测试和模型效果对比

使用示例：
    # 配置多个模型
    template = MultiModelTemplate()
    template.add_model("gpt-4", OpenAITemplate(), {
        "api_key": "your-key",
        "model_name": "gpt-4"
    })
    template.add_model("claude", AnthropicTemplate(), {
        "api_key": "your-key", 
        "model_name": "claude-3-sonnet-20240229"
    })
    
    # 智能路由
    result = template.run("复杂推理任务", prefer_quality=True)
    
    # 对比测试
    comparison = template.compare_models("测试问题", ["gpt-4", "claude"])
"""

import time
import asyncio
import random
from typing import Dict, Any, Optional, List, Union, Iterator, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import json

from .openai_template import OpenAITemplate, OpenAIResponse
from .anthropic_template import AnthropicTemplate, AnthropicResponse  
from .local_llm_template import LocalLLMTemplate, LocalLLMResponse
from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    APIError,
    ErrorCodes
)

logger = get_logger(__name__)


class RoutingStrategy(Enum):
    """路由策略枚举"""
    ROUND_ROBIN = "round_robin"      # 轮询
    RANDOM = "random"                # 随机
    COST_OPTIMIZED = "cost_optimized" # 成本优化
    PERFORMANCE = "performance"      # 性能优先
    QUALITY = "quality"              # 质量优先
    FASTEST = "fastest"              # 速度优先
    SMART = "smart"                  # 智能路由


@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str                                    # 模型名称
    template: TemplateBase                       # 模板实例
    setup_params: Dict[str, Any]                # 设置参数
    priority: int = 1                           # 优先级 (1-10)
    weight: float = 1.0                         # 权重
    enabled: bool = True                        # 是否启用
    max_requests_per_minute: int = 60          # 每分钟最大请求数
    cost_per_1k_tokens: float = 0.002         # 每1K tokens成本
    tags: List[str] = field(default_factory=list)  # 标签
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.tags:
            # 根据模板类型自动添加标签
            if isinstance(self.template, OpenAITemplate):
                self.tags.extend(["openai", "cloud", "commercial"])
            elif isinstance(self.template, AnthropicTemplate):
                self.tags.extend(["anthropic", "claude", "cloud", "commercial"])
            elif isinstance(self.template, LocalLLMTemplate):
                self.tags.extend(["local", "free", "private"])


@dataclass
class ModelResponse:
    """统一的模型响应数据类"""
    model_name: str                              # 模型名称
    content: str                                 # 响应内容
    response_time: float                         # 响应时间
    tokens_used: int                            # 使用的token数
    estimated_cost: float                       # 预估成本
    success: bool = True                        # 是否成功
    error_message: str = ""                     # 错误信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    @classmethod
    def from_openai_response(cls, model_name: str, response: OpenAIResponse) -> "ModelResponse":
        """从OpenAI响应创建"""
        return cls(
            model_name=model_name,
            content=response.content,
            response_time=response.response_time,
            tokens_used=response.total_tokens,
            estimated_cost=response.estimated_cost,
            metadata={
                "model": response.model,
                "finish_reason": response.finish_reason,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens
            }
        )
    
    @classmethod
    def from_anthropic_response(cls, model_name: str, response: AnthropicResponse) -> "ModelResponse":
        """从Anthropic响应创建"""
        return cls(
            model_name=model_name,
            content=response.content,
            response_time=response.response_time,
            tokens_used=response.input_tokens + response.output_tokens,
            estimated_cost=response.estimated_cost,
            metadata={
                "model": response.model,
                "stop_reason": response.stop_reason,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens
            }
        )
    
    @classmethod
    def from_local_response(cls, model_name: str, response: LocalLLMResponse) -> "ModelResponse":
        """从本地模型响应创建"""
        return cls(
            model_name=model_name,
            content=response.content,
            response_time=response.generation_time,
            tokens_used=response.total_tokens,
            estimated_cost=0.0,  # 本地模型无成本
            metadata={
                "backend": response.backend,
                "tokens_per_second": response.tokens_per_second,
                "memory_usage": response.memory_usage
            }
        )
    
    @classmethod
    def from_error(cls, model_name: str, error: Exception) -> "ModelResponse":
        """从错误创建响应"""
        return cls(
            model_name=model_name,
            content="",
            response_time=0.0,
            tokens_used=0,
            estimated_cost=0.0,
            success=False,
            error_message=str(error)
        )


@dataclass
class ComparisonResult:
    """模型对比结果数据类"""
    query: str                                   # 查询文本
    responses: List[ModelResponse]               # 各模型响应
    fastest_model: str                          # 最快模型
    cheapest_model: str                         # 最便宜模型
    best_quality_model: str                     # 最高质量模型（基于启发式评估）
    total_cost: float                           # 总成本
    comparison_time: float                      # 对比耗时
    
    def get_response_by_model(self, model_name: str) -> Optional[ModelResponse]:
        """根据模型名称获取响应"""
        for response in self.responses:
            if response.model_name == model_name:
                return response
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        successful_responses = [r for r in self.responses if r.success]
        
        if not successful_responses:
            return {"error": "No successful responses"}
        
        response_times = [r.response_time for r in successful_responses]
        costs = [r.estimated_cost for r in successful_responses]
        tokens = [r.tokens_used for r in successful_responses]
        
        return {
            "total_models": len(self.responses),
            "successful_models": len(successful_responses),
            "failed_models": len(self.responses) - len(successful_responses),
            "average_response_time": statistics.mean(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "total_cost": sum(costs),
            "average_cost": statistics.mean(costs),
            "total_tokens": sum(tokens),
            "average_tokens": statistics.mean(tokens)
        }


class MultiModelTemplate(TemplateBase[str, ModelResponse]):
    """
    多模型对比和切换模板
    
    提供多个LLM模型的统一管理、智能路由和性能对比功能。
    
    核心特性：
    1. 模型池管理：统一管理不同类型的模型实例
    2. 智能路由：根据多种策略选择最优模型
    3. 负载均衡：分散请求到不同模型，避免单点过载
    4. 故障转移：自动检测故障并切换到备用模型
    5. 性能监控：实时监控各模型的性能和健康状态
    6. 成本控制：根据预算优化模型选择
    7. A/B测试：支持模型效果对比和实验
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化多模型模板
        
        Args:
            config: 模板配置，None时使用默认配置
        """
        super().__init__(config or self._create_default_config())
        
        # 模型池
        self.models: Dict[str, ModelConfig] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
        # 路由配置
        self.routing_strategy = RoutingStrategy.SMART
        self.fallback_models: List[str] = []
        self.current_model_index = 0  # 用于轮询
        
        # 实验和测试
        self.ab_test_config: Dict[str, Any] = {}
        self.comparison_history: List[ComparisonResult] = []
        
        # 性能监控
        self.total_requests = 0
        self.successful_requests = 0
        self.total_cost = 0.0
        self.total_response_time = 0.0
        
        logger.debug("Multi-model template initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="MultiModelTemplate",
            version="1.0.0",
            description="多模型管理和对比模板，支持智能路由和性能优化",
            template_type=TemplateType.LLM,
            author="LangChain Learning Project",
            async_enabled=True,
            cache_enabled=True,
            timeout=120.0,  # 多模型可能需要更长时间
            retry_count=2
        )
        
        # 添加参数定义
        config.add_parameter("routing_strategy", str, False, "smart", "路由策略")
        config.add_parameter("fallback_models", list, False, [], "备用模型列表")
        config.add_parameter("enable_comparison", bool, False, False, "是否启用对比模式")
        config.add_parameter("max_parallel_requests", int, False, 3, "最大并发请求数")
        config.add_parameter("cost_threshold", float, False, 0.1, "成本阈值")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置模板参数
        
        Args:
            **parameters: 配置参数
            
        主要参数：
            routing_strategy (str): 路由策略
            fallback_models (list): 备用模型列表
            enable_comparison (bool): 是否启用对比模式
            max_parallel_requests (int): 最大并发请求数
            cost_threshold (float): 成本阈值
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置路由策略
            strategy_str = parameters.get("routing_strategy", "smart")
            try:
                self.routing_strategy = RoutingStrategy(strategy_str)
            except ValueError:
                raise ValidationError(f"Invalid routing strategy: {strategy_str}")
            
            # 设置其他参数
            self.fallback_models = parameters.get("fallback_models", [])
            self.enable_comparison = parameters.get("enable_comparison", False)
            self.max_parallel_requests = parameters.get("max_parallel_requests", 3)
            self.cost_threshold = parameters.get("cost_threshold", 0.1)
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            # 更新状态
            self.status = self.status.CONFIGURED
            
            logger.info(f"Multi-model template configured with strategy: {self.routing_strategy.value}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup multi-model template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def add_model(
        self, 
        name: str, 
        template: TemplateBase, 
        setup_params: Dict[str, Any],
        **model_config_kwargs
    ) -> None:
        """
        添加模型到模型池
        
        Args:
            name: 模型名称
            template: 模板实例
            setup_params: 模板设置参数
            **model_config_kwargs: 模型配置的额外参数
        """
        try:
            # 设置模板
            template.setup(**setup_params)
            
            # 创建模型配置
            model_config = ModelConfig(
                name=name,
                template=template,
                setup_params=setup_params,
                **model_config_kwargs
            )
            
            # 添加到模型池
            self.models[name] = model_config
            self.model_stats[name] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "total_response_time": 0.0,
                "total_cost": 0.0,
                "average_response_time": 0.0,
                "average_cost": 0.0,
                "last_used": 0.0,
                "health_score": 1.0
            }
            
            logger.info(f"Added model to pool: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add model {name}: {str(e)}")
            raise ConfigurationError(f"Failed to add model {name}: {str(e)}")
    
    def remove_model(self, name: str) -> None:
        """
        从模型池中移除模型
        
        Args:
            name: 模型名称
        """
        if name in self.models:
            del self.models[name]
            del self.model_stats[name]
            logger.info(f"Removed model from pool: {name}")
        else:
            logger.warning(f"Model not found in pool: {name}")
    
    def execute(self, input_data: str, **kwargs) -> ModelResponse:
        """
        执行模型调用（使用智能路由选择模型）
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
                - preferred_model: 偏好模型
                - prefer_quality: 优先质量
                - prefer_speed: 优先速度
                - prefer_cost: 优先成本
                - max_cost: 最大成本限制
                
        Returns:
            统一的模型响应对象
        """
        if not self.models:
            raise RuntimeError("No models configured. Please add models first.")
        
        try:
            # 选择模型
            selected_model = self._route_request(input_data, kwargs)
            
            # 执行调用
            result = self._call_model(selected_model, input_data, kwargs)
            
            # 更新统计信息
            self._update_model_stats(selected_model, result)
            self._update_global_stats(result)
            
            return result
            
        except Exception as e:
            # 尝试故障转移
            if hasattr(e, '__class__') and 'API' in e.__class__.__name__:
                fallback_result = self._try_fallback(input_data, kwargs)
                if fallback_result:
                    return fallback_result
            
            # 如果故障转移也失败，抛出原始异常
            raise APIError(
                f"Multi-model call failed: {str(e)}",
                error_code=ErrorCodes.API_REQUEST_FAILED,
                cause=e
            )
    
    async def execute_async(self, input_data: str, **kwargs) -> ModelResponse:
        """
        执行异步模型调用
        
        Args:
            input_data: 输入文本
            **kwargs: 额外参数
            
        Returns:
            统一的模型响应对象
        """
        if not self.models:
            raise RuntimeError("No models configured. Please add models first.")
        
        try:
            # 选择模型
            selected_model = self._route_request(input_data, kwargs)
            
            # 执行异步调用
            result = await self._call_model_async(selected_model, input_data, kwargs)
            
            # 更新统计信息
            self._update_model_stats(selected_model, result)
            self._update_global_stats(result)
            
            return result
            
        except Exception as e:
            # 尝试异步故障转移
            fallback_result = await self._try_fallback_async(input_data, kwargs)
            if fallback_result:
                return fallback_result
            
            raise APIError(
                f"Async multi-model call failed: {str(e)}",
                error_code=ErrorCodes.API_REQUEST_FAILED,
                cause=e
            )
    
    def compare_models(
        self, 
        input_data: str, 
        model_names: Optional[List[str]] = None,
        **kwargs
    ) -> ComparisonResult:
        """
        对比多个模型的性能
        
        Args:
            input_data: 输入文本
            model_names: 要对比的模型名称列表，None表示对比所有模型
            **kwargs: 额外参数
            
        Returns:
            对比结果
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        # 过滤有效的模型
        valid_models = [name for name in model_names if name in self.models and self.models[name].enabled]
        
        if not valid_models:
            raise ValueError("No valid models found for comparison")
        
        start_time = time.time()
        responses = []
        
        # 并发执行多个模型
        with ThreadPoolExecutor(max_workers=min(len(valid_models), self.max_parallel_requests)) as executor:
            future_to_model = {
                executor.submit(self._call_model, model_name, input_data, kwargs): model_name
                for model_name in valid_models
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    responses.append(result)
                    self._update_model_stats(model_name, result)
                except Exception as e:
                    error_response = ModelResponse.from_error(model_name, e)
                    responses.append(error_response)
                    logger.error(f"Model {model_name} failed during comparison: {str(e)}")
        
        comparison_time = time.time() - start_time
        
        # 分析结果
        successful_responses = [r for r in responses if r.success]
        
        if not successful_responses:
            raise APIError("All models failed during comparison")
        
        # 找出最优模型
        fastest_model = min(successful_responses, key=lambda r: r.response_time).model_name
        cheapest_model = min(successful_responses, key=lambda r: r.estimated_cost).model_name
        
        # 质量评估（简单的启发式方法）
        best_quality_model = self._evaluate_quality(successful_responses)
        
        # 创建对比结果
        comparison_result = ComparisonResult(
            query=input_data,
            responses=responses,
            fastest_model=fastest_model,
            cheapest_model=cheapest_model,
            best_quality_model=best_quality_model,
            total_cost=sum(r.estimated_cost for r in successful_responses),
            comparison_time=comparison_time
        )
        
        # 保存对比历史
        self.comparison_history.append(comparison_result)
        if len(self.comparison_history) > 100:  # 保持历史记录在合理范围
            self.comparison_history = self.comparison_history[-50:]
        
        logger.info(f"Model comparison completed: {len(successful_responses)}/{len(responses)} successful")
        
        return comparison_result
    
    async def compare_models_async(
        self, 
        input_data: str, 
        model_names: Optional[List[str]] = None,
        **kwargs
    ) -> ComparisonResult:
        """
        异步对比多个模型的性能
        
        Args:
            input_data: 输入文本
            model_names: 要对比的模型名称列表
            **kwargs: 额外参数
            
        Returns:
            对比结果
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        valid_models = [name for name in model_names if name in self.models and self.models[name].enabled]
        
        if not valid_models:
            raise ValueError("No valid models found for comparison")
        
        start_time = time.time()
        
        # 并发执行异步调用
        tasks = []
        for model_name in valid_models:
            task = self._call_model_async(model_name, input_data, kwargs)
            tasks.append((model_name, task))
        
        responses = []
        for model_name, task in tasks:
            try:
                result = await task
                responses.append(result)
                self._update_model_stats(model_name, result)
            except Exception as e:
                error_response = ModelResponse.from_error(model_name, e)
                responses.append(error_response)
                logger.error(f"Model {model_name} failed during async comparison: {str(e)}")
        
        comparison_time = time.time() - start_time
        
        # 分析结果（与同步版本相同的逻辑）
        successful_responses = [r for r in responses if r.success]
        
        if not successful_responses:
            raise APIError("All models failed during async comparison")
        
        fastest_model = min(successful_responses, key=lambda r: r.response_time).model_name
        cheapest_model = min(successful_responses, key=lambda r: r.estimated_cost).model_name
        best_quality_model = self._evaluate_quality(successful_responses)
        
        comparison_result = ComparisonResult(
            query=input_data,
            responses=responses,
            fastest_model=fastest_model,
            cheapest_model=cheapest_model,
            best_quality_model=best_quality_model,
            total_cost=sum(r.estimated_cost for r in successful_responses),
            comparison_time=comparison_time
        )
        
        self.comparison_history.append(comparison_result)
        if len(self.comparison_history) > 100:
            self.comparison_history = self.comparison_history[-50:]
        
        logger.info(f"Async model comparison completed: {len(successful_responses)}/{len(responses)} successful")
        
        return comparison_result
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "多模型管理和对比模板使用示例",
            "setup_parameters": {
                "routing_strategy": "smart",
                "fallback_models": ["gpt-3.5-turbo", "local-llama"],
                "enable_comparison": True,
                "max_parallel_requests": 3
            },
            "model_configuration": [
                {
                    "name": "gpt-4",
                    "template": "OpenAITemplate",
                    "setup_params": {
                        "api_key": "${OPENAI_API_KEY}",
                        "model_name": "gpt-4"
                    },
                    "priority": 3,
                    "cost_per_1k_tokens": 0.06
                },
                {
                    "name": "claude",
                    "template": "AnthropicTemplate",
                    "setup_params": {
                        "api_key": "${ANTHROPIC_API_KEY}",
                        "model_name": "claude-3-sonnet-20240229"
                    },
                    "priority": 2,
                    "cost_per_1k_tokens": 0.015
                }
            ],
            "usage_code": '''
# 配置多模型系统
from templates.llm import MultiModelTemplate, OpenAITemplate, AnthropicTemplate

template = MultiModelTemplate()
template.setup(
    routing_strategy="smart",
    fallback_models=["claude", "local-llama"]
)

# 添加模型
template.add_model("gpt-4", OpenAITemplate(), {
    "api_key": "your-key",
    "model_name": "gpt-4"
}, priority=3, cost_per_1k_tokens=0.06)

template.add_model("claude", AnthropicTemplate(), {
    "api_key": "your-key",
    "model_name": "claude-3-sonnet-20240229"
}, priority=2, cost_per_1k_tokens=0.015)

# 智能路由调用
result = template.run("复杂的推理问题", prefer_quality=True)
print(f"使用模型: {result.model_name}")
print(f"响应: {result.content}")

# 模型对比
comparison = template.compare_models("测试问题")
print(f"最快模型: {comparison.fastest_model}")
print(f"最便宜模型: {comparison.cheapest_model}")
print(f"最优质量模型: {comparison.best_quality_model}")

# 获取统计信息
stats = template.get_model_statistics()
for model_name, stat in stats.items():
    print(f"{model_name}: 成功率 {stat['success_rate']:.2%}")
'''
        }
    
    def _route_request(self, input_data: str, kwargs: Dict[str, Any]) -> str:
        """
        根据路由策略选择模型
        
        Args:
            input_data: 输入数据
            kwargs: 额外参数
            
        Returns:
            选中的模型名称
        """
        available_models = [
            name for name, config in self.models.items() 
            if config.enabled and self._is_model_healthy(name)
        ]
        
        if not available_models:
            raise RuntimeError("No healthy models available")
        
        # 检查是否指定了偏好模型
        preferred_model = kwargs.get("preferred_model")
        if preferred_model and preferred_model in available_models:
            return preferred_model
        
        # 根据路由策略选择
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_models)
        elif self.routing_strategy == RoutingStrategy.RANDOM:
            return random.choice(available_models)
        elif self.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(available_models, kwargs)
        elif self.routing_strategy == RoutingStrategy.PERFORMANCE:
            return self._performance_select(available_models)
        elif self.routing_strategy == RoutingStrategy.QUALITY:
            return self._quality_select(available_models)
        elif self.routing_strategy == RoutingStrategy.FASTEST:
            return self._fastest_select(available_models)
        elif self.routing_strategy == RoutingStrategy.SMART:
            return self._smart_select(available_models, input_data, kwargs)
        else:
            return available_models[0]
    
    def _round_robin_select(self, available_models: List[str]) -> str:
        """轮询选择模型"""
        model = available_models[self.current_model_index % len(available_models)]
        self.current_model_index += 1
        return model
    
    def _cost_optimized_select(self, available_models: List[str], kwargs: Dict[str, Any]) -> str:
        """成本优化选择"""
        max_cost = kwargs.get("max_cost", self.cost_threshold)
        
        # 过滤超出成本限制的模型
        affordable_models = [
            name for name in available_models
            if self.models[name].cost_per_1k_tokens <= max_cost
        ]
        
        if not affordable_models:
            affordable_models = available_models  # 如果都超出限制，使用最便宜的
        
        # 选择最便宜的模型
        return min(affordable_models, key=lambda name: self.models[name].cost_per_1k_tokens)
    
    def _performance_select(self, available_models: List[str]) -> str:
        """性能优化选择"""
        # 根据平均响应时间选择最快的模型
        model_scores = {}
        for name in available_models:
            stats = self.model_stats[name]
            if stats["requests"] > 0:
                model_scores[name] = stats["average_response_time"]
            else:
                model_scores[name] = float('inf')  # 未使用过的模型优先级最低
        
        return min(model_scores.keys(), key=lambda name: model_scores[name])
    
    def _quality_select(self, available_models: List[str]) -> str:
        """质量优化选择"""
        # 根据优先级和健康分数选择最优质的模型
        model_scores = {}
        for name in available_models:
            priority = self.models[name].priority
            health_score = self.model_stats[name]["health_score"]
            model_scores[name] = priority * health_score
        
        return max(model_scores.keys(), key=lambda name: model_scores[name])
    
    def _fastest_select(self, available_models: List[str]) -> str:
        """速度优先选择"""
        return self._performance_select(available_models)
    
    def _smart_select(self, available_models: List[str], input_data: str, kwargs: Dict[str, Any]) -> str:
        """智能选择（综合考虑多个因素）"""
        # 分析请求特征
        input_length = len(input_data)
        prefer_quality = kwargs.get("prefer_quality", False)
        prefer_speed = kwargs.get("prefer_speed", False)
        prefer_cost = kwargs.get("prefer_cost", False)
        
        model_scores = {}
        
        for name in available_models:
            config = self.models[name]
            stats = self.model_stats[name]
            
            # 基础分数（优先级和健康分数）
            score = config.priority * stats["health_score"]
            
            # 根据偏好调整分数
            if prefer_quality:
                score *= 1.5 if config.priority >= 3 else 1.0
            
            if prefer_speed:
                if stats["requests"] > 0:
                    avg_time = stats["average_response_time"]
                    score *= 2.0 if avg_time < 2.0 else (1.0 if avg_time < 5.0 else 0.5)
            
            if prefer_cost:
                cost = config.cost_per_1k_tokens
                score *= 2.0 if cost < 0.01 else (1.0 if cost < 0.05 else 0.5)
            
            # 根据输入长度调整（长输入偏向高质量模型）
            if input_length > 1000:
                score *= 1.2 if config.priority >= 3 else 0.8
            
            model_scores[name] = score
        
        return max(model_scores.keys(), key=lambda name: model_scores[name])
    
    def _call_model(self, model_name: str, input_data: str, kwargs: Dict[str, Any]) -> ModelResponse:
        """
        调用指定模型
        
        Args:
            model_name: 模型名称
            input_data: 输入数据
            kwargs: 额外参数
            
        Returns:
            统一的模型响应
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        template = self.models[model_name].template
        
        try:
            # 根据模板类型调用相应方法
            if isinstance(template, OpenAITemplate):
                response = template.execute(input_data, **kwargs)
                return ModelResponse.from_openai_response(model_name, response)
            elif isinstance(template, AnthropicTemplate):
                response = template.execute(input_data, **kwargs)
                return ModelResponse.from_anthropic_response(model_name, response)
            elif isinstance(template, LocalLLMTemplate):
                response = template.execute(input_data, **kwargs)
                return ModelResponse.from_local_response(model_name, response)
            else:
                # 通用模板调用
                response = template.execute(input_data, **kwargs)
                return ModelResponse(
                    model_name=model_name,
                    content=str(response),
                    response_time=0.0,
                    tokens_used=0,
                    estimated_cost=0.0
                )
                
        except Exception as e:
            return ModelResponse.from_error(model_name, e)
    
    async def _call_model_async(self, model_name: str, input_data: str, kwargs: Dict[str, Any]) -> ModelResponse:
        """异步调用指定模型"""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        template = self.models[model_name].template
        
        try:
            # 根据模板类型调用相应的异步方法
            if isinstance(template, OpenAITemplate):
                response = await template.execute_async(input_data, **kwargs)
                return ModelResponse.from_openai_response(model_name, response)
            elif isinstance(template, AnthropicTemplate):
                response = await template.execute_async(input_data, **kwargs)
                return ModelResponse.from_anthropic_response(model_name, response)
            elif isinstance(template, LocalLLMTemplate):
                response = await template.execute_async(input_data, **kwargs)
                return ModelResponse.from_local_response(model_name, response)
            else:
                # 通用模板异步调用
                response = await template.execute_async(input_data, **kwargs)
                return ModelResponse(
                    model_name=model_name,
                    content=str(response),
                    response_time=0.0,
                    tokens_used=0,
                    estimated_cost=0.0
                )
                
        except Exception as e:
            return ModelResponse.from_error(model_name, e)
    
    def _try_fallback(self, input_data: str, kwargs: Dict[str, Any]) -> Optional[ModelResponse]:
        """尝试故障转移"""
        for fallback_model in self.fallback_models:
            if fallback_model in self.models and self._is_model_healthy(fallback_model):
                try:
                    logger.warning(f"Attempting fallback to model: {fallback_model}")
                    result = self._call_model(fallback_model, input_data, kwargs)
                    if result.success:
                        self._update_model_stats(fallback_model, result)
                        return result
                except Exception as e:
                    logger.error(f"Fallback model {fallback_model} also failed: {str(e)}")
                    continue
        
        return None
    
    async def _try_fallback_async(self, input_data: str, kwargs: Dict[str, Any]) -> Optional[ModelResponse]:
        """异步故障转移"""
        for fallback_model in self.fallback_models:
            if fallback_model in self.models and self._is_model_healthy(fallback_model):
                try:
                    logger.warning(f"Attempting async fallback to model: {fallback_model}")
                    result = await self._call_model_async(fallback_model, input_data, kwargs)
                    if result.success:
                        self._update_model_stats(fallback_model, result)
                        return result
                except Exception as e:
                    logger.error(f"Async fallback model {fallback_model} also failed: {str(e)}")
                    continue
        
        return None
    
    def _is_model_healthy(self, model_name: str) -> bool:
        """检查模型健康状态"""
        if model_name not in self.model_stats:
            return True  # 新模型默认健康
        
        stats = self.model_stats[model_name]
        
        # 如果最近没有请求，认为是健康的
        if stats["requests"] == 0:
            return True
        
        # 计算成功率
        success_rate = stats["successes"] / stats["requests"]
        
        # 健康阈值：成功率 > 50%
        return success_rate > 0.5
    
    def _evaluate_quality(self, responses: List[ModelResponse]) -> str:
        """
        评估响应质量（简单的启发式方法）
        
        Args:
            responses: 成功的响应列表
            
        Returns:
            最高质量模型的名称
        """
        quality_scores = {}
        
        for response in responses:
            score = 0
            
            # 响应长度得分（适中长度得分较高）
            content_length = len(response.content)
            if 100 <= content_length <= 2000:
                score += 3
            elif content_length > 50:
                score += 2
            else:
                score += 1
            
            # 模型优先级得分
            if response.model_name in self.models:
                priority = self.models[response.model_name].priority
                score += priority
            
            # 响应时间得分（适中响应时间得分较高）
            if 1.0 <= response.response_time <= 10.0:
                score += 2
            elif response.response_time <= 20.0:
                score += 1
            
            quality_scores[response.model_name] = score
        
        return max(quality_scores.keys(), key=lambda name: quality_scores[name])
    
    def _update_model_stats(self, model_name: str, response: ModelResponse) -> None:
        """更新模型统计信息"""
        if model_name not in self.model_stats:
            return
        
        stats = self.model_stats[model_name]
        stats["requests"] += 1
        stats["last_used"] = time.time()
        
        if response.success:
            stats["successes"] += 1
            stats["total_response_time"] += response.response_time
            stats["total_cost"] += response.estimated_cost
            
            # 更新平均值
            stats["average_response_time"] = stats["total_response_time"] / stats["successes"]
            stats["average_cost"] = stats["total_cost"] / stats["successes"]
            
            # 更新健康分数
            success_rate = stats["successes"] / stats["requests"]
            stats["health_score"] = min(1.0, success_rate * 1.2)  # 给予一定容错性
        else:
            stats["failures"] += 1
            # 降低健康分数
            success_rate = stats["successes"] / stats["requests"]
            stats["health_score"] = max(0.1, success_rate)
    
    def _update_global_stats(self, response: ModelResponse) -> None:
        """更新全局统计信息"""
        self.total_requests += 1
        
        if response.success:
            self.successful_requests += 1
            self.total_response_time += response.response_time
            self.total_cost += response.estimated_cost
    
    def get_model_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型的统计信息"""
        result = {}
        
        for model_name, stats in self.model_stats.items():
            model_stats = stats.copy()
            if stats["requests"] > 0:
                model_stats["success_rate"] = stats["successes"] / stats["requests"]
                model_stats["failure_rate"] = stats["failures"] / stats["requests"]
            else:
                model_stats["success_rate"] = 0.0
                model_stats["failure_rate"] = 0.0
            
            result[model_name] = model_stats
        
        return result
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        avg_response_time = (
            self.total_response_time / self.successful_requests
            if self.successful_requests > 0 else 0.0
        )
        
        avg_cost_per_request = (
            self.total_cost / self.successful_requests
            if self.successful_requests > 0 else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.total_requests - self.successful_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "total_cost": round(self.total_cost, 6),
            "average_response_time": round(avg_response_time, 3),
            "average_cost_per_request": round(avg_cost_per_request, 6),
            "active_models": len([name for name, config in self.models.items() if config.enabled]),
            "total_models": len(self.models),
            "routing_strategy": self.routing_strategy.value
        }
    
    def get_comparison_history(self, limit: int = 10) -> List[ComparisonResult]:
        """获取对比历史记录"""
        return self.comparison_history[-limit:] if self.comparison_history else []