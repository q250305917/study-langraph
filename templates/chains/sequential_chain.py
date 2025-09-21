"""
顺序链模板（SequentialChainTemplate）

本模块提供了顺序执行的链模板，支持多个步骤依次执行的工作流。
这是Chain模板系统的核心组件之一，适用于需要步骤化处理的复杂任务。

核心特性：
1. 步骤依次执行：严格按照定义顺序执行各个步骤
2. 数据流传递：前一步的输出作为下一步的输入
3. 错误处理：任何一步失败都会中断整个链的执行
4. 状态管理：跟踪每个步骤的执行状态和结果
5. 动态配置：支持运行时动态调整步骤配置
6. 性能监控：监控每个步骤的执行时间和资源使用

设计原理：
- 责任链模式：每个步骤处理特定的任务，形成处理链条
- 流水线模式：数据在步骤间顺序流动和转换
- 状态机模式：管理链的执行状态转换
- 策略模式：支持不同类型的执行步骤
- 观察者模式：监控步骤执行过程和状态变化

使用场景：
- 文档处理：分段加载→分割→向量化→存储
- 内容生成：主题分析→大纲生成→内容写作→后处理
- 数据分析：数据清洗→特征提取→模型训练→结果评估
- 工作流自动化：表单验证→业务处理→结果通知→日志记录
- 多步推理：问题分解→子问题求解→结果合并→答案生成
"""

import asyncio
import time
import copy
import uuid
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from pathlib import Path

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)

# 泛型类型定义
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
StepInputType = TypeVar('StepInputType')
StepOutputType = TypeVar('StepOutputType')


class StepStatus(Enum):
    """步骤执行状态枚举"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 执行完成
    FAILED = "failed"         # 执行失败
    SKIPPED = "skipped"       # 跳过执行
    CANCELLED = "cancelled"   # 取消执行


class ErrorHandlingStrategy(Enum):
    """错误处理策略枚举"""
    FAIL_FAST = "fail_fast"           # 快速失败，立即停止
    CONTINUE_ON_ERROR = "continue"    # 继续执行，记录错误
    RETRY_ON_ERROR = "retry"          # 重试失败的步骤
    SKIP_ON_ERROR = "skip"            # 跳过失败的步骤


@dataclass
class StepConfig:
    """
    步骤配置类
    
    定义链中每个步骤的配置信息，包括名称、描述、执行函数、
    参数、依赖关系、错误处理策略等。
    """
    
    # === 基本信息 ===
    name: str                                    # 步骤名称
    description: str = ""                        # 步骤描述
    step_id: Optional[str] = None               # 步骤唯一标识
    
    # === 执行配置 ===
    executor: Optional[Callable] = None          # 执行函数
    template: Optional[TemplateBase] = None      # 模板实例
    
    # === 参数配置 ===
    input_keys: List[str] = field(default_factory=list)    # 输入参数键
    output_keys: List[str] = field(default_factory=list)   # 输出参数键
    parameters: Dict[str, Any] = field(default_factory=dict)  # 固定参数
    
    # === 依赖关系 ===
    dependencies: List[str] = field(default_factory=list)   # 依赖的步骤ID
    condition: Optional[Callable] = None         # 执行条件函数
    
    # === 执行控制 ===
    timeout: Optional[float] = None              # 超时时间（秒）
    retry_count: int = 0                         # 重试次数
    retry_delay: float = 1.0                     # 重试延迟（秒）
    
    # === 错误处理 ===
    error_strategy: ErrorHandlingStrategy = ErrorHandlingStrategy.FAIL_FAST
    error_handler: Optional[Callable] = None     # 自定义错误处理函数
    
    # === 性能配置 ===
    cache_enabled: bool = False                  # 是否启用缓存
    cache_key_generator: Optional[Callable] = None  # 缓存键生成函数
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.step_id:
            self.step_id = f"step_{uuid.uuid4().hex[:8]}"
        
        # 验证配置有效性
        if not self.executor and not self.template:
            raise ConfigurationError("Step must have either executor or template")
        
        if self.executor and self.template:
            raise ConfigurationError("Step cannot have both executor and template")


@dataclass
class StepResult:
    """
    步骤执行结果类
    
    包含步骤执行的完整信息，包括状态、输入输出、
    执行时间、错误信息等。
    """
    
    # === 基本信息 ===
    step_id: str                                 # 步骤ID
    step_name: str                               # 步骤名称
    status: StepStatus                           # 执行状态
    
    # === 执行信息 ===
    start_time: Optional[float] = None           # 开始时间
    end_time: Optional[float] = None             # 结束时间
    execution_time: Optional[float] = None       # 执行时长
    
    # === 数据信息 ===
    input_data: Any = None                       # 输入数据
    output_data: Any = None                      # 输出数据
    parameters: Dict[str, Any] = field(default_factory=dict)  # 执行参数
    
    # === 错误信息 ===
    error: Optional[Exception] = None            # 错误异常
    error_message: Optional[str] = None          # 错误消息
    traceback: Optional[str] = None              # 错误堆栈
    
    # === 性能信息 ===
    memory_usage: Optional[float] = None         # 内存使用量
    cache_hit: bool = False                      # 是否命中缓存
    
    def is_successful(self) -> bool:
        """判断步骤是否执行成功"""
        return self.status == StepStatus.COMPLETED
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "success": self.is_successful(),
            "cache_hit": self.cache_hit,
            "error_message": self.error_message
        }


class ChainContext:
    """
    链执行上下文
    
    管理链执行过程中的数据流、状态信息和共享资源。
    提供数据传递、状态跟踪、错误处理等核心功能。
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """
        初始化链上下文
        
        Args:
            initial_data: 初始数据
        """
        self.data: Dict[str, Any] = initial_data or {}
        self.step_results: Dict[str, StepResult] = {}
        self.shared_resources: Dict[str, Any] = {}
        self.execution_metadata: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # 执行统计
        self.total_steps = 0
        self.completed_steps = 0
        self.failed_steps = 0
        self.skipped_steps = 0
        
        logger.debug("Initialized ChainContext")
    
    def set_data(self, key: str, value: Any) -> None:
        """设置数据"""
        with self._lock:
            self.data[key] = value
            logger.debug(f"Set context data: {key}")
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """获取数据"""
        with self._lock:
            return self.data.get(key, default)
    
    def update_data(self, data: Dict[str, Any]) -> None:
        """批量更新数据"""
        with self._lock:
            self.data.update(data)
            logger.debug(f"Updated context data with {len(data)} items")
    
    def add_step_result(self, result: StepResult) -> None:
        """添加步骤结果"""
        with self._lock:
            self.step_results[result.step_id] = result
            
            # 更新执行统计
            if result.status == StepStatus.COMPLETED:
                self.completed_steps += 1
            elif result.status == StepStatus.FAILED:
                self.failed_steps += 1
            elif result.status == StepStatus.SKIPPED:
                self.skipped_steps += 1
            
            logger.debug(f"Added step result: {result.step_name} ({result.status.value})")
    
    def get_step_result(self, step_id: str) -> Optional[StepResult]:
        """获取步骤结果"""
        with self._lock:
            return self.step_results.get(step_id)
    
    def get_step_output(self, step_id: str) -> Any:
        """获取步骤输出数据"""
        result = self.get_step_result(step_id)
        return result.output_data if result else None
    
    def has_failed_steps(self) -> bool:
        """检查是否有失败的步骤"""
        with self._lock:
            return self.failed_steps > 0
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        with self._lock:
            return {
                "total_steps": self.total_steps,
                "completed_steps": self.completed_steps,
                "failed_steps": self.failed_steps,
                "skipped_steps": self.skipped_steps,
                "success_rate": self.completed_steps / self.total_steps if self.total_steps > 0 else 0,
                "data_keys": list(self.data.keys()),
                "step_results": [result.get_execution_summary() for result in self.step_results.values()]
            }


class SequentialChainTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    顺序链模板类
    
    实现多个步骤依次执行的工作流。每个步骤的输出会传递给下一个步骤，
    形成完整的数据处理链条。
    
    核心功能：
    1. 步骤管理：动态添加、删除、修改步骤
    2. 执行控制：支持暂停、恢复、取消执行
    3. 错误处理：灵活的错误处理策略
    4. 性能优化：缓存、并行优化等
    5. 状态监控：实时监控执行状态和进度
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化顺序链模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config)
        
        # 步骤配置
        self.steps: List[StepConfig] = []
        self.step_cache: Dict[str, Any] = {}
        
        # 执行控制
        self.context: Optional[ChainContext] = None
        self.is_cancelled = False
        self.is_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始为非暂停状态
        
        # 性能配置
        self.enable_caching = True
        self.max_cache_size = 1000
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        
        logger.info(f"Initialized SequentialChainTemplate: {self.config.name}")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="SequentialChainTemplate",
            version="1.0.0",
            description="顺序执行的链模板，支持多步骤工作流",
            template_type=TemplateType.CHAIN,
            async_enabled=True,
            cache_enabled=True
        )
        
        # 定义参数模式
        config.add_parameter(
            "steps", list, required=True,
            description="链步骤配置列表",
            examples=[
                [
                    {"name": "step1", "executor": "function1"},
                    {"name": "step2", "executor": "function2"}
                ]
            ]
        )
        
        config.add_parameter(
            "error_strategy", str, required=False, default="fail_fast",
            description="错误处理策略",
            constraints={"allowed_values": ["fail_fast", "continue", "retry", "skip"]}
        )
        
        config.add_parameter(
            "enable_caching", bool, required=False, default=True,
            description="是否启用步骤结果缓存"
        )
        
        config.add_parameter(
            "timeout", float, required=False,
            description="整个链的超时时间（秒）",
            constraints={"min_value": 0}
        )
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置链参数
        
        Args:
            **parameters: 链配置参数
                - steps: 步骤配置列表
                - error_strategy: 错误处理策略
                - enable_caching: 是否启用缓存
                - timeout: 超时时间
        
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        # 验证参数
        self.validate_parameters(parameters)
        
        # 保存设置参数
        self._setup_parameters = parameters.copy()
        
        # 配置步骤
        steps_config = parameters.get("steps", [])
        self._configure_steps(steps_config)
        
        # 配置错误处理策略
        error_strategy = parameters.get("error_strategy", "fail_fast")
        self.default_error_strategy = ErrorHandlingStrategy(error_strategy)
        
        # 配置缓存
        self.enable_caching = parameters.get("enable_caching", True)
        
        # 配置超时
        self.chain_timeout = parameters.get("timeout")
        
        # 配置线程池
        max_workers = parameters.get("max_workers", 4)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # 更新状态
        self.status = self.status.__class__.CONFIGURED
        
        logger.info(f"Configured SequentialChain with {len(self.steps)} steps")
    
    def _configure_steps(self, steps_config: List[Dict[str, Any]]) -> None:
        """
        配置步骤列表
        
        Args:
            steps_config: 步骤配置列表
        """
        self.steps.clear()
        
        for i, step_dict in enumerate(steps_config):
            try:
                # 创建步骤配置
                step_config = self._create_step_config(step_dict, i)
                self.steps.append(step_config)
                
                logger.debug(f"Configured step: {step_config.name}")
                
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to configure step {i}: {str(e)}",
                    error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                    cause=e
                )
    
    def _create_step_config(self, step_dict: Dict[str, Any], index: int) -> StepConfig:
        """
        创建步骤配置
        
        Args:
            step_dict: 步骤字典配置
            index: 步骤索引
        
        Returns:
            步骤配置对象
        """
        # 基本信息
        name = step_dict.get("name", f"step_{index}")
        description = step_dict.get("description", "")
        step_id = step_dict.get("step_id", f"step_{index}_{uuid.uuid4().hex[:8]}")
        
        # 执行器配置
        executor = step_dict.get("executor")
        template = step_dict.get("template")
        
        # 参数配置
        input_keys = step_dict.get("input_keys", [])
        output_keys = step_dict.get("output_keys", [])
        parameters = step_dict.get("parameters", {})
        
        # 依赖关系
        dependencies = step_dict.get("dependencies", [])
        condition = step_dict.get("condition")
        
        # 执行控制
        timeout = step_dict.get("timeout")
        retry_count = step_dict.get("retry_count", 0)
        retry_delay = step_dict.get("retry_delay", 1.0)
        
        # 错误处理
        error_strategy_str = step_dict.get("error_strategy", self.default_error_strategy.value)
        error_strategy = ErrorHandlingStrategy(error_strategy_str)
        error_handler = step_dict.get("error_handler")
        
        # 缓存配置
        cache_enabled = step_dict.get("cache_enabled", False)
        cache_key_generator = step_dict.get("cache_key_generator")
        
        return StepConfig(
            name=name,
            description=description,
            step_id=step_id,
            executor=executor,
            template=template,
            input_keys=input_keys,
            output_keys=output_keys,
            parameters=parameters,
            dependencies=dependencies,
            condition=condition,
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            error_strategy=error_strategy,
            error_handler=error_handler,
            cache_enabled=cache_enabled,
            cache_key_generator=cache_key_generator
        )
    
    def add_step(
        self,
        name: str,
        executor: Optional[Callable] = None,
        template: Optional[TemplateBase] = None,
        **kwargs
    ) -> str:
        """
        添加步骤
        
        Args:
            name: 步骤名称
            executor: 执行函数
            template: 模板实例
            **kwargs: 其他步骤配置参数
        
        Returns:
            步骤ID
        """
        step_config = StepConfig(
            name=name,
            executor=executor,
            template=template,
            **kwargs
        )
        
        self.steps.append(step_config)
        logger.info(f"Added step: {name} ({step_config.step_id})")
        
        return step_config.step_id
    
    def remove_step(self, step_id: str) -> bool:
        """
        移除步骤
        
        Args:
            step_id: 步骤ID
        
        Returns:
            是否移除成功
        """
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                removed_step = self.steps.pop(i)
                logger.info(f"Removed step: {removed_step.name} ({step_id})")
                return True
        
        logger.warning(f"Step not found for removal: {step_id}")
        return False
    
    def get_step(self, step_id: str) -> Optional[StepConfig]:
        """
        获取步骤配置
        
        Args:
            step_id: 步骤ID
        
        Returns:
            步骤配置，如果不存在则返回None
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行顺序链（同步版本）
        
        Args:
            input_data: 输入数据
            **kwargs: 执行参数
        
        Returns:
            执行结果
        
        Raises:
            ValidationError: 输入验证失败
            ConfigurationError: 配置错误
            Exception: 执行过程中的异常
        """
        if not self.steps:
            raise ConfigurationError("No steps configured for execution")
        
        # 初始化执行上下文
        self.context = ChainContext(input_data)
        self.context.total_steps = len(self.steps)
        self.is_cancelled = False
        self.is_paused = False
        
        logger.info(f"Starting sequential chain execution with {len(self.steps)} steps")
        
        try:
            # 依次执行每个步骤
            for step in self.steps:
                # 检查取消状态
                if self.is_cancelled:
                    logger.info("Chain execution cancelled")
                    break
                
                # 检查暂停状态
                self._pause_event.wait()
                
                # 执行步骤
                step_result = self._execute_step(step, self.context)
                
                # 处理步骤结果
                if step_result.status == StepStatus.FAILED:
                    if step.error_strategy == ErrorHandlingStrategy.FAIL_FAST:
                        raise Exception(f"Step '{step.name}' failed: {step_result.error_message}")
                    elif step.error_strategy == ErrorHandlingStrategy.CONTINUE_ON_ERROR:
                        logger.warning(f"Step '{step.name}' failed, continuing: {step_result.error_message}")
                        continue
                    elif step.error_strategy == ErrorHandlingStrategy.SKIP_ON_ERROR:
                        logger.warning(f"Step '{step.name}' failed, skipping: {step_result.error_message}")
                        continue
                
                # 更新上下文数据
                if step_result.output_data is not None:
                    if isinstance(step_result.output_data, dict):
                        self.context.update_data(step_result.output_data)
                    else:
                        # 如果输出不是字典，使用步骤名作为键
                        self.context.set_data(step.name, step_result.output_data)
            
            # 返回最终结果
            final_result = {
                "status": "completed" if not self.context.has_failed_steps() else "partial",
                "data": self.context.data.copy(),
                "summary": self.context.get_execution_summary()
            }
            
            logger.info(f"Sequential chain execution completed")
            return final_result
            
        except Exception as e:
            logger.error(f"Sequential chain execution failed: {str(e)}")
            
            # 构建错误结果
            error_result = {
                "status": "failed",
                "error": str(e),
                "data": self.context.data.copy() if self.context else {},
                "summary": self.context.get_execution_summary() if self.context else {}
            }
            
            raise Exception(f"Chain execution failed: {str(e)}") from e
    
    async def execute_async(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行顺序链（异步版本）
        
        Args:
            input_data: 输入数据
            **kwargs: 执行参数
        
        Returns:
            执行结果
        """
        if not self.steps:
            raise ConfigurationError("No steps configured for execution")
        
        # 初始化执行上下文
        self.context = ChainContext(input_data)
        self.context.total_steps = len(self.steps)
        self.is_cancelled = False
        self.is_paused = False
        
        logger.info(f"Starting async sequential chain execution with {len(self.steps)} steps")
        
        try:
            # 依次执行每个步骤
            for step in self.steps:
                # 检查取消状态
                if self.is_cancelled:
                    logger.info("Async chain execution cancelled")
                    break
                
                # 检查暂停状态
                while self.is_paused:
                    await asyncio.sleep(0.1)
                
                # 异步执行步骤
                step_result = await self._execute_step_async(step, self.context)
                
                # 处理步骤结果（与同步版本相同的逻辑）
                if step_result.status == StepStatus.FAILED:
                    if step.error_strategy == ErrorHandlingStrategy.FAIL_FAST:
                        raise Exception(f"Step '{step.name}' failed: {step_result.error_message}")
                    elif step.error_strategy == ErrorHandlingStrategy.CONTINUE_ON_ERROR:
                        logger.warning(f"Step '{step.name}' failed, continuing: {step_result.error_message}")
                        continue
                    elif step.error_strategy == ErrorHandlingStrategy.SKIP_ON_ERROR:
                        logger.warning(f"Step '{step.name}' failed, skipping: {step_result.error_message}")
                        continue
                
                # 更新上下文数据
                if step_result.output_data is not None:
                    if isinstance(step_result.output_data, dict):
                        self.context.update_data(step_result.output_data)
                    else:
                        self.context.set_data(step.name, step_result.output_data)
            
            # 返回最终结果
            final_result = {
                "status": "completed" if not self.context.has_failed_steps() else "partial",
                "data": self.context.data.copy(),
                "summary": self.context.get_execution_summary()
            }
            
            logger.info(f"Async sequential chain execution completed")
            return final_result
            
        except Exception as e:
            logger.error(f"Async sequential chain execution failed: {str(e)}")
            
            error_result = {
                "status": "failed",
                "error": str(e),
                "data": self.context.data.copy() if self.context else {},
                "summary": self.context.get_execution_summary() if self.context else {}
            }
            
            raise Exception(f"Async chain execution failed: {str(e)}") from e
    
    def _execute_step(self, step: StepConfig, context: ChainContext) -> StepResult:
        """
        执行单个步骤（同步版本）
        
        Args:
            step: 步骤配置
            context: 执行上下文
        
        Returns:
            步骤执行结果
        """
        result = StepResult(
            step_id=step.step_id,
            step_name=step.name,
            status=StepStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            logger.debug(f"Executing step: {step.name}")
            result.status = StepStatus.RUNNING
            
            # 检查执行条件
            if step.condition and not step.condition(context):
                logger.info(f"Step condition not met, skipping: {step.name}")
                result.status = StepStatus.SKIPPED
                result.end_time = time.time()
                result.execution_time = result.end_time - result.start_time
                context.add_step_result(result)
                return result
            
            # 准备输入数据
            step_input = self._prepare_step_input(step, context)
            result.input_data = step_input
            result.parameters = step.parameters.copy()
            
            # 检查缓存
            if step.cache_enabled and self.enable_caching:
                cache_key = self._generate_cache_key(step, step_input)
                if cache_key in self.step_cache:
                    logger.debug(f"Cache hit for step: {step.name}")
                    result.output_data = self.step_cache[cache_key]
                    result.cache_hit = True
                    result.status = StepStatus.COMPLETED
                    result.end_time = time.time()
                    result.execution_time = result.end_time - result.start_time
                    context.add_step_result(result)
                    return result
            
            # 执行步骤逻辑
            if step.executor:
                # 使用执行函数
                output = step.executor(step_input, **step.parameters)
            elif step.template:
                # 使用模板
                output = step.template.run(step_input, **step.parameters)
            else:
                raise ConfigurationError(f"Step '{step.name}' has no executor or template")
            
            result.output_data = output
            result.status = StepStatus.COMPLETED
            
            # 缓存结果
            if step.cache_enabled and self.enable_caching:
                cache_key = self._generate_cache_key(step, step_input)
                self.step_cache[cache_key] = output
                
                # 限制缓存大小
                if len(self.step_cache) > self.max_cache_size:
                    # 简单的LRU策略：删除最早的一半
                    keys_to_remove = list(self.step_cache.keys())[:len(self.step_cache) // 2]
                    for key in keys_to_remove:
                        del self.step_cache[key]
            
            logger.debug(f"Step completed successfully: {step.name}")
            
        except Exception as e:
            result.error = e
            result.error_message = str(e)
            result.status = StepStatus.FAILED
            
            # 处理错误
            if step.error_handler:
                try:
                    step.error_handler(e, result, context)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {str(handler_error)}")
            
            logger.error(f"Step failed: {step.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            context.add_step_result(result)
        
        return result
    
    async def _execute_step_async(self, step: StepConfig, context: ChainContext) -> StepResult:
        """
        执行单个步骤（异步版本）
        
        Args:
            step: 步骤配置
            context: 执行上下文
        
        Returns:
            步骤执行结果
        """
        result = StepResult(
            step_id=step.step_id,
            step_name=step.name,
            status=StepStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            logger.debug(f"Async executing step: {step.name}")
            result.status = StepStatus.RUNNING
            
            # 检查执行条件
            if step.condition:
                condition_result = step.condition(context)
                if asyncio.iscoroutine(condition_result):
                    condition_result = await condition_result
                
                if not condition_result:
                    logger.info(f"Step condition not met, skipping: {step.name}")
                    result.status = StepStatus.SKIPPED
                    result.end_time = time.time()
                    result.execution_time = result.end_time - result.start_time
                    context.add_step_result(result)
                    return result
            
            # 准备输入数据
            step_input = self._prepare_step_input(step, context)
            result.input_data = step_input
            result.parameters = step.parameters.copy()
            
            # 检查缓存
            if step.cache_enabled and self.enable_caching:
                cache_key = self._generate_cache_key(step, step_input)
                if cache_key in self.step_cache:
                    logger.debug(f"Cache hit for step: {step.name}")
                    result.output_data = self.step_cache[cache_key]
                    result.cache_hit = True
                    result.status = StepStatus.COMPLETED
                    result.end_time = time.time()
                    result.execution_time = result.end_time - result.start_time
                    context.add_step_result(result)
                    return result
            
            # 异步执行步骤逻辑
            if step.executor:
                # 使用执行函数
                if asyncio.iscoroutinefunction(step.executor):
                    output = await step.executor(step_input, **step.parameters)
                else:
                    # 在线程池中执行同步函数
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        self.thread_pool, 
                        step.executor, 
                        step_input, 
                        step.parameters
                    )
            elif step.template:
                # 使用模板的异步方法
                if hasattr(step.template, 'run_async'):
                    output = await step.template.run_async(step_input, **step.parameters)
                else:
                    # 在线程池中执行同步方法
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        self.thread_pool,
                        step.template.run,
                        step_input,
                        step.parameters
                    )
            else:
                raise ConfigurationError(f"Step '{step.name}' has no executor or template")
            
            result.output_data = output
            result.status = StepStatus.COMPLETED
            
            # 缓存结果
            if step.cache_enabled and self.enable_caching:
                cache_key = self._generate_cache_key(step, step_input)
                self.step_cache[cache_key] = output
                
                # 限制缓存大小
                if len(self.step_cache) > self.max_cache_size:
                    keys_to_remove = list(self.step_cache.keys())[:len(self.step_cache) // 2]
                    for key in keys_to_remove:
                        del self.step_cache[key]
            
            logger.debug(f"Async step completed successfully: {step.name}")
            
        except Exception as e:
            result.error = e
            result.error_message = str(e)
            result.status = StepStatus.FAILED
            
            # 处理错误
            if step.error_handler:
                try:
                    if asyncio.iscoroutinefunction(step.error_handler):
                        await step.error_handler(e, result, context)
                    else:
                        step.error_handler(e, result, context)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {str(handler_error)}")
            
            logger.error(f"Async step failed: {step.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            context.add_step_result(result)
        
        return result
    
    def _prepare_step_input(self, step: StepConfig, context: ChainContext) -> Any:
        """
        准备步骤输入数据
        
        Args:
            step: 步骤配置
            context: 执行上下文
        
        Returns:
            步骤输入数据
        """
        if not step.input_keys:
            # 如果没有指定输入键，返回整个上下文数据
            return context.data.copy()
        
        # 根据输入键提取数据
        step_input = {}
        for key in step.input_keys:
            if key in context.data:
                step_input[key] = context.data[key]
            else:
                logger.warning(f"Input key '{key}' not found in context for step '{step.name}'")
        
        return step_input
    
    def _generate_cache_key(self, step: StepConfig, step_input: Any) -> str:
        """
        生成缓存键
        
        Args:
            step: 步骤配置
            step_input: 步骤输入
        
        Returns:
            缓存键
        """
        if step.cache_key_generator:
            return step.cache_key_generator(step, step_input)
        
        # 默认缓存键生成策略
        import hashlib
        import json
        
        key_data = {
            "step_id": step.step_id,
            "input": step_input,
            "parameters": step.parameters
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def pause(self) -> None:
        """暂停链执行"""
        self.is_paused = True
        self._pause_event.clear()
        logger.info("Chain execution paused")
    
    def resume(self) -> None:
        """恢复链执行"""
        self.is_paused = False
        self._pause_event.set()
        logger.info("Chain execution resumed")
    
    def cancel(self) -> None:
        """取消链执行"""
        self.is_cancelled = True
        logger.info("Chain execution cancelled")
    
    def clear_cache(self) -> None:
        """清空步骤缓存"""
        self.step_cache.clear()
        logger.info("Step cache cleared")
    
    def get_example(self) -> Dict[str, Any]:
        """
        获取使用示例
        
        Returns:
            使用示例字典
        """
        return {
            "setup_parameters": {
                "steps": [
                    {
                        "name": "数据预处理",
                        "executor": "preprocess_data",
                        "input_keys": ["raw_data"],
                        "output_keys": ["processed_data"],
                        "description": "清洗和预处理原始数据"
                    },
                    {
                        "name": "特征提取",
                        "executor": "extract_features",
                        "input_keys": ["processed_data"],
                        "output_keys": ["features"],
                        "description": "从处理后的数据中提取特征"
                    },
                    {
                        "name": "模型预测",
                        "executor": "predict",
                        "input_keys": ["features"],
                        "output_keys": ["prediction"],
                        "description": "使用模型进行预测"
                    }
                ],
                "error_strategy": "fail_fast",
                "enable_caching": True,
                "timeout": 300
            },
            "execute_parameters": {
                "raw_data": "输入的原始数据"
            },
            "expected_output": {
                "status": "completed",
                "data": {
                    "processed_data": "处理后的数据",
                    "features": "提取的特征",
                    "prediction": "预测结果"
                },
                "summary": {
                    "total_steps": 3,
                    "completed_steps": 3,
                    "failed_steps": 0,
                    "success_rate": 1.0
                }
            },
            "usage_code": '''
# 使用示例
from templates.chains.sequential_chain import SequentialChainTemplate

# 定义步骤执行函数
def preprocess_data(data, **kwargs):
    """数据预处理函数"""
    # 实现数据预处理逻辑
    return {"processed_data": f"processed_{data['raw_data']}"}

def extract_features(data, **kwargs):
    """特征提取函数"""
    # 实现特征提取逻辑
    return {"features": f"features_from_{data['processed_data']}"}

def predict(data, **kwargs):
    """预测函数"""
    # 实现预测逻辑
    return {"prediction": f"prediction_for_{data['features']}"}

# 创建和配置链
chain = SequentialChainTemplate()
chain.setup(
    steps=[
        {
            "name": "数据预处理",
            "executor": preprocess_data,
            "input_keys": ["raw_data"],
            "description": "清洗和预处理原始数据"
        },
        {
            "name": "特征提取", 
            "executor": extract_features,
            "input_keys": ["processed_data"],
            "description": "从处理后的数据中提取特征"
        },
        {
            "name": "模型预测",
            "executor": predict,
            "input_keys": ["features"],
            "description": "使用模型进行预测"
        }
    ],
    error_strategy="fail_fast",
    enable_caching=True
)

# 执行链
result = chain.run({"raw_data": "sample_data"})
print("执行结果:", result["data"])
print("执行摘要:", result["summary"])

# 异步执行
import asyncio

async def main():
    result = await chain.run_async({"raw_data": "sample_data"})
    print("异步执行结果:", result["data"])

asyncio.run(main())
'''
        }
    
    def __del__(self):
        """析构函数，清理资源"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)


# 注册模板到工厂
def register_sequential_chain_template():
    """注册顺序链模板到全局工厂"""
    from ..base.template_base import register_template
    register_template("sequential_chain", SequentialChainTemplate)


# 自动注册
register_sequential_chain_template()