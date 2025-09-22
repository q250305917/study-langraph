"""
LangChain学习项目的基础链类模块

本模块定义了项目中所有链式操作的基础接口和实现，包括：
- BaseChain抽象基类：定义链式调用的统一接口
- ChainComposer链组合器：支持多个链的串联和并联
- ChainContext链上下文：管理链执行过程中的状态和数据
- ChainMiddleware中间件：提供可插拔的处理逻辑
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, validator

from .logger import get_logger, log_function_call, log_performance
from .exceptions import (
    ChainExecutionError,
    ValidationError,
    TimeoutError,
    ErrorCodes,
    exception_handler,
    retry_on_exception
)

logger = get_logger(__name__)

T = TypeVar('T')
U = TypeVar('U')


class ChainStatus(Enum):
    """链执行状态枚举"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 执行完成
    FAILED = "failed"        # 执行失败
    CANCELLED = "cancelled"  # 执行取消


class ChainType(Enum):
    """链类型枚举"""
    SEQUENTIAL = "sequential"  # 顺序执行
    PARALLEL = "parallel"      # 并行执行
    CONDITIONAL = "conditional"  # 条件执行
    LOOP = "loop"             # 循环执行


@dataclass
class ChainMetadata:
    """
    链元数据
    
    存储链的基本信息和执行统计数据。
    """
    name: str
    chain_type: ChainType
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # 执行统计
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[float] = None


class ChainInput(BaseModel):
    """
    链输入数据模型
    
    定义链的输入数据结构，支持数据验证和类型转换。
    """
    
    # 主要输入数据
    data: Dict[str, Any] = Field(default_factory=dict, description="主要输入数据")
    
    # 执行配置
    config: Dict[str, Any] = Field(default_factory=dict, description="执行配置")
    
    # 上下文信息
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")
    
    # 执行选项
    timeout: Optional[float] = Field(default=None, description="超时时间（秒）")
    retry_count: int = Field(default=0, description="重试次数")
    
    class Config:
        # 允许额外字段
        extra = "allow"
    
    @validator('timeout')
    def validate_timeout(cls, v):
        """验证超时时间"""
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @validator('retry_count')
    def validate_retry_count(cls, v):
        """验证重试次数"""
        if v < 0:
            raise ValueError("Retry count must be non-negative")
        return v


class ChainOutput(BaseModel):
    """
    链输出数据模型
    
    定义链的输出数据结构，包含执行结果和元信息。
    """
    
    # 主要输出数据
    data: Dict[str, Any] = Field(default_factory=dict, description="主要输出数据")
    
    # 执行信息
    execution_id: str = Field(description="执行ID")
    status: ChainStatus = Field(description="执行状态")
    start_time: float = Field(description="开始时间")
    end_time: Optional[float] = Field(default=None, description="结束时间")
    execution_time: Optional[float] = Field(default=None, description="执行时间")
    
    # 错误信息
    error: Optional[str] = Field(default=None, description="错误信息")
    error_code: Optional[str] = Field(default=None, description="错误码")
    
    # 中间结果
    intermediate_results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="中间执行结果"
    )
    
    # 性能指标
    metrics: Dict[str, Any] = Field(default_factory=dict, description="性能指标")
    
    class Config:
        # 允许额外字段
        extra = "allow"
    
    @property
    def is_successful(self) -> bool:
        """检查执行是否成功"""
        return self.status == ChainStatus.COMPLETED
    
    @property
    def duration(self) -> Optional[float]:
        """获取执行时长"""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None


class ChainContext:
    """
    链执行上下文
    
    管理链执行过程中的状态、数据和配置信息。
    支持上下文的嵌套和继承。
    """
    
    def __init__(
        self,
        execution_id: Optional[str] = None,
        parent_context: Optional['ChainContext'] = None
    ):
        """
        初始化链上下文
        
        Args:
            execution_id: 执行ID，None则自动生成
            parent_context: 父上下文，用于嵌套执行
        """
        self.execution_id = execution_id or str(uuid.uuid4())
        self.parent_context = parent_context
        
        # 数据存储
        self._data: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        
        # 执行状态
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status = ChainStatus.PENDING
        
        # 子上下文管理
        self.child_contexts: List['ChainContext'] = []
        
        logger.debug(f"Created chain context: {self.execution_id}")
    
    def set_data(self, key: str, value: Any) -> None:
        """设置数据"""
        self._data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """
        获取数据
        
        如果当前上下文中不存在该数据，会向上查找父上下文。
        """
        if key in self._data:
            return self._data[key]
        elif self.parent_context:
            return self.parent_context.get_data(key, default)
        else:
            return default
    
    def set_config(self, key: str, value: Any) -> None:
        """设置配置"""
        self._config[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置"""
        if key in self._config:
            return self._config[key]
        elif self.parent_context:
            return self.parent_context.get_config(key, default)
        else:
            return default
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self._metadata.get(key, default)
    
    def create_child_context(self) -> 'ChainContext':
        """创建子上下文"""
        child = ChainContext(parent_context=self)
        self.child_contexts.append(child)
        return child
    
    def start_execution(self) -> None:
        """开始执行"""
        self.start_time = time.time()
        self.status = ChainStatus.RUNNING
    
    def complete_execution(self) -> None:
        """完成执行"""
        self.end_time = time.time()
        self.status = ChainStatus.COMPLETED
    
    def fail_execution(self, error: str) -> None:
        """执行失败"""
        self.end_time = time.time()
        self.status = ChainStatus.FAILED
        self.set_metadata("error", error)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "data": self._data,
            "config": self._config,
            "metadata": self._metadata,
            "child_contexts": [child.to_dict() for child in self.child_contexts]
        }


class ChainMiddleware(ABC):
    """
    链中间件抽象基类
    
    定义了中间件的通用接口，支持在链执行的不同阶段插入自定义逻辑。
    """
    
    @abstractmethod
    async def before_execute(self, context: ChainContext, inputs: ChainInput) -> None:
        """
        执行前处理
        
        Args:
            context: 链上下文
            inputs: 输入数据
        """
        pass
    
    @abstractmethod
    async def after_execute(self, context: ChainContext, output: ChainOutput) -> None:
        """
        执行后处理
        
        Args:
            context: 链上下文
            output: 输出数据
        """
        pass
    
    @abstractmethod
    async def on_error(self, context: ChainContext, error: Exception) -> None:
        """
        错误处理
        
        Args:
            context: 链上下文
            error: 异常对象
        """
        pass


class LoggingMiddleware(ChainMiddleware):
    """日志中间件，记录链的执行过程"""
    
    async def before_execute(self, context: ChainContext, inputs: ChainInput) -> None:
        logger.info(f"Chain execution started: {context.execution_id}")
        logger.debug(f"Input data keys: {list(inputs.data.keys())}")
    
    async def after_execute(self, context: ChainContext, output: ChainOutput) -> None:
        duration = output.duration or 0
        logger.info(
            f"Chain execution completed: {context.execution_id} "
            f"(duration: {duration:.3f}s, status: {output.status.value})"
        )
    
    async def on_error(self, context: ChainContext, error: Exception) -> None:
        logger.error(f"Chain execution failed: {context.execution_id} - {str(error)}")


class MetricsMiddleware(ChainMiddleware):
    """性能指标中间件，收集执行统计信息"""
    
    async def before_execute(self, context: ChainContext, inputs: ChainInput) -> None:
        context.set_metadata("start_memory", self._get_memory_usage())
        context.set_metadata("input_size", len(str(inputs.data)))
    
    async def after_execute(self, context: ChainContext, output: ChainOutput) -> None:
        output.metrics.update({
            "memory_usage": self._get_memory_usage() - context.get_metadata("start_memory", 0),
            "input_size": context.get_metadata("input_size", 0),
            "output_size": len(str(output.data))
        })
    
    async def on_error(self, context: ChainContext, error: Exception) -> None:
        context.set_metadata("error_type", type(error).__name__)
    
    def _get_memory_usage(self) -> int:
        """获取内存使用量（简化实现）"""
        import psutil
        return psutil.Process().memory_info().rss


class BaseChain(ABC, Generic[T, U]):
    """
    链式调用的抽象基类
    
    定义了链式操作的核心接口，包括执行、组合、监控等功能。
    所有具体的链实现都应该继承此类。
    """
    
    def __init__(
        self,
        name: str,
        metadata: Optional[ChainMetadata] = None,
        middlewares: Optional[List[ChainMiddleware]] = None
    ):
        """
        初始化基础链
        
        Args:
            name: 链名称
            metadata: 元数据信息
            middlewares: 中间件列表
        """
        self.name = name
        self.metadata = metadata or ChainMetadata(
            name=name,
            chain_type=ChainType.SEQUENTIAL
        )
        self.middlewares = middlewares or []
        
        # 添加默认中间件
        self._add_default_middlewares()
        
        logger.debug(f"Initialized chain: {self.name}")
    
    def _add_default_middlewares(self) -> None:
        """添加默认中间件"""
        # 检查是否已存在相同类型的中间件
        existing_types = {type(m) for m in self.middlewares}
        
        if LoggingMiddleware not in existing_types:
            self.middlewares.append(LoggingMiddleware())
        
        if MetricsMiddleware not in existing_types:
            self.middlewares.append(MetricsMiddleware())
    
    @abstractmethod
    async def _execute(self, context: ChainContext, inputs: ChainInput) -> ChainOutput:
        """
        子类必须实现的核心执行方法
        
        Args:
            context: 执行上下文
            inputs: 输入数据
            
        Returns:
            执行结果
            
        Raises:
            ChainExecutionError: 执行失败
        """
        pass
    
    async def run(self, inputs: Union[Dict[str, Any], ChainInput], **kwargs) -> ChainOutput:
        """
        执行链的主要入口方法
        
        Args:
            inputs: 输入数据，可以是字典或ChainInput对象
            **kwargs: 额外的配置参数
            
        Returns:
            执行结果
            
        Raises:
            ChainExecutionError: 执行失败
            TimeoutError: 执行超时
            ValidationError: 输入验证失败
        """
        # 输入数据标准化
        if isinstance(inputs, dict):
            chain_input = ChainInput(data=inputs, config=kwargs)
        else:
            chain_input = inputs
            # 合并额外配置
            chain_input.config.update(kwargs)
        
        # 创建执行上下文
        context = ChainContext()
        
        # 创建输出对象
        output = ChainOutput(
            execution_id=context.execution_id,
            status=ChainStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            # 输入验证
            await self._validate_inputs(chain_input)
            
            # 执行前处理（中间件）
            await self._run_before_middlewares(context, chain_input)
            
            # 开始执行
            context.start_execution()
            output.status = ChainStatus.RUNNING
            
            # 执行核心逻辑（支持超时）
            if chain_input.timeout:
                output = await asyncio.wait_for(
                    self._execute_with_retry(context, chain_input),
                    timeout=chain_input.timeout
                )
            else:
                output = await self._execute_with_retry(context, chain_input)
            
            # 执行成功
            context.complete_execution()
            output.status = ChainStatus.COMPLETED
            output.end_time = time.time()
            output.execution_time = output.end_time - output.start_time
            
            # 执行后处理（中间件）
            await self._run_after_middlewares(context, output)
            
            # 更新统计信息
            self._update_metrics(output)
            
            return output
            
        except asyncio.TimeoutError as e:
            # 超时处理
            context.fail_execution("Execution timeout")
            output.status = ChainStatus.FAILED
            output.end_time = time.time()
            output.error = "Execution timeout"
            output.error_code = ErrorCodes.EXECUTION_TIMEOUT
            
            await self._run_error_middlewares(context, e)
            
            raise TimeoutError(
                f"Chain execution timeout after {chain_input.timeout}s",
                error_code=ErrorCodes.EXECUTION_TIMEOUT,
                context={"chain_name": self.name, "timeout": chain_input.timeout}
            )
            
        except Exception as e:
            # 一般错误处理
            context.fail_execution(str(e))
            output.status = ChainStatus.FAILED
            output.end_time = time.time()
            output.error = str(e)
            output.error_code = getattr(e, 'error_code', 'UNKNOWN_ERROR')
            
            await self._run_error_middlewares(context, e)
            
            # 包装为链执行异常
            if not isinstance(e, ChainExecutionError):
                raise ChainExecutionError(
                    f"Chain execution failed: {str(e)}",
                    error_code=ErrorCodes.CHAIN_EXECUTION_ERROR,
                    context={"chain_name": self.name},
                    cause=e
                )
            raise
    
    async def _execute_with_retry(self, context: ChainContext, inputs: ChainInput) -> ChainOutput:
        """带重试的执行方法"""
        last_exception = None
        
        for attempt in range(inputs.retry_count + 1):
            try:
                return await self._execute(context, inputs)
            except Exception as e:
                last_exception = e
                
                if attempt < inputs.retry_count:
                    logger.warning(
                        f"Chain execution failed (attempt {attempt + 1}), "
                        f"retrying: {str(e)}"
                    )
                    # 可以在这里添加退避策略
                    await asyncio.sleep(2 ** attempt)
                else:
                    # 最后一次尝试失败
                    raise
        
        # 理论上不会到达这里
        raise last_exception
    
    async def _validate_inputs(self, inputs: ChainInput) -> None:
        """
        验证输入数据
        
        子类可以重写此方法来实现自定义的输入验证逻辑。
        """
        try:
            # 基础验证由Pydantic自动完成
            # 这里可以添加额外的业务逻辑验证
            pass
        except Exception as e:
            raise ValidationError(
                f"Input validation failed: {str(e)}",
                error_code=ErrorCodes.VALIDATION_TYPE_ERROR,
                context={"chain_name": self.name},
                cause=e
            )
    
    async def _run_before_middlewares(self, context: ChainContext, inputs: ChainInput) -> None:
        """运行执行前中间件"""
        for middleware in self.middlewares:
            try:
                await middleware.before_execute(context, inputs)
            except Exception as e:
                logger.warning(f"Middleware {type(middleware).__name__} failed: {e}")
    
    async def _run_after_middlewares(self, context: ChainContext, output: ChainOutput) -> None:
        """运行执行后中间件"""
        for middleware in self.middlewares:
            try:
                await middleware.after_execute(context, output)
            except Exception as e:
                logger.warning(f"Middleware {type(middleware).__name__} failed: {e}")
    
    async def _run_error_middlewares(self, context: ChainContext, error: Exception) -> None:
        """运行错误处理中间件"""
        for middleware in self.middlewares:
            try:
                await middleware.on_error(context, error)
            except Exception as e:
                logger.warning(f"Error middleware {type(middleware).__name__} failed: {e}")
    
    def _update_metrics(self, output: ChainOutput) -> None:
        """更新执行统计信息"""
        self.metadata.total_executions += 1
        
        if output.is_successful:
            self.metadata.successful_executions += 1
        else:
            self.metadata.failed_executions += 1
        
        if output.execution_time is not None:
            # 更新平均执行时间（指数移动平均）
            if self.metadata.average_execution_time == 0:
                self.metadata.average_execution_time = output.execution_time
            else:
                alpha = 0.1  # 平滑因子
                self.metadata.average_execution_time = (
                    alpha * output.execution_time + 
                    (1 - alpha) * self.metadata.average_execution_time
                )
            
            self.metadata.last_execution_time = output.execution_time
    
    def add_middleware(self, middleware: ChainMiddleware) -> None:
        """添加中间件"""
        self.middlewares.append(middleware)
        logger.debug(f"Added middleware {type(middleware).__name__} to chain {self.name}")
    
    def compose(self, other: 'BaseChain') -> 'ChainComposer':
        """
        与另一个链组合
        
        Args:
            other: 另一个链实例
            
        Returns:
            链组合器
        """
        return ChainComposer([self, other])
    
    def __or__(self, other: 'BaseChain') -> 'ChainComposer':
        """支持 | 操作符进行链组合"""
        return self.compose(other)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取链的性能指标"""
        success_rate = 0.0
        if self.metadata.total_executions > 0:
            success_rate = self.metadata.successful_executions / self.metadata.total_executions
        
        return {
            "name": self.name,
            "total_executions": self.metadata.total_executions,
            "successful_executions": self.metadata.successful_executions,
            "failed_executions": self.metadata.failed_executions,
            "success_rate": success_rate,
            "average_execution_time": self.metadata.average_execution_time,
            "last_execution_time": self.metadata.last_execution_time
        }


class ChainComposer(BaseChain):
    """
    链组合器
    
    支持多个链的串联执行，前一个链的输出作为后一个链的输入。
    """
    
    def __init__(
        self,
        chains: List[BaseChain],
        name: Optional[str] = None,
        execution_mode: ChainType = ChainType.SEQUENTIAL
    ):
        """
        初始化链组合器
        
        Args:
            chains: 要组合的链列表
            name: 组合器名称
            execution_mode: 执行模式（顺序或并行）
        """
        if not chains:
            raise ValueError("Chains list cannot be empty")
        
        composer_name = name or f"Composer({', '.join(chain.name for chain in chains)})"
        
        metadata = ChainMetadata(
            name=composer_name,
            chain_type=execution_mode,
            description=f"Composed chain with {len(chains)} chains"
        )
        
        super().__init__(composer_name, metadata)
        
        self.chains = chains
        self.execution_mode = execution_mode
    
    async def _execute(self, context: ChainContext, inputs: ChainInput) -> ChainOutput:
        """执行组合链"""
        if self.execution_mode == ChainType.SEQUENTIAL:
            return await self._execute_sequential(context, inputs)
        elif self.execution_mode == ChainType.PARALLEL:
            return await self._execute_parallel(context, inputs)
        else:
            raise ChainExecutionError(
                f"Unsupported execution mode: {self.execution_mode}",
                error_code=ErrorCodes.CHAIN_EXECUTION_ERROR
            )
    
    async def _execute_sequential(self, context: ChainContext, inputs: ChainInput) -> ChainOutput:
        """顺序执行多个链"""
        current_data = inputs.data.copy()
        intermediate_results = []
        
        # 创建组合输出
        output = ChainOutput(
            execution_id=context.execution_id,
            status=ChainStatus.RUNNING,
            start_time=time.time()
        )
        
        for i, chain in enumerate(self.chains):
            logger.debug(f"Executing chain {i+1}/{len(self.chains)}: {chain.name}")
            
            # 创建子上下文
            child_context = context.create_child_context()
            
            # 准备输入数据
            chain_input = ChainInput(
                data=current_data,
                config=inputs.config,
                context=inputs.context
            )
            
            try:
                # 执行当前链
                chain_output = await chain.run(chain_input)
                
                # 保存中间结果
                intermediate_results.append({
                    "chain_name": chain.name,
                    "execution_id": chain_output.execution_id,
                    "status": chain_output.status.value,
                    "execution_time": chain_output.execution_time,
                    "data": chain_output.data
                })
                
                # 将输出作为下一个链的输入
                current_data = chain_output.data
                
            except Exception as e:
                # 链执行失败，记录错误并中断
                intermediate_results.append({
                    "chain_name": chain.name,
                    "status": ChainStatus.FAILED.value,
                    "error": str(e)
                })
                
                raise ChainExecutionError(
                    f"Chain {chain.name} failed in sequential execution: {str(e)}",
                    error_code=ErrorCodes.CHAIN_EXECUTION_ERROR,
                    context={"failed_chain": chain.name, "chain_index": i},
                    cause=e
                )
        
        # 设置最终输出
        output.data = current_data
        output.intermediate_results = intermediate_results
        output.status = ChainStatus.COMPLETED
        
        return output
    
    async def _execute_parallel(self, context: ChainContext, inputs: ChainInput) -> ChainOutput:
        """并行执行多个链"""
        logger.debug(f"Executing {len(self.chains)} chains in parallel")
        
        # 创建组合输出
        output = ChainOutput(
            execution_id=context.execution_id,
            status=ChainStatus.RUNNING,
            start_time=time.time()
        )
        
        # 为每个链创建独立的输入副本
        chain_tasks = []
        for chain in self.chains:
            chain_input = ChainInput(
                data=inputs.data.copy(),
                config=inputs.config.copy(),
                context=inputs.context.copy()
            )
            
            task = asyncio.create_task(
                chain.run(chain_input),
                name=f"chain_{chain.name}"
            )
            chain_tasks.append((chain, task))
        
        # 等待所有链执行完成
        results = []
        errors = []
        
        for chain, task in chain_tasks:
            try:
                chain_output = await task
                results.append({
                    "chain_name": chain.name,
                    "execution_id": chain_output.execution_id,
                    "status": chain_output.status.value,
                    "execution_time": chain_output.execution_time,
                    "data": chain_output.data
                })
            except Exception as e:
                error_info = {
                    "chain_name": chain.name,
                    "status": ChainStatus.FAILED.value,
                    "error": str(e)
                }
                results.append(error_info)
                errors.append((chain.name, e))
        
        # 如果有错误，抛出异常
        if errors:
            error_msg = f"Failed chains in parallel execution: {[name for name, _ in errors]}"
            raise ChainExecutionError(
                error_msg,
                error_code=ErrorCodes.CHAIN_EXECUTION_ERROR,
                context={"failed_chains": errors}
            )
        
        # 合并所有链的输出数据
        merged_data = {}
        for result in results:
            if isinstance(result.get("data"), dict):
                merged_data.update(result["data"])
            else:
                # 如果输出不是字典，使用链名称作为键
                merged_data[result["chain_name"]] = result["data"]
        
        output.data = merged_data
        output.intermediate_results = results
        output.status = ChainStatus.COMPLETED
        
        return output
    
    def add_chain(self, chain: BaseChain) -> None:
        """添加链到组合器"""
        self.chains.append(chain)
        self.metadata.description = f"Composed chain with {len(self.chains)} chains"
        logger.debug(f"Added chain {chain.name} to composer")
    
    def __len__(self) -> int:
        """返回包含的链数量"""
        return len(self.chains)
    
    def __getitem__(self, index: int) -> BaseChain:
        """通过索引访问链"""
        return self.chains[index]