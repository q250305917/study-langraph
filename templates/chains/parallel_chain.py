"""
并行链模板（ParallelChainTemplate）

本模块提供了并行执行的链模板，支持多个分支同时执行的工作流。
这是Chain模板系统的重要组件，适用于需要并发处理的复杂任务。

核心特性：
1. 并行执行：多个分支同时执行，提高处理效率
2. 结果聚合：自动收集和合并各分支的执行结果
3. 错误隔离：单个分支的错误不影响其他分支
4. 资源控制：限制并发数量，避免资源过度使用
5. 超时控制：支持整体和分支级别的超时设置
6. 动态调度：根据系统负载动态调整执行策略

设计原理：
- 生产者消费者模式：任务分发和结果收集的解耦
- 线程池模式：高效的并发任务执行
- Future模式：异步任务状态管理
- 分治策略：将复杂任务分解为独立的并行子任务
- 栅栏同步：等待所有分支完成后进行结果聚合

使用场景：
- 数据并行处理：同时处理多个数据源或分片
- 多模型推理：同时使用多个模型进行推理比较
- 批量任务处理：并行执行大量独立的小任务
- 多源信息聚合：同时从多个数据源获取信息
- A/B测试执行：并行执行不同的算法或策略
"""

import asyncio
import time
import copy
import uuid
import threading
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed, wait
import queue
import multiprocessing
from pathlib import Path

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)


class BranchStatus(Enum):
    """分支执行状态枚举"""
    PENDING = "pending"       # 等待执行
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 执行完成
    FAILED = "failed"         # 执行失败
    TIMEOUT = "timeout"       # 执行超时
    CANCELLED = "cancelled"   # 取消执行


class ExecutionMode(Enum):
    """执行模式枚举"""
    THREAD = "thread"         # 线程并行
    PROCESS = "process"       # 进程并行
    ASYNC = "async"           # 异步并发
    HYBRID = "hybrid"         # 混合模式


class AggregationStrategy(Enum):
    """结果聚合策略枚举"""
    ALL = "all"               # 等待所有分支完成
    FIRST = "first"           # 第一个完成即返回
    MAJORITY = "majority"     # 大多数完成即返回
    CUSTOM = "custom"         # 自定义聚合逻辑


@dataclass
class BranchConfig:
    """
    分支配置类
    
    定义并行链中每个分支的配置信息，包括名称、描述、
    执行函数、参数、优先级、资源限制等。
    """
    
    # === 基本信息 ===
    name: str                                    # 分支名称
    description: str = ""                        # 分支描述
    branch_id: Optional[str] = None             # 分支唯一标识
    
    # === 执行配置 ===
    executor: Optional[Callable] = None          # 执行函数
    template: Optional[TemplateBase] = None      # 模板实例
    
    # === 参数配置 ===
    input_data: Any = None                       # 输入数据
    parameters: Dict[str, Any] = field(default_factory=dict)  # 执行参数
    
    # === 执行控制 ===
    priority: int = 0                            # 执行优先级（数字越大优先级越高）
    timeout: Optional[float] = None              # 超时时间（秒）
    retry_count: int = 0                         # 重试次数
    retry_delay: float = 1.0                     # 重试延迟（秒）
    
    # === 资源控制 ===
    execution_mode: ExecutionMode = ExecutionMode.THREAD  # 执行模式
    resource_limit: Dict[str, Any] = field(default_factory=dict)  # 资源限制
    
    # === 依赖关系 ===
    dependencies: List[str] = field(default_factory=list)  # 依赖的分支ID
    condition: Optional[Callable] = None         # 执行条件函数
    
    # === 错误处理 ===
    error_handler: Optional[Callable] = None     # 自定义错误处理函数
    ignore_errors: bool = False                  # 是否忽略错误
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.branch_id:
            self.branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        
        # 验证配置有效性
        if not self.executor and not self.template:
            raise ConfigurationError("Branch must have either executor or template")
        
        if self.executor and self.template:
            raise ConfigurationError("Branch cannot have both executor and template")


@dataclass
class BranchResult:
    """
    分支执行结果类
    
    包含分支执行的完整信息，包括状态、输入输出、
    执行时间、错误信息等。
    """
    
    # === 基本信息 ===
    branch_id: str                               # 分支ID
    branch_name: str                             # 分支名称
    status: BranchStatus                         # 执行状态
    
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
    retry_count: int = 0                         # 实际重试次数
    
    # === 性能信息 ===
    memory_usage: Optional[float] = None         # 内存使用量
    cpu_usage: Optional[float] = None            # CPU使用率
    
    def is_successful(self) -> bool:
        """判断分支是否执行成功"""
        return self.status == BranchStatus.COMPLETED
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "branch_id": self.branch_id,
            "branch_name": self.branch_name,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "success": self.is_successful(),
            "retry_count": self.retry_count,
            "error_message": self.error_message
        }


class ParallelExecutor:
    """
    并行执行器
    
    负责管理和执行并行任务，支持多种执行模式和策略。
    提供任务调度、资源管理、结果收集等核心功能。
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        execution_mode: ExecutionMode = ExecutionMode.THREAD,
        timeout: Optional[float] = None
    ):
        """
        初始化并行执行器
        
        Args:
            max_workers: 最大工作线程/进程数
            execution_mode: 执行模式
            timeout: 默认超时时间
        """
        self.max_workers = max_workers
        self.execution_mode = execution_mode
        self.default_timeout = timeout
        
        # 执行器实例
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # 任务管理
        self.running_tasks: Dict[str, Future] = {}
        self.task_results: Dict[str, BranchResult] = {}
        
        # 同步控制
        self._lock = threading.Lock()
        self._shutdown = False
        
        logger.debug(f"Initialized ParallelExecutor with {max_workers} workers")
    
    def __enter__(self):
        """上下文管理器入口"""
        self._initialize_executors()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.shutdown()
    
    def _initialize_executors(self) -> None:
        """初始化执行器"""
        if self.execution_mode in [ExecutionMode.THREAD, ExecutionMode.HYBRID]:
            self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        if self.execution_mode in [ExecutionMode.PROCESS, ExecutionMode.HYBRID]:
            # 进程池的工作进程数通常设置为CPU核心数
            process_workers = min(self.max_workers, multiprocessing.cpu_count())
            self.process_executor = ProcessPoolExecutor(max_workers=process_workers)
        
        logger.debug(f"Initialized executors for mode: {self.execution_mode.value}")
    
    def submit_branch(self, branch: BranchConfig) -> Future:
        """
        提交分支任务
        
        Args:
            branch: 分支配置
        
        Returns:
            Future对象
        """
        if self._shutdown:
            raise RuntimeError("Executor has been shutdown")
        
        # 根据执行模式选择执行器
        if branch.execution_mode == ExecutionMode.THREAD:
            if not self.thread_executor:
                self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
            executor = self.thread_executor
        elif branch.execution_mode == ExecutionMode.PROCESS:
            if not self.process_executor:
                process_workers = min(self.max_workers, multiprocessing.cpu_count())
                self.process_executor = ProcessPoolExecutor(max_workers=process_workers)
            executor = self.process_executor
        else:
            # 默认使用线程执行器
            if not self.thread_executor:
                self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
            executor = self.thread_executor
        
        # 提交任务
        future = executor.submit(self._execute_branch, branch)
        
        with self._lock:
            self.running_tasks[branch.branch_id] = future
        
        logger.debug(f"Submitted branch task: {branch.name}")
        return future
    
    def _execute_branch(self, branch: BranchConfig) -> BranchResult:
        """
        执行分支任务
        
        Args:
            branch: 分支配置
        
        Returns:
            分支执行结果
        """
        result = BranchResult(
            branch_id=branch.branch_id,
            branch_name=branch.name,
            status=BranchStatus.PENDING,
            start_time=time.time(),
            input_data=branch.input_data,
            parameters=branch.parameters.copy()
        )
        
        try:
            logger.debug(f"Executing branch: {branch.name}")
            result.status = BranchStatus.RUNNING
            
            # 检查执行条件
            if branch.condition and not branch.condition():
                logger.info(f"Branch condition not met, skipping: {branch.name}")
                result.status = BranchStatus.COMPLETED
                result.output_data = None
                return result
            
            # 执行分支逻辑
            if branch.executor:
                # 使用执行函数
                output = branch.executor(branch.input_data, **branch.parameters)
            elif branch.template:
                # 使用模板
                output = branch.template.run(branch.input_data, **branch.parameters)
            else:
                raise ConfigurationError(f"Branch '{branch.name}' has no executor or template")
            
            result.output_data = output
            result.status = BranchStatus.COMPLETED
            
            logger.debug(f"Branch completed successfully: {branch.name}")
            
        except Exception as e:
            result.error = e
            result.error_message = str(e)
            result.status = BranchStatus.FAILED
            
            # 处理错误
            if branch.error_handler:
                try:
                    branch.error_handler(e, result)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {str(handler_error)}")
            
            if not branch.ignore_errors:
                logger.error(f"Branch failed: {branch.name} - {str(e)}")
            else:
                logger.warning(f"Branch failed (ignored): {branch.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
            
            # 记录结果
            with self._lock:
                self.task_results[branch.branch_id] = result
                if branch.branch_id in self.running_tasks:
                    del self.running_tasks[branch.branch_id]
        
        return result
    
    def wait_for_completion(
        self,
        branch_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        return_when: str = "ALL_COMPLETED"
    ) -> Tuple[Set[Future], Set[Future]]:
        """
        等待分支完成
        
        Args:
            branch_ids: 要等待的分支ID列表，None表示等待所有分支
            timeout: 超时时间
            return_when: 返回条件 ("ALL_COMPLETED", "FIRST_COMPLETED")
        
        Returns:
            (已完成的Future集合, 未完成的Future集合)
        """
        with self._lock:
            if branch_ids:
                futures = [self.running_tasks[bid] for bid in branch_ids if bid in self.running_tasks]
            else:
                futures = list(self.running_tasks.values())
        
        if not futures:
            return set(), set()
        
        # 等待任务完成
        timeout_value = timeout or self.default_timeout
        done, not_done = wait(futures, timeout=timeout_value, return_when=return_when)
        
        logger.debug(f"Waited for {len(futures)} branches, {len(done)} completed")
        return done, not_done
    
    def get_results(self, branch_ids: Optional[List[str]] = None) -> Dict[str, BranchResult]:
        """
        获取分支执行结果
        
        Args:
            branch_ids: 要获取结果的分支ID列表，None表示获取所有结果
        
        Returns:
            分支结果字典
        """
        with self._lock:
            if branch_ids:
                return {bid: self.task_results[bid] for bid in branch_ids if bid in self.task_results}
            else:
                return self.task_results.copy()
    
    def cancel_branch(self, branch_id: str) -> bool:
        """
        取消分支执行
        
        Args:
            branch_id: 分支ID
        
        Returns:
            是否取消成功
        """
        with self._lock:
            if branch_id in self.running_tasks:
                future = self.running_tasks[branch_id]
                if future.cancel():
                    logger.info(f"Cancelled branch: {branch_id}")
                    return True
                else:
                    logger.warning(f"Failed to cancel branch (already running): {branch_id}")
                    return False
            else:
                logger.warning(f"Branch not found for cancellation: {branch_id}")
                return False
    
    def shutdown(self, wait: bool = True) -> None:
        """
        关闭执行器
        
        Args:
            wait: 是否等待任务完成
        """
        self._shutdown = True
        
        if self.thread_executor:
            self.thread_executor.shutdown(wait=wait)
            self.thread_executor = None
        
        if self.process_executor:
            self.process_executor.shutdown(wait=wait)
            self.process_executor = None
        
        logger.debug("ParallelExecutor shutdown completed")


class ParallelChainTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    并行链模板类
    
    实现多个分支同时执行的工作流。各分支可以独立执行，
    最后聚合所有分支的结果。
    
    核心功能：
    1. 分支管理：动态添加、删除、配置分支
    2. 并行执行：多种并行模式和资源控制
    3. 结果聚合：灵活的结果收集和合并策略
    4. 错误处理：分支级别的错误隔离和处理
    5. 性能监控：实时监控执行状态和性能指标
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化并行链模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config)
        
        # 分支配置
        self.branches: List[BranchConfig] = []
        
        # 执行配置
        self.max_workers = 4
        self.execution_mode = ExecutionMode.THREAD
        self.aggregation_strategy = AggregationStrategy.ALL
        self.timeout = None
        
        # 结果聚合
        self.aggregator: Optional[Callable] = None
        self.result_filter: Optional[Callable] = None
        
        # 执行状态
        self.executor: Optional[ParallelExecutor] = None
        self.is_cancelled = False
        
        logger.info(f"Initialized ParallelChainTemplate: {self.config.name}")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="ParallelChainTemplate",
            version="1.0.0",
            description="并行执行的链模板，支持多分支同时执行",
            template_type=TemplateType.CHAIN,
            async_enabled=True,
            cache_enabled=False
        )
        
        # 定义参数模式
        config.add_parameter(
            "branches", list, required=True,
            description="并行分支配置列表",
            examples=[
                [
                    {"name": "branch1", "executor": "function1"},
                    {"name": "branch2", "executor": "function2"}
                ]
            ]
        )
        
        config.add_parameter(
            "max_workers", int, required=False, default=4,
            description="最大并发工作线程数",
            constraints={"min_value": 1, "max_value": 32}
        )
        
        config.add_parameter(
            "execution_mode", str, required=False, default="thread",
            description="执行模式",
            constraints={"allowed_values": ["thread", "process", "async", "hybrid"]}
        )
        
        config.add_parameter(
            "aggregation_strategy", str, required=False, default="all",
            description="结果聚合策略",
            constraints={"allowed_values": ["all", "first", "majority", "custom"]}
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
                - branches: 分支配置列表
                - max_workers: 最大并发数
                - execution_mode: 执行模式
                - aggregation_strategy: 聚合策略
                - timeout: 超时时间
        
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        # 验证参数
        self.validate_parameters(parameters)
        
        # 保存设置参数
        self._setup_parameters = parameters.copy()
        
        # 配置分支
        branches_config = parameters.get("branches", [])
        self._configure_branches(branches_config)
        
        # 配置执行参数
        self.max_workers = parameters.get("max_workers", 4)
        execution_mode_str = parameters.get("execution_mode", "thread")
        self.execution_mode = ExecutionMode(execution_mode_str)
        
        # 配置聚合策略
        aggregation_strategy_str = parameters.get("aggregation_strategy", "all")
        self.aggregation_strategy = AggregationStrategy(aggregation_strategy_str)
        
        # 配置超时
        self.timeout = parameters.get("timeout")
        
        # 配置自定义聚合器
        self.aggregator = parameters.get("aggregator")
        self.result_filter = parameters.get("result_filter")
        
        # 更新状态
        self.status = self.status.__class__.CONFIGURED
        
        logger.info(f"Configured ParallelChain with {len(self.branches)} branches")
    
    def _configure_branches(self, branches_config: List[Dict[str, Any]]) -> None:
        """
        配置分支列表
        
        Args:
            branches_config: 分支配置列表
        """
        self.branches.clear()
        
        for i, branch_dict in enumerate(branches_config):
            try:
                # 创建分支配置
                branch_config = self._create_branch_config(branch_dict, i)
                self.branches.append(branch_config)
                
                logger.debug(f"Configured branch: {branch_config.name}")
                
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to configure branch {i}: {str(e)}",
                    error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                    cause=e
                )
    
    def _create_branch_config(self, branch_dict: Dict[str, Any], index: int) -> BranchConfig:
        """
        创建分支配置
        
        Args:
            branch_dict: 分支字典配置
            index: 分支索引
        
        Returns:
            分支配置对象
        """
        # 基本信息
        name = branch_dict.get("name", f"branch_{index}")
        description = branch_dict.get("description", "")
        branch_id = branch_dict.get("branch_id", f"branch_{index}_{uuid.uuid4().hex[:8]}")
        
        # 执行器配置
        executor = branch_dict.get("executor")
        template = branch_dict.get("template")
        
        # 参数配置
        input_data = branch_dict.get("input_data")
        parameters = branch_dict.get("parameters", {})
        
        # 执行控制
        priority = branch_dict.get("priority", 0)
        timeout = branch_dict.get("timeout")
        retry_count = branch_dict.get("retry_count", 0)
        retry_delay = branch_dict.get("retry_delay", 1.0)
        
        # 执行模式
        execution_mode_str = branch_dict.get("execution_mode", self.execution_mode.value)
        execution_mode = ExecutionMode(execution_mode_str)
        
        # 资源限制
        resource_limit = branch_dict.get("resource_limit", {})
        
        # 依赖关系
        dependencies = branch_dict.get("dependencies", [])
        condition = branch_dict.get("condition")
        
        # 错误处理
        error_handler = branch_dict.get("error_handler")
        ignore_errors = branch_dict.get("ignore_errors", False)
        
        return BranchConfig(
            name=name,
            description=description,
            branch_id=branch_id,
            executor=executor,
            template=template,
            input_data=input_data,
            parameters=parameters,
            priority=priority,
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            execution_mode=execution_mode,
            resource_limit=resource_limit,
            dependencies=dependencies,
            condition=condition,
            error_handler=error_handler,
            ignore_errors=ignore_errors
        )
    
    def add_branch(
        self,
        name: str,
        executor: Optional[Callable] = None,
        template: Optional[TemplateBase] = None,
        input_data: Any = None,
        **kwargs
    ) -> str:
        """
        添加分支
        
        Args:
            name: 分支名称
            executor: 执行函数
            template: 模板实例
            input_data: 输入数据
            **kwargs: 其他分支配置参数
        
        Returns:
            分支ID
        """
        branch_config = BranchConfig(
            name=name,
            executor=executor,
            template=template,
            input_data=input_data,
            **kwargs
        )
        
        self.branches.append(branch_config)
        logger.info(f"Added branch: {name} ({branch_config.branch_id})")
        
        return branch_config.branch_id
    
    def remove_branch(self, branch_id: str) -> bool:
        """
        移除分支
        
        Args:
            branch_id: 分支ID
        
        Returns:
            是否移除成功
        """
        for i, branch in enumerate(self.branches):
            if branch.branch_id == branch_id:
                removed_branch = self.branches.pop(i)
                logger.info(f"Removed branch: {removed_branch.name} ({branch_id})")
                return True
        
        logger.warning(f"Branch not found for removal: {branch_id}")
        return False
    
    def get_branch(self, branch_id: str) -> Optional[BranchConfig]:
        """
        获取分支配置
        
        Args:
            branch_id: 分支ID
        
        Returns:
            分支配置，如果不存在则返回None
        """
        for branch in self.branches:
            if branch.branch_id == branch_id:
                return branch
        return None
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行并行链（同步版本）
        
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
        if not self.branches:
            raise ConfigurationError("No branches configured for execution")
        
        logger.info(f"Starting parallel chain execution with {len(self.branches)} branches")
        
        # 初始化执行器
        with ParallelExecutor(
            max_workers=self.max_workers,
            execution_mode=self.execution_mode,
            timeout=self.timeout
        ) as executor:
            self.executor = executor
            self.is_cancelled = False
            
            try:
                # 准备分支输入数据
                self._prepare_branch_inputs(input_data)
                
                # 按优先级排序分支
                sorted_branches = sorted(self.branches, key=lambda b: b.priority, reverse=True)
                
                # 提交所有分支任务
                submitted_futures = {}
                for branch in sorted_branches:
                    if self.is_cancelled:
                        break
                    
                    future = executor.submit_branch(branch)
                    submitted_futures[branch.branch_id] = future
                
                # 等待分支完成
                if self.aggregation_strategy == AggregationStrategy.ALL:
                    done, not_done = executor.wait_for_completion(timeout=self.timeout)
                elif self.aggregation_strategy == AggregationStrategy.FIRST:
                    done, not_done = executor.wait_for_completion(timeout=self.timeout, return_when="FIRST_COMPLETED")
                else:
                    # 其他策略的处理
                    done, not_done = executor.wait_for_completion(timeout=self.timeout)
                
                # 收集结果
                results = executor.get_results()
                
                # 聚合结果
                final_result = self._aggregate_results(results, input_data)
                
                logger.info(f"Parallel chain execution completed")
                return final_result
                
            except Exception as e:
                logger.error(f"Parallel chain execution failed: {str(e)}")
                
                # 取消未完成的任务
                for branch_id in submitted_futures:
                    executor.cancel_branch(branch_id)
                
                raise Exception(f"Parallel chain execution failed: {str(e)}") from e
            
            finally:
                self.executor = None
    
    async def execute_async(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行并行链（异步版本）
        
        Args:
            input_data: 输入数据
            **kwargs: 执行参数
        
        Returns:
            执行结果
        """
        if not self.branches:
            raise ConfigurationError("No branches configured for execution")
        
        logger.info(f"Starting async parallel chain execution with {len(self.branches)} branches")
        
        try:
            # 准备分支输入数据
            self._prepare_branch_inputs(input_data)
            
            # 创建异步任务
            tasks = []
            for branch in self.branches:
                if self.is_cancelled:
                    break
                
                task = asyncio.create_task(self._execute_branch_async(branch))
                tasks.append(task)
            
            # 根据聚合策略等待任务完成
            if self.aggregation_strategy == AggregationStrategy.ALL:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            elif self.aggregation_strategy == AggregationStrategy.FIRST:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # 取消其他任务
                for task in pending:
                    task.cancel()
                results = [task.result() for task in done]
            else:
                # 其他策略的处理
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果，将异常转换为BranchResult
            processed_results = {}
            for i, result in enumerate(results):
                branch = self.branches[i]
                if isinstance(result, Exception):
                    processed_results[branch.branch_id] = BranchResult(
                        branch_id=branch.branch_id,
                        branch_name=branch.name,
                        status=BranchStatus.FAILED,
                        error=result,
                        error_message=str(result)
                    )
                else:
                    processed_results[branch.branch_id] = result
            
            # 聚合结果
            final_result = self._aggregate_results(processed_results, input_data)
            
            logger.info(f"Async parallel chain execution completed")
            return final_result
            
        except Exception as e:
            logger.error(f"Async parallel chain execution failed: {str(e)}")
            raise Exception(f"Async parallel chain execution failed: {str(e)}") from e
    
    async def _execute_branch_async(self, branch: BranchConfig) -> BranchResult:
        """
        异步执行分支
        
        Args:
            branch: 分支配置
        
        Returns:
            分支执行结果
        """
        result = BranchResult(
            branch_id=branch.branch_id,
            branch_name=branch.name,
            status=BranchStatus.PENDING,
            start_time=time.time(),
            input_data=branch.input_data,
            parameters=branch.parameters.copy()
        )
        
        try:
            logger.debug(f"Async executing branch: {branch.name}")
            result.status = BranchStatus.RUNNING
            
            # 检查执行条件
            if branch.condition:
                condition_result = branch.condition()
                if asyncio.iscoroutine(condition_result):
                    condition_result = await condition_result
                
                if not condition_result:
                    logger.info(f"Branch condition not met, skipping: {branch.name}")
                    result.status = BranchStatus.COMPLETED
                    result.output_data = None
                    return result
            
            # 异步执行分支逻辑
            if branch.executor:
                # 使用执行函数
                if asyncio.iscoroutinefunction(branch.executor):
                    output = await branch.executor(branch.input_data, **branch.parameters)
                else:
                    # 在线程池中执行同步函数
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None, 
                        branch.executor, 
                        branch.input_data,
                        branch.parameters
                    )
            elif branch.template:
                # 使用模板的异步方法
                if hasattr(branch.template, 'run_async'):
                    output = await branch.template.run_async(branch.input_data, **branch.parameters)
                else:
                    # 在线程池中执行同步方法
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None,
                        branch.template.run,
                        branch.input_data,
                        branch.parameters
                    )
            else:
                raise ConfigurationError(f"Branch '{branch.name}' has no executor or template")
            
            result.output_data = output
            result.status = BranchStatus.COMPLETED
            
            logger.debug(f"Async branch completed successfully: {branch.name}")
            
        except Exception as e:
            result.error = e
            result.error_message = str(e)
            result.status = BranchStatus.FAILED
            
            # 处理错误
            if branch.error_handler:
                try:
                    if asyncio.iscoroutinefunction(branch.error_handler):
                        await branch.error_handler(e, result)
                    else:
                        branch.error_handler(e, result)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {str(handler_error)}")
            
            if not branch.ignore_errors:
                logger.error(f"Async branch failed: {branch.name} - {str(e)}")
            else:
                logger.warning(f"Async branch failed (ignored): {branch.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
        
        return result
    
    def _prepare_branch_inputs(self, input_data: Dict[str, Any]) -> None:
        """
        准备分支输入数据
        
        Args:
            input_data: 原始输入数据
        """
        for branch in self.branches:
            if branch.input_data is None:
                # 如果分支没有特定的输入数据，使用共同的输入数据
                branch.input_data = input_data.copy()
    
    def _aggregate_results(
        self, 
        branch_results: Dict[str, BranchResult], 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        聚合分支结果
        
        Args:
            branch_results: 分支结果字典
            input_data: 原始输入数据
        
        Returns:
            聚合后的最终结果
        """
        # 基本统计信息
        total_branches = len(self.branches)
        completed_branches = sum(1 for r in branch_results.values() if r.is_successful())
        failed_branches = sum(1 for r in branch_results.values() if not r.is_successful())
        
        # 收集成功的分支输出
        successful_outputs = {}
        failed_outputs = {}
        
        for branch_id, result in branch_results.items():
            if result.is_successful():
                successful_outputs[branch_id] = result.output_data
            else:
                failed_outputs[branch_id] = {
                    "error": result.error_message,
                    "status": result.status.value
                }
        
        # 应用结果过滤器
        if self.result_filter:
            successful_outputs = self.result_filter(successful_outputs)
        
        # 应用自定义聚合器
        if self.aggregator:
            aggregated_data = self.aggregator(successful_outputs, failed_outputs, input_data)
        else:
            # 默认聚合策略
            aggregated_data = successful_outputs
        
        # 构建最终结果
        final_result = {
            "status": "completed" if failed_branches == 0 else "partial",
            "input_data": input_data,
            "aggregated_data": aggregated_data,
            "branch_results": {
                "successful": successful_outputs,
                "failed": failed_outputs
            },
            "summary": {
                "total_branches": total_branches,
                "completed_branches": completed_branches,
                "failed_branches": failed_branches,
                "success_rate": completed_branches / total_branches if total_branches > 0 else 0,
                "execution_details": [result.get_execution_summary() for result in branch_results.values()]
            }
        }
        
        return final_result
    
    def cancel(self) -> None:
        """取消链执行"""
        self.is_cancelled = True
        if self.executor:
            # 取消所有正在运行的分支
            for branch in self.branches:
                self.executor.cancel_branch(branch.branch_id)
        logger.info("Parallel chain execution cancelled")
    
    def get_example(self) -> Dict[str, Any]:
        """
        获取使用示例
        
        Returns:
            使用示例字典
        """
        return {
            "setup_parameters": {
                "branches": [
                    {
                        "name": "模型A预测",
                        "executor": "model_a_predict",
                        "priority": 1,
                        "timeout": 30,
                        "description": "使用模型A进行预测"
                    },
                    {
                        "name": "模型B预测",
                        "executor": "model_b_predict",
                        "priority": 1,
                        "timeout": 30,
                        "description": "使用模型B进行预测"
                    },
                    {
                        "name": "规则引擎",
                        "executor": "rule_engine",
                        "priority": 0,
                        "timeout": 10,
                        "description": "基于规则的预测"
                    }
                ],
                "max_workers": 3,
                "execution_mode": "thread",
                "aggregation_strategy": "all",
                "timeout": 60
            },
            "execute_parameters": {
                "input_text": "需要预测的文本数据",
                "features": ["feature1", "feature2", "feature3"]
            },
            "expected_output": {
                "status": "completed",
                "aggregated_data": {
                    "branch_0": "模型A的预测结果",
                    "branch_1": "模型B的预测结果",
                    "branch_2": "规则引擎的预测结果"
                },
                "summary": {
                    "total_branches": 3,
                    "completed_branches": 3,
                    "failed_branches": 0,
                    "success_rate": 1.0
                }
            },
            "usage_code": '''
# 使用示例
from templates.chains.parallel_chain import ParallelChainTemplate

# 定义分支执行函数
def model_a_predict(data, **kwargs):
    """模型A预测函数"""
    # 实现模型A的预测逻辑
    return f"ModelA_result_for_{data.get('input_text', '')}"

def model_b_predict(data, **kwargs):
    """模型B预测函数"""
    # 实现模型B的预测逻辑
    return f"ModelB_result_for_{data.get('input_text', '')}"

def rule_engine(data, **kwargs):
    """规则引擎函数"""
    # 实现基于规则的预测逻辑
    return f"Rule_result_for_{data.get('input_text', '')}"

# 创建和配置并行链
chain = ParallelChainTemplate()
chain.setup(
    branches=[
        {
            "name": "模型A预测",
            "executor": model_a_predict,
            "priority": 1,
            "timeout": 30
        },
        {
            "name": "模型B预测", 
            "executor": model_b_predict,
            "priority": 1,
            "timeout": 30
        },
        {
            "name": "规则引擎",
            "executor": rule_engine,
            "priority": 0,
            "timeout": 10
        }
    ],
    max_workers=3,
    execution_mode="thread",
    aggregation_strategy="all"
)

# 执行并行链
input_data = {
    "input_text": "这是一个测试文本",
    "features": ["feature1", "feature2", "feature3"]
}

result = chain.run(input_data)
print("并行执行结果:", result["aggregated_data"])
print("执行摘要:", result["summary"])

# 异步执行
import asyncio

async def main():
    result = await chain.run_async(input_data)
    print("异步并行执行结果:", result["aggregated_data"])

asyncio.run(main())
'''
        }


# 注册模板到工厂
def register_parallel_chain_template():
    """注册并行链模板到全局工厂"""
    from ..base.template_base import register_template
    register_template("parallel_chain", ParallelChainTemplate)


# 自动注册
register_parallel_chain_template()