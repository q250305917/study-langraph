"""
管道链模板（PipelineChainTemplate）

本模块提供了复杂工作流编排的管道链模板，支持顺序、并行、条件等多种模式的组合。
这是Chain模板系统的高级组件，适用于需要复杂工作流编排的企业级应用。

核心特性：
1. 工作流编排：支持复杂的工作流定义和执行
2. 多模式组合：可以组合顺序、并行、条件等多种执行模式
3. 阶段管理：将复杂任务分解为多个阶段，每个阶段可以包含多个步骤
4. 数据流控制：精确控制数据在不同阶段和步骤间的流动
5. 错误恢复：支持阶段级别的错误处理和恢复机制
6. 动态调整：运行时动态调整工作流结构和参数

设计原理：
- 管道模式：将复杂处理分解为多个阶段的管道
- 组合模式：组合不同类型的链模板构建复杂工作流
- 策略模式：支持不同的阶段执行策略
- 观察者模式：监控工作流执行状态和进度
- 命令模式：将阶段操作封装为可执行的命令

使用场景：
- 数据处理管道：ETL、数据清洗、特征工程等
- 机器学习工作流：数据预处理→训练→验证→部署
- 内容处理：文档解析→内容分析→格式转换→发布
- 业务流程：订单处理→支付→库存→发货→通知
- CI/CD流水线：代码检查→构建→测试→部署→监控
"""

import asyncio
import time
import copy
import uuid
import threading
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from .sequential_chain import SequentialChainTemplate
from .parallel_chain import ParallelChainTemplate  
from .conditional_chain import ConditionalChainTemplate
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)


class StageType(Enum):
    """阶段类型枚举"""
    SEQUENTIAL = "sequential"       # 顺序执行阶段
    PARALLEL = "parallel"           # 并行执行阶段
    CONDITIONAL = "conditional"     # 条件选择阶段
    PIPELINE = "pipeline"           # 嵌套管道阶段
    CUSTOM = "custom"               # 自定义阶段


class StageStatus(Enum):
    """阶段状态枚举"""
    PENDING = "pending"             # 等待执行
    RUNNING = "running"             # 执行中
    COMPLETED = "completed"         # 执行完成
    FAILED = "failed"               # 执行失败
    SKIPPED = "skipped"             # 跳过执行
    CANCELLED = "cancelled"         # 取消执行
    PAUSED = "paused"               # 暂停执行


class DataFlowMode(Enum):
    """数据流模式枚举"""
    ACCUMULATE = "accumulate"       # 累积模式：保留所有阶段的输出
    PIPELINE = "pipeline"           # 管道模式：只传递上一阶段的输出
    MERGE = "merge"                 # 合并模式：合并所有阶段的输出
    CUSTOM = "custom"               # 自定义模式：使用自定义数据流函数


class RecoveryStrategy(Enum):
    """恢复策略枚举"""
    FAIL_FAST = "fail_fast"         # 快速失败
    RETRY_STAGE = "retry_stage"     # 重试当前阶段
    SKIP_STAGE = "skip_stage"       # 跳过当前阶段
    FALLBACK = "fallback"           # 执行回退逻辑
    MANUAL = "manual"               # 手动干预


@dataclass
class StageConfig:
    """
    阶段配置类
    
    定义管道中每个阶段的配置信息，包括类型、模板、参数、
    依赖关系、错误处理等。
    """
    
    # === 基本信息 ===
    name: str                                    # 阶段名称
    stage_type: StageType                        # 阶段类型
    description: str = ""                        # 阶段描述
    stage_id: Optional[str] = None              # 阶段唯一标识
    
    # === 执行配置 ===
    template: Optional[TemplateBase] = None      # 使用的模板实例
    template_config: Optional[Dict[str, Any]] = None  # 模板配置
    executor: Optional[Callable] = None          # 自定义执行函数
    
    # === 参数配置 ===
    input_mapping: Dict[str, str] = field(default_factory=dict)     # 输入映射
    output_mapping: Dict[str, str] = field(default_factory=dict)    # 输出映射
    parameters: Dict[str, Any] = field(default_factory=dict)        # 阶段参数
    
    # === 依赖关系 ===
    dependencies: List[str] = field(default_factory=list)           # 依赖的阶段ID
    wait_for_all: bool = True                    # 是否等待所有依赖完成
    
    # === 执行控制 ===
    priority: int = 0                            # 执行优先级
    timeout: Optional[float] = None              # 超时时间
    retry_count: int = 0                         # 重试次数
    retry_delay: float = 1.0                     # 重试延迟
    
    # === 条件控制 ===
    condition: Optional[Callable] = None         # 执行条件
    skip_on_condition: bool = True               # 条件不满足时是否跳过
    
    # === 错误处理 ===
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.FAIL_FAST
    fallback_stage: Optional[str] = None         # 回退阶段ID
    error_handler: Optional[Callable] = None     # 错误处理函数
    
    # === 性能配置 ===
    cache_enabled: bool = False                  # 是否启用缓存
    resource_limit: Dict[str, Any] = field(default_factory=dict)    # 资源限制
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.stage_id:
            self.stage_id = f"stage_{uuid.uuid4().hex[:8]}"
        
        # 验证配置有效性
        if not self.template and not self.executor:
            raise ConfigurationError("Stage must have either template or executor")


@dataclass
class StageResult:
    """
    阶段执行结果类
    
    包含阶段执行的完整信息。
    """
    
    # === 基本信息 ===
    stage_id: str                                # 阶段ID
    stage_name: str                              # 阶段名称
    stage_type: StageType                        # 阶段类型
    status: StageStatus                          # 执行状态
    
    # === 执行信息 ===
    start_time: Optional[float] = None           # 开始时间
    end_time: Optional[float] = None             # 结束时间
    execution_time: Optional[float] = None       # 执行时长
    retry_count: int = 0                         # 实际重试次数
    
    # === 数据信息 ===
    input_data: Any = None                       # 输入数据
    output_data: Any = None                      # 输出数据
    intermediate_data: Dict[str, Any] = field(default_factory=dict)  # 中间数据
    
    # === 错误信息 ===
    error: Optional[Exception] = None            # 错误异常
    error_message: Optional[str] = None          # 错误消息
    recovery_actions: List[str] = field(default_factory=list)       # 恢复动作
    
    # === 性能信息 ===
    memory_usage: Optional[float] = None         # 内存使用量
    cpu_usage: Optional[float] = None            # CPU使用率
    cache_hit: bool = False                      # 是否命中缓存
    
    # === 子结果 ===
    sub_results: List[Dict[str, Any]] = field(default_factory=list) # 子阶段结果
    
    def is_successful(self) -> bool:
        """判断阶段是否执行成功"""
        return self.status == StageStatus.COMPLETED
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "stage_type": self.stage_type.value,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "success": self.is_successful(),
            "retry_count": self.retry_count,
            "cache_hit": self.cache_hit,
            "error_message": self.error_message,
            "sub_results_count": len(self.sub_results)
        }


class PipelineContext:
    """
    管道执行上下文
    
    管理管道执行过程中的数据流、状态信息和共享资源。
    提供阶段间的数据传递、状态跟踪、错误处理等核心功能。
    """
    
    def __init__(
        self, 
        initial_data: Optional[Dict[str, Any]] = None,
        data_flow_mode: DataFlowMode = DataFlowMode.PIPELINE
    ):
        """
        初始化管道上下文
        
        Args:
            initial_data: 初始数据
            data_flow_mode: 数据流模式
        """
        self.data_flow_mode = data_flow_mode
        
        # 数据管理
        self.initial_data = initial_data or {}
        self.current_data = self.initial_data.copy()
        self.stage_outputs: Dict[str, Any] = {}          # 各阶段输出
        self.intermediate_data: Dict[str, Any] = {}      # 中间数据
        self.shared_resources: Dict[str, Any] = {}       # 共享资源
        
        # 执行状态
        self.stage_results: Dict[str, StageResult] = {}
        self.execution_order: List[str] = []             # 执行顺序
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # 依赖图
        
        # 统计信息
        self.total_stages = 0
        self.completed_stages = 0
        self.failed_stages = 0
        self.skipped_stages = 0
        
        # 同步控制
        self._lock = threading.Lock()
        
        logger.debug("Initialized PipelineContext")
    
    def set_data(self, key: str, value: Any) -> None:
        """设置当前数据"""
        with self._lock:
            self.current_data[key] = value
            logger.debug(f"Set context data: {key}")
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """获取当前数据"""
        with self._lock:
            return self.current_data.get(key, default)
    
    def update_data(self, data: Dict[str, Any]) -> None:
        """批量更新当前数据"""
        with self._lock:
            if self.data_flow_mode == DataFlowMode.ACCUMULATE:
                # 累积模式：保留所有数据
                self.current_data.update(data)
            elif self.data_flow_mode == DataFlowMode.PIPELINE:
                # 管道模式：替换数据
                self.current_data = data.copy()
            elif self.data_flow_mode == DataFlowMode.MERGE:
                # 合并模式：智能合并
                self._merge_data(data)
            
            logger.debug(f"Updated context data with {len(data)} items")
    
    def _merge_data(self, new_data: Dict[str, Any]) -> None:
        """智能合并数据"""
        for key, value in new_data.items():
            if key in self.current_data:
                # 如果键已存在，尝试合并
                existing_value = self.current_data[key]
                if isinstance(existing_value, dict) and isinstance(value, dict):
                    existing_value.update(value)
                elif isinstance(existing_value, list) and isinstance(value, list):
                    existing_value.extend(value)
                else:
                    # 其他情况直接替换
                    self.current_data[key] = value
            else:
                self.current_data[key] = value
    
    def add_stage_result(self, result: StageResult) -> None:
        """添加阶段结果"""
        with self._lock:
            self.stage_results[result.stage_id] = result
            self.execution_order.append(result.stage_id)
            
            # 更新统计
            if result.status == StageStatus.COMPLETED:
                self.completed_stages += 1
                # 保存阶段输出
                if result.output_data is not None:
                    self.stage_outputs[result.stage_id] = result.output_data
            elif result.status == StageStatus.FAILED:
                self.failed_stages += 1
            elif result.status == StageStatus.SKIPPED:
                self.skipped_stages += 1
            
            logger.debug(f"Added stage result: {result.stage_name} ({result.status.value})")
    
    def get_stage_result(self, stage_id: str) -> Optional[StageResult]:
        """获取阶段结果"""
        with self._lock:
            return self.stage_results.get(stage_id)
    
    def get_stage_output(self, stage_id: str) -> Any:
        """获取阶段输出"""
        with self._lock:
            return self.stage_outputs.get(stage_id)
    
    def has_failed_stages(self) -> bool:
        """检查是否有失败的阶段"""
        with self._lock:
            return self.failed_stages > 0
    
    def add_dependency(self, stage_id: str, dependency_id: str) -> None:
        """添加依赖关系"""
        with self._lock:
            self.dependency_graph[stage_id].add(dependency_id)
    
    def get_dependencies(self, stage_id: str) -> Set[str]:
        """获取阶段依赖"""
        with self._lock:
            return self.dependency_graph.get(stage_id, set())
    
    def are_dependencies_satisfied(self, stage_id: str) -> bool:
        """检查依赖是否满足"""
        dependencies = self.get_dependencies(stage_id)
        for dep_id in dependencies:
            dep_result = self.get_stage_result(dep_id)
            if not dep_result or not dep_result.is_successful():
                return False
        return True
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        with self._lock:
            return {
                "total_stages": self.total_stages,
                "completed_stages": self.completed_stages,
                "failed_stages": self.failed_stages,
                "skipped_stages": self.skipped_stages,
                "success_rate": self.completed_stages / self.total_stages if self.total_stages > 0 else 0,
                "execution_order": self.execution_order.copy(),
                "data_keys": list(self.current_data.keys()),
                "stage_results": [result.get_execution_summary() for result in self.stage_results.values()]
            }


class PipelineChainTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    管道链模板类
    
    实现复杂工作流编排的管道链。支持多种执行模式的组合，
    提供企业级的工作流管理功能。
    
    核心功能：
    1. 阶段管理：动态添加、删除、配置工作流阶段
    2. 依赖解析：自动解析和管理阶段间的依赖关系
    3. 执行编排：智能调度和执行各个阶段
    4. 数据流控制：灵活的数据流管理和传递
    5. 错误恢复：完善的错误处理和恢复机制
    6. 性能监控：全面的执行监控和性能分析
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化管道链模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config)
        
        # 阶段配置
        self.stages: List[StageConfig] = []
        self.stage_templates: Dict[str, TemplateBase] = {}  # 阶段模板缓存
        
        # 执行配置
        self.data_flow_mode = DataFlowMode.PIPELINE
        self.max_parallel_stages = 4
        self.global_timeout = None
        
        # 执行状态
        self.context: Optional[PipelineContext] = None
        self.is_cancelled = False
        self.is_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始为非暂停状态
        
        # 线程池
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        
        logger.info(f"Initialized PipelineChainTemplate: {self.config.name}")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="PipelineChainTemplate",
            version="1.0.0",
            description="复杂工作流编排的管道链模板",
            template_type=TemplateType.CHAIN,
            async_enabled=True,
            cache_enabled=False
        )
        
        # 定义参数模式
        config.add_parameter(
            "stages", list, required=True,
            description="管道阶段配置列表",
            examples=[
                [
                    {"name": "stage1", "stage_type": "sequential", "template_config": {...}},
                    {"name": "stage2", "stage_type": "parallel", "template_config": {...}}
                ]
            ]
        )
        
        config.add_parameter(
            "data_flow_mode", str, required=False, default="pipeline",
            description="数据流模式",
            constraints={"allowed_values": ["accumulate", "pipeline", "merge", "custom"]}
        )
        
        config.add_parameter(
            "max_parallel_stages", int, required=False, default=4,
            description="最大并行阶段数",
            constraints={"min_value": 1, "max_value": 16}
        )
        
        config.add_parameter(
            "global_timeout", float, required=False,
            description="全局超时时间（秒）",
            constraints={"min_value": 0}
        )
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置管道参数
        
        Args:
            **parameters: 管道配置参数
                - stages: 阶段配置列表
                - data_flow_mode: 数据流模式
                - max_parallel_stages: 最大并行阶段数
                - global_timeout: 全局超时时间
        
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        # 验证参数
        self.validate_parameters(parameters)
        
        # 保存设置参数
        self._setup_parameters = parameters.copy()
        
        # 配置阶段
        stages_config = parameters.get("stages", [])
        self._configure_stages(stages_config)
        
        # 配置数据流模式
        data_flow_mode_str = parameters.get("data_flow_mode", "pipeline")
        self.data_flow_mode = DataFlowMode(data_flow_mode_str)
        
        # 配置并行设置
        self.max_parallel_stages = parameters.get("max_parallel_stages", 4)
        
        # 配置超时
        self.global_timeout = parameters.get("global_timeout")
        
        # 配置线程池
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_parallel_stages)
        
        # 更新状态
        self.status = self.status.__class__.CONFIGURED
        
        logger.info(f"Configured PipelineChain with {len(self.stages)} stages")
    
    def _configure_stages(self, stages_config: List[Dict[str, Any]]) -> None:
        """
        配置阶段列表
        
        Args:
            stages_config: 阶段配置列表
        """
        self.stages.clear()
        self.stage_templates.clear()
        
        for i, stage_dict in enumerate(stages_config):
            try:
                # 创建阶段配置
                stage_config = self._create_stage_config(stage_dict, i)
                self.stages.append(stage_config)
                
                # 创建阶段模板
                if stage_config.template_config:
                    template = self._create_stage_template(stage_config)
                    self.stage_templates[stage_config.stage_id] = template
                
                logger.debug(f"Configured stage: {stage_config.name}")
                
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to configure stage {i}: {str(e)}",
                    error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                    cause=e
                )
    
    def _create_stage_config(self, stage_dict: Dict[str, Any], index: int) -> StageConfig:
        """
        创建阶段配置
        
        Args:
            stage_dict: 阶段字典配置
            index: 阶段索引
        
        Returns:
            阶段配置对象
        """
        # 基本信息
        name = stage_dict.get("name", f"stage_{index}")
        stage_type_str = stage_dict.get("stage_type", "sequential")
        stage_type = StageType(stage_type_str)
        description = stage_dict.get("description", "")
        stage_id = stage_dict.get("stage_id", f"stage_{index}_{uuid.uuid4().hex[:8]}")
        
        # 执行配置
        template = stage_dict.get("template")
        template_config = stage_dict.get("template_config")
        executor = stage_dict.get("executor")
        
        # 参数配置
        input_mapping = stage_dict.get("input_mapping", {})
        output_mapping = stage_dict.get("output_mapping", {})
        parameters = stage_dict.get("parameters", {})
        
        # 依赖关系
        dependencies = stage_dict.get("dependencies", [])
        wait_for_all = stage_dict.get("wait_for_all", True)
        
        # 执行控制
        priority = stage_dict.get("priority", 0)
        timeout = stage_dict.get("timeout")
        retry_count = stage_dict.get("retry_count", 0)
        retry_delay = stage_dict.get("retry_delay", 1.0)
        
        # 条件控制
        condition = stage_dict.get("condition")
        skip_on_condition = stage_dict.get("skip_on_condition", True)
        
        # 错误处理
        recovery_strategy_str = stage_dict.get("recovery_strategy", "fail_fast")
        recovery_strategy = RecoveryStrategy(recovery_strategy_str)
        fallback_stage = stage_dict.get("fallback_stage")
        error_handler = stage_dict.get("error_handler")
        
        # 性能配置
        cache_enabled = stage_dict.get("cache_enabled", False)
        resource_limit = stage_dict.get("resource_limit", {})
        
        return StageConfig(
            name=name,
            stage_type=stage_type,
            description=description,
            stage_id=stage_id,
            template=template,
            template_config=template_config,
            executor=executor,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            parameters=parameters,
            dependencies=dependencies,
            wait_for_all=wait_for_all,
            priority=priority,
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            condition=condition,
            skip_on_condition=skip_on_condition,
            recovery_strategy=recovery_strategy,
            fallback_stage=fallback_stage,
            error_handler=error_handler,
            cache_enabled=cache_enabled,
            resource_limit=resource_limit
        )
    
    def _create_stage_template(self, stage_config: StageConfig) -> TemplateBase:
        """
        创建阶段模板
        
        Args:
            stage_config: 阶段配置
        
        Returns:
            模板实例
        """
        if stage_config.template:
            return stage_config.template
        
        if not stage_config.template_config:
            raise ConfigurationError(f"Stage '{stage_config.name}' has no template or template_config")
        
        # 根据阶段类型创建对应的模板
        if stage_config.stage_type == StageType.SEQUENTIAL:
            template = SequentialChainTemplate()
        elif stage_config.stage_type == StageType.PARALLEL:
            template = ParallelChainTemplate()
        elif stage_config.stage_type == StageType.CONDITIONAL:
            template = ConditionalChainTemplate()
        elif stage_config.stage_type == StageType.PIPELINE:
            template = PipelineChainTemplate()
        else:
            raise ConfigurationError(f"Unsupported stage type: {stage_config.stage_type}")
        
        # 设置模板配置
        template.setup(**stage_config.template_config)
        
        return template
    
    def add_stage(
        self,
        name: str,
        stage_type: StageType,
        template: Optional[TemplateBase] = None,
        template_config: Optional[Dict[str, Any]] = None,
        executor: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        添加阶段
        
        Args:
            name: 阶段名称
            stage_type: 阶段类型
            template: 模板实例
            template_config: 模板配置
            executor: 自定义执行函数
            dependencies: 依赖的阶段ID列表
            **kwargs: 其他阶段配置参数
        
        Returns:
            阶段ID
        """
        stage_config = StageConfig(
            name=name,
            stage_type=stage_type,
            template=template,
            template_config=template_config,
            executor=executor,
            dependencies=dependencies or [],
            **kwargs
        )
        
        self.stages.append(stage_config)
        
        # 创建模板
        if template_config and not template:
            stage_template = self._create_stage_template(stage_config)
            self.stage_templates[stage_config.stage_id] = stage_template
        
        logger.info(f"Added stage: {name} ({stage_config.stage_id})")
        return stage_config.stage_id
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行管道链（同步版本）
        
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
        if not self.stages:
            raise ConfigurationError("No stages configured for execution")
        
        logger.info(f"Starting pipeline chain execution with {len(self.stages)} stages")
        
        # 初始化执行上下文
        self.context = PipelineContext(input_data, self.data_flow_mode)
        self.context.total_stages = len(self.stages)
        self.is_cancelled = False
        self.is_paused = False
        
        # 构建依赖图
        self._build_dependency_graph()
        
        try:
            # 拓扑排序确定执行顺序
            execution_plan = self._create_execution_plan()
            
            # 依次执行各个阶段
            for stage_group in execution_plan:
                if self.is_cancelled:
                    logger.info("Pipeline execution cancelled")
                    break
                
                # 检查暂停状态
                self._pause_event.wait()
                
                # 并行执行当前组中的阶段
                self._execute_stage_group(stage_group)
            
            # 构建最终结果
            final_result = {
                "status": "completed" if not self.context.has_failed_stages() else "partial",
                "input_data": input_data,
                "output_data": self.context.current_data.copy(),
                "stage_outputs": self.context.stage_outputs.copy(),
                "summary": self.context.get_execution_summary()
            }
            
            logger.info(f"Pipeline chain execution completed")
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline chain execution failed: {str(e)}")
            
            error_result = {
                "status": "failed",
                "error": str(e),
                "input_data": input_data,
                "output_data": self.context.current_data.copy() if self.context else {},
                "summary": self.context.get_execution_summary() if self.context else {}
            }
            
            raise Exception(f"Pipeline chain execution failed: {str(e)}") from e
    
    async def execute_async(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行管道链（异步版本）
        
        Args:
            input_data: 输入数据
            **kwargs: 执行参数
        
        Returns:
            执行结果
        """
        if not self.stages:
            raise ConfigurationError("No stages configured for execution")
        
        logger.info(f"Starting async pipeline chain execution with {len(self.stages)} stages")
        
        # 初始化执行上下文
        self.context = PipelineContext(input_data, self.data_flow_mode)
        self.context.total_stages = len(self.stages)
        self.is_cancelled = False
        self.is_paused = False
        
        # 构建依赖图
        self._build_dependency_graph()
        
        try:
            # 拓扑排序确定执行顺序
            execution_plan = self._create_execution_plan()
            
            # 依次执行各个阶段
            for stage_group in execution_plan:
                if self.is_cancelled:
                    logger.info("Async pipeline execution cancelled")
                    break
                
                # 检查暂停状态
                while self.is_paused:
                    await asyncio.sleep(0.1)
                
                # 异步并行执行当前组中的阶段
                await self._execute_stage_group_async(stage_group)
            
            # 构建最终结果
            final_result = {
                "status": "completed" if not self.context.has_failed_stages() else "partial",
                "input_data": input_data,
                "output_data": self.context.current_data.copy(),
                "stage_outputs": self.context.stage_outputs.copy(),
                "summary": self.context.get_execution_summary()
            }
            
            logger.info(f"Async pipeline chain execution completed")
            return final_result
            
        except Exception as e:
            logger.error(f"Async pipeline chain execution failed: {str(e)}")
            raise Exception(f"Async pipeline chain execution failed: {str(e)}") from e
    
    def _build_dependency_graph(self) -> None:
        """构建依赖图"""
        for stage in self.stages:
            for dep_id in stage.dependencies:
                self.context.add_dependency(stage.stage_id, dep_id)
    
    def _create_execution_plan(self) -> List[List[StageConfig]]:
        """
        创建执行计划（拓扑排序）
        
        Returns:
            执行计划，每个元素是可以并行执行的阶段组
        """
        # 计算入度
        in_degree = defaultdict(int)
        stage_map = {stage.stage_id: stage for stage in self.stages}
        
        for stage in self.stages:
            for dep_id in stage.dependencies:
                in_degree[stage.stage_id] += 1
        
        # 拓扑排序
        execution_plan = []
        queue = deque()
        
        # 找到所有入度为0的阶段
        for stage in self.stages:
            if in_degree[stage.stage_id] == 0:
                queue.append(stage)
        
        while queue:
            # 当前批次可以并行执行的阶段
            current_batch = []
            next_queue = deque()
            
            # 处理当前批次
            while queue:
                stage = queue.popleft()
                current_batch.append(stage)
                
                # 更新依赖此阶段的其他阶段的入度
                for other_stage in self.stages:
                    if stage.stage_id in other_stage.dependencies:
                        in_degree[other_stage.stage_id] -= 1
                        if in_degree[other_stage.stage_id] == 0:
                            next_queue.append(other_stage)
            
            if current_batch:
                execution_plan.append(current_batch)
            
            queue = next_queue
        
        # 检查是否有循环依赖
        total_scheduled = sum(len(batch) for batch in execution_plan)
        if total_scheduled != len(self.stages):
            raise ConfigurationError("Circular dependency detected in pipeline stages")
        
        return execution_plan
    
    def _execute_stage_group(self, stage_group: List[StageConfig]) -> None:
        """
        执行阶段组（同步版本）
        
        Args:
            stage_group: 阶段组
        """
        if len(stage_group) == 1:
            # 单个阶段，直接执行
            stage_result = self._execute_stage(stage_group[0])
            self._process_stage_result(stage_result)
        else:
            # 多个阶段，并行执行
            futures = []
            for stage in stage_group:
                future = self.thread_pool.submit(self._execute_stage, stage)
                futures.append((stage, future))
            
            # 等待所有阶段完成
            for stage, future in futures:
                try:
                    stage_result = future.result(timeout=stage.timeout)
                    self._process_stage_result(stage_result)
                except Exception as e:
                    # 创建失败的阶段结果
                    stage_result = StageResult(
                        stage_id=stage.stage_id,
                        stage_name=stage.name,
                        stage_type=stage.stage_type,
                        status=StageStatus.FAILED,
                        error=e,
                        error_message=str(e)
                    )
                    self._process_stage_result(stage_result)
    
    async def _execute_stage_group_async(self, stage_group: List[StageConfig]) -> None:
        """
        执行阶段组（异步版本）
        
        Args:
            stage_group: 阶段组
        """
        if len(stage_group) == 1:
            # 单个阶段，直接执行
            stage_result = await self._execute_stage_async(stage_group[0])
            self._process_stage_result(stage_result)
        else:
            # 多个阶段，并行执行
            tasks = []
            for stage in stage_group:
                task = asyncio.create_task(self._execute_stage_async(stage))
                tasks.append(task)
            
            # 等待所有阶段完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # 创建失败的阶段结果
                    stage = stage_group[i]
                    stage_result = StageResult(
                        stage_id=stage.stage_id,
                        stage_name=stage.name,
                        stage_type=stage.stage_type,
                        status=StageStatus.FAILED,
                        error=result,
                        error_message=str(result)
                    )
                else:
                    stage_result = result
                
                self._process_stage_result(stage_result)
    
    def _execute_stage(self, stage: StageConfig) -> StageResult:
        """
        执行单个阶段（同步版本）
        
        Args:
            stage: 阶段配置
        
        Returns:
            阶段执行结果
        """
        result = StageResult(
            stage_id=stage.stage_id,
            stage_name=stage.name,
            stage_type=stage.stage_type,
            status=StageStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            logger.debug(f"Executing stage: {stage.name}")
            result.status = StageStatus.RUNNING
            
            # 检查执行条件
            if stage.condition and not stage.condition(self.context):
                if stage.skip_on_condition:
                    logger.info(f"Stage condition not met, skipping: {stage.name}")
                    result.status = StageStatus.SKIPPED
                    result.output_data = None
                    return result
                else:
                    raise ConfigurationError(f"Stage condition not met: {stage.name}")
            
            # 检查依赖
            if not self.context.are_dependencies_satisfied(stage.stage_id):
                raise ConfigurationError(f"Stage dependencies not satisfied: {stage.name}")
            
            # 准备输入数据
            stage_input = self._prepare_stage_input(stage)
            result.input_data = stage_input
            
            # 执行阶段逻辑
            if stage.executor:
                # 使用自定义执行函数
                output = stage.executor(stage_input, self.context, **stage.parameters)
            elif stage.stage_id in self.stage_templates:
                # 使用模板
                template = self.stage_templates[stage.stage_id]
                output = template.run(stage_input, **stage.parameters)
            elif stage.template:
                # 使用直接提供的模板
                output = stage.template.run(stage_input, **stage.parameters)
            else:
                raise ConfigurationError(f"Stage '{stage.name}' has no executor or template")
            
            # 应用输出映射
            mapped_output = self._apply_output_mapping(stage, output)
            result.output_data = mapped_output
            result.status = StageStatus.COMPLETED
            
            logger.debug(f"Stage completed successfully: {stage.name}")
            
        except Exception as e:
            result.error = e
            result.error_message = str(e)
            result.status = StageStatus.FAILED
            
            # 处理错误
            if stage.error_handler:
                try:
                    stage.error_handler(e, result, self.context)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {str(handler_error)}")
            
            logger.error(f"Stage failed: {stage.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
        
        return result
    
    async def _execute_stage_async(self, stage: StageConfig) -> StageResult:
        """
        执行单个阶段（异步版本）
        
        Args:
            stage: 阶段配置
        
        Returns:
            阶段执行结果
        """
        result = StageResult(
            stage_id=stage.stage_id,
            stage_name=stage.name,
            stage_type=stage.stage_type,
            status=StageStatus.PENDING,
            start_time=time.time()
        )
        
        try:
            logger.debug(f"Async executing stage: {stage.name}")
            result.status = StageStatus.RUNNING
            
            # 检查执行条件
            if stage.condition:
                condition_result = stage.condition(self.context)
                if asyncio.iscoroutine(condition_result):
                    condition_result = await condition_result
                
                if not condition_result:
                    if stage.skip_on_condition:
                        logger.info(f"Stage condition not met, skipping: {stage.name}")
                        result.status = StageStatus.SKIPPED
                        result.output_data = None
                        return result
                    else:
                        raise ConfigurationError(f"Stage condition not met: {stage.name}")
            
            # 检查依赖
            if not self.context.are_dependencies_satisfied(stage.stage_id):
                raise ConfigurationError(f"Stage dependencies not satisfied: {stage.name}")
            
            # 准备输入数据
            stage_input = self._prepare_stage_input(stage)
            result.input_data = stage_input
            
            # 异步执行阶段逻辑
            if stage.executor:
                # 使用自定义执行函数
                if asyncio.iscoroutinefunction(stage.executor):
                    output = await stage.executor(stage_input, self.context, **stage.parameters)
                else:
                    # 在线程池中执行同步函数
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        self.thread_pool,
                        stage.executor,
                        stage_input,
                        self.context,
                        stage.parameters
                    )
            elif stage.stage_id in self.stage_templates:
                # 使用模板的异步方法
                template = self.stage_templates[stage.stage_id]
                if hasattr(template, 'run_async'):
                    output = await template.run_async(stage_input, **stage.parameters)
                else:
                    # 在线程池中执行同步方法
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        self.thread_pool,
                        template.run,
                        stage_input,
                        stage.parameters
                    )
            elif stage.template:
                # 使用直接提供的模板
                if hasattr(stage.template, 'run_async'):
                    output = await stage.template.run_async(stage_input, **stage.parameters)
                else:
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        self.thread_pool,
                        stage.template.run,
                        stage_input,
                        stage.parameters
                    )
            else:
                raise ConfigurationError(f"Stage '{stage.name}' has no executor or template")
            
            # 应用输出映射
            mapped_output = self._apply_output_mapping(stage, output)
            result.output_data = mapped_output
            result.status = StageStatus.COMPLETED
            
            logger.debug(f"Async stage completed successfully: {stage.name}")
            
        except Exception as e:
            result.error = e
            result.error_message = str(e)
            result.status = StageStatus.FAILED
            
            # 处理错误
            if stage.error_handler:
                try:
                    if asyncio.iscoroutinefunction(stage.error_handler):
                        await stage.error_handler(e, result, self.context)
                    else:
                        stage.error_handler(e, result, self.context)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {str(handler_error)}")
            
            logger.error(f"Async stage failed: {stage.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
        
        return result
    
    def _prepare_stage_input(self, stage: StageConfig) -> Any:
        """
        准备阶段输入数据
        
        Args:
            stage: 阶段配置
        
        Returns:
            阶段输入数据
        """
        if not stage.input_mapping:
            # 如果没有输入映射，返回当前上下文数据
            return self.context.current_data.copy()
        
        # 应用输入映射
        stage_input = {}
        for target_key, source_key in stage.input_mapping.items():
            if source_key in self.context.current_data:
                stage_input[target_key] = self.context.current_data[source_key]
            elif '.' in source_key:
                # 支持嵌套路径，如 "stage1.output.result"
                value = self._get_nested_value(source_key)
                if value is not None:
                    stage_input[target_key] = value
            else:
                logger.warning(f"Input mapping key '{source_key}' not found for stage '{stage.name}'")
        
        return stage_input
    
    def _apply_output_mapping(self, stage: StageConfig, output: Any) -> Any:
        """
        应用输出映射
        
        Args:
            stage: 阶段配置
            output: 原始输出
        
        Returns:
            映射后的输出
        """
        if not stage.output_mapping:
            return output
        
        if not isinstance(output, dict):
            # 如果输出不是字典，无法应用映射
            return output
        
        mapped_output = {}
        for source_key, target_key in stage.output_mapping.items():
            if source_key in output:
                mapped_output[target_key] = output[source_key]
            else:
                logger.warning(f"Output mapping key '{source_key}' not found in stage output")
        
        return mapped_output
    
    def _get_nested_value(self, path: str) -> Any:
        """
        获取嵌套路径的值
        
        Args:
            path: 嵌套路径，如 "stage1.output.result"
        
        Returns:
            路径对应的值
        """
        parts = path.split('.')
        
        # 检查是否是阶段输出路径
        if len(parts) >= 2 and parts[0] in self.context.stage_outputs:
            value = self.context.stage_outputs[parts[0]]
            for part in parts[1:]:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        
        # 检查当前数据
        value = self.context.current_data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _process_stage_result(self, result: StageResult) -> None:
        """
        处理阶段结果
        
        Args:
            result: 阶段结果
        """
        # 添加到上下文
        self.context.add_stage_result(result)
        
        # 更新数据流
        if result.is_successful() and result.output_data is not None:
            self.context.update_data(result.output_data)
    
    def pause(self) -> None:
        """暂停管道执行"""
        self.is_paused = True
        self._pause_event.clear()
        logger.info("Pipeline execution paused")
    
    def resume(self) -> None:
        """恢复管道执行"""
        self.is_paused = False
        self._pause_event.set()
        logger.info("Pipeline execution resumed")
    
    def cancel(self) -> None:
        """取消管道执行"""
        self.is_cancelled = True
        logger.info("Pipeline execution cancelled")
    
    def get_example(self) -> Dict[str, Any]:
        """
        获取使用示例
        
        Returns:
            使用示例字典
        """
        return {
            "setup_parameters": {
                "stages": [
                    {
                        "name": "数据预处理阶段",
                        "stage_type": "sequential",
                        "template_config": {
                            "steps": [
                                {"name": "数据清洗", "executor": "clean_data"},
                                {"name": "数据验证", "executor": "validate_data"}
                            ]
                        },
                        "output_mapping": {"processed_data": "clean_data"}
                    },
                    {
                        "name": "模型推理阶段",
                        "stage_type": "parallel",
                        "template_config": {
                            "branches": [
                                {"name": "模型A", "executor": "model_a_predict"},
                                {"name": "模型B", "executor": "model_b_predict"}
                            ]
                        },
                        "dependencies": ["stage_0"],
                        "input_mapping": {"data": "clean_data"}
                    },
                    {
                        "name": "结果聚合阶段",
                        "stage_type": "conditional",
                        "template_config": {
                            "branches": [
                                {
                                    "name": "高置信度分支",
                                    "condition": {
                                        "type": "value",
                                        "field_path": "confidence",
                                        "operator": "gt",
                                        "value": 0.8
                                    },
                                    "executor": "high_confidence_process"
                                }
                            ],
                            "default_branch": {
                                "name": "普通处理",
                                "executor": "normal_process"
                            }
                        },
                        "dependencies": ["stage_1"]
                    }
                ],
                "data_flow_mode": "pipeline",
                "max_parallel_stages": 4,
                "global_timeout": 300
            },
            "execute_parameters": {
                "raw_data": "原始数据",
                "config": {"model_version": "v1.0"}
            },
            "expected_output": {
                "status": "completed",
                "output_data": "最终处理结果",
                "stage_outputs": {
                    "stage_0": "预处理结果",
                    "stage_1": "推理结果",
                    "stage_2": "聚合结果"
                },
                "summary": {
                    "total_stages": 3,
                    "completed_stages": 3,
                    "success_rate": 1.0
                }
            },
            "usage_code": '''
# 使用示例
from templates.chains.pipeline_chain import PipelineChainTemplate

# 定义各个执行函数
def clean_data(data, **kwargs):
    """数据清洗函数"""
    return {"clean_data": f"cleaned_{data.get('raw_data', '')}"}

def validate_data(data, **kwargs):
    """数据验证函数"""
    return {"validated": True, "clean_data": data.get("clean_data")}

def model_a_predict(data, **kwargs):
    """模型A预测函数"""
    return {"prediction_a": f"result_a_for_{data.get('data', '')}"}

def model_b_predict(data, **kwargs):
    """模型B预测函数"""
    return {"prediction_b": f"result_b_for_{data.get('data', '')}"}

def high_confidence_process(data, **kwargs):
    """高置信度处理函数"""
    return {"final_result": "high_confidence_result"}

def normal_process(data, **kwargs):
    """普通处理函数"""
    return {"final_result": "normal_result"}

# 创建和配置管道链
pipeline = PipelineChainTemplate()
pipeline.setup(
    stages=[
        {
            "name": "数据预处理阶段",
            "stage_type": "sequential",
            "template_config": {
                "steps": [
                    {"name": "数据清洗", "executor": clean_data},
                    {"name": "数据验证", "executor": validate_data}
                ]
            }
        },
        {
            "name": "模型推理阶段", 
            "stage_type": "parallel",
            "template_config": {
                "branches": [
                    {"name": "模型A", "executor": model_a_predict},
                    {"name": "模型B", "executor": model_b_predict}
                ]
            },
            "dependencies": ["stage_0"]
        },
        {
            "name": "结果聚合阶段",
            "stage_type": "conditional",
            "template_config": {
                "branches": [
                    {
                        "name": "高置信度分支",
                        "condition": {
                            "type": "expression",
                            "expression": "confidence > 0.8"
                        },
                        "executor": high_confidence_process
                    }
                ],
                "default_branch": {
                    "name": "普通处理",
                    "executor": normal_process
                }
            },
            "dependencies": ["stage_1"]
        }
    ],
    data_flow_mode="pipeline",
    max_parallel_stages=2
)

# 执行管道
input_data = {
    "raw_data": "sample_input_data",
    "confidence": 0.9
}

result = pipeline.run(input_data)
print("管道执行结果:", result["output_data"])
print("各阶段输出:", result["stage_outputs"])
print("执行摘要:", result["summary"])

# 异步执行
import asyncio

async def main():
    result = await pipeline.run_async(input_data)
    print("异步执行结果:", result["output_data"])

asyncio.run(main())
'''
        }
    
    def __del__(self):
        """析构函数，清理资源"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)


# 注册模板到工厂
def register_pipeline_chain_template():
    """注册管道链模板到全局工厂"""
    from ..base.template_base import register_template
    register_template("pipeline_chain", PipelineChainTemplate)


# 自动注册
register_pipeline_chain_template()