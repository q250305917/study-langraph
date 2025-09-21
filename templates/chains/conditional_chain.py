"""
条件链模板（ConditionalChainTemplate）

本模块提供了基于条件分支执行的链模板，支持根据条件动态选择执行路径。
这是Chain模板系统的决策组件，适用于需要智能分支选择的复杂工作流。

核心特性：
1. 条件分支：根据条件函数或规则动态选择执行路径
2. 多层嵌套：支持条件的多层嵌套和复杂逻辑组合
3. 动态路由：根据运行时数据动态决定执行路径
4. 回退机制：支持默认分支和错误回退路径
5. 状态传递：在不同分支间传递和维护状态信息
6. 智能缓存：缓存条件判断结果，避免重复计算

设计原理：
- 策略模式：每个条件分支代表不同的执行策略
- 状态机模式：根据条件转换执行状态和路径
- 责任链模式：条件按优先级依次判断和处理
- 装饰器模式：为条件判断添加缓存、日志等功能
- 模板方法模式：定义条件判断和分支执行的通用流程

使用场景：
- 智能路由：根据用户类型、权限、偏好等选择处理路径
- 业务规则：实现复杂的业务逻辑判断和分支处理
- 错误处理：根据错误类型选择不同的恢复策略
- 性能优化：根据数据大小、系统负载选择最优算法
- 内容分发：根据内容类型、用户画像选择处理方式
"""

import time
import copy
import uuid
import asyncio
import threading
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import operator
import re
from pathlib import Path

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)


class ConditionType(Enum):
    """条件类型枚举"""
    FUNCTION = "function"         # 函数条件
    EXPRESSION = "expression"     # 表达式条件
    RULE = "rule"                # 规则条件
    VALUE = "value"              # 值条件
    REGEX = "regex"              # 正则表达式条件
    COMPOSITE = "composite"       # 复合条件


class LogicalOperator(Enum):
    """逻辑操作符枚举"""
    AND = "and"                  # 逻辑与
    OR = "or"                    # 逻辑或
    NOT = "not"                  # 逻辑非
    XOR = "xor"                  # 逻辑异或


class ComparisonOperator(Enum):
    """比较操作符枚举"""
    EQ = "eq"                    # 等于
    NE = "ne"                    # 不等于
    GT = "gt"                    # 大于
    LT = "lt"                    # 小于
    GE = "ge"                    # 大于等于
    LE = "le"                    # 小于等于
    IN = "in"                    # 包含
    NOT_IN = "not_in"            # 不包含
    STARTSWITH = "startswith"    # 以...开始
    ENDSWITH = "endswith"        # 以...结束
    CONTAINS = "contains"        # 包含子串
    MATCHES = "matches"          # 正则匹配


class BranchStatus(Enum):
    """分支状态枚举"""
    PENDING = "pending"          # 等待执行
    SELECTED = "selected"        # 被选中
    EXECUTED = "executed"        # 已执行
    SKIPPED = "skipped"          # 被跳过
    FAILED = "failed"            # 执行失败


@dataclass
class ConditionConfig:
    """
    条件配置类
    
    定义条件判断的各种配置，支持多种条件类型和组合方式。
    """
    
    # === 基本信息 ===
    name: str                                    # 条件名称
    condition_type: ConditionType                # 条件类型
    description: str = ""                        # 条件描述
    
    # === 条件定义 ===
    function: Optional[Callable] = None          # 条件函数
    expression: Optional[str] = None             # 条件表达式
    field_path: Optional[str] = None             # 数据字段路径
    operator: Optional[ComparisonOperator] = None  # 比较操作符
    value: Any = None                            # 比较值
    regex_pattern: Optional[str] = None          # 正则表达式模式
    
    # === 复合条件 ===
    sub_conditions: List['ConditionConfig'] = field(default_factory=list)  # 子条件列表
    logical_operator: LogicalOperator = LogicalOperator.AND  # 逻辑操作符
    
    # === 执行控制 ===
    priority: int = 0                            # 优先级
    cache_enabled: bool = True                   # 是否启用缓存
    timeout: Optional[float] = None              # 超时时间
    
    def __post_init__(self):
        """初始化后处理"""
        # 验证条件配置的有效性
        if self.condition_type == ConditionType.FUNCTION and not self.function:
            raise ConfigurationError("Function condition must have a function")
        
        if self.condition_type == ConditionType.EXPRESSION and not self.expression:
            raise ConfigurationError("Expression condition must have an expression")
        
        if self.condition_type == ConditionType.VALUE and self.operator is None:
            raise ConfigurationError("Value condition must have an operator")
        
        if self.condition_type == ConditionType.REGEX and not self.regex_pattern:
            raise ConfigurationError("Regex condition must have a pattern")
        
        if self.condition_type == ConditionType.COMPOSITE and not self.sub_conditions:
            raise ConfigurationError("Composite condition must have sub-conditions")


@dataclass
class BranchConfig:
    """
    分支配置类
    
    定义条件链中每个分支的配置信息，包括条件、执行器、
    参数等信息。
    """
    
    # === 基本信息 ===
    name: str                                    # 分支名称
    description: str = ""                        # 分支描述
    branch_id: Optional[str] = None             # 分支唯一标识
    
    # === 条件配置 ===
    condition: Optional[ConditionConfig] = None  # 分支条件
    is_default: bool = False                     # 是否为默认分支
    
    # === 执行配置 ===
    executor: Optional[Callable] = None          # 执行函数
    template: Optional[TemplateBase] = None      # 模板实例
    
    # === 参数配置 ===
    input_keys: List[str] = field(default_factory=list)    # 输入参数键
    output_keys: List[str] = field(default_factory=list)   # 输出参数键
    parameters: Dict[str, Any] = field(default_factory=dict)  # 固定参数
    
    # === 执行控制 ===
    priority: int = 0                            # 执行优先级
    timeout: Optional[float] = None              # 超时时间
    retry_count: int = 0                         # 重试次数
    
    # === 错误处理 ===
    error_handler: Optional[Callable] = None     # 错误处理函数
    fallback_branch: Optional[str] = None        # 回退分支ID
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.branch_id:
            self.branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        
        # 验证配置有效性
        if not self.is_default and not self.condition:
            raise ConfigurationError("Non-default branch must have a condition")
        
        if not self.executor and not self.template:
            raise ConfigurationError("Branch must have either executor or template")


@dataclass
class BranchResult:
    """
    分支执行结果类
    
    包含分支执行的完整信息。
    """
    
    # === 基本信息 ===
    branch_id: str                               # 分支ID
    branch_name: str                             # 分支名称
    status: BranchStatus                         # 执行状态
    
    # === 条件判断信息 ===
    condition_result: Optional[bool] = None      # 条件判断结果
    condition_evaluation_time: Optional[float] = None  # 条件评估时间
    
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
    
    def is_successful(self) -> bool:
        """判断分支是否执行成功"""
        return self.status == BranchStatus.EXECUTED and self.error is None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "branch_id": self.branch_id,
            "branch_name": self.branch_name,
            "status": self.status.value,
            "condition_result": self.condition_result,
            "execution_time": self.execution_time,
            "success": self.is_successful(),
            "error_message": self.error_message
        }


class ConditionEvaluator:
    """
    条件评估器
    
    负责评估各种类型的条件，支持缓存、超时控制等功能。
    """
    
    def __init__(self, cache_enabled: bool = True, max_cache_size: int = 1000):
        """
        初始化条件评估器
        
        Args:
            cache_enabled: 是否启用缓存
            max_cache_size: 最大缓存大小
        """
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.condition_cache: Dict[str, Tuple[bool, float]] = {}  # (result, timestamp)
        self._lock = threading.Lock()
        
        # 编译器缓存
        self.expression_cache: Dict[str, Any] = {}
        self.regex_cache: Dict[str, re.Pattern] = {}
        
        logger.debug("Initialized ConditionEvaluator")
    
    def evaluate(self, condition: ConditionConfig, data: Dict[str, Any]) -> bool:
        """
        评估条件
        
        Args:
            condition: 条件配置
            data: 输入数据
        
        Returns:
            条件评估结果
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(condition, data)
        
        # 检查缓存
        if condition.cache_enabled and self.cache_enabled:
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for condition: {condition.name}")
                return cached_result
        
        start_time = time.time()
        
        try:
            # 根据条件类型进行评估
            if condition.condition_type == ConditionType.FUNCTION:
                result = self._evaluate_function_condition(condition, data)
            elif condition.condition_type == ConditionType.EXPRESSION:
                result = self._evaluate_expression_condition(condition, data)
            elif condition.condition_type == ConditionType.VALUE:
                result = self._evaluate_value_condition(condition, data)
            elif condition.condition_type == ConditionType.REGEX:
                result = self._evaluate_regex_condition(condition, data)
            elif condition.condition_type == ConditionType.COMPOSITE:
                result = self._evaluate_composite_condition(condition, data)
            else:
                raise ConfigurationError(f"Unsupported condition type: {condition.condition_type}")
            
            # 缓存结果
            if condition.cache_enabled and self.cache_enabled:
                self._cache_result(cache_key, result)
            
            evaluation_time = time.time() - start_time
            logger.debug(f"Evaluated condition '{condition.name}': {result} (time: {evaluation_time:.3f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {condition.name} - {str(e)}")
            raise
    
    def _evaluate_function_condition(self, condition: ConditionConfig, data: Dict[str, Any]) -> bool:
        """评估函数条件"""
        try:
            if asyncio.iscoroutinefunction(condition.function):
                # 异步函数需要在异步上下文中执行
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在已有的事件循环中，需要创建任务
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, condition.function(data))
                        return future.result(timeout=condition.timeout)
                else:
                    return asyncio.run(condition.function(data))
            else:
                return condition.function(data)
        except Exception as e:
            logger.error(f"Function condition evaluation failed: {str(e)}")
            return False
    
    def _evaluate_expression_condition(self, condition: ConditionConfig, data: Dict[str, Any]) -> bool:
        """评估表达式条件"""
        try:
            # 简单的表达式评估（生产环境建议使用更安全的评估器）
            # 替换数据变量
            expression = condition.expression
            for key, value in data.items():
                # 简单的变量替换
                expression = expression.replace(f"${key}", str(value))
                expression = expression.replace(f"{{{key}}}", str(value))
            
            # 安全的表达式评估
            allowed_names = {
                "__builtins__": {},
                "True": True,
                "False": False,
                "None": None,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
            }
            
            # 添加数据变量
            allowed_names.update(data)
            
            result = eval(expression, allowed_names)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Expression condition evaluation failed: {str(e)}")
            return False
    
    def _evaluate_value_condition(self, condition: ConditionConfig, data: Dict[str, Any]) -> bool:
        """评估值条件"""
        try:
            # 获取字段值
            field_value = self._get_field_value(data, condition.field_path)
            compare_value = condition.value
            
            # 根据操作符进行比较
            if condition.operator == ComparisonOperator.EQ:
                return field_value == compare_value
            elif condition.operator == ComparisonOperator.NE:
                return field_value != compare_value
            elif condition.operator == ComparisonOperator.GT:
                return field_value > compare_value
            elif condition.operator == ComparisonOperator.LT:
                return field_value < compare_value
            elif condition.operator == ComparisonOperator.GE:
                return field_value >= compare_value
            elif condition.operator == ComparisonOperator.LE:
                return field_value <= compare_value
            elif condition.operator == ComparisonOperator.IN:
                return field_value in compare_value
            elif condition.operator == ComparisonOperator.NOT_IN:
                return field_value not in compare_value
            elif condition.operator == ComparisonOperator.STARTSWITH:
                return str(field_value).startswith(str(compare_value))
            elif condition.operator == ComparisonOperator.ENDSWITH:
                return str(field_value).endswith(str(compare_value))
            elif condition.operator == ComparisonOperator.CONTAINS:
                return str(compare_value) in str(field_value)
            elif condition.operator == ComparisonOperator.MATCHES:
                return re.search(str(compare_value), str(field_value)) is not None
            else:
                raise ConfigurationError(f"Unsupported operator: {condition.operator}")
            
        except Exception as e:
            logger.error(f"Value condition evaluation failed: {str(e)}")
            return False
    
    def _evaluate_regex_condition(self, condition: ConditionConfig, data: Dict[str, Any]) -> bool:
        """评估正则表达式条件"""
        try:
            # 获取字段值
            field_value = self._get_field_value(data, condition.field_path)
            text_value = str(field_value)
            
            # 获取或编译正则表达式
            pattern = self._get_regex_pattern(condition.regex_pattern)
            
            # 执行匹配
            return pattern.search(text_value) is not None
            
        except Exception as e:
            logger.error(f"Regex condition evaluation failed: {str(e)}")
            return False
    
    def _evaluate_composite_condition(self, condition: ConditionConfig, data: Dict[str, Any]) -> bool:
        """评估复合条件"""
        try:
            if not condition.sub_conditions:
                return True
            
            # 评估所有子条件
            sub_results = []
            for sub_condition in condition.sub_conditions:
                sub_result = self.evaluate(sub_condition, data)
                sub_results.append(sub_result)
            
            # 根据逻辑操作符组合结果
            if condition.logical_operator == LogicalOperator.AND:
                return all(sub_results)
            elif condition.logical_operator == LogicalOperator.OR:
                return any(sub_results)
            elif condition.logical_operator == LogicalOperator.NOT:
                # NOT操作符只使用第一个子条件
                return not sub_results[0] if sub_results else True
            elif condition.logical_operator == LogicalOperator.XOR:
                # XOR操作：有且仅有一个条件为真
                true_count = sum(sub_results)
                return true_count == 1
            else:
                raise ConfigurationError(f"Unsupported logical operator: {condition.logical_operator}")
            
        except Exception as e:
            logger.error(f"Composite condition evaluation failed: {str(e)}")
            return False
    
    def _get_field_value(self, data: Dict[str, Any], field_path: Optional[str]) -> Any:
        """
        获取字段值（支持嵌套路径）
        
        Args:
            data: 数据字典
            field_path: 字段路径，如 "user.profile.age"
        
        Returns:
            字段值
        """
        if not field_path:
            return data
        
        value = data
        for key in field_path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif hasattr(value, key):
                value = getattr(value, key)
            else:
                return None
        
        return value
    
    def _get_regex_pattern(self, pattern_string: str) -> re.Pattern:
        """获取或编译正则表达式模式"""
        if pattern_string in self.regex_cache:
            return self.regex_cache[pattern_string]
        
        try:
            pattern = re.compile(pattern_string)
            
            # 限制缓存大小
            if len(self.regex_cache) >= self.max_cache_size:
                # 简单的清理策略：清除一半
                keys_to_remove = list(self.regex_cache.keys())[:len(self.regex_cache) // 2]
                for key in keys_to_remove:
                    del self.regex_cache[key]
            
            self.regex_cache[pattern_string] = pattern
            return pattern
            
        except re.error as e:
            raise ConfigurationError(f"Invalid regex pattern: {pattern_string} - {str(e)}")
    
    def _generate_cache_key(self, condition: ConditionConfig, data: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        import json
        
        # 简化的缓存键生成
        key_data = {
            "condition_name": condition.name,
            "condition_type": condition.condition_type.value,
            "data_keys": list(data.keys()),
            "data_hash": hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:8]
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[bool]:
        """获取缓存结果"""
        with self._lock:
            if cache_key in self.condition_cache:
                result, timestamp = self.condition_cache[cache_key]
                # 简单的TTL检查（5分钟）
                if time.time() - timestamp < 300:
                    return result
                else:
                    del self.condition_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: bool) -> None:
        """缓存结果"""
        with self._lock:
            # 限制缓存大小
            if len(self.condition_cache) >= self.max_cache_size:
                # 删除最旧的一半条目
                sorted_items = sorted(
                    self.condition_cache.items(),
                    key=lambda x: x[1][1]  # 按时间戳排序
                )
                for key, _ in sorted_items[:len(sorted_items) // 2]:
                    del self.condition_cache[key]
            
            self.condition_cache[cache_key] = (result, time.time())
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._lock:
            self.condition_cache.clear()
            self.expression_cache.clear()
            self.regex_cache.clear()
        logger.info("Condition cache cleared")


class ConditionalChainTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    条件链模板类
    
    实现基于条件分支的工作流。根据条件动态选择执行路径，
    支持复杂的业务逻辑和决策流程。
    
    核心功能：
    1. 条件评估：支持多种条件类型和复杂逻辑组合
    2. 动态路由：根据条件结果选择执行分支
    3. 回退机制：支持默认分支和错误回退
    4. 状态管理：维护分支执行状态和数据流
    5. 性能优化：条件缓存和智能评估
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化条件链模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config)
        
        # 分支配置
        self.branches: List[BranchConfig] = []
        self.default_branch: Optional[BranchConfig] = None
        
        # 条件评估器
        self.evaluator = ConditionEvaluator()
        
        # 执行状态
        self.selected_branch: Optional[BranchConfig] = None
        self.execution_path: List[str] = []  # 记录执行路径
        
        logger.info(f"Initialized ConditionalChainTemplate: {self.config.name}")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="ConditionalChainTemplate",
            version="1.0.0",
            description="基于条件分支的链模板，支持动态路径选择",
            template_type=TemplateType.CHAIN,
            async_enabled=True,
            cache_enabled=True
        )
        
        # 定义参数模式
        config.add_parameter(
            "branches", list, required=True,
            description="条件分支配置列表",
            examples=[
                [
                    {"name": "branch1", "condition": {...}, "executor": "function1"},
                    {"name": "branch2", "condition": {...}, "executor": "function2"}
                ]
            ]
        )
        
        config.add_parameter(
            "default_branch", dict, required=False,
            description="默认分支配置（当所有条件都不满足时执行）"
        )
        
        config.add_parameter(
            "evaluation_strategy", str, required=False, default="first_match",
            description="条件评估策略",
            constraints={"allowed_values": ["first_match", "best_match", "all_match"]}
        )
        
        config.add_parameter(
            "cache_enabled", bool, required=False, default=True,
            description="是否启用条件缓存"
        )
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置链参数
        
        Args:
            **parameters: 链配置参数
                - branches: 分支配置列表
                - default_branch: 默认分支配置
                - evaluation_strategy: 评估策略
                - cache_enabled: 是否启用缓存
        
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
        
        # 配置默认分支
        default_branch_config = parameters.get("default_branch")
        if default_branch_config:
            self.default_branch = self._create_branch_config(default_branch_config, -1, is_default=True)
        
        # 配置评估策略
        self.evaluation_strategy = parameters.get("evaluation_strategy", "first_match")
        
        # 配置缓存
        cache_enabled = parameters.get("cache_enabled", True)
        self.evaluator.cache_enabled = cache_enabled
        
        # 更新状态
        self.status = self.status.__class__.CONFIGURED
        
        logger.info(f"Configured ConditionalChain with {len(self.branches)} branches")
    
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
    
    def _create_branch_config(
        self, 
        branch_dict: Dict[str, Any], 
        index: int,
        is_default: bool = False
    ) -> BranchConfig:
        """
        创建分支配置
        
        Args:
            branch_dict: 分支字典配置
            index: 分支索引
            is_default: 是否为默认分支
        
        Returns:
            分支配置对象
        """
        # 基本信息
        name = branch_dict.get("name", f"branch_{index}")
        description = branch_dict.get("description", "")
        branch_id = branch_dict.get("branch_id", f"branch_{index}_{uuid.uuid4().hex[:8]}")
        
        # 条件配置
        condition_dict = branch_dict.get("condition")
        condition = None
        if condition_dict and not is_default:
            condition = self._create_condition_config(condition_dict)
        
        # 执行器配置
        executor = branch_dict.get("executor")
        template = branch_dict.get("template")
        
        # 参数配置
        input_keys = branch_dict.get("input_keys", [])
        output_keys = branch_dict.get("output_keys", [])
        parameters = branch_dict.get("parameters", {})
        
        # 执行控制
        priority = branch_dict.get("priority", 0)
        timeout = branch_dict.get("timeout")
        retry_count = branch_dict.get("retry_count", 0)
        
        # 错误处理
        error_handler = branch_dict.get("error_handler")
        fallback_branch = branch_dict.get("fallback_branch")
        
        return BranchConfig(
            name=name,
            description=description,
            branch_id=branch_id,
            condition=condition,
            is_default=is_default,
            executor=executor,
            template=template,
            input_keys=input_keys,
            output_keys=output_keys,
            parameters=parameters,
            priority=priority,
            timeout=timeout,
            retry_count=retry_count,
            error_handler=error_handler,
            fallback_branch=fallback_branch
        )
    
    def _create_condition_config(self, condition_dict: Dict[str, Any]) -> ConditionConfig:
        """
        创建条件配置
        
        Args:
            condition_dict: 条件字典配置
        
        Returns:
            条件配置对象
        """
        # 基本信息
        name = condition_dict.get("name", "condition")
        condition_type_str = condition_dict.get("type", "function")
        condition_type = ConditionType(condition_type_str)
        description = condition_dict.get("description", "")
        
        # 条件定义
        function = condition_dict.get("function")
        expression = condition_dict.get("expression")
        field_path = condition_dict.get("field_path")
        operator_str = condition_dict.get("operator")
        operator = ComparisonOperator(operator_str) if operator_str else None
        value = condition_dict.get("value")
        regex_pattern = condition_dict.get("regex_pattern")
        
        # 复合条件
        sub_conditions_list = condition_dict.get("sub_conditions", [])
        sub_conditions = [self._create_condition_config(sub_dict) for sub_dict in sub_conditions_list]
        logical_operator_str = condition_dict.get("logical_operator", "and")
        logical_operator = LogicalOperator(logical_operator_str)
        
        # 执行控制
        priority = condition_dict.get("priority", 0)
        cache_enabled = condition_dict.get("cache_enabled", True)
        timeout = condition_dict.get("timeout")
        
        return ConditionConfig(
            name=name,
            condition_type=condition_type,
            description=description,
            function=function,
            expression=expression,
            field_path=field_path,
            operator=operator,
            value=value,
            regex_pattern=regex_pattern,
            sub_conditions=sub_conditions,
            logical_operator=logical_operator,
            priority=priority,
            cache_enabled=cache_enabled,
            timeout=timeout
        )
    
    def add_branch(
        self,
        name: str,
        condition: Optional[ConditionConfig] = None,
        executor: Optional[Callable] = None,
        template: Optional[TemplateBase] = None,
        is_default: bool = False,
        **kwargs
    ) -> str:
        """
        添加分支
        
        Args:
            name: 分支名称
            condition: 分支条件
            executor: 执行函数
            template: 模板实例
            is_default: 是否为默认分支
            **kwargs: 其他分支配置参数
        
        Returns:
            分支ID
        """
        branch_config = BranchConfig(
            name=name,
            condition=condition,
            executor=executor,
            template=template,
            is_default=is_default,
            **kwargs
        )
        
        if is_default:
            self.default_branch = branch_config
            logger.info(f"Set default branch: {name}")
        else:
            self.branches.append(branch_config)
            logger.info(f"Added branch: {name} ({branch_config.branch_id})")
        
        return branch_config.branch_id
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行条件链（同步版本）
        
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
        if not self.branches and not self.default_branch:
            raise ConfigurationError("No branches configured for execution")
        
        logger.info(f"Starting conditional chain execution")
        
        # 重置执行状态
        self.selected_branch = None
        self.execution_path.clear()
        
        try:
            # 选择执行分支
            selected_branch = self._select_branch(input_data)
            
            if not selected_branch:
                if self.default_branch:
                    selected_branch = self.default_branch
                    logger.info("Using default branch")
                else:
                    raise ConfigurationError("No branch selected and no default branch configured")
            
            self.selected_branch = selected_branch
            self.execution_path.append(selected_branch.branch_id)
            
            # 执行选中的分支
            branch_result = self._execute_branch(selected_branch, input_data)
            
            # 构建最终结果
            final_result = {
                "status": "completed" if branch_result.is_successful() else "failed",
                "selected_branch": {
                    "id": selected_branch.branch_id,
                    "name": selected_branch.name,
                    "is_default": selected_branch.is_default
                },
                "execution_path": self.execution_path.copy(),
                "input_data": input_data,
                "output_data": branch_result.output_data,
                "branch_result": branch_result.get_execution_summary(),
                "condition_evaluations": self._get_condition_evaluations(input_data)
            }
            
            logger.info(f"Conditional chain execution completed: {selected_branch.name}")
            return final_result
            
        except Exception as e:
            logger.error(f"Conditional chain execution failed: {str(e)}")
            
            error_result = {
                "status": "failed",
                "error": str(e),
                "input_data": input_data,
                "execution_path": self.execution_path.copy(),
                "selected_branch": {
                    "id": self.selected_branch.branch_id,
                    "name": self.selected_branch.name
                } if self.selected_branch else None
            }
            
            raise Exception(f"Conditional chain execution failed: {str(e)}") from e
    
    async def execute_async(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行条件链（异步版本）
        
        Args:
            input_data: 输入数据
            **kwargs: 执行参数
        
        Returns:
            执行结果
        """
        if not self.branches and not self.default_branch:
            raise ConfigurationError("No branches configured for execution")
        
        logger.info(f"Starting async conditional chain execution")
        
        # 重置执行状态
        self.selected_branch = None
        self.execution_path.clear()
        
        try:
            # 异步选择执行分支
            selected_branch = await self._select_branch_async(input_data)
            
            if not selected_branch:
                if self.default_branch:
                    selected_branch = self.default_branch
                    logger.info("Using default branch")
                else:
                    raise ConfigurationError("No branch selected and no default branch configured")
            
            self.selected_branch = selected_branch
            self.execution_path.append(selected_branch.branch_id)
            
            # 异步执行选中的分支
            branch_result = await self._execute_branch_async(selected_branch, input_data)
            
            # 构建最终结果
            final_result = {
                "status": "completed" if branch_result.is_successful() else "failed",
                "selected_branch": {
                    "id": selected_branch.branch_id,
                    "name": selected_branch.name,
                    "is_default": selected_branch.is_default
                },
                "execution_path": self.execution_path.copy(),
                "input_data": input_data,
                "output_data": branch_result.output_data,
                "branch_result": branch_result.get_execution_summary(),
                "condition_evaluations": self._get_condition_evaluations(input_data)
            }
            
            logger.info(f"Async conditional chain execution completed: {selected_branch.name}")
            return final_result
            
        except Exception as e:
            logger.error(f"Async conditional chain execution failed: {str(e)}")
            raise Exception(f"Async conditional chain execution failed: {str(e)}") from e
    
    def _select_branch(self, input_data: Dict[str, Any]) -> Optional[BranchConfig]:
        """
        选择执行分支（同步版本）
        
        Args:
            input_data: 输入数据
        
        Returns:
            选中的分支，如果没有匹配则返回None
        """
        # 按优先级排序分支
        sorted_branches = sorted(self.branches, key=lambda b: b.priority, reverse=True)
        
        for branch in sorted_branches:
            if branch.condition:
                try:
                    condition_result = self.evaluator.evaluate(branch.condition, input_data)
                    if condition_result:
                        logger.info(f"Branch condition matched: {branch.name}")
                        return branch
                except Exception as e:
                    logger.error(f"Condition evaluation failed for branch {branch.name}: {str(e)}")
                    continue
        
        logger.info("No branch condition matched")
        return None
    
    async def _select_branch_async(self, input_data: Dict[str, Any]) -> Optional[BranchConfig]:
        """
        选择执行分支（异步版本）
        
        Args:
            input_data: 输入数据
        
        Returns:
            选中的分支，如果没有匹配则返回None
        """
        # 按优先级排序分支
        sorted_branches = sorted(self.branches, key=lambda b: b.priority, reverse=True)
        
        for branch in sorted_branches:
            if branch.condition:
                try:
                    # 对于异步条件评估，这里简化处理
                    # 实际实现中可能需要支持异步条件函数
                    condition_result = self.evaluator.evaluate(branch.condition, input_data)
                    if condition_result:
                        logger.info(f"Branch condition matched: {branch.name}")
                        return branch
                except Exception as e:
                    logger.error(f"Condition evaluation failed for branch {branch.name}: {str(e)}")
                    continue
        
        logger.info("No branch condition matched")
        return None
    
    def _execute_branch(self, branch: BranchConfig, input_data: Dict[str, Any]) -> BranchResult:
        """
        执行分支（同步版本）
        
        Args:
            branch: 分支配置
            input_data: 输入数据
        
        Returns:
            分支执行结果
        """
        result = BranchResult(
            branch_id=branch.branch_id,
            branch_name=branch.name,
            status=BranchStatus.SELECTED,
            start_time=time.time(),
            input_data=input_data,
            parameters=branch.parameters.copy()
        )
        
        try:
            logger.debug(f"Executing branch: {branch.name}")
            
            # 准备输入数据
            branch_input = self._prepare_branch_input(branch, input_data)
            result.input_data = branch_input
            
            # 执行分支逻辑
            if branch.executor:
                # 使用执行函数
                output = branch.executor(branch_input, **branch.parameters)
            elif branch.template:
                # 使用模板
                output = branch.template.run(branch_input, **branch.parameters)
            else:
                raise ConfigurationError(f"Branch '{branch.name}' has no executor or template")
            
            result.output_data = output
            result.status = BranchStatus.EXECUTED
            
            logger.debug(f"Branch executed successfully: {branch.name}")
            
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
            
            logger.error(f"Branch execution failed: {branch.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
        
        return result
    
    async def _execute_branch_async(self, branch: BranchConfig, input_data: Dict[str, Any]) -> BranchResult:
        """
        执行分支（异步版本）
        
        Args:
            branch: 分支配置
            input_data: 输入数据
        
        Returns:
            分支执行结果
        """
        result = BranchResult(
            branch_id=branch.branch_id,
            branch_name=branch.name,
            status=BranchStatus.SELECTED,
            start_time=time.time(),
            input_data=input_data,
            parameters=branch.parameters.copy()
        )
        
        try:
            logger.debug(f"Async executing branch: {branch.name}")
            
            # 准备输入数据
            branch_input = self._prepare_branch_input(branch, input_data)
            result.input_data = branch_input
            
            # 异步执行分支逻辑
            if branch.executor:
                # 使用执行函数
                if asyncio.iscoroutinefunction(branch.executor):
                    output = await branch.executor(branch_input, **branch.parameters)
                else:
                    # 在线程池中执行同步函数
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None, 
                        branch.executor, 
                        branch_input,
                        branch.parameters
                    )
            elif branch.template:
                # 使用模板的异步方法
                if hasattr(branch.template, 'run_async'):
                    output = await branch.template.run_async(branch_input, **branch.parameters)
                else:
                    # 在线程池中执行同步方法
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None,
                        branch.template.run,
                        branch_input,
                        branch.parameters
                    )
            else:
                raise ConfigurationError(f"Branch '{branch.name}' has no executor or template")
            
            result.output_data = output
            result.status = BranchStatus.EXECUTED
            
            logger.debug(f"Async branch executed successfully: {branch.name}")
            
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
            
            logger.error(f"Async branch execution failed: {branch.name} - {str(e)}")
        
        finally:
            result.end_time = time.time()
            result.execution_time = result.end_time - result.start_time
        
        return result
    
    def _prepare_branch_input(self, branch: BranchConfig, input_data: Dict[str, Any]) -> Any:
        """
        准备分支输入数据
        
        Args:
            branch: 分支配置
            input_data: 原始输入数据
        
        Returns:
            分支输入数据
        """
        if not branch.input_keys:
            # 如果没有指定输入键，返回整个输入数据
            return input_data.copy()
        
        # 根据输入键提取数据
        branch_input = {}
        for key in branch.input_keys:
            if key in input_data:
                branch_input[key] = input_data[key]
            else:
                logger.warning(f"Input key '{key}' not found in input data for branch '{branch.name}'")
        
        return branch_input
    
    def _get_condition_evaluations(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取所有条件的评估结果
        
        Args:
            input_data: 输入数据
        
        Returns:
            条件评估结果列表
        """
        evaluations = []
        
        for branch in self.branches:
            if branch.condition:
                try:
                    start_time = time.time()
                    result = self.evaluator.evaluate(branch.condition, input_data)
                    evaluation_time = time.time() - start_time
                    
                    evaluations.append({
                        "branch_id": branch.branch_id,
                        "branch_name": branch.name,
                        "condition_name": branch.condition.name,
                        "result": result,
                        "evaluation_time": evaluation_time
                    })
                except Exception as e:
                    evaluations.append({
                        "branch_id": branch.branch_id,
                        "branch_name": branch.name,
                        "condition_name": branch.condition.name,
                        "result": False,
                        "error": str(e)
                    })
        
        return evaluations
    
    def clear_cache(self) -> None:
        """清空条件缓存"""
        self.evaluator.clear_cache()
        logger.info("Condition cache cleared")
    
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
                        "name": "高价值用户分支",
                        "condition": {
                            "name": "high_value_user",
                            "type": "value",
                            "field_path": "user.value_score",
                            "operator": "gt",
                            "value": 80
                        },
                        "executor": "handle_high_value_user",
                        "priority": 2,
                        "description": "处理高价值用户的特殊逻辑"
                    },
                    {
                        "name": "新用户分支",
                        "condition": {
                            "name": "new_user",
                            "type": "value",
                            "field_path": "user.registration_days",
                            "operator": "lt",
                            "value": 30
                        },
                        "executor": "handle_new_user",
                        "priority": 1,
                        "description": "处理新用户的特殊逻辑"
                    },
                    {
                        "name": "VIP用户分支",
                        "condition": {
                            "name": "vip_user",
                            "type": "value",
                            "field_path": "user.level",
                            "operator": "in",
                            "value": ["VIP", "SVIP", "PREMIUM"]
                        },
                        "executor": "handle_vip_user",
                        "priority": 3,
                        "description": "处理VIP用户的特殊逻辑"
                    }
                ],
                "default_branch": {
                    "name": "普通用户分支",
                    "executor": "handle_normal_user",
                    "description": "处理普通用户的默认逻辑"
                },
                "evaluation_strategy": "first_match",
                "cache_enabled": True
            },
            "execute_parameters": {
                "user": {
                    "id": "12345",
                    "level": "VIP",
                    "value_score": 85,
                    "registration_days": 45
                },
                "request_type": "purchase"
            },
            "expected_output": {
                "status": "completed",
                "selected_branch": {
                    "id": "branch_2",
                    "name": "VIP用户分支",
                    "is_default": False
                },
                "output_data": "VIP用户处理结果",
                "condition_evaluations": [
                    {"branch_name": "高价值用户分支", "result": True},
                    {"branch_name": "新用户分支", "result": False},
                    {"branch_name": "VIP用户分支", "result": True}
                ]
            },
            "usage_code": '''
# 使用示例
from templates.chains.conditional_chain import ConditionalChainTemplate, ConditionConfig, ComparisonOperator, ConditionType

# 定义分支执行函数
def handle_high_value_user(data, **kwargs):
    """高价值用户处理逻辑"""
    user = data.get('user', {})
    return f"高价值用户 {user.get('id')} 的特殊处理结果"

def handle_new_user(data, **kwargs):
    """新用户处理逻辑"""
    user = data.get('user', {})
    return f"新用户 {user.get('id')} 的欢迎处理"

def handle_vip_user(data, **kwargs):
    """VIP用户处理逻辑"""
    user = data.get('user', {})
    return f"VIP用户 {user.get('id')} 的尊享服务"

def handle_normal_user(data, **kwargs):
    """普通用户处理逻辑"""
    user = data.get('user', {})
    return f"普通用户 {user.get('id')} 的标准服务"

# 创建条件配置
high_value_condition = ConditionConfig(
    name="high_value_user",
    condition_type=ConditionType.VALUE,
    field_path="user.value_score",
    operator=ComparisonOperator.GT,
    value=80
)

vip_condition = ConditionConfig(
    name="vip_user",
    condition_type=ConditionType.VALUE,
    field_path="user.level",
    operator=ComparisonOperator.IN,
    value=["VIP", "SVIP", "PREMIUM"]
)

# 创建和配置条件链
chain = ConditionalChainTemplate()
chain.setup(
    branches=[
        {
            "name": "高价值用户分支",
            "condition": {
                "name": "high_value_user",
                "type": "value",
                "field_path": "user.value_score",
                "operator": "gt",
                "value": 80
            },
            "executor": handle_high_value_user,
            "priority": 2
        },
        {
            "name": "VIP用户分支",
            "condition": {
                "name": "vip_user", 
                "type": "value",
                "field_path": "user.level",
                "operator": "in",
                "value": ["VIP", "SVIP", "PREMIUM"]
            },
            "executor": handle_vip_user,
            "priority": 3
        }
    ],
    default_branch={
        "name": "普通用户分支",
        "executor": handle_normal_user
    }
)

# 执行条件链
input_data = {
    "user": {
        "id": "12345",
        "level": "VIP",
        "value_score": 85,
        "registration_days": 45
    },
    "request_type": "purchase"
}

result = chain.run(input_data)
print("选中分支:", result["selected_branch"]["name"])
print("执行结果:", result["output_data"])

# 异步执行
import asyncio

async def main():
    result = await chain.run_async(input_data)
    print("异步执行结果:", result["output_data"])

asyncio.run(main())
'''
        }


# 注册模板到工厂
def register_conditional_chain_template():
    """注册条件链模板到全局工厂"""
    from ..base.template_base import register_template
    register_template("conditional_chain", ConditionalChainTemplate)


# 自动注册
register_conditional_chain_template()