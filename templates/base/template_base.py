"""
模板系统的核心基础架构模块

本模块定义了整个模板系统的基础架构，包括：
- TemplateConfig: 模板配置数据类，定义模板的元数据和参数结构
- TemplateBase: 所有模板的抽象基类，定义统一的模板接口
- TemplateFactory: 模板工厂，支持动态创建和管理模板实例

设计原理：
1. 统一接口：所有模板都继承自TemplateBase，提供一致的使用体验
2. 配置驱动：通过TemplateConfig定义模板行为，支持灵活的参数化配置
3. 生命周期管理：从设置、验证、执行到结果处理的完整生命周期
4. 扩展性：模块化设计，便于添加新的模板类型和功能
5. 类型安全：完整的类型注解，提高代码质量和开发体验
"""

import os
import abc
import uuid
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Union, Type, TypeVar, Generic, 
    Callable, Awaitable, Protocol
)
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator

# 导入项目核心模块
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)

T = TypeVar('T')  # 输入类型
U = TypeVar('U')  # 输出类型


class TemplateStatus(Enum):
    """模板执行状态枚举"""
    INITIALIZED = "initialized"    # 已初始化
    CONFIGURED = "configured"      # 已配置  
    READY = "ready"               # 准备就绪
    RUNNING = "running"           # 执行中
    COMPLETED = "completed"       # 执行完成
    FAILED = "failed"             # 执行失败
    CANCELLED = "cancelled"       # 执行取消


class TemplateType(Enum):
    """模板类型枚举"""
    LLM = "llm"                   # LLM模型模板
    PROMPT = "prompt"             # 提示词模板  
    CHAIN = "chain"               # 链组合模板
    AGENT = "agent"               # 代理模板
    DATA = "data"                 # 数据处理模板
    MEMORY = "memory"             # 记忆系统模板
    EVALUATION = "evaluation"     # 评估模板
    CUSTOM = "custom"             # 自定义模板


@dataclass
class ParameterSchema:
    """
    参数模式定义
    
    定义模板参数的结构、类型和约束条件，用于参数验证和文档生成。
    """
    name: str                                    # 参数名称
    type: Type                                   # 参数类型
    required: bool = True                        # 是否必需
    default: Any = None                          # 默认值
    description: str = ""                        # 参数描述
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件
    examples: List[Any] = field(default_factory=list)         # 示例值
    
    def __post_init__(self):
        """初始化后处理，设置默认值和验证约束条件"""
        if not self.required and self.default is None:
            # 为非必需参数生成合适的默认值
            if self.type == str:
                self.default = ""
            elif self.type == int:
                self.default = 0
            elif self.type == float:
                self.default = 0.0
            elif self.type == bool:
                self.default = False
            elif self.type == list:
                self.default = []
            elif self.type == dict:
                self.default = {}


@dataclass
class TemplateConfig:
    """
    模板配置数据类
    
    定义模板的元数据、参数结构和执行配置。这是模板系统的核心配置结构，
    用于描述模板的行为、依赖关系和使用方式。
    
    设计思路：
    1. 元数据管理：包含模板的基本信息，如名称、版本、描述等
    2. 参数定义：定义模板接受的参数及其类型、约束和默认值
    3. 依赖管理：声明模板的依赖关系，支持依赖检查和自动安装
    4. 示例提供：内置使用示例，便于学习和测试
    5. 执行配置：定义模板的执行行为和性能参数
    """
    
    # === 基本元数据 ===
    name: str                                    # 模板名称
    version: str = "1.0.0"                      # 模板版本
    description: str = ""                       # 模板描述
    template_type: TemplateType = TemplateType.CUSTOM  # 模板类型
    author: str = ""                            # 作者信息
    
    # === 参数定义 ===
    parameters: Dict[str, ParameterSchema] = field(default_factory=dict)  # 参数模式
    required_parameters: List[str] = field(default_factory=list)          # 必需参数列表
    optional_parameters: List[str] = field(default_factory=list)          # 可选参数列表
    
    # === 依赖管理 ===
    dependencies: List[str] = field(default_factory=list)                 # Python依赖包
    template_dependencies: List[str] = field(default_factory=list)        # 模板依赖
    
    # === 使用示例 ===
    examples: List[Dict[str, Any]] = field(default_factory=list)          # 使用示例
    
    # === 执行配置 ===
    timeout: Optional[float] = None              # 超时时间（秒）
    retry_count: int = 0                         # 重试次数
    async_enabled: bool = False                  # 是否支持异步执行
    
    # === 文档和标签 ===
    tags: List[str] = field(default_factory=list)           # 标签
    documentation_url: Optional[str] = None                 # 文档链接
    source_url: Optional[str] = None                        # 源码链接
    
    # === 性能配置 ===
    cache_enabled: bool = False                  # 是否启用缓存
    cache_ttl: int = 3600                        # 缓存生存时间（秒）
    max_memory_usage: Optional[int] = None       # 最大内存使用量（MB）
    
    def __post_init__(self):
        """初始化后处理，自动生成必需和可选参数列表"""
        if not self.required_parameters and not self.optional_parameters:
            # 根据parameters自动生成参数列表
            for param_name, param_schema in self.parameters.items():
                if param_schema.required:
                    self.required_parameters.append(param_name)
                else:
                    self.optional_parameters.append(param_name)
    
    def get_parameter_schema(self, param_name: str) -> Optional[ParameterSchema]:
        """获取指定参数的模式定义"""
        return self.parameters.get(param_name)
    
    def add_parameter(
        self, 
        name: str, 
        param_type: Type, 
        required: bool = True,
        default: Any = None,
        description: str = "",
        **kwargs
    ) -> None:
        """
        添加参数定义
        
        Args:
            name: 参数名称
            param_type: 参数类型
            required: 是否必需
            default: 默认值
            description: 参数描述
            **kwargs: 其他参数（constraints, examples等）
        """
        param_schema = ParameterSchema(
            name=name,
            type=param_type,
            required=required,
            default=default,
            description=description,
            **kwargs
        )
        self.parameters[name] = param_schema
        
        # 更新参数列表
        if required and name not in self.required_parameters:
            self.required_parameters.append(name)
        elif not required and name not in self.optional_parameters:
            self.optional_parameters.append(name)
    
    def validate_structure(self) -> bool:
        """
        验证配置结构的有效性
        
        Returns:
            True如果配置有效，否则抛出异常
            
        Raises:
            ValidationError: 配置结构无效
        """
        try:
            # 验证基本字段
            if not self.name:
                raise ValidationError("Template name cannot be empty")
            
            if not self.version:
                raise ValidationError("Template version cannot be empty")
            
            # 验证参数定义
            for param_name, param_schema in self.parameters.items():
                if not param_name:
                    raise ValidationError("Parameter name cannot be empty")
                
                if param_schema.required and param_schema.default is not None:
                    logger.warning(
                        f"Required parameter '{param_name}' has default value, "
                        "which may cause confusion"
                    )
            
            # 验证依赖关系
            for dep in self.template_dependencies:
                if dep == self.name:
                    raise ValidationError(f"Template cannot depend on itself: {dep}")
            
            # 验证超时和重试配置
            if self.timeout is not None and self.timeout <= 0:
                raise ValidationError("Timeout must be positive")
            
            if self.retry_count < 0:
                raise ValidationError("Retry count must be non-negative")
            
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                f"Template config validation failed: {str(e)}",
                error_code=ErrorCodes.VALIDATION_TYPE_ERROR,
                cause=e
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == "parameters":
                # 特殊处理参数模式
                result[field_name] = {
                    name: {
                        "type": schema.type.__name__,
                        "required": schema.required,
                        "default": schema.default,
                        "description": schema.description,
                        "constraints": schema.constraints,
                        "examples": schema.examples
                    }
                    for name, schema in field_value.items()
                }
            elif isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateConfig":
        """从字典创建配置实例"""
        # 提取参数数据并转换为ParameterSchema对象
        parameters_data = data.get("parameters", {})
        parameters = {}
        
        for name, schema_data in parameters_data.items():
            # 重构类型信息
            type_name = schema_data.get("type", "str")
            param_type = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict
            }.get(type_name, str)
            
            parameters[name] = ParameterSchema(
                name=name,
                type=param_type,
                required=schema_data.get("required", True),
                default=schema_data.get("default"),
                description=schema_data.get("description", ""),
                constraints=schema_data.get("constraints", {}),
                examples=schema_data.get("examples", [])
            )
        
        # 处理枚举类型
        template_type = data.get("template_type", "custom")
        if isinstance(template_type, str):
            template_type = TemplateType(template_type)
        
        # 创建配置实例
        config_data = data.copy()
        config_data["parameters"] = parameters
        config_data["template_type"] = template_type
        
        return cls(**config_data)


class TemplateBase(ABC, Generic[T, U]):
    """
    模板抽象基类
    
    定义了所有模板的统一接口和基础功能。所有具体的模板实现都应该继承此类。
    
    核心设计理念：
    1. 统一接口：提供一致的setup、execute、get_example方法
    2. 生命周期管理：从初始化到执行完成的完整状态管理
    3. 参数验证：内置参数验证机制，确保输入数据的有效性
    4. 错误处理：统一的错误处理和日志记录
    5. 性能监控：执行时间、内存使用等性能指标收集
    6. 异步支持：支持同步和异步两种执行模式
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化模板基类
        
        Args:
            config: 模板配置，None则创建默认配置
        """
        self.config = config or self._create_default_config()
        self.status = TemplateStatus.INITIALIZED
        self.execution_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error_message: Optional[str] = None
        
        # 性能指标
        self.metrics: Dict[str, Any] = {}
        self._setup_parameters: Dict[str, Any] = {}
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        
        # 验证配置
        self.config.validate_structure()
        
        logger.debug(f"Initialized template: {self.config.name}")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置，子类可以重写以提供特定的默认配置"""
        return TemplateConfig(
            name=self.__class__.__name__,
            description=f"Default configuration for {self.__class__.__name__}"
        )
    
    @abstractmethod
    def setup(self, **parameters) -> None:
        """
        设置模板参数
        
        子类必须实现此方法来处理模板的初始化配置。
        
        Args:
            **parameters: 模板参数
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        pass
    
    @abstractmethod
    def execute(self, input_data: T, **kwargs) -> U:
        """
        执行模板逻辑（同步版本）
        
        子类必须实现此方法来定义模板的核心逻辑。
        
        Args:
            input_data: 输入数据
            **kwargs: 额外的执行参数
            
        Returns:
            执行结果
            
        Raises:
            Exception: 执行过程中的任何异常
        """
        pass
    
    async def execute_async(self, input_data: T, **kwargs) -> U:
        """
        执行模板逻辑（异步版本）
        
        默认实现是将同步execute方法包装在异步上下文中执行。
        如果子类支持原生异步操作，可以重写此方法。
        
        Args:
            input_data: 输入数据
            **kwargs: 额外的执行参数
            
        Returns:
            执行结果
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data, **kwargs)
    
    @abstractmethod
    def get_example(self) -> Dict[str, Any]:
        """
        获取使用示例
        
        返回模板的使用示例，包括setup参数、execute参数和预期输出。
        
        Returns:
            示例字典，包含setup_parameters、execute_parameters、expected_output等
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        验证参数有效性
        
        Args:
            parameters: 要验证的参数字典
            
        Returns:
            True如果参数有效
            
        Raises:
            ValidationError: 参数验证失败
        """
        try:
            # 检查必需参数
            for param_name in self.config.required_parameters:
                if param_name not in parameters:
                    raise ValidationError(
                        f"Required parameter missing: {param_name}",
                        error_code=ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING
                    )
            
            # 检查参数类型和约束
            for param_name, param_value in parameters.items():
                if param_name in self.config.parameters:
                    param_schema = self.config.parameters[param_name]
                    
                    # 类型检查
                    if not isinstance(param_value, param_schema.type):
                        # 尝试类型转换
                        try:
                            param_value = param_schema.type(param_value)
                            parameters[param_name] = param_value
                        except (ValueError, TypeError):
                            raise ValidationError(
                                f"Parameter '{param_name}' must be of type {param_schema.type.__name__}, "
                                f"got {type(param_value).__name__}",
                                error_code=ErrorCodes.VALIDATION_TYPE_ERROR
                            )
                    
                    # 约束检查
                    if param_schema.constraints:
                        self._validate_constraints(param_name, param_value, param_schema.constraints)
            
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                f"Parameter validation failed: {str(e)}",
                error_code=ErrorCodes.VALIDATION_TYPE_ERROR,
                cause=e
            )
    
    def _validate_constraints(self, param_name: str, value: Any, constraints: Dict[str, Any]) -> None:
        """验证参数约束条件"""
        for constraint_name, constraint_value in constraints.items():
            if constraint_name == "min_value" and value < constraint_value:
                raise ValidationError(
                    f"Parameter '{param_name}' must be >= {constraint_value}, got {value}"
                )
            elif constraint_name == "max_value" and value > constraint_value:
                raise ValidationError(
                    f"Parameter '{param_name}' must be <= {constraint_value}, got {value}"
                )
            elif constraint_name == "min_length" and len(value) < constraint_value:
                raise ValidationError(
                    f"Parameter '{param_name}' length must be >= {constraint_value}, got {len(value)}"
                )
            elif constraint_name == "max_length" and len(value) > constraint_value:
                raise ValidationError(
                    f"Parameter '{param_name}' length must be <= {constraint_value}, got {len(value)}"
                )
            elif constraint_name == "allowed_values" and value not in constraint_value:
                raise ValidationError(
                    f"Parameter '{param_name}' must be one of {constraint_value}, got {value}"
                )
            elif constraint_name == "pattern" and isinstance(value, str):
                import re
                if not re.match(constraint_value, value):
                    raise ValidationError(
                        f"Parameter '{param_name}' must match pattern {constraint_value}, got {value}"
                    )
    
    def run(self, input_data: T, **kwargs) -> U:
        """
        运行模板的主要入口方法（同步版本）
        
        提供完整的生命周期管理，包括状态跟踪、性能监控和错误处理。
        
        Args:
            input_data: 输入数据
            **kwargs: 额外的执行参数
            
        Returns:
            执行结果
        """
        self.execution_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.status = TemplateStatus.RUNNING
        
        try:
            logger.info(f"Starting template execution: {self.config.name} ({self.execution_id})")
            
            # 执行核心逻辑
            result = self.execute(input_data, **kwargs)
            
            # 执行成功
            self.end_time = time.time()
            self.status = TemplateStatus.COMPLETED
            execution_time = self.end_time - self.start_time
            
            # 更新性能指标
            self.metrics.update({
                "execution_time": execution_time,
                "input_size": len(str(input_data)) if input_data else 0,
                "output_size": len(str(result)) if result else 0,
                "success": True
            })
            
            # 记录执行历史
            self._record_execution(True, execution_time, kwargs)
            
            logger.info(
                f"Template execution completed: {self.config.name} "
                f"(duration: {execution_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            # 执行失败
            self.end_time = time.time()
            self.status = TemplateStatus.FAILED
            self.error_message = str(e)
            execution_time = self.end_time - self.start_time if self.start_time else 0
            
            # 更新性能指标
            self.metrics.update({
                "execution_time": execution_time,
                "error": str(e),
                "success": False
            })
            
            # 记录执行历史
            self._record_execution(False, execution_time, kwargs, str(e))
            
            logger.error(f"Template execution failed: {self.config.name} - {str(e)}")
            raise
    
    async def run_async(self, input_data: T, **kwargs) -> U:
        """
        运行模板的主要入口方法（异步版本）
        
        Args:
            input_data: 输入数据
            **kwargs: 额外的执行参数
            
        Returns:
            执行结果
        """
        if not self.config.async_enabled:
            # 如果模板不支持异步，使用同步方法
            return self.run(input_data, **kwargs)
        
        self.execution_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.status = TemplateStatus.RUNNING
        
        try:
            logger.info(f"Starting async template execution: {self.config.name} ({self.execution_id})")
            
            # 执行核心逻辑
            result = await self.execute_async(input_data, **kwargs)
            
            # 执行成功
            self.end_time = time.time()
            self.status = TemplateStatus.COMPLETED
            execution_time = self.end_time - self.start_time
            
            # 更新性能指标
            self.metrics.update({
                "execution_time": execution_time,
                "input_size": len(str(input_data)) if input_data else 0,
                "output_size": len(str(result)) if result else 0,
                "success": True,
                "async": True
            })
            
            # 记录执行历史
            self._record_execution(True, execution_time, kwargs)
            
            logger.info(
                f"Async template execution completed: {self.config.name} "
                f"(duration: {execution_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            # 执行失败
            self.end_time = time.time()
            self.status = TemplateStatus.FAILED
            self.error_message = str(e)
            execution_time = self.end_time - self.start_time if self.start_time else 0
            
            # 更新性能指标
            self.metrics.update({
                "execution_time": execution_time,
                "error": str(e),
                "success": False,
                "async": True
            })
            
            # 记录执行历史
            self._record_execution(False, execution_time, kwargs, str(e))
            
            logger.error(f"Async template execution failed: {self.config.name} - {str(e)}")
            raise
    
    def _record_execution(
        self, 
        success: bool, 
        execution_time: float, 
        kwargs: Dict[str, Any],
        error: Optional[str] = None
    ) -> None:
        """记录执行历史"""
        record = {
            "execution_id": self.execution_id,
            "timestamp": self.start_time,
            "execution_time": execution_time,
            "success": success,
            "parameters": kwargs.copy(),
            "setup_parameters": self._setup_parameters.copy()
        }
        
        if error:
            record["error"] = error
        
        self.execution_history.append(record)
        
        # 保持历史记录数量在合理范围内
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]
    
    def get_status(self) -> Dict[str, Any]:
        """获取模板状态信息"""
        return {
            "name": self.config.name,
            "status": self.status.value,
            "execution_id": self.execution_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": (self.end_time - self.start_time) if self.start_time and self.end_time else None,
            "error_message": self.error_message,
            "metrics": self.metrics.copy(),
            "total_executions": len(self.execution_history),
            "successful_executions": sum(1 for h in self.execution_history if h["success"]),
            "failed_executions": sum(1 for h in self.execution_history if not h["success"])
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        successful_executions = [h for h in self.execution_history if h["success"]]
        failed_executions = [h for h in self.execution_history if not h["success"]]
        
        metrics = {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "success_rate": len(successful_executions) / len(self.execution_history) if self.execution_history else 0
        }
        
        if successful_executions:
            execution_times = [h["execution_time"] for h in successful_executions]
            metrics.update({
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times)
            })
        
        return metrics
    
    def reset(self) -> None:
        """重置模板状态"""
        self.status = TemplateStatus.INITIALIZED
        self.execution_id = None
        self.start_time = None
        self.end_time = None
        self.error_message = None
        self.metrics.clear()
        logger.debug(f"Reset template: {self.config.name}")
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(name='{self.config.name}', status='{self.status.value}')"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.config.name}', "
            f"type='{self.config.template_type.value}', "
            f"status='{self.status.value}', "
            f"executions={len(self.execution_history)})"
        )


# 模板工厂相关类型定义
TemplateConstructor = Callable[[TemplateConfig], TemplateBase]


class TemplateFactory:
    """
    模板工厂类
    
    负责模板的注册、创建和管理。支持动态加载模板类型，
    提供统一的模板实例创建接口。
    
    核心功能：
    1. 模板注册：支持运行时注册新的模板类型
    2. 动态创建：根据配置动态创建模板实例
    3. 类型管理：维护模板类型到构造函数的映射
    4. 配置验证：在创建前验证模板配置的有效性
    5. 实例缓存：可选的实例缓存机制，提高性能
    """
    
    def __init__(self):
        """初始化模板工厂"""
        self._template_registry: Dict[str, TemplateConstructor] = {}
        self._instance_cache: Dict[str, TemplateBase] = {}
        self._cache_enabled = False
        
        logger.debug("Initialized TemplateFactory")
    
    def register_template(
        self, 
        template_type: str, 
        constructor: TemplateConstructor
    ) -> None:
        """
        注册模板类型
        
        Args:
            template_type: 模板类型名称
            constructor: 模板构造函数
        """
        if template_type in self._template_registry:
            logger.warning(f"Template type '{template_type}' already registered, overwriting")
        
        self._template_registry[template_type] = constructor
        logger.info(f"Registered template type: {template_type}")
    
    def create_template(
        self, 
        template_type: str, 
        config: Optional[TemplateConfig] = None
    ) -> TemplateBase:
        """
        创建模板实例
        
        Args:
            template_type: 模板类型
            config: 模板配置
            
        Returns:
            模板实例
            
        Raises:
            ValueError: 未知的模板类型
            ConfigurationError: 配置错误
        """
        if template_type not in self._template_registry:
            available_types = list(self._template_registry.keys())
            raise ValueError(
                f"Unknown template type: {template_type}. "
                f"Available types: {available_types}"
            )
        
        # 检查缓存
        cache_key = f"{template_type}:{config.name if config else 'default'}"
        if self._cache_enabled and cache_key in self._instance_cache:
            logger.debug(f"Returning cached template instance: {cache_key}")
            return self._instance_cache[cache_key]
        
        try:
            # 创建实例
            constructor = self._template_registry[template_type]
            instance = constructor(config)
            
            # 缓存实例
            if self._cache_enabled:
                self._instance_cache[cache_key] = instance
            
            logger.info(f"Created template instance: {template_type} ({instance.config.name})")
            return instance
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create template '{template_type}': {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def get_available_types(self) -> List[str]:
        """获取可用的模板类型列表"""
        return list(self._template_registry.keys())
    
    def enable_cache(self, enabled: bool = True) -> None:
        """启用或禁用实例缓存"""
        self._cache_enabled = enabled
        if not enabled:
            self._instance_cache.clear()
        logger.info(f"Template instance cache {'enabled' if enabled else 'disabled'}")
    
    def clear_cache(self) -> None:
        """清空实例缓存"""
        self._instance_cache.clear()
        logger.info("Template instance cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "enabled": self._cache_enabled,
            "cached_instances": len(self._instance_cache),
            "cache_keys": list(self._instance_cache.keys())
        }


# 全局模板工厂实例
_global_factory: Optional[TemplateFactory] = None


def get_template_factory() -> TemplateFactory:
    """获取全局模板工厂实例"""
    global _global_factory
    if _global_factory is None:
        _global_factory = TemplateFactory()
    return _global_factory


def register_template(template_type: str, constructor: TemplateConstructor) -> None:
    """注册模板类型的便捷函数"""
    factory = get_template_factory()
    factory.register_template(template_type, constructor)


def create_template(
    template_type: str, 
    config: Optional[TemplateConfig] = None
) -> TemplateBase:
    """创建模板实例的便捷函数"""
    factory = get_template_factory()
    return factory.create_template(template_type, config)