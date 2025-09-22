"""
LangChain学习项目的工具管理模块

本模块实现了完整的工具管理系统，包括：
- BaseTool抽象基类：定义工具的通用接口
- ToolManager工具管理器：负责工具的注册、发现和调用
- ToolRegistry工具注册表：管理工具的元数据和索引
- 内置工具实现：提供常用的工具功能
"""

import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type, Callable, get_type_hints
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from pathlib import Path

from pydantic import BaseModel, Field, validator

from .logger import get_logger, log_function_call
from .exceptions import (
    ToolError,
    ValidationError,
    ResourceError,
    ErrorCodes,
    exception_handler
)

logger = get_logger(__name__)


class ToolType(Enum):
    """工具类型枚举"""
    FUNCTION = "function"           # 函数工具
    CLASS = "class"                # 类工具
    API = "api"                    # API工具
    COMMAND = "command"            # 命令行工具
    CHAIN = "chain"                # 链式工具
    COMPOSITE = "composite"        # 复合工具


class ToolStatus(Enum):
    """工具状态枚举"""
    ACTIVE = "active"              # 活跃状态
    INACTIVE = "inactive"          # 非活跃状态
    DEPRECATED = "deprecated"      # 已弃用
    EXPERIMENTAL = "experimental"  # 实验性
    MAINTENANCE = "maintenance"    # 维护中


@dataclass
class ToolSchema:
    """
    工具模式定义
    
    描述工具的输入输出格式、参数类型等信息。
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        验证参数
        
        Args:
            params: 要验证的参数字典
            
        Raises:
            ValidationError: 参数验证失败
        """
        # 检查必需参数
        missing_params = []
        for required_param in self.required_parameters:
            if required_param not in params:
                missing_params.append(required_param)
        
        if missing_params:
            raise ValidationError(
                f"Missing required parameters: {missing_params}",
                error_code=ErrorCodes.VALIDATION_REQUIRED_ERROR,
                context={"tool_name": self.name, "missing_params": missing_params}
            )
        
        # 验证参数类型（简化实现）
        for param_name, param_value in params.items():
            if param_name in self.parameters:
                param_info = self.parameters[param_name]
                expected_type = param_info.get("type")
                
                if expected_type and not self._check_type(param_value, expected_type):
                    raise ValidationError(
                        f"Parameter '{param_name}' type mismatch. "
                        f"Expected: {expected_type}, Got: {type(param_value).__name__}",
                        error_code=ErrorCodes.VALIDATION_TYPE_ERROR,
                        context={
                            "tool_name": self.name,
                            "parameter": param_name,
                            "expected_type": expected_type,
                            "actual_type": type(param_value).__name__
                        }
                    )
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """检查值的类型是否匹配"""
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": lambda x: True
        }
        
        check_func = type_map.get(expected_type.lower())
        if check_func:
            if expected_type.lower() == "any":
                return True
            return isinstance(value, check_func)
        
        return True  # 未知类型默认通过


@dataclass
class ToolMetadata:
    """
    工具元数据
    
    存储工具的详细信息和统计数据。
    """
    name: str
    tool_type: ToolType
    status: ToolStatus = ToolStatus.ACTIVE
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # 使用统计
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_execution_time: float = 0.0
    last_used_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class ToolInput(BaseModel):
    """工具输入数据模型"""
    
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    context: Dict[str, Any] = Field(default_factory=dict, description="执行上下文")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置信息")
    
    class Config:
        extra = "allow"


class ToolOutput(BaseModel):
    """工具输出数据模型"""
    
    result: Any = Field(description="执行结果")
    success: bool = Field(description="是否成功")
    error: Optional[str] = Field(default=None, description="错误信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    execution_time: Optional[float] = Field(default=None, description="执行时间")
    
    class Config:
        extra = "allow"


class BaseTool(ABC):
    """
    工具抽象基类
    
    定义了工具的通用接口，所有具体工具都应该继承此类。
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        schema: Optional[ToolSchema] = None,
        metadata: Optional[ToolMetadata] = None
    ):
        """
        初始化工具
        
        Args:
            name: 工具名称
            description: 工具描述
            schema: 工具模式
            metadata: 工具元数据
        """
        self.name = name
        self.description = description
        self.schema = schema or self._create_default_schema()
        self.metadata = metadata or ToolMetadata(
            name=name,
            tool_type=ToolType.FUNCTION,
            description=description
        )
        
        logger.debug(f"Initialized tool: {self.name}")
    
    def _create_default_schema(self) -> ToolSchema:
        """创建默认的工具模式"""
        return ToolSchema(
            name=self.name,
            description=self.description
        )
    
    @abstractmethod
    async def _execute(self, inputs: ToolInput) -> ToolOutput:
        """
        子类必须实现的核心执行方法
        
        Args:
            inputs: 工具输入
            
        Returns:
            工具输出
            
        Raises:
            ToolError: 工具执行失败
        """
        pass
    
    async def run(self, **kwargs) -> ToolOutput:
        """
        执行工具的主要入口方法
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            工具输出
            
        Raises:
            ToolError: 工具执行失败
            ValidationError: 参数验证失败
        """
        import time
        
        start_time = time.time()
        
        try:
            # 准备输入数据
            tool_input = ToolInput(parameters=kwargs)
            
            # 参数验证
            self.schema.validate_parameters(kwargs)
            
            # 执行工具
            result = await self._execute(tool_input)
            
            # 更新统计信息
            execution_time = time.time() - start_time
            self._update_metrics(True, execution_time)
            
            # 设置执行时间
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            # 更新失败统计
            execution_time = time.time() - start_time
            self._update_metrics(False, execution_time)
            
            # 包装异常
            if not isinstance(e, (ToolError, ValidationError)):
                raise ToolError(
                    f"Tool execution failed: {str(e)}",
                    error_code=ErrorCodes.TOOL_EXECUTION_ERROR,
                    context={"tool_name": self.name},
                    cause=e
                )
            raise
    
    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """更新工具统计信息"""
        import time
        
        self.metadata.usage_count += 1
        self.metadata.last_used_time = time.time()
        
        if success:
            self.metadata.success_count += 1
        else:
            self.metadata.failure_count += 1
        
        # 更新平均执行时间（指数移动平均）
        if self.metadata.average_execution_time == 0:
            self.metadata.average_execution_time = execution_time
        else:
            alpha = 0.1
            self.metadata.average_execution_time = (
                alpha * execution_time + 
                (1 - alpha) * self.metadata.average_execution_time
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具模式的字典表示"""
        return {
            "name": self.schema.name,
            "description": self.schema.description,
            "parameters": self.schema.parameters,
            "required_parameters": self.schema.required_parameters,
            "return_type": self.schema.return_type,
            "examples": self.schema.examples
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取工具的性能指标"""
        return {
            "name": self.metadata.name,
            "usage_count": self.metadata.usage_count,
            "success_count": self.metadata.success_count,
            "failure_count": self.metadata.failure_count,
            "success_rate": self.metadata.success_rate,
            "average_execution_time": self.metadata.average_execution_time,
            "last_used_time": self.metadata.last_used_time
        }


class FunctionTool(BaseTool):
    """
    函数工具
    
    将普通函数包装为工具，支持同步和异步函数。
    """
    
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        初始化函数工具
        
        Args:
            func: 要包装的函数
            name: 工具名称，None则使用函数名
            description: 工具描述，None则使用函数的docstring
        """
        self.func = func
        
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()
        
        # 自动生成模式
        schema = self._create_schema_from_function(func)
        
        metadata = ToolMetadata(
            name=tool_name,
            tool_type=ToolType.FUNCTION,
            description=tool_description
        )
        
        super().__init__(tool_name, tool_description, schema, metadata)
    
    def _create_schema_from_function(self, func: Callable) -> ToolSchema:
        """从函数签名创建工具模式"""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        parameters = {}
        required_parameters = []
        
        for param_name, param in sig.parameters.items():
            param_info = {
                "description": f"Parameter {param_name}"
            }
            
            # 获取类型信息
            if param_name in type_hints:
                param_type = type_hints[param_name]
                param_info["type"] = getattr(param_type, "__name__", str(param_type))
            
            # 获取默认值
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
            else:
                required_parameters.append(param_name)
            
            parameters[param_name] = param_info
        
        # 获取返回类型
        return_type = None
        if "return" in type_hints:
            return_type_hint = type_hints["return"]
            return_type = getattr(return_type_hint, "__name__", str(return_type_hint))
        
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
            required_parameters=required_parameters,
            return_type=return_type
        )
    
    async def _execute(self, inputs: ToolInput) -> ToolOutput:
        """执行函数工具"""
        try:
            # 检查函数是否为异步
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**inputs.parameters)
            else:
                result = self.func(**inputs.parameters)
            
            return ToolOutput(
                result=result,
                success=True,
                metadata={"function_name": self.func.__name__}
            )
            
        except Exception as e:
            return ToolOutput(
                result=None,
                success=False,
                error=str(e),
                metadata={"function_name": self.func.__name__}
            )


class ToolRegistry:
    """
    工具注册表
    
    管理工具的注册、索引和查询功能。
    """
    
    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, BaseTool] = {}
        self._tags_index: Dict[str, List[str]] = {}
        self._type_index: Dict[ToolType, List[str]] = {}
        
        logger.debug("Initialized tool registry")
    
    def register(self, tool: BaseTool) -> None:
        """
        注册工具
        
        Args:
            tool: 要注册的工具实例
            
        Raises:
            ToolError: 工具名称冲突
        """
        if tool.name in self._tools:
            raise ToolError(
                f"Tool '{tool.name}' already registered",
                error_code=ErrorCodes.TOOL_REGISTRATION_ERROR,
                context={"tool_name": tool.name}
            )
        
        self._tools[tool.name] = tool
        
        # 更新索引
        self._update_indexes(tool)
        
        logger.info(f"Registered tool: {tool.name}")
    
    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> FunctionTool:
        """
        注册函数为工具
        
        Args:
            func: 要注册的函数
            name: 工具名称
            description: 工具描述
            
        Returns:
            创建的函数工具实例
        """
        tool = FunctionTool(func, name, description)
        self.register(tool)
        return tool
    
    def unregister(self, name: str) -> None:
        """
        注销工具
        
        Args:
            name: 工具名称
        """
        if name in self._tools:
            tool = self._tools[name]
            del self._tools[name]
            
            # 更新索引
            self._remove_from_indexes(tool)
            
            logger.info(f"Unregistered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        获取工具实例
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例，不存在则返回None
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """获取所有工具名称列表"""
        return list(self._tools.keys())
    
    def search_by_tag(self, tag: str) -> List[BaseTool]:
        """
        按标签搜索工具
        
        Args:
            tag: 标签名称
            
        Returns:
            匹配的工具列表
        """
        tool_names = self._tags_index.get(tag, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def search_by_type(self, tool_type: ToolType) -> List[BaseTool]:
        """
        按类型搜索工具
        
        Args:
            tool_type: 工具类型
            
        Returns:
            匹配的工具列表
        """
        tool_names = self._type_index.get(tool_type, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def search_by_name(self, pattern: str) -> List[BaseTool]:
        """
        按名称模式搜索工具
        
        Args:
            pattern: 名称模式（支持通配符*）
            
        Returns:
            匹配的工具列表
        """
        import fnmatch
        
        matching_tools = []
        for tool_name, tool in self._tools.items():
            if fnmatch.fnmatch(tool_name.lower(), pattern.lower()):
                matching_tools.append(tool)
        
        return matching_tools
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        total_tools = len(self._tools)
        active_tools = sum(
            1 for tool in self._tools.values()
            if tool.metadata.status == ToolStatus.ACTIVE
        )
        
        type_counts = {}
        for tool_type in ToolType:
            type_counts[tool_type.value] = len(self._type_index.get(tool_type, []))
        
        return {
            "total_tools": total_tools,
            "active_tools": active_tools,
            "type_distribution": type_counts,
            "total_executions": sum(tool.metadata.usage_count for tool in self._tools.values()),
            "total_successes": sum(tool.metadata.success_count for tool in self._tools.values())
        }
    
    def _update_indexes(self, tool: BaseTool) -> None:
        """更新索引"""
        # 标签索引
        for tag in tool.metadata.tags:
            if tag not in self._tags_index:
                self._tags_index[tag] = []
            self._tags_index[tag].append(tool.name)
        
        # 类型索引
        tool_type = tool.metadata.tool_type
        if tool_type not in self._type_index:
            self._type_index[tool_type] = []
        self._type_index[tool_type].append(tool.name)
    
    def _remove_from_indexes(self, tool: BaseTool) -> None:
        """从索引中移除工具"""
        # 从标签索引移除
        for tag in tool.metadata.tags:
            if tag in self._tags_index:
                self._tags_index[tag] = [
                    name for name in self._tags_index[tag]
                    if name != tool.name
                ]
                if not self._tags_index[tag]:
                    del self._tags_index[tag]
        
        # 从类型索引移除
        tool_type = tool.metadata.tool_type
        if tool_type in self._type_index:
            self._type_index[tool_type] = [
                name for name in self._type_index[tool_type]
                if name != tool.name
            ]
            if not self._type_index[tool_type]:
                del self._type_index[tool_type]


class ToolManager:
    """
    工具管理器
    
    提供工具的统一管理接口，包括注册、发现、调用和监控功能。
    """
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        """
        初始化工具管理器
        
        Args:
            registry: 工具注册表，None则创建新的注册表
        """
        self.registry = registry or ToolRegistry()
        self._middleware_stack: List[Callable] = []
        
        # 注册内置工具
        self._register_builtin_tools()
        
        logger.info("Tool manager initialized")
    
    def register_tool(self, tool: BaseTool) -> None:
        """注册工具"""
        self.registry.register(tool)
    
    def register_function(
        self,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Union[FunctionTool, Callable]:
        """
        注册函数为工具（支持装饰器语法）
        
        Args:
            func: 函数对象
            name: 工具名称
            description: 工具描述
            
        Returns:
            FunctionTool实例或装饰器函数
            
        Usage:
            # 直接注册
            tool = manager.register_function(my_function)
            
            # 装饰器语法
            @manager.register_function
            def my_tool():
                pass
            
            # 带参数的装饰器
            @manager.register_function(name="custom_name", description="Custom tool")
            def my_tool():
                pass
        """
        if func is None:
            # 装饰器模式
            def decorator(f):
                return self.registry.register_function(f, name, description)
            return decorator
        else:
            # 直接调用模式
            return self.registry.register_function(func, name, description)
    
    async def call_tool(self, name: str, **kwargs) -> ToolOutput:
        """
        调用工具
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具输出
            
        Raises:
            ToolError: 工具不存在或执行失败
        """
        tool = self.registry.get_tool(name)
        if tool is None:
            raise ToolError(
                f"Tool '{name}' not found",
                error_code=ErrorCodes.TOOL_NOT_FOUND,
                context={"tool_name": name}
            )
        
        # 应用中间件
        for middleware in self._middleware_stack:
            kwargs = await self._apply_middleware(middleware, tool, kwargs)
        
        # 执行工具
        return await tool.run(**kwargs)
    
    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具模式"""
        tool = self.registry.get_tool(name)
        return tool.get_schema() if tool else None
    
    def list_tools(self, include_metrics: bool = False) -> List[Dict[str, Any]]:
        """
        列出所有工具
        
        Args:
            include_metrics: 是否包含性能指标
            
        Returns:
            工具信息列表
        """
        tools_info = []
        
        for tool_name in self.registry.list_tools():
            tool = self.registry.get_tool(tool_name)
            if tool:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "type": tool.metadata.tool_type.value,
                    "status": tool.metadata.status.value,
                    "version": tool.metadata.version,
                    "tags": tool.metadata.tags
                }
                
                if include_metrics:
                    tool_info["metrics"] = tool.get_metrics()
                
                tools_info.append(tool_info)
        
        return tools_info
    
    def search_tools(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        tool_type: Optional[ToolType] = None
    ) -> List[BaseTool]:
        """
        搜索工具
        
        Args:
            query: 名称查询字符串
            tags: 标签列表
            tool_type: 工具类型
            
        Returns:
            匹配的工具列表
        """
        results = set()
        
        # 按名称搜索
        if query:
            results.update(self.registry.search_by_name(query))
        
        # 按标签搜索
        if tags:
            for tag in tags:
                results.update(self.registry.search_by_tag(tag))
        
        # 按类型搜索
        if tool_type:
            if results:
                # 交集
                type_results = set(self.registry.search_by_type(tool_type))
                results = results.intersection(type_results)
            else:
                results.update(self.registry.search_by_type(tool_type))
        
        # 如果没有指定任何条件，返回所有工具
        if not any([query, tags, tool_type]):
            results.update(self.registry._tools.values())
        
        return list(results)
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        添加中间件
        
        Args:
            middleware: 中间件函数
        """
        self._middleware_stack.append(middleware)
        logger.debug(f"Added middleware: {middleware.__name__}")
    
    async def _apply_middleware(
        self,
        middleware: Callable,
        tool: BaseTool,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用中间件"""
        try:
            if asyncio.iscoroutinefunction(middleware):
                return await middleware(tool, kwargs)
            else:
                return middleware(tool, kwargs)
        except Exception as e:
            logger.warning(f"Middleware {middleware.__name__} failed: {e}")
            return kwargs
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取工具管理器统计信息"""
        return self.registry.get_statistics()
    
    def _register_builtin_tools(self) -> None:
        """注册内置工具"""
        
        # 字符串处理工具
        @self.register_function(name="string_length", description="获取字符串长度")
        def get_string_length(text: str) -> int:
            """获取字符串的长度"""
            return len(text)
        
        @self.register_function(name="string_upper", description="转换为大写")
        def string_to_upper(text: str) -> str:
            """将字符串转换为大写"""
            return text.upper()
        
        @self.register_function(name="string_lower", description="转换为小写")
        def string_to_lower(text: str) -> str:
            """将字符串转换为小写"""
            return text.lower()
        
        # 数学计算工具
        @self.register_function(name="math_add", description="加法运算")
        def math_add(a: float, b: float) -> float:
            """执行加法运算"""
            return a + b
        
        @self.register_function(name="math_multiply", description="乘法运算")
        def math_multiply(a: float, b: float) -> float:
            """执行乘法运算"""
            return a * b
        
        # 文件操作工具
        @self.register_function(name="read_file", description="读取文件内容")
        async def read_file(file_path: str) -> str:
            """读取文件内容"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                raise ToolError(f"Failed to read file: {e}")
        
        # JSON处理工具
        @self.register_function(name="parse_json", description="解析JSON字符串")
        def parse_json(json_str: str) -> Dict[str, Any]:
            """解析JSON字符串为Python对象"""
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid JSON: {e}")
        
        logger.debug("Registered builtin tools")


# 全局工具管理器实例
_global_manager: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    """
    获取全局工具管理器实例
    
    Returns:
        工具管理器实例
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = ToolManager()
    
    return _global_manager


def register_tool(tool: BaseTool) -> None:
    """注册工具到全局管理器"""
    get_tool_manager().register_tool(tool)


def register_function(
    func: Optional[Callable] = None,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Union[FunctionTool, Callable]:
    """注册函数为工具到全局管理器"""
    return get_tool_manager().register_function(func, name, description)


async def call_tool(name: str, **kwargs) -> ToolOutput:
    """调用全局管理器中的工具"""
    return await get_tool_manager().call_tool(name, **kwargs)