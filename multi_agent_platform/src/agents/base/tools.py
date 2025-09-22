"""
智能体工具系统模块

提供智能体的工具集成功能，包括：
- 工具抽象接口
- 工具注册和发现
- 工具执行管理
- 工具结果处理
"""

from typing import Dict, Any, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import json
import inspect
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """工具类型枚举"""
    EXTERNAL_API = "external_api"        # 外部API调用
    DATABASE = "database"                # 数据库操作
    FILE_SYSTEM = "file_system"          # 文件系统操作
    COMPUTATION = "computation"          # 计算工具
    COMMUNICATION = "communication"      # 通信工具
    VISUALIZATION = "visualization"      # 可视化工具
    WEB_SCRAPING = "web_scraping"       # 网页抓取
    DATA_PROCESSING = "data_processing"  # 数据处理
    CUSTOM = "custom"                    # 自定义工具


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }
    
    @classmethod
    def success_result(cls, data: Any, execution_time: float = 0.0, 
                      metadata: Dict[str, Any] = None) -> 'ToolResult':
        """创建成功结果"""
        return cls(
            success=True,
            data=data,
            execution_time=execution_time,
            metadata=metadata or {}
        )
    
    @classmethod
    def error_result(cls, error: str, execution_time: float = 0.0,
                    metadata: Dict[str, Any] = None) -> 'ToolResult':
        """创建错误结果"""
        return cls(
            success=False,
            error=error,
            execution_time=execution_time,
            metadata=metadata or {}
        )


class BaseTool(ABC):
    """工具基础抽象类"""
    
    def __init__(self, name: str, description: str, tool_type: ToolType):
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self.usage_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        self.created_time = datetime.now()
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取工具参数定义"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数"""
        param_def = self.get_parameters()
        required_params = {k for k, v in param_def.items() 
                          if v.get('required', False)}
        
        # 检查必需参数
        provided_params = set(parameters.keys())
        missing_params = required_params - provided_params
        
        if missing_params:
            raise ValueError(f"缺少必需参数: {missing_params}")
        
        # 检查参数类型（简单检查）
        for param_name, param_value in parameters.items():
            if param_name in param_def:
                expected_type = param_def[param_name].get('type')
                if expected_type and not isinstance(param_value, expected_type):
                    # 尝试类型转换
                    try:
                        parameters[param_name] = expected_type(param_value)
                    except (ValueError, TypeError):
                        raise TypeError(f"参数 {param_name} 类型错误，期望 {expected_type.__name__}")
        
        return True
    
    async def run(self, **kwargs) -> ToolResult:
        """运行工具（包含统计和验证）"""
        start_time = datetime.now()
        
        try:
            # 验证参数
            self.validate_parameters(kwargs)
            
            # 执行工具
            result = await self.execute(**kwargs)
            
            # 更新统计
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            self.usage_count += 1
            self.total_execution_time += execution_time
            
            if not result.success:
                self.error_count += 1
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.usage_count += 1
            self.error_count += 1
            self.total_execution_time += execution_time
            
            logger.error(f"工具 {self.name} 执行失败: {e}")
            return ToolResult.error_result(str(e), execution_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        avg_execution_time = (self.total_execution_time / self.usage_count 
                             if self.usage_count > 0 else 0)
        error_rate = (self.error_count / self.usage_count 
                     if self.usage_count > 0 else 0)
        
        return {
            'name': self.name,
            'type': self.tool_type.value,
            'usage_count': self.usage_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'avg_execution_time': avg_execution_time,
            'total_execution_time': self.total_execution_time,
            'created_time': self.created_time.isoformat()
        }


class FunctionTool(BaseTool):
    """函数工具包装器"""
    
    def __init__(self, name: str, func: Callable, description: str = "",
                 tool_type: ToolType = ToolType.CUSTOM):
        super().__init__(name, description or func.__doc__ or name, tool_type)
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)
        
        # 解析函数参数
        self.signature = inspect.signature(func)
        self._parameters = self._parse_parameters()
    
    def _parse_parameters(self) -> Dict[str, Any]:
        """解析函数参数"""
        params = {}
        
        for param_name, param in self.signature.parameters.items():
            param_info = {
                'name': param_name,
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None
            }
            
            # 尝试获取类型注解
            if param.annotation != inspect.Parameter.empty:
                param_info['type'] = param.annotation
            
            params[param_name] = param_info
        
        return params
    
    async def execute(self, **kwargs) -> ToolResult:
        """执行函数"""
        try:
            if self.is_async:
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
            
            return ToolResult.success_result(result)
            
        except Exception as e:
            return ToolResult.error_result(str(e))
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取参数定义"""
        return self._parameters


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolType, List[str]] = {
            tool_type: [] for tool_type in ToolType
        }
    
    def register_tool(self, tool: BaseTool) -> None:
        """注册工具"""
        if tool.name in self._tools:
            logger.warning(f"工具 {tool.name} 已存在，将被覆盖")
        
        self._tools[tool.name] = tool
        
        # 添加到分类
        if tool.name not in self._categories[tool.tool_type]:
            self._categories[tool.tool_type].append(tool.name)
        
        logger.info(f"工具 {tool.name} 注册成功")
    
    def register_function(self, name: str, func: Callable, 
                         description: str = "", tool_type: ToolType = ToolType.CUSTOM) -> None:
        """注册函数作为工具"""
        tool = FunctionTool(name, func, description, tool_type)
        self.register_tool(tool)
    
    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        if name in self._tools:
            tool = self._tools.pop(name)
            
            # 从分类中移除
            if name in self._categories[tool.tool_type]:
                self._categories[tool.tool_type].remove(name)
            
            logger.info(f"工具 {name} 注销成功")
            return True
        
        return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[str]:
        """列出工具"""
        if tool_type:
            return self._categories[tool_type].copy()
        return list(self._tools.keys())
    
    def search_tools(self, query: str) -> List[str]:
        """搜索工具"""
        query_lower = query.lower()
        results = []
        
        for name, tool in self._tools.items():
            if (query_lower in name.lower() or 
                query_lower in tool.description.lower()):
                results.append(name)
        
        return results
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具信息"""
        tool = self._tools.get(name)
        if not tool:
            return None
        
        return {
            'name': tool.name,
            'description': tool.description,
            'type': tool.tool_type.value,
            'parameters': tool.get_parameters(),
            'stats': tool.get_stats()
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计"""
        total_tools = len(self._tools)
        tools_by_type = {
            tool_type.value: len(tools) 
            for tool_type, tools in self._categories.items()
        }
        
        total_usage = sum(tool.usage_count for tool in self._tools.values())
        total_errors = sum(tool.error_count for tool in self._tools.values())
        
        return {
            'total_tools': total_tools,
            'tools_by_type': tools_by_type,
            'total_usage': total_usage,
            'total_errors': total_errors,
            'error_rate': total_errors / total_usage if total_usage > 0 else 0
        }


class AgentTools:
    """智能体工具管理器"""
    
    def __init__(self, agent_id: str, registry: Optional[ToolRegistry] = None):
        self.agent_id = agent_id
        self.registry = registry or ToolRegistry()
        self.enabled_tools: List[str] = []
        self.tool_configs: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def enable_tool(self, tool_name: str, config: Dict[str, Any] = None) -> bool:
        """启用工具"""
        if tool_name not in self.registry._tools:
            logger.error(f"工具 {tool_name} 不存在")
            return False
        
        if tool_name not in self.enabled_tools:
            self.enabled_tools.append(tool_name)
        
        if config:
            self.tool_configs[tool_name] = config
        
        logger.info(f"智能体 {self.agent_id} 启用工具 {tool_name}")
        return True
    
    def disable_tool(self, tool_name: str) -> bool:
        """禁用工具"""
        if tool_name in self.enabled_tools:
            self.enabled_tools.remove(tool_name)
            self.tool_configs.pop(tool_name, None)
            logger.info(f"智能体 {self.agent_id} 禁用工具 {tool_name}")
            return True
        return False
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """检查工具是否启用"""
        return tool_name in self.enabled_tools
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具"""
        if not self.is_tool_enabled(tool_name):
            return ToolResult.error_result(f"工具 {tool_name} 未启用")
        
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult.error_result(f"工具 {tool_name} 不存在")
        
        # 合并工具配置
        tool_config = self.tool_configs.get(tool_name, {})
        merged_kwargs = {**tool_config, **kwargs}
        
        # 记录执行开始
        execution_record = {
            'tool_name': tool_name,
            'agent_id': self.agent_id,
            'start_time': datetime.now().isoformat(),
            'parameters': kwargs  # 不包含配置，只记录用户参数
        }
        
        # 执行工具
        result = await tool.run(**merged_kwargs)
        
        # 记录执行结果
        execution_record.update({
            'end_time': datetime.now().isoformat(),
            'success': result.success,
            'execution_time': result.execution_time,
            'error': result.error
        })
        
        # 添加到历史记录
        self._add_to_history(execution_record)
        
        return result
    
    def _add_to_history(self, record: Dict[str, Any]) -> None:
        """添加到执行历史"""
        self._execution_history.append(record)
        
        # 限制历史记录大小
        if len(self._execution_history) > self.max_history_size:
            self._execution_history = self._execution_history[-self.max_history_size:]
    
    def get_enabled_tools(self) -> List[Dict[str, Any]]:
        """获取启用的工具列表"""
        tools_info = []
        
        for tool_name in self.enabled_tools:
            tool_info = self.registry.get_tool_info(tool_name)
            if tool_info:
                # 添加配置信息
                tool_info['config'] = self.tool_configs.get(tool_name, {})
                tools_info.append(tool_info)
        
        return tools_info
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取执行历史"""
        if limit:
            return self._execution_history[-limit:]
        return self._execution_history.copy()
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """获取工具使用统计"""
        tool_usage = {}
        
        for record in self._execution_history:
            tool_name = record['tool_name']
            if tool_name not in tool_usage:
                tool_usage[tool_name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'total_execution_time': 0.0
                }
            
            stats = tool_usage[tool_name]
            stats['total_calls'] += 1
            stats['total_execution_time'] += record.get('execution_time', 0)
            
            if record['success']:
                stats['successful_calls'] += 1
            else:
                stats['failed_calls'] += 1
        
        # 计算平均值和错误率
        for tool_name, stats in tool_usage.items():
            if stats['total_calls'] > 0:
                stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_calls']
                stats['error_rate'] = stats['failed_calls'] / stats['total_calls']
            else:
                stats['avg_execution_time'] = 0
                stats['error_rate'] = 0
        
        return tool_usage
    
    def clear_history(self) -> None:
        """清空执行历史"""
        self._execution_history.clear()
    
    def export_config(self) -> Dict[str, Any]:
        """导出工具配置"""
        return {
            'agent_id': self.agent_id,
            'enabled_tools': self.enabled_tools.copy(),
            'tool_configs': self.tool_configs.copy()
        }
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """导入工具配置"""
        self.enabled_tools = config.get('enabled_tools', [])
        self.tool_configs = config.get('tool_configs', {})


# 全局工具注册表实例
global_tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> None:
    """注册工具到全局注册表"""
    global_tool_registry.register_tool(tool)


def register_function_tool(name: str, func: Callable, 
                          description: str = "", tool_type: ToolType = ToolType.CUSTOM) -> None:
    """注册函数工具到全局注册表"""
    global_tool_registry.register_function(name, func, description, tool_type)


def get_global_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    return global_tool_registry