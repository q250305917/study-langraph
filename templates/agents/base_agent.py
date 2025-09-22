"""
基础Agent抽象类

定义Agent的核心接口和基础功能，为所有具体Agent实现提供统一的架构基础。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..base.template_base import TemplateBase


class AgentState(Enum):
    """Agent执行状态枚举"""
    IDLE = "idle"              # 空闲状态
    THINKING = "thinking"      # 思考中
    ACTING = "acting"          # 执行动作中
    RESPONDING = "responding"  # 生成回复中
    ERROR = "error"           # 错误状态
    COMPLETED = "completed"   # 完成状态


@dataclass
class ToolDefinition:
    """工具定义类"""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    async_func: bool = False


@dataclass
class ExecutionMetrics:
    """执行指标类"""
    start_time: float = 0.0
    end_time: float = 0.0
    think_time: float = 0.0
    act_time: float = 0.0
    respond_time: float = 0.0
    total_time: float = 0.0
    success: bool = False
    error_count: int = 0
    
    def duration(self) -> float:
        """计算总执行时间"""
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0


class BaseAgent(TemplateBase[str, str], ABC):
    """
    基础Agent抽象类
    
    定义Agent的核心接口和基础功能：
    1. 状态管理 - 跟踪Agent的执行状态
    2. 工具系统 - 注册和调用外部工具
    3. 决策流程 - think -> act -> respond 标准流程
    4. 记忆集成 - 与记忆系统的集成
    5. 性能监控 - 执行指标收集和分析
    """
    
    def __init__(self, agent_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id or f"agent_{int(time.time())}"
        self.state = AgentState.IDLE
        self.tools: Dict[str, ToolDefinition] = {}
        self.memory = None
        self.execution_history: List[ExecutionMetrics] = []
        self.current_metrics = ExecutionMetrics()
        
    @property 
    def default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "max_iterations": 10,
            "timeout": 30.0,
            "enable_memory": True,
            "enable_metrics": True,
            "log_level": "INFO"
        }
    
    def register_tool(self, tool: ToolDefinition):
        """
        注册工具
        
        Args:
            tool: 工具定义
        """
        self.tools[tool.name] = tool
        self.logger.info(f"已注册工具: {tool.name}")
    
    def register_tools(self, tools: List[ToolDefinition]):
        """
        批量注册工具
        
        Args:
            tools: 工具定义列表
        """
        for tool in tools:
            self.register_tool(tool)
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ValueError: 工具不存在
        """
        if tool_name not in self.tools:
            raise ValueError(f"工具 {tool_name} 未注册")
        
        tool = self.tools[tool_name]
        
        try:
            if tool.async_func:
                result = await tool.func(**kwargs)
            else:
                result = tool.func(**kwargs)
            
            self.logger.debug(f"工具 {tool_name} 执行成功")
            return result
            
        except Exception as e:
            self.logger.error(f"工具 {tool_name} 执行失败: {e}")
            raise
    
    def set_memory(self, memory):
        """
        设置记忆系统
        
        Args:
            memory: 记忆系统实例
        """
        self.memory = memory
        self.logger.info(f"已设置记忆系统: {type(memory).__name__}")
    
    def _update_state(self, new_state: AgentState):
        """
        更新Agent状态
        
        Args:
            new_state: 新状态
        """
        old_state = self.state
        self.state = new_state
        self.logger.debug(f"状态变更: {old_state.value} -> {new_state.value}")
    
    def _start_metrics(self):
        """开始指标收集"""
        if self.config.get("enable_metrics", True):
            self.current_metrics = ExecutionMetrics()
            self.current_metrics.start_time = time.time()
    
    def _end_metrics(self, success: bool = True):
        """结束指标收集"""
        if self.config.get("enable_metrics", True):
            self.current_metrics.end_time = time.time()
            self.current_metrics.total_time = self.current_metrics.duration()
            self.current_metrics.success = success
            self.execution_history.append(self.current_metrics)
    
    async def execute_async(self, input_data: str, **kwargs) -> str:
        """
        异步执行Agent主流程
        
        Args:
            input_data: 输入数据
            **kwargs: 额外参数
            
        Returns:
            Agent输出结果
            
        Raises:
            Exception: 执行过程中的各种异常
        """
        self._start_metrics()
        session_id = kwargs.get("session_id", "default")
        
        try:
            # 状态：开始思考
            self._update_state(AgentState.THINKING)
            think_start = time.time()
            
            # 构建上下文
            context = await self._build_context(input_data, session_id, **kwargs)
            
            # 思考阶段
            decision = await self.think(input_data, context)
            self.current_metrics.think_time = time.time() - think_start
            
            # 状态：执行动作
            self._update_state(AgentState.ACTING)
            act_start = time.time()
            
            # 行动阶段
            action_result = await self.act(decision, context)
            self.current_metrics.act_time = time.time() - act_start
            
            # 状态：生成回复
            self._update_state(AgentState.RESPONDING)
            respond_start = time.time()
            
            # 回复阶段
            response = await self.respond(action_result, context)
            self.current_metrics.respond_time = time.time() - respond_start
            
            # 更新记忆
            if self.memory and self.config.get("enable_memory", True):
                await self._update_memory(input_data, response, context)
            
            # 状态：完成
            self._update_state(AgentState.COMPLETED)
            self._end_metrics(success=True)
            
            return response
            
        except Exception as e:
            self._update_state(AgentState.ERROR)
            self.current_metrics.error_count += 1
            self._end_metrics(success=False)
            self.logger.error(f"Agent执行失败: {e}")
            raise
    
    async def _build_context(self, input_data: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        构建执行上下文
        
        Args:
            input_data: 输入数据
            session_id: 会话ID
            **kwargs: 额外参数
            
        Returns:
            上下文字典
        """
        context = {
            "input": input_data,
            "session_id": session_id,
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "state": self.state.value,
            "available_tools": list(self.tools.keys()),
            **kwargs
        }
        
        # 加载记忆
        if self.memory:
            try:
                memory_data = await self.memory.get_memory(session_id)
                context["memory"] = memory_data
            except Exception as e:
                self.logger.warning(f"获取记忆失败: {e}")
                context["memory"] = {}
        
        return context
    
    async def _update_memory(self, input_data: str, response: str, context: Dict[str, Any]):
        """
        更新记忆系统
        
        Args:
            input_data: 输入数据
            response: 输出响应
            context: 执行上下文
        """
        if not self.memory:
            return
            
        try:
            await self.memory.add_message(
                session_id=context["session_id"],
                role="user",
                content=input_data
            )
            
            await self.memory.add_message(
                session_id=context["session_id"],
                role="assistant", 
                content=response
            )
            
            self.logger.debug("记忆更新成功")
            
        except Exception as e:
            self.logger.error(f"记忆更新失败: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取性能指标摘要
        
        Returns:
            指标摘要字典
        """
        if not self.execution_history:
            return {"total_executions": 0}
        
        successful = [m for m in self.execution_history if m.success]
        failed = [m for m in self.execution_history if not m.success]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "success_rate": len(successful) / len(self.execution_history) if self.execution_history else 0,
            "avg_execution_time": sum(m.total_time for m in self.execution_history) / len(self.execution_history),
            "avg_think_time": sum(m.think_time for m in self.execution_history) / len(self.execution_history),
            "avg_act_time": sum(m.act_time for m in self.execution_history) / len(self.execution_history),
            "avg_respond_time": sum(m.respond_time for m in self.execution_history) / len(self.execution_history),
        }
    
    # 抽象方法 - 子类必须实现
    
    @abstractmethod
    async def think(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考和决策过程
        
        Args:
            input_data: 输入数据
            context: 执行上下文
            
        Returns:
            决策结果字典
        """
        pass
    
    @abstractmethod
    async def act(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行动作
        
        Args:
            decision: 决策结果
            context: 执行上下文
            
        Returns:
            动作执行结果
        """
        pass
    
    @abstractmethod
    async def respond(self, action_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        生成最终回复
        
        Args:
            action_result: 动作执行结果
            context: 执行上下文
            
        Returns:
            最终回复文本
        """
        pass