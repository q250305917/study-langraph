"""
BaseAgent 基础测试用例

测试基础Agent抽象类的核心功能。
"""

import pytest
import asyncio
import time
from typing import Any, Dict
from unittest.mock import Mock, AsyncMock

from templates.agents.base_agent import (
    BaseAgent, AgentState, ToolDefinition, ExecutionMetrics
)


class MockAgent(BaseAgent):
    """用于测试的模拟Agent实现"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.think_called = False
        self.act_called = False
        self.respond_called = False
    
    async def think(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        self.think_called = True
        return {
            "decision": "mock_decision",
            "confidence": 0.9,
            "reasoning": "This is a mock decision"
        }
    
    async def act(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        self.act_called = True
        return {
            "action_result": "mock_action_completed",
            "data": {"processed": True}
        }
    
    async def respond(self, action_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        self.respond_called = True
        return f"模拟回复：{action_result['action_result']}"


class TestBaseAgent:
    """BaseAgent测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.agent = MockAgent(agent_id="test_agent")
    
    def test_agent_initialization(self):
        """测试Agent初始化"""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.state == AgentState.IDLE
        assert isinstance(self.agent.tools, dict)
        assert isinstance(self.agent.execution_history, list)
        assert len(self.agent.execution_history) == 0
    
    def test_default_config(self):
        """测试默认配置"""
        config = self.agent.default_config
        assert "max_iterations" in config
        assert "timeout" in config
        assert "enable_memory" in config
        assert "enable_metrics" in config
        assert config["max_iterations"] == 10
        assert config["timeout"] == 30.0
    
    def test_tool_registration(self):
        """测试工具注册"""
        def mock_tool_func(x: int, y: int) -> int:
            return x + y
        
        tool = ToolDefinition(
            name="add_numbers",
            description="添加两个数字",
            func=mock_tool_func,
            parameters={"x": {"type": "int"}, "y": {"type": "int"}}
        )
        
        self.agent.register_tool(tool)
        
        assert "add_numbers" in self.agent.tools
        assert self.agent.tools["add_numbers"].name == "add_numbers"
        assert self.agent.tools["add_numbers"].func == mock_tool_func
    
    def test_batch_tool_registration(self):
        """测试批量工具注册"""
        tools = [
            ToolDefinition(
                name="tool1",
                description="工具1",
                func=lambda x: x * 2
            ),
            ToolDefinition(
                name="tool2", 
                description="工具2",
                func=lambda x: x + 1
            )
        ]
        
        self.agent.register_tools(tools)
        
        assert len(self.agent.tools) == 2
        assert "tool1" in self.agent.tools
        assert "tool2" in self.agent.tools
    
    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """测试工具调用"""
        def add_func(x: int, y: int) -> int:
            return x + y
        
        tool = ToolDefinition(
            name="add",
            description="加法工具",
            func=add_func
        )
        
        self.agent.register_tool(tool)
        
        result = await self.agent.call_tool("add", x=3, y=5)
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_async_tool_calling(self):
        """测试异步工具调用"""
        async def async_multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.01)  # 模拟异步操作
            return x * y
        
        tool = ToolDefinition(
            name="multiply",
            description="乘法工具",
            func=async_multiply,
            async_func=True
        )
        
        self.agent.register_tool(tool)
        
        result = await self.agent.call_tool("multiply", x=4, y=6)
        assert result == 24
    
    @pytest.mark.asyncio
    async def test_tool_calling_error(self):
        """测试工具调用错误处理"""
        with pytest.raises(ValueError, match="工具 nonexistent 未注册"):
            await self.agent.call_tool("nonexistent", param="value")
    
    def test_memory_setting(self):
        """测试记忆系统设置"""
        mock_memory = Mock()
        mock_memory.__class__.__name__ = "MockMemory"
        
        self.agent.set_memory(mock_memory)
        
        assert self.agent.memory == mock_memory
    
    def test_state_management(self):
        """测试状态管理"""
        assert self.agent.state == AgentState.IDLE
        
        self.agent._update_state(AgentState.THINKING)
        assert self.agent.state == AgentState.THINKING
        
        self.agent._update_state(AgentState.ACTING)
        assert self.agent.state == AgentState.ACTING
        
        self.agent._update_state(AgentState.COMPLETED)
        assert self.agent.state == AgentState.COMPLETED
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        self.agent._start_metrics()
        
        assert self.agent.current_metrics.start_time > 0
        assert self.agent.current_metrics.end_time == 0.0
        assert self.agent.current_metrics.success == False
    
    def test_metrics_completion(self):
        """测试指标完成"""
        self.agent._start_metrics()
        time.sleep(0.01)  # 确保有执行时间
        self.agent._end_metrics(success=True)
        
        assert self.agent.current_metrics.end_time > self.agent.current_metrics.start_time
        assert self.agent.current_metrics.success == True
        assert len(self.agent.execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_execute_async_flow(self):
        """测试完整的异步执行流程"""
        input_data = "测试输入"
        result = await self.agent.execute_async(input_data, session_id="test_session")
        
        # 验证执行流程
        assert self.agent.think_called
        assert self.agent.act_called
        assert self.agent.respond_called
        
        # 验证结果
        assert result == "模拟回复：mock_action_completed"
        
        # 验证状态
        assert self.agent.state == AgentState.COMPLETED
        
        # 验证指标
        assert len(self.agent.execution_history) == 1
        assert self.agent.execution_history[0].success == True
    
    @pytest.mark.asyncio
    async def test_execute_with_memory(self):
        """测试带记忆的执行"""
        # 设置模拟记忆系统
        mock_memory = AsyncMock()
        mock_memory.get_memory.return_value = {"previous": "data"}
        mock_memory.add_message = AsyncMock()
        
        self.agent.set_memory(mock_memory)
        
        result = await self.agent.execute_async("测试输入", session_id="test_session")
        
        # 验证记忆系统调用
        mock_memory.get_memory.assert_called_once_with("test_session")
        assert mock_memory.add_message.call_count == 2  # user和assistant消息
    
    @pytest.mark.asyncio
    async def test_context_building(self):
        """测试上下文构建"""
        context = await self.agent._build_context(
            "测试输入", 
            "test_session",
            extra_param="extra_value"
        )
        
        assert context["input"] == "测试输入"
        assert context["session_id"] == "test_session"
        assert context["agent_id"] == "test_agent"
        assert context["extra_param"] == "extra_value"
        assert "timestamp" in context
        assert "state" in context
        assert "available_tools" in context
    
    def test_metrics_summary(self):
        """测试指标摘要"""
        # 添加一些模拟执行历史
        metrics1 = ExecutionMetrics()
        metrics1.success = True
        metrics1.total_time = 1.5
        metrics1.think_time = 0.5
        metrics1.act_time = 0.7
        metrics1.respond_time = 0.3
        
        metrics2 = ExecutionMetrics()
        metrics2.success = False
        metrics2.total_time = 2.0
        metrics2.think_time = 0.8
        metrics2.act_time = 0.9
        metrics2.respond_time = 0.3
        
        self.agent.execution_history = [metrics1, metrics2]
        
        summary = self.agent.get_metrics_summary()
        
        assert summary["total_executions"] == 2
        assert summary["successful_executions"] == 1
        assert summary["failed_executions"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["avg_execution_time"] == 1.75
        assert summary["avg_think_time"] == 0.65
        assert summary["avg_act_time"] == 0.8
        assert summary["avg_respond_time"] == 0.3
    
    def test_empty_metrics_summary(self):
        """测试空指标摘要"""
        summary = self.agent.get_metrics_summary()
        assert summary["total_executions"] == 0
    
    @pytest.mark.asyncio
    async def test_execution_error_handling(self):
        """测试执行错误处理"""
        # 创建一个会抛出异常的Agent
        class ErrorAgent(BaseAgent):
            async def think(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
                raise ValueError("Think error")
            
            async def act(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
                return {}
            
            async def respond(self, action_result: Dict[str, Any], context: Dict[str, Any]) -> str:
                return ""
        
        error_agent = ErrorAgent()
        
        with pytest.raises(ValueError, match="Think error"):
            await error_agent.execute_async("测试输入")
        
        # 验证错误状态
        assert error_agent.state == AgentState.ERROR
        assert len(error_agent.execution_history) == 1
        assert error_agent.execution_history[0].success == False
        assert error_agent.execution_history[0].error_count == 1
    
    @pytest.mark.asyncio
    async def test_memory_error_handling(self):
        """测试记忆系统错误处理"""
        # 设置会抛出异常的模拟记忆系统
        mock_memory = AsyncMock()
        mock_memory.get_memory.side_effect = Exception("Memory error")
        mock_memory.add_message.side_effect = Exception("Memory add error")
        
        self.agent.set_memory(mock_memory)
        
        # 执行应该成功，但记忆操作失败
        result = await self.agent.execute_async("测试输入", session_id="test_session")
        
        assert result == "模拟回复：mock_action_completed"
        assert self.agent.state == AgentState.COMPLETED
    
    def test_configuration_override(self):
        """测试配置覆盖"""
        custom_config = {
            "max_iterations": 20,
            "custom_param": "custom_value"
        }
        
        agent = MockAgent(config=custom_config)
        
        assert agent.config["max_iterations"] == 20
        assert agent.config["custom_param"] == "custom_value"
        # 默认配置应该仍然存在
        assert "timeout" in agent.config
        assert "enable_memory" in agent.config
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """测试并发执行"""
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                self.agent.execute_async(f"输入 {i}", session_id=f"session_{i}")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert "mock_action_completed" in result
        
        # 验证执行历史
        assert len(self.agent.execution_history) == 5
        assert all(metric.success for metric in self.agent.execution_history)
    
    def test_tool_definition_validation(self):
        """测试工具定义验证"""
        # 测试有效的工具定义
        valid_tool = ToolDefinition(
            name="valid_tool",
            description="有效的工具",
            func=lambda x: x
        )
        
        assert valid_tool.name == "valid_tool"
        assert valid_tool.description == "有效的工具"
        assert callable(valid_tool.func)
        assert valid_tool.async_func == False
        assert isinstance(valid_tool.parameters, dict)


if __name__ == "__main__":
    pytest.main([__file__])