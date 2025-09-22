"""
工具管理模块的单元测试

测试ToolManager、BaseTool、FunctionTool和工具注册表功能。
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_learning.core.tools import (
    ToolManager,
    BaseTool,
    FunctionTool,
    ToolType,
    ToolStatus,
    ToolRegistry,
    ToolInput,
    ToolOutput,
    ToolSchema,
    ToolMetadata,
    get_tool_manager,
    register_tool,
    register_function,
    call_tool
)
from langchain_learning.core.exceptions import ToolError, ValidationError


class MockTool(BaseTool):
    """用于测试的模拟工具"""
    
    async def _execute(self, inputs: ToolInput) -> ToolOutput:
        """模拟工具执行"""
        if inputs.parameters.get('fail'):
            raise Exception("Simulated failure")
        
        result = inputs.parameters.get('input', 'default') + '_processed'
        return ToolOutput(
            result=result,
            success=True,
            metadata={"mock": True}
        )


class TestToolSchema:
    """测试工具模式"""
    
    def test_basic_schema(self):
        """测试基本模式创建"""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            parameters={
                "input": {"type": "str", "description": "Input text"},
                "count": {"type": "int", "description": "Count value"}
            },
            required_parameters=["input"]
        )
        
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert "input" in schema.parameters
        assert "input" in schema.required_parameters
    
    def test_parameter_validation(self):
        """测试参数验证"""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            parameters={
                "required_param": {"type": "str"},
                "optional_param": {"type": "int"}
            },
            required_parameters=["required_param"]
        )
        
        # 测试有效参数
        schema.validate_parameters({
            "required_param": "test",
            "optional_param": 123
        })
        
        # 测试缺少必需参数
        with pytest.raises(ValidationError) as exc_info:
            schema.validate_parameters({"optional_param": 123})
        
        assert "required_param" in str(exc_info.value)
        
        # 测试类型不匹配
        with pytest.raises(ValidationError) as exc_info:
            schema.validate_parameters({
                "required_param": 123  # 应该是字符串
            })
        
        assert "type mismatch" in str(exc_info.value)


class TestToolMetadata:
    """测试工具元数据"""
    
    def test_basic_metadata(self):
        """测试基本元数据"""
        metadata = ToolMetadata(
            name="test_tool",
            tool_type=ToolType.FUNCTION,
            status=ToolStatus.ACTIVE,
            version="1.0.0",
            author="test_author",
            description="Test tool description"
        )
        
        assert metadata.name == "test_tool"
        assert metadata.tool_type == ToolType.FUNCTION
        assert metadata.status == ToolStatus.ACTIVE
        assert metadata.success_rate == 0.0  # 初始为0
    
    def test_success_rate_calculation(self):
        """测试成功率计算"""
        metadata = ToolMetadata(
            name="test_tool",
            tool_type=ToolType.FUNCTION
        )
        
        # 初始成功率为0
        assert metadata.success_rate == 0.0
        
        # 模拟执行统计
        metadata.usage_count = 10
        metadata.success_count = 8
        metadata.failure_count = 2
        
        assert metadata.success_rate == 0.8


class TestBaseTool:
    """测试基础工具类"""
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """测试工具执行"""
        tool = MockTool(
            name="mock_tool",
            description="A mock tool for testing"
        )
        
        # 测试成功执行
        result = await tool.run(input="test")
        assert result.success is True
        assert result.result == "test_processed"
        assert result.execution_time > 0
        
        # 验证统计信息更新
        assert tool.metadata.usage_count == 1
        assert tool.metadata.success_count == 1
        assert tool.metadata.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_tool_failure(self):
        """测试工具执行失败"""
        tool = MockTool(
            name="failing_tool",
            description="A tool that fails"
        )
        
        # 测试执行失败
        result = await tool.run(fail=True)
        assert result.success is False
        assert "Simulated failure" in result.error
        
        # 验证统计信息更新
        assert tool.metadata.usage_count == 1
        assert tool.metadata.success_count == 0
        assert tool.metadata.failure_count == 1
    
    def test_get_info(self):
        """测试获取工具信息"""
        tool = MockTool(
            name="info_tool",
            description="Tool for info testing",
            metadata=ToolMetadata(
                name="info_tool",
                tool_type=ToolType.FUNCTION,
                tags=["test", "mock"],
                dependencies=["dependency1"]
            )
        )
        
        info = tool.get_info()
        assert info["name"] == "info_tool"
        assert info["description"] == "Tool for info testing"
        assert info["type"] == ToolType.FUNCTION.value
        assert "test" in info["tags"]
        assert "dependency1" in info["dependencies"]
        assert "metrics" in info


class TestFunctionTool:
    """测试函数工具"""
    
    def test_sync_function_wrapping(self):
        """测试同步函数包装"""
        def add_numbers(a: int, b: int = 10) -> int:
            """Add two numbers together"""
            return a + b
        
        tool = FunctionTool(add_numbers, name="adder", description="Adds numbers")
        
        assert tool.name == "adder"
        assert tool.description == "Adds numbers"
        assert tool.metadata.tool_type == ToolType.FUNCTION
        
        # 检查自动生成的模式
        schema = tool.schema
        assert "a" in schema.parameters
        assert "b" in schema.parameters
        assert "a" in schema.required_parameters
        assert "b" not in schema.required_parameters  # 有默认值
    
    @pytest.mark.asyncio
    async def test_sync_function_execution(self):
        """测试同步函数执行"""
        def multiply(x: int, y: int) -> int:
            return x * y
        
        tool = FunctionTool(multiply)
        result = await tool.run(x=5, y=3)
        
        assert result.success is True
        assert result.result == 15
    
    @pytest.mark.asyncio
    async def test_async_function_wrapping(self):
        """测试异步函数包装"""
        async def async_process(data: str) -> str:
            """Process data asynchronously"""
            await asyncio.sleep(0.01)  # 模拟异步操作
            return f"processed_{data}"
        
        tool = FunctionTool(async_process)
        result = await tool.run(data="test")
        
        assert result.success is True
        assert result.result == "processed_test"
    
    @pytest.mark.asyncio
    async def test_function_with_exception(self):
        """测试带异常的函数"""
        def failing_function(value: str) -> str:
            if value == "error":
                raise ValueError("Invalid value")
            return f"valid_{value}"
        
        tool = FunctionTool(failing_function)
        
        # 测试正常执行
        result = await tool.run(value="test")
        assert result.success is True
        assert result.result == "valid_test"
        
        # 测试异常情况
        result = await tool.run(value="error")
        assert result.success is False
        assert "Invalid value" in result.error


class TestToolRegistry:
    """测试工具注册表"""
    
    def test_tool_registration(self):
        """测试工具注册"""
        registry = ToolRegistry()
        tool = MockTool(
            name="registry_test_tool",
            description="Tool for registry testing",
            metadata=ToolMetadata(
                name="registry_test_tool",
                tool_type=ToolType.FUNCTION,
                tags=["test", "registry"]
            )
        )
        
        # 注册工具
        registry.register(tool)
        
        # 验证注册成功
        assert "registry_test_tool" in registry.list_tools()
        retrieved_tool = registry.get_tool("registry_test_tool")
        assert retrieved_tool is tool
    
    def test_duplicate_registration(self):
        """测试重复注册"""
        registry = ToolRegistry()
        tool1 = MockTool(name="duplicate_tool", description="First tool")
        tool2 = MockTool(name="duplicate_tool", description="Second tool")
        
        registry.register(tool1)
        
        # 重复注册应该抛出异常
        with pytest.raises(ToolError) as exc_info:
            registry.register(tool2)
        
        assert "already registered" in str(exc_info.value)
    
    def test_function_registration(self):
        """测试函数注册"""
        registry = ToolRegistry()
        
        def test_function(x: int) -> int:
            return x * 2
        
        tool = registry.register_function(test_function, description="Test function")
        
        assert tool.name == "test_function"
        assert tool.description == "Test function"
        assert "test_function" in registry.list_tools()
    
    def test_search_by_tag(self):
        """测试按标签搜索"""
        registry = ToolRegistry()
        
        tool1 = MockTool(
            name="tool1",
            metadata=ToolMetadata(name="tool1", tool_type=ToolType.FUNCTION, tags=["math", "basic"])
        )
        tool2 = MockTool(
            name="tool2", 
            metadata=ToolMetadata(name="tool2", tool_type=ToolType.FUNCTION, tags=["text", "processing"])
        )
        tool3 = MockTool(
            name="tool3",
            metadata=ToolMetadata(name="tool3", tool_type=ToolType.FUNCTION, tags=["math", "advanced"])
        )
        
        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)
        
        # 搜索数学相关工具
        math_tools = registry.search_by_tag("math")
        assert len(math_tools) == 2
        assert tool1 in math_tools
        assert tool3 in math_tools
        
        # 搜索文本处理工具
        text_tools = registry.search_by_tag("text")
        assert len(text_tools) == 1
        assert tool2 in text_tools
    
    def test_search_by_type(self):
        """测试按类型搜索"""
        registry = ToolRegistry()
        
        function_tool = MockTool(
            name="function_tool",
            metadata=ToolMetadata(name="function_tool", tool_type=ToolType.FUNCTION)
        )
        api_tool = MockTool(
            name="api_tool",
            metadata=ToolMetadata(name="api_tool", tool_type=ToolType.API)
        )
        
        registry.register(function_tool)
        registry.register(api_tool)
        
        function_tools = registry.search_by_type(ToolType.FUNCTION)
        assert len(function_tools) == 1
        assert function_tool in function_tools
        
        api_tools = registry.search_by_type(ToolType.API)
        assert len(api_tools) == 1
        assert api_tool in api_tools
    
    def test_search_by_name(self):
        """测试按名称搜索"""
        registry = ToolRegistry()
        
        tools = [
            MockTool(name="math_add"),
            MockTool(name="math_subtract"),
            MockTool(name="text_clean"),
            MockTool(name="text_format")
        ]
        
        for tool in tools:
            registry.register(tool)
        
        # 搜索math开头的工具
        math_tools = registry.search_by_name("math_*")
        assert len(math_tools) == 2
        
        # 搜索包含text的工具
        text_tools = registry.search_by_name("*text*")
        assert len(text_tools) == 2
    
    def test_statistics(self):
        """测试统计信息"""
        registry = ToolRegistry()
        
        tools = [
            MockTool(name="tool1", metadata=ToolMetadata(name="tool1", tool_type=ToolType.FUNCTION)),
            MockTool(name="tool2", metadata=ToolMetadata(name="tool2", tool_type=ToolType.API)),
            MockTool(name="tool3", metadata=ToolMetadata(name="tool3", tool_type=ToolType.FUNCTION))
        ]
        
        for tool in tools:
            registry.register(tool)
        
        stats = registry.get_statistics()
        
        assert stats["total_tools"] == 3
        assert stats["type_distribution"][ToolType.FUNCTION.value] == 2
        assert stats["type_distribution"][ToolType.API.value] == 1


class TestToolManager:
    """测试工具管理器"""
    
    def test_tool_manager_initialization(self):
        """测试工具管理器初始化"""
        manager = ToolManager()
        
        # 验证内置工具已注册
        tools = manager.list_tools()
        tool_names = [tool["name"] for tool in tools]
        
        assert "string_length" in tool_names
        assert "math_add" in tool_names
        assert "read_file" in tool_names
    
    @pytest.mark.asyncio
    async def test_tool_call(self):
        """测试工具调用"""
        manager = ToolManager()
        
        # 测试内置工具调用
        result = await manager.call_tool("string_length", text="hello world")
        assert result.success is True
        assert result.result == 11
        
        result = await manager.call_tool("math_add", a=5, b=3)
        assert result.success is True
        assert result.result == 8
    
    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """测试工具不存在的情况"""
        manager = ToolManager()
        
        with pytest.raises(ToolError) as exc_info:
            await manager.call_tool("nonexistent_tool")
        
        assert "not found" in str(exc_info.value)
    
    def test_register_function_decorator(self):
        """测试函数注册装饰器"""
        manager = ToolManager()
        
        @manager.register_function
        def custom_tool(x: int, y: int = 1) -> int:
            """A custom tool for testing"""
            return x + y
        
        # 验证工具已注册
        tools = manager.list_tools()
        tool_names = [tool["name"] for tool in tools]
        assert "custom_tool" in tool_names
        
        # 验证工具可以调用
        import asyncio
        result = asyncio.run(manager.call_tool("custom_tool", x=10, y=5))
        assert result.success is True
        assert result.result == 15
    
    def test_register_function_with_params(self):
        """测试带参数的函数注册装饰器"""
        manager = ToolManager()
        
        @manager.register_function(name="special_adder", description="Special addition tool")
        def add_special(a: float, b: float) -> float:
            return a + b + 0.1
        
        # 获取工具模式
        schema = manager.get_tool_schema("special_adder")
        assert schema is not None
        assert schema["name"] == "special_adder"
        assert schema["description"] == "Special addition tool"
    
    def test_search_tools(self):
        """测试工具搜索"""
        manager = ToolManager()
        
        # 按名称查询搜索
        math_tools = manager.search_tools(query="math")
        assert len(math_tools) > 0
        
        # 按类型搜索
        function_tools = manager.search_tools(tool_type=ToolType.FUNCTION)
        assert len(function_tools) > 0
    
    def test_middleware(self):
        """测试中间件功能"""
        manager = ToolManager()
        
        # 添加日志中间件
        call_log = []
        
        async def logging_middleware(tool, kwargs):
            call_log.append(f"Called {tool.name} with {kwargs}")
            return kwargs
        
        manager.add_middleware(logging_middleware)
        
        # 调用工具
        import asyncio
        asyncio.run(manager.call_tool("string_length", text="test"))
        
        # 验证中间件被调用
        assert len(call_log) == 1
        assert "string_length" in call_log[0]


class TestGlobalToolFunctions:
    """测试全局工具函数"""
    
    def test_get_global_manager(self):
        """测试获取全局管理器"""
        manager1 = get_tool_manager()
        manager2 = get_tool_manager()
        
        # 应该返回同一个实例
        assert manager1 is manager2
    
    def test_global_register_function(self):
        """测试全局函数注册"""
        @register_function(name="global_test_tool")
        def global_tool(value: str) -> str:
            return f"global_{value}"
        
        # 验证工具已注册到全局管理器
        manager = get_tool_manager()
        tools = manager.list_tools()
        tool_names = [tool["name"] for tool in tools]
        assert "global_test_tool" in tool_names
    
    @pytest.mark.asyncio
    async def test_global_call_tool(self):
        """测试全局工具调用"""
        result = await call_tool("string_length", text="global test")
        assert result.success is True
        assert result.result == 11