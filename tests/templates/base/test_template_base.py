"""
TemplateBase和相关组件的单元测试

测试模板基类、配置数据类、工厂模式等核心功能的正确性。
包括正常流程测试、边界情况测试和错误处理测试。
"""

import pytest
import asyncio
import time
from typing import Any, Dict
from unittest.mock import Mock, patch

from templates.base.template_base import (
    TemplateBase, TemplateConfig, ParameterSchema, TemplateStatus, 
    TemplateType, TemplateFactory, get_template_factory
)
from src.langchain_learning.core.exceptions import ValidationError, ConfigurationError


class MockTemplate(TemplateBase[Dict[str, Any], str]):
    """用于测试的模拟模板类"""
    
    def __init__(self, config=None, should_fail=False, execution_time=0.1):
        super().__init__(config)
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.setup_called = False
        self.setup_params = {}
    
    def setup(self, **parameters):
        """设置模板参数"""
        self.validate_parameters(parameters)
        self.setup_called = True
        self.setup_params = parameters.copy()
        self._setup_parameters = parameters.copy()
        self.status = TemplateStatus.CONFIGURED
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> str:
        """执行模板逻辑"""
        time.sleep(self.execution_time)
        
        if self.should_fail:
            raise RuntimeError("Simulated execution failure")
        
        return f"Processed: {input_data.get('message', 'No message')}"
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "param1": "value1",
                "param2": 42
            },
            "execute_parameters": {
                "message": "Hello, World!"
            },
            "expected_output": "Processed: Hello, World!"
        }


class TestParameterSchema:
    """ParameterSchema的单元测试"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        schema = ParameterSchema(
            name="test_param",
            type=str,
            required=True,
            description="A test parameter"
        )
        
        assert schema.name == "test_param"
        assert schema.type == str
        assert schema.required is True
        assert schema.description == "A test parameter"
        assert schema.default is None
    
    def test_optional_parameter_defaults(self):
        """测试可选参数的默认值设置"""
        schema = ParameterSchema(
            name="optional_param",
            type=int,
            required=False
        )
        
        assert schema.required is False
        assert schema.default == 0  # 自动设置的默认值
    
    def test_different_type_defaults(self):
        """测试不同类型的默认值"""
        test_cases = [
            (str, ""),
            (int, 0),
            (float, 0.0),
            (bool, False),
            (list, []),
            (dict, {})
        ]
        
        for param_type, expected_default in test_cases:
            schema = ParameterSchema(
                name="test",
                type=param_type,
                required=False
            )
            assert schema.default == expected_default


class TestTemplateConfig:
    """TemplateConfig的单元测试"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        config = TemplateConfig(
            name="TestTemplate",
            version="2.0.0",
            description="A test template"
        )
        
        assert config.name == "TestTemplate"
        assert config.version == "2.0.0"
        assert config.description == "A test template"
        assert config.template_type == TemplateType.CUSTOM
        assert len(config.parameters) == 0
    
    def test_parameter_management(self):
        """测试参数管理功能"""
        config = TemplateConfig(name="TestTemplate")
        
        # 添加参数
        config.add_parameter(
            name="input_text",
            param_type=str,
            required=True,
            description="Input text to process"
        )
        
        config.add_parameter(
            name="max_length",
            param_type=int,
            required=False,
            default=100,
            description="Maximum output length"
        )
        
        assert len(config.parameters) == 2
        assert "input_text" in config.required_parameters
        assert "max_length" in config.optional_parameters
        
        # 获取参数模式
        text_schema = config.get_parameter_schema("input_text")
        assert text_schema is not None
        assert text_schema.type == str
        assert text_schema.required is True
        
        length_schema = config.get_parameter_schema("max_length")
        assert length_schema is not None
        assert length_schema.type == int
        assert length_schema.required is False
        assert length_schema.default == 100
    
    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        config = TemplateConfig(
            name="ValidTemplate",
            version="1.0.0"
        )
        assert config.validate_structure() is True
        
        # 无效配置 - 空名称
        with pytest.raises(ValidationError):
            invalid_config = TemplateConfig(name="", version="1.0.0")
            invalid_config.validate_structure()
        
        # 无效配置 - 负超时时间
        with pytest.raises(ValidationError):
            invalid_config = TemplateConfig(
                name="InvalidTemplate",
                timeout=-1
            )
            invalid_config.validate_structure()
        
        # 无效配置 - 负重试次数
        with pytest.raises(ValidationError):
            invalid_config = TemplateConfig(
                name="InvalidTemplate",
                retry_count=-1
            )
            invalid_config.validate_structure()
    
    def test_to_dict_conversion(self):
        """测试字典转换"""
        config = TemplateConfig(
            name="TestTemplate",
            template_type=TemplateType.LLM,
            tags=["test", "example"]
        )
        
        config.add_parameter("param1", str, True, description="Test parameter")
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "TestTemplate"
        assert config_dict["template_type"] == "llm"
        assert config_dict["tags"] == ["test", "example"]
        assert "param1" in config_dict["parameters"]
        assert config_dict["parameters"]["param1"]["type"] == "str"
        assert config_dict["parameters"]["param1"]["required"] is True
    
    def test_from_dict_creation(self):
        """测试从字典创建配置"""
        config_dict = {
            "name": "FromDictTemplate",
            "version": "1.5.0",
            "template_type": "agent",
            "parameters": {
                "input_param": {
                    "type": "str",
                    "required": True,
                    "description": "Input parameter"
                },
                "output_param": {
                    "type": "int",
                    "required": False,
                    "default": 42
                }
            }
        }
        
        config = TemplateConfig.from_dict(config_dict)
        
        assert config.name == "FromDictTemplate"
        assert config.version == "1.5.0"
        assert config.template_type == TemplateType.AGENT
        assert len(config.parameters) == 2
        
        input_schema = config.get_parameter_schema("input_param")
        assert input_schema.type == str
        assert input_schema.required is True
        
        output_schema = config.get_parameter_schema("output_param")
        assert output_schema.type == int
        assert output_schema.required is False
        assert output_schema.default == 42


class TestTemplateBase:
    """TemplateBase的单元测试"""
    
    def test_basic_initialization(self):
        """测试基本初始化"""
        template = MockTemplate()
        
        assert template.status == TemplateStatus.INITIALIZED
        assert template.config.name == "MockTemplate"
        assert template.execution_id is None
        assert len(template.execution_history) == 0
    
    def test_custom_config_initialization(self):
        """测试自定义配置初始化"""
        config = TemplateConfig(
            name="CustomTemplate",
            version="2.0.0",
            description="A custom template"
        )
        
        template = MockTemplate(config)
        
        assert template.config.name == "CustomTemplate"
        assert template.config.version == "2.0.0"
        assert template.config.description == "A custom template"
    
    def test_parameter_validation(self):
        """测试参数验证"""
        config = TemplateConfig(name="TestTemplate")
        config.add_parameter("required_param", str, True)
        config.add_parameter("optional_param", int, False, default=10)
        
        template = MockTemplate(config)
        
        # 有效参数
        valid_params = {
            "required_param": "test_value",
            "optional_param": 20
        }
        assert template.validate_parameters(valid_params) is True
        
        # 缺少必需参数
        with pytest.raises(ValidationError):
            template.validate_parameters({"optional_param": 20})
        
        # 错误的参数类型
        with pytest.raises(ValidationError):
            template.validate_parameters({
                "required_param": "test_value",
                "optional_param": "not_an_int"
            })
    
    def test_setup_and_execution(self):
        """测试设置和执行流程"""
        config = TemplateConfig(name="TestTemplate")
        config.add_parameter("test_param", str, True)
        
        template = MockTemplate(config)
        
        # 设置参数
        template.setup(test_param="test_value")
        assert template.setup_called is True
        assert template.setup_params["test_param"] == "test_value"
        assert template.status == TemplateStatus.CONFIGURED
        
        # 执行模板
        input_data = {"message": "Hello, World!"}
        result = template.run(input_data)
        
        assert result == "Processed: Hello, World!"
        assert template.status == TemplateStatus.COMPLETED
        assert template.execution_id is not None
        assert len(template.execution_history) == 1
    
    def test_execution_failure_handling(self):
        """测试执行失败处理"""
        template = MockTemplate(should_fail=True)
        
        input_data = {"message": "This will fail"}
        
        with pytest.raises(RuntimeError):
            template.run(input_data)
        
        assert template.status == TemplateStatus.FAILED
        assert template.error_message == "Simulated execution failure"
        assert len(template.execution_history) == 1
        assert template.execution_history[0]["success"] is False
    
    def test_metrics_collection(self):
        """测试性能指标收集"""
        template = MockTemplate(execution_time=0.05)  # 50ms
        
        input_data = {"message": "Test metrics"}
        result = template.run(input_data)
        
        metrics = template.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["successful_executions"] == 1
        assert metrics["failed_executions"] == 0
        assert metrics["success_rate"] == 1.0
        assert "avg_execution_time" in metrics
        assert metrics["avg_execution_time"] > 0
    
    def test_multiple_executions(self):
        """测试多次执行"""
        template = MockTemplate()
        
        # 执行多次
        for i in range(5):
            input_data = {"message": f"Message {i}"}
            template.run(input_data)
        
        metrics = template.get_metrics()
        assert metrics["total_executions"] == 5
        assert metrics["successful_executions"] == 5
        assert metrics["success_rate"] == 1.0
        
        # 检查执行历史
        assert len(template.execution_history) == 5
        for i, record in enumerate(template.execution_history):
            assert record["success"] is True
            assert "execution_time" in record
    
    def test_status_information(self):
        """测试状态信息"""
        template = MockTemplate()
        
        status = template.get_status()
        assert status["name"] == "MockTemplate"
        assert status["status"] == "initialized"
        assert status["total_executions"] == 0
        
        # 执行后检查状态
        template.run({"message": "Test"})
        status = template.get_status()
        assert status["status"] == "completed"
        assert status["total_executions"] == 1
        assert status["successful_executions"] == 1
    
    def test_template_reset(self):
        """测试模板重置"""
        template = MockTemplate()
        
        # 执行一次
        template.run({"message": "Test"})
        assert template.status == TemplateStatus.COMPLETED
        assert template.execution_id is not None
        
        # 重置
        template.reset()
        assert template.status == TemplateStatus.INITIALIZED
        assert template.execution_id is None
        assert template.start_time is None
        assert template.end_time is None
        assert template.error_message is None
        assert len(template.metrics) == 0
    
    @pytest.mark.asyncio
    async def test_async_execution(self):
        """测试异步执行"""
        config = TemplateConfig(name="AsyncTemplate", async_enabled=True)
        template = MockTemplate(config, execution_time=0.01)
        
        input_data = {"message": "Async test"}
        result = await template.run_async(input_data)
        
        assert result == "Processed: Async test"
        assert template.status == TemplateStatus.COMPLETED
        assert template.metrics["async"] is True
    
    @pytest.mark.asyncio
    async def test_async_fallback_to_sync(self):
        """测试异步回退到同步执行"""
        # async_enabled=False的情况下，async执行应该回退到同步
        config = TemplateConfig(name="SyncTemplate", async_enabled=False)
        template = MockTemplate(config)
        
        input_data = {"message": "Fallback test"}
        result = await template.run_async(input_data)
        
        assert result == "Processed: Fallback test"
        assert template.status == TemplateStatus.COMPLETED
        # 应该没有async标记
        assert "async" not in template.metrics or template.metrics.get("async") is not True


class TestTemplateFactory:
    """TemplateFactory的单元测试"""
    
    def test_factory_initialization(self):
        """测试工厂初始化"""
        factory = TemplateFactory()
        assert len(factory.get_available_types()) == 0
        assert factory.get_cache_info()["enabled"] is False
    
    def test_template_registration(self):
        """测试模板注册"""
        factory = TemplateFactory()
        
        def mock_constructor(config):
            return MockTemplate(config)
        
        factory.register_template("mock", mock_constructor)
        
        available_types = factory.get_available_types()
        assert "mock" in available_types
        assert len(available_types) == 1
    
    def test_template_creation(self):
        """测试模板创建"""
        factory = TemplateFactory()
        
        def mock_constructor(config):
            return MockTemplate(config)
        
        factory.register_template("mock", mock_constructor)
        
        # 使用默认配置创建
        template = factory.create_template("mock")
        assert isinstance(template, MockTemplate)
        assert template.config.name == "MockTemplate"
        
        # 使用自定义配置创建
        custom_config = TemplateConfig(
            name="CustomMockTemplate",
            description="A custom mock template"
        )
        template = factory.create_template("mock", custom_config)
        assert template.config.name == "CustomMockTemplate"
        assert template.config.description == "A custom mock template"
    
    def test_unknown_template_type(self):
        """测试未知模板类型"""
        factory = TemplateFactory()
        
        with pytest.raises(ValueError, match="Unknown template type"):
            factory.create_template("unknown_type")
    
    def test_template_caching(self):
        """测试模板实例缓存"""
        factory = TemplateFactory()
        factory.enable_cache(True)
        
        def mock_constructor(config):
            return MockTemplate(config)
        
        factory.register_template("mock", mock_constructor)
        
        config = TemplateConfig(name="CachedTemplate")
        
        # 第一次创建
        template1 = factory.create_template("mock", config)
        
        # 第二次创建，应该返回缓存的实例
        template2 = factory.create_template("mock", config)
        
        assert template1 is template2  # 应该是同一个实例
        
        cache_info = factory.get_cache_info()
        assert cache_info["enabled"] is True
        assert cache_info["cached_instances"] == 1
    
    def test_cache_clearing(self):
        """测试缓存清理"""
        factory = TemplateFactory()
        factory.enable_cache(True)
        
        def mock_constructor(config):
            return MockTemplate(config)
        
        factory.register_template("mock", mock_constructor)
        
        # 创建一个实例以填充缓存
        template = factory.create_template("mock")
        assert factory.get_cache_info()["cached_instances"] == 1
        
        # 清理缓存
        factory.clear_cache()
        assert factory.get_cache_info()["cached_instances"] == 0
        
        # 禁用缓存
        factory.enable_cache(False)
        assert factory.get_cache_info()["enabled"] is False
    
    def test_global_factory(self):
        """测试全局工厂实例"""
        # 获取全局工厂
        factory1 = get_template_factory()
        factory2 = get_template_factory()
        
        # 应该是同一个实例
        assert factory1 is factory2
        
        # 注册模板类型到全局工厂
        def mock_constructor(config):
            return MockTemplate(config)
        
        factory1.register_template("global_mock", mock_constructor)
        
        # 通过另一个引用应该能看到注册的类型
        assert "global_mock" in factory2.get_available_types()


if __name__ == "__main__":
    pytest.main([__file__])