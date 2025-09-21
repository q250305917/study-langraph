#!/usr/bin/env python3
"""
模板基础框架演示脚本

演示如何使用模板系统的核心组件，包括：
- 创建和配置模板
- 参数验证
- 配置加载
- 模板执行和监控

这个脚本展示了Stream A基础框架的完整功能。
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from templates.base.template_base import (
    TemplateBase, TemplateConfig, ParameterSchema, TemplateType,
    TemplateFactory, get_template_factory
)
from templates.base.parameter_validator import (
    ParameterValidator, ValidationLevel,
    create_email_validator, create_positive_number_validator
)
from templates.base.config_loader import ConfigLoader, ConfigSource, ConfigSourceType


class DemoTemplate(TemplateBase[Dict[str, Any], str]):
    """
    演示模板类
    
    一个简单的文本处理模板，展示模板系统的基本功能。
    """
    
    def __init__(self, config=None):
        # 如果没有提供配置，创建默认配置
        if config is None:
            config = self._create_demo_config()
        super().__init__(config)
        self.processing_settings = {}
    
    def _create_demo_config(self) -> TemplateConfig:
        """创建演示配置"""
        config = TemplateConfig(
            name="DemoTemplate",
            version="1.0.0",
            description="一个演示模板系统功能的示例模板",
            template_type=TemplateType.CUSTOM,
            author="LangChain Learning Project"
        )
        
        # 添加参数定义
        config.add_parameter(
            name="input_text",
            param_type=str,
            required=True,
            description="要处理的输入文本",
            constraints={"min_length": 1, "max_length": 1000}
        )
        
        config.add_parameter(
            name="operation",
            param_type=str,
            required=True,
            description="要执行的操作",
            constraints={"allowed_values": ["upper", "lower", "title", "reverse"]}
        )
        
        config.add_parameter(
            name="repeat_count",
            param_type=int,
            required=False,
            default=1,
            description="重复次数",
            constraints={"min_value": 1, "max_value": 10}
        )
        
        config.add_parameter(
            name="add_prefix",
            param_type=bool,
            required=False,
            default=False,
            description="是否添加前缀"
        )
        
        return config
    
    def setup(self, **parameters):
        """设置模板参数"""
        # 验证参数
        self.validate_parameters(parameters)
        
        # 保存设置
        self.processing_settings = parameters.copy()
        self._setup_parameters = parameters.copy()
        
        print(f"✅ 模板设置完成: {parameters}")
        
    def execute(self, input_data: Dict[str, Any], **kwargs) -> str:
        """执行文本处理"""
        if not self.processing_settings:
            raise RuntimeError("请先调用setup()方法设置模板参数")
        
        # 获取输入文本
        text = input_data.get("text", "")
        if not text:
            raise ValueError("输入数据必须包含'text'字段")
        
        # 执行操作
        operation = self.processing_settings["operation"]
        repeat_count = self.processing_settings.get("repeat_count", 1)
        add_prefix = self.processing_settings.get("add_prefix", False)
        
        # 文本处理
        if operation == "upper":
            result = text.upper()
        elif operation == "lower":
            result = text.lower()
        elif operation == "title":
            result = text.title()
        elif operation == "reverse":
            result = text[::-1]
        else:
            result = text
        
        # 重复处理
        if repeat_count > 1:
            result = " ".join([result] * repeat_count)
        
        # 添加前缀
        if add_prefix:
            result = f"[处理结果] {result}"
        
        return result
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "input_text": "Hello, World!",
                "operation": "upper",
                "repeat_count": 2,
                "add_prefix": True
            },
            "execute_parameters": {
                "text": "Hello, World!"
            },
            "expected_output": "[处理结果] HELLO, WORLD! HELLO, WORLD!"
        }


def demonstrate_parameter_validation():
    """演示参数验证功能"""
    print("\n🔍 === 参数验证演示 ===")
    
    # 创建参数验证器
    validator = ParameterValidator(ValidationLevel.STRICT)
    
    # 添加各种验证规则
    validator.add_type_validator("name", str)
    validator.add_length_validator("name", min_length=2, max_length=50)
    validator.add_pattern_validator("email", r'^[^@]+@[^@]+\.[^@]+$')
    validator.add_range_validator("age", min_value=0, max_value=150)
    validator.add_allowed_values_validator("status", ["active", "inactive", "pending"])
    
    # 测试有效数据
    valid_data = {
        "name": "张三",
        "email": "zhangsan@example.com",
        "age": 25,
        "status": "active"
    }
    
    print(f"验证有效数据: {valid_data}")
    result = validator.validate(valid_data)
    print(f"验证结果: {result}")
    
    # 测试无效数据
    invalid_data = {
        "name": "X",  # 太短
        "email": "invalid-email",  # 格式错误
        "age": -5,  # 负数
        "status": "unknown"  # 不在允许值中
    }
    
    print(f"\n验证无效数据: {invalid_data}")
    result = validator.validate(invalid_data)
    print(f"验证结果: {result}")
    if result.errors:
        print("错误详情:")
        for error in result.errors:
            print(f"  - {error}")


def demonstrate_config_loader():
    """演示配置加载功能"""
    print("\n⚙️ === 配置加载演示 ===")
    
    # 创建配置加载器
    loader = ConfigLoader(cache_enabled=False)
    
    # 添加字典配置源
    base_config = {
        "name": "ConfigDemo",
        "version": "1.0.0",
        "template_type": "custom",
        "parameters": {
            "text_param": {
                "type": "str",
                "required": True,
                "description": "文本参数"
            }
        }
    }
    
    loader.add_dict_source(base_config, priority=10)
    
    # 模拟环境变量覆盖
    os.environ["TEMPLATE_VERSION"] = "2.0.0"
    os.environ["TEMPLATE_DEBUG"] = "true"
    
    try:
        loader.add_env_source("TEMPLATE_", priority=20)
        
        # 加载配置
        config = loader.load_config()
        
        print(f"加载的配置:")
        print(f"  名称: {config.name}")
        print(f"  版本: {config.version}")  # 应该被环境变量覆盖为2.0.0
        print(f"  类型: {config.template_type}")
        print(f"  参数数量: {len(config.parameters)}")
        
    finally:
        # 清理环境变量
        if "TEMPLATE_VERSION" in os.environ:
            del os.environ["TEMPLATE_VERSION"]
        if "TEMPLATE_DEBUG" in os.environ:
            del os.environ["TEMPLATE_DEBUG"]


def demonstrate_template_usage():
    """演示模板使用"""
    print("\n🚀 === 模板使用演示 ===")
    
    # 创建模板实例
    template = DemoTemplate()
    
    print(f"模板状态: {template.get_status()}")
    
    # 设置模板参数
    setup_params = {
        "input_text": "Hello, World!",
        "operation": "upper",
        "repeat_count": 2,
        "add_prefix": True
    }
    
    template.setup(**setup_params)
    
    # 执行模板
    input_data = {"text": "Hello, World!"}
    
    print(f"执行输入: {input_data}")
    result = template.run(input_data)
    print(f"执行结果: {result}")
    
    # 查看执行状态和指标
    status = template.get_status()
    print(f"\n执行后状态:")
    print(f"  状态: {status['status']}")
    print(f"  执行时间: {status['execution_time']:.3f}秒")
    print(f"  成功执行次数: {status['successful_executions']}")
    
    metrics = template.get_metrics()
    print(f"\n性能指标:")
    print(f"  总执行次数: {metrics['total_executions']}")
    print(f"  成功率: {metrics['success_rate']:.2%}")
    print(f"  平均执行时间: {metrics.get('avg_execution_time', 0):.3f}秒")


async def demonstrate_async_template():
    """演示异步模板执行"""
    print("\n⚡ === 异步执行演示 ===")
    
    # 创建支持异步的配置
    config = TemplateConfig(
        name="AsyncDemoTemplate",
        async_enabled=True
    )
    
    template = DemoTemplate(config)
    
    # 设置参数
    template.setup(
        input_text="Async Test",
        operation="title",
        repeat_count=1,
        add_prefix=True
    )
    
    # 异步执行
    input_data = {"text": "async execution test"}
    
    print(f"异步执行输入: {input_data}")
    result = await template.run_async(input_data)
    print(f"异步执行结果: {result}")
    
    # 检查异步标记
    metrics = template.get_metrics()
    print(f"异步执行指标: {metrics}")


def demonstrate_template_factory():
    """演示模板工厂"""
    print("\n🏭 === 模板工厂演示 ===")
    
    # 获取全局工厂
    factory = get_template_factory()
    
    # 注册模板类型
    def create_demo_template(config):
        return DemoTemplate(config)
    
    factory.register_template("demo", create_demo_template)
    
    print(f"可用模板类型: {factory.get_available_types()}")
    
    # 创建模板实例
    template = factory.create_template("demo")
    print(f"创建的模板: {template}")
    
    # 使用自定义配置创建
    custom_config = TemplateConfig(
        name="FactoryCreatedTemplate",
        description="通过工厂创建的模板"
    )
    
    custom_template = factory.create_template("demo", custom_config)
    print(f"自定义配置模板: {custom_template.config.name}")


def demonstrate_error_handling():
    """演示错误处理"""
    print("\n❌ === 错误处理演示 ===")
    
    template = DemoTemplate()
    
    try:
        # 尝试在没有设置参数的情况下执行
        template.run({"text": "test"})
    except Exception as e:
        print(f"捕获到预期错误: {e}")
    
    # 设置无效参数
    try:
        template.setup(
            input_text="test",
            operation="invalid_operation",  # 无效操作
            repeat_count=15  # 超出范围
        )
    except Exception as e:
        print(f"参数验证错误: {e}")
    
    # 设置有效参数但提供无效输入
    template.setup(
        input_text="test",
        operation="upper",
        repeat_count=1
    )
    
    try:
        # 缺少必需的输入字段
        template.run({"wrong_field": "data"})
    except Exception as e:
        print(f"输入验证错误: {e}")


async def main():
    """主演示函数"""
    print("🎯 模板基础框架演示")
    print("=" * 50)
    
    # 演示各个组件
    demonstrate_parameter_validation()
    demonstrate_config_loader()
    demonstrate_template_usage()
    await demonstrate_async_template()
    demonstrate_template_factory()
    demonstrate_error_handling()
    
    print("\n✨ === 演示完成 ===")
    print("Stream A基础框架提供了强大而灵活的模板系统基础。")
    print("其他Stream可以基于这些组件开发具体的LangChain模板。")


if __name__ == "__main__":
    asyncio.run(main())