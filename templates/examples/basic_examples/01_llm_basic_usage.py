#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM模板基础使用示例

本示例演示如何使用LLM模板进行基本的文本生成任务，包括：
1. 基础LLM调用
2. 参数配置和调优
3. 错误处理
4. 性能监控

作者: LangChain Learning Project
版本: 1.0.0
"""

import os
import sys
import asyncio
from typing import Dict, Any

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from templates.llm.openai_template import OpenAITemplate
from templates.base.template_base import TemplateConfig, ParameterSchema


def basic_llm_usage():
    """基础LLM使用示例"""
    print("=== 基础LLM使用示例 ===")
    
    # 1. 创建LLM模板实例
    llm_template = OpenAITemplate()
    
    # 2. 配置模板参数
    llm_template.setup(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000,
        timeout=30.0
    )
    
    # 3. 执行文本生成
    prompt = "请介绍一下LangChain框架的主要特点和应用场景。"
    
    try:
        result = llm_template.run(prompt)
        print(f"生成结果:\n{result.content}")
        
        # 4. 查看执行状态
        status = llm_template.get_status()
        print(f"\n执行状态: {status['status']}")
        print(f"执行时间: {status['execution_time']:.2f}秒")
        
    except Exception as e:
        print(f"执行失败: {str(e)}")


def parameter_tuning_example():
    """参数调优示例"""
    print("\n=== 参数调优示例 ===")
    
    # 测试不同温度参数的效果
    temperatures = [0.1, 0.7, 1.2]
    prompt = "创作一首关于春天的诗歌。"
    
    for temp in temperatures:
        print(f"\n--- 温度参数: {temp} ---")
        
        llm_template = OpenAITemplate()
        llm_template.setup(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=temp,
            max_tokens=500
        )
        
        try:
            result = llm_template.run(prompt)
            print(f"生成结果:\n{result.content[:200]}...")  # 只显示前200字符
            
        except Exception as e:
            print(f"执行失败: {str(e)}")


def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 1. API密钥错误处理
    print("1. 测试无效API密钥:")
    llm_template = OpenAITemplate()
    
    try:
        llm_template.setup(
            api_key="invalid-api-key",
            model_name="gpt-3.5-turbo"
        )
        result = llm_template.run("Hello")
        
    except Exception as e:
        print(f"预期的认证错误: {type(e).__name__}")
    
    # 2. 参数验证错误
    print("\n2. 测试无效参数:")
    try:
        llm_template.setup(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="invalid-model",
            temperature=5.0  # 超出有效范围
        )
        
    except Exception as e:
        print(f"预期的参数错误: {type(e).__name__}")
    
    # 3. 网络超时处理
    print("\n3. 测试超时处理:")
    llm_template = OpenAITemplate()
    llm_template.setup(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        timeout=0.1  # 设置很短的超时时间
    )
    
    try:
        result = llm_template.run("生成一篇长文章")
        
    except Exception as e:
        print(f"预期的超时错误: {type(e).__name__}")


def performance_monitoring_example():
    """性能监控示例"""
    print("\n=== 性能监控示例 ===")
    
    llm_template = OpenAITemplate()
    llm_template.setup(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        enable_metrics=True  # 启用性能指标收集
    )
    
    # 执行多次调用以收集性能数据
    prompts = [
        "解释什么是人工智能。",
        "描述机器学习的基本概念。",
        "介绍深度学习的应用领域。",
        "分析自然语言处理的发展趋势。"
    ]
    
    print("执行多次调用...")
    for i, prompt in enumerate(prompts, 1):
        try:
            result = llm_template.run(prompt)
            print(f"调用 {i} 完成")
            
        except Exception as e:
            print(f"调用 {i} 失败: {str(e)}")
    
    # 查看性能指标
    metrics = llm_template.get_metrics()
    print(f"\n=== 性能统计 ===")
    print(f"总执行次数: {metrics['total_executions']}")
    print(f"成功次数: {metrics['successful_executions']}")
    print(f"失败次数: {metrics['failed_executions']}")
    print(f"成功率: {metrics['success_rate']:.2%}")
    
    if 'avg_execution_time' in metrics:
        print(f"平均执行时间: {metrics['avg_execution_time']:.2f}秒")
        print(f"最快执行时间: {metrics['min_execution_time']:.2f}秒")
        print(f"最慢执行时间: {metrics['max_execution_time']:.2f}秒")


async def async_execution_example():
    """异步执行示例"""
    print("\n=== 异步执行示例 ===")
    
    # 创建支持异步的LLM模板
    llm_template = OpenAITemplate()
    llm_template.setup(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        async_enabled=True
    )
    
    # 准备多个并发任务
    prompts = [
        "介绍Python编程语言的特点。",
        "解释JavaScript的主要用途。",
        "描述Java语言的优势。"
    ]
    
    print("开始并发执行...")
    start_time = asyncio.get_event_loop().time()
    
    # 创建异步任务
    tasks = [llm_template.run_async(prompt) for prompt in prompts]
    
    try:
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        print(f"并发执行完成，总耗时: {total_time:.2f}秒")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- 结果 {i} ---")
            print(f"{result.content[:100]}...")  # 显示前100字符
            
    except Exception as e:
        print(f"异步执行失败: {str(e)}")


def template_configuration_example():
    """模板配置示例"""
    print("\n=== 模板配置示例 ===")
    
    # 1. 使用自定义配置创建模板
    custom_config = TemplateConfig(
        name="CustomLLMTemplate",
        description="自定义LLM模板配置",
        version="1.0.0"
    )
    
    # 添加自定义参数
    custom_config.add_parameter(
        name="custom_system_prompt",
        param_type=str,
        required=False,
        default="你是一个专业的AI助手",
        description="自定义系统提示词"
    )
    
    # 使用自定义配置创建模板
    llm_template = OpenAITemplate(config=custom_config)
    
    # 2. 查看模板配置信息
    print("模板配置信息:")
    print(f"名称: {llm_template.config.name}")
    print(f"描述: {llm_template.config.description}")
    print(f"版本: {llm_template.config.version}")
    
    # 3. 查看参数定义
    print("\n参数定义:")
    for param_name, param_schema in llm_template.config.parameters.items():
        print(f"- {param_name}: {param_schema.type.__name__} "
              f"(必需: {param_schema.required}, "
              f"默认值: {param_schema.default})")
    
    # 4. 获取使用示例
    example = llm_template.get_example()
    print(f"\n使用示例:")
    print(f"设置参数: {example['setup_parameters']}")
    print(f"执行参数: {example['execute_parameters']}")


def main():
    """主函数"""
    print("LangChain Learning - LLM模板基础使用示例")
    print("=" * 50)
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY环境变量")
        print("请先设置API密钥:")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # 运行基础示例
        basic_llm_usage()
        
        # 运行参数调优示例
        parameter_tuning_example()
        
        # 运行错误处理示例
        error_handling_example()
        
        # 运行性能监控示例
        performance_monitoring_example()
        
        # 运行模板配置示例
        template_configuration_example()
        
        # 运行异步执行示例
        print("\n开始异步执行示例...")
        asyncio.run(async_execution_example())
        
        print("\n所有示例执行完成！")
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
        
    except Exception as e:
        print(f"示例执行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()