#!/usr/bin/env python3
"""
LLM模板使用演示脚本

本脚本演示了如何使用各种LLM模板，包括：
1. OpenAI模板 - GPT系列模型
2. Anthropic模板 - Claude系列模型
3. 本地LLM模板 - Ollama/LlamaCpp等
4. 多模型模板 - 模型对比和切换

运行要求：
- 设置相应的API密钥环境变量
- 安装必要的依赖包
- 对于本地模型，确保Ollama服务运行

使用方法：
    python llm_templates_demo.py [--demo {openai,anthropic,local,multi,all}]
"""

import os
import sys
import asyncio
import argparse
import time
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入LLM模板
from templates.llm import OpenAITemplate, AnthropicTemplate, LocalLLMTemplate, MultiModelTemplate


class LLMTemplateDemo:
    """LLM模板演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.test_prompt = "请用中文简要介绍一下人工智能的三个主要应用领域，每个领域用一句话概括。"
        self.results: Dict[str, Any] = {}
    
    def print_header(self, title: str) -> None:
        """打印标题"""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def print_result(self, model_name: str, result: Any, response_time: float = None) -> None:
        """打印结果"""
        print(f"\n【{model_name}】")
        print("-" * 40)
        
        if hasattr(result, 'content'):
            print(f"响应内容: {result.content}")
            
            if hasattr(result, 'response_time'):
                print(f"响应时间: {result.response_time:.2f}秒")
            elif response_time:
                print(f"响应时间: {response_time:.2f}秒")
                
            if hasattr(result, 'total_tokens'):
                print(f"Token使用: {result.total_tokens}")
            elif hasattr(result, 'tokens_used'):
                print(f"Token使用: {result.tokens_used}")
                
            if hasattr(result, 'estimated_cost'):
                if result.estimated_cost > 0:
                    print(f"预估成本: ${result.estimated_cost:.6f}")
                else:
                    print("成本: 免费（本地模型）")
        else:
            print(f"响应: {result}")
        
        print("-" * 40)
    
    def demo_openai_template(self) -> None:
        """演示OpenAI模板"""
        self.print_header("OpenAI模板演示")
        
        # 检查API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  未设置OPENAI_API_KEY环境变量，跳过OpenAI演示")
            return
        
        try:
            # 创建模板实例
            template = OpenAITemplate()
            
            print("🔧 配置OpenAI模板...")
            template.setup(
                api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=200
            )
            
            print("📝 执行同步调用...")
            start_time = time.time()
            result = template.run(self.test_prompt)
            sync_time = time.time() - start_time
            
            self.print_result("GPT-3.5-turbo (同步)", result)
            self.results["openai_sync"] = result
            
            # 演示流式输出
            print("\n📝 演示流式输出...")
            print("流式响应: ", end="", flush=True)
            
            for chunk in template.stream("用一句话介绍机器学习"):
                print(chunk, end="", flush=True)
                time.sleep(0.05)  # 模拟实时显示
            print("\n")
            
            # 演示异步调用
            print("📝 执行异步调用...")
            
            async def async_demo():
                start_time = time.time()
                result = await template.run_async("简述深度学习的优势")
                async_time = time.time() - start_time
                self.print_result("GPT-3.5-turbo (异步)", result, async_time)
                return result
            
            # 运行异步演示
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async_result = loop.run_until_complete(async_demo())
                self.results["openai_async"] = async_result
            finally:
                loop.close()
            
            # 显示统计信息
            stats = template.get_statistics()
            print(f"\n📊 统计信息:")
            print(f"   总请求数: {stats['total_requests']}")
            print(f"   总Token使用: {stats['total_tokens_used']}")
            print(f"   总成本: ${stats['total_cost']:.6f}")
            print(f"   平均响应时间: {stats.get('average_response_time', 0):.2f}秒")
            
        except Exception as e:
            print(f"❌ OpenAI演示失败: {str(e)}")
    
    def demo_anthropic_template(self) -> None:
        """演示Anthropic模板"""
        self.print_header("Anthropic模板演示")
        
        # 检查API密钥
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  未设置ANTHROPIC_API_KEY环境变量，跳过Anthropic演示")
            return
        
        try:
            # 创建模板实例
            template = AnthropicTemplate()
            
            print("🔧 配置Anthropic模板...")
            template.setup(
                api_key=api_key,
                model_name="claude-3-haiku-20240307",  # 使用较便宜的模型
                max_tokens=200,
                temperature=0.7,
                system_prompt="你是一个专业的AI助手，请提供准确和有用的信息。"
            )
            
            print("📝 执行调用...")
            start_time = time.time()
            result = template.run(self.test_prompt)
            response_time = time.time() - start_time
            
            self.print_result("Claude-3-Haiku", result)
            self.results["anthropic"] = result
            
            # 演示流式输出
            print("\n📝 演示流式输出...")
            print("流式响应: ", end="", flush=True)
            
            for chunk in template.stream("解释什么是自然语言处理"):
                print(chunk, end="", flush=True)
                time.sleep(0.05)
            print("\n")
            
            # 显示统计信息
            stats = template.get_statistics()
            print(f"\n📊 统计信息:")
            print(f"   总请求数: {stats['total_requests']}")
            print(f"   总Token使用: {stats['total_tokens_used']}")
            print(f"   总成本: ${stats['total_cost']:.6f}")
            
        except Exception as e:
            print(f"❌ Anthropic演示失败: {str(e)}")
    
    def demo_local_llm_template(self) -> None:
        """演示本地LLM模板"""
        self.print_header("本地LLM模板演示")
        
        try:
            # 创建模板实例
            template = LocalLLMTemplate()
            
            print("🔧 配置本地LLM模板（Ollama后端）...")
            template.setup(
                backend="ollama",
                model_name="llama2",  # 如果没有此模型会尝试下载
                base_url="http://localhost:11434",
                temperature=0.7,
                max_tokens=200,
                system_prompt="You are a helpful AI assistant. Please respond in Chinese."
            )
            
            # 检查健康状态
            health = template.check_health()
            print(f"健康检查: {health}")
            
            if health.get("status") != "healthy":
                print("⚠️  本地LLM服务不可用，请确保Ollama正在运行且模型已下载")
                print("   启动Ollama: ollama serve")
                print("   下载模型: ollama pull llama2")
                return
            
            print("📝 执行调用...")
            start_time = time.time()
            result = template.run(self.test_prompt)
            response_time = time.time() - start_time
            
            self.print_result("Llama2 (本地)", result)
            self.results["local"] = result
            
            # 演示流式输出
            print("\n📝 演示流式输出...")
            print("流式响应: ", end="", flush=True)
            
            try:
                for chunk in template.stream("What is machine learning?"):
                    print(chunk, end="", flush=True)
                    time.sleep(0.1)
                print("\n")
            except Exception as e:
                print(f"流式输出失败: {str(e)}")
            
            # 显示统计信息
            stats = template.get_statistics()
            print(f"\n📊 统计信息:")
            print(f"   总请求数: {stats['total_requests']}")
            print(f"   总生成时间: {stats['total_generation_time']:.2f}秒")
            print(f"   平均生成速度: {stats.get('average_tokens_per_second', 0):.1f} tokens/秒")
            
        except Exception as e:
            print(f"❌ 本地LLM演示失败: {str(e)}")
            print("   请确保Ollama服务正在运行：ollama serve")
    
    def demo_multi_model_template(self) -> None:
        """演示多模型模板"""
        self.print_header("多模型模板演示")
        
        try:
            # 创建多模型模板
            template = MultiModelTemplate()
            
            print("🔧 配置多模型系统...")
            template.setup(
                routing_strategy="smart",
                fallback_models=[],
                max_parallel_requests=3
            )
            
            # 添加可用的模型
            models_added = 0
            
            # 尝试添加OpenAI模型
            if os.getenv("OPENAI_API_KEY"):
                try:
                    openai_template = OpenAITemplate()
                    template.add_model(
                        name="gpt-3.5",
                        template=openai_template,
                        setup_params={
                            "api_key": os.getenv("OPENAI_API_KEY"),
                            "model_name": "gpt-3.5-turbo",
                            "temperature": 0.7,
                            "max_tokens": 150
                        },
                        priority=2,
                        cost_per_1k_tokens=0.002,
                        tags=["fast", "commercial"]
                    )
                    models_added += 1
                    print("✅ 已添加OpenAI模型")
                except Exception as e:
                    print(f"⚠️  添加OpenAI模型失败: {str(e)}")
            
            # 尝试添加Anthropic模型
            if os.getenv("ANTHROPIC_API_KEY"):
                try:
                    anthropic_template = AnthropicTemplate()
                    template.add_model(
                        name="claude",
                        template=anthropic_template,
                        setup_params={
                            "api_key": os.getenv("ANTHROPIC_API_KEY"),
                            "model_name": "claude-3-haiku-20240307",
                            "max_tokens": 150,
                            "temperature": 0.7
                        },
                        priority=3,
                        cost_per_1k_tokens=0.00125,
                        tags=["quality", "commercial"]
                    )
                    models_added += 1
                    print("✅ 已添加Anthropic模型")
                except Exception as e:
                    print(f"⚠️  添加Anthropic模型失败: {str(e)}")
            
            # 尝试添加本地模型
            try:
                local_template = LocalLLMTemplate()
                # 先检查健康状态
                local_template.setup(
                    backend="ollama",
                    model_name="llama2",
                    base_url="http://localhost:11434"
                )
                health = local_template.check_health()
                
                if health.get("status") == "healthy":
                    template.add_model(
                        name="local-llama",
                        template=local_template,
                        setup_params={
                            "backend": "ollama",
                            "model_name": "llama2",
                            "temperature": 0.7,
                            "max_tokens": 150
                        },
                        priority=1,
                        cost_per_1k_tokens=0.0,  # 免费
                        tags=["free", "local", "private"]
                    )
                    models_added += 1
                    print("✅ 已添加本地模型")
                else:
                    print("⚠️  本地模型不可用")
            except Exception as e:
                print(f"⚠️  添加本地模型失败: {str(e)}")
            
            if models_added == 0:
                print("❌ 没有可用的模型，请设置API密钥或启动本地服务")
                return
            
            print(f"\n📋 已配置 {models_added} 个模型")
            
            # 演示智能路由
            print("\n📝 演示智能路由...")
            result = template.run(self.test_prompt)
            self.print_result(f"智能路由选择: {result.model_name}", result)
            
            # 演示模型对比（如果有多个模型）
            if models_added > 1:
                print("\n📝 演示模型对比...")
                comparison = template.compare_models("什么是机器学习？")
                
                print(f"\n🏆 对比结果:")
                print(f"   最快模型: {comparison.fastest_model}")
                print(f"   最便宜模型: {comparison.cheapest_model}")
                print(f"   最优质量模型: {comparison.best_quality_model}")
                print(f"   总成本: ${comparison.total_cost:.6f}")
                print(f"   对比耗时: {comparison.comparison_time:.2f}秒")
                
                print(f"\n📊 各模型响应:")
                for response in comparison.responses:
                    if response.success:
                        print(f"   {response.model_name}: {response.response_time:.2f}s, "
                              f"{response.tokens_used} tokens, ${response.estimated_cost:.6f}")
                    else:
                        print(f"   {response.model_name}: 失败 - {response.error_message}")
            
            # 演示不同偏好的路由
            if models_added > 1:
                print("\n📝 演示偏好路由...")
                
                # 成本优先
                result_cost = template.run("简述AI", prefer_cost=True)
                print(f"成本优先选择: {result_cost.model_name}")
                
                # 质量优先
                result_quality = template.run("简述AI", prefer_quality=True)
                print(f"质量优先选择: {result_quality.model_name}")
                
                # 速度优先
                result_speed = template.run("简述AI", prefer_speed=True)
                print(f"速度优先选择: {result_speed.model_name}")
            
            # 显示全局统计
            global_stats = template.get_global_statistics()
            print(f"\n📊 全局统计:")
            print(f"   总请求数: {global_stats['total_requests']}")
            print(f"   成功率: {global_stats['success_rate']:.1%}")
            print(f"   总成本: ${global_stats['total_cost']:.6f}")
            print(f"   平均响应时间: {global_stats['average_response_time']:.2f}秒")
            
            # 显示各模型统计
            model_stats = template.get_model_statistics()
            print(f"\n📊 各模型统计:")
            for model_name, stats in model_stats.items():
                print(f"   {model_name}: {stats['requests']}次请求, "
                      f"成功率{stats['success_rate']:.1%}, "
                      f"平均{stats.get('average_response_time', 0):.2f}秒")
                      
        except Exception as e:
            print(f"❌ 多模型演示失败: {str(e)}")
    
    def run_all_demos(self) -> None:
        """运行所有演示"""
        print("🚀 开始LLM模板演示")
        print("本演示将展示OpenAI、Anthropic、本地LLM和多模型模板的功能")
        
        # 运行各个演示
        self.demo_openai_template()
        self.demo_anthropic_template()
        self.demo_local_llm_template()
        self.demo_multi_model_template()
        
        # 总结
        self.print_header("演示总结")
        print("✨ 演示完成！")
        
        if self.results:
            print(f"\n📋 共完成 {len(self.results)} 个模板演示")
            for model_name, result in self.results.items():
                print(f"   ✅ {model_name}: {len(result.content) if hasattr(result, 'content') else 'N/A'} 字符")
        
        print("\n💡 使用建议:")
        print("   - 对于云端API：OpenAI适合快速原型，Anthropic适合复杂推理")
        print("   - 对于本地部署：使用Ollama可以完全控制数据隐私")
        print("   - 对于生产环境：使用多模型模板实现负载均衡和故障转移")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM模板演示脚本")
    parser.add_argument(
        "--demo",
        choices=["openai", "anthropic", "local", "multi", "all"],
        default="all",
        help="选择要运行的演示"
    )
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = LLMTemplateDemo()
    
    # 根据参数运行相应的演示
    if args.demo == "openai":
        demo.demo_openai_template()
    elif args.demo == "anthropic":
        demo.demo_anthropic_template()
    elif args.demo == "local":
        demo.demo_local_llm_template()
    elif args.demo == "multi":
        demo.demo_multi_model_template()
    else:
        demo.run_all_demos()


if __name__ == "__main__":
    main()