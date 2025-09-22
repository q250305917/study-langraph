#!/usr/bin/env python3
"""
LangChain学习辅助工具类

提供学习过程中常用的辅助功能，包括代码示例、调试工具、性能监控等。
"""

import time
import json
import logging
import functools
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import inspect
import sys
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeExecutionTimer:
    """代码执行时间计时器"""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.execution_time = None
    
    def __enter__(self):
        """进入上下文管理器"""
        self.start_time = time.time()
        if self.description:
            print(f"⏱️ 开始执行: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        
        if exc_type is None:
            print(f"✅ 执行完成: {self.description if self.description else '代码块'} "
                  f"耗时 {self.execution_time:.3f} 秒")
        else:
            print(f"❌ 执行失败: {self.description if self.description else '代码块'} "
                  f"耗时 {self.execution_time:.3f} 秒")
            print(f"错误类型: {exc_type.__name__}")
            print(f"错误信息: {exc_val}")
    
    def get_execution_time(self) -> float:
        """获取执行时间"""
        return self.execution_time if self.execution_time else 0.0

def time_execution(description: str = ""):
    """装饰器：自动计时函数执行时间"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_desc = description or f"函数 {func.__name__}"
            with CodeExecutionTimer(func_desc):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class LangChainDebugger:
    """LangChain调试辅助工具"""
    
    @staticmethod
    def inspect_chain_components(chain, verbose: bool = True):
        """检查Chain的组成部分"""
        print("🔍 Chain组件检查")
        print("=" * 40)
        
        # 检查Chain类型
        chain_type = type(chain).__name__
        print(f"Chain类型: {chain_type}")
        
        # 检查是否有input_variables
        if hasattr(chain, 'input_variables'):
            print(f"输入变量: {chain.input_variables}")
        
        # 检查是否有output_variables
        if hasattr(chain, 'output_variables'):
            print(f"输出变量: {chain.output_variables}")
        
        # 检查是否有memory
        if hasattr(chain, 'memory') and chain.memory:
            print(f"记忆系统: {type(chain.memory).__name__}")
        
        # 检查是否有verbose设置
        if hasattr(chain, 'verbose'):
            print(f"详细模式: {chain.verbose}")
        
        if verbose:
            # 尝试获取更多详细信息
            try:
                chain_dict = chain.dict() if hasattr(chain, 'dict') else {}
                if chain_dict:
                    print(f"\\n详细配置:")
                    for key, value in chain_dict.items():
                        if key not in ['llm', 'memory']:  # 避免打印复杂对象
                            print(f"  {key}: {value}")
            except Exception as e:
                print(f"无法获取详细配置: {e}")
    
    @staticmethod
    def inspect_agent_tools(agent_executor, verbose: bool = True):
        """检查Agent的工具配置"""
        print("🛠️ Agent工具检查")
        print("=" * 40)
        
        if hasattr(agent_executor, 'tools'):
            tools = agent_executor.tools
            print(f"工具数量: {len(tools)}")
            
            for i, tool in enumerate(tools, 1):
                print(f"\\n工具{i}: {tool.name}")
                print(f"  描述: {tool.description}")
                
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    print(f"  参数模式: {tool.args_schema}")
                
                if verbose and hasattr(tool, 'func'):
                    # 获取函数签名
                    try:
                        sig = inspect.signature(tool.func)
                        print(f"  函数签名: {sig}")
                    except Exception:
                        pass
        
        # 检查Agent配置
        if hasattr(agent_executor, 'agent'):
            agent = agent_executor.agent
            print(f"\\nAgent类型: {type(agent).__name__}")
            
            if hasattr(agent_executor, 'max_iterations'):
                print(f"最大迭代次数: {agent_executor.max_iterations}")
    
    @staticmethod
    def trace_execution(func: Callable, *args, **kwargs):
        """跟踪函数执行过程"""
        print(f"🔍 开始跟踪执行: {func.__name__}")
        print("=" * 50)
        
        # 记录输入参数
        print("📥 输入参数:")
        if args:
            print(f"  位置参数: {args}")
        if kwargs:
            print(f"  关键字参数: {kwargs}")
        
        try:
            # 执行函数
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # 记录结果
            print(f"\\n📤 执行结果:")
            print(f"  返回值类型: {type(result).__name__}")
            print(f"  执行时间: {end_time - start_time:.3f} 秒")
            
            # 如果结果是字典或有合理的字符串表示，显示部分内容
            if isinstance(result, dict):
                print(f"  结果键: {list(result.keys())}")
            elif isinstance(result, str) and len(result) > 100:
                print(f"  结果预览: {result[:100]}...")
            else:
                print(f"  结果: {result}")
            
            return result
            
        except Exception as e:
            print(f"\\n❌ 执行失败:")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {e}")
            print(f"  错误堆栈:")
            traceback.print_exc()
            raise

class LearningMetrics:
    """学习指标收集器"""
    
    def __init__(self):
        self.metrics = {
            'execution_times': [],
            'success_count': 0,
            'error_count': 0,
            'tool_usage': {},
            'session_start': datetime.now()
        }
    
    def record_execution(self, operation: str, execution_time: float, success: bool = True):
        """记录执行指标"""
        self.metrics['execution_times'].append({
            'operation': operation,
            'time': execution_time,
            'timestamp': datetime.now(),
            'success': success
        })
        
        if success:
            self.metrics['success_count'] += 1
        else:
            self.metrics['error_count'] += 1
    
    def record_tool_usage(self, tool_name: str):
        """记录工具使用情况"""
        if tool_name not in self.metrics['tool_usage']:
            self.metrics['tool_usage'][tool_name] = 0
        self.metrics['tool_usage'][tool_name] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取学习指标摘要"""
        total_operations = len(self.metrics['execution_times'])
        session_duration = datetime.now() - self.metrics['session_start']
        
        if self.metrics['execution_times']:
            avg_time = sum(op['time'] for op in self.metrics['execution_times']) / total_operations
            max_time = max(op['time'] for op in self.metrics['execution_times'])
            min_time = min(op['time'] for op in self.metrics['execution_times'])
        else:
            avg_time = max_time = min_time = 0
        
        return {
            'session_duration': session_duration.total_seconds(),
            'total_operations': total_operations,
            'success_rate': self.metrics['success_count'] / max(total_operations, 1) * 100,
            'average_execution_time': avg_time,
            'max_execution_time': max_time,
            'min_execution_time': min_time,
            'tool_usage_stats': self.metrics['tool_usage'],
            'most_used_tool': max(self.metrics['tool_usage'].items(), key=lambda x: x[1])[0] if self.metrics['tool_usage'] else None
        }
    
    def display_summary(self):
        """显示学习指标摘要"""
        summary = self.get_summary()
        
        print("📊 学习会话指标摘要")
        print("=" * 40)
        print(f"会话时长: {summary['session_duration']:.1f} 秒")
        print(f"总操作数: {summary['total_operations']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print(f"平均执行时间: {summary['average_execution_time']:.3f} 秒")
        print(f"最长执行时间: {summary['max_execution_time']:.3f} 秒")
        print(f"最短执行时间: {summary['min_execution_time']:.3f} 秒")
        
        if summary['tool_usage_stats']:
            print(f"\\n🛠️ 工具使用统计:")
            for tool, count in sorted(summary['tool_usage_stats'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count} 次")
            
            if summary['most_used_tool']:
                print(f"\\n🏆 最常用工具: {summary['most_used_tool']}")

class ExampleCodeRunner:
    """示例代码运行器"""
    
    def __init__(self, enable_metrics: bool = True):
        self.enable_metrics = enable_metrics
        if enable_metrics:
            self.metrics = LearningMetrics()
    
    def run_example(self, example_name: str, code_func: Callable, *args, **kwargs):
        """运行示例代码"""
        print(f"🚀 运行示例: {example_name}")
        print("=" * 50)
        
        try:
            with CodeExecutionTimer(f"示例 {example_name}") as timer:
                result = code_func(*args, **kwargs)
            
            if self.enable_metrics:
                self.metrics.record_execution(example_name, timer.get_execution_time(), True)
            
            print(f"\\n✅ 示例 '{example_name}' 执行成功")
            return result
            
        except Exception as e:
            if self.enable_metrics:
                execution_time = timer.get_execution_time() if 'timer' in locals() else 0
                self.metrics.record_execution(example_name, execution_time, False)
            
            print(f"\\n❌ 示例 '{example_name}' 执行失败: {e}")
            raise
    
    def run_multiple_examples(self, examples: List[Dict[str, Any]]):
        """批量运行多个示例"""
        print(f"🎯 批量运行 {len(examples)} 个示例")
        print("=" * 60)
        
        results = {}
        for i, example in enumerate(examples, 1):
            print(f"\\n[{i}/{len(examples)}] ", end="")
            
            try:
                result = self.run_example(
                    example['name'],
                    example['func'],
                    *example.get('args', []),
                    **example.get('kwargs', {})
                )
                results[example['name']] = {'success': True, 'result': result}
                
            except Exception as e:
                results[example['name']] = {'success': False, 'error': str(e)}
                print(f"继续执行下一个示例...")
        
        # 显示批量执行摘要
        self._display_batch_summary(results)
        
        if self.enable_metrics:
            self.metrics.display_summary()
        
        return results
    
    def _display_batch_summary(self, results: Dict[str, Dict]):
        """显示批量执行摘要"""
        success_count = sum(1 for r in results.values() if r['success'])
        total_count = len(results)
        
        print(f"\\n📊 批量执行摘要")
        print("=" * 40)
        print(f"总示例数: {total_count}")
        print(f"成功执行: {success_count}")
        print(f"执行失败: {total_count - success_count}")
        print(f"成功率: {success_count / total_count * 100:.1f}%")
        
        # 显示失败的示例
        failed_examples = [name for name, result in results.items() if not result['success']]
        if failed_examples:
            print(f"\\n❌ 失败的示例:")
            for example_name in failed_examples:
                error = results[example_name]['error']
                print(f"  • {example_name}: {error}")

class ConfigurationHelper:
    """配置辅助工具"""
    
    @staticmethod
    def check_environment():
        """检查学习环境配置"""
        print("🔍 检查学习环境配置")
        print("=" * 40)
        
        # 检查Python版本
        python_version = sys.version_info
        print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version >= (3, 9):
            print("✅ Python版本满足要求 (>= 3.9)")
        else:
            print("⚠️ Python版本可能过低，建议升级到3.9+")
        
        # 检查必需的包
        required_packages = [
            'langchain',
            'langchain_openai',
            'openai',
            'pandas',
            'matplotlib',
            'jupyter',
            'python_dotenv'
        ]
        
        print(f"\\n📦 检查必需包:")
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}: 已安装")
            except ImportError:
                print(f"❌ {package}: 未安装")
        
        # 检查环境变量
        print(f"\\n🔑 检查环境变量:")
        env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        
        import os
        for var in env_vars:
            if os.getenv(var):
                print(f"✅ {var}: 已配置")
            else:
                print(f"⚠️ {var}: 未配置")
    
    @staticmethod
    def generate_config_template() -> str:
        """生成配置文件模板"""
        template = """# LangChain学习环境配置文件
# 复制此文件为 .env 并填入您的API密钥

# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_organization_id_here

# Anthropic API配置  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 其他配置
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langchain-learning

# Jupyter配置
JUPYTER_ENABLE_LAB=true
JUPYTER_PORT=8888
"""
        return template
    
    @staticmethod
    def save_config_template(file_path: str = ".env.example"):
        """保存配置文件模板"""
        template = ConfigurationHelper.generate_config_template()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(template)
            print(f"✅ 配置模板已保存到: {file_path}")
            print("💡 请复制此文件为 .env 并填入您的API密钥")
        except Exception as e:
            print(f"❌ 保存配置模板失败: {e}")

class LearningPathGuide:
    """学习路径指导"""
    
    @staticmethod
    def get_learning_roadmap() -> Dict[str, List[str]]:
        """获取完整的学习路线图"""
        return {
            "基础概念": [
                "01_langchain_introduction.ipynb - LangChain基础介绍",
                "02_llm_basics.ipynb - 大语言模型基础",
                "03_prompts_templates.ipynb - 提示词和模板"
            ],
            "核心组件": [
                "01_chains_introduction.ipynb - 链的介绍和使用",
                "02_agents_basics.ipynb - 智能代理基础",
                "03_memory_systems.ipynb - 记忆系统"
            ],
            "高级应用": [
                "01_rag_systems.ipynb - RAG检索增强生成",
                "02_multi_agent.ipynb - 多代理系统",
                "03_evaluation.ipynb - 评估和优化"
            ],
            "实战项目": [
                "01_chatbot_project.ipynb - 聊天机器人项目",
                "02_qa_system.ipynb - 问答系统项目",
                "03_document_analysis.ipynb - 文档分析项目"
            ]
        }
    
    @staticmethod
    def recommend_next_steps(completed_notebooks: List[str]) -> List[str]:
        """根据已完成的notebook推荐下一步学习内容"""
        roadmap = LearningPathGuide.get_learning_roadmap()
        
        # 展平所有课程
        all_courses = []
        for category, courses in roadmap.items():
            for course in courses:
                notebook_name = course.split(" - ")[0]
                all_courses.append((category, notebook_name, course))
        
        # 找到下一个应该学习的课程
        recommendations = []
        
        for category, notebook_name, full_description in all_courses:
            if notebook_name not in completed_notebooks:
                recommendations.append(full_description)
                if len(recommendations) >= 3:  # 限制推荐数量
                    break
        
        return recommendations
    
    @staticmethod
    def display_progress(completed_notebooks: List[str]):
        """显示学习进度"""
        roadmap = LearningPathGuide.get_learning_roadmap()
        
        print("📚 学习进度概览")
        print("=" * 50)
        
        total_courses = 0
        completed_courses = 0
        
        for category, courses in roadmap.items():
            print(f"\\n📖 {category}:")
            category_completed = 0
            
            for course in courses:
                notebook_name = course.split(" - ")[0]
                total_courses += 1
                
                if notebook_name in completed_notebooks:
                    print(f"  ✅ {course}")
                    completed_courses += 1
                    category_completed += 1
                else:
                    print(f"  ⭕ {course}")
            
            completion_rate = category_completed / len(courses) * 100
            print(f"  📊 分类完成度: {completion_rate:.1f}%")
        
        overall_completion = completed_courses / total_courses * 100
        print(f"\\n🎯 总体完成度: {overall_completion:.1f}% ({completed_courses}/{total_courses})")
        
        # 推荐下一步
        recommendations = LearningPathGuide.recommend_next_steps(completed_notebooks)
        if recommendations:
            print(f"\\n📈 推荐下一步学习:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

# 全局学习指标实例
global_metrics = LearningMetrics()

# 便捷函数
def quick_timer(description: str = ""):
    """快速创建计时器上下文管理器"""
    return CodeExecutionTimer(description)

def debug_chain(chain, verbose: bool = True):
    """快速调试Chain"""
    return LangChainDebugger.inspect_chain_components(chain, verbose)

def debug_agent(agent_executor, verbose: bool = True):
    """快速调试Agent"""
    return LangChainDebugger.inspect_agent_tools(agent_executor, verbose)

def check_env():
    """快速检查环境"""
    ConfigurationHelper.check_environment()

def show_roadmap():
    """显示学习路线图"""
    roadmap = LearningPathGuide.get_learning_roadmap()
    
    print("🗺️ LangChain学习路线图")
    print("=" * 50)
    
    for category, courses in roadmap.items():
        print(f"\\n📚 {category}:")
        for i, course in enumerate(courses, 1):
            print(f"  {i}. {course}")

if __name__ == "__main__":
    # 运行环境检查
    print("🔧 LangChain学习辅助工具初始化")
    print("=" * 50)
    
    check_env()
    print("\\n")
    show_roadmap()
    
    print("\\n💡 使用提示:")
    print("  • 使用 quick_timer('描述') 来计时代码执行")
    print("  • 使用 debug_chain(chain) 来调试链")
    print("  • 使用 debug_agent(agent) 来调试代理")
    print("  • 使用 check_env() 来检查环境配置")
    print("  • 使用 show_roadmap() 来查看学习路线图")