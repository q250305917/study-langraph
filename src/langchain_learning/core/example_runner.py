"""
LangChain学习项目的统一示例运行器模块

本模块实现了统一的示例运行和管理系统，支持：
- 示例的注册、发现和执行
- 多种示例类型（代码片段、完整示例、交互式示例）
- 依赖管理和环境检查
- 执行结果记录和分析
- 示例的分类和标签管理
- 批量执行和并行处理
"""

import asyncio
import inspect
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import json
from datetime import datetime

from pydantic import BaseModel, Field, validator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.syntax import Syntax

from .logger import get_logger
from .config import get_config_value
from .llm_factory import get_llm_factory, LLMInstance
from .prompt_manager import get_prompt_manager
from .tools import get_tool_manager
from .exceptions import (
    ValidationError,
    ResourceError,
    ChainExecutionError,
    ErrorCodes
)

logger = get_logger(__name__)


class ExampleType(Enum):
    """示例类型枚举"""
    CODE_SNIPPET = "code_snippet"     # 代码片段
    TUTORIAL = "tutorial"             # 教程示例
    DEMO = "demo"                     # 演示示例
    TEST_CASE = "test_case"           # 测试用例
    BENCHMARK = "benchmark"           # 性能测试
    INTERACTIVE = "interactive"       # 交互式示例
    NOTEBOOK = "notebook"             # Jupyter笔记本


class ExampleStatus(Enum):
    """示例状态枚举"""
    PENDING = "pending"               # 等待执行
    RUNNING = "running"               # 执行中
    COMPLETED = "completed"           # 执行完成
    FAILED = "failed"                # 执行失败
    SKIPPED = "skipped"               # 跳过执行


@dataclass
class ExampleMetadata:
    """
    示例元数据
    
    记录示例的基本信息和统计数据。
    """
    name: str
    example_type: ExampleType
    category: str = ""
    description: str = ""
    author: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # 难度和时间估计
    difficulty: str = "beginner"      # beginner, intermediate, advanced
    estimated_time: int = 0           # 预估执行时间（秒）
    
    # 统计信息
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


class ExampleInput(BaseModel):
    """示例输入数据模型"""
    
    parameters: Dict[str, Any] = Field(default_factory=dict, description="示例参数")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置选项")
    environment: Dict[str, str] = Field(default_factory=dict, description="环境变量")
    
    class Config:
        extra = "allow"


class ExampleOutput(BaseModel):
    """示例输出数据模型"""
    
    status: ExampleStatus = Field(description="执行状态")
    result: Any = Field(default=None, description="执行结果")
    output: str = Field(default="", description="标准输出")
    error: Optional[str] = Field(default=None, description="错误信息")
    execution_time: float = Field(description="执行时间")
    start_time: datetime = Field(description="开始时间")
    end_time: Optional[datetime] = Field(default=None, description="结束时间")
    
    # 性能指标
    memory_usage: Optional[int] = Field(default=None, description="内存使用量")
    cpu_usage: Optional[float] = Field(default=None, description="CPU使用率")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @property
    def is_successful(self) -> bool:
        """检查执行是否成功"""
        return self.status == ExampleStatus.COMPLETED
    
    @property
    def duration(self) -> Optional[float]:
        """获取执行时长"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class BaseExample(ABC):
    """
    示例抽象基类
    
    定义示例的通用接口，所有具体示例都应该继承此类。
    """
    
    def __init__(
        self,
        name: str,
        metadata: Optional[ExampleMetadata] = None
    ):
        """
        初始化示例
        
        Args:
            name: 示例名称
            metadata: 示例元数据
        """
        self.name = name
        self.metadata = metadata or ExampleMetadata(
            name=name,
            example_type=ExampleType.CODE_SNIPPET
        )
        
        logger.debug(f"Initialized example: {self.name}")
    
    @abstractmethod
    async def execute(self, inputs: ExampleInput) -> ExampleOutput:
        """
        执行示例的核心方法
        
        Args:
            inputs: 示例输入
            
        Returns:
            示例输出
            
        Raises:
            Exception: 执行失败
        """
        pass
    
    async def run(self, **kwargs) -> ExampleOutput:
        """
        执行示例的主要入口方法
        
        Args:
            **kwargs: 示例参数
            
        Returns:
            示例输出
        """
        start_time = datetime.now()
        
        try:
            # 准备输入数据
            inputs = ExampleInput(parameters=kwargs)
            
            # 环境检查
            await self._check_environment(inputs)
            
            # 依赖检查
            await self._check_dependencies()
            
            # 执行示例
            result = await self.execute(inputs)
            
            # 更新统计信息
            execution_time = time.time() - start_time.timestamp()
            self._update_metrics(True, execution_time)
            
            result.status = ExampleStatus.COMPLETED
            result.end_time = datetime.now()
            
            return result
            
        except Exception as e:
            # 处理执行失败
            execution_time = time.time() - start_time.timestamp()
            self._update_metrics(False, execution_time)
            
            error_output = ExampleOutput(
                status=ExampleStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                start_time=start_time,
                end_time=datetime.now(),
                metadata={"exception_type": type(e).__name__}
            )
            
            logger.error(f"Example execution failed: {self.name} - {e}")
            return error_output
    
    async def _check_environment(self, inputs: ExampleInput) -> None:
        """检查环境要求"""
        # 子类可以重写此方法来实现自定义的环境检查
        pass
    
    async def _check_dependencies(self) -> None:
        """检查依赖要求"""
        for dependency in self.metadata.dependencies:
            try:
                __import__(dependency)
            except ImportError:
                raise ResourceError(
                    f"Missing dependency: {dependency}",
                    error_code=ErrorCodes.VALIDATION_REQUIRED_ERROR,
                    context={"example": self.name, "dependency": dependency}
                )
    
    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """更新统计信息"""
        self.metadata.execution_count += 1
        self.metadata.last_execution_time = time.time()
        
        if success:
            self.metadata.success_count += 1
        else:
            self.metadata.failure_count += 1
        
        # 更新平均执行时间
        if self.metadata.average_execution_time == 0:
            self.metadata.average_execution_time = execution_time
        else:
            alpha = 0.1
            self.metadata.average_execution_time = (
                alpha * execution_time + 
                (1 - alpha) * self.metadata.average_execution_time
            )
    
    def get_info(self) -> Dict[str, Any]:
        """获取示例信息"""
        return {
            "name": self.name,
            "type": self.metadata.example_type.value,
            "category": self.metadata.category,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "version": self.metadata.version,
            "tags": self.metadata.tags,
            "difficulty": self.metadata.difficulty,
            "estimated_time": self.metadata.estimated_time,
            "dependencies": self.metadata.dependencies,
            "metrics": {
                "execution_count": self.metadata.execution_count,
                "success_rate": self.metadata.success_rate,
                "average_execution_time": self.metadata.average_execution_time
            }
        }


class CodeExample(BaseExample):
    """
    代码示例
    
    执行Python代码片段的示例。
    """
    
    def __init__(
        self,
        name: str,
        code: str,
        description: str = "",
        **kwargs
    ):
        """
        初始化代码示例
        
        Args:
            name: 示例名称
            code: 代码内容
            description: 示例描述
            **kwargs: 其他元数据
        """
        metadata = ExampleMetadata(
            name=name,
            example_type=ExampleType.CODE_SNIPPET,
            description=description,
            **kwargs
        )
        
        super().__init__(name, metadata)
        self.code = code
    
    async def execute(self, inputs: ExampleInput) -> ExampleOutput:
        """执行代码示例"""
        import io
        import contextlib
        
        start_time = datetime.now()
        
        # 创建输出捕获
        output_buffer = io.StringIO()
        
        # 准备执行环境
        exec_globals = {
            "__name__": "__main__",
            **inputs.parameters
        }
        
        # 注入核心组件
        exec_globals.update({
            "llm_factory": get_llm_factory(),
            "prompt_manager": get_prompt_manager(),
            "tool_manager": get_tool_manager(),
            "logger": logger
        })
        
        try:
            # 执行代码并捕获输出
            with contextlib.redirect_stdout(output_buffer):
                exec(self.code, exec_globals)
            
            # 获取结果
            result = exec_globals.get("result")
            output = output_buffer.getvalue()
            
            return ExampleOutput(
                status=ExampleStatus.COMPLETED,
                result=result,
                output=output,
                execution_time=0,  # 将在run方法中设置
                start_time=start_time
            )
            
        except Exception as e:
            return ExampleOutput(
                status=ExampleStatus.FAILED,
                error=str(e),
                output=output_buffer.getvalue(),
                execution_time=0,
                start_time=start_time,
                metadata={"traceback": traceback.format_exc()}
            )


class ChainExample(BaseExample):
    """
    链式示例
    
    执行LangChain链的示例。
    """
    
    def __init__(
        self,
        name: str,
        chain_factory: Callable,
        description: str = "",
        **kwargs
    ):
        """
        初始化链式示例
        
        Args:
            name: 示例名称
            chain_factory: 创建链的工厂函数
            description: 示例描述
            **kwargs: 其他元数据
        """
        metadata = ExampleMetadata(
            name=name,
            example_type=ExampleType.DEMO,
            description=description,
            **kwargs
        )
        
        super().__init__(name, metadata)
        self.chain_factory = chain_factory
    
    async def execute(self, inputs: ExampleInput) -> ExampleOutput:
        """执行链式示例"""
        start_time = datetime.now()
        
        try:
            # 创建链
            chain = self.chain_factory()
            
            # 执行链
            result = await chain.run(inputs.parameters)
            
            return ExampleOutput(
                status=ExampleStatus.COMPLETED,
                result=result,
                execution_time=0,
                start_time=start_time
            )
            
        except Exception as e:
            return ExampleOutput(
                status=ExampleStatus.FAILED,
                error=str(e),
                execution_time=0,
                start_time=start_time
            )


class ExampleRegistry:
    """
    示例注册表
    
    管理示例的注册、索引和查询。
    """
    
    def __init__(self):
        """初始化示例注册表"""
        self._examples: Dict[str, BaseExample] = {}
        self._categories: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
        self._types: Dict[ExampleType, List[str]] = {}
        
        logger.debug("Initialized example registry")
    
    def register(self, example: BaseExample) -> None:
        """
        注册示例
        
        Args:
            example: 要注册的示例实例
        """
        if example.name in self._examples:
            logger.warning(f"Example '{example.name}' already registered, overwriting")
        
        self._examples[example.name] = example
        
        # 更新索引
        self._update_indexes(example)
        
        logger.info(f"Registered example: {example.name}")
    
    def get(self, name: str) -> Optional[BaseExample]:
        """获取示例"""
        return self._examples.get(name)
    
    def list_examples(self) -> List[str]:
        """获取所有示例名称"""
        return list(self._examples.keys())
    
    def get_by_category(self, category: str) -> List[BaseExample]:
        """按分类获取示例"""
        example_names = self._categories.get(category, [])
        return [self._examples[name] for name in example_names if name in self._examples]
    
    def get_by_tag(self, tag: str) -> List[BaseExample]:
        """按标签获取示例"""
        example_names = self._tags.get(tag, [])
        return [self._examples[name] for name in example_names if name in self._examples]
    
    def get_by_type(self, example_type: ExampleType) -> List[BaseExample]:
        """按类型获取示例"""
        example_names = self._types.get(example_type, [])
        return [self._examples[name] for name in example_names if name in self._examples]
    
    def search(self, query: str) -> List[BaseExample]:
        """搜索示例"""
        query = query.lower()
        matching_examples = []
        
        for example in self._examples.values():
            if (query in example.name.lower() or
                query in example.metadata.description.lower() or
                any(query in tag.lower() for tag in example.metadata.tags)):
                matching_examples.append(example)
        
        return matching_examples
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        total_examples = len(self._examples)
        
        type_counts = {}
        for example_type in ExampleType:
            count = len(self._types.get(example_type, []))
            if count > 0:
                type_counts[example_type.value] = count
        
        category_counts = {
            category: len(examples) 
            for category, examples in self._categories.items()
        }
        
        return {
            "total_examples": total_examples,
            "type_distribution": type_counts,
            "category_distribution": category_counts,
            "total_executions": sum(
                example.metadata.execution_count 
                for example in self._examples.values()
            )
        }
    
    def _update_indexes(self, example: BaseExample) -> None:
        """更新索引"""
        # 分类索引
        category = example.metadata.category
        if category:
            if category not in self._categories:
                self._categories[category] = []
            if example.name not in self._categories[category]:
                self._categories[category].append(example.name)
        
        # 标签索引
        for tag in example.metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            if example.name not in self._tags[tag]:
                self._tags[tag].append(example.name)
        
        # 类型索引
        example_type = example.metadata.example_type
        if example_type not in self._types:
            self._types[example_type] = []
        if example.name not in self._types[example_type]:
            self._types[example_type].append(example.name)


class ExampleRunner:
    """
    示例运行器
    
    提供示例的统一执行和管理接口。
    """
    
    def __init__(self):
        """初始化示例运行器"""
        self.registry = ExampleRegistry()
        self.console = Console()
        
        # 注册内置示例
        self._register_builtin_examples()
        
        logger.info("Example runner initialized")
    
    def register_example(self, example: BaseExample) -> None:
        """注册示例"""
        self.registry.register(example)
    
    def register_code_example(
        self,
        name: str,
        code: str,
        description: str = "",
        **kwargs
    ) -> CodeExample:
        """注册代码示例"""
        example = CodeExample(name, code, description, **kwargs)
        self.register_example(example)
        return example
    
    def register_chain_example(
        self,
        name: str,
        chain_factory: Callable,
        description: str = "",
        **kwargs
    ) -> ChainExample:
        """注册链式示例"""
        example = ChainExample(name, chain_factory, description, **kwargs)
        self.register_example(example)
        return example
    
    async def run_example(self, name: str, **kwargs) -> ExampleOutput:
        """
        运行单个示例
        
        Args:
            name: 示例名称
            **kwargs: 示例参数
            
        Returns:
            示例输出
            
        Raises:
            ResourceError: 示例不存在
        """
        example = self.registry.get(name)
        if not example:
            raise ResourceError(
                f"Example '{name}' not found",
                error_code=ErrorCodes.FILE_NOT_FOUND,
                context={"example_name": name}
            )
        
        logger.info(f"Running example: {name}")
        
        # 显示执行信息
        self._display_example_info(example)
        
        # 执行示例
        result = await example.run(**kwargs)
        
        # 显示结果
        self._display_result(example, result)
        
        return result
    
    async def run_examples_by_category(
        self,
        category: str,
        **kwargs
    ) -> List[ExampleOutput]:
        """按分类批量运行示例"""
        examples = self.registry.get_by_category(category)
        if not examples:
            logger.warning(f"No examples found in category: {category}")
            return []
        
        logger.info(f"Running {len(examples)} examples in category: {category}")
        
        results = []
        for example in examples:
            try:
                result = await example.run(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run example {example.name}: {e}")
        
        return results
    
    async def run_examples_parallel(
        self,
        example_names: List[str],
        **kwargs
    ) -> List[ExampleOutput]:
        """并行运行多个示例"""
        tasks = []
        
        for name in example_names:
            example = self.registry.get(name)
            if example:
                task = asyncio.create_task(example.run(**kwargs))
                tasks.append((name, task))
            else:
                logger.warning(f"Example not found: {name}")
        
        results = []
        for name, task in tasks:
            try:
                result = await task
                results.append(result)
                logger.info(f"Completed example: {name}")
            except Exception as e:
                logger.error(f"Failed to run example {name}: {e}")
        
        return results
    
    def list_examples(
        self,
        category: Optional[str] = None,
        example_type: Optional[ExampleType] = None,
        tag: Optional[str] = None
    ) -> List[BaseExample]:
        """
        列出示例
        
        Args:
            category: 分类筛选
            example_type: 类型筛选
            tag: 标签筛选
            
        Returns:
            匹配的示例列表
        """
        if category:
            return self.registry.get_by_category(category)
        elif example_type:
            return self.registry.get_by_type(example_type)
        elif tag:
            return self.registry.get_by_tag(tag)
        else:
            return [
                self.registry.get(name) 
                for name in self.registry.list_examples()
            ]
    
    def search_examples(self, query: str) -> List[BaseExample]:
        """搜索示例"""
        return self.registry.search(query)
    
    def display_examples_table(
        self,
        examples: Optional[List[BaseExample]] = None
    ) -> None:
        """显示示例表格"""
        if examples is None:
            examples = [
                self.registry.get(name) 
                for name in self.registry.list_examples()
            ]
        
        table = Table(title="LangChain Learning Examples")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Category", style="green")
        table.add_column("Description", style="white")
        table.add_column("Success Rate", justify="right", style="yellow")
        table.add_column("Avg Time", justify="right", style="blue")
        
        for example in examples:
            success_rate = f"{example.metadata.success_rate:.1%}"
            avg_time = f"{example.metadata.average_execution_time:.2f}s"
            
            table.add_row(
                example.name,
                example.metadata.example_type.value,
                example.metadata.category,
                example.metadata.description[:50] + "..." 
                if len(example.metadata.description) > 50 
                else example.metadata.description,
                success_rate,
                avg_time
            )
        
        self.console.print(table)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取运行器统计信息"""
        return self.registry.get_statistics()
    
    def _display_example_info(self, example: BaseExample) -> None:
        """显示示例信息"""
        info_text = f"[bold cyan]{example.name}[/bold cyan]\n"
        info_text += f"Type: {example.metadata.example_type.value}\n"
        info_text += f"Category: {example.metadata.category}\n"
        if example.metadata.description:
            info_text += f"Description: {example.metadata.description}\n"
        if example.metadata.tags:
            info_text += f"Tags: {', '.join(example.metadata.tags)}\n"
        
        panel = Panel(info_text, title="Example Info", border_style="blue")
        self.console.print(panel)
    
    def _display_result(self, example: BaseExample, result: ExampleOutput) -> None:
        """显示执行结果"""
        if result.is_successful:
            status_color = "green"
            status_text = "✓ SUCCESS"
        else:
            status_color = "red"
            status_text = "✗ FAILED"
        
        result_text = f"[bold {status_color}]{status_text}[/bold {status_color}]\n"
        result_text += f"Execution time: {result.execution_time:.3f}s\n"
        
        if result.output:
            result_text += f"\nOutput:\n{result.output}"
        
        if result.error:
            result_text += f"\n[red]Error: {result.error}[/red]"
        
        panel = Panel(result_text, title="Execution Result", border_style=status_color)
        self.console.print(panel)
    
    def _register_builtin_examples(self) -> None:
        """注册内置示例"""
        
        # 基础LLM调用示例
        basic_llm_code = '''
# 基础LLM调用示例
llm = llm_factory.create_llm_from_config()
response = await llm.generate("你好，请介绍一下LangChain框架。")
print(f"LLM响应: {response}")
result = response
'''
        
        self.register_code_example(
            name="basic_llm_call",
            code=basic_llm_code,
            description="演示如何创建和调用LLM实例",
            category="基础教程",
            tags=["LLM", "基础", "调用"],
            difficulty="beginner",
            estimated_time=10
        )
        
        # 提示词模板示例
        prompt_template_code = '''
# 提示词模板示例
template = prompt_manager.create_template(
    name="greeting_template",
    content="你好，{{name}}！今天是{{date}}，祝你{{wish}}！",
    template_type=prompt_manager.PromptType.USER
)

rendered = template.render(
    name="用户",
    date="2024年",
    wish="学习愉快"
)
print(f"渲染结果: {rendered}")
result = rendered
'''
        
        self.register_code_example(
            name="prompt_template",
            code=prompt_template_code,
            description="演示如何创建和使用提示词模板",
            category="提示词管理",
            tags=["提示词", "模板", "渲染"],
            difficulty="beginner",
            estimated_time=15
        )
        
        # 工具调用示例
        tool_usage_code = '''
# 工具调用示例
# 注册自定义工具
@tool_manager.register_function(name="calculate_sum", description="计算两个数的和")
def calculate_sum(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b

# 调用工具
result1 = await tool_manager.call_tool("calculate_sum", a=10, b=20)
print(f"工具调用结果: {result1.result}")

# 调用内置工具
result2 = await tool_manager.call_tool("string_upper", text="hello world")
print(f"字符串转大写: {result2.result}")

result = {"sum": result1.result, "upper": result2.result}
'''
        
        self.register_code_example(
            name="tool_usage",
            code=tool_usage_code,
            description="演示如何注册和调用工具",
            category="工具管理",
            tags=["工具", "调用", "注册"],
            difficulty="intermediate",
            estimated_time=20
        )
        
        logger.debug("Registered builtin examples")


# 全局示例运行器实例
_global_runner: Optional[ExampleRunner] = None


def get_example_runner() -> ExampleRunner:
    """
    获取全局示例运行器实例
    
    Returns:
        示例运行器实例
    """
    global _global_runner
    
    if _global_runner is None:
        _global_runner = ExampleRunner()
    
    return _global_runner


def run_example(name: str, **kwargs) -> ExampleOutput:
    """运行示例的便捷函数"""
    import asyncio
    runner = get_example_runner()
    return asyncio.run(runner.run_example(name, **kwargs))


def list_examples(**filters) -> List[BaseExample]:
    """列出示例的便捷函数"""
    return get_example_runner().list_examples(**filters)


def register_example(example: BaseExample) -> None:
    """注册示例的便捷函数"""
    get_example_runner().register_example(example)