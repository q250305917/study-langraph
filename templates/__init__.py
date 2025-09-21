"""
LangChain学习项目模板系统

本模块提供了一套完整的参数化示例代码模板系统，覆盖LangChain的所有核心应用场景。
这些模板帮助学习者快速理解和应用不同的LangChain功能，同时提供可复用的代码结构。

模块结构：
- base/: 模板系统基础架构
- llm/: LLM模型使用模板
- prompts/: 提示词模板
- chains/: 链组合模板  
- agents/: 代理模板
- data/: 数据处理模板
- memory/: 记忆系统模板
- evaluation/: 评估模板
- configs/: 配置文件系统
"""

__version__ = "0.1.0"
__author__ = "LangChain Learning Project"

# 导入基础模板类供外部使用
from .base.template_base import TemplateBase, TemplateConfig
from .base.config_loader import ConfigLoader
from .base.parameter_validator import ParameterValidator

__all__ = [
    "TemplateBase",
    "TemplateConfig", 
    "ConfigLoader",
    "ParameterValidator"
]