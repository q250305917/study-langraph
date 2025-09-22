"""
Prompt模板模块

本模块提供了一套完整的提示词模板系统，支持多种类型的文本生成和对话任务。
包含对话模板、补全模板、少样本学习模板和角色扮演模板。

核心组件：
- ChatTemplate: 多轮对话模板，支持角色设定和历史管理
- CompletionTemplate: 文本补全模板，支持代码生成和内容创作
- FewShotTemplate: 少样本学习模板，支持智能示例选择
- RolePlayingTemplate: 角色扮演模板，支持专业角色设定

设计特点：
1. 统一接口：所有模板都继承自TemplateBase，提供一致的使用体验
2. 灵活配置：支持丰富的参数配置和动态调整
3. 高度集成：可以与LLM模板无缝集成使用
4. 易于扩展：模块化设计，便于添加新的模板类型
5. 生产就绪：包含完整的错误处理和性能优化

使用示例：
    >>> from templates.prompts import ChatTemplate, CompletionTemplate
    >>> from templates.llm import OpenAITemplate
    >>> 
    >>> # 创建LLM模板
    >>> llm = OpenAITemplate()
    >>> llm.setup(api_key="your-key", model_name="gpt-3.5-turbo")
    >>> 
    >>> # 创建对话模板
    >>> chat = ChatTemplate()
    >>> chat.setup(
    ...     role_name="Python助手",
    ...     expertise=["Python编程", "算法设计"],
    ...     llm_template=llm
    ... )
    >>> 
    >>> # 进行对话
    >>> response = chat.run("如何使用Python实现快速排序？")
    >>> print(response.message.content)

版本历史：
- v1.0.0: 初始版本，实现基础的四种模板类型
"""

# 导入所有模板类
from .chat_template import (
    ChatTemplate,
    Message,
    MessageRole,
    ConversationContext,
    ConversationHistory,
    ConversationState,
    ChatResponse
)

from .completion_template import (
    CompletionTemplate,
    CompletionType,
    CompletionStrategy,
    CompletionContext,
    CompletionResult
)

from .few_shot_template import (
    FewShotTemplate,
    Example,
    ExampleType,
    SelectionStrategy,
    FewShotContext,
    FewShotResult,
    ExampleDatabase,
    ExampleSelector,
    SimilaritySelector,
    DiverseSelector,
    AdaptiveSelector
)

from .role_playing_template import (
    RolePlayingTemplate,
    RoleProfile,
    RoleType,
    RoleState,
    InteractionMode,
    RoleContext,
    RoleResponse
)

# 定义公开的API
__all__ = [
    # 主要模板类
    "ChatTemplate",
    "CompletionTemplate", 
    "FewShotTemplate",
    "RolePlayingTemplate",
    
    # 对话模板相关
    "Message",
    "MessageRole",
    "ConversationContext",
    "ConversationHistory", 
    "ConversationState",
    "ChatResponse",
    
    # 补全模板相关
    "CompletionType",
    "CompletionStrategy",
    "CompletionContext",
    "CompletionResult",
    
    # 少样本学习相关
    "Example",
    "ExampleType",
    "SelectionStrategy",
    "FewShotContext",
    "FewShotResult",
    "ExampleDatabase",
    "ExampleSelector",
    "SimilaritySelector",
    "DiverseSelector", 
    "AdaptiveSelector",
    
    # 角色扮演相关
    "RoleProfile",
    "RoleType",
    "RoleState",
    "InteractionMode",
    "RoleContext",
    "RoleResponse",
]

# 版本信息
__version__ = "1.0.0"
__author__ = "LangChain Learning Project"

# 模块级文档
__doc__ = """
Prompt模板模块 - 提供完整的提示词模板系统

本模块包含四个核心模板类：

1. ChatTemplate - 多轮对话模板
   - 支持角色设定和个性化
   - 智能对话历史管理
   - 动态参数替换和条件分支

2. CompletionTemplate - 文本补全模板  
   - 支持多种补全策略
   - 文本续写、代码生成、内容创作
   - 质量控制和格式化输出

3. FewShotTemplate - 少样本学习模板
   - 智能示例选择和匹配
   - 支持多种选择策略
   - 动态示例管理和质量评估

4. RolePlayingTemplate - 角色扮演模板
   - 专业角色设定和管理
   - 情境模拟和行为一致性
   - 多种互动模式支持

每个模板都提供：
- 统一的setup()配置接口
- 灵活的execute()执行方法  
- 丰富的get_example()使用示例
- 完整的错误处理和日志记录
"""


def get_available_templates():
    """
    获取可用的模板列表
    
    Returns:
        Dict[str, type]: 模板名称到类的映射
    """
    return {
        "chat": ChatTemplate,
        "completion": CompletionTemplate,
        "few_shot": FewShotTemplate,
        "role_playing": RolePlayingTemplate
    }


def create_template(template_type: str, **kwargs):
    """
    便捷的模板创建函数
    
    Args:
        template_type: 模板类型 ("chat", "completion", "few_shot", "role_playing")
        **kwargs: 传递给模板的参数
        
    Returns:
        创建的模板实例
        
    Raises:
        ValueError: 不支持的模板类型
        
    Example:
        >>> template = create_template("chat", role_name="助手")
        >>> template.setup(llm_template=llm)
    """
    templates = get_available_templates()
    
    if template_type not in templates:
        available = list(templates.keys())
        raise ValueError(f"Unknown template type: {template_type}. Available: {available}")
    
    template_class = templates[template_type]
    return template_class(**kwargs)


def get_template_info(template_type: str = None):
    """
    获取模板信息
    
    Args:
        template_type: 模板类型，None表示获取所有模板信息
        
    Returns:
        模板信息字典或列表
    """
    templates = get_available_templates()
    
    if template_type:
        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template_class = templates[template_type]
        return {
            "name": template_class.__name__,
            "description": template_class.__doc__.split('\n')[1].strip() if template_class.__doc__ else "",
            "module": template_class.__module__,
            "version": __version__
        }
    else:
        info = {}
        for name, template_class in templates.items():
            info[name] = {
                "name": template_class.__name__,
                "description": template_class.__doc__.split('\n')[1].strip() if template_class.__doc__ else "",
                "module": template_class.__module__
            }
        return info


# 设置默认的日志级别
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)