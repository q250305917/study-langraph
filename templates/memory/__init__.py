"""
记忆系统模板模块

本模块提供了完整的记忆系统模板，用于管理对话历史和上下文信息：

模板类型：
- ConversationMemoryTemplate: 对话记忆模板，管理对话历史和上下文维护
- SummaryMemoryTemplate: 摘要记忆模板，提供智能压缩和关键信息提取

核心功能：
1. 对话历史管理 - 存储、检索和更新对话记录
2. 上下文维护 - 保持对话的连续性和相关性
3. 智能压缩 - 对长对话进行摘要压缩
4. 关键信息提取 - 从对话中提取重要信息
5. 多种存储后端 - 支持内存、文件、数据库等存储方式
6. 向量记忆 - 基于向量相似度的记忆检索

使用场景：
- 聊天机器人对话管理
- 客服系统会话记录
- 教学辅助对话跟踪
- 知识问答系统
- 多轮对话应用

设计原理：
- 模块化设计，易于扩展和定制
- 支持多种存储后端和检索策略
- 内置智能压缩和摘要功能
- 完整的生命周期管理
- 丰富的配置选项和性能优化
"""

from .conversation_memory import ConversationMemoryTemplate
from .summary_memory import SummaryMemoryTemplate

__all__ = [
    "ConversationMemoryTemplate",
    "SummaryMemoryTemplate",
]

# 模块版本信息
__version__ = "1.0.0"
__author__ = "LangChain Learning Team"
__description__ = "记忆系统模板模块"