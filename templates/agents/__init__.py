"""
Agent模板模块

本模块提供各种Agent模板的实现，支持不同类型的AI智能体开发需求。

模块结构：
1. BaseAgent - Agent基础抽象类，定义统一接口和核心功能
2. ConversationalAgent - 对话Agent模板，支持多轮对话和上下文维护
3. ToolCallingAgent - 工具调用Agent模板，支持外部工具集成和调用
4. RAGAgent - 检索增强生成Agent模板，集成知识库查询和答案生成
5. CollaborativeAgent - 协作Agent模板，支持多Agent协作和任务分发

核心特性：
- 基于TemplateBase的统一架构
- 完整的异步支持
- 灵活的配置管理
- 丰富的工具集成
- 智能的状态管理
- 详细的性能监控
"""

from .base_agent import BaseAgent
from .conversational_agent import ConversationalAgent
from .tool_calling_agent import ToolCallingAgent
from .rag_agent import RAGAgent
from .collaborative_agent import CollaborativeAgent

__all__ = [
    "BaseAgent",
    "ConversationalAgent", 
    "ToolCallingAgent",
    "RAGAgent",
    "CollaborativeAgent"
]

# Agent工厂函数
def create_agent(agent_type: str, **kwargs):
    """
    创建指定类型的Agent实例
    
    Args:
        agent_type: Agent类型 ("conversational", "tool_calling", "rag", "collaborative")
        **kwargs: Agent初始化参数
        
    Returns:
        相应的Agent实例
        
    Raises:
        ValueError: 不支持的Agent类型
    """
    agent_map = {
        "conversational": ConversationalAgent,
        "tool_calling": ToolCallingAgent,
        "rag": RAGAgent,
        "collaborative": CollaborativeAgent
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"不支持的Agent类型: {agent_type}")
        
    return agent_map[agent_type](**kwargs)