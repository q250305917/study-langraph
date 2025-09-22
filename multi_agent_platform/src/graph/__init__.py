"""
LangGraph工作流框架模块

提供基于LangGraph的智能体协作工作流功能，包括：
- 状态图定义和管理
- 工作流节点实现
- 条件路由和决策
- 工作流执行引擎
"""

from .state import WorkflowState, AgentWorkflowState
from .nodes import WorkflowNode, AgentNode, CoordinatorNode, DecisionNode
from .edges import ConditionalEdge, StaticEdge, DynamicEdge
from .workflow import WorkflowGraph, WorkflowEngine, WorkflowBuilder

__all__ = [
    # 状态管理
    "WorkflowState",
    "AgentWorkflowState",
    
    # 节点类型
    "WorkflowNode",
    "AgentNode", 
    "CoordinatorNode",
    "DecisionNode",
    
    # 边类型
    "ConditionalEdge",
    "StaticEdge",
    "DynamicEdge",
    
    # 工作流引擎
    "WorkflowGraph",
    "WorkflowEngine",
    "WorkflowBuilder"
]