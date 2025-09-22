"""
Agent模板测试模块

本模块包含对各种Agent模板的完整测试用例。

测试覆盖：
1. BaseAgent - 基础Agent抽象类测试
2. ConversationalAgent - 对话Agent模板测试
3. ToolCallingAgent - 工具调用Agent模板测试
4. RAGAgent - 检索增强生成Agent模板测试
5. CollaborativeAgent - 协作Agent模板测试
6. 集成测试 - 多个Agent模板的协同工作测试
7. 性能测试 - 大量请求处理测试
8. 错误处理测试 - 异常情况处理测试
"""

__all__ = [
    "test_base_agent",
    "test_conversational_agent", 
    "test_tool_calling_agent",
    "test_rag_agent",
    "test_collaborative_agent",
    "test_agent_integration"
]

# 测试工具和常量
TEST_SESSION_ID = "test_session_123"
TEST_TIMEOUT = 10.0

# 模拟数据
MOCK_USER_INPUTS = [
    "你好，我想了解Python编程",
    "请帮我计算2+3*4的结果",
    "什么是机器学习？",
    "请帮我搜索相关资料",
    "我今天心情不太好"
]

MOCK_AGENT_RESPONSES = [
    "很高兴为您介绍Python编程！",
    "计算结果是14",
    "机器学习是人工智能的一个分支...",
    "我已经为您搜索到了相关资料",
    "理解您的感受，让我们聊聊看"
]