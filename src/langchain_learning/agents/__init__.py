"""
代理模块 (agents)

本模块包含 LangChain 中各种智能代理（Agent）的实现、示例和最佳实践。
代理是能够根据输入自主决定使用哪些工具和执行哪些操作的智能系统。

主要包含的代理类型：
- Zero-shot ReAct Agent: 零样本推理-行动代理
- Conversational Agent: 对话式代理
- Plan-and-execute Agent: 计划-执行代理
- Self-ask with Search Agent: 自问自答搜索代理
- Custom Agent: 自定义代理实现

代理组件：
- Tools: 代理可使用的工具集合
- Memory: 代理的记忆系统
- Planning: 代理的规划能力
- Execution: 代理的执行引擎
- Monitoring: 代理的监控和调试

每种代理都包含：
1. 基础实现和配置
2. 工具集成和管理
3. 记忆系统配置
4. 错误处理和恢复
5. 性能监控和优化
6. 安全性考虑
"""

# 版本信息
__version__ = "0.1.0"

# 计划实现的代理类型
PLANNED_AGENTS = [
    "react_agent",         # ReAct 代理（推理-行动）
    "conversational_agent", # 对话式代理
    "plan_execute_agent",  # 计划-执行代理
    "self_ask_agent",      # 自问自答代理
    "structured_chat_agent", # 结构化聊天代理
    "openai_functions_agent", # OpenAI 函数调用代理
    "xml_agent",           # XML 格式代理
    "json_agent",          # JSON 格式代理
    "custom_agent",        # 自定义代理
    "multi_agent_system",  # 多代理系统
]

# 代理类型描述
AGENT_DESCRIPTIONS = {
    "react_agent": "ReAct代理：结合推理和行动的智能代理",
    "conversational_agent": "对话代理：支持多轮对话的智能助手",
    "plan_execute_agent": "计划执行代理：先制定计划再逐步执行的代理",
    "self_ask_agent": "自问代理：通过自问自答来解决复杂问题的代理",
    "structured_chat_agent": "结构化聊天代理：使用结构化输出的聊天代理",
    "openai_functions_agent": "函数调用代理：利用OpenAI函数调用能力的代理",
    "xml_agent": "XML代理：使用XML格式进行工具调用的代理",
    "json_agent": "JSON代理：使用JSON格式进行工具调用的代理",
    "custom_agent": "自定义代理：展示如何创建自定义代理的实现",
    "multi_agent_system": "多代理系统：多个代理协作完成复杂任务的系统",
}

# 代理能力分类
AGENT_CAPABILITIES = {
    "reasoning": ["react_agent", "plan_execute_agent", "self_ask_agent"],
    "conversation": ["conversational_agent", "structured_chat_agent"],
    "function_calling": ["openai_functions_agent", "xml_agent", "json_agent"],
    "planning": ["plan_execute_agent"],
    "collaboration": ["multi_agent_system"],
    "customization": ["custom_agent"],
}

# 尝试导入已实现的代理模块
_available_agents = []
_import_errors = []

# 尝试导入各种代理的实现
for agent_type in PLANNED_AGENTS:
    try:
        # 动态导入模块
        module = __import__(f".{agent_type}", package=__name__, level=1)
        _available_agents.append(agent_type)
    except ImportError as e:
        _import_errors.append((agent_type, str(e)))

def get_available_agents():
    """
    获取当前可用的代理类型列表
    
    Returns:
        list: 已成功导入的代理类型名称列表
    """
    return _available_agents.copy()

def get_import_errors():
    """
    获取导入失败的代理和错误信息
    
    Returns:
        list: 包含 (代理类型, 错误信息) 元组的列表
    """
    return _import_errors.copy()

def get_agent_status():
    """
    获取所有代理的状态信息
    
    Returns:
        dict: 包含代理状态的字典
    """
    status = {}
    
    for agent in PLANNED_AGENTS:
        if agent in _available_agents:
            status[agent] = {
                "available": True,
                "description": AGENT_DESCRIPTIONS.get(agent, "无描述"),
                "error": None
            }
        else:
            error_info = next(
                (error for name, error in _import_errors if name == agent),
                "代理未实现"
            )
            status[agent] = {
                "available": False,
                "description": AGENT_DESCRIPTIONS.get(agent, "无描述"), 
                "error": error_info
            }
    
    return status

def get_agents_by_capability(capability):
    """
    根据能力获取代理列表
    
    Args:
        capability (str): 能力类型 (reasoning, conversation, function_calling, etc.)
        
    Returns:
        list: 具备指定能力的代理类型列表
    """
    return AGENT_CAPABILITIES.get(capability, [])

def print_agent_status():
    """
    打印所有代理的状态信息
    """
    print(f"\n🤖 代理模块状态 (agents v{__version__}):")
    print("=" * 60)
    
    status = get_agent_status()
    
    for agent, info in status.items():
        status_icon = "✅" if info["available"] else "❌"
        print(f"{status_icon} {agent:20} - {info['description']}")
        
        if not info["available"] and info["error"] != "代理未实现":
            print(f"   错误: {info['error']}")
    
    print(f"\n📊 统计: {len(_available_agents)}/{len(PLANNED_AGENTS)} 个代理可用")
    
    if _import_errors:
        print(f"\n⚠️  导入错误: {len(_import_errors)} 个")

def print_agent_capabilities():
    """
    打印代理能力分类
    """
    print(f"\n🎯 代理能力分类:")
    print("=" * 40)
    
    for capability, agents in AGENT_CAPABILITIES.items():
        print(f"\n🔸 {capability.upper()}:")
        for agent in agents:
            status = "✅" if agent in _available_agents else "❌"
            description = AGENT_DESCRIPTIONS.get(agent, "无描述")
            print(f"  {status} {agent}: {description}")

def get_agent_examples():
    """
    获取各种代理的使用示例
    
    Returns:
        dict: 包含代理类型和示例代码的字典
    """
    examples = {
        "react_agent": """
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import OpenAI

# 准备工具
tools = [
    DuckDuckGoSearchRun(),
    # 可以添加更多工具
]

# 创建 ReAct 代理
agent = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行代理
result = agent.run("2024年世界杯在哪里举办？")
        """,
        
        "conversational_agent": """
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import OpenAI

# 准备记忆和工具
memory = ConversationBufferMemory(memory_key="chat_history")
tools = [DuckDuckGoSearchRun()]

# 创建对话代理
agent = initialize_agent(
    tools=tools,
    llm=OpenAI(temperature=0),
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 进行多轮对话
response1 = agent.run("我想了解一下人工智能的发展历史")
response2 = agent.run("那深度学习是什么时候开始兴起的？")
        """,
        
        "plan_execute_agent": """
from langchain.agents import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun

# 准备工具
tools = [DuckDuckGoSearchRun()]

# 创建规划器和执行器
planner = load_chat_planner(OpenAI(temperature=0))
executor = load_agent_executor(OpenAI(temperature=0), tools, verbose=True)

# 创建计划-执行代理
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 运行复杂任务
result = agent.run("研究并比较Python和JavaScript在机器学习领域的应用")
        """
    }
    
    return examples

def print_agent_examples():
    """
    打印代理的使用示例
    """
    examples = get_agent_examples()
    
    print(f"\n📝 代理使用示例:")
    print("=" * 60)
    
    for agent_type, example in examples.items():
        print(f"\n🤖 {agent_type} 示例:")
        print("-" * 40)
        print(example)

# 定义公共接口
__all__ = [
    # 版本信息
    "__version__",
    
    # 状态查询函数
    "get_available_agents",
    "get_import_errors",
    "get_agent_status",
    "print_agent_status",
    
    # 能力相关函数
    "get_agents_by_capability",
    "print_agent_capabilities",
    
    # 示例函数
    "get_agent_examples",
    "print_agent_examples",
    
    # 代理描述和分类
    "PLANNED_AGENTS",
    "AGENT_DESCRIPTIONS",
    "AGENT_CAPABILITIES",
]

# 如果直接运行此模块，显示状态信息和示例
if __name__ == "__main__":
    print_agent_status()
    print_agent_capabilities()
    print_agent_examples()