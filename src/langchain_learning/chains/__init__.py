"""
链模块 (chains)

本模块包含 LangChain 中各种链（Chain）的实现、示例和最佳实践。
链是 LangChain 的核心概念之一，用于组合和编排不同的组件来完成复杂任务。

主要包含的链类型：
- LLM Chain: 基础的大语言模型链
- Conversation Chain: 对话链，支持上下文记忆
- Sequential Chain: 顺序链，按顺序执行多个步骤
- Router Chain: 路由链，根据输入选择不同的处理路径
- Transform Chain: 转换链，对数据进行预处理和后处理
- Custom Chain: 自定义链的实现示例

每种链都包含：
1. 基础实现和配置
2. 使用示例和最佳实践
3. 错误处理和异常管理
4. 性能优化技巧
5. 测试用例
"""

# 版本信息
__version__ = "0.1.0"

# 计划实现的链类型
PLANNED_CHAINS = [
    "llm_chain",           # LLM 基础链
    "conversation_chain",  # 对话链
    "sequential_chain",    # 顺序链
    "parallel_chain",      # 并行链
    "router_chain",        # 路由链
    "transform_chain",     # 转换链
    "summarization_chain", # 摘要链
    "qa_chain",           # 问答链
    "retrieval_chain",    # 检索链
    "custom_chain",       # 自定义链
]

# 链类型描述
CHAIN_DESCRIPTIONS = {
    "llm_chain": "LLM基础链：最基本的语言模型调用链",
    "conversation_chain": "对话链：支持上下文记忆的对话处理链",
    "sequential_chain": "顺序链：按顺序执行多个处理步骤的链",
    "parallel_chain": "并行链：并行执行多个任务的链",
    "router_chain": "路由链：根据输入内容路由到不同处理分支的链",
    "transform_chain": "转换链：对输入数据进行预处理和转换的链",
    "summarization_chain": "摘要链：文本摘要和内容总结的链",
    "qa_chain": "问答链：基于文档或知识库的问答链",
    "retrieval_chain": "检索链：结合向量检索的增强生成链",
    "custom_chain": "自定义链：展示如何创建自定义链的实现",
}

# 尝试导入已实现的链模块
_available_chains = []
_import_errors = []

# 尝试导入各种链的实现
for chain_type in PLANNED_CHAINS:
    try:
        # 动态导入模块
        module = __import__(f".{chain_type}", package=__name__, level=1)
        _available_chains.append(chain_type)
    except ImportError as e:
        _import_errors.append((chain_type, str(e)))

def get_available_chains():
    """
    获取当前可用的链类型列表
    
    Returns:
        list: 已成功导入的链类型名称列表
    """
    return _available_chains.copy()

def get_import_errors():
    """
    获取导入失败的链和错误信息
    
    Returns:
        list: 包含 (链类型, 错误信息) 元组的列表
    """
    return _import_errors.copy()

def get_chain_status():
    """
    获取所有链的状态信息
    
    Returns:
        dict: 包含链状态的字典
    """
    status = {}
    
    for chain in PLANNED_CHAINS:
        if chain in _available_chains:
            status[chain] = {
                "available": True,
                "description": CHAIN_DESCRIPTIONS.get(chain, "无描述"),
                "error": None
            }
        else:
            error_info = next(
                (error for name, error in _import_errors if name == chain),
                "链未实现"
            )
            status[chain] = {
                "available": False,
                "description": CHAIN_DESCRIPTIONS.get(chain, "无描述"),
                "error": error_info
            }
    
    return status

def print_chain_status():
    """
    打印所有链的状态信息
    """
    print(f"\n🔗 链模块状态 (chains v{__version__}):")
    print("=" * 60)
    
    status = get_chain_status()
    
    for chain, info in status.items():
        status_icon = "✅" if info["available"] else "❌"
        print(f"{status_icon} {chain:18} - {info['description']}")
        
        if not info["available"] and info["error"] != "链未实现":
            print(f"   错误: {info['error']}")
    
    print(f"\n📊 统计: {len(_available_chains)}/{len(PLANNED_CHAINS)} 个链可用")
    
    if _import_errors:
        print(f"\n⚠️  导入错误: {len(_import_errors)} 个")

def get_chain_examples():
    """
    获取各种链的使用示例
    
    Returns:
        dict: 包含链类型和示例代码的字典
    """
    examples = {
        "llm_chain": """
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# 创建提示模板
prompt = PromptTemplate(
    input_variables=["topic"],
    template="请写一篇关于{topic}的简短介绍。"
)

# 创建 LLM 链
llm_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=prompt
)

# 运行链
result = llm_chain.run(topic="人工智能")
print(result)
        """,
        
        "conversation_chain": """
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# 创建对话链
conversation = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=ConversationBufferMemory(),
    verbose=True
)

# 进行对话
response1 = conversation.predict(input="你好，我是张三")
response2 = conversation.predict(input="我刚才说我叫什么名字？")
        """,
        
        "sequential_chain": """
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# 第一个链：生成大纲
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="为{topic}创建一个详细的大纲。"
)
outline_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=outline_prompt,
    output_key="outline"
)

# 第二个链：根据大纲写文章
article_prompt = PromptTemplate(
    input_variables=["outline"],
    template="根据以下大纲写一篇文章：\\n{outline}"
)
article_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=article_prompt,
    output_key="article"
)

# 创建顺序链
overall_chain = SequentialChain(
    chains=[outline_chain, article_chain],
    input_variables=["topic"],
    output_variables=["outline", "article"],
    verbose=True
)

# 运行链
result = overall_chain({"topic": "机器学习入门"})
        """
    }
    
    return examples

def print_chain_examples():
    """
    打印链的使用示例
    """
    examples = get_chain_examples()
    
    print(f"\n📝 链使用示例:")
    print("=" * 60)
    
    for chain_type, example in examples.items():
        print(f"\n🔗 {chain_type} 示例:")
        print("-" * 40)
        print(example)

# 定义公共接口
__all__ = [
    # 版本信息
    "__version__",
    
    # 状态查询函数
    "get_available_chains",
    "get_import_errors",
    "get_chain_status", 
    "print_chain_status",
    
    # 示例函数
    "get_chain_examples",
    "print_chain_examples",
    
    # 链描述
    "PLANNED_CHAINS",
    "CHAIN_DESCRIPTIONS",
]

# 如果直接运行此模块，显示状态信息和示例
if __name__ == "__main__":
    print_chain_status()
    print_chain_examples()