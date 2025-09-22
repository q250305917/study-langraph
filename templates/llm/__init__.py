"""
LLM模板模块

本模块提供了各种大语言模型的接入模板，包括：
- OpenAI模型 (GPT-3.5, GPT-4系列)
- Anthropic模型 (Claude系列)
- 本地模型 (Ollama, LlamaCpp等)
- 多模型对比和切换

所有LLM模板都基于统一的接口设计，支持：
1. 流式输出和批量输出
2. 异步和同步调用
3. 错误处理和重试机制
4. 性能监控和统计
5. 参数验证和配置管理
6. 缓存和优化机制

使用示例：
    from templates.llm import OpenAITemplate
    
    # 创建OpenAI模板实例
    template = OpenAITemplate()
    
    # 配置模板参数
    template.setup(
        api_key="your-api-key",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # 执行调用
    result = template.run("介绍一下LangChain")
    print(result)
"""

from .openai_template import OpenAITemplate
from .anthropic_template import AnthropicTemplate  
from .local_llm_template import LocalLLMTemplate
from .multi_model_template import MultiModelTemplate

# 版本信息
__version__ = "1.0.0"

# 导出的类列表
__all__ = [
    "OpenAITemplate",
    "AnthropicTemplate", 
    "LocalLLMTemplate",
    "MultiModelTemplate"
]

# 支持的模型类型
SUPPORTED_MODELS = {
    "openai": [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k", 
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-32k",
        "text-davinci-003",
        "text-davinci-002"
    ],
    "anthropic": [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229", 
        "claude-3-opus-20240229",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2"
    ],
    "local": [
        "llama2",
        "llama2:7b",
        "llama2:13b", 
        "llama2:70b",
        "codellama",
        "mistral",
        "vicuna",
        "alpaca"
    ]
}

# 模板工厂函数
def create_llm_template(model_type: str, model_name: str = None):
    """
    创建LLM模板实例的工厂函数
    
    Args:
        model_type: 模型类型 ("openai", "anthropic", "local", "multi")
        model_name: 具体的模型名称 (可选)
        
    Returns:
        相应的LLM模板实例
        
    Raises:
        ValueError: 不支持的模型类型
    """
    if model_type.lower() == "openai":
        return OpenAITemplate()
    elif model_type.lower() == "anthropic":
        return AnthropicTemplate()
    elif model_type.lower() == "local":
        return LocalLLMTemplate()
    elif model_type.lower() == "multi":
        return MultiModelTemplate()
    else:
        raise ValueError(
            f"不支持的模型类型: {model_type}. "
            f"支持的类型: openai, anthropic, local, multi"
        )

def get_supported_models(model_type: str = None):
    """
    获取支持的模型列表
    
    Args:
        model_type: 模型类型，None表示获取所有类型
        
    Returns:
        支持的模型列表或字典
    """
    if model_type is None:
        return SUPPORTED_MODELS
    elif model_type.lower() in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_type.lower()]
    else:
        return []

def is_model_supported(model_type: str, model_name: str = None):
    """
    检查指定模型是否被支持
    
    Args:
        model_type: 模型类型
        model_name: 模型名称
        
    Returns:
        bool: 是否支持该模型
    """
    if model_type.lower() not in SUPPORTED_MODELS:
        return False
    
    if model_name is None:
        return True
        
    return model_name in SUPPORTED_MODELS[model_type.lower()]