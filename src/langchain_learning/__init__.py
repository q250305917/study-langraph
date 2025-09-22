"""
LangChain 学习项目

这是一个专门用于学习 LangChain 框架和相关技术栈的项目。
本项目包含了 LangChain 的各种核心概念、组件使用示例和实践项目。

主要模块：
- core: 核心配置、工具函数和基础类
- chains: 各种链（Chain）的实现和示例
- agents: 智能代理的实现和使用案例
- tools: 自定义工具和工具集成
- utils: 通用工具函数和辅助类

作者: LangChain 学习者
版本: 0.1.0
许可: MIT License
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "LangChain Learner"
__email__ = "learner@example.com"
__license__ = "MIT"

# 导入核心模块，方便外部直接使用
try:
    from .core.config import settings, get_settings
    from .core.logger import get_logger
    from .core.exceptions import (
        LangChainLearningError,
        ConfigurationError,
        ModelError,
        ChainError,
        AgentError,
        ToolError,
    )
    
    # 设置默认日志记录器
    logger = get_logger(__name__)
    logger.info(f"LangChain Learning 项目已初始化，版本: {__version__}")
    
except ImportError as e:
    # 如果核心模块还没有创建，提供一个简单的日志记录器
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"部分核心模块尚未创建，当前版本: {__version__}")

# 定义包的公共接口
__all__ = [
    # 版本和元信息
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    
    # 核心组件（如果可用）
    "settings",
    "get_settings", 
    "get_logger",
    "logger",
    
    # 异常类（如果可用）
    "LangChainLearningError",
    "ConfigurationError",
    "ModelError", 
    "ChainError",
    "AgentError",
    "ToolError",
]

# 项目元数据
PROJECT_METADATA = {
    "name": "langchain-learning",
    "version": __version__,
    "description": "LangChain学习项目：深入学习LangChain框架和相关技术栈",
    "author": __author__,
    "license": __license__,
    "python_requires": ">=3.9",
    "keywords": ["langchain", "llm", "ai", "learning", "python"],
    "github": "https://github.com/your-username/langchain-learning",
    "documentation": "https://your-username.github.io/langchain-learning",
}

def get_project_info():
    """
    获取项目信息
    
    Returns:
        dict: 包含项目元数据的字典
    """
    return PROJECT_METADATA.copy()

def print_welcome():
    """
    打印欢迎信息和项目简介
    """
    welcome_message = f"""
    🚀 欢迎使用 LangChain 学习项目！
    
    📋 项目信息：
    - 名称: {PROJECT_METADATA['name']}
    - 版本: {PROJECT_METADATA['version']}
    - 描述: {PROJECT_METADATA['description']}
    - 作者: {PROJECT_METADATA['author']}
    
    📁 主要模块：
    - core: 核心配置和基础组件
    - chains: LangChain 链的实现和示例
    - agents: 智能代理的使用案例
    - tools: 自定义工具和集成
    - utils: 通用工具和辅助函数
    
    🔗 相关链接：
    - GitHub: {PROJECT_METADATA['github']}
    - 文档: {PROJECT_METADATA['documentation']}
    
    💡 快速开始：
    1. 查看 examples/ 目录中的示例代码
    2. 阅读 docs/ 目录中的文档
    3. 运行 notebooks/ 中的 Jupyter 笔记本
    
    Happy Learning! 🎓
    """
    print(welcome_message)

# 如果直接运行此模块，显示欢迎信息
if __name__ == "__main__":
    print_welcome()