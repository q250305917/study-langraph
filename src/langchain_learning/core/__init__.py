"""
LangChain学习项目的核心模块

本模块包含项目的核心组件，为整个项目提供基础功能：
- 异常处理和错误管理 (exceptions)
- 日志系统和监控 (logger)
- 配置管理和加载 (config)
- 链式调用基础框架 (base)
- LLM工厂和实例管理 (llm_factory)
- 提示词模板管理 (prompt_manager)
- 工具注册和调用 (tools)
- 示例运行和管理 (example_runner)
"""

# 导入核心异常类
from .exceptions import (
    LangChainLearningError,
    ConfigurationError,
    ChainExecutionError,
    LLMError,
    ToolError,
    ValidationError,
    ResourceError,
    TimeoutError,
    ErrorCodes,
    exception_handler,
    retry_on_exception,
    validate_input
)

# 导入日志系统
from .logger import (
    setup_logging,
    get_logger,
    log_function_call,
    log_performance,
    log_context
)

# 导入配置管理
from .config import (
    ConfigLoader,
    ConfigSource,
    EnvironmentConfigSource,
    FileConfigSource,
    ArgumentConfigSource,
    AppConfig,
    setup_config,
    get_config,
    get_config_value
)

# 导入基础链类
from .base import (
    BaseChain,
    ChainComposer,
    ChainInput,
    ChainOutput,
    ChainContext,
    ChainMiddleware,
    ChainStatus,
    ChainType,
    ChainMetadata,
    LoggingMiddleware,
    MetricsMiddleware
)

# 导入LLM工厂
from .llm_factory import (
    LLMFactory,
    LLMInstance,
    LLMConfig,
    LLMProvider,
    ModelInfo,
    ModelCapability,
    get_llm_factory,
    create_llm,
    create_llm_from_config
)

# 导入提示词管理
from .prompt_manager import (
    PromptManager,
    PromptTemplate,
    PromptType,
    TemplateFormat,
    TemplateRegistry,
    get_prompt_manager,
    create_template,
    render_template,
    get_template
)

# 导入工具管理
from .tools import (
    ToolManager,
    BaseTool,
    FunctionTool,
    ToolType,
    ToolStatus,
    ToolRegistry,
    ToolInput,
    ToolOutput,
    ToolSchema,
    ToolMetadata,
    get_tool_manager,
    register_tool,
    register_function,
    call_tool
)

# 导入示例运行器
from .example_runner import (
    ExampleRunner,
    BaseExample,
    CodeExample,
    ChainExample,
    ExampleType,
    ExampleStatus,
    ExampleInput,
    ExampleOutput,
    ExampleMetadata,
    ExampleRegistry,
    get_example_runner,
    run_example,
    list_examples,
    register_example
)

# 版本信息
__version__ = "1.0.0"

# 导出的公共接口
__all__ = [
    # 异常处理
    "LangChainLearningError",
    "ConfigurationError",
    "ChainExecutionError", 
    "LLMError",
    "ToolError",
    "ValidationError",
    "ResourceError",
    "TimeoutError",
    "ErrorCodes",
    "exception_handler",
    "retry_on_exception",
    "validate_input",
    
    # 日志系统
    "setup_logging",
    "get_logger",
    "log_function_call",
    "log_performance",
    "log_context",
    
    # 配置管理
    "ConfigLoader",
    "ConfigSource",
    "EnvironmentConfigSource",
    "FileConfigSource", 
    "ArgumentConfigSource",
    "AppConfig",
    "setup_config",
    "get_config",
    "get_config_value",
    
    # 基础链类
    "BaseChain",
    "ChainComposer",
    "ChainInput",
    "ChainOutput",
    "ChainContext",
    "ChainMiddleware",
    "ChainStatus",
    "ChainType",
    "ChainMetadata",
    "LoggingMiddleware",
    "MetricsMiddleware",
    
    # LLM工厂
    "LLMFactory",
    "LLMInstance",
    "LLMConfig",
    "LLMProvider",
    "ModelInfo",
    "ModelCapability",
    "get_llm_factory",
    "create_llm",
    "create_llm_from_config",
    
    # 提示词管理
    "PromptManager",
    "PromptTemplate",
    "PromptType",
    "TemplateFormat",
    "TemplateRegistry",
    "get_prompt_manager",
    "create_template",
    "render_template",
    "get_template",
    
    # 工具管理
    "ToolManager",
    "BaseTool",
    "FunctionTool",
    "ToolType",
    "ToolStatus",
    "ToolRegistry",
    "ToolInput",
    "ToolOutput",
    "ToolSchema",
    "ToolMetadata",
    "get_tool_manager",
    "register_tool",
    "register_function",
    "call_tool",
    
    # 示例运行器
    "ExampleRunner",
    "BaseExample",
    "CodeExample",
    "ChainExample",
    "ExampleType",
    "ExampleStatus",
    "ExampleInput",
    "ExampleOutput",
    "ExampleMetadata",
    "ExampleRegistry",
    "get_example_runner",
    "run_example",
    "list_examples",
    "register_example",
]