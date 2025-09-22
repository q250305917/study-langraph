"""
LangChain学习项目的异常处理模块

本模块定义了项目中使用的所有自定义异常类，提供统一的错误处理机制。
采用层次化的异常设计，便于精确的错误捕获和处理。
"""

from typing import Any, Dict, Optional
import functools
import traceback
from loguru import logger


class LangChainLearningError(Exception):
    """
    LangChain学习项目的基础异常类
    
    所有项目自定义异常的基类，提供统一的异常接口。
    包含错误码、错误消息和额外的上下文信息。
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        初始化异常
        
        Args:
            message: 错误描述信息
            error_code: 错误码，便于程序化处理
            context: 额外的上下文信息
            cause: 导致此异常的原始异常
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        
    def __str__(self) -> str:
        """返回格式化的错误信息"""
        error_info = f"[{self.error_code}] {self.message}"
        if self.context:
            error_info += f" | Context: {self.context}"
        if self.cause:
            error_info += f" | Caused by: {str(self.cause)}"
        return error_info
    
    def to_dict(self) -> Dict[str, Any]:
        """将异常信息转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None
        }


class ConfigurationError(LangChainLearningError):
    """
    配置相关的异常
    
    当配置加载、解析或验证失败时抛出此异常。
    包括环境变量缺失、配置文件格式错误、参数验证失败等情况。
    """
    pass


class ChainExecutionError(LangChainLearningError):
    """
    链式调用执行异常
    
    当链式调用过程中发生错误时抛出，包括：
    - 链的初始化失败
    - 中间步骤执行错误  
    - 输入输出格式不匹配
    - 超时或资源限制
    """
    pass


class LLMError(LangChainLearningError):
    """
    LLM相关异常
    
    与语言模型交互时的各种错误：
    - API调用失败
    - 认证错误
    - 配额超限
    - 模型不可用
    """
    pass


class ToolError(LangChainLearningError):
    """
    工具执行异常
    
    工具注册、发现或执行过程中的错误：
    - 工具注册失败
    - 工具不存在
    - 工具执行失败
    - 参数验证错误
    """
    pass


class ValidationError(LangChainLearningError):
    """
    数据验证异常
    
    输入数据格式、类型或内容验证失败时抛出：
    - 类型检查失败
    - 数据格式不正确
    - 必需字段缺失
    - 值范围超限
    """
    pass


class ResourceError(LangChainLearningError):
    """
    资源访问异常
    
    文件、网络或其他资源访问失败：
    - 文件不存在或权限不足
    - 网络连接失败
    - 内存不足
    - 外部服务不可用
    """
    pass


class TimeoutError(LangChainLearningError):
    """
    超时异常
    
    操作执行超过设定的时间限制：
    - API调用超时
    - 任务执行超时
    - 等待响应超时
    """
    pass


# 错误码定义
class ErrorCodes:
    """错误码常量定义"""
    
    # 配置错误
    CONFIG_FILE_NOT_FOUND = "CFG_001"
    CONFIG_PARSE_ERROR = "CFG_002"
    CONFIG_VALIDATION_ERROR = "CFG_003"
    ENV_VAR_MISSING = "CFG_004"
    
    # 链执行错误
    CHAIN_INIT_ERROR = "CHN_001"
    CHAIN_EXECUTION_ERROR = "CHN_002"
    CHAIN_INPUT_ERROR = "CHN_003"
    CHAIN_OUTPUT_ERROR = "CHN_004"
    
    # LLM错误
    LLM_API_ERROR = "LLM_001"
    LLM_AUTH_ERROR = "LLM_002"
    LLM_QUOTA_ERROR = "LLM_003"
    LLM_MODEL_ERROR = "LLM_004"
    
    # 工具错误
    TOOL_NOT_FOUND = "TOL_001"
    TOOL_EXECUTION_ERROR = "TOL_002"
    TOOL_REGISTRATION_ERROR = "TOL_003"
    TOOL_VALIDATION_ERROR = "TOL_004"
    
    # 验证错误
    VALIDATION_TYPE_ERROR = "VAL_001"
    VALIDATION_FORMAT_ERROR = "VAL_002"
    VALIDATION_REQUIRED_ERROR = "VAL_003"
    VALIDATION_RANGE_ERROR = "VAL_004"
    
    # 资源错误
    FILE_NOT_FOUND = "RES_001"
    PERMISSION_DENIED = "RES_002"
    NETWORK_ERROR = "RES_003"
    MEMORY_ERROR = "RES_004"
    
    # 超时错误
    API_TIMEOUT = "TIM_001"
    EXECUTION_TIMEOUT = "TIM_002"
    RESPONSE_TIMEOUT = "TIM_003"


def exception_handler(
    reraise: bool = True,
    default_return: Any = None,
    log_level: str = "error"
) -> Any:
    """
    异常处理装饰器
    
    用于统一处理函数执行过程中的异常，提供日志记录和可选的重新抛出。
    
    Args:
        reraise: 是否重新抛出异常，默认True
        default_return: 捕获异常时的默认返回值
        log_level: 日志级别，默认为error
        
    Returns:
        装饰器函数
        
    Usage:
        @exception_handler(reraise=False, default_return={})
        def some_function():
            # 可能抛出异常的代码
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LangChainLearningError as e:
                # 记录自定义异常
                getattr(logger, log_level)(
                    f"Function {func.__name__} failed: {e}"
                )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # 记录其他异常并包装为自定义异常
                error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
                getattr(logger, log_level)(f"{error_msg}\n{traceback.format_exc()}")
                
                if reraise:
                    # 包装为自定义异常并重新抛出
                    raise LangChainLearningError(
                        message=error_msg,
                        error_code="UNEXPECTED_ERROR",
                        cause=e
                    )
                return default_return
        return wrapper
    return decorator


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    异常重试装饰器
    
    当函数执行失败时自动重试，支持指数退避策略。
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 退避因子，每次重试延迟时间的倍数
        exceptions: 需要重试的异常类型元组
        
    Returns:
        装饰器函数
        
    Usage:
        @retry_on_exception(max_retries=3, delay=1.0, exceptions=(LLMError,))
        def call_llm_api():
            # 可能需要重试的API调用
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # 最后一次尝试失败，抛出异常
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{max_retries} retries: {e}"
                        )
                        raise
                    
                    # 记录重试信息
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}), "
                        f"retrying in {current_delay:.1f}s: {e}"
                    )
                    
                    # 等待后重试
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    
            # 理论上不会到达这里
            raise last_exception
            
        return wrapper
    return decorator


def validate_input(**validations) -> Any:
    """
    输入验证装饰器
    
    在函数执行前验证输入参数，支持类型检查、值范围检查等。
    
    Args:
        **validations: 验证规则字典，键为参数名，值为验证函数
        
    Returns:
        装饰器函数
        
    Usage:
        @validate_input(
            name=lambda x: isinstance(x, str) and len(x) > 0,
            age=lambda x: isinstance(x, int) and 0 <= x <= 150
        )
        def create_person(name: str, age: int):
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            
            # 获取函数参数信息
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证每个参数
            for param_name, validation_func in validations.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validation_func(value):
                            raise ValidationError(
                                f"Validation failed for parameter '{param_name}' "
                                f"with value: {value}",
                                error_code=ErrorCodes.VALIDATION_TYPE_ERROR,
                                context={"parameter": param_name, "value": value}
                            )
                    except Exception as e:
                        if isinstance(e, ValidationError):
                            raise
                        raise ValidationError(
                            f"Validation error for parameter '{param_name}': {str(e)}",
                            error_code=ErrorCodes.VALIDATION_TYPE_ERROR,
                            context={"parameter": param_name, "value": value},
                            cause=e
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator