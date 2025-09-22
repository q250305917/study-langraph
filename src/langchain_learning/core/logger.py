"""
LangChain学习项目的日志系统模块

本模块基于loguru库实现了功能丰富的日志系统，支持：
- 多级别日志记录（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- 文件输出和控制台输出
- 日志格式化和彩色输出
- 日志轮转和归档
- 结构化日志记录
- 性能监控和统计
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import json
import functools
import time

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler


class LoggerConfig:
    """日志配置类，管理日志系统的各种设置"""
    
    def __init__(
        self,
        level: str = "INFO",
        log_dir: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        file_output: bool = True,
        json_format: bool = False,
        colorize: bool = True,
        rotation: str = "1 day",
        retention: str = "30 days",
        compression: str = "gz"
    ):
        """
        初始化日志配置
        
        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录，None则使用项目根目录下的logs
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            json_format: 是否使用JSON格式（便于日志分析）
            colorize: 是否使用彩色输出
            rotation: 日志轮转策略，如"1 day", "100 MB"
            retention: 日志保留时间，如"30 days"
            compression: 压缩格式，如"gz", "bz2"
        """
        self.level = level.upper()
        
        # 设置日志目录
        if log_dir is None:
            # 查找项目根目录（包含pyproject.toml的目录）
            current_path = Path(__file__).parent
            while current_path.parent != current_path:
                if (current_path / "pyproject.toml").exists():
                    self.log_dir = current_path / "logs"
                    break
                current_path = current_path.parent
            else:
                # 如果找不到项目根目录，使用当前目录
                self.log_dir = Path.cwd() / "logs"
        else:
            self.log_dir = Path(log_dir)
            
        self.console_output = console_output
        self.file_output = file_output
        self.json_format = json_format
        self.colorize = colorize
        self.rotation = rotation
        self.retention = retention
        self.compression = compression
        
        # 确保日志目录存在
        if self.file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)


class StructuredLogger:
    """
    结构化日志记录器
    
    提供结构化的日志记录功能，便于后续的日志分析和监控。
    支持添加上下文信息、请求追踪、性能监控等功能。
    """
    
    def __init__(self, config: LoggerConfig):
        """
        初始化结构化日志记录器
        
        Args:
            config: 日志配置对象
        """
        self.config = config
        self.console = Console()
        self._setup_logger()
        
    def _setup_logger(self) -> None:
        """配置loguru日志系统"""
        # 移除默认handler
        logger.remove()
        
        # 控制台输出配置
        if self.config.console_output:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            
            logger.add(
                sys.stderr,
                format=console_format,
                level=self.config.level,
                colorize=self.config.colorize,
                backtrace=True,
                diagnose=True
            )
        
        # 文件输出配置
        if self.config.file_output:
            # 普通日志文件
            if self.config.json_format:
                file_format = self._json_formatter
            else:
                file_format = (
                    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "{extra} | "
                    "{message}"
                )
            
            logger.add(
                self.config.log_dir / "app.log",
                format=file_format,
                level=self.config.level,
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                encoding="utf-8",
                backtrace=True,
                diagnose=True
            )
            
            # 错误日志单独记录
            logger.add(
                self.config.log_dir / "error.log",
                format=file_format,
                level="ERROR",
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                encoding="utf-8",
                backtrace=True,
                diagnose=True
            )
    
    def _json_formatter(self, record: Dict[str, Any]) -> str:
        """JSON格式化器"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "extra": record.get("extra", {})
        }
        
        # 添加异常信息
        if record.get("exception"):
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_entry, ensure_ascii=False)
    
    def get_logger(self, name: str) -> Any:
        """
        获取带名称的日志记录器
        
        Args:
            name: 日志记录器名称，通常使用模块名
            
        Returns:
            配置好的日志记录器实例
        """
        return logger.bind(name=name)


# 全局日志配置
_global_config: Optional[LoggerConfig] = None
_global_logger: Optional[StructuredLogger] = None


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    colorize: bool = True,
    rotation: str = "1 day",
    retention: str = "30 days",
    compression: str = "gz"
) -> None:
    """
    设置全局日志配置
    
    这是项目中配置日志系统的主要接口，应该在应用启动时调用。
    
    Args:
        level: 日志级别
        log_dir: 日志文件目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        json_format: 是否使用JSON格式
        colorize: 是否使用彩色输出
        rotation: 日志轮转策略
        retention: 日志保留时间
        compression: 压缩格式
    """
    global _global_config, _global_logger
    
    _global_config = LoggerConfig(
        level=level,
        log_dir=log_dir,
        console_output=console_output,
        file_output=file_output,
        json_format=json_format,
        colorize=colorize,
        rotation=rotation,
        retention=retention,
        compression=compression
    )
    
    _global_logger = StructuredLogger(_global_config)


def get_logger(name: str = "langchain_learning") -> Any:
    """
    获取日志记录器实例
    
    这是项目中获取日志记录器的标准方式。如果全局日志配置尚未设置，
    将使用默认配置进行初始化。
    
    Args:
        name: 日志记录器名称，建议使用模块的__name__
        
    Returns:
        配置好的日志记录器实例
        
    Usage:
        from langchain_learning.core.logger import get_logger
        
        logger = get_logger(__name__)
        logger.info("This is an info message")
        logger.error("This is an error message")
    """
    global _global_logger
    
    if _global_logger is None:
        # 使用默认配置初始化
        setup_logging()
    
    return _global_logger.get_logger(name)


def log_function_call(
    include_args: bool = True,
    include_result: bool = False,
    level: str = "DEBUG"
) -> Any:
    """
    函数调用日志装饰器
    
    自动记录函数的调用信息，包括参数、返回值和执行时间。
    适用于调试和性能监控。
    
    Args:
        include_args: 是否记录函数参数
        include_result: 是否记录返回值
        level: 日志级别
        
    Returns:
        装饰器函数
        
    Usage:
        @log_function_call(include_args=True, include_result=True)
        def my_function(arg1, arg2):
            return arg1 + arg2
    """
    def decorator(func):
        func_logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # 记录函数调用开始
            log_data = {"function": func.__name__}
            if include_args:
                log_data["args"] = args
                log_data["kwargs"] = kwargs
            
            func_logger.bind(**log_data).log(
                level, f"Calling function {func.__name__}"
            )
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录执行成功
                execution_time = time.time() - start_time
                success_data = {
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s",
                    "status": "success"
                }
                
                if include_result:
                    success_data["result"] = result
                
                func_logger.bind(**success_data).log(
                    level, f"Function {func.__name__} completed successfully"
                )
                
                return result
                
            except Exception as e:
                # 记录执行失败
                execution_time = time.time() - start_time
                error_data = {
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s",
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                
                func_logger.bind(**error_data).error(
                    f"Function {func.__name__} failed: {e}"
                )
                raise
        
        return wrapper
    return decorator


def log_performance(threshold: float = 1.0, level: str = "WARNING") -> Any:
    """
    性能监控装饰器
    
    监控函数执行时间，当超过阈值时记录警告日志。
    用于识别性能瓶颈和慢查询。
    
    Args:
        threshold: 时间阈值（秒），超过此值将记录日志
        level: 日志级别
        
    Returns:
        装饰器函数
        
    Usage:
        @log_performance(threshold=2.0, level="WARNING")
        def slow_function():
            time.sleep(3)  # 模拟慢操作
    """
    def decorator(func):
        func_logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold:
                perf_data = {
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s",
                    "threshold": f"{threshold:.3f}s",
                    "performance_issue": True
                }
                
                func_logger.bind(**perf_data).log(
                    level, 
                    f"Function {func.__name__} took {execution_time:.3f}s "
                    f"(threshold: {threshold:.3f}s)"
                )
            
            return result
        return wrapper
    return decorator


def log_context(**context_data) -> Any:
    """
    添加上下文信息的装饰器
    
    为函数调用添加额外的上下文信息，便于日志分析和问题定位。
    
    Args:
        **context_data: 上下文数据键值对
        
    Returns:
        装饰器函数
        
    Usage:
        @log_context(component="config_loader", operation="load")
        def load_config():
            pass
    """
    def decorator(func):
        func_logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 绑定上下文数据到日志记录器
            context_logger = func_logger.bind(**context_data)
            
            # 临时替换模块中的logger（如果存在）
            original_logger = getattr(func.__globals__.get('logger'), '_core', None)
            if 'logger' in func.__globals__:
                func.__globals__['logger'] = context_logger
            
            try:
                return func(*args, **kwargs)
            finally:
                # 恢复原始logger
                if original_logger and 'logger' in func.__globals__:
                    func.__globals__['logger'] = get_logger(func.__module__)
        
        return wrapper
    return decorator


# 初始化默认日志配置
if _global_logger is None:
    setup_logging()