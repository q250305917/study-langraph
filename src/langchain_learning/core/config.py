"""
LangChain学习项目的配置管理模块

本模块实现了灵活的配置管理系统，支持：
- 多种配置源（环境变量、文件、命令行参数）
- 层级配置和配置继承
- 配置验证和类型转换
- 动态配置重载
- 配置加密和安全存储
"""

import os
import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import argparse

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from .logger import get_logger
from .exceptions import (
    ConfigurationError,
    ValidationError,
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)

T = TypeVar('T')


class ConfigFormat(Enum):
    """配置文件格式枚举"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class ConfigSource(ABC):
    """
    配置源抽象基类
    
    定义了配置源的通用接口，所有具体的配置源都需要实现这个接口。
    支持配置的加载、监听变化和优先级管理。
    """
    
    def __init__(self, priority: int = 0):
        """
        初始化配置源
        
        Args:
            priority: 优先级，数值越大优先级越高
        """
        self.priority = priority
        self._cache: Dict[str, Any] = {}
        self._last_modified: Optional[float] = None
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        加载配置数据
        
        Returns:
            配置字典
            
        Raises:
            ConfigurationError: 配置加载失败
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查配置源是否可用
        
        Returns:
            True如果配置源可用，否则False
        """
        pass
    
    def get_cache(self) -> Dict[str, Any]:
        """获取缓存的配置数据"""
        return self._cache.copy()
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._last_modified = None


class EnvironmentConfigSource(ConfigSource):
    """
    环境变量配置源
    
    从系统环境变量中加载配置，支持前缀过滤和类型转换。
    """
    
    def __init__(self, prefix: str = "LANGCHAIN_", priority: int = 100):
        """
        初始化环境变量配置源
        
        Args:
            prefix: 环境变量前缀，只加载以此前缀开头的变量
            priority: 优先级
        """
        super().__init__(priority)
        self.prefix = prefix.upper()
    
    def load(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        try:
            config = {}
            for key, value in os.environ.items():
                if key.startswith(self.prefix):
                    # 移除前缀并转换为小写
                    clean_key = key[len(self.prefix):].lower()
                    config[clean_key] = self._convert_value(value)
            
            self._cache = config
            logger.debug(f"Loaded {len(config)} config items from environment variables")
            return config
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load environment variables: {str(e)}",
                error_code=ErrorCodes.CONFIG_PARSE_ERROR,
                cause=e
            )
    
    def is_available(self) -> bool:
        """环境变量配置源总是可用的"""
        return True
    
    def _convert_value(self, value: str) -> Any:
        """
        转换环境变量值的类型
        
        支持以下类型转换：
        - bool: "true", "false", "1", "0", "yes", "no"
        - int: 数字字符串
        - float: 浮点数字符串  
        - list: 逗号分隔的字符串
        - dict: JSON格式字符串
        """
        if not value:
            return value
        
        # 布尔值转换
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        if value.lower() in ("false", "0", "no", "off"):
            return False
        
        # 尝试转换为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 尝试解析JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 尝试解析逗号分隔的列表
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        return value


class FileConfigSource(ConfigSource):
    """
    文件配置源
    
    从配置文件中加载配置，支持多种文件格式。
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        format: Optional[ConfigFormat] = None,
        priority: int = 50,
        required: bool = False
    ):
        """
        初始化文件配置源
        
        Args:
            file_path: 配置文件路径
            format: 文件格式，None则根据文件扩展名自动判断
            priority: 优先级
            required: 是否为必需文件，True时文件不存在会抛出异常
        """
        super().__init__(priority)
        self.file_path = Path(file_path)
        self.format = format or self._detect_format()
        self.required = required
    
    def load(self) -> Dict[str, Any]:
        """从文件加载配置"""
        if not self.is_available():
            if self.required:
                raise ConfigurationError(
                    f"Required config file not found: {self.file_path}",
                    error_code=ErrorCodes.CONFIG_FILE_NOT_FOUND,
                    context={"file_path": str(self.file_path)}
                )
            return {}
        
        try:
            # 检查文件是否被修改
            current_modified = self.file_path.stat().st_mtime
            if current_modified == self._last_modified and self._cache:
                return self._cache.copy()
            
            # 读取文件内容
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 根据格式解析内容
            if self.format == ConfigFormat.JSON:
                config = json.loads(content)
            elif self.format == ConfigFormat.YAML:
                config = yaml.safe_load(content)
            elif self.format == ConfigFormat.ENV:
                config = self._parse_env_file(content)
            else:
                raise ConfigurationError(
                    f"Unsupported config format: {self.format}",
                    error_code=ErrorCodes.CONFIG_PARSE_ERROR
                )
            
            self._cache = config
            self._last_modified = current_modified
            logger.debug(f"Loaded config from file: {self.file_path}")
            return config
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(
                f"Failed to parse config file {self.file_path}: {str(e)}",
                error_code=ErrorCodes.CONFIG_PARSE_ERROR,
                context={"file_path": str(self.file_path)},
                cause=e
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config file {self.file_path}: {str(e)}",
                error_code=ErrorCodes.CONFIG_FILE_NOT_FOUND,
                context={"file_path": str(self.file_path)},
                cause=e
            )
    
    def is_available(self) -> bool:
        """检查文件是否存在且可读"""
        return self.file_path.exists() and self.file_path.is_file()
    
    def _detect_format(self) -> ConfigFormat:
        """根据文件扩展名检测格式"""
        suffix = self.file_path.suffix.lower()
        format_map = {
            '.json': ConfigFormat.JSON,
            '.yaml': ConfigFormat.YAML,
            '.yml': ConfigFormat.YAML,
            '.env': ConfigFormat.ENV,
        }
        return format_map.get(suffix, ConfigFormat.JSON)
    
    def _parse_env_file(self, content: str) -> Dict[str, Any]:
        """解析.env文件格式"""
        config = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip().strip('"\'')
        return config


class ArgumentConfigSource(ConfigSource):
    """
    命令行参数配置源
    
    从命令行参数中加载配置，支持argparse参数解析。
    """
    
    def __init__(
        self,
        parser: Optional[argparse.ArgumentParser] = None,
        args: Optional[List[str]] = None,
        priority: int = 200
    ):
        """
        初始化命令行参数配置源
        
        Args:
            parser: argparse解析器，None则创建默认解析器
            args: 参数列表，None则使用sys.argv
            priority: 优先级
        """
        super().__init__(priority)
        self.parser = parser or self._create_default_parser()
        self.args = args
    
    def load(self) -> Dict[str, Any]:
        """从命令行参数加载配置"""
        try:
            parsed_args = self.parser.parse_args(self.args)
            config = {
                key: value for key, value in vars(parsed_args).items()
                if value is not None
            }
            
            self._cache = config
            logger.debug(f"Loaded {len(config)} config items from command line arguments")
            return config
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse command line arguments: {str(e)}",
                error_code=ErrorCodes.CONFIG_PARSE_ERROR,
                cause=e
            )
    
    def is_available(self) -> bool:
        """命令行参数配置源总是可用的"""
        return True
    
    def _create_default_parser(self) -> argparse.ArgumentParser:
        """创建默认的参数解析器"""
        parser = argparse.ArgumentParser(description="LangChain Learning Project")
        parser.add_argument(
            "--config", "-c",
            type=str,
            help="Configuration file path"
        )
        parser.add_argument(
            "--log-level", "-l",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level"
        )
        parser.add_argument(
            "--env", "-e",
            type=str,
            help="Environment name"
        )
        return parser


class ConfigSchema(BaseModel):
    """
    配置模式基类
    
    使用Pydantic进行配置验证和类型转换。
    子类可以继承此类来定义具体的配置模式。
    """
    
    class Config:
        # 允许从环境变量加载配置
        env_prefix = "LANGCHAIN_"
        # 区分大小写
        case_sensitive = False
        # 允许额外字段
        extra = "allow"


class AppConfig(ConfigSchema):
    """
    应用程序主配置
    
    定义了应用程序的核心配置项，包括日志、数据库、API等设置。
    """
    
    # 应用程序设置
    app_name: str = Field(default="langchain-learning", description="应用名称")
    version: str = Field(default="0.1.0", description="应用版本")
    environment: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    
    # 日志设置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")
    
    # API设置
    api_host: str = Field(default="localhost", description="API服务器主机")
    api_port: int = Field(default=8000, description="API服务器端口")
    api_prefix: str = Field(default="/api/v1", description="API路径前缀")
    
    # LLM设置
    default_llm_provider: str = Field(default="openai", description="默认LLM提供商")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API密钥")
    llm_model: str = Field(default="gpt-3.5-turbo", description="默认LLM模型")
    llm_temperature: float = Field(default=0.7, description="LLM温度参数")
    llm_max_tokens: int = Field(default=1000, description="LLM最大令牌数")
    
    # 数据存储设置
    data_dir: str = Field(default="./data", description="数据目录")
    cache_dir: str = Field(default="./cache", description="缓存目录")
    
    @validator('environment')
    def validate_environment(cls, v):
        """验证环境名称"""
        allowed_envs = ['development', 'testing', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @validator('api_port')
    def validate_api_port(cls, v):
        """验证端口号"""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class ConfigLoader:
    """
    配置加载器
    
    管理多个配置源，提供统一的配置访问接口。
    支持配置的层级合并、验证和动态重载。
    """
    
    def __init__(self, schema_class: Type[BaseModel] = AppConfig):
        """
        初始化配置加载器
        
        Args:
            schema_class: 配置模式类，用于验证配置数据
        """
        self.schema_class = schema_class
        self.sources: List[ConfigSource] = []
        self._merged_config: Dict[str, Any] = {}
        self._config_instance: Optional[BaseModel] = None
        
        logger.debug(f"Initialized ConfigLoader with schema: {schema_class.__name__}")
    
    def add_source(self, source: ConfigSource) -> None:
        """
        添加配置源
        
        Args:
            source: 配置源实例
        """
        self.sources.append(source)
        # 按优先级排序，优先级高的在前
        self.sources.sort(key=lambda s: s.priority, reverse=True)
        logger.debug(f"Added config source: {source.__class__.__name__} (priority: {source.priority})")
    
    def add_file_source(
        self,
        file_path: Union[str, Path],
        format: Optional[ConfigFormat] = None,
        priority: int = 50,
        required: bool = False
    ) -> None:
        """添加文件配置源的便捷方法"""
        source = FileConfigSource(file_path, format, priority, required)
        self.add_source(source)
    
    def add_env_source(self, prefix: str = "LANGCHAIN_", priority: int = 100) -> None:
        """添加环境变量配置源的便捷方法"""
        source = EnvironmentConfigSource(prefix, priority)
        self.add_source(source)
    
    def add_args_source(
        self,
        parser: Optional[argparse.ArgumentParser] = None,
        args: Optional[List[str]] = None,
        priority: int = 200
    ) -> None:
        """添加命令行参数配置源的便捷方法"""
        source = ArgumentConfigSource(parser, args, priority)
        self.add_source(source)
    
    def load(self) -> BaseModel:
        """
        加载并合并所有配置源的配置
        
        Returns:
            验证后的配置实例
            
        Raises:
            ConfigurationError: 配置加载或验证失败
        """
        try:
            # 加载所有配置源
            configs = []
            for source in self.sources:
                if source.is_available():
                    try:
                        config = source.load()
                        configs.append((source.priority, config))
                        logger.debug(f"Loaded config from {source.__class__.__name__}")
                    except Exception as e:
                        logger.warning(f"Failed to load config from {source.__class__.__name__}: {e}")
                        # 继续处理其他配置源
                        continue
            
            # 按优先级合并配置
            self._merged_config = self._merge_configs(configs)
            
            # 验证配置
            self._config_instance = self.schema_class(**self._merged_config)
            
            logger.info(f"Successfully loaded configuration with {len(self._merged_config)} items")
            return self._config_instance
            
        except Exception as e:
            if isinstance(e, (ConfigurationError, ValidationError)):
                raise
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        if self._config_instance is None:
            raise ConfigurationError(
                "Configuration not loaded. Call load() first.",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR
            )
        
        try:
            # 支持嵌套键访问，如 "database.host"
            value = self._config_instance
            for part in key.split('.'):
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        except Exception:
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置数据"""
        if self._config_instance is None:
            raise ConfigurationError(
                "Configuration not loaded. Call load() first.",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR
            )
        return self._config_instance.dict()
    
    def reload(self) -> BaseModel:
        """重新加载配置"""
        # 清空所有配置源的缓存
        for source in self.sources:
            source.clear_cache()
        
        # 重新加载
        return self.load()
    
    def _merge_configs(self, configs: List[tuple]) -> Dict[str, Any]:
        """
        合并配置数据
        
        Args:
            configs: 配置数据列表，每个元素是(优先级, 配置字典)的元组
            
        Returns:
            合并后的配置字典
        """
        merged = {}
        
        # 按优先级从低到高合并，高优先级覆盖低优先级
        for priority, config in sorted(configs, key=lambda x: x[0]):
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            override: 覆盖字典
            
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self._deep_merge(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result


# 全局配置加载器实例
_global_loader: Optional[ConfigLoader] = None


def setup_config(
    schema_class: Type[BaseModel] = AppConfig,
    config_files: Optional[List[Union[str, Path]]] = None,
    env_prefix: str = "LANGCHAIN_",
    include_args: bool = True
) -> ConfigLoader:
    """
    设置全局配置加载器
    
    Args:
        schema_class: 配置模式类
        config_files: 配置文件路径列表
        env_prefix: 环境变量前缀
        include_args: 是否包含命令行参数
        
    Returns:
        配置加载器实例
    """
    global _global_loader
    
    loader = ConfigLoader(schema_class)
    
    # 添加文件配置源
    if config_files:
        for file_path in config_files:
            loader.add_file_source(file_path, priority=50)
    else:
        # 添加默认配置文件
        project_root = Path(__file__).parent.parent.parent.parent
        default_files = [
            project_root / "configs" / "default.yaml",
            project_root / "configs" / "local.yaml",
            Path.cwd() / ".env"
        ]
        for file_path in default_files:
            loader.add_file_source(file_path, priority=50, required=False)
    
    # 添加环境变量配置源
    loader.add_env_source(env_prefix, priority=100)
    
    # 添加命令行参数配置源
    if include_args:
        loader.add_args_source(priority=200)
    
    _global_loader = loader
    return loader


def get_config() -> BaseModel:
    """
    获取全局配置实例
    
    Returns:
        配置实例
        
    Raises:
        ConfigurationError: 配置未初始化
    """
    global _global_loader
    
    if _global_loader is None:
        # 使用默认配置初始化
        _global_loader = setup_config()
        _global_loader.load()
    
    if _global_loader._config_instance is None:
        _global_loader.load()
    
    return _global_loader._config_instance


def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取配置值的便捷函数
    
    Args:
        key: 配置键
        default: 默认值
        
    Returns:
        配置值
    """
    global _global_loader
    
    if _global_loader is None:
        _global_loader = setup_config()
        _global_loader.load()
    
    return _global_loader.get(key, default)