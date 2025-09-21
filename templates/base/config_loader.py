"""
模板配置加载器模块

本模块提供了灵活强大的配置加载功能，专门用于模板系统的配置管理。
支持多种配置源、配置合并、环境变量替换和动态配置重载。

核心特性：
1. 多配置源：支持YAML文件、JSON文件、环境变量、字典等多种配置源
2. 配置合并：按优先级智能合并多个配置源的数据
3. 环境变量：支持环境变量替换和默认值设置
4. 配置继承：支持配置文件之间的继承关系
5. 动态重载：监控配置文件变化并自动重载
6. 配置验证：内置配置结构验证和错误报告
7. 缓存机制：提高配置加载性能
8. 安全加载：防止配置注入和恶意代码执行

设计原理：
- 策略模式：不同的配置源使用不同的加载策略
- 组合模式：组合多个配置源形成完整的配置
- 观察者模式：监控配置变化并通知相关组件
- 工厂模式：根据配置类型创建相应的加载器
"""

import os
import json
import yaml
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .template_base import TemplateConfig, ParameterSchema, TemplateType
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ConfigurationError, 
    ValidationError,
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class ConfigFormat(Enum):
    """配置文件格式枚举"""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"
    DICT = "dict"
    AUTO = "auto"  # 自动检测格式


class ConfigSourceType(Enum):
    """配置源类型枚举"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DICT = "dict"
    URL = "url"
    DATABASE = "database"


@dataclass
class ConfigSource:
    """
    配置源描述
    
    定义配置源的基本信息和加载参数。
    """
    source_type: ConfigSourceType    # 配置源类型
    source_path: str                 # 配置源路径（文件路径、URL、数据库连接等）
    format: ConfigFormat = ConfigFormat.AUTO  # 配置格式
    priority: int = 0                # 优先级，数值越大优先级越高
    required: bool = False           # 是否为必需配置源
    encoding: str = "utf-8"          # 文件编码
    watch: bool = False              # 是否监控文件变化
    cache_ttl: int = 300            # 缓存生存时间（秒）
    
    # 加载选项
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.format == ConfigFormat.AUTO:
            self.format = self._detect_format()
    
    def _detect_format(self) -> ConfigFormat:
        """自动检测配置格式"""
        if self.source_type == ConfigSourceType.FILE:
            path = Path(self.source_path)
            suffix = path.suffix.lower()
            if suffix in ['.yaml', '.yml']:
                return ConfigFormat.YAML
            elif suffix == '.json':
                return ConfigFormat.JSON
            elif suffix == '.env':
                return ConfigFormat.ENV
        elif self.source_type == ConfigSourceType.ENVIRONMENT:
            return ConfigFormat.ENV
        
        # 默认使用YAML格式
        return ConfigFormat.YAML


class ConfigCache:
    """
    配置缓存管理器
    
    管理配置数据的缓存，提高加载性能并减少文件I/O操作。
    """
    
    def __init__(self):
        """初始化配置缓存"""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的配置数据
        
        Args:
            key: 缓存键
            
        Returns:
            配置数据，如果缓存不存在或已过期则返回None
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            # 检查是否过期
            current_time = time.time()
            if key in self._timestamps and key in self._ttl:
                if current_time - self._timestamps[key] > self._ttl[key]:
                    # 缓存已过期，清除
                    self._remove(key)
                    return None
            
            return self._cache[key].copy()
    
    def set(self, key: str, data: Dict[str, Any], ttl: int = 300) -> None:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            data: 配置数据
            ttl: 生存时间（秒）
        """
        with self._lock:
            self._cache[key] = data.copy()
            self._timestamps[key] = time.time()
            self._ttl[key] = ttl
    
    def remove(self, key: str) -> None:
        """删除缓存项"""
        with self._lock:
            self._remove(key)
    
    def _remove(self, key: str) -> None:
        """内部删除方法（不加锁）"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttl.pop(key, None)
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttl.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                "total_entries": len(self._cache),
                "cache_keys": list(self._cache.keys()),
                "memory_usage": sum(len(str(data)) for data in self._cache.values())
            }


class ConfigWatcher(FileSystemEventHandler):
    """
    配置文件监控器
    
    监控配置文件的变化并触发重载。
    """
    
    def __init__(self, reload_callback: Callable[[str], None]):
        """
        初始化配置监控器
        
        Args:
            reload_callback: 重载回调函数
        """
        self.reload_callback = reload_callback
        self.watched_files: Set[str] = set()
    
    def add_file(self, file_path: str) -> None:
        """添加监控文件"""
        self.watched_files.add(os.path.abspath(file_path))
    
    def on_modified(self, event) -> None:
        """文件修改事件处理"""
        if not event.is_directory:
            file_path = os.path.abspath(event.src_path)
            if file_path in self.watched_files:
                logger.info(f"Config file changed: {file_path}")
                try:
                    self.reload_callback(file_path)
                except Exception as e:
                    logger.error(f"Failed to reload config file {file_path}: {str(e)}")


class BaseConfigLoader(ABC):
    """
    配置加载器抽象基类
    
    定义配置加载器的通用接口，所有具体的加载器都应该继承此类。
    """
    
    def __init__(self, cache: Optional[ConfigCache] = None):
        """
        初始化配置加载器
        
        Args:
            cache: 配置缓存实例
        """
        self.cache = cache or ConfigCache()
    
    @abstractmethod
    def load(self, source: ConfigSource) -> Dict[str, Any]:
        """
        加载配置数据
        
        Args:
            source: 配置源描述
            
        Returns:
            配置数据字典
            
        Raises:
            ConfigurationError: 配置加载失败
        """
        pass
    
    def _get_cache_key(self, source: ConfigSource) -> str:
        """生成缓存键"""
        return f"{source.source_type.value}:{source.source_path}:{source.format.value}"
    
    def _process_environment_variables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理环境变量替换
        
        支持${VAR_NAME}和${VAR_NAME:default_value}格式的环境变量引用。
        
        Args:
            data: 原始配置数据
            
        Returns:
            处理后的配置数据
        """
        def replace_env_vars(value):
            if isinstance(value, str):
                # 匹配${VAR_NAME}或${VAR_NAME:default}格式
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replace_match(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replace_match, value)
            elif isinstance(value, dict):
                return {k: replace_env_vars(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_env_vars(item) for item in value]
            else:
                return value
        
        return replace_env_vars(data)


class FileConfigLoader(BaseConfigLoader):
    """
    文件配置加载器
    
    从配置文件中加载配置数据，支持YAML、JSON等格式。
    """
    
    def load(self, source: ConfigSource) -> Dict[str, Any]:
        """从文件加载配置"""
        cache_key = self._get_cache_key(source)
        
        # 检查缓存
        if source.cache_ttl > 0:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded config from cache: {source.source_path}")
                return cached_data
        
        file_path = Path(source.source_path)
        
        # 检查文件是否存在
        if not file_path.exists():
            if source.required:
                raise ConfigurationError(
                    f"Required config file not found: {source.source_path}",
                    error_code=ErrorCodes.CONFIG_FILE_NOT_FOUND
                )
            logger.warning(f"Config file not found: {source.source_path}")
            return {}
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding=source.encoding) as f:
                content = f.read()
            
            # 根据格式解析内容
            if source.format == ConfigFormat.YAML:
                data = yaml.safe_load(content) or {}
            elif source.format == ConfigFormat.JSON:
                data = json.loads(content)
            elif source.format == ConfigFormat.ENV:
                data = self._parse_env_file(content)
            else:
                raise ConfigurationError(
                    f"Unsupported config format: {source.format.value}",
                    error_code=ErrorCodes.CONFIG_PARSE_ERROR
                )
            
            # 处理环境变量替换
            data = self._process_environment_variables(data)
            
            # 处理配置继承
            if 'extends' in data:
                data = self._handle_inheritance(data, file_path.parent)
            
            # 缓存结果
            if source.cache_ttl > 0:
                self.cache.set(cache_key, data, source.cache_ttl)
            
            logger.debug(f"Loaded config from file: {source.source_path}")
            return data
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(
                f"Failed to parse config file {source.source_path}: {str(e)}",
                error_code=ErrorCodes.CONFIG_PARSE_ERROR,
                cause=e
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config file {source.source_path}: {str(e)}",
                error_code=ErrorCodes.CONFIG_FILE_NOT_FOUND,
                cause=e
            )
    
    def _parse_env_file(self, content: str) -> Dict[str, Any]:
        """解析.env文件格式"""
        data = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # 尝试类型转换
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                data[key] = value
        return data
    
    def _handle_inheritance(self, data: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """
        处理配置继承
        
        Args:
            data: 当前配置数据
            base_dir: 基础目录，用于解析相对路径
            
        Returns:
            合并后的配置数据
        """
        extends = data.pop('extends', None)
        if not extends:
            return data
        
        if isinstance(extends, str):
            extends = [extends]
        
        merged_data = {}
        
        # 按顺序加载父配置
        for parent_config in extends:
            parent_path = base_dir / parent_config
            if parent_path.exists():
                try:
                    parent_source = ConfigSource(
                        source_type=ConfigSourceType.FILE,
                        source_path=str(parent_path),
                        format=ConfigFormat.AUTO
                    )
                    parent_data = self.load(parent_source)
                    merged_data = self._deep_merge(merged_data, parent_data)
                except Exception as e:
                    logger.warning(f"Failed to load parent config {parent_config}: {str(e)}")
        
        # 当前配置覆盖父配置
        return self._deep_merge(merged_data, data)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class EnvironmentConfigLoader(BaseConfigLoader):
    """
    环境变量配置加载器
    
    从环境变量中加载配置数据。
    """
    
    def load(self, source: ConfigSource) -> Dict[str, Any]:
        """从环境变量加载配置"""
        prefix = source.options.get('prefix', 'TEMPLATE_')
        prefix = prefix.upper()
        
        cache_key = f"env:{prefix}"
        
        # 检查缓存
        if source.cache_ttl > 0:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Loaded environment config from cache: {prefix}")
                return cached_data
        
        data = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = key[len(prefix):].lower()
                
                # 处理嵌套键（使用双下划线分隔）
                if '__' in config_key:
                    self._set_nested_value(data, config_key.split('__'), self._convert_env_value(value))
                else:
                    data[config_key] = self._convert_env_value(value)
        
        # 缓存结果
        if source.cache_ttl > 0:
            self.cache.set(cache_key, data, source.cache_ttl)
        
        logger.debug(f"Loaded {len(data)} config items from environment variables")
        return data
    
    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值的类型"""
        if not value:
            return value
        
        # 布尔值转换
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        if value.lower() in ("false", "0", "no", "off"):
            return False
        
        # 数字转换
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON转换
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 逗号分隔的列表
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], keys: List[str], value: Any) -> None:
        """设置嵌套字典值"""
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value


class DictConfigLoader(BaseConfigLoader):
    """
    字典配置加载器
    
    从Python字典中加载配置数据。
    """
    
    def load(self, source: ConfigSource) -> Dict[str, Any]:
        """从字典加载配置"""
        # 对于字典源，source_path实际是字典数据
        if isinstance(source.source_path, dict):
            data = source.source_path.copy()
        else:
            # 尝试从模块导入
            try:
                import importlib
                module_name, attr_name = source.source_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                data = getattr(module, attr_name)
                if not isinstance(data, dict):
                    raise ConfigurationError(f"Config object must be a dictionary: {source.source_path}")
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load config from {source.source_path}: {str(e)}",
                    error_code=ErrorCodes.CONFIG_FILE_NOT_FOUND,
                    cause=e
                )
        
        # 处理环境变量替换
        data = self._process_environment_variables(data)
        
        logger.debug(f"Loaded config from dictionary: {len(data)} items")
        return data


class ConfigLoader:
    """
    模板配置加载器主类
    
    统一管理多个配置源，提供完整的配置加载、合并和验证功能。
    支持配置继承、环境变量替换、文件监控等高级特性。
    
    核心功能：
    1. 多源加载：支持从多种配置源加载数据
    2. 智能合并：按优先级合并配置数据
    3. 配置验证：验证配置结构和数据有效性
    4. 动态重载：监控配置变化并自动重载
    5. 缓存机制：提高配置加载性能
    6. 错误处理：详细的错误报告和恢复机制
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        初始化配置加载器
        
        Args:
            cache_enabled: 是否启用配置缓存
        """
        self.cache = ConfigCache() if cache_enabled else None
        self.sources: List[ConfigSource] = []
        self.loaders = {
            ConfigSourceType.FILE: FileConfigLoader(self.cache),
            ConfigSourceType.ENVIRONMENT: EnvironmentConfigLoader(self.cache),
            ConfigSourceType.DICT: DictConfigLoader(self.cache)
        }
        
        # 文件监控
        self.observer: Optional[Observer] = None
        self.watcher: Optional[ConfigWatcher] = None
        self.reload_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.debug("Initialized ConfigLoader")
    
    def add_source(self, source: ConfigSource) -> None:
        """
        添加配置源
        
        Args:
            source: 配置源描述
        """
        self.sources.append(source)
        # 按优先级排序，优先级高的在后面（后加载会覆盖前面的）
        self.sources.sort(key=lambda s: s.priority)
        
        # 如果启用了文件监控，添加到监控列表
        if source.watch and source.source_type == ConfigSourceType.FILE:
            self._add_file_watch(source.source_path)
        
        logger.debug(
            f"Added config source: {source.source_type.value}:{source.source_path} "
            f"(priority: {source.priority})"
        )
    
    def add_file_source(
        self,
        file_path: Union[str, Path],
        format: ConfigFormat = ConfigFormat.AUTO,
        priority: int = 0,
        required: bool = False,
        watch: bool = False
    ) -> None:
        """添加文件配置源的便捷方法"""
        source = ConfigSource(
            source_type=ConfigSourceType.FILE,
            source_path=str(file_path),
            format=format,
            priority=priority,
            required=required,
            watch=watch
        )
        self.add_source(source)
    
    def add_env_source(
        self,
        prefix: str = "TEMPLATE_",
        priority: int = 100
    ) -> None:
        """添加环境变量配置源的便捷方法"""
        source = ConfigSource(
            source_type=ConfigSourceType.ENVIRONMENT,
            source_path="environment",
            format=ConfigFormat.ENV,
            priority=priority,
            options={"prefix": prefix}
        )
        self.add_source(source)
    
    def add_dict_source(
        self,
        data: Dict[str, Any],
        priority: int = 50
    ) -> None:
        """添加字典配置源的便捷方法"""
        source = ConfigSource(
            source_type=ConfigSourceType.DICT,
            source_path=data,  # 直接传递字典数据
            format=ConfigFormat.DICT,
            priority=priority
        )
        self.add_source(source)
    
    def load_config(self, template_name: Optional[str] = None) -> TemplateConfig:
        """
        加载并合并所有配置源的数据
        
        Args:
            template_name: 模板名称，用于查找特定的模板配置
            
        Returns:
            模板配置实例
            
        Raises:
            ConfigurationError: 配置加载或验证失败
        """
        try:
            # 加载所有配置源
            all_configs = []
            for source in self.sources:
                try:
                    if source.source_type in self.loaders:
                        loader = self.loaders[source.source_type]
                        config_data = loader.load(source)
                        all_configs.append((source.priority, config_data))
                        logger.debug(f"Loaded config from {source.source_type.value}: {source.source_path}")
                    else:
                        logger.warning(f"No loader available for source type: {source.source_type.value}")
                except Exception as e:
                    if source.required:
                        raise ConfigurationError(
                            f"Failed to load required config source {source.source_path}: {str(e)}",
                            error_code=ErrorCodes.CONFIG_FILE_NOT_FOUND,
                            cause=e
                        )
                    else:
                        logger.warning(f"Failed to load optional config source {source.source_path}: {str(e)}")
            
            # 合并配置数据
            merged_config = self._merge_configs(all_configs)
            
            # 如果指定了模板名称，查找特定模板的配置
            if template_name:
                template_configs = merged_config.get('templates', {})
                if template_name in template_configs:
                    template_data = template_configs[template_name]
                    # 合并全局配置和模板特定配置
                    global_config = {k: v for k, v in merged_config.items() if k != 'templates'}
                    merged_config = self._deep_merge(global_config, template_data)
            
            # 转换为TemplateConfig对象
            template_config = self._dict_to_template_config(merged_config, template_name)
            
            # 验证配置
            template_config.validate_structure()
            
            logger.info(f"Successfully loaded template config: {template_config.name}")
            return template_config
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load template configuration: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def load_from_file(self, file_path: Union[str, Path]) -> TemplateConfig:
        """
        从单个文件加载配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            模板配置实例
        """
        source = ConfigSource(
            source_type=ConfigSourceType.FILE,
            source_path=str(file_path),
            format=ConfigFormat.AUTO,
            required=True
        )
        
        loader = self.loaders[ConfigSourceType.FILE]
        config_data = loader.load(source)
        template_config = self._dict_to_template_config(config_data)
        template_config.validate_structure()
        
        return template_config
    
    def load_default(self, template_name: str) -> TemplateConfig:
        """
        加载默认配置
        
        Args:
            template_name: 模板名称
            
        Returns:
            默认模板配置实例
        """
        # 尝试从多个可能的位置加载默认配置
        possible_paths = [
            f"templates/configs/template_configs/{template_name.lower()}.yaml",
            f"templates/configs/{template_name.lower()}.yaml",
            f"configs/{template_name.lower()}.yaml",
            f"{template_name.lower()}.yaml"
        ]
        
        for config_path in possible_paths:
            if Path(config_path).exists():
                return self.load_from_file(config_path)
        
        # 如果没有找到配置文件，返回基本默认配置
        logger.warning(f"No config file found for template: {template_name}, using minimal default")
        return TemplateConfig(
            name=template_name,
            description=f"Default configuration for {template_name} template"
        )
    
    def reload(self) -> None:
        """重新加载所有配置"""
        if self.cache:
            self.cache.clear()
        
        # 触发重载回调
        try:
            config = self.load_config()
            for callback in self.reload_callbacks:
                try:
                    callback(config.to_dict())
                except Exception as e:
                    logger.error(f"Config reload callback failed: {str(e)}")
        except Exception as e:
            logger.error(f"Config reload failed: {str(e)}")
    
    def add_reload_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加配置重载回调函数"""
        self.reload_callbacks.append(callback)
    
    def start_file_watching(self) -> None:
        """启动文件监控"""
        if self.observer is None:
            self.watcher = ConfigWatcher(self._on_file_changed)
            self.observer = Observer()
            
            # 添加已有的文件监控
            watched_dirs = set()
            for source in self.sources:
                if source.watch and source.source_type == ConfigSourceType.FILE:
                    file_path = Path(source.source_path)
                    if file_path.exists():
                        dir_path = file_path.parent
                        if dir_path not in watched_dirs:
                            self.observer.schedule(self.watcher, str(dir_path), recursive=False)
                            watched_dirs.add(dir_path)
                        self.watcher.add_file(str(file_path))
            
            self.observer.start()
            logger.info("Started config file watching")
    
    def stop_file_watching(self) -> None:
        """停止文件监控"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.watcher = None
            logger.info("Stopped config file watching")
    
    def _add_file_watch(self, file_path: str) -> None:
        """添加文件监控"""
        if self.watcher:
            self.watcher.add_file(file_path)
            
            # 如果observer已经启动，添加目录监控
            if self.observer and self.observer.is_alive():
                dir_path = Path(file_path).parent
                if dir_path.exists():
                    self.observer.schedule(self.watcher, str(dir_path), recursive=False)
    
    def _on_file_changed(self, file_path: str) -> None:
        """文件变化回调"""
        logger.info(f"Reloading config due to file change: {file_path}")
        self.reload()
    
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
        """深度合并两个字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_template_config(
        self, 
        data: Dict[str, Any], 
        template_name: Optional[str] = None
    ) -> TemplateConfig:
        """
        将字典数据转换为TemplateConfig对象
        
        Args:
            data: 配置字典
            template_name: 模板名称
            
        Returns:
            TemplateConfig实例
        """
        # 提取基本字段
        config_data = {
            "name": data.get("name", template_name or "UnknownTemplate"),
            "version": data.get("version", "1.0.0"),
            "description": data.get("description", ""),
            "author": data.get("author", ""),
            "dependencies": data.get("dependencies", []),
            "template_dependencies": data.get("template_dependencies", []),
            "examples": data.get("examples", []),
            "timeout": data.get("timeout"),
            "retry_count": data.get("retry_count", 0),
            "async_enabled": data.get("async_enabled", False),
            "tags": data.get("tags", []),
            "documentation_url": data.get("documentation_url"),
            "source_url": data.get("source_url"),
            "cache_enabled": data.get("cache_enabled", False),
            "cache_ttl": data.get("cache_ttl", 3600),
            "max_memory_usage": data.get("max_memory_usage")
        }
        
        # 处理模板类型
        template_type_str = data.get("template_type", "custom")
        if isinstance(template_type_str, str):
            try:
                config_data["template_type"] = TemplateType(template_type_str.lower())
            except ValueError:
                config_data["template_type"] = TemplateType.CUSTOM
        else:
            config_data["template_type"] = template_type_str
        
        # 处理参数定义
        parameters = {}
        parameters_data = data.get("parameters", {})
        
        for param_name, param_info in parameters_data.items():
            if isinstance(param_info, dict):
                # 详细参数定义
                param_type_str = param_info.get("type", "str")
                param_type = {
                    "str": str, "string": str,
                    "int": int, "integer": int,
                    "float": float, "number": float,
                    "bool": bool, "boolean": bool,
                    "list": list, "array": list,
                    "dict": dict, "object": dict,
                    "path": Path
                }.get(param_type_str.lower(), str)
                
                parameters[param_name] = ParameterSchema(
                    name=param_name,
                    type=param_type,
                    required=param_info.get("required", True),
                    default=param_info.get("default"),
                    description=param_info.get("description", ""),
                    constraints=param_info.get("constraints", {}),
                    examples=param_info.get("examples", [])
                )
            else:
                # 简单参数定义（只指定类型）
                if isinstance(param_info, str):
                    param_type = {
                        "str": str, "int": int, "float": float, 
                        "bool": bool, "list": list, "dict": dict
                    }.get(param_info.lower(), str)
                else:
                    param_type = param_info
                
                parameters[param_name] = ParameterSchema(
                    name=param_name,
                    type=param_type,
                    required=True,
                    description=""
                )
        
        config_data["parameters"] = parameters
        
        return TemplateConfig(**config_data)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_disabled": True}
    
    def clear_cache(self) -> None:
        """清空配置缓存"""
        if self.cache:
            self.cache.clear()
            logger.info("Cleared config cache")
    
    def __del__(self):
        """析构函数，清理资源"""
        self.stop_file_watching()