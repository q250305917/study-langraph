"""
配置管理系统
功能：处理YAML配置文件，管理文档生成的各种配置选项
作者：自动文档生成系统
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """源代码配置"""
    directories: List[str] = field(default_factory=lambda: ["src/", "examples/"])
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__/*", "*.pyc", "test_*"])
    recursive: bool = True
    encoding: str = "utf-8"


@dataclass
class OutputConfig:
    """输出配置"""
    directory: str = "docs/generated/"
    formats: List[str] = field(default_factory=lambda: ["markdown", "html"])
    clean_before_generate: bool = True
    create_index: bool = True
    organize_by_module: bool = True


@dataclass
class APIDocConfig:
    """API文档配置"""
    include_private: bool = False
    include_source_links: bool = True
    include_inheritance: bool = True
    include_examples: bool = True
    sort_members: bool = True
    group_by_type: bool = True
    generate_class_diagrams: bool = False


@dataclass
class TutorialConfig:
    """教程配置"""
    auto_generate: bool = True
    include_output: bool = True
    execute_code: bool = False
    language: str = "zh"
    difficulty_levels: List[str] = field(default_factory=lambda: ["初级", "中级", "高级"])
    format_style: str = "step_by_step"


@dataclass
class ThemeConfig:
    """主题配置"""
    name: str = "default"
    custom_css: Optional[str] = None
    custom_js: Optional[str] = None
    syntax_highlighting: bool = True
    color_scheme: str = "light"
    font_family: str = "system"


@dataclass
class PublishingConfig:
    """发布配置"""
    auto_deploy: bool = False
    github_pages: bool = False
    confluence_space: Optional[str] = None
    s3_bucket: Optional[str] = None
    custom_domain: Optional[str] = None


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    directory: str = ".doc_cache/"
    max_age_hours: int = 24
    cache_templates: bool = True
    cache_assets: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5


@dataclass
class DocumentationConfig:
    """完整的文档配置"""
    # 基本信息
    project_name: str = "项目文档"
    project_version: str = "1.0.0"
    project_description: str = ""
    project_url: str = ""
    author: str = ""
    
    # 各模块配置
    source: SourceConfig = field(default_factory=SourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    api_docs: APIDocConfig = field(default_factory=APIDocConfig)
    tutorials: TutorialConfig = field(default_factory=TutorialConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    publishing: PublishingConfig = field(default_factory=PublishingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 扩展配置
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentationConfig':
        """从字典创建配置实例"""
        # 提取各部分配置
        source_data = data.get('source', {})
        output_data = data.get('output', {})
        api_docs_data = data.get('api_docs', {})
        tutorials_data = data.get('tutorials', {})
        theme_data = data.get('theme', {})
        publishing_data = data.get('publishing', {})
        cache_data = data.get('cache', {})
        logging_data = data.get('logging', {})
        
        return cls(
            project_name=data.get('project_name', '项目文档'),
            project_version=data.get('project_version', '1.0.0'),
            project_description=data.get('project_description', ''),
            project_url=data.get('project_url', ''),
            author=data.get('author', ''),
            source=SourceConfig(**source_data),
            output=OutputConfig(**output_data),
            api_docs=APIDocConfig(**api_docs_data),
            tutorials=TutorialConfig(**tutorials_data),
            theme=ThemeConfig(**theme_data),
            publishing=PublishingConfig(**publishing_data),
            cache=CacheConfig(**cache_data),
            logging=LoggingConfig(**logging_data),
            custom=data.get('custom', {})
        )


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则查找默认位置
        """
        self.config_path = config_path or self._find_config_file()
        self.config: Optional[DocumentationConfig] = None
        self._load_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """
        查找配置文件
        
        Returns:
            配置文件路径或None
        """
        # 查找配置文件的常见位置
        possible_paths = [
            Path("docs.yaml"),
            Path("docs.yml"),
            Path(".docs.yaml"),
            Path(".docs.yml"),
            Path("config/docs.yaml"),
            Path("config/docs.yml"),
            Path("docs/config.yaml"),
            Path("docs/config.yml")
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found config file: {path}")
                return path
        
        logger.warning("No config file found, using default configuration")
        return None
    
    def _load_config(self):
        """加载配置文件"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                
                self.config = DocumentationConfig.from_dict(config_data)
                logger.info(f"Loaded configuration from: {self.config_path}")
                
            except Exception as e:
                logger.error(f"Error loading config file {self.config_path}: {e}")
                self.config = DocumentationConfig()
        else:
            # 使用默认配置
            self.config = DocumentationConfig()
            logger.info("Using default configuration")
    
    def save_config(self, config_path: Optional[Path] = None):
        """
        保存配置到文件
        
        Args:
            config_path: 保存路径，如果为None则使用当前配置路径
        """
        if not self.config:
            logger.error("No configuration to save")
            return
        
        save_path = config_path or self.config_path
        if not save_path:
            save_path = Path("docs.yaml")
        
        try:
            # 确保目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为字典并保存
            config_dict = self.config.to_dict()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, 
                         default_flow_style=False, 
                         allow_unicode=True,
                         sort_keys=False,
                         indent=2)
            
            logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {e}")
    
    def get_config(self) -> DocumentationConfig:
        """
        获取当前配置
        
        Returns:
            文档配置对象
        """
        return self.config or DocumentationConfig()
    
    def update_config(self, **kwargs):
        """
        更新配置
        
        Args:
            **kwargs: 要更新的配置项
        """
        if not self.config:
            self.config = DocumentationConfig()
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                # 存储到custom字段
                self.config.custom[key] = value
        
        logger.debug(f"Updated configuration: {list(kwargs.keys())}")
    
    def get_source_paths(self) -> List[Path]:
        """
        获取源代码路径列表
        
        Returns:
            源代码路径列表
        """
        config = self.get_config()
        paths = []
        
        for directory in config.source.directories:
            path = Path(directory)
            if path.exists():
                paths.append(path)
            else:
                logger.warning(f"Source directory not found: {directory}")
        
        return paths
    
    def get_output_path(self) -> Path:
        """
        获取输出路径
        
        Returns:
            输出路径
        """
        config = self.get_config()
        return Path(config.output.directory)
    
    def should_include_file(self, file_path: Path) -> bool:
        """
        判断是否应该包含某个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否包含
        """
        config = self.get_config()
        
        # 检查包含模式
        include_match = False
        for pattern in config.source.include_patterns:
            if file_path.match(pattern):
                include_match = True
                break
        
        if not include_match:
            return False
        
        # 检查排除模式
        for pattern in config.source.exclude_patterns:
            if file_path.match(pattern):
                return False
        
        return True
    
    def get_cache_path(self) -> Path:
        """
        获取缓存路径
        
        Returns:
            缓存路径
        """
        config = self.get_config()
        cache_path = Path(config.cache.directory)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def is_cache_enabled(self) -> bool:
        """
        检查是否启用缓存
        
        Returns:
            是否启用缓存
        """
        config = self.get_config()
        return config.cache.enabled
    
    def get_theme_path(self) -> Optional[Path]:
        """
        获取主题路径
        
        Returns:
            主题路径或None
        """
        config = self.get_config()
        if config.theme.custom_css:
            css_path = Path(config.theme.custom_css)
            if css_path.exists():
                return css_path.parent
        return None
    
    def validate_config(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误信息列表
        """
        errors = []
        config = self.get_config()
        
        # 验证源目录
        for directory in config.source.directories:
            if not Path(directory).exists():
                errors.append(f"Source directory not found: {directory}")
        
        # 验证输出格式
        valid_formats = ['markdown', 'html', 'pdf']
        for format_name in config.output.formats:
            if format_name not in valid_formats:
                errors.append(f"Invalid output format: {format_name}")
        
        # 验证主题文件
        if config.theme.custom_css:
            css_path = Path(config.theme.custom_css)
            if not css_path.exists():
                errors.append(f"Custom CSS file not found: {config.theme.custom_css}")
        
        if config.theme.custom_js:
            js_path = Path(config.theme.custom_js)
            if not js_path.exists():
                errors.append(f"Custom JS file not found: {config.theme.custom_js}")
        
        # 验证日志级别
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level not in valid_log_levels:
            errors.append(f"Invalid logging level: {config.logging.level}")
        
        return errors


def create_default_config() -> DocumentationConfig:
    """
    创建默认配置
    
    Returns:
        默认配置对象
    """
    return DocumentationConfig()


def create_config_template(output_path: Path):
    """
    创建配置模板文件
    
    Args:
        output_path: 输出文件路径
    """
    config = create_default_config()
    
    # 添加注释的YAML内容
    yaml_content = """# 文档生成配置文件
# 项目基本信息
project_name: "LangChain学习项目"
project_version: "1.0.0"
project_description: "LangChain学习和实践项目的文档"
project_url: "https://github.com/example/langchain-learning"
author: "开发团队"

# 源代码配置
source:
  directories:
    - "src/"
    - "examples/"
  include_patterns:
    - "*.py"
  exclude_patterns:
    - "__pycache__/*"
    - "*.pyc"
    - "test_*"
  recursive: true
  encoding: "utf-8"

# 输出配置
output:
  directory: "docs/generated/"
  formats:
    - "markdown"
    - "html"
  clean_before_generate: true
  create_index: true
  organize_by_module: true

# API文档配置
api_docs:
  include_private: false
  include_source_links: true
  include_inheritance: true
  include_examples: true
  sort_members: true
  group_by_type: true
  generate_class_diagrams: false

# 教程配置
tutorials:
  auto_generate: true
  include_output: true
  execute_code: false
  language: "zh"
  difficulty_levels:
    - "初级"
    - "中级"
    - "高级"
  format_style: "step_by_step"

# 主题配置
theme:
  name: "default"
  custom_css: null
  custom_js: null
  syntax_highlighting: true
  color_scheme: "light"
  font_family: "system"

# 发布配置
publishing:
  auto_deploy: false
  github_pages: false
  confluence_space: null
  s3_bucket: null
  custom_domain: null

# 缓存配置
cache:
  enabled: true
  directory: ".doc_cache/"
  max_age_hours: 24
  cache_templates: true
  cache_assets: true

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null
  max_file_size: "10MB"
  backup_count: 5

# 自定义配置
custom:
  # 在这里添加项目特定的配置
  experimental_features: false
  debug_mode: false
"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        logger.info(f"Configuration template created: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating config template: {e}")


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    
    if _config_manager is None or config_path:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def get_config() -> DocumentationConfig:
    """
    获取当前配置的便捷函数
    
    Returns:
        文档配置对象
    """
    return get_config_manager().get_config()


if __name__ == "__main__":
    # 测试配置管理
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create-template":
            output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs.yaml")
            create_config_template(output_path)
            
        elif command == "validate":
            config_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            manager = ConfigManager(config_path)
            errors = manager.validate_config()
            
            if errors:
                print("配置验证失败:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("配置验证通过")
                
        elif command == "show":
            config_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
            manager = ConfigManager(config_path)
            config = manager.get_config()
            
            print("当前配置:")
            print(f"  项目名称: {config.project_name}")
            print(f"  项目版本: {config.project_version}")
            print(f"  源目录: {config.source.directories}")
            print(f"  输出目录: {config.output.directory}")
            print(f"  输出格式: {config.output.formats}")
            
    else:
        print("用法:")
        print("  python config.py create-template [output_path]  # 创建配置模板")
        print("  python config.py validate [config_path]        # 验证配置")
        print("  python config.py show [config_path]            # 显示配置")