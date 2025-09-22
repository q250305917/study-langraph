"""
基础格式化器
功能：定义格式化器的基类和通用接口
作者：自动文档生成系统
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FormatterConfig:
    """格式化器配置"""
    output_format: str           # 输出格式名称
    theme: str = "default"       # 主题名称
    include_toc: bool = True     # 是否包含目录
    include_nav: bool = True     # 是否包含导航
    syntax_highlight: bool = True # 是否启用语法高亮
    custom_css: Optional[Path] = None     # 自定义样式文件
    custom_js: Optional[Path] = None      # 自定义脚本文件
    meta_data: Dict[str, Any] = None      # 元数据
    
    def __post_init__(self):
        if self.meta_data is None:
            self.meta_data = {}


class BaseFormatter(ABC):
    """格式化器抽象基类"""
    
    def __init__(self, config: FormatterConfig):
        """
        初始化格式化器
        
        Args:
            config: 格式化器配置
        """
        self.config = config
        self.format_name = config.output_format
        
    @abstractmethod
    def format_content(self, content: str, **kwargs) -> str:
        """
        格式化内容的核心方法
        
        Args:
            content: 要格式化的内容
            **kwargs: 额外的格式化参数
            
        Returns:
            格式化后的内容
        """
        pass
    
    @abstractmethod
    def save_to_file(self, content: str, output_path: Path, **kwargs) -> bool:
        """
        保存格式化内容到文件
        
        Args:
            content: 格式化后的内容
            output_path: 输出文件路径
            **kwargs: 额外参数
            
        Returns:
            保存是否成功
        """
        pass
    
    def get_file_extension(self) -> str:
        """
        获取输出文件的扩展名
        
        Returns:
            文件扩展名（包含点号）
        """
        extension_map = {
            'markdown': '.md',
            'html': '.html',
            'pdf': '.pdf',
            'docx': '.docx',
            'rst': '.rst'
        }
        return extension_map.get(self.format_name, '.txt')
    
    def validate_content(self, content: str) -> bool:
        """
        验证内容是否有效
        
        Args:
            content: 要验证的内容
            
        Returns:
            内容是否有效
        """
        if not content or not content.strip():
            logger.warning("Content is empty or whitespace only")
            return False
        return True
    
    def preprocess_content(self, content: str) -> str:
        """
        预处理内容（子类可重写）
        
        Args:
            content: 原始内容
            
        Returns:
            预处理后的内容
        """
        return content.strip()
    
    def postprocess_content(self, content: str) -> str:
        """
        后处理内容（子类可重写）
        
        Args:
            content: 格式化后的内容
            
        Returns:
            后处理后的内容
        """
        return content
    
    def format_and_save(self, content: str, output_path: Path, **kwargs) -> bool:
        """
        格式化内容并保存到文件的便捷方法
        
        Args:
            content: 要格式化的内容
            output_path: 输出文件路径
            **kwargs: 额外参数
            
        Returns:
            操作是否成功
        """
        try:
            # 验证内容
            if not self.validate_content(content):
                return False
            
            # 预处理
            content = self.preprocess_content(content)
            
            # 格式化
            formatted_content = self.format_content(content, **kwargs)
            
            # 后处理
            formatted_content = self.postprocess_content(formatted_content)
            
            # 保存到文件
            return self.save_to_file(formatted_content, output_path, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in format_and_save: {e}")
            return False
    
    def get_supported_features(self) -> List[str]:
        """
        获取格式化器支持的特性列表
        
        Returns:
            支持的特性列表
        """
        return [
            'basic_formatting',
            'code_blocks',
            'tables',
            'links',
            'images'
        ]
    
    def supports_feature(self, feature: str) -> bool:
        """
        检查是否支持特定特性
        
        Args:
            feature: 特性名称
            
        Returns:
            是否支持该特性
        """
        return feature in self.get_supported_features()


class FormatterRegistry:
    """格式化器注册表"""
    
    def __init__(self):
        """初始化注册表"""
        self._formatters = {}
    
    def register(self, name: str, formatter_class: type):
        """
        注册格式化器
        
        Args:
            name: 格式化器名称
            formatter_class: 格式化器类
        """
        if not issubclass(formatter_class, BaseFormatter):
            raise TypeError("Formatter class must inherit from BaseFormatter")
        
        self._formatters[name] = formatter_class
        logger.debug(f"Registered formatter: {name}")
    
    def get_formatter(self, name: str, config: FormatterConfig) -> Optional[BaseFormatter]:
        """
        获取格式化器实例
        
        Args:
            name: 格式化器名称
            config: 格式化器配置
            
        Returns:
            格式化器实例或None
        """
        formatter_class = self._formatters.get(name)
        if formatter_class:
            return formatter_class(config)
        return None
    
    def list_formatters(self) -> List[str]:
        """
        列出所有已注册的格式化器
        
        Returns:
            格式化器名称列表
        """
        return list(self._formatters.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        检查格式化器是否已注册
        
        Args:
            name: 格式化器名称
            
        Returns:
            是否已注册
        """
        return name in self._formatters


# 全局格式化器注册表实例
formatter_registry = FormatterRegistry()


class MultiFormatter:
    """多格式化器 - 支持同时输出多种格式"""
    
    def __init__(self, formatters: List[BaseFormatter]):
        """
        初始化多格式化器
        
        Args:
            formatters: 格式化器列表
        """
        self.formatters = formatters
    
    def format_and_save_all(self, content: str, base_output_path: Path, **kwargs) -> Dict[str, bool]:
        """
        使用所有格式化器格式化并保存内容
        
        Args:
            content: 要格式化的内容
            base_output_path: 基础输出路径（不含扩展名）
            **kwargs: 额外参数
            
        Returns:
            各格式化器的处理结果字典
        """
        results = {}
        
        for formatter in self.formatters:
            try:
                # 构建输出文件路径
                output_path = base_output_path.with_suffix(formatter.get_file_extension())
                
                # 格式化并保存
                success = formatter.format_and_save(content, output_path, **kwargs)
                results[formatter.format_name] = success
                
                if success:
                    logger.info(f"Successfully generated {formatter.format_name} format")
                else:
                    logger.error(f"Failed to generate {formatter.format_name} format")
                    
            except Exception as e:
                logger.error(f"Error processing {formatter.format_name} format: {e}")
                results[formatter.format_name] = False
        
        return results
    
    def add_formatter(self, formatter: BaseFormatter):
        """
        添加格式化器
        
        Args:
            formatter: 要添加的格式化器
        """
        self.formatters.append(formatter)
    
    def remove_formatter(self, format_name: str):
        """
        移除格式化器
        
        Args:
            format_name: 要移除的格式化器名称
        """
        self.formatters = [f for f in self.formatters if f.format_name != format_name]
    
    def get_formatter(self, format_name: str) -> Optional[BaseFormatter]:
        """
        获取指定格式的格式化器
        
        Args:
            format_name: 格式名称
            
        Returns:
            格式化器实例或None
        """
        for formatter in self.formatters:
            if formatter.format_name == format_name:
                return formatter
        return None


# 工具函数
def create_formatter(format_name: str, config: Optional[FormatterConfig] = None) -> Optional[BaseFormatter]:
    """
    创建格式化器的便捷函数
    
    Args:
        format_name: 格式名称
        config: 格式化器配置，如果为None则使用默认配置
        
    Returns:
        格式化器实例或None
    """
    if config is None:
        config = FormatterConfig(output_format=format_name)
    
    return formatter_registry.get_formatter(format_name, config)


def create_multi_formatter(format_names: List[str], 
                          base_config: Optional[FormatterConfig] = None) -> MultiFormatter:
    """
    创建多格式化器的便捷函数
    
    Args:
        format_names: 格式名称列表
        base_config: 基础配置，会为每种格式创建副本
        
    Returns:
        多格式化器实例
    """
    formatters = []
    
    for format_name in format_names:
        # 为每种格式创建独立配置
        if base_config:
            config = FormatterConfig(
                output_format=format_name,
                theme=base_config.theme,
                include_toc=base_config.include_toc,
                include_nav=base_config.include_nav,
                syntax_highlight=base_config.syntax_highlight,
                custom_css=base_config.custom_css,
                custom_js=base_config.custom_js,
                meta_data=base_config.meta_data.copy() if base_config.meta_data else {}
            )
        else:
            config = FormatterConfig(output_format=format_name)
        
        formatter = create_formatter(format_name, config)
        if formatter:
            formatters.append(formatter)
        else:
            logger.warning(f"Failed to create formatter for format: {format_name}")
    
    return MultiFormatter(formatters)