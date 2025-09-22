"""
文档自动生成系统
功能：为LangChain学习项目提供完整的文档自动化生成解决方案
作者：自动文档生成系统
版本：1.0.0
"""

from .code_parser import CodeParser, CodeElement, NodeType
from .template_engine import TemplateEngine, TemplateManager
from .api_generator import APIDocumentationGenerator, APIDocumentationConfig
from .tutorial_generator import TutorialGenerator, TutorialConfig
from .config import ConfigManager, DocumentationConfig
from .cli import DocumentationCLI

# 格式化器
from .formatters import (
    BaseFormatter,
    FormatterConfig,
    MarkdownFormatter,
    HTMLFormatter,
    PDFFormatter
)

__version__ = "1.0.0"
__author__ = "文档生成系统团队"
__description__ = "自动化文档生成工具，支持API文档、教程生成和多格式输出"

__all__ = [
    # 核心组件
    'CodeParser',
    'CodeElement', 
    'NodeType',
    'TemplateEngine',
    'TemplateManager',
    
    # 生成器
    'APIDocumentationGenerator',
    'APIDocumentationConfig',
    'TutorialGenerator',
    'TutorialConfig',
    
    # 配置管理
    'ConfigManager',
    'DocumentationConfig',
    
    # 格式化器
    'BaseFormatter',
    'FormatterConfig',
    'MarkdownFormatter',
    'HTMLFormatter',
    'PDFFormatter',
    
    # 命令行工具
    'DocumentationCLI',
    
    # 便捷函数
    'generate_api_docs',
    'generate_tutorials',
    'generate_all_docs'
]


def generate_api_docs(source_path, output_path, config=None):
    """
    生成API文档的便捷函数
    
    Args:
        source_path: 源代码路径
        output_path: 输出路径
        config: API文档配置，如果为None则使用默认配置
        
    Returns:
        生成是否成功
    """
    from pathlib import Path
    
    if config is None:
        config = APIDocumentationConfig()
    
    generator = APIDocumentationGenerator(config)
    return generator.generate_documentation(Path(source_path), Path(output_path))


def generate_tutorials(examples_path, output_path, config=None):
    """
    生成教程文档的便捷函数
    
    Args:
        examples_path: 示例代码路径
        output_path: 输出路径
        config: 教程配置，如果为None则使用默认配置
        
    Returns:
        生成是否成功
    """
    from pathlib import Path
    
    if config is None:
        config = TutorialConfig()
    
    generator = TutorialGenerator(config)
    return generator.generate_tutorial_documentation(Path(examples_path), Path(output_path))


def generate_all_docs(config_path=None, output_path=None):
    """
    生成所有类型文档的便捷函数
    
    Args:
        config_path: 配置文件路径，如果为None则查找默认配置
        output_path: 输出路径，如果为None则使用配置中的路径
        
    Returns:
        生成是否成功
    """
    from pathlib import Path
    
    # 加载配置
    config_manager = ConfigManager(Path(config_path) if config_path else None)
    config = config_manager.get_config()
    
    # 确定输出路径
    if output_path:
        output_dir = Path(output_path)
    else:
        output_dir = Path(config.output.directory)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    try:
        # 生成API文档
        source_paths = config_manager.get_source_paths()
        if source_paths:
            api_config = APIDocumentationConfig(
                include_private=config.api_docs.include_private,
                include_source=config.api_docs.include_source_links,
                include_inheritance=config.api_docs.include_inheritance,
                include_examples=config.api_docs.include_examples,
                sort_members=config.api_docs.sort_members
            )
            
            api_generator = APIDocumentationGenerator(api_config)
            for source_path in source_paths:
                api_output = output_dir / "api"
                if not api_generator.generate_documentation(source_path, api_output):
                    success = False
        
        # 生成教程文档
        examples_dir = Path("examples")
        if examples_dir.exists():
            tutorial_config = TutorialConfig(
                auto_generate=config.tutorials.auto_generate,
                include_output=config.tutorials.include_output,
                code_execution=config.tutorials.execute_code,
                language=config.tutorials.language,
                difficulty_levels=config.tutorials.difficulty_levels,
                format_style=config.tutorials.format_style
            )
            
            tutorial_generator = TutorialGenerator(tutorial_config)
            tutorial_output = output_dir / "tutorials"
            if not tutorial_generator.generate_tutorial_documentation(examples_dir, tutorial_output):
                success = False
        
        return success
        
    except Exception as e:
        import logging
        logging.error(f"文档生成失败: {e}")
        return False


# 检查依赖
def check_dependencies():
    """
    检查文档生成所需的依赖
    
    Returns:
        依赖状态字典
    """
    dependencies = {
        'required': {},
        'optional': {}
    }
    
    # 检查必需依赖
    try:
        import yaml
        dependencies['required']['yaml'] = True
    except ImportError:
        dependencies['required']['yaml'] = False
    
    try:
        import jinja2
        dependencies['required']['jinja2'] = True
    except ImportError:
        dependencies['required']['jinja2'] = False
    
    # 检查可选依赖
    try:
        import weasyprint
        dependencies['optional']['weasyprint'] = True
    except ImportError:
        dependencies['optional']['weasyprint'] = False
    
    try:
        import subprocess
        result = subprocess.run(['wkhtmltopdf', '--version'], 
                              capture_output=True, timeout=5)
        dependencies['optional']['wkhtmltopdf'] = result.returncode == 0
    except:
        dependencies['optional']['wkhtmltopdf'] = False
    
    return dependencies


def get_version():
    """获取版本信息"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__
    }


if __name__ == "__main__":
    # 简单的模块测试
    print(f"文档生成系统 v{__version__}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")
    
    print("\n依赖检查:")
    deps = check_dependencies()
    
    print("必需依赖:")
    for name, available in deps['required'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    
    print("可选依赖:")
    for name, available in deps['optional'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    
    # 检查是否可以运行基本功能
    all_required = all(deps['required'].values())
    if all_required:
        print("\n✓ 所有必需依赖已满足，可以正常使用")
    else:
        print("\n✗ 缺少必需依赖，请先安装：")
        for name, available in deps['required'].items():
            if not available:
                print(f"  pip install {name}")