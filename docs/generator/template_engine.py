"""
模板引擎系统
功能：基于Jinja2的文档模板渲染引擎，支持自定义过滤器和全局变量
作者：自动文档生成系统
"""

import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from jinja2.filters import FILTERS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateEngine:
    """文档模板引擎 - 基于Jinja2实现"""
    
    def __init__(self, template_dirs: Optional[List[Path]] = None):
        """
        初始化模板引擎
        
        Args:
            template_dirs: 模板目录列表，如果为None则使用默认目录
        """
        # 设置默认模板目录
        if template_dirs is None:
            current_dir = Path(__file__).parent.parent
            template_dirs = [current_dir / "templates"]
        
        self.template_dirs = [Path(d) for d in template_dirs]
        
        # 确保模板目录存在
        for template_dir in self.template_dirs:
            template_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化Jinja2环境
        self.env = Environment(
            loader=FileSystemLoader([str(d) for d in self.template_dirs]),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # 注册自定义过滤器
        self._register_filters()
        
        # 设置全局变量
        self._setup_globals()
        
        logger.info(f"Template engine initialized with directories: {self.template_dirs}")
    
    def _register_filters(self):
        """注册自定义过滤器"""
        
        @self.env.filter('code_block')
        def code_block_filter(text: str, language: str = 'python') -> str:
            """
            将文本包装为代码块
            
            Args:
                text: 要包装的文本
                language: 代码语言
                
            Returns:
                格式化的代码块
            """
            if not text:
                return ""
            return f"```{language}\n{text}\n```"
        
        @self.env.filter('indent_code')
        def indent_code_filter(text: str, spaces: int = 4) -> str:
            """
            缩进代码文本
            
            Args:
                text: 要缩进的文本
                spaces: 缩进空格数
                
            Returns:
                缩进后的文本
            """
            if not text:
                return ""
            indent = " " * spaces
            return "\n".join(f"{indent}{line}" if line.strip() else line 
                           for line in text.split("\n"))
        
        @self.env.filter('format_docstring')
        def format_docstring_filter(docstring: str) -> str:
            """
            格式化文档字符串，处理缩进和空行
            
            Args:
                docstring: 原始文档字符串
                
            Returns:
                格式化后的文档字符串
            """
            if not docstring:
                return ""
            
            lines = docstring.strip().split('\n')
            if not lines:
                return ""
            
            # 移除第一行前的缩进
            first_line = lines[0]
            
            # 如果只有一行，直接返回
            if len(lines) == 1:
                return first_line.strip()
            
            # 处理多行文档字符串
            # 找到最小缩进（忽略空行）
            min_indent = float('inf')
            for line in lines[1:]:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            if min_indent == float('inf'):
                min_indent = 0
            
            # 移除公共缩进
            formatted_lines = [first_line.strip()]
            for line in lines[1:]:
                if line.strip():
                    formatted_lines.append(line[min_indent:])
                else:
                    formatted_lines.append("")
            
            return '\n'.join(formatted_lines)
        
        @self.env.filter('snake_to_title')
        def snake_to_title_filter(text: str) -> str:
            """
            将snake_case转换为Title Case
            
            Args:
                text: snake_case文本
                
            Returns:
                Title Case文本
            """
            return ' '.join(word.capitalize() for word in text.split('_'))
        
        @self.env.filter('format_type')
        def format_type_filter(type_str: Optional[str]) -> str:
            """
            格式化类型注解字符串
            
            Args:
                type_str: 类型注解字符串
                
            Returns:
                格式化后的类型字符串
            """
            if not type_str:
                return "Any"
            
            # 简化常见的类型注解
            type_mappings = {
                'typing.List': 'List',
                'typing.Dict': 'Dict',
                'typing.Optional': 'Optional',
                'typing.Union': 'Union',
                'typing.Callable': 'Callable',
                'typing.Any': 'Any',
            }
            
            formatted = type_str
            for full_type, short_type in type_mappings.items():
                formatted = formatted.replace(full_type, short_type)
            
            return formatted
        
        @self.env.filter('to_anchor')
        def to_anchor_filter(text: str) -> str:
            """
            将文本转换为HTML锚点ID
            
            Args:
                text: 原始文本
                
            Returns:
                锚点ID
            """
            # 转换为小写，替换特殊字符为连字符
            anchor = re.sub(r'[^\w\s-]', '', text.lower())
            anchor = re.sub(r'[-\s]+', '-', anchor)
            return anchor.strip('-')
        
        @self.env.filter('word_count')
        def word_count_filter(text: str) -> int:
            """
            计算文本词数
            
            Args:
                text: 文本内容
                
            Returns:
                词数
            """
            if not text:
                return 0
            return len(text.split())
        
        @self.env.filter('truncate_words')
        def truncate_words_filter(text: str, length: int = 50, 
                                suffix: str = "...") -> str:
            """
            按词数截断文本
            
            Args:
                text: 原始文本
                length: 最大词数
                suffix: 截断后缀
                
            Returns:
                截断后的文本
            """
            if not text:
                return ""
            
            words = text.split()
            if len(words) <= length:
                return text
            
            return ' '.join(words[:length]) + suffix
        
        @self.env.filter('json_pretty')
        def json_pretty_filter(obj: Any, indent: int = 2) -> str:
            """
            将对象格式化为美观的JSON字符串
            
            Args:
                obj: 要格式化的对象
                indent: 缩进空格数
                
            Returns:
                格式化的JSON字符串
            """
            try:
                return json.dumps(obj, indent=indent, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(obj)
        
        logger.info("Custom filters registered successfully")
    
    def _setup_globals(self):
        """设置全局变量"""
        self.env.globals.update({
            'now': datetime.now,
            'today': datetime.now().strftime('%Y-%m-%d'),
            'year': datetime.now().year,
            'version': '1.0.0',  # 可以从配置文件读取
            'generator_name': 'LangChain Learning Doc Generator'
        })
        
        logger.info("Global variables set up successfully")
    
    def add_global(self, name: str, value: Any):
        """
        添加全局变量
        
        Args:
            name: 变量名
            value: 变量值
        """
        self.env.globals[name] = value
        logger.debug(f"Added global variable: {name}")
    
    def add_filter(self, name: str, func: Callable):
        """
        添加自定义过滤器
        
        Args:
            name: 过滤器名称
            func: 过滤器函数
        """
        self.env.filters[name] = func
        logger.debug(f"Added custom filter: {name}")
    
    def load_template(self, template_name: str) -> Template:
        """
        加载模板文件
        
        Args:
            template_name: 模板文件名
            
        Returns:
            加载的模板对象
            
        Raises:
            TemplateNotFound: 当模板文件不存在时
        """
        try:
            template = self.env.get_template(template_name)
            logger.debug(f"Loaded template: {template_name}")
            return template
        except TemplateNotFound as e:
            logger.error(f"Template not found: {template_name}")
            raise e
    
    def render_template(self, template_name: str, 
                       context: Dict[str, Any]) -> str:
        """
        渲染模板
        
        Args:
            template_name: 模板文件名
            context: 模板上下文变量
            
        Returns:
            渲染后的内容
        """
        template = self.load_template(template_name)
        return self.render(template, context)
    
    def render(self, template: Union[Template, str], 
              context: Dict[str, Any]) -> str:
        """
        渲染模板
        
        Args:
            template: 模板对象或模板字符串
            context: 模板上下文变量
            
        Returns:
            渲染后的内容
        """
        try:
            if isinstance(template, str):
                # 如果是字符串，创建临时模板
                template_obj = self.env.from_string(template)
            else:
                template_obj = template
            
            rendered = template_obj.render(**context)
            logger.debug("Template rendered successfully")
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            raise e
    
    def render_string(self, template_string: str, 
                     context: Dict[str, Any]) -> str:
        """
        渲染模板字符串
        
        Args:
            template_string: 模板字符串
            context: 模板上下文变量
            
        Returns:
            渲染后的内容
        """
        template = self.env.from_string(template_string)
        return self.render(template, context)
    
    def list_templates(self) -> List[str]:
        """
        列出所有可用的模板
        
        Returns:
            模板名称列表
        """
        templates = []
        for template_dir in self.template_dirs:
            if template_dir.exists():
                for template_file in template_dir.rglob("*.md"):
                    relative_path = template_file.relative_to(template_dir)
                    templates.append(str(relative_path))
        
        return sorted(templates)
    
    def template_exists(self, template_name: str) -> bool:
        """
        检查模板是否存在
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板是否存在
        """
        try:
            self.env.get_template(template_name)
            return True
        except TemplateNotFound:
            return False
    
    def get_template_source(self, template_name: str) -> str:
        """
        获取模板源码
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板源码
        """
        template = self.load_template(template_name)
        return template.source


class TemplateManager:
    """模板管理器 - 管理多种模板类型"""
    
    def __init__(self, template_engine: TemplateEngine):
        """
        初始化模板管理器
        
        Args:
            template_engine: 模板引擎实例
        """
        self.engine = template_engine
        self.template_registry = {}
        
        # 注册默认模板类型
        self._register_default_templates()
    
    def _register_default_templates(self):
        """注册默认模板类型"""
        self.template_registry.update({
            'api': {
                'module': 'api_module.md',
                'class': 'api_class.md',
                'function': 'api_function.md'
            },
            'tutorial': {
                'main': 'tutorial_main.md',
                'section': 'tutorial_section.md',
                'example': 'tutorial_example.md'
            },
            'documentation': {
                'index': 'doc_index.md',
                'readme': 'doc_readme.md',
                'guide': 'doc_guide.md'
            }
        })
    
    def register_template(self, category: str, template_type: str, 
                         template_name: str):
        """
        注册模板
        
        Args:
            category: 模板分类
            template_type: 模板类型
            template_name: 模板文件名
        """
        if category not in self.template_registry:
            self.template_registry[category] = {}
        
        self.template_registry[category][template_type] = template_name
        logger.debug(f"Registered template: {category}.{template_type} -> {template_name}")
    
    def get_template(self, category: str, template_type: str) -> Optional[str]:
        """
        获取模板文件名
        
        Args:
            category: 模板分类
            template_type: 模板类型
            
        Returns:
            模板文件名或None
        """
        return self.template_registry.get(category, {}).get(template_type)
    
    def render_by_type(self, category: str, template_type: str, 
                      context: Dict[str, Any]) -> str:
        """
        根据类型渲染模板
        
        Args:
            category: 模板分类
            template_type: 模板类型
            context: 模板上下文
            
        Returns:
            渲染后的内容
        """
        template_name = self.get_template(category, template_type)
        if not template_name:
            raise ValueError(f"Template not found: {category}.{template_type}")
        
        return self.engine.render_template(template_name, context)
    
    def list_template_types(self) -> Dict[str, List[str]]:
        """
        列出所有模板类型
        
        Returns:
            按分类组织的模板类型字典
        """
        return {category: list(templates.keys()) 
                for category, templates in self.template_registry.items()}


# 工具函数
def create_default_engine(template_dir: Optional[Path] = None) -> TemplateEngine:
    """
    创建默认配置的模板引擎
    
    Args:
        template_dir: 模板目录，如果为None则使用默认目录
        
    Returns:
        配置好的模板引擎实例
    """
    template_dirs = [template_dir] if template_dir else None
    return TemplateEngine(template_dirs)


def render_template_file(template_path: Path, context: Dict[str, Any], 
                        output_path: Optional[Path] = None) -> str:
    """
    渲染模板文件的便捷函数
    
    Args:
        template_path: 模板文件路径
        context: 模板上下文
        output_path: 输出文件路径，如果提供则写入文件
        
    Returns:
        渲染后的内容
    """
    # 创建模板引擎
    template_dir = template_path.parent
    engine = TemplateEngine([template_dir])
    
    # 渲染模板
    content = engine.render_template(template_path.name, context)
    
    # 如果指定了输出路径，写入文件
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Rendered template saved to: {output_path}")
    
    return content


if __name__ == "__main__":
    # 测试模板引擎
    engine = create_default_engine()
    
    # 测试字符串渲染
    template_str = """
# {{ title | upper }}

生成时间：{{ now().strftime('%Y-%m-%d %H:%M:%S') }}

{% for item in items %}
- {{ item | snake_to_title }}
{% endfor %}

代码示例：
{{ code | code_block('python') }}
"""
    
    context = {
        'title': 'test template',
        'items': ['hello_world', 'test_function', 'api_endpoint'],
        'code': 'def hello():\n    print("Hello, World!")'
    }
    
    result = engine.render_string(template_str, context)
    print("Template rendering test:")
    print(result)