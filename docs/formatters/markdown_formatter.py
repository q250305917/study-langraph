"""
Markdown格式化器
功能：将文档内容格式化为Markdown格式，支持表格、代码块、链接等
作者：自动文档生成系统
"""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .base_formatter import BaseFormatter, FormatterConfig, formatter_registry

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarkdownFormatter(BaseFormatter):
    """Markdown格式化器"""
    
    def __init__(self, config: FormatterConfig):
        """
        初始化Markdown格式化器
        
        Args:
            config: 格式化器配置
        """
        super().__init__(config)
        
        # Markdown特定配置
        self.use_tables = True
        self.use_code_blocks = True
        self.use_toc = config.include_toc
        self.indent_size = 2
        
    def format_content(self, content: str, **kwargs) -> str:
        """
        格式化内容为Markdown
        
        Args:
            content: 要格式化的内容
            **kwargs: 额外参数，支持：
                - title: 文档标题
                - toc: 是否生成目录
                - metadata: 文档元数据
                
        Returns:
            格式化后的Markdown内容
        """
        try:
            # 获取参数
            title = kwargs.get('title', '')
            include_toc = kwargs.get('toc', self.use_toc)
            metadata = kwargs.get('metadata', {})
            
            # 构建Markdown文档
            markdown_parts = []
            
            # 添加前言元数据（YAML Front Matter）
            if metadata:
                markdown_parts.append(self._format_front_matter(metadata))
            
            # 添加主标题
            if title:
                markdown_parts.append(f"# {title}\n")
            
            # 添加目录
            if include_toc:
                toc = self._generate_toc(content)
                if toc:
                    markdown_parts.append(toc)
            
            # 处理主要内容
            formatted_content = self._process_content(content)
            markdown_parts.append(formatted_content)
            
            return '\n'.join(markdown_parts)
            
        except Exception as e:
            logger.error(f"Error formatting Markdown content: {e}")
            return content
    
    def save_to_file(self, content: str, output_path: Path, **kwargs) -> bool:
        """
        保存Markdown内容到文件
        
        Args:
            content: Markdown内容
            output_path: 输出文件路径
            **kwargs: 额外参数
            
        Returns:
            保存是否成功
        """
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 确保文件扩展名正确
            if output_path.suffix.lower() not in ['.md', '.markdown']:
                output_path = output_path.with_suffix('.md')
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Markdown file saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Markdown file: {e}")
            return False
    
    def _format_front_matter(self, metadata: Dict[str, Any]) -> str:
        """
        格式化YAML Front Matter
        
        Args:
            metadata: 元数据字典
            
        Returns:
            格式化的Front Matter
        """
        if not metadata:
            return ""
        
        lines = ["---"]
        for key, value in metadata.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            elif isinstance(value, str) and '\n' in value:
                lines.append(f"{key}: |")
                for line in value.split('\n'):
                    lines.append(f"  {line}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("---\n")
        
        return '\n'.join(lines)
    
    def _generate_toc(self, content: str) -> str:
        """
        生成目录
        
        Args:
            content: 文档内容
            
        Returns:
            目录Markdown
        """
        # 提取标题
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        
        if not headings:
            return ""
        
        toc_lines = ["## 目录\n"]
        
        for level_str, title in headings:
            level = len(level_str)
            indent = "  " * (level - 1)
            
            # 生成锚点链接
            anchor = self._create_anchor(title)
            toc_lines.append(f"{indent}- [{title}](#{anchor})")
        
        toc_lines.append("")  # 空行
        return '\n'.join(toc_lines)
    
    def _create_anchor(self, text: str) -> str:
        """
        创建锚点链接
        
        Args:
            text: 标题文本
            
        Returns:
            锚点字符串
        """
        # 转换为小写，替换特殊字符
        anchor = text.lower()
        anchor = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', anchor)  # 保留中文字符
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')
    
    def _process_content(self, content: str) -> str:
        """
        处理主要内容
        
        Args:
            content: 原始内容
            
        Returns:
            处理后的内容
        """
        # 处理代码块
        content = self._format_code_blocks(content)
        
        # 处理表格
        content = self._format_tables(content)
        
        # 处理链接
        content = self._format_links(content)
        
        # 处理强调和粗体
        content = self._format_emphasis(content)
        
        # 处理列表
        content = self._format_lists(content)
        
        return content
    
    def _format_code_blocks(self, content: str) -> str:
        """
        格式化代码块
        
        Args:
            content: 内容
            
        Returns:
            格式化后的内容
        """
        if not self.use_code_blocks:
            return content
        
        # 处理行内代码
        content = re.sub(r'`([^`]+)`', r'`\1`', content)
        
        # 处理代码块（如果没有语言标识，添加python）
        def fix_code_block(match):
            code = match.group(1)
            if code.strip():
                return f"```python\n{code}\n```"
            return match.group(0)
        
        content = re.sub(r'```\n(.*?)\n```', fix_code_block, content, flags=re.DOTALL)
        
        return content
    
    def _format_tables(self, content: str) -> str:
        """
        格式化表格（确保Markdown表格格式正确）
        
        Args:
            content: 内容
            
        Returns:
            格式化后的内容
        """
        if not self.use_tables:
            return content
        
        # 这里可以添加表格格式化逻辑
        # 例如，确保表格有正确的分隔符和对齐
        
        return content
    
    def _format_links(self, content: str) -> str:
        """
        格式化链接
        
        Args:
            content: 内容
            
        Returns:
            格式化后的内容
        """
        # 确保链接格式正确
        # 将裸露的URL转换为链接格式
        url_pattern = r'(https?://[^\s<>"]+)'
        content = re.sub(url_pattern, r'[\1](\1)', content)
        
        return content
    
    def _format_emphasis(self, content: str) -> str:
        """
        格式化强调和粗体
        
        Args:
            content: 内容
            
        Returns:
            格式化后的内容
        """
        # 确保强调标记正确
        # 这里可以添加更复杂的强调处理逻辑
        
        return content
    
    def _format_lists(self, content: str) -> str:
        """
        格式化列表
        
        Args:
            content: 内容
            
        Returns:
            格式化后的内容
        """
        # 确保列表格式正确
        # 这里可以添加列表格式化逻辑
        
        return content
    
    def get_supported_features(self) -> List[str]:
        """
        获取支持的特性列表
        
        Returns:
            支持的特性列表
        """
        return [
            'basic_formatting',
            'code_blocks',
            'tables',
            'links',
            'images',
            'emphasis',
            'lists',
            'headings',
            'front_matter',
            'toc',
            'anchors'
        ]
    
    def format_api_documentation(self, api_data: Dict[str, Any]) -> str:
        """
        格式化API文档数据为Markdown
        
        Args:
            api_data: API文档数据
            
        Returns:
            格式化的Markdown文档
        """
        markdown_parts = []
        
        # 模块标题
        module_name = api_data.get('name', 'Unknown Module')
        markdown_parts.append(f"# {module_name} API文档\n")
        
        # 模块描述
        if api_data.get('docstring'):
            markdown_parts.append(f"{api_data['docstring']}\n")
        
        # 模块信息
        if api_data.get('file_info'):
            info = api_data['file_info']
            markdown_parts.append("## 模块信息\n")
            markdown_parts.append(f"- **文件路径**: `{api_data.get('path', '')}`")
            markdown_parts.append(f"- **行数**: {info.get('line_count', 0)}")
            markdown_parts.append(f"- **文件大小**: {info.get('size_bytes', 0)} bytes\n")
        
        # 类文档
        if api_data.get('classes'):
            markdown_parts.append("## 类\n")
            for class_info in api_data['classes']:
                markdown_parts.append(self._format_class_docs(class_info))
        
        # 函数文档
        if api_data.get('functions'):
            markdown_parts.append("## 函数\n")
            for func_info in api_data['functions']:
                markdown_parts.append(self._format_function_docs(func_info))
        
        return '\n'.join(markdown_parts)
    
    def _format_class_docs(self, class_info: Dict[str, Any]) -> str:
        """
        格式化类文档
        
        Args:
            class_info: 类信息
            
        Returns:
            格式化的类文档
        """
        parts = []
        
        # 类标题
        parts.append(f"### {class_info['name']}\n")
        
        # 类描述
        if class_info.get('docstring'):
            parts.append(f"{class_info['docstring']}\n")
        
        # 基类
        if class_info.get('base_classes'):
            bases = ', '.join(f"`{base}`" for base in class_info['base_classes'])
            parts.append(f"**继承自**: {bases}\n")
        
        # 方法
        if class_info.get('methods'):
            parts.append("#### 方法\n")
            for method in class_info['methods']:
                parts.append(self._format_method_docs(method))
        
        # 属性
        if class_info.get('properties'):
            parts.append("#### 属性\n")
            for prop in class_info['properties']:
                parts.append(self._format_property_docs(prop))
        
        return '\n'.join(parts)
    
    def _format_function_docs(self, func_info: Dict[str, Any]) -> str:
        """
        格式化函数文档
        
        Args:
            func_info: 函数信息
            
        Returns:
            格式化的函数文档
        """
        parts = []
        
        # 函数签名
        signature = self._build_function_signature(func_info)
        parts.append(f"### {signature}\n")
        
        # 函数描述
        if func_info.get('docstring'):
            parts.append(f"{func_info['docstring']}\n")
        
        # 参数
        if func_info.get('parameters'):
            parts.append("**参数**:\n")
            for param in func_info['parameters']:
                param_doc = f"- `{param['name']}`"
                if param.get('type'):
                    param_doc += f" ({param['type']})"
                if param.get('default'):
                    param_doc += f" = {param['default']}"
                parts.append(param_doc)
            parts.append("")
        
        # 返回值
        if func_info.get('return_type'):
            parts.append(f"**返回**: {func_info['return_type']}\n")
        
        # 示例
        if func_info.get('examples'):
            parts.append("**示例**:\n")
            for example in func_info['examples']:
                parts.append(f"```python\n{example}\n```\n")
        
        return '\n'.join(parts)
    
    def _format_method_docs(self, method_info: Dict[str, Any]) -> str:
        """格式化方法文档"""
        return self._format_function_docs(method_info)
    
    def _format_property_docs(self, prop_info: Dict[str, Any]) -> str:
        """格式化属性文档"""
        parts = []
        parts.append(f"##### {prop_info['name']}\n")
        if prop_info.get('docstring'):
            parts.append(f"{prop_info['docstring']}\n")
        return '\n'.join(parts)
    
    def _build_function_signature(self, func_info: Dict[str, Any]) -> str:
        """
        构建函数签名
        
        Args:
            func_info: 函数信息
            
        Returns:
            函数签名字符串
        """
        name = func_info['name']
        params = []
        
        for param in func_info.get('parameters', []):
            param_str = param['name']
            if param.get('type'):
                param_str += f": {param['type']}"
            if param.get('default'):
                param_str += f" = {param['default']}"
            params.append(param_str)
        
        signature = f"{name}({', '.join(params)})"
        
        if func_info.get('return_type'):
            signature += f" -> {func_info['return_type']}"
        
        return signature


# 注册Markdown格式化器
formatter_registry.register('markdown', MarkdownFormatter)


# 便捷函数
def create_markdown_formatter(config: Optional[FormatterConfig] = None) -> MarkdownFormatter:
    """
    创建Markdown格式化器的便捷函数
    
    Args:
        config: 格式化器配置
        
    Returns:
        Markdown格式化器实例
    """
    if config is None:
        config = FormatterConfig(output_format='markdown')
    
    return MarkdownFormatter(config)


if __name__ == "__main__":
    # 测试Markdown格式化器
    config = FormatterConfig(output_format='markdown', include_toc=True)
    formatter = MarkdownFormatter(config)
    
    test_content = """
# 测试文档

这是一个测试文档。

## 第一章

这是第一章的内容。

```python
def hello():
    print("Hello, World!")
```

## 第二章

这是第二章的内容。
"""
    
    result = formatter.format_content(
        test_content,
        title="测试API文档",
        metadata={'author': 'Test', 'version': '1.0'}
    )
    
    print("Formatted Markdown:")
    print(result)