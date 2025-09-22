"""
HTML格式化器
功能：将文档内容格式化为HTML格式，支持主题、样式和交互功能
作者：自动文档生成系统
"""

import re
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .base_formatter import BaseFormatter, FormatterConfig, formatter_registry

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTMLFormatter(BaseFormatter):
    """HTML格式化器"""
    
    def __init__(self, config: FormatterConfig):
        """
        初始化HTML格式化器
        
        Args:
            config: 格式化器配置
        """
        super().__init__(config)
        
        # HTML特定配置
        self.theme = config.theme
        self.include_navigation = config.include_nav
        self.include_toc = config.include_toc
        self.syntax_highlight = config.syntax_highlight
        
        # HTML模板组件
        self.html_template = self._load_html_template()
        self.css_styles = self._load_css_styles()
        self.js_scripts = self._load_js_scripts()
    
    def format_content(self, content: str, **kwargs) -> str:
        """
        格式化内容为HTML
        
        Args:
            content: 要格式化的内容（Markdown格式）
            **kwargs: 额外参数，支持：
                - title: 文档标题
                - toc: 是否生成目录
                - metadata: 文档元数据
                - sidebar: 侧边栏内容
                
        Returns:
            格式化后的HTML内容
        """
        try:
            # 获取参数
            title = kwargs.get('title', '文档')
            include_toc = kwargs.get('toc', self.include_toc)
            metadata = kwargs.get('metadata', {})
            sidebar = kwargs.get('sidebar', '')
            
            # 将Markdown转换为HTML
            html_content = self._markdown_to_html(content)
            
            # 生成目录
            toc_html = ""
            if include_toc:
                toc_html = self._generate_toc_html(html_content)
            
            # 构建完整HTML文档
            full_html = self._build_html_document(
                title=title,
                content=html_content,
                toc=toc_html,
                sidebar=sidebar,
                metadata=metadata
            )
            
            return full_html
            
        except Exception as e:
            logger.error(f"Error formatting HTML content: {e}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
    def save_to_file(self, content: str, output_path: Path, **kwargs) -> bool:
        """
        保存HTML内容到文件
        
        Args:
            content: HTML内容
            output_path: 输出文件路径
            **kwargs: 额外参数，支持：
                - copy_assets: 是否复制CSS/JS资源文件
                
        Returns:
            保存是否成功
        """
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 确保文件扩展名正确
            if output_path.suffix.lower() != '.html':
                output_path = output_path.with_suffix('.html')
            
            # 写入HTML文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 复制资源文件
            if kwargs.get('copy_assets', True):
                self._copy_assets(output_path.parent)
            
            logger.info(f"HTML file saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving HTML file: {e}")
            return False
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """
        将Markdown转换为HTML
        
        Args:
            markdown_content: Markdown内容
            
        Returns:
            HTML内容
        """
        html = markdown_content
        
        # 处理标题
        html = self._convert_headings(html)
        
        # 处理代码块
        html = self._convert_code_blocks(html)
        
        # 处理行内代码
        html = self._convert_inline_code(html)
        
        # 处理强调和粗体
        html = self._convert_emphasis(html)
        
        # 处理链接
        html = self._convert_links(html)
        
        # 处理图片
        html = self._convert_images(html)
        
        # 处理列表
        html = self._convert_lists(html)
        
        # 处理表格
        html = self._convert_tables(html)
        
        # 处理段落
        html = self._convert_paragraphs(html)
        
        return html
    
    def _convert_headings(self, content: str) -> str:
        """转换标题"""
        def heading_replacer(match):
            level = len(match.group(1))
            text = match.group(2).strip()
            anchor = self._create_anchor_id(text)
            return f'<h{level} id="{anchor}">{text}</h{level}>'
        
        return re.sub(r'^(#{1,6})\s+(.+)$', heading_replacer, content, flags=re.MULTILINE)
    
    def _convert_code_blocks(self, content: str) -> str:
        """转换代码块"""
        def code_block_replacer(match):
            language = match.group(1) or 'python'
            code = match.group(2)
            
            # HTML转义
            code = self._escape_html(code)
            
            # 语法高亮
            if self.syntax_highlight:
                return f'<pre><code class="language-{language}" data-lang="{language}">{code}</code></pre>'
            else:
                return f'<pre><code>{code}</code></pre>'
        
        return re.sub(r'```(\w*)\n(.*?)\n```', code_block_replacer, content, flags=re.DOTALL)
    
    def _convert_inline_code(self, content: str) -> str:
        """转换行内代码"""
        def inline_code_replacer(match):
            code = self._escape_html(match.group(1))
            return f'<code>{code}</code>'
        
        return re.sub(r'`([^`]+)`', inline_code_replacer, content)
    
    def _convert_emphasis(self, content: str) -> str:
        """转换强调和粗体"""
        # 粗体
        content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', content)
        
        # 斜体
        content = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', content)
        content = re.sub(r'_([^_]+)_', r'<em>\1</em>', content)
        
        return content
    
    def _convert_links(self, content: str) -> str:
        """转换链接"""
        # Markdown链接格式 [text](url)
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', content)
        
        # 自动链接
        url_pattern = r'(https?://[^\s<>"]+)'
        content = re.sub(url_pattern, r'<a href="\1">\1</a>', content)
        
        return content
    
    def _convert_images(self, content: str) -> str:
        """转换图片"""
        # Markdown图片格式 ![alt](src)
        def img_replacer(match):
            alt = match.group(1)
            src = match.group(2)
            return f'<img src="{src}" alt="{alt}" class="doc-image">'
        
        return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_replacer, content)
    
    def _convert_lists(self, content: str) -> str:
        """转换列表"""
        lines = content.split('\n')
        result_lines = []
        in_ul = False
        in_ol = False
        
        for line in lines:
            stripped = line.strip()
            
            # 无序列表
            if re.match(r'^[-*+]\s+', stripped):
                if not in_ul:
                    if in_ol:
                        result_lines.append('</ol>')
                        in_ol = False
                    result_lines.append('<ul>')
                    in_ul = True
                
                item_text = re.sub(r'^[-*+]\s+', '', stripped)
                result_lines.append(f'  <li>{item_text}</li>')
            
            # 有序列表
            elif re.match(r'^\d+\.\s+', stripped):
                if not in_ol:
                    if in_ul:
                        result_lines.append('</ul>')
                        in_ul = False
                    result_lines.append('<ol>')
                    in_ol = True
                
                item_text = re.sub(r'^\d+\.\s+', '', stripped)
                result_lines.append(f'  <li>{item_text}</li>')
            
            else:
                # 结束列表
                if in_ul:
                    result_lines.append('</ul>')
                    in_ul = False
                if in_ol:
                    result_lines.append('</ol>')
                    in_ol = False
                
                result_lines.append(line)
        
        # 关闭未关闭的列表
        if in_ul:
            result_lines.append('</ul>')
        if in_ol:
            result_lines.append('</ol>')
        
        return '\n'.join(result_lines)
    
    def _convert_tables(self, content: str) -> str:
        """转换表格（简单实现）"""
        # 这里可以实现更复杂的表格转换逻辑
        return content
    
    def _convert_paragraphs(self, content: str) -> str:
        """转换段落"""
        # 将连续的文本行包装为段落
        lines = content.split('\n')
        result_lines = []
        current_paragraph = []
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过已经是HTML标签的行
            if re.match(r'^\s*<[^>]+>', line):
                if current_paragraph:
                    para_content = ' '.join(current_paragraph)
                    if para_content.strip():
                        result_lines.append(f'<p>{para_content}</p>')
                    current_paragraph = []
                result_lines.append(line)
            elif stripped == '':
                # 空行，结束当前段落
                if current_paragraph:
                    para_content = ' '.join(current_paragraph)
                    if para_content.strip():
                        result_lines.append(f'<p>{para_content}</p>')
                    current_paragraph = []
                result_lines.append('')
            else:
                # 普通文本行
                current_paragraph.append(stripped)
        
        # 处理最后一个段落
        if current_paragraph:
            para_content = ' '.join(current_paragraph)
            if para_content.strip():
                result_lines.append(f'<p>{para_content}</p>')
        
        return '\n'.join(result_lines)
    
    def _generate_toc_html(self, html_content: str) -> str:
        """
        生成HTML目录
        
        Args:
            html_content: HTML内容
            
        Returns:
            目录HTML
        """
        # 提取标题
        headings = re.findall(r'<h([1-6])[^>]*id="([^"]*)"[^>]*>([^<]+)</h[1-6]>', html_content)
        
        if not headings:
            return ""
        
        toc_html = ['<nav class="table-of-contents">', '<h3>目录</h3>', '<ul>']
        
        for level, anchor, title in headings:
            level_int = int(level)
            indent_class = f"toc-level-{level_int}"
            toc_html.append(f'  <li class="{indent_class}"><a href="#{anchor}">{title}</a></li>')
        
        toc_html.extend(['</ul>', '</nav>'])
        
        return '\n'.join(toc_html)
    
    def _build_html_document(self, title: str, content: str, toc: str, 
                           sidebar: str, metadata: Dict[str, Any]) -> str:
        """
        构建完整的HTML文档
        
        Args:
            title: 文档标题
            content: 主要内容
            toc: 目录HTML
            sidebar: 侧边栏内容
            metadata: 元数据
            
        Returns:
            完整的HTML文档
        """
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="zh-CN">',
            '<head>',
            f'    <meta charset="UTF-8">',
            f'    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'    <title>{self._escape_html(title)}</title>',
        ]
        
        # 添加元数据
        for key, value in metadata.items():
            if key in ['description', 'keywords', 'author']:
                html_parts.append(f'    <meta name="{key}" content="{self._escape_html(str(value))}">') 
        
        # 添加样式
        html_parts.extend([
            '    <style>',
            self.css_styles,
            '    </style>'
        ])
        
        html_parts.extend([
            '</head>',
            '<body>',
            '    <div class="container">',
        ])
        
        # 添加导航栏
        if self.include_navigation:
            html_parts.extend([
                '        <nav class="navbar">',
                f'            <h1 class="site-title">{self._escape_html(title)}</h1>',
                '        </nav>'
            ])
        
        html_parts.append('        <div class="main-content">')
        
        # 添加侧边栏
        if sidebar or toc:
            html_parts.extend([
                '            <aside class="sidebar">',
                toc,
                sidebar,
                '            </aside>'
            ])
        
        # 添加主要内容
        html_parts.extend([
            '            <main class="content">',
            content,
            '            </main>',
        ])
        
        html_parts.extend([
            '        </div>',
            '    </div>',
        ])
        
        # 添加脚本
        if self.js_scripts:
            html_parts.extend([
                '    <script>',
                self.js_scripts,
                '    </script>'
            ])
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    def _load_html_template(self) -> str:
        """加载HTML模板"""
        # 这里可以从文件加载模板
        return ""
    
    def _load_css_styles(self) -> str:
        """加载CSS样式"""
        # 默认样式
        default_css = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .container {
            display: flex;
            flex-direction: column;
        }
        
        .navbar {
            background-color: #f8f9fa;
            padding: 1rem;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
        
        .site-title {
            margin: 0;
            color: #2c3e50;
        }
        
        .main-content {
            display: flex;
            gap: 2rem;
        }
        
        .sidebar {
            flex: 0 0 300px;
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            height: fit-content;
        }
        
        .content {
            flex: 1;
        }
        
        .table-of-contents h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        
        .table-of-contents ul {
            list-style: none;
            padding-left: 0;
        }
        
        .table-of-contents li {
            margin: 0.5rem 0;
        }
        
        .toc-level-2 { padding-left: 1rem; }
        .toc-level-3 { padding-left: 2rem; }
        .toc-level-4 { padding-left: 3rem; }
        .toc-level-5 { padding-left: 4rem; }
        .toc-level-6 { padding-left: 5rem; }
        
        .table-of-contents a {
            color: #007bff;
            text-decoration: none;
        }
        
        .table-of-contents a:hover {
            text-decoration: underline;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        h1 { border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }
        h2 { border-bottom: 1px solid #eee; padding-bottom: 0.3rem; }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 1rem;
            overflow-x: auto;
        }
        
        code {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 3px;
            padding: 0.2rem 0.4rem;
            font-size: 0.9em;
        }
        
        pre code {
            background: none;
            border: none;
            padding: 0;
        }
        
        .doc-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 0.75rem;
            text-align: left;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
            }
            
            .sidebar {
                flex: none;
            }
        }
        """
        
        # 如果有自定义CSS文件，加载它
        if self.config.custom_css and self.config.custom_css.exists():
            try:
                with open(self.config.custom_css, 'r', encoding='utf-8') as f:
                    custom_css = f.read()
                return default_css + '\n' + custom_css
            except Exception as e:
                logger.warning(f"Failed to load custom CSS: {e}")
        
        return default_css
    
    def _load_js_scripts(self) -> str:
        """加载JavaScript脚本"""
        default_js = """
        // 语法高亮支持
        document.addEventListener('DOMContentLoaded', function() {
            // 可以在这里添加代码高亮库的初始化
            
            // 添加目录点击平滑滚动
            const tocLinks = document.querySelectorAll('.table-of-contents a[href^="#"]');
            tocLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
        });
        """
        
        # 如果有自定义JS文件，加载它
        if self.config.custom_js and self.config.custom_js.exists():
            try:
                with open(self.config.custom_js, 'r', encoding='utf-8') as f:
                    custom_js = f.read()
                return default_js + '\n' + custom_js
            except Exception as e:
                logger.warning(f"Failed to load custom JS: {e}")
        
        return default_js
    
    def _copy_assets(self, output_dir: Path):
        """复制资源文件到输出目录"""
        # 这里可以复制CSS、JS、图片等资源文件
        pass
    
    def _escape_html(self, text: str) -> str:
        """HTML转义"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))
    
    def _create_anchor_id(self, text: str) -> str:
        """创建锚点ID"""
        # 转换为小写，替换特殊字符
        anchor = text.lower()
        anchor = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', anchor)  # 保留中文字符
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor.strip('-')
    
    def get_supported_features(self) -> List[str]:
        """获取支持的特性列表"""
        return [
            'basic_formatting',
            'code_blocks',
            'tables',
            'links',
            'images',
            'emphasis',
            'lists',
            'headings',
            'toc',
            'navigation',
            'themes',
            'syntax_highlighting',
            'responsive_design'
        ]


# 注册HTML格式化器
formatter_registry.register('html', HTMLFormatter)


# 便捷函数
def create_html_formatter(config: Optional[FormatterConfig] = None) -> HTMLFormatter:
    """
    创建HTML格式化器的便捷函数
    
    Args:
        config: 格式化器配置
        
    Returns:
        HTML格式化器实例
    """
    if config is None:
        config = FormatterConfig(output_format='html')
    
    return HTMLFormatter(config)


if __name__ == "__main__":
    # 测试HTML格式化器
    config = FormatterConfig(output_format='html', include_toc=True)
    formatter = HTMLFormatter(config)
    
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

这是第二章的内容，包含一个链接：[GitHub](https://github.com)
"""
    
    result = formatter.format_content(
        test_content,
        title="测试API文档",
        metadata={'author': 'Test', 'description': '这是一个测试文档'}
    )
    
    print("Formatted HTML:")
    print(result)