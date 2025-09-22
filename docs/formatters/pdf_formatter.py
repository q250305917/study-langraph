"""
PDF格式化器
功能：将文档内容格式化为PDF格式，支持目录、页眉页脚和自定义样式
作者：自动文档生成系统
"""

import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .base_formatter import BaseFormatter, FormatterConfig, formatter_registry
from .html_formatter import HTMLFormatter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFFormatter(BaseFormatter):
    """PDF格式化器 - 基于HTML到PDF的转换"""
    
    def __init__(self, config: FormatterConfig):
        """
        初始化PDF格式化器
        
        Args:
            config: 格式化器配置
        """
        super().__init__(config)
        
        # PDF特定配置
        self.page_size = "A4"
        self.orientation = "portrait"  # portrait 或 landscape
        self.margin_top = "1in"
        self.margin_bottom = "1in"
        self.margin_left = "1in"
        self.margin_right = "1in"
        self.include_header = True
        self.include_footer = True
        self.enable_toc = config.include_toc
        
        # HTML格式化器用于中间步骤
        html_config = FormatterConfig(
            output_format='html',
            theme=config.theme,
            include_toc=config.include_toc,
            include_nav=False,  # PDF中通常不需要导航栏
            syntax_highlight=config.syntax_highlight,
            custom_css=config.custom_css
        )
        self.html_formatter = HTMLFormatter(html_config)
        
        # 检查PDF生成工具
        self.pdf_engine = self._detect_pdf_engine()
    
    def format_content(self, content: str, **kwargs) -> str:
        """
        格式化内容为PDF（实际返回HTML，在save_to_file中转换为PDF）
        
        Args:
            content: 要格式化的内容
            **kwargs: 额外参数
                
        Returns:
            中间HTML内容（用于PDF转换）
        """
        try:
            # 使用HTML格式化器处理内容
            html_content = self.html_formatter.format_content(content, **kwargs)
            
            # 添加PDF特定的样式
            html_content = self._add_pdf_styles(html_content)
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error formatting PDF content: {e}")
            return content
    
    def save_to_file(self, content: str, output_path: Path, **kwargs) -> bool:
        """
        保存PDF内容到文件
        
        Args:
            content: HTML内容（将转换为PDF）
            output_path: 输出文件路径
            **kwargs: 额外参数，支持：
                - page_size: 页面大小
                - orientation: 页面方向
                - margins: 页边距
                
        Returns:
            保存是否成功
        """
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 确保文件扩展名正确
            if output_path.suffix.lower() != '.pdf':
                output_path = output_path.with_suffix('.pdf')
            
            # 更新PDF选项
            self._update_pdf_options(kwargs)
            
            # 转换为PDF
            success = self._convert_html_to_pdf(content, output_path)
            
            if success:
                logger.info(f"PDF file saved to: {output_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving PDF file: {e}")
            return False
    
    def _detect_pdf_engine(self) -> Optional[str]:
        """
        检测可用的PDF生成引擎
        
        Returns:
            可用的PDF引擎名称或None
        """
        engines = [
            ('wkhtmltopdf', ['wkhtmltopdf', '--version']),
            ('weasyprint', ['weasyprint', '--version']),
            ('puppeteer', ['node', '-e', 'require("puppeteer"); console.log("ok")']),
            ('playwright', ['python', '-c', 'import playwright; print("ok")'])
        ]
        
        for engine_name, test_cmd in engines:
            try:
                result = subprocess.run(
                    test_cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"Found PDF engine: {engine_name}")
                    return engine_name
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        logger.warning("No PDF generation engine found. Available options: wkhtmltopdf, weasyprint, puppeteer, playwright")
        return None
    
    def _add_pdf_styles(self, html_content: str) -> str:
        """
        添加PDF特定的样式
        
        Args:
            html_content: HTML内容
            
        Returns:
            添加了PDF样式的HTML内容
        """
        pdf_css = """
        <style>
        /* PDF特定样式 */
        @media print {
            body {
                font-size: 12pt;
                line-height: 1.4;
            }
            
            .sidebar {
                display: none; /* 在PDF中隐藏侧边栏 */
            }
            
            .main-content {
                flex-direction: column;
            }
            
            .content {
                width: 100%;
            }
            
            h1, h2, h3, h4, h5, h6 {
                page-break-after: avoid;
                break-after: avoid;
            }
            
            pre, blockquote {
                page-break-inside: avoid;
                break-inside: avoid;
            }
            
            .page-break {
                page-break-before: always;
                break-before: page;
            }
            
            .no-break {
                page-break-inside: avoid;
                break-inside: avoid;
            }
            
            /* 目录样式 */
            .table-of-contents {
                page-break-after: always;
                break-after: page;
            }
            
            .table-of-contents a {
                color: black;
                text-decoration: none;
            }
            
            .table-of-contents a::after {
                content: leader('.') target-counter(attr(href), page);
            }
        }
        
        /* 页眉页脚 */
        @page {
            size: """ + self.page_size + """;
            margin: """ + self.margin_top + """ """ + self.margin_right + """ """ + self.margin_bottom + """ """ + self.margin_left + """;
            
            @top-center {
                content: "文档标题";
                font-size: 10pt;
                color: #666;
            }
            
            @bottom-center {
                content: "第 " counter(page) " 页，共 " counter(pages) " 页";
                font-size: 10pt;
                color: #666;
            }
        }
        </style>
        """
        
        # 在</head>标签前插入PDF样式
        if '</head>' in html_content:
            html_content = html_content.replace('</head>', pdf_css + '\n</head>')
        else:
            # 如果没有head标签，在body前添加
            html_content = pdf_css + '\n' + html_content
        
        return html_content
    
    def _update_pdf_options(self, kwargs: Dict[str, Any]):
        """
        更新PDF选项
        
        Args:
            kwargs: 用户提供的选项
        """
        if 'page_size' in kwargs:
            self.page_size = kwargs['page_size']
        
        if 'orientation' in kwargs:
            self.orientation = kwargs['orientation']
        
        if 'margins' in kwargs:
            margins = kwargs['margins']
            if isinstance(margins, dict):
                self.margin_top = margins.get('top', self.margin_top)
                self.margin_bottom = margins.get('bottom', self.margin_bottom)
                self.margin_left = margins.get('left', self.margin_left)
                self.margin_right = margins.get('right', self.margin_right)
            elif isinstance(margins, str):
                # 所有边距相同
                self.margin_top = self.margin_bottom = self.margin_left = self.margin_right = margins
    
    def _convert_html_to_pdf(self, html_content: str, output_path: Path) -> bool:
        """
        将HTML内容转换为PDF
        
        Args:
            html_content: HTML内容
            output_path: 输出文件路径
            
        Returns:
            转换是否成功
        """
        if not self.pdf_engine:
            logger.error("No PDF generation engine available")
            return False
        
        try:
            if self.pdf_engine == 'wkhtmltopdf':
                return self._convert_with_wkhtmltopdf(html_content, output_path)
            elif self.pdf_engine == 'weasyprint':
                return self._convert_with_weasyprint(html_content, output_path)
            elif self.pdf_engine == 'puppeteer':
                return self._convert_with_puppeteer(html_content, output_path)
            elif self.pdf_engine == 'playwright':
                return self._convert_with_playwright(html_content, output_path)
            else:
                logger.error(f"Unsupported PDF engine: {self.pdf_engine}")
                return False
                
        except Exception as e:
            logger.error(f"Error converting HTML to PDF: {e}")
            return False
    
    def _convert_with_wkhtmltopdf(self, html_content: str, output_path: Path) -> bool:
        """使用wkhtmltopdf转换"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(html_content)
            tmp_html_path = tmp_file.name
        
        try:
            cmd = [
                'wkhtmltopdf',
                '--page-size', self.page_size,
                '--orientation', self.orientation,
                '--margin-top', self.margin_top,
                '--margin-bottom', self.margin_bottom,
                '--margin-left', self.margin_left,
                '--margin-right', self.margin_right,
                '--encoding', 'UTF-8'
            ]
            
            if self.enable_toc:
                cmd.append('--enable-toc')
            
            cmd.extend([tmp_html_path, str(output_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"wkhtmltopdf error: {result.stderr}")
                return False
                
        finally:
            Path(tmp_html_path).unlink(missing_ok=True)
    
    def _convert_with_weasyprint(self, html_content: str, output_path: Path) -> bool:
        """使用WeasyPrint转换"""
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            # 创建字体配置
            font_config = FontConfiguration()
            
            # 添加自定义CSS
            css_content = f"""
            @page {{
                size: {self.page_size};
                margin: {self.margin_top} {self.margin_right} {self.margin_bottom} {self.margin_left};
            }}
            """
            
            # 转换为PDF
            html_doc = HTML(string=html_content)
            css_doc = CSS(string=css_content, font_config=font_config)
            
            html_doc.write_pdf(str(output_path), stylesheets=[css_doc], font_config=font_config)
            return True
            
        except ImportError:
            logger.error("WeasyPrint not installed. Please install it with: pip install weasyprint")
            return False
    
    def _convert_with_puppeteer(self, html_content: str, output_path: Path) -> bool:
        """使用Puppeteer转换"""
        # 创建临时HTML文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(html_content)
            tmp_html_path = tmp_file.name
        
        # 创建Node.js脚本
        js_script = f"""
        const puppeteer = require('puppeteer');
        const path = require('path');

        (async () => {{
            const browser = await puppeteer.launch();
            const page = await browser.newPage();
            
            await page.goto('file://{tmp_html_path}', {{ waitUntil: 'networkidle0' }});
            
            await page.pdf({{
                path: '{output_path}',
                format: '{self.page_size}',
                margin: {{
                    top: '{self.margin_top}',
                    bottom: '{self.margin_bottom}',
                    left: '{self.margin_left}',
                    right: '{self.margin_right}'
                }},
                printBackground: true
            }});
            
            await browser.close();
            console.log('PDF generated successfully');
        }})();
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as tmp_js:
            tmp_js.write(js_script)
            tmp_js_path = tmp_js.name
        
        try:
            result = subprocess.run(
                ['node', tmp_js_path], 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"Puppeteer error: {result.stderr}")
                return False
                
        finally:
            Path(tmp_html_path).unlink(missing_ok=True)
            Path(tmp_js_path).unlink(missing_ok=True)
    
    def _convert_with_playwright(self, html_content: str, output_path: Path) -> bool:
        """使用Playwright转换"""
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                
                # 设置内容
                page.set_content(html_content)
                
                # 等待页面加载完成
                page.wait_for_load_state('networkidle')
                
                # 生成PDF
                page.pdf(
                    path=str(output_path),
                    format=self.page_size,
                    margin={
                        'top': self.margin_top,
                        'bottom': self.margin_bottom,
                        'left': self.margin_left,
                        'right': self.margin_right
                    },
                    print_background=True
                )
                
                browser.close()
                return True
                
        except ImportError:
            logger.error("Playwright not installed. Please install it with: pip install playwright")
            return False
    
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
            'page_breaks',
            'headers_footers',
            'custom_margins',
            'page_numbering'
        ]


# 注册PDF格式化器
formatter_registry.register('pdf', PDFFormatter)


# 便捷函数
def create_pdf_formatter(config: Optional[FormatterConfig] = None) -> PDFFormatter:
    """
    创建PDF格式化器的便捷函数
    
    Args:
        config: 格式化器配置
        
    Returns:
        PDF格式化器实例
    """
    if config is None:
        config = FormatterConfig(output_format='pdf')
    
    return PDFFormatter(config)


def check_pdf_dependencies() -> Dict[str, bool]:
    """
    检查PDF生成依赖
    
    Returns:
        各种PDF引擎的可用性状态
    """
    dependencies = {}
    
    # 检查wkhtmltopdf
    try:
        subprocess.run(['wkhtmltopdf', '--version'], capture_output=True, timeout=5)
        dependencies['wkhtmltopdf'] = True
    except:
        dependencies['wkhtmltopdf'] = False
    
    # 检查WeasyPrint
    try:
        import weasyprint
        dependencies['weasyprint'] = True
    except ImportError:
        dependencies['weasyprint'] = False
    
    # 检查Puppeteer
    try:
        result = subprocess.run(
            ['node', '-e', 'require("puppeteer"); console.log("ok")'],
            capture_output=True, timeout=5
        )
        dependencies['puppeteer'] = result.returncode == 0
    except:
        dependencies['puppeteer'] = False
    
    # 检查Playwright
    try:
        import playwright
        dependencies['playwright'] = True
    except ImportError:
        dependencies['playwright'] = False
    
    return dependencies


if __name__ == "__main__":
    # 测试PDF格式化器
    print("Checking PDF dependencies:")
    deps = check_pdf_dependencies()
    for engine, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {engine}")
    
    if any(deps.values()):
        config = FormatterConfig(output_format='pdf', include_toc=True)
        formatter = PDFFormatter(config)
        
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
            title="测试PDF文档"
        )
        
        print("\nFormatted content ready for PDF conversion")
    else:
        print("\nNo PDF generation engines available. Please install one of:")
        print("  - wkhtmltopdf")
        print("  - weasyprint (pip install weasyprint)")
        print("  - puppeteer (npm install puppeteer)")
        print("  - playwright (pip install playwright)")