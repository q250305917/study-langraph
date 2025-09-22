#!/usr/bin/env python3
"""
文档生成系统使用示例
功能：演示如何使用文档自动生成系统的各种功能
作者：自动文档生成系统
"""

import sys
from pathlib import Path

# 添加generator模块到路径
sys.path.insert(0, str(Path(__file__).parent))

from generator import (
    CodeParser,
    APIDocumentationGenerator,
    APIDocumentationConfig,
    TutorialGenerator,
    TutorialConfig,
    TemplateEngine,
    ConfigManager,
    DocumentationConfig,
    MarkdownFormatter,
    HTMLFormatter,
    FormatterConfig,
    generate_api_docs,
    generate_tutorials,
    generate_all_docs
)


def example_code_parsing():
    """示例1：代码解析功能"""
    print("=" * 50)
    print("示例1：代码解析功能")
    print("=" * 50)
    
    # 创建代码解析器
    parser = CodeParser()
    
    # 解析单个文件
    file_path = Path("../src/langchain_learning/__init__.py")
    if file_path.exists():
        print(f"正在解析文件: {file_path}")
        module = parser.parse_file(file_path)
        
        if module:
            print(f"模块名称: {module.name}")
            print(f"文档字符串: {module.docstring.content if module.docstring else '无'}")
            print(f"类数量: {len([c for c in module.children if c.node_type.value == 'class'])}")
            print(f"函数数量: {len([c for c in module.children if c.node_type.value == 'function'])}")
        else:
            print("文件解析失败")
    else:
        print(f"文件不存在: {file_path}")
    
    print()


def example_api_documentation():
    """示例2：API文档生成"""
    print("=" * 50)
    print("示例2：API文档生成")
    print("=" * 50)
    
    # 创建API文档配置
    api_config = APIDocumentationConfig(
        include_private=False,
        include_source=True,
        include_inheritance=True,
        include_examples=True,
        sort_members=True
    )
    
    # 创建API文档生成器
    generator = APIDocumentationGenerator(api_config)
    
    # 设置源路径和输出路径
    source_path = Path("../src/langchain_learning/")
    output_path = Path("output/api_docs/")
    
    if source_path.exists():
        print(f"正在生成API文档...")
        print(f"源路径: {source_path}")
        print(f"输出路径: {output_path}")
        
        success = generator.generate_documentation(source_path, output_path)
        
        if success:
            print("✓ API文档生成成功!")
            print(f"文档位置: {output_path}")
        else:
            print("✗ API文档生成失败")
    else:
        print(f"源路径不存在: {source_path}")
    
    print()


def example_tutorial_generation():
    """示例3：教程文档生成"""
    print("=" * 50)
    print("示例3：教程文档生成")
    print("=" * 50)
    
    # 创建教程配置
    tutorial_config = TutorialConfig(
        auto_generate=True,
        include_output=True,
        code_execution=False,
        language="zh",
        difficulty_levels=["初级", "中级", "高级"],
        format_style="step_by_step"
    )
    
    # 创建教程生成器
    generator = TutorialGenerator(tutorial_config)
    
    # 设置示例路径和输出路径
    examples_path = Path("../examples/")
    output_path = Path("output/tutorials/")
    
    if examples_path.exists():
        print(f"正在生成教程文档...")
        print(f"示例路径: {examples_path}")
        print(f"输出路径: {output_path}")
        
        success = generator.generate_tutorial_documentation(examples_path, output_path)
        
        if success:
            print("✓ 教程文档生成成功!")
            print(f"文档位置: {output_path}")
        else:
            print("✗ 教程文档生成失败")
    else:
        print(f"示例路径不存在: {examples_path}")
        print("可以创建一些示例文件来测试教程生成功能")
    
    print()


def example_template_usage():
    """示例4：模板引擎使用"""
    print("=" * 50)
    print("示例4：模板引擎使用")
    print("=" * 50)
    
    # 创建模板引擎
    engine = TemplateEngine()
    
    # 使用字符串模板
    template_string = """
# {{ title }}

作者：{{ author }}
日期：{{ today }}

## 描述
{{ description }}

## 代码示例
```python
{{ code | code_block('python') }}
```

## 参数列表
{% for param in parameters %}
- **{{ param.name }}**: {{ param.description }}
{% endfor %}
"""
    
    # 准备上下文数据
    context = {
        'title': '示例文档',
        'author': '文档生成系统',
        'description': '这是一个使用模板引擎生成的示例文档',
        'code': 'def hello():\n    print("Hello, World!")',
        'parameters': [
            {'name': 'name', 'description': '用户名称'},
            {'name': 'age', 'description': '用户年龄'}
        ]
    }
    
    # 渲染模板
    result = engine.render_string(template_string, context)
    
    print("模板渲染结果:")
    print("-" * 30)
    print(result)
    print("-" * 30)
    print()


def example_formatters():
    """示例5：多格式输出"""
    print("=" * 50)
    print("示例5：多格式输出")
    print("=" * 50)
    
    # 准备测试内容
    test_content = """
# 测试文档

这是一个测试文档，用于演示多格式输出功能。

## 功能特性

- 支持Markdown格式
- 支持HTML格式  
- 支持PDF格式（需要额外依赖）

## 代码示例

```python
def example_function():
    print("这是一个示例函数")
    return True
```

## 表格示例

| 格式 | 支持 | 说明 |
|------|------|------|
| Markdown | ✓ | 默认支持 |
| HTML | ✓ | 内置支持 |
| PDF | △ | 需要额外工具 |
"""
    
    # 输出目录
    output_dir = Path("output/formats/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试Markdown格式化器
    print("正在生成Markdown格式...")
    md_config = FormatterConfig(output_format='markdown', include_toc=True)
    md_formatter = MarkdownFormatter(md_config)
    
    md_result = md_formatter.format_content(
        test_content,
        title="测试文档",
        metadata={'author': '文档生成系统', 'version': '1.0'}
    )
    
    md_file = output_dir / "test_document.md"
    md_formatter.save_to_file(md_result, md_file)
    print(f"✓ Markdown文档已保存: {md_file}")
    
    # 测试HTML格式化器
    print("正在生成HTML格式...")
    html_config = FormatterConfig(output_format='html', include_toc=True)
    html_formatter = HTMLFormatter(html_config)
    
    html_result = html_formatter.format_content(
        test_content,
        title="测试文档",
        metadata={'author': '文档生成系统', 'description': '测试HTML输出'}
    )
    
    html_file = output_dir / "test_document.html"
    html_formatter.save_to_file(html_result, html_file)
    print(f"✓ HTML文档已保存: {html_file}")
    
    # 测试PDF格式化器（如果可用）
    try:
        from generator.formatters.pdf_formatter import PDFFormatter, check_pdf_dependencies
        
        pdf_deps = check_pdf_dependencies()
        if any(pdf_deps.values()):
            print("正在生成PDF格式...")
            pdf_config = FormatterConfig(output_format='pdf', include_toc=True)
            pdf_formatter = PDFFormatter(pdf_config)
            
            pdf_result = pdf_formatter.format_content(
                test_content,
                title="测试文档"
            )
            
            pdf_file = output_dir / "test_document.pdf"
            if pdf_formatter.save_to_file(pdf_result, pdf_file):
                print(f"✓ PDF文档已保存: {pdf_file}")
            else:
                print("✗ PDF生成失败")
        else:
            print("△ PDF生成不可用（缺少依赖）")
            print("  可用的PDF引擎:", list(pdf_deps.keys()))
    except Exception as e:
        print(f"△ PDF测试失败: {e}")
    
    print()


def example_config_management():
    """示例6：配置管理"""
    print("=" * 50)
    print("示例6：配置管理")
    print("=" * 50)
    
    # 创建配置管理器
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print("当前配置:")
    print(f"  项目名称: {config.project_name}")
    print(f"  项目版本: {config.project_version}")
    print(f"  源目录: {config.source.directories}")
    print(f"  输出目录: {config.output.directory}")
    print(f"  输出格式: {config.output.formats}")
    print(f"  包含私有成员: {config.api_docs.include_private}")
    print(f"  教程语言: {config.tutorials.language}")
    
    # 创建示例配置文件
    from generator.config import create_config_template
    
    config_file = Path("output/example_config.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在创建示例配置文件: {config_file}")
    create_config_template(config_file)
    print("✓ 示例配置文件已创建")
    
    print()


def example_convenience_functions():
    """示例7：便捷函数使用"""
    print("=" * 50)
    print("示例7：便捷函数使用")
    print("=" * 50)
    
    # 使用便捷函数生成API文档
    source_path = "../src/langchain_learning/"
    api_output = "output/convenience/api/"
    
    if Path(source_path).exists():
        print("使用便捷函数生成API文档...")
        success = generate_api_docs(source_path, api_output)
        if success:
            print(f"✓ API文档生成成功: {api_output}")
        else:
            print("✗ API文档生成失败")
    else:
        print(f"源路径不存在: {source_path}")
    
    # 使用便捷函数生成教程文档
    examples_path = "../examples/"
    tutorial_output = "output/convenience/tutorials/"
    
    if Path(examples_path).exists():
        print("使用便捷函数生成教程文档...")
        success = generate_tutorials(examples_path, tutorial_output)
        if success:
            print(f"✓ 教程文档生成成功: {tutorial_output}")
        else:
            print("✗ 教程文档生成失败")
    else:
        print(f"示例路径不存在: {examples_path}")
    
    print()


def create_sample_files():
    """创建示例文件用于测试"""
    print("=" * 50)
    print("创建测试文件")
    print("=" * 50)
    
    # 创建示例源文件
    sample_dir = Path("sample_project")
    sample_dir.mkdir(exist_ok=True)
    
    # 创建示例Python模块
    sample_module = sample_dir / "sample_module.py"
    with open(sample_module, 'w', encoding='utf-8') as f:
        f.write('''"""
示例模块
功能：演示文档生成功能
作者：文档生成系统
"""

class SampleClass:
    """示例类，用于演示API文档生成功能"""
    
    def __init__(self, name: str, age: int = 0):
        """
        初始化示例类
        
        Args:
            name: 用户名称
            age: 用户年龄，默认为0
        """
        self.name = name
        self.age = age
    
    def greet(self, message: str = "Hello") -> str:
        """
        返回问候消息
        
        Args:
            message: 问候消息，默认为"Hello"
            
        Returns:
            格式化的问候字符串
            
        Examples:
            >>> obj = SampleClass("World", 25)
            >>> obj.greet()
            'Hello, World! (25岁)'
            >>> obj.greet("Hi")
            'Hi, World! (25岁)'
        """
        return f"{message}, {self.name}! ({self.age}岁)"
    
    @property
    def info(self) -> str:
        """获取用户信息"""
        return f"{self.name}, {self.age}岁"
    
    @staticmethod
    def validate_name(name: str) -> bool:
        """
        验证名称是否有效
        
        Args:
            name: 要验证的名称
            
        Returns:
            名称是否有效
        """
        return len(name) > 0 and name.isalpha()


def calculate_age_in_days(years: int) -> int:
    """
    计算年龄对应的天数（简化计算）
    
    Args:
        years: 年龄（年）
        
    Returns:
        大概的天数
        
    Examples:
        >>> calculate_age_in_days(1)
        365
        >>> calculate_age_in_days(2)
        730
    """
    return years * 365


# 模块常量
DEFAULT_GREETING = "Hello"
MAX_AGE = 150
''')
    
    # 创建示例教程文件
    tutorial_dir = sample_dir / "tutorials"
    tutorial_dir.mkdir(exist_ok=True)
    
    tutorial_file = tutorial_dir / "basic_tutorial.py"
    with open(tutorial_file, 'w', encoding='utf-8') as f:
        f.write('''"""
基础使用教程
标题：SampleClass基础使用指南
作者：文档生成系统
难度：初级
时间：5分钟
类型：tutorial
标签：基础,入门,示例
"""

# 步骤1：导入模块
# 首先导入我们需要使用的类

from sample_module import SampleClass, calculate_age_in_days

# 步骤2：创建实例
# 创建一个SampleClass的实例

person = SampleClass("张三", 25)

# 步骤3：使用方法
# 调用实例的方法获取问候消息

greeting = person.greet()
print(f"默认问候: {greeting}")

custom_greeting = person.greet("你好")
print(f"自定义问候: {custom_greeting}")

# 步骤4：使用属性
# 访问实例的属性

info = person.info
print(f"用户信息: {info}")

# 步骤5：使用静态方法
# 调用类的静态方法

is_valid = SampleClass.validate_name("张三")
print(f"名称有效性: {is_valid}")

# 步骤6：使用模块函数
# 调用模块级别的函数

days = calculate_age_in_days(25)
print(f"25岁大约等于 {days} 天")

# 总结
# 通过这个教程，我们学会了：
# 1. 如何创建类实例
# 2. 如何调用实例方法
# 3. 如何访问属性
# 4. 如何使用静态方法
# 5. 如何调用模块函数
''')
    
    print(f"✓ 示例文件已创建: {sample_dir}")
    return sample_dir


def run_all_examples():
    """运行所有示例"""
    print("文档生成系统使用示例")
    print("=" * 70)
    
    # 首先创建示例文件
    sample_dir = create_sample_files()
    
    # 运行各种示例
    example_code_parsing()
    example_template_usage()
    example_formatters()
    example_config_management()
    
    # 使用示例文件测试生成功能
    print("=" * 50)
    print("使用示例文件测试文档生成")
    print("=" * 50)
    
    # 测试API文档生成
    print("测试API文档生成...")
    api_success = generate_api_docs(
        sample_dir / "sample_module.py",
        "output/sample_api/"
    )
    if api_success:
        print("✓ 示例API文档生成成功")
    else:
        print("✗ 示例API文档生成失败")
    
    # 测试教程文档生成
    print("测试教程文档生成...")
    tutorial_success = generate_tutorials(
        sample_dir / "tutorials/",
        "output/sample_tutorials/"
    )
    if tutorial_success:
        print("✓ 示例教程文档生成成功")
    else:
        print("✗ 示例教程文档生成失败")
    
    print("\n" + "=" * 70)
    print("所有示例运行完成！")
    print("生成的文件位于 output/ 目录中")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()