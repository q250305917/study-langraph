# 文档自动生成系统

LangChain学习项目的自动化文档生成解决方案，支持从Python代码自动生成API文档、教程文档，并提供多种输出格式。

## 🚀 快速开始

### 一键生成文档
```bash
python docs/quick_start.py
```

### 使用命令行工具
```bash
# 生成所有文档
python -m docs.generator.cli generate --all

# 生成API文档
python -m docs.generator.cli api src/

# 生成教程文档
python -m docs.generator.cli tutorial examples/

# 启动本地服务器预览
python -m docs.generator.cli serve
```

## ✨ 主要功能

### 📖 API文档生成
- 自动提取Python代码的文档字符串
- 支持类型注解和参数信息
- 生成继承关系和交叉引用
- 提取代码示例和用法

### 📚 教程文档生成
- 从示例代码自动生成步骤化教程
- 智能解析代码注释和结构
- 支持多难度级别分类
- 包含代码执行结果

### 🎨 多格式输出
- **Markdown** (.md) - GitHub兼容格式
- **HTML** (.html) - 响应式网页格式
- **PDF** (.pdf) - 打印友好格式

### ⚙️ 灵活配置
- YAML配置文件驱动
- 模板化和主题支持
- 增量更新和缓存
- 监视文件变化自动重生成

## 📁 项目结构

```
docs/
├── generator/              # 核心生成器模块
│   ├── code_parser.py     # 代码解析器
│   ├── api_generator.py   # API文档生成器
│   ├── tutorial_generator.py # 教程生成器
│   ├── template_engine.py # 模板引擎
│   ├── formatters/        # 输出格式化器
│   └── cli.py            # 命令行工具
├── templates/             # 文档模板
├── assets/               # 静态资源
├── example_usage.py      # 使用示例
├── quick_start.py        # 快速开始
└── docs.yaml            # 配置文件
```

## 🛠️ 安装依赖

### 必需依赖
```bash
pip install pyyaml jinja2
```

### 可选依赖（PDF生成）
```bash
# 选择其中一种
pip install weasyprint           # Python PDF库
npm install puppeteer           # Node.js PDF库
pip install playwright          # 微软开发的浏览器自动化库
# 或安装 wkhtmltopdf 系统工具
```

## 📖 配置说明

编辑 `docs.yaml` 文件来自定义生成行为：

```yaml
# 项目信息
project_name: "LangChain学习项目"
project_version: "1.0.0"

# 源代码配置
source:
  directories: ["src/", "examples/"]
  include_patterns: ["*.py"]

# 输出配置
output:
  directory: "docs/generated/"
  formats: ["markdown", "html"]

# API文档配置
api_docs:
  include_private: false
  include_examples: true

# 教程配置
tutorials:
  language: "zh"
  auto_generate: true
```

## 💡 使用示例

### 编程接口
```python
from docs.generator import generate_all_docs, APIDocumentationGenerator

# 使用便捷函数
generate_all_docs(config_path="docs.yaml")

# 细粒度控制
from docs.generator import APIDocumentationConfig
config = APIDocumentationConfig(include_private=False)
generator = APIDocumentationGenerator(config)
generator.generate_documentation("src/", "docs/api/")
```

### 命令行工具
```bash
# 初始化项目
python -m docs.generator.cli init

# 生成配置模板
python -m docs.generator.cli config init

# 验证配置
python -m docs.generator.cli config validate

# 监视文件变化
python -m docs.generator.cli watch --source src/

# 启动文档服务器
python -m docs.generator.cli serve --port 8080
```

## 🎯 核心特性

### 智能代码解析
- 基于AST的深度代码分析
- 支持复杂的Python语法结构
- 提取完整的类型信息和文档

### 强大的模板系统
- 基于Jinja2的模板引擎
- 20+内置过滤器和函数
- 支持模板继承和复用

### 多语言支持
- 中英双语文档生成
- 本地化模板系统
- Unicode字符完美支持

### 性能优化
- 增量更新机制
- 智能缓存系统
- 并行处理大型项目

## 🔧 高级用法

### 自定义模板
在 `docs/templates/` 目录创建自定义模板：

```jinja2
---
title: "{{ module.name }} - API文档"
---

# {{ module.name }}

{{ module.docstring }}

{% for class in module.classes %}
## {{ class.name }}
{{ class.docstring }}
{% endfor %}
```

### 自定义格式化器
```python
from docs.generator.formatters import BaseFormatter

class CustomFormatter(BaseFormatter):
    def format_content(self, content, **kwargs):
        # 自定义格式化逻辑
        return formatted_content
```

### 配置hooks
```yaml
custom:
  pre_generate_hooks:
    - "scripts/validate_code.py"
  post_generate_hooks:
    - "scripts/deploy_docs.py"
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 获取帮助

- 查看 [示例代码](example_usage.py)
- 阅读 [配置文档](docs.yaml)
- 提交 [Issue](../../issues)

---

**提示**: 运行 `python docs/example_usage.py` 查看完整的功能演示！