---
title: "{{ index.title or '文档索引' }}"
description: "{{ index.description or '项目文档的主索引页面' }}"
author: "{{ generator_info.name }}"
version: "{{ generator_info.version }}"
generated_at: "{{ now().strftime('%Y-%m-%d %H:%M:%S') }}"
---

# 文档中心

欢迎来到项目文档中心！这里包含了完整的API参考、教程指南和开发资源。

{% if index.modules or index.total_modules %}
## 统计信息

{% if index.total_modules %}
- **模块总数**: {{ index.total_modules }}
{% endif %}
{% if index.total_classes %}
- **类总数**: {{ index.total_classes }}
{% endif %}
{% if index.total_functions %}
- **函数总数**: {{ index.total_functions }}
{% endif %}
{% if index.total_tutorials %}
- **教程数量**: {{ index.total_tutorials }}
{% endif %}
- **最后更新**: {{ now().strftime('%Y-%m-%d %H:%M:%S') }}
{% endif %}

{% if index.modules %}
## API参考文档

### 模块列表

{% for module in index.modules %}
#### [{{ module.name }}]({{ module.name }}.md)

{% if module.docstring %}
{{ module.docstring | truncate_words(30) }}
{% else %}
{{ module.name }} 模块的API文档。
{% endif %}

- **文件路径**: `{{ module.path }}`
{% if module.class_count %}
- **类数量**: {{ module.class_count }}
{% endif %}
{% if module.function_count %}
- **函数数量**: {{ module.function_count }}
{% endif %}

---

{% endfor %}
{% endif %}

{% if index.by_type %}
## 教程文档

{% for type_name, type_data in index.by_type.items() %}
### {{ type_name | snake_to_title }} ({{ type_data.count }}个)

{% for tutorial in type_data.tutorials %}
#### [{{ tutorial.title }}]({{ tutorial.title | to_anchor }}.md)

{% if tutorial.description %}
{{ tutorial.description | truncate_words(25) }}
{% endif %}

- **难度**: {{ tutorial.difficulty }}
{% if tutorial.estimated_time %}
- **预计时间**: {{ tutorial.estimated_time }}
{% endif %}
{% if tutorial.tags %}
- **标签**: {% for tag in tutorial.tags %}`{{ tag }}`{% if not loop.last %} {% endif %}{% endfor %}
{% endif %}

---

{% endfor %}
{% endfor %}
{% endif %}

{% if index.by_difficulty %}
## 按难度分类

{% for difficulty, count in index.by_difficulty.items() %}
### {{ difficulty }} ({{ count }}个教程)

点击查看所有 [{{ difficulty }}教程](tutorials-{{ difficulty | to_anchor }}.md)

{% endfor %}
{% endif %}

{% if index.featured %}
## 推荐教程

{% for tutorial in index.featured %}
### [{{ tutorial.title }}]({{ tutorial.title | to_anchor }}.md)

{% if tutorial.description %}
{{ tutorial.description | truncate_words(20) }}
{% endif %}

**难度**: {{ tutorial.difficulty }}

---

{% endfor %}
{% endif %}

## 快速开始

### 新手入门

如果您是第一次使用本项目，建议按以下顺序学习：

1. **阅读基础教程** - 了解项目的核心概念
2. **查看快速开始指南** - 快速上手基本功能
3. **探索API文档** - 深入了解各个模块和类
4. **尝试高级教程** - 学习更复杂的使用场景

### 开发者指南

如果您是开发者，可能需要：

1. **API参考文档** - 查看完整的API说明
2. **架构设计文档** - 了解项目架构
3. **贡献指南** - 参与项目开发
4. **测试文档** - 运行和编写测试

## 文档结构

```
docs/
├── api/                    # API参考文档
│   ├── index.md           # API索引
│   └── [module].md        # 各模块文档
├── tutorials/             # 教程文档
│   ├── index.md          # 教程索引
│   ├── basic/            # 基础教程
│   ├── advanced/         # 高级教程
│   └── examples/         # 示例代码
├── guides/               # 开发指南
│   ├── installation.md  # 安装指南
│   ├── configuration.md # 配置说明
│   └── contributing.md  # 贡献指南
└── assets/              # 资源文件
    ├── css/             # 样式文件
    ├── js/              # 脚本文件
    └── images/          # 图片资源
```

## 搜索文档

- 使用页面内搜索（Ctrl+F / Cmd+F）快速查找内容
- 查看左侧导航栏浏览不同章节
- 使用目录跳转到具体部分

## 文档贡献

我们欢迎对文档的改进建议！如果您发现：

- 文档错误或过时信息
- 缺失的重要内容
- 可以改进的示例代码
- 翻译或语言问题

请通过以下方式贡献：

1. **提交Issue** - 报告文档问题
2. **提交Pull Request** - 直接修改文档
3. **参与讨论** - 在社区论坛提出建议
4. **分享经验** - 编写教程和最佳实践

## 获取帮助

如果您在使用过程中遇到问题：

### 常用资源

- [FAQ - 常见问题](faq.md)
- [故障排除指南](troubleshooting.md)
- [社区论坛](community.md)
- [官方文档](official-docs.md)

### 联系方式

- **GitHub Issues**: 报告bug和功能请求
- **讨论区**: 技术讨论和经验分享
- **邮件**: 商业合作和重要问题

## 版本说明

{% if config.version %}
- **当前版本**: {{ config.version }}
{% endif %}
- **文档版本**: {{ generator_info.version }}
- **兼容性**: 请查看具体模块的版本要求

## 更新日志

- **{{ now().strftime('%Y-%m-%d') }}**: 文档更新，添加新的API和教程
- **定期更新**: 文档会随代码更新自动同步

---

## 技术规格

### 文档生成

- **生成工具**: {{ generator_info.name }}
- **模板引擎**: Jinja2
- **支持格式**: Markdown, HTML, PDF
- **自动化**: 集成CI/CD流程

### 浏览要求

- **现代浏览器**: Chrome, Firefox, Safari, Edge
- **移动设备**: 支持响应式设计
- **搜索功能**: 内置全文搜索
- **离线访问**: 支持离线浏览

---

*本文档由 {{ generator_info.name }} v{{ generator_info.version }} 自动生成于 {{ now().strftime('%Y-%m-%d %H:%M:%S') }}*

**最后更新**: {{ now().strftime('%Y-%m-%d %H:%M:%S') }}