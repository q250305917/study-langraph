---
title: "{{ tutorial.metadata.title }}"
description: "{{ tutorial.metadata.description or '教程文档' }}"
author: "{{ tutorial.metadata.author or generator_info.name }}"
difficulty: "{{ tutorial.metadata.difficulty }}"
estimated_time: "{{ tutorial.metadata.estimated_time }}"
type: "{{ tutorial.metadata.example_type.value }}"
tags: {{ tutorial.metadata.tags | json_pretty }}
created_at: "{{ tutorial.metadata.created_date or now().strftime('%Y-%m-%d') }}"
updated_at: "{{ tutorial.metadata.updated_date or now().strftime('%Y-%m-%d') }}"
version: "{{ tutorial.metadata.version }}"
---

# {{ tutorial.metadata.title }}

{% if tutorial.metadata.description %}
{{ tutorial.metadata.description }}
{% endif %}

## 教程信息

{% if tutorial.metadata.difficulty %}
- **难度级别**: {{ tutorial.metadata.difficulty }}
{% endif %}
{% if tutorial.metadata.estimated_time %}
- **预计时间**: {{ tutorial.metadata.estimated_time }}
{% endif %}
{% if tutorial.metadata.example_type %}
- **教程类型**: {{ tutorial.metadata.example_type.value | snake_to_title }}
{% endif %}
{% if tutorial.metadata.author %}
- **作者**: {{ tutorial.metadata.author }}
{% endif %}
{% if tutorial.metadata.version %}
- **版本**: {{ tutorial.metadata.version }}
{% endif %}

{% if tutorial.metadata.tags %}
## 标签

{% for tag in tutorial.metadata.tags %}
`{{ tag }}`{% if not loop.last %} {% endif %}
{% endfor %}
{% endif %}

{% if tutorial.metadata.prerequisites %}
## 前置要求

在开始本教程之前，您需要了解：

{% for prereq in tutorial.metadata.prerequisites %}
- {{ prereq }}
{% endfor %}
{% endif %}

{% if config.include_toc and tutorial.sections %}
## 目录

{% for section in tutorial.sections %}
- [{{ section.title }}](#{{ section.title | to_anchor }})
{% if section.subsections %}
{% for subsection in section.subsections %}
  - [{{ subsection.title }}](#{{ subsection.title | to_anchor }})
{% endfor %}
{% endif %}
{% endfor %}

---
{% endif %}

{% if tutorial.introduction %}
## 介绍

{{ tutorial.introduction | format_docstring }}
{% endif %}

{% for section in tutorial.sections %}
## {{ section.title }}

{% if section.description %}
{{ section.description }}
{% endif %}

{% if section.difficulty and section.difficulty != tutorial.metadata.difficulty %}
**难度**: {{ section.difficulty }}
{% endif %}

{% if section.prerequisites %}
**本节前置要求**:
{% for prereq in section.prerequisites %}
- {{ prereq }}
{% endfor %}
{% endif %}

{% for code_block in section.code_blocks %}
{% if code_block.description %}
### {{ code_block.description }}

{% endif %}
{% if code_block.content %}
```{{ code_block.language }}
{{ code_block.content }}
```

{% if config.include_output and code_block.expected_output %}
**输出结果**:

```
{{ code_block.expected_output }}
```
{% endif %}

{% if not code_block.is_executable %}
> **注意**: 此代码片段仅用于演示，可能无法直接执行。
{% endif %}

{% if code_block.tags %}
**标签**: {% for tag in code_block.tags %}`{{ tag }}`{% if not loop.last %} {% endif %}{% endfor %}
{% endif %}

{% endif %}
{% endfor %}

{% if section.subsections %}
{% for subsection in section.subsections %}
### {{ subsection.title }}

{% if subsection.description %}
{{ subsection.description }}
{% endif %}

{% for code_block in subsection.code_blocks %}
{% if code_block.description %}
#### {{ code_block.description }}

{% endif %}
{% if code_block.content %}
```{{ code_block.language }}
{{ code_block.content }}
```

{% if config.include_output and code_block.expected_output %}
**输出结果**:

```
{{ code_block.expected_output }}
```
{% endif %}
{% endif %}
{% endfor %}

{% endfor %}
{% endif %}

---

{% endfor %}

{% if tutorial.conclusion %}
## 总结

{{ tutorial.conclusion | format_docstring }}
{% endif %}

## 下一步

完成本教程后，您可以：

{% if tutorial.metadata.example_type.value == 'basic' %}
- 尝试中级教程，深入了解更多功能
- 阅读相关的API文档
- 修改示例代码，探索不同的参数和配置
{% elif tutorial.metadata.example_type.value == 'advanced' %}
- 在实际项目中应用这些技术
- 查看源代码实现，了解底层原理
- 贡献您的改进建议或bug报告
{% else %}
- 探索相关的教程和文档
- 在实际场景中应用所学知识
- 参与社区讨论和交流
{% endif %}

{% if tutorial.references %}
## 参考资料

{% for reference in tutorial.references %}
- {{ reference }}
{% endfor %}
{% endif %}

## 相关资源

- [API参考文档](../api/)
- [更多教程](../tutorials/)
- [示例代码仓库](../examples/)

## 问题反馈

如果您在学习过程中遇到问题或有改进建议，请：

1. 检查常见问题解答
2. 查看相关的API文档
3. 在社区论坛提问
4. 提交issue或pull request

## 版权声明

{% if tutorial.metadata.author %}
本教程由 {{ tutorial.metadata.author }} 创作。
{% endif %}

## 更新日志

{% if tutorial.metadata.updated_date and tutorial.metadata.updated_date != tutorial.metadata.created_date %}
- **{{ tutorial.metadata.updated_date }}**: 教程更新
{% endif %}
{% if tutorial.metadata.created_date %}
- **{{ tutorial.metadata.created_date }}**: 教程创建
{% endif %}

---

## 生成信息

- **生成工具**: {{ generator_info.name }}
- **生成时间**: {{ now().strftime('%Y-%m-%d %H:%M:%S') }}
- **配置**: {{ config.format_style }}格式

---

*此教程由 {{ generator_info.name }} 自动生成并整理*