---
title: "{{ module.name }} - API文档"
description: "{{ module.docstring or '模块API参考文档' | truncate(100) }}"
author: "{{ generator_info.name }}"
version: "{{ generator_info.version }}"
generated_at: "{{ now().strftime('%Y-%m-%d %H:%M:%S') }}"
module_path: "{{ module.path }}"
---

# {{ module.name }} 模块

{% if module.docstring %}
{{ module.docstring | format_docstring }}
{% else %}
{{ module.name }} 模块的API参考文档。
{% endif %}

## 模块信息

- **文件路径**: `{{ module.path }}`
{% if module.file_info %}
- **代码行数**: {{ module.file_info.line_count }}
- **文件大小**: {{ module.file_info.size_bytes }} bytes
{% endif %}
- **生成时间**: {{ now().strftime('%Y-%m-%d %H:%M:%S') }}

{% if config.include_toc and (module.classes or module.functions) %}
## 目录

{% if module.classes %}
### 类

{% for class in module.classes %}
- [{{ class.name }}](#{{ class.name | to_anchor }})
{% endfor %}
{% endif %}

{% if module.functions %}
### 函数

{% for function in module.functions %}
- [{{ function.name }}](#{{ function.name | to_anchor }})
{% endfor %}
{% endif %}

{% if module.variables %}
### 变量

{% for variable in module.variables %}
- [{{ variable.name }}](#{{ variable.name | to_anchor }})
{% endfor %}
{% endif %}

---
{% endif %}

{% if module.classes %}
## 类

{% for class in module.classes %}
### {{ class.name }}

{% if class.docstring %}
{{ class.docstring | format_docstring }}
{% endif %}

{% if class.base_classes %}
**继承关系**:
{% for base in class.base_classes %}
- `{{ base }}`
{% endfor %}
{% endif %}

**定义位置**: 第 {{ class.line_number }} 行

{% if class.methods %}
#### 方法

{% for method in class.methods %}
##### {{ method.name }}

```python
{{ method.name }}(
{%- for param in method.parameters -%}
{%- if not loop.first %}, {% endif -%}
{{ param.name }}
{%- if param.type %}: {{ param.type | format_type }}{% endif -%}
{%- if param.default %} = {{ param.default }}{% endif -%}
{%- endfor -%}
)
{%- if method.return_type %} -> {{ method.return_type | format_type }}{% endif %}
```

{% if method.docstring %}
{{ method.docstring | format_docstring }}
{% endif %}

{% if method.parameters %}
**参数**:

{% for param in method.parameters %}
- `{{ param.name }}`{% if param.type %} ({{ param.type | format_type }}){% endif %}{% if param.default %} = {{ param.default }}{% endif %}
{% endfor %}
{% endif %}

{% if method.return_type %}
**返回值**: {{ method.return_type | format_type }}
{% endif %}

{% if method.examples %}
**示例**:

{% for example in method.examples %}
{{ example | code_block('python') }}
{% endfor %}
{% endif %}

---

{% endfor %}
{% endif %}

{% if class.properties %}
#### 属性

{% for property in class.properties %}
##### {{ property.name }}

{% if property.docstring %}
{{ property.docstring | format_docstring }}
{% endif %}

{% if property.return_type %}
**类型**: {{ property.return_type | format_type }}
{% endif %}

---

{% endfor %}
{% endif %}

{% if class.attributes %}
#### 类属性

{% for attr in class.attributes %}
##### {{ attr.name }}

{% if attr.type_annotation %}
**类型**: {{ attr.type_annotation | format_type }}
{% endif %}

{% if attr.default_value %}
**默认值**: `{{ attr.default_value }}`
{% endif %}

---

{% endfor %}
{% endif %}

---

{% endfor %}
{% endif %}

{% if module.functions %}
## 函数

{% for function in module.functions %}
### {{ function.name }}

```python
{{ function.name }}(
{%- for param in function.parameters -%}
{%- if not loop.first %}, {% endif -%}
{{ param.name }}
{%- if param.type %}: {{ param.type | format_type }}{% endif -%}
{%- if param.default %} = {{ param.default }}{% endif -%}
{%- endfor -%}
)
{%- if function.return_type %} -> {{ function.return_type | format_type }}{% endif %}
```

{% if function.docstring %}
{{ function.docstring | format_docstring }}
{% endif %}

{% if function.parameters %}
**参数**:

{% for param in function.parameters %}
- `{{ param.name }}`{% if param.type %} ({{ param.type | format_type }}){% endif %}{% if param.default %} = {{ param.default }}{% endif %}
{% endfor %}
{% endif %}

{% if function.return_type %}
**返回值**: {{ function.return_type | format_type }}
{% endif %}

{% if function.examples %}
**示例**:

{% for example in function.examples %}
{{ example | code_block('python') }}
{% endfor %}
{% endif %}

**定义位置**: 第 {{ function.line_number }} 行

---

{% endfor %}
{% endif %}

{% if module.variables %}
## 模块变量

{% for variable in module.variables %}
### {{ variable.name }}

{% if variable.type_annotation %}
**类型**: {{ variable.type_annotation | format_type }}
{% endif %}

{% if variable.default_value %}
**值**: `{{ variable.default_value }}`
{% endif %}

**定义位置**: 第 {{ variable.line_number }} 行

---

{% endfor %}
{% endif %}

## 生成信息

- **生成工具**: {{ generator_info.name }}
- **版本**: {{ generator_info.version }}
- **生成时间**: {{ now().strftime('%Y-%m-%d %H:%M:%S') }}
- **源文件**: {{ module.path }}

---

*此文档由 {{ generator_info.name }} 自动生成*