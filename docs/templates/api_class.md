---
title: "{{ class.name }} - 类文档"
description: "{{ class.docstring or class.name + ' 类的详细文档' | truncate(100) }}"
author: "{{ generator_info.name }}"
version: "{{ generator_info.version }}"
generated_at: "{{ now().strftime('%Y-%m-%d %H:%M:%S') }}"
class_name: "{{ class.name }}"
module_name: "{{ module.name }}"
---

# {{ class.name }} 类

{% if class.docstring %}
{{ class.docstring | format_docstring }}
{% else %}
{{ class.name }} 类的详细API文档。
{% endif %}

## 类信息

- **类名**: `{{ class.name }}`
- **模块**: `{{ module.name }}`
- **完整名称**: `{{ class.full_name }}`
- **定义位置**: 第 {{ class.line_number }} 行

{% if class.base_classes %}
## 继承关系

此类继承自以下类：

{% for base in class.base_classes %}
- `{{ base }}`
{% endfor %}
{% endif %}

{% if config.include_toc and (class.methods or class.properties or class.attributes) %}
## 目录

{% if class.methods %}
### 方法

{% for method in class.methods %}
- [{{ method.name }}](#{{ method.name | to_anchor }})
{% endfor %}
{% endif %}

{% if class.properties %}
### 属性

{% for property in class.properties %}
- [{{ property.name }}](#{{ property.name | to_anchor }})
{% endfor %}
{% endif %}

{% if class.attributes %}
### 类属性

{% for attr in class.attributes %}
- [{{ attr.name }}](#{{ attr.name | to_anchor }})
{% endfor %}
{% endif %}

---
{% endif %}

{% if class.methods %}
## 方法

{% for method in class.methods %}
### {{ method.name }}

```python
{% if method.is_classmethod %}@classmethod
{% elif method.is_staticmethod %}@staticmethod
{% endif %}def {{ method.name }}(
{%- if not method.is_staticmethod and not method.is_classmethod %}self{% if method.parameters %}, {% endif %}{% endif -%}
{%- for param in method.parameters -%}
{%- if param.name not in ['self', 'cls'] -%}
{%- if not loop.first and (not method.is_staticmethod and not method.is_classmethod) %}, {% elif loop.first and (method.is_staticmethod or method.is_classmethod) %}{% else %}, {% endif -%}
{{ param.name }}
{%- if param.type %}: {{ param.type | format_type }}{% endif -%}
{%- if param.default %} = {{ param.default }}{% endif -%}
{%- endif -%}
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
{% if param.name not in ['self', 'cls'] %}
- `{{ param.name }}`{% if param.type %} ({{ param.type | format_type }}){% endif %}{% if param.default %} = {{ param.default }}{% endif %}
{% endif %}
{% endfor %}
{% endif %}

{% if method.return_type %}
**返回值**: {{ method.return_type | format_type }}
{% endif %}

{% if method.is_async %}
**注意**: 这是一个异步方法，需要使用 `await` 调用。
{% endif %}

{% if method.is_classmethod %}
**注意**: 这是一个类方法，可以通过类直接调用。
{% endif %}

{% if method.is_staticmethod %}
**注意**: 这是一个静态方法，不需要实例或类参数。
{% endif %}

{% if method.examples %}
**示例**:

{% for example in method.examples %}
{{ example | code_block('python') }}
{% endfor %}
{% endif %}

**定义位置**: 第 {{ method.line_number }} 行

---

{% endfor %}
{% endif %}

{% if class.properties %}
## 属性

{% for property in class.properties %}
### {{ property.name }}

```python
@property
def {{ property.name }}(self)
{%- if property.return_type %} -> {{ property.return_type | format_type }}{% endif %}
```

{% if property.docstring %}
{{ property.docstring | format_docstring }}
{% endif %}

{% if property.return_type %}
**返回类型**: {{ property.return_type | format_type }}
{% endif %}

{% if property.examples %}
**示例**:

{% for example in property.examples %}
{{ example | code_block('python') }}
{% endfor %}
{% endif %}

**定义位置**: 第 {{ property.line_number }} 行

---

{% endfor %}
{% endif %}

{% if class.attributes %}
## 类属性

{% for attr in class.attributes %}
### {{ attr.name }}

{% if attr.type_annotation %}
**类型**: {{ attr.type_annotation | format_type }}
{% endif %}

{% if attr.default_value %}
**默认值**: `{{ attr.default_value }}`
{% endif %}

**定义位置**: 第 {{ attr.line_number }} 行

---

{% endfor %}
{% endif %}

## 使用示例

```python
# 导入类
from {{ module.name }} import {{ class.name }}

# 创建实例
{% if class.methods %}
{% for method in class.methods %}
{% if method.name == '__init__' %}
instance = {{ class.name }}(
{%- for param in method.parameters -%}
{%- if param.name != 'self' -%}
{%- if not loop.first %}, {% endif -%}
{%- if param.default -%}
{{ param.name }}={{ param.default }}
{%- else -%}
{{ param.name }}=...
{%- endif -%}
{%- endif -%}
{%- endfor -%}
)
{% break %}
{% endif %}
{% endfor %}
{% else %}
instance = {{ class.name }}()
{% endif %}

# 使用实例
# TODO: 添加具体的使用示例
```

## 相关链接

- [{{ module.name }} 模块文档]({{ module.name }}.md)
{% if class.base_classes %}
- 基类文档:
{% for base in class.base_classes %}
  - [{{ base }}]({{ base | replace('.', '/') }}.md)
{% endfor %}
{% endif %}

## 生成信息

- **生成工具**: {{ generator_info.name }}
- **版本**: {{ generator_info.version }}
- **生成时间**: {{ now().strftime('%Y-%m-%d %H:%M:%S') }}
- **源模块**: {{ module.name }}

---

*此文档由 {{ generator_info.name }} 自动生成*