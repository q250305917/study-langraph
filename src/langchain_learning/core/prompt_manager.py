"""
LangChain学习项目的提示词管理模块

本模块实现了完整的提示词模板管理系统，支持：
- 模板的创建、存储和检索
- 动态变量替换和格式化
- 多语言模板支持
- 模板版本控制和历史记录
- 提示词优化和A/B测试
- 模板验证和错误检查
"""

import json
import yaml
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import time
from datetime import datetime

from pydantic import BaseModel, Field, validator
from jinja2 import Template, Environment, FileSystemLoader, meta

from .logger import get_logger
from .exceptions import (
    ValidationError,
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class PromptType(Enum):
    """提示词类型枚举"""
    SYSTEM = "system"              # 系统提示词
    USER = "user"                  # 用户提示词
    ASSISTANT = "assistant"        # 助手提示词
    FUNCTION = "function"          # 函数调用提示词
    CHAT = "chat"                  # 聊天对话提示词
    COMPLETION = "completion"      # 文本补全提示词
    INSTRUCTION = "instruction"    # 指令提示词


class TemplateFormat(Enum):
    """模板格式枚举"""
    PLAIN = "plain"                # 纯文本
    JINJA2 = "jinja2"             # Jinja2模板
    F_STRING = "f_string"         # Python f-string
    MUSTACHE = "mustache"         # Mustache模板


@dataclass
class PromptVersion:
    """
    提示词版本信息
    
    记录模板的版本历史和变更信息。
    """
    version: str
    content: str
    variables: List[str]
    created_at: datetime
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # 性能统计
    usage_count: int = 0
    success_rate: float = 0.0
    average_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "version": self.version,
            "content": self.content,
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "average_tokens": self.average_tokens
        }


class PromptTemplate(BaseModel):
    """
    提示词模板模型
    
    定义提示词模板的结构和元数据。
    """
    
    # 基本信息
    name: str = Field(description="模板名称")
    content: str = Field(description="模板内容")
    template_type: PromptType = Field(description="模板类型")
    format: TemplateFormat = Field(default=TemplateFormat.JINJA2, description="模板格式")
    
    # 元数据
    description: str = Field(default="", description="模板描述")
    author: str = Field(default="", description="作者")
    version: str = Field(default="1.0.0", description="版本")
    language: str = Field(default="zh", description="语言代码")
    tags: List[str] = Field(default_factory=list, description="标签")
    
    # 变量信息
    variables: List[str] = Field(default_factory=list, description="模板变量")
    required_variables: List[str] = Field(default_factory=list, description="必需变量")
    default_values: Dict[str, Any] = Field(default_factory=dict, description="默认值")
    
    # 配置选项
    max_tokens: Optional[int] = Field(default=None, description="最大令牌数")
    temperature: Optional[float] = Field(default=None, description="温度参数")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: Optional[datetime] = Field(default=None, description="更新时间")
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('name')
    def validate_name(cls, v):
        """验证模板名称"""
        if not v or not v.strip():
            raise ValueError("Template name cannot be empty")
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', v):
            raise ValueError("Template name can only contain letters, numbers, underscore, dash and dot")
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        """验证模板内容"""
        if not v or not v.strip():
            raise ValueError("Template content cannot be empty")
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """验证语言代码"""
        # 支持常见的语言代码
        valid_languages = ['zh', 'en', 'ja', 'ko', 'fr', 'de', 'es', 'it', 'ru']
        if v not in valid_languages:
            logger.warning(f"Unsupported language code: {v}")
        return v
    
    def extract_variables(self) -> List[str]:
        """提取模板中的变量"""
        if self.format == TemplateFormat.JINJA2:
            env = Environment()
            ast = env.parse(self.content)
            variables = list(meta.find_undeclared_variables(ast))
            self.variables = variables
            return variables
        elif self.format == TemplateFormat.F_STRING:
            # 简单的f-string变量提取
            pattern = r'\{([^}]+)\}'
            variables = list(set(re.findall(pattern, self.content)))
            self.variables = variables
            return variables
        else:
            return self.variables
    
    def render(self, **kwargs) -> str:
        """
        渲染模板
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            渲染后的文本
            
        Raises:
            ValidationError: 变量验证失败
        """
        # 验证必需变量
        missing_vars = []
        for var in self.required_variables:
            if var not in kwargs and var not in self.default_values:
                missing_vars.append(var)
        
        if missing_vars:
            raise ValidationError(
                f"Missing required variables: {missing_vars}",
                error_code=ErrorCodes.VALIDATION_REQUIRED_ERROR,
                context={"template": self.name, "missing_variables": missing_vars}
            )
        
        # 合并默认值
        render_kwargs = {**self.default_values, **kwargs}
        
        try:
            if self.format == TemplateFormat.JINJA2:
                template = Template(self.content)
                return template.render(**render_kwargs)
            elif self.format == TemplateFormat.F_STRING:
                return self.content.format(**render_kwargs)
            elif self.format == TemplateFormat.PLAIN:
                return self.content
            else:
                raise ValidationError(
                    f"Unsupported template format: {self.format}",
                    error_code=ErrorCodes.VALIDATION_FORMAT_ERROR
                )
        except Exception as e:
            raise ValidationError(
                f"Template rendering failed: {str(e)}",
                error_code=ErrorCodes.VALIDATION_FORMAT_ERROR,
                context={"template": self.name, "variables": render_kwargs},
                cause=e
            )
    
    def validate_syntax(self) -> bool:
        """
        验证模板语法
        
        Returns:
            True如果语法正确
            
        Raises:
            ValidationError: 语法错误
        """
        try:
            if self.format == TemplateFormat.JINJA2:
                env = Environment()
                env.parse(self.content)
            elif self.format == TemplateFormat.F_STRING:
                # 尝试格式化一个空字典
                self.content.format()
            return True
        except Exception as e:
            raise ValidationError(
                f"Template syntax error: {str(e)}",
                error_code=ErrorCodes.VALIDATION_FORMAT_ERROR,
                context={"template": self.name, "format": self.format.value},
                cause=e
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()
    
    def clone(self, new_name: Optional[str] = None) -> 'PromptTemplate':
        """
        克隆模板
        
        Args:
            new_name: 新模板名称
            
        Returns:
            克隆的模板
        """
        data = self.dict()
        if new_name:
            data['name'] = new_name
        data['created_at'] = datetime.now()
        data['updated_at'] = None
        return PromptTemplate(**data)


class TemplateRegistry:
    """
    模板注册表
    
    管理模板的存储、索引和查询。
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        初始化模板注册表
        
        Args:
            storage_path: 存储路径，None则使用内存存储
        """
        self.storage_path = storage_path
        self._templates: Dict[str, PromptTemplate] = {}
        self._versions: Dict[str, List[PromptVersion]] = {}
        self._index: Dict[str, List[str]] = {
            'type': {},
            'tags': {},
            'language': {},
            'author': {}
        }
        
        # 如果指定了存储路径，从文件加载
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_storage()
        
        logger.debug(f"Initialized template registry with storage: {storage_path}")
    
    def register(self, template: PromptTemplate) -> None:
        """
        注册模板
        
        Args:
            template: 要注册的模板
            
        Raises:
            ValidationError: 模板验证失败
        """
        # 验证模板语法
        template.validate_syntax()
        
        # 提取变量
        template.extract_variables()
        
        # 检查模板是否已存在
        if template.name in self._templates:
            # 创建新版本
            self._create_version(template)
        
        # 注册模板
        template.updated_at = datetime.now()
        self._templates[template.name] = template
        
        # 更新索引
        self._update_index(template)
        
        # 保存到存储
        if self.storage_path:
            self._save_to_storage(template)
        
        logger.info(f"Registered template: {template.name}")
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """
        获取模板
        
        Args:
            name: 模板名称
            version: 版本号，None则获取最新版本
            
        Returns:
            模板实例，不存在则返回None
        """
        if version:
            # 获取指定版本
            versions = self._versions.get(name, [])
            for v in versions:
                if v.version == version:
                    # 重构模板对象
                    template_data = self._templates[name].dict()
                    template_data['content'] = v.content
                    template_data['version'] = v.version
                    template_data['variables'] = v.variables
                    return PromptTemplate(**template_data)
            return None
        else:
            # 获取最新版本
            return self._templates.get(name)
    
    def list_templates(self, 
                      template_type: Optional[PromptType] = None,
                      tags: Optional[List[str]] = None,
                      language: Optional[str] = None,
                      author: Optional[str] = None) -> List[PromptTemplate]:
        """
        列出模板
        
        Args:
            template_type: 模板类型筛选
            tags: 标签筛选
            language: 语言筛选
            author: 作者筛选
            
        Returns:
            匹配的模板列表
        """
        templates = list(self._templates.values())
        
        # 按类型筛选
        if template_type:
            templates = [t for t in templates if t.template_type == template_type]
        
        # 按标签筛选
        if tags:
            templates = [
                t for t in templates 
                if any(tag in t.tags for tag in tags)
            ]
        
        # 按语言筛选
        if language:
            templates = [t for t in templates if t.language == language]
        
        # 按作者筛选
        if author:
            templates = [t for t in templates if t.author == author]
        
        return templates
    
    def search(self, query: str) -> List[PromptTemplate]:
        """
        搜索模板
        
        Args:
            query: 搜索关键词
            
        Returns:
            匹配的模板列表
        """
        query = query.lower()
        matching_templates = []
        
        for template in self._templates.values():
            # 搜索名称、描述、内容、标签
            if (query in template.name.lower() or
                query in template.description.lower() or
                query in template.content.lower() or
                any(query in tag.lower() for tag in template.tags)):
                matching_templates.append(template)
        
        return matching_templates
    
    def delete(self, name: str) -> bool:
        """
        删除模板
        
        Args:
            name: 模板名称
            
        Returns:
            True如果删除成功
        """
        if name in self._templates:
            template = self._templates[name]
            
            # 从内存中删除
            del self._templates[name]
            
            # 删除版本历史
            if name in self._versions:
                del self._versions[name]
            
            # 更新索引
            self._remove_from_index(template)
            
            # 从存储中删除
            if self.storage_path:
                template_file = self.storage_path / f"{name}.yaml"
                if template_file.exists():
                    template_file.unlink()
            
            logger.info(f"Deleted template: {name}")
            return True
        
        return False
    
    def get_versions(self, name: str) -> List[PromptVersion]:
        """获取模板的版本历史"""
        return self._versions.get(name, [])
    
    def export_template(self, name: str, format: str = "yaml") -> str:
        """
        导出模板
        
        Args:
            name: 模板名称
            format: 导出格式 (yaml, json)
            
        Returns:
            导出的文本内容
        """
        template = self.get(name)
        if not template:
            raise ResourceError(
                f"Template '{name}' not found",
                error_code=ErrorCodes.FILE_NOT_FOUND
            )
        
        data = template.to_dict()
        
        if format == "yaml":
            return yaml.dump(data, allow_unicode=True, default_flow_style=False)
        elif format == "json":
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            raise ValidationError(
                f"Unsupported export format: {format}",
                error_code=ErrorCodes.VALIDATION_FORMAT_ERROR
            )
    
    def import_template(self, content: str, format: str = "yaml") -> PromptTemplate:
        """
        导入模板
        
        Args:
            content: 模板内容
            format: 导入格式 (yaml, json)
            
        Returns:
            导入的模板
        """
        try:
            if format == "yaml":
                data = yaml.safe_load(content)
            elif format == "json":
                data = json.loads(content)
            else:
                raise ValidationError(
                    f"Unsupported import format: {format}",
                    error_code=ErrorCodes.VALIDATION_FORMAT_ERROR
                )
            
            # 处理datetime字段
            if 'created_at' in data and isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if 'updated_at' in data and isinstance(data['updated_at'], str):
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
            template = PromptTemplate(**data)
            self.register(template)
            return template
            
        except Exception as e:
            raise ValidationError(
                f"Failed to import template: {str(e)}",
                error_code=ErrorCodes.VALIDATION_FORMAT_ERROR,
                cause=e
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        total_templates = len(self._templates)
        
        type_counts = {}
        for template_type in PromptType:
            count = len([t for t in self._templates.values() if t.template_type == template_type])
            if count > 0:
                type_counts[template_type.value] = count
        
        language_counts = {}
        for template in self._templates.values():
            language_counts[template.language] = language_counts.get(template.language, 0) + 1
        
        return {
            "total_templates": total_templates,
            "type_distribution": type_counts,
            "language_distribution": language_counts,
            "total_versions": sum(len(versions) for versions in self._versions.values())
        }
    
    def _create_version(self, template: PromptTemplate) -> None:
        """创建模板版本"""
        if template.name not in self._versions:
            self._versions[template.name] = []
        
        version = PromptVersion(
            version=template.version,
            content=template.content,
            variables=template.variables,
            created_at=datetime.now(),
            created_by=template.author,
            description=template.description,
            tags=template.tags.copy()
        )
        
        self._versions[template.name].append(version)
        
        # 限制版本数量，保留最新的10个版本
        if len(self._versions[template.name]) > 10:
            self._versions[template.name] = self._versions[template.name][-10:]
    
    def _update_index(self, template: PromptTemplate) -> None:
        """更新索引"""
        # 类型索引
        type_key = template.template_type.value
        if type_key not in self._index['type']:
            self._index['type'][type_key] = []
        if template.name not in self._index['type'][type_key]:
            self._index['type'][type_key].append(template.name)
        
        # 标签索引
        for tag in template.tags:
            if tag not in self._index['tags']:
                self._index['tags'][tag] = []
            if template.name not in self._index['tags'][tag]:
                self._index['tags'][tag].append(template.name)
        
        # 语言索引
        if template.language not in self._index['language']:
            self._index['language'][template.language] = []
        if template.name not in self._index['language'][template.language]:
            self._index['language'][template.language].append(template.name)
        
        # 作者索引
        if template.author and template.author not in self._index['author']:
            self._index['author'][template.author] = []
        if template.author and template.name not in self._index['author'][template.author]:
            self._index['author'][template.author].append(template.name)
    
    def _remove_from_index(self, template: PromptTemplate) -> None:
        """从索引中移除模板"""
        # 从类型索引移除
        type_key = template.template_type.value
        if type_key in self._index['type']:
            self._index['type'][type_key] = [
                name for name in self._index['type'][type_key] 
                if name != template.name
            ]
        
        # 从标签索引移除
        for tag in template.tags:
            if tag in self._index['tags']:
                self._index['tags'][tag] = [
                    name for name in self._index['tags'][tag] 
                    if name != template.name
                ]
        
        # 从语言索引移除
        if template.language in self._index['language']:
            self._index['language'][template.language] = [
                name for name in self._index['language'][template.language] 
                if name != template.name
            ]
        
        # 从作者索引移除
        if template.author in self._index['author']:
            self._index['author'][template.author] = [
                name for name in self._index['author'][template.author] 
                if name != template.name
            ]
    
    def _load_from_storage(self) -> None:
        """从存储加载模板"""
        if not self.storage_path.exists():
            return
        
        for template_file in self.storage_path.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                template = self.import_template(content, "yaml")
                logger.debug(f"Loaded template from storage: {template.name}")
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
    
    def _save_to_storage(self, template: PromptTemplate) -> None:
        """保存模板到存储"""
        template_file = self.storage_path / f"{template.name}.yaml"
        content = self.export_template(template.name, "yaml")
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(content)


class PromptManager:
    """
    提示词管理器
    
    提供提示词模板的统一管理接口。
    """
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """
        初始化提示词管理器
        
        Args:
            storage_path: 存储路径
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # 使用项目根目录下的templates目录
            project_root = Path(__file__).parent.parent.parent.parent
            self.storage_path = project_root / "templates" / "prompts"
        
        self.registry = TemplateRegistry(self.storage_path)
        
        # 注册内置模板
        self._register_builtin_templates()
        
        logger.info("Prompt manager initialized")
    
    def create_template(
        self,
        name: str,
        content: str,
        template_type: PromptType = PromptType.USER,
        **kwargs
    ) -> PromptTemplate:
        """
        创建模板
        
        Args:
            name: 模板名称
            content: 模板内容
            template_type: 模板类型
            **kwargs: 其他参数
            
        Returns:
            创建的模板
        """
        template = PromptTemplate(
            name=name,
            content=content,
            template_type=template_type,
            **kwargs
        )
        
        self.registry.register(template)
        return template
    
    def get_template(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.registry.get(name, version)
    
    def render_template(self, name: str, **kwargs) -> str:
        """
        渲染模板
        
        Args:
            name: 模板名称
            **kwargs: 模板变量
            
        Returns:
            渲染后的文本
            
        Raises:
            ResourceError: 模板不存在
            ValidationError: 渲染失败
        """
        template = self.get_template(name)
        if not template:
            raise ResourceError(
                f"Template '{name}' not found",
                error_code=ErrorCodes.FILE_NOT_FOUND,
                context={"template_name": name}
            )
        
        return template.render(**kwargs)
    
    def list_templates(self, **filters) -> List[PromptTemplate]:
        """列出模板"""
        return self.registry.list_templates(**filters)
    
    def search_templates(self, query: str) -> List[PromptTemplate]:
        """搜索模板"""
        return self.registry.search(query)
    
    def delete_template(self, name: str) -> bool:
        """删除模板"""
        return self.registry.delete(name)
    
    def import_templates_from_directory(self, directory: Union[str, Path]) -> List[PromptTemplate]:
        """
        从目录批量导入模板
        
        Args:
            directory: 模板目录
            
        Returns:
            导入的模板列表
        """
        directory = Path(directory)
        imported_templates = []
        
        for template_file in directory.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                template = self.registry.import_template(content, "yaml")
                imported_templates.append(template)
                logger.info(f"Imported template: {template.name}")
            except Exception as e:
                logger.warning(f"Failed to import template {template_file}: {e}")
        
        return imported_templates
    
    def export_all_templates(self, directory: Union[str, Path]) -> None:
        """
        导出所有模板到目录
        
        Args:
            directory: 导出目录
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for template in self.list_templates():
            content = self.registry.export_template(template.name, "yaml")
            template_file = directory / f"{template.name}.yaml"
            
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Exported {len(self.list_templates())} templates to {directory}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.registry.get_statistics()
    
    def _register_builtin_templates(self) -> None:
        """注册内置模板"""
        
        # 通用系统提示词
        system_prompt = PromptTemplate(
            name="default_system",
            content="你是一个有用的AI助手。请以友好、准确和有帮助的方式回答用户的问题。",
            template_type=PromptType.SYSTEM,
            description="默认的系统提示词",
            language="zh",
            tags=["系统", "通用"]
        )
        
        # 代码解释提示词
        code_explanation = PromptTemplate(
            name="code_explanation",
            content="""请详细解释以下{{language}}代码的功能和实现原理：

```{{language}}
{{code}}
```

请从以下几个方面进行分析：
1. 代码的主要功能
2. 关键算法和数据结构
3. 代码的执行流程
4. 可能的优化建议""",
            template_type=PromptType.USER,
            format=TemplateFormat.JINJA2,
            description="代码解释模板",
            language="zh",
            tags=["代码", "解释", "编程"],
            variables=["language", "code"],
            required_variables=["code"],
            default_values={"language": "Python"}
        )
        
        # 文档总结提示词
        document_summary = PromptTemplate(
            name="document_summary",
            content="""请为以下文档内容生成一个简洁的总结：

{{document}}

总结要求：
- 长度控制在{{max_length}}字以内
- 包含主要观点和关键信息
- 语言简洁明了""",
            template_type=PromptType.USER,
            format=TemplateFormat.JINJA2,
            description="文档总结模板",
            language="zh",
            tags=["文档", "总结", "信息提取"],
            variables=["document", "max_length"],
            required_variables=["document"],
            default_values={"max_length": 200}
        )
        
        # 问答对话提示词
        qa_chat = PromptTemplate(
            name="qa_chat",
            content="""基于以下上下文信息回答用户问题：

上下文：
{{context}}

用户问题：{{question}}

请提供准确、详细的答案。如果上下文中没有相关信息，请说明无法找到相关信息。""",
            template_type=PromptType.CHAT,
            format=TemplateFormat.JINJA2,
            description="问答对话模板",
            language="zh",
            tags=["问答", "对话", "检索增强"],
            variables=["context", "question"],
            required_variables=["question"],
            default_values={"context": ""}
        )
        
        # 注册模板
        templates = [system_prompt, code_explanation, document_summary, qa_chat]
        for template in templates:
            try:
                self.registry.register(template)
            except Exception as e:
                logger.warning(f"Failed to register builtin template {template.name}: {e}")
        
        logger.debug("Registered builtin templates")


# 全局提示词管理器实例
_global_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """
    获取全局提示词管理器实例
    
    Returns:
        提示词管理器实例
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = PromptManager()
    
    return _global_manager


def create_template(
    name: str,
    content: str,
    template_type: PromptType = PromptType.USER,
    **kwargs
) -> PromptTemplate:
    """创建模板的便捷函数"""
    return get_prompt_manager().create_template(name, content, template_type, **kwargs)


def render_template(name: str, **kwargs) -> str:
    """渲染模板的便捷函数"""
    return get_prompt_manager().render_template(name, **kwargs)


def get_template(name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
    """获取模板的便捷函数"""
    return get_prompt_manager().get_template(name, version)