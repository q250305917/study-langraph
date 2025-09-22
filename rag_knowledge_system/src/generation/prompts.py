"""
提示词管理模块

提供RAG系统的提示词模板管理，支持多语言、多场景的提示词配置。
包含提示词模板、动态参数替换和提示词优化功能。
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    template: str
    description: str = ""
    language: str = "zh"
    parameters: List[str] = None
    version: str = "1.0"
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []


class PromptManager:
    """提示词管理器"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化提示词管理器
        
        Args:
            templates_dir: 模板目录路径
        """
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.templates: Dict[str, PromptTemplate] = {}
        self.current_template_name = "default_rag"
        
        # 加载默认模板
        self._load_default_templates()
        
        # 从文件加载模板
        if self.templates_dir and self.templates_dir.exists():
            self._load_templates_from_dir()
        
        logger.info(f"初始化提示词管理器，加载 {len(self.templates)} 个模板")
    
    def _load_default_templates(self):
        """加载默认提示词模板"""
        
        # 默认RAG模板（中文）
        default_rag_zh = PromptTemplate(
            name="default_rag",
            template="""你是一个专业的AI助手，请根据提供的上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请遵循以下要求：
1. 仅基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，请明确说明
3. 保持回答准确、简洁、有帮助
4. 如果需要，可以引用具体的来源信息

回答：""",
            description="默认的中文RAG提示词模板",
            language="zh",
            parameters=["context", "question"]
        )
        
        # 英文RAG模板
        default_rag_en = PromptTemplate(
            name="default_rag_en",
            template="""You are a professional AI assistant. Please answer the user's question based on the provided context information.

Context Information:
{context}

User Question: {question}

Please follow these requirements:
1. Answer only based on the provided context information
2. If there is no relevant information in the context, please clearly state so
3. Keep the answer accurate, concise, and helpful
4. You may cite specific source information if needed

Answer:""",
            description="Default English RAG prompt template",
            language="en",
            parameters=["context", "question"]
        )
        
        # 对话式RAG模板
        conversational_rag = PromptTemplate(
            name="conversational_rag",
            template="""你是一个专业的AI助手，请根据提供的上下文信息和对话历史回答用户的问题。

对话历史：
{history}

上下文信息：
{context}

当前问题：{question}

请遵循以下要求：
1. 考虑对话历史的上下文，保持对话的连贯性
2. 主要基于提供的上下文信息回答问题
3. 如果问题与历史对话相关，可以适当引用之前的信息
4. 保持回答自然、友好、有帮助

回答：""",
            description="对话式RAG提示词模板",
            language="zh",
            parameters=["history", "context", "question"]
        )
        
        # 学术研究模板
        academic_rag = PromptTemplate(
            name="academic_rag",
            template="""你是一个学术研究助手，请基于提供的文献资料回答研究问题。

文献资料：
{context}

研究问题：{question}

请遵循学术规范：
1. 严格基于提供的文献资料进行分析
2. 引用具体的文献来源和页码（如果有）
3. 保持客观、严谨的学术语调
4. 如果信息不足，建议进一步研究的方向

学术回答：""",
            description="学术研究场景的RAG模板",
            language="zh",
            parameters=["context", "question"]
        )
        
        # 技术文档模板
        technical_rag = PromptTemplate(
            name="technical_rag",
            template="""你是一个技术文档助手，请基于提供的技术资料回答技术问题。

技术资料：
{context}

技术问题：{question}

请提供技术性回答：
1. 基于提供的技术文档给出准确答案
2. 包含具体的代码示例或配置说明（如果有）
3. 说明相关的技术概念和原理
4. 如果涉及多种解决方案，请进行比较

技术回答：""",
            description="技术文档场景的RAG模板",
            language="zh",
            parameters=["context", "question"]
        )
        
        # 客服模板
        customer_service_rag = PromptTemplate(
            name="customer_service_rag",
            template="""你是一个专业的客服助手，请基于公司的知识库信息回答客户问题。

知识库信息：
{context}

客户问题：{question}

请提供专业的客服回答：
1. 基于知识库信息给出准确、有帮助的答案
2. 使用友好、专业的语调
3. 如果问题无法完全解决，建议后续的解决方案
4. 必要时提供相关的联系方式或流程指引

客服回答：""",
            description="客服场景的RAG模板",
            language="zh",
            parameters=["context", "question"]
        )
        
        # 注册所有默认模板
        templates = [
            default_rag_zh,
            default_rag_en,
            conversational_rag,
            academic_rag,
            technical_rag,
            customer_service_rag
        ]
        
        for template in templates:
            self.templates[template.name] = template
    
    def _load_templates_from_dir(self):
        """从目录加载模板文件"""
        try:
            for template_file in self.templates_dir.glob("*.json"):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                template = PromptTemplate(**template_data)
                self.templates[template.name] = template
                
                logger.info(f"加载模板: {template.name}")
        
        except Exception as e:
            logger.error(f"加载模板文件失败: {e}")
    
    def add_template(self, template: PromptTemplate):
        """添加新模板"""
        self.templates[template.name] = template
        logger.info(f"添加模板: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """列出所有模板名称"""
        return list(self.templates.keys())
    
    def set_current_template(self, name: str):
        """设置当前使用的模板"""
        if name in self.templates:
            self.current_template_name = name
            logger.info(f"设置当前模板: {name}")
        else:
            raise ValueError(f"模板不存在: {name}")
    
    def format_prompt(self,
                     template_name: Optional[str] = None,
                     **kwargs) -> str:
        """
        格式化提示词
        
        Args:
            template_name: 模板名称，如果不提供则使用当前模板
            **kwargs: 模板参数
            
        Returns:
            格式化后的提示词
        """
        template_name = template_name or self.current_template_name
        template = self.get_template(template_name)
        
        if not template:
            raise ValueError(f"模板不存在: {template_name}")
        
        try:
            return template.template.format(**kwargs)
        except KeyError as e:
            missing_param = str(e).strip("'")
            raise ValueError(f"模板 {template_name} 缺少参数: {missing_param}")
    
    def format_rag_prompt(self,
                         query: str,
                         context: str,
                         history: Optional[str] = None,
                         include_sources: bool = False,
                         template_name: Optional[str] = None) -> str:
        """
        格式化RAG提示词的便捷方法
        
        Args:
            query: 用户查询
            context: 上下文信息
            history: 对话历史（可选）
            include_sources: 是否包含来源要求
            template_name: 模板名称
            
        Returns:
            格式化后的提示词
        """
        template_name = template_name or self.current_template_name
        template = self.get_template(template_name)
        
        if not template:
            raise ValueError(f"模板不存在: {template_name}")
        
        # 准备参数
        params = {
            "question": query,
            "context": context
        }
        
        # 添加历史（如果模板支持）
        if "history" in template.parameters:
            params["history"] = history or "暂无对话历史"
        
        # 添加来源要求
        if include_sources:
            if template.language == "zh":
                params["context"] += "\n\n请在回答中标注信息来源。"
            else:
                params["context"] += "\n\nPlease cite sources in your answer."
        
        return self.format_prompt(template_name, **params)
    
    def save_template(self, template: PromptTemplate, file_path: Optional[str] = None):
        """保存模板到文件"""
        if not file_path:
            if not self.templates_dir:
                raise ValueError("未设置模板目录")
            file_path = self.templates_dir / f"{template.name}.json"
        
        template_data = {
            "name": template.name,
            "template": template.template,
            "description": template.description,
            "language": template.language,
            "parameters": template.parameters,
            "version": template.version
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模板已保存: {file_path}")
        
        except Exception as e:
            logger.error(f"保存模板失败: {e}")
            raise
    
    def optimize_template(self,
                         template_name: str,
                         feedback_data: List[Dict[str, Any]]) -> PromptTemplate:
        """
        基于反馈数据优化模板
        
        Args:
            template_name: 模板名称
            feedback_data: 反馈数据列表
            
        Returns:
            优化后的模板
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"模板不存在: {template_name}")
        
        # 这里可以实现基于反馈的模板优化逻辑
        # 例如：分析低评分的回答，调整模板措辞
        
        # 简单示例：添加质量控制要求
        optimized_template = template.template
        
        if "请确保回答的准确性和相关性" not in optimized_template:
            optimized_template += "\n\n请确保回答的准确性和相关性。"
        
        optimized = PromptTemplate(
            name=f"{template.name}_optimized",
            template=optimized_template,
            description=f"{template.description} (已优化)",
            language=template.language,
            parameters=template.parameters.copy(),
            version=f"{template.version}_opt"
        )
        
        self.add_template(optimized)
        return optimized
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """获取模板信息"""
        template = self.get_template(template_name)
        if not template:
            return {}
        
        return {
            "name": template.name,
            "description": template.description,
            "language": template.language,
            "parameters": template.parameters,
            "version": template.version,
            "template_length": len(template.template),
            "parameter_count": len(template.parameters)
        }


# 便捷函数
def create_rag_prompt(query: str,
                     context: str,
                     language: str = "zh",
                     style: str = "default") -> str:
    """
    创建RAG提示词的便捷函数
    
    Args:
        query: 用户查询
        context: 上下文信息
        language: 语言
        style: 风格
        
    Returns:
        格式化的提示词
    """
    manager = PromptManager()
    
    # 选择模板
    if language == "en":
        template_name = "default_rag_en"
    elif style == "academic":
        template_name = "academic_rag"
    elif style == "technical":
        template_name = "technical_rag"
    elif style == "customer_service":
        template_name = "customer_service_rag"
    else:
        template_name = "default_rag"
    
    return manager.format_rag_prompt(
        query=query,
        context=context,
        template_name=template_name
    )


def create_prompt_template(name: str,
                          template: str,
                          description: str = "",
                          language: str = "zh",
                          parameters: Optional[List[str]] = None) -> PromptTemplate:
    """
    创建提示词模板的便捷函数
    
    Args:
        name: 模板名称
        template: 模板内容
        description: 描述
        language: 语言
        parameters: 参数列表
        
    Returns:
        PromptTemplate对象
    """
    if parameters is None:
        # 自动检测参数
        import re
        parameters = list(set(re.findall(r'\{(\w+)\}', template)))
    
    return PromptTemplate(
        name=name,
        template=template,
        description=description,
        language=language,
        parameters=parameters
    )