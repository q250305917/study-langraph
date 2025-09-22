"""
补全模板（CompletionTemplate）

本模块提供了强大的文本补全模板系统，专门用于文本续写、代码生成、内容创作等场景。
支持多种补全策略、上下文感知、格式化输出等高级功能。

核心特性：
1. 文本续写：智能续写各种类型的文本内容
2. 代码生成：生成高质量的代码片段和完整程序
3. 内容创作：协助创作文章、故事、诗歌等文学作品
4. 格式化输出：支持多种输出格式和结构化内容
5. 上下文感知：根据上下文智能调整补全策略
6. 质量控制：内置质量评估和优化机制

设计原理：
- 策略模式：支持不同的补全策略和生成方法
- 模板方法模式：定义补全流程的通用结构
- 工厂模式：根据任务类型创建相应的补全器
- 装饰器模式：增强补全功能和后处理
- 责任链模式：多步骤的内容生成和优化

使用场景：
- 代码补全：IDE中的智能代码补全
- 文档生成：自动生成技术文档和API文档
- 创意写作：辅助小说、剧本、诗歌创作
- 邮件撰写：自动撰写商务邮件和回复
- 报告生成：自动生成分析报告和总结
"""

import re
import json
import time
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)


class CompletionType(Enum):
    """补全类型枚举"""
    TEXT = "text"                # 通用文本补全
    CODE = "code"                # 代码补全
    ARTICLE = "article"          # 文章写作
    EMAIL = "email"              # 邮件撰写
    STORY = "story"              # 故事创作
    POETRY = "poetry"            # 诗歌创作
    DIALOGUE = "dialogue"        # 对话补全
    SUMMARY = "summary"          # 摘要生成
    TRANSLATION = "translation"  # 翻译补全
    TECHNICAL = "technical"      # 技术文档


class CompletionStrategy(Enum):
    """补全策略枚举"""
    CONTINUE = "continue"        # 续写模式
    EXPAND = "expand"            # 扩展模式
    COMPLETE = "complete"        # 完成模式
    REWRITE = "rewrite"          # 重写模式
    ENHANCE = "enhance"          # 增强模式
    OPTIMIZE = "optimize"        # 优化模式


@dataclass
class CompletionContext:
    """
    补全上下文数据类
    
    管理补全任务的上下文信息和参数。
    """
    completion_type: CompletionType      # 补全类型
    strategy: CompletionStrategy         # 补全策略
    input_text: str                      # 输入文本
    target_length: Optional[int] = None  # 目标长度
    style: str = "neutral"               # 写作风格
    tone: str = "professional"           # 语调
    audience: str = "general"            # 目标受众
    domain: str = "general"              # 专业领域
    language: str = "zh-CN"              # 语言
    format_requirements: Dict[str, Any] = field(default_factory=dict)  # 格式要求
    constraints: Dict[str, Any] = field(default_factory=dict)          # 约束条件
    metadata: Dict[str, Any] = field(default_factory=dict)             # 元数据


@dataclass
class CompletionResult:
    """
    补全结果数据类
    
    封装补全操作的结果和相关信息。
    """
    completed_text: str                  # 补全后的文本
    original_text: str                   # 原始文本
    context: CompletionContext           # 补全上下文
    confidence: float = 1.0              # 置信度
    quality_score: float = 1.0           # 质量分数
    processing_time: float = 0.0         # 处理时间
    token_count: int = 0                 # Token数量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 结果元数据
    
    # 分析信息
    added_length: int = 0                # 新增长度
    completion_ratio: float = 0.0        # 补全比例
    keywords: List[str] = field(default_factory=list)       # 关键词
    suggestions: List[str] = field(default_factory=list)    # 改进建议


class CompletionTemplate(TemplateBase[CompletionContext, CompletionResult]):
    """
    补全模板类
    
    提供强大的文本补全功能，支持多种类型的内容生成和续写任务。
    
    核心功能：
    1. 多类型补全：支持文本、代码、文章等多种内容类型
    2. 策略选择：提供多种补全策略和生成方法
    3. 上下文感知：智能分析输入内容并调整补全策略
    4. 质量控制：内置质量评估和优化机制
    5. 格式化输出：支持多种输出格式和结构化内容
    6. 性能优化：高效的生成算法和缓存机制
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化补全模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config or self._create_default_config())
        
        # 补全配置
        self.default_completion_type = CompletionType.TEXT
        self.default_strategy = CompletionStrategy.CONTINUE
        self.default_target_length = 500
        self.default_style = "neutral"
        self.default_tone = "professional"
        self.max_length = 2000
        self.min_quality_score = 0.5
        
        # 集成的LLM模板
        self.llm_template = None
        
        # 补全策略映射
        self.strategy_prompts = {}
        self.type_prompts = {}
        
        # 质量评估器
        self.quality_evaluators = {}
        
        # 后处理器
        self.post_processors = {}
        
        logger.debug("CompletionTemplate initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="CompletionTemplate",
            version="1.0.0",
            description="文本补全模板，支持文本续写、代码生成等",
            template_type=TemplateType.PROMPT,
            author="LangChain Learning Project",
            async_enabled=True
        )
        
        # 添加参数定义
        config.add_parameter("completion_type", str, False, "text", "补全类型")
        config.add_parameter("strategy", str, False, "continue", "补全策略")
        config.add_parameter("target_length", int, False, 500, "目标长度")
        config.add_parameter("style", str, False, "neutral", "写作风格")
        config.add_parameter("tone", str, False, "professional", "语调")
        config.add_parameter("audience", str, False, "general", "目标受众")
        config.add_parameter("domain", str, False, "general", "专业领域")
        config.add_parameter("language", str, False, "zh-CN", "语言")
        config.add_parameter("max_length", int, False, 2000, "最大长度限制")
        config.add_parameter("enable_quality_check", bool, False, True, "启用质量检查")
        config.add_parameter("format_requirements", dict, False, {}, "格式要求")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置补全模板参数
        
        Args:
            **parameters: 设置参数
            
        主要参数：
            completion_type (str): 补全类型
            strategy (str): 补全策略
            target_length (int): 目标长度
            style (str): 写作风格
            tone (str): 语调
            audience (str): 目标受众
            domain (str): 专业领域
            language (str): 语言
            max_length (int): 最大长度限制
            enable_quality_check (bool): 启用质量检查
            llm_template: 集成的LLM模板实例
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置基本参数
            completion_type_str = parameters.get("completion_type", "text")
            self.default_completion_type = CompletionType(completion_type_str.lower())
            
            strategy_str = parameters.get("strategy", "continue")
            self.default_strategy = CompletionStrategy(strategy_str.lower())
            
            self.default_target_length = parameters.get("target_length", 500)
            self.default_style = parameters.get("style", "neutral")
            self.default_tone = parameters.get("tone", "professional")
            self.default_audience = parameters.get("audience", "general")
            self.default_domain = parameters.get("domain", "general")
            self.language = parameters.get("language", "zh-CN")
            self.max_length = parameters.get("max_length", 2000)
            self.enable_quality_check = parameters.get("enable_quality_check", True)
            self.format_requirements = parameters.get("format_requirements", {})
            
            # 设置LLM模板
            self.llm_template = parameters.get("llm_template")
            if not self.llm_template:
                logger.warning("No LLM template provided, will need to set one later")
            
            # 初始化策略提示词
            self._initialize_strategy_prompts()
            self._initialize_type_prompts()
            
            # 初始化质量评估器
            self._initialize_quality_evaluators()
            
            # 初始化后处理器
            self._initialize_post_processors()
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            logger.info(f"CompletionTemplate configured with type: {self.default_completion_type.value}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup completion template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def execute(self, input_data: CompletionContext, **kwargs) -> CompletionResult:
        """
        执行补全任务
        
        Args:
            input_data: 补全上下文
            **kwargs: 额外参数
                
        Returns:
            补全结果对象
        """
        try:
            start_time = time.time()
            
            # 验证输入
            if not isinstance(input_data, CompletionContext):
                # 尝试从字符串创建上下文
                if isinstance(input_data, str):
                    input_data = self._create_context_from_text(input_data, kwargs)
                else:
                    raise ValidationError(f"Invalid input type: {type(input_data)}")
            
            # 分析输入文本
            analyzed_context = self._analyze_input(input_data)
            
            # 选择补全策略
            strategy = self._select_strategy(analyzed_context)
            
            # 构建提示词
            prompt = self._build_completion_prompt(analyzed_context, strategy)
            
            # 调用LLM进行补全
            if not self.llm_template:
                raise ConfigurationError("No LLM template configured")
            
            llm_response = self.llm_template.execute(prompt, **kwargs)
            completion_text = self._extract_content(llm_response)
            
            # 后处理
            processed_text = self._post_process(completion_text, analyzed_context)
            
            # 质量评估
            quality_score = 1.0
            if self.enable_quality_check:
                quality_score = self._evaluate_quality(processed_text, analyzed_context)
            
            # 创建结果对象
            result = CompletionResult(
                completed_text=processed_text,
                original_text=analyzed_context.input_text,
                context=analyzed_context,
                quality_score=quality_score,
                processing_time=time.time() - start_time
            )
            
            # 添加分析信息
            result.added_length = len(processed_text) - len(analyzed_context.input_text)
            result.completion_ratio = result.added_length / len(analyzed_context.input_text) if analyzed_context.input_text else 0
            result.keywords = self._extract_keywords(processed_text)
            result.suggestions = self._generate_suggestions(result)
            
            # 提取token计数（如果LLM响应提供）
            if hasattr(llm_response, 'total_tokens'):
                result.token_count = llm_response.total_tokens
            
            logger.debug(
                f"Completion finished: {result.added_length} chars added, "
                f"quality: {quality_score:.2f}, time: {result.processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Completion execution failed: {str(e)}")
            # 返回错误结果
            return CompletionResult(
                completed_text=input_data.input_text if hasattr(input_data, 'input_text') else str(input_data),
                original_text=input_data.input_text if hasattr(input_data, 'input_text') else str(input_data),
                context=input_data if isinstance(input_data, CompletionContext) else CompletionContext(
                    completion_type=CompletionType.TEXT,
                    strategy=CompletionStrategy.CONTINUE,
                    input_text=str(input_data)
                ),
                confidence=0.0,
                quality_score=0.0,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                metadata={"error": str(e)}
            )
    
    def _create_context_from_text(self, text: str, kwargs: Dict[str, Any]) -> CompletionContext:
        """
        从文本创建补全上下文
        
        Args:
            text: 输入文本
            kwargs: 额外参数
            
        Returns:
            补全上下文对象
        """
        completion_type_str = kwargs.get("completion_type", self.default_completion_type.value)
        strategy_str = kwargs.get("strategy", self.default_strategy.value)
        
        return CompletionContext(
            completion_type=CompletionType(completion_type_str.lower()),
            strategy=CompletionStrategy(strategy_str.lower()),
            input_text=text,
            target_length=kwargs.get("target_length", self.default_target_length),
            style=kwargs.get("style", self.default_style),
            tone=kwargs.get("tone", self.default_tone),
            audience=kwargs.get("audience", getattr(self, 'default_audience', 'general')),
            domain=kwargs.get("domain", getattr(self, 'default_domain', 'general')),
            language=kwargs.get("language", self.language),
            format_requirements=kwargs.get("format_requirements", self.format_requirements),
            constraints=kwargs.get("constraints", {})
        )
    
    def _analyze_input(self, context: CompletionContext) -> CompletionContext:
        """
        分析输入文本并增强上下文
        
        Args:
            context: 原始上下文
            
        Returns:
            增强后的上下文
        """
        # 分析文本特征
        text = context.input_text
        
        # 检测语言（简单实现）
        if context.language == "auto":
            context.language = self._detect_language(text)
        
        # 分析文本类型（如果未指定）
        if context.completion_type == CompletionType.TEXT:
            detected_type = self._detect_text_type(text)
            if detected_type:
                context.completion_type = detected_type
        
        # 分析写作风格（如果未指定）
        if context.style == "auto":
            context.style = self._detect_style(text)
        
        # 提取关键信息
        context.metadata.update({
            "input_length": len(text),
            "word_count": len(text.split()),
            "has_code": bool(re.search(r'```|`[^`]+`|\b(def|class|function|var|let|const)\b', text)),
            "has_urls": bool(re.search(r'https?://\S+', text)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            "sentence_count": len(re.split(r'[.!?。！？]', text)),
            "paragraph_count": len(text.split('\n\n'))
        })
        
        return context
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言（简单实现）"""
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        if chinese_chars / total_chars > 0.3:
            return "zh-CN"
        else:
            return "en-US"
    
    def _detect_text_type(self, text: str) -> Optional[CompletionType]:
        """检测文本类型"""
        text_lower = text.lower()
        
        # 代码特征
        if re.search(r'(def|class|function|import|#include|public|private)', text):
            return CompletionType.CODE
        
        # 邮件特征
        if re.search(r'(dear|hello|regards|sincerely|best|subject|from|to)', text_lower):
            return CompletionType.EMAIL
        
        # 技术文档特征
        if re.search(r'(api|documentation|readme|installation|configuration)', text_lower):
            return CompletionType.TECHNICAL
        
        # 故事特征
        if re.search(r'(once upon|chapter|characters?|plot)', text_lower):
            return CompletionType.STORY
        
        return None
    
    def _detect_style(self, text: str) -> str:
        """检测写作风格"""
        text_lower = text.lower()
        
        # 正式风格
        if re.search(r'(therefore|furthermore|however|nevertheless|consequently)', text_lower):
            return "formal"
        
        # 随意风格
        if re.search(r'(yeah|okay|gonna|wanna|btw)', text_lower):
            return "casual"
        
        # 技术风格
        if re.search(r'(implementation|algorithm|optimization|performance)', text_lower):
            return "technical"
        
        return "neutral"    
    def _select_strategy(self, context: CompletionContext) -> CompletionStrategy:
        """
        选择补全策略
        
        Args:
            context: 补全上下文
            
        Returns:
            选择的策略
        """
        # 基于上下文智能选择策略
        text = context.input_text
        
        # 如果文本很短，使用扩展策略
        if len(text) < 50:
            return CompletionStrategy.EXPAND
        
        # 如果文本看起来不完整，使用续写策略
        if not text.rstrip().endswith(('.', '!', '?', '。', '！', '？')):
            return CompletionStrategy.CONTINUE
        
        # 如果是代码且有语法错误，使用完成策略
        if context.completion_type == CompletionType.CODE:
            return CompletionStrategy.COMPLETE
        
        # 默认使用配置的策略
        return context.strategy
    
    def _initialize_strategy_prompts(self) -> None:
        """初始化策略提示词"""
        self.strategy_prompts = {
            CompletionStrategy.CONTINUE: """
基于以下内容，自然地续写下去，保持风格和语调的一致性：

{input_text}

续写要求：
- 保持与原文相同的写作风格和语调
- 内容要连贯，逻辑要清晰
- 目标长度约 {target_length} 字符
- 语言：{language}
- 受众：{audience}
""",
            CompletionStrategy.EXPAND: """
请扩展以下内容，增加更多细节和深度：

{input_text}

扩展要求：
- 保留原有内容的核心思想
- 添加相关的细节、例子或解释
- 使内容更加丰富和完整
- 目标长度约 {target_length} 字符
- 风格：{style}
- 语调：{tone}
""",
            CompletionStrategy.COMPLETE: """
请完成以下未完成的内容：

{input_text}

完成要求：
- 分析内容的意图和目标
- 补充缺失的部分
- 确保内容完整和连贯
- 保持专业性和准确性
- 目标长度约 {target_length} 字符
""",
            CompletionStrategy.REWRITE: """
请重写以下内容，改进表达和结构：

{input_text}

重写要求：
- 保持原意不变
- 改进语言表达和文章结构
- 使内容更加清晰易懂
- 风格：{style}
- 语调：{tone}
- 受众：{audience}
""",
            CompletionStrategy.ENHANCE: """
请增强以下内容，使其更加生动和吸引人：

{input_text}

增强要求：
- 保留原有信息
- 使用更生动的语言
- 增加吸引力和感染力
- 适合目标受众：{audience}
- 风格：{style}
""",
            CompletionStrategy.OPTIMIZE: """
请优化以下内容，提高质量和效果：

{input_text}

优化要求：
- 改进逻辑结构
- 提高可读性
- 消除冗余和模糊
- 专业领域：{domain}
- 目标受众：{audience}
"""
        }
    
    def _initialize_type_prompts(self) -> None:
        """初始化类型特定的提示词"""
        self.type_prompts = {
            CompletionType.CODE: """
作为一个专业的程序员，请{strategy_action}以下代码：

{input_text}

代码要求：
- 遵循最佳编程实践
- 代码要规范、可读性强
- 添加必要的注释
- 考虑性能和安全性
- 确保代码的完整性和正确性
""",
            CompletionType.ARTICLE: """
作为一个专业的撰稿人，请{strategy_action}以下文章：

{input_text}

文章要求：
- 结构清晰，逻辑严密
- 语言流畅，表达准确
- 内容有深度和价值
- 适合目标读者：{audience}
- 写作风格：{style}
""",
            CompletionType.EMAIL: """
作为一个商务沟通专家，请{strategy_action}以下邮件：

{input_text}

邮件要求：
- 语调：{tone}
- 保持专业和礼貌
- 内容简洁明了
- 目的明确
- 考虑收件人的背景
""",
            CompletionType.STORY: """
作为一个创意作家，请{strategy_action}以下故事：

{input_text}

故事要求：
- 情节生动有趣
- 人物形象鲜明
- 语言富有感染力
- 保持故事的连贯性
- 风格：{style}
""",
            CompletionType.TECHNICAL: """
作为一个技术文档专家，请{strategy_action}以下技术内容：

{input_text}

技术文档要求：
- 准确性和专业性
- 结构化和系统性
- 易于理解和操作
- 包含必要的示例
- 专业领域：{domain}
""",
            CompletionType.POETRY: """
作为一个诗人，请{strategy_action}以下诗歌：

{input_text}

诗歌要求：
- 富有韵律和节奏感
- 意境优美，情感丰富
- 语言精炼，表达深刻
- 风格：{style}
- 保持诗歌的艺术性
"""
        }
    
    def _build_completion_prompt(self, context: CompletionContext, strategy: CompletionStrategy) -> str:
        """
        构建补全提示词
        
        Args:
            context: 补全上下文
            strategy: 补全策略
            
        Returns:
            构建的提示词
        """
        # 获取策略动作描述
        strategy_actions = {
            CompletionStrategy.CONTINUE: "续写",
            CompletionStrategy.EXPAND: "扩展",
            CompletionStrategy.COMPLETE: "完成",
            CompletionStrategy.REWRITE: "重写",
            CompletionStrategy.ENHANCE: "增强",
            CompletionStrategy.OPTIMIZE: "优化"
        }
        
        # 准备模板变量
        template_vars = {
            "input_text": context.input_text,
            "target_length": context.target_length or self.default_target_length,
            "style": context.style,
            "tone": context.tone,
            "audience": context.audience,
            "domain": context.domain,
            "language": context.language,
            "strategy_action": strategy_actions.get(strategy, "处理")
        }
        
        # 选择提示词模板
        if context.completion_type in self.type_prompts:
            prompt_template = self.type_prompts[context.completion_type]
        else:
            prompt_template = self.strategy_prompts.get(strategy, self.strategy_prompts[CompletionStrategy.CONTINUE])
        
        # 应用模板变量
        prompt = prompt_template.format(**template_vars)
        
        # 添加格式要求
        if context.format_requirements:
            format_text = self._build_format_requirements(context.format_requirements)
            prompt += f"\n\n格式要求：\n{format_text}"
        
        # 添加约束条件
        if context.constraints:
            constraints_text = self._build_constraints_text(context.constraints)
            prompt += f"\n\n约束条件：\n{constraints_text}"
        
        return prompt
    
    def _build_format_requirements(self, requirements: Dict[str, Any]) -> str:
        """构建格式要求文本"""
        format_parts = []
        
        if requirements.get("structure"):
            format_parts.append(f"结构：{requirements['structure']}")
        
        if requirements.get("output_format"):
            format_parts.append(f"输出格式：{requirements['output_format']}")
        
        if requirements.get("sections"):
            sections = ", ".join(requirements["sections"])
            format_parts.append(f"包含以下部分：{sections}")
        
        if requirements.get("length_limit"):
            format_parts.append(f"长度限制：{requirements['length_limit']} 字符")
        
        if requirements.get("style_guide"):
            format_parts.append(f"风格指南：{requirements['style_guide']}")
        
        return "\n".join([f"- {part}" for part in format_parts])
    
    def _build_constraints_text(self, constraints: Dict[str, Any]) -> str:
        """构建约束条件文本"""
        constraint_parts = []
        
        if constraints.get("forbidden_words"):
            words = ", ".join(constraints["forbidden_words"])
            constraint_parts.append(f"禁用词汇：{words}")
        
        if constraints.get("required_keywords"):
            keywords = ", ".join(constraints["required_keywords"])
            constraint_parts.append(f"必须包含：{keywords}")
        
        if constraints.get("max_length"):
            constraint_parts.append(f"最大长度：{constraints['max_length']} 字符")
        
        if constraints.get("min_length"):
            constraint_parts.append(f"最小长度：{constraints['min_length']} 字符")
        
        if constraints.get("content_restrictions"):
            constraint_parts.append(f"内容限制：{constraints['content_restrictions']}")
        
        return "\n".join([f"- {part}" for part in constraint_parts])
    
    def _extract_content(self, llm_response: Any) -> str:
        """从LLM响应中提取内容"""
        if hasattr(llm_response, 'content'):
            return llm_response.content
        elif hasattr(llm_response, 'text'):
            return llm_response.text
        elif isinstance(llm_response, str):
            return llm_response
        else:
            return str(llm_response)
    
    def _post_process(self, text: str, context: CompletionContext) -> str:
        """
        后处理生成的文本
        
        Args:
            text: 生成的文本
            context: 补全上下文
            
        Returns:
            后处理的文本
        """
        # 基本清理
        processed_text = text.strip()
        
        # 应用类型特定的后处理
        if context.completion_type in self.post_processors:
            processor = self.post_processors[context.completion_type]
            processed_text = processor(processed_text, context)
        
        # 长度控制
        if len(processed_text) > self.max_length:
            processed_text = self._truncate_text(processed_text, self.max_length)
        
        # 质量检查
        processed_text = self._basic_quality_check(processed_text, context)
        
        return processed_text
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """智能截断文本"""
        if len(text) <= max_length:
            return text
        
        # 尝试在句子边界截断
        sentences = re.split(r'[.!?。！？]', text[:max_length])
        if len(sentences) > 1:
            return ".".join(sentences[:-1]) + "."
        
        # 尝试在段落边界截断
        paragraphs = text[:max_length].split('\n\n')
        if len(paragraphs) > 1:
            return '\n\n'.join(paragraphs[:-1])
        
        # 简单截断
        return text[:max_length].rstrip() + "..."
    
    def _basic_quality_check(self, text: str, context: CompletionContext) -> str:
        """基本质量检查和修复"""
        # 移除多余的空行
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # 修复标点符号
        text = re.sub(r'\s+([.!?。！？,，;；:])', r'\1', text)
        
        # 修复引号
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        return text.strip()
    
    def _initialize_quality_evaluators(self) -> None:
        """初始化质量评估器"""
        self.quality_evaluators = {
            "coherence": self._evaluate_coherence,
            "completeness": self._evaluate_completeness,
            "relevance": self._evaluate_relevance,
            "style_consistency": self._evaluate_style_consistency,
            "technical_accuracy": self._evaluate_technical_accuracy
        }
    
    def _evaluate_quality(self, text: str, context: CompletionContext) -> float:
        """
        评估文本质量
        
        Args:
            text: 要评估的文本
            context: 补全上下文
            
        Returns:
            质量分数 (0.0-1.0)
        """
        scores = []
        
        # 运行所有评估器
        for name, evaluator in self.quality_evaluators.items():
            try:
                score = evaluator(text, context)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Quality evaluator {name} failed: {str(e)}")
                scores.append(0.5)  # 默认中等分数
        
        # 计算平均分数
        return sum(scores) / len(scores) if scores else 0.5
    
    def _evaluate_coherence(self, text: str, context: CompletionContext) -> float:
        """评估连贯性"""
        # 简单的连贯性检查
        sentences = re.split(r'[.!?。！？]', text)
        if len(sentences) < 2:
            return 1.0
        
        # 检查句子长度变化
        lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # 长度变化太大可能表示连贯性问题
        coherence_score = max(0.0, 1.0 - variance / (avg_length ** 2))
        return min(1.0, coherence_score)
    
    def _evaluate_completeness(self, text: str, context: CompletionContext) -> float:
        """评估完整性"""
        # 检查是否以完整的句子结束
        if text.rstrip().endswith(('.', '!', '?', '。', '！', '？')):
            return 1.0
        elif text.rstrip().endswith((',', ';', '，', '；', ':')):
            return 0.3
        else:
            return 0.6
    
    def _evaluate_relevance(self, text: str, context: CompletionContext) -> float:
        """评估相关性"""
        input_words = set(context.input_text.lower().split())
        output_words = set(text.lower().split())
        
        if not input_words:
            return 1.0
        
        # 计算词汇重叠度
        overlap = len(input_words.intersection(output_words))
        relevance_score = overlap / len(input_words)
        
        return min(1.0, relevance_score * 2)  # 放大相关性分数
    
    def _evaluate_style_consistency(self, text: str, context: CompletionContext) -> float:
        """评估风格一致性"""
        # 简单的风格检查
        if context.style == "formal":
            informal_words = ['yeah', 'okay', 'gonna', 'wanna']
            if any(word in text.lower() for word in informal_words):
                return 0.3
        elif context.style == "casual":
            formal_words = ['therefore', 'furthermore', 'nevertheless']
            if any(word in text.lower() for word in formal_words):
                return 0.3
        
        return 1.0
    
    def _evaluate_technical_accuracy(self, text: str, context: CompletionContext) -> float:
        """评估技术准确性"""
        if context.completion_type == CompletionType.CODE:
            # 检查代码的基本语法特征
            if re.search(r'[(){}[\]]+', text):
                return 0.8
            else:
                return 0.4
        
        return 1.0
    
    def _initialize_post_processors(self) -> None:
        """初始化后处理器"""
        self.post_processors = {
            CompletionType.CODE: self._post_process_code,
            CompletionType.EMAIL: self._post_process_email,
            CompletionType.ARTICLE: self._post_process_article
        }
    
    def _post_process_code(self, text: str, context: CompletionContext) -> str:
        """代码后处理"""
        # 确保代码块有适当的格式
        if '```' not in text and context.input_text.strip().startswith('```'):
            text = '```\n' + text + '\n```'
        
        return text
    
    def _post_process_email(self, text: str, context: CompletionContext) -> str:
        """邮件后处理"""
        # 确保邮件有适当的结尾
        if not text.rstrip().endswith(('regards', '谢谢', '此致', 'best')):
            if context.language == "zh-CN":
                text += "\n\n此致\n敬礼！"
            else:
                text += "\n\nBest regards,"
        
        return text
    
    def _post_process_article(self, text: str, context: CompletionContext) -> str:
        """文章后处理"""
        # 确保段落格式正确
        paragraphs = text.split('\n\n')
        processed_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if para:
                processed_paragraphs.append(para)
        
        return '\n\n'.join(processed_paragraphs)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # 过滤常见停用词
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     '的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那', '有', '没'}
        
        keywords = [word for word in words if word not in stop_words]
        
        # 返回频率最高的词
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]
    
    def _generate_suggestions(self, result: CompletionResult) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if result.quality_score < 0.7:
            suggestions.append("考虑重新生成以提高质量")
        
        if result.added_length < 50:
            suggestions.append("内容可能过于简短，考虑扩展")
        
        if result.completion_ratio > 3:
            suggestions.append("生成内容过长，考虑精简")
        
        if result.context.completion_type == CompletionType.CODE and 'TODO' in result.completed_text:
            suggestions.append("代码中包含TODO项，需要进一步完善")
        
        return suggestions
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "文本补全模板使用示例",
            "setup_parameters": {
                "completion_type": "article",
                "strategy": "continue",
                "target_length": 800,
                "style": "professional",
                "tone": "informative",
                "language": "zh-CN",
                "max_length": 2000
            },
            "execute_parameters": {
                "input_data": "人工智能技术的快速发展正在改变我们的生活方式。从智能手机到自动驾驶汽车，AI的应用已经深入到各个领域",
                "completion_type": "article",
                "target_length": 500,
                "style": "professional"
            },
            "expected_output": {
                "type": "CompletionResult",
                "fields": {
                    "completed_text": "续写后的完整文章内容",
                    "quality_score": "0.8-1.0之间的质量分数",
                    "keywords": "提取的关键词列表",
                    "suggestions": "改进建议"
                }
            },
            "advanced_usage": '''
# 文本续写
from templates.prompts import CompletionTemplate
from templates.prompts.completion_template import CompletionContext, CompletionType, CompletionStrategy

template = CompletionTemplate()
template.setup(
    completion_type="article",
    style="professional",
    llm_template=openai_template
)

# 方式1：直接使用字符串
result = template.run(
    "人工智能的发展历程可以追溯到...",
    completion_type="article",
    target_length=500
)

# 方式2：使用上下文对象
context = CompletionContext(
    completion_type=CompletionType.ARTICLE,
    strategy=CompletionStrategy.EXPAND,
    input_text="深度学习是机器学习的一个重要分支",
    target_length=800,
    style="academic",
    tone="informative",
    audience="researchers"
)

result = template.run(context)

# 代码补全
code_context = CompletionContext(
    completion_type=CompletionType.CODE,
    strategy=CompletionStrategy.COMPLETE,
    input_text="""
def fibonacci(n):
    if n <= 1:
        return n
    # TODO: 实现递归逻辑
""",
    target_length=200,
    domain="algorithms"
)

code_result = template.run(code_context)

# 邮件撰写
email_result = template.run(
    "尊敬的张总，\n\n关于下周的项目会议安排",
    completion_type="email",
    tone="formal",
    target_length=300
)

# 带格式要求的补全
format_result = template.run(
    "机器学习的主要算法类型包括",
    completion_type="technical",
    format_requirements={
        "structure": "列表格式",
        "sections": ["监督学习", "无监督学习", "强化学习"],
        "output_format": "markdown"
    },
    constraints={
        "required_keywords": ["分类", "聚类", "回归"],
        "max_length": 1000
    }
)

print(f"生成文本: {result.completed_text}")
print(f"质量分数: {result.quality_score}")
print(f"关键词: {result.keywords}")
print(f"建议: {result.suggestions}")
'''
        }
    
    # 工具方法
    def set_llm_template(self, llm_template) -> None:
        """设置LLM模板"""
        self.llm_template = llm_template
        logger.info("LLM template updated")
    
    def add_custom_post_processor(self, completion_type: CompletionType, processor: Callable) -> None:
        """添加自定义后处理器"""
        self.post_processors[completion_type] = processor
        logger.info(f"Added custom post processor for {completion_type.value}")
    
    def add_custom_quality_evaluator(self, name: str, evaluator: Callable) -> None:
        """添加自定义质量评估器"""
        self.quality_evaluators[name] = evaluator
        logger.info(f"Added custom quality evaluator: {name}")
    
    # 便捷方法
    def complete_text(self, text: str, target_length: int = 500, **kwargs) -> str:
        """便捷的文本补全方法"""
        result = self.run(text, target_length=target_length, **kwargs)
        return result.completed_text
    
    def complete_code(self, code: str, **kwargs) -> str:
        """便捷的代码补全方法"""
        result = self.run(code, completion_type="code", **kwargs)
        return result.completed_text
    
    def write_article(self, topic: str, target_length: int = 1000, **kwargs) -> str:
        """便捷的文章写作方法"""
        result = self.run(topic, completion_type="article", target_length=target_length, **kwargs)
        return result.completed_text
    
    def compose_email(self, start_text: str, tone: str = "professional", **kwargs) -> str:
        """便捷的邮件撰写方法"""
        result = self.run(start_text, completion_type="email", tone=tone, **kwargs)
        return result.completed_text