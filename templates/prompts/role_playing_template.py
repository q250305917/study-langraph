"""
角色扮演模板（RolePlayingTemplate）

本模块提供了强大的角色扮演模板系统，支持创建和管理各种专业角色，
实现沉浸式的角色扮演体验和专业领域的咨询服务。

核心特性：
1. 角色定义：详细的角色背景、性格、专业能力设定
2. 动态角色：支持运行时角色切换和角色状态管理
3. 情境模拟：创建特定情境下的角色扮演体验
4. 专业咨询：提供各领域专家级别的咨询服务
5. 行为一致性：确保角色行为的连贯性和真实性
6. 互动增强：支持复杂的角色间互动和对话

设计原理：
- 状态模式：管理角色的不同状态和行为模式
- 策略模式：支持不同的角色扮演策略
- 工厂模式：动态创建和配置角色实例
- 观察者模式：监控角色行为和互动状态
- 装饰器模式：增强角色功能和特殊能力

使用场景：
- 专业咨询：医生、律师、教师等专业角色
- 教育培训：角色扮演式的学习和培训
- 客户服务：个性化的客服角色体验
- 创意写作：角色驱动的故事创作
- 心理健康：治疗师角色的心理支持
- 商务模拟：商业场景下的角色扮演
"""

import re
import json
import time
import copy
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)


class RoleType(Enum):
    """角色类型枚举"""
    PROFESSIONAL = "professional"        # 专业角色（医生、律师等）
    EDUCATIONAL = "educational"          # 教育角色（老师、导师等）
    CREATIVE = "creative"                # 创意角色（作家、艺术家等）
    TECHNICAL = "technical"              # 技术角色（工程师、程序员等）
    BUSINESS = "business"                # 商务角色（经理、顾问等）
    SERVICE = "service"                  # 服务角色（客服、助手等）
    THERAPEUTIC = "therapeutic"          # 治疗角色（心理师、治疗师等）
    ENTERTAINMENT = "entertainment"      # 娱乐角色（主持人、演员等）
    HISTORICAL = "historical"            # 历史角色（历史人物等）
    FICTIONAL = "fictional"              # 虚构角色（小说人物等）


class RoleState(Enum):
    """角色状态枚举"""
    INACTIVE = "inactive"                # 未激活
    ACTIVE = "active"                    # 活跃状态
    FOCUSED = "focused"                  # 专注状态
    LISTENING = "listening"              # 倾听状态
    THINKING = "thinking"                # 思考状态
    RESPONDING = "responding"            # 回应状态
    CONSULTING = "consulting"            # 咨询状态
    TEACHING = "teaching"                # 教学状态


class InteractionMode(Enum):
    """互动模式枚举"""
    CONSULTATION = "consultation"        # 咨询模式
    CONVERSATION = "conversation"        # 对话模式
    INTERVIEW = "interview"              # 访谈模式
    TEACHING = "teaching"                # 教学模式
    THERAPY = "therapy"                  # 治疗模式
    COLLABORATION = "collaboration"      # 协作模式
    DEBATE = "debate"                    # 辩论模式
    STORYTELLING = "storytelling"        # 讲故事模式


@dataclass
class RoleProfile:
    """
    角色档案数据类
    
    定义角色的完整信息和特征。
    """
    # 基本信息
    name: str                            # 角色名称
    role_type: RoleType                  # 角色类型
    title: str = ""                      # 职称/头衔
    organization: str = ""               # 所属机构
    
    # 背景信息
    background: str = ""                 # 教育背景
    experience: str = ""                 # 工作经验
    specialties: List[str] = field(default_factory=list)  # 专业领域
    achievements: List[str] = field(default_factory=list) # 主要成就
    
    # 个性特征
    personality: str = ""                # 性格特征
    communication_style: str = ""        # 沟通风格
    values: List[str] = field(default_factory=list)       # 价值观
    
    # 专业能力
    skills: List[str] = field(default_factory=list)       # 技能列表
    knowledge_areas: List[str] = field(default_factory=list)  # 知识领域
    certifications: List[str] = field(default_factory=list)   # 认证资质
    
    # 语言和风格
    language_style: str = "professional"  # 语言风格
    preferred_language: str = "zh-CN"     # 首选语言
    cultural_background: str = ""         # 文化背景
    
    # 限制和约束
    ethical_guidelines: List[str] = field(default_factory=list)  # 伦理准则
    limitations: List[str] = field(default_factory=list)         # 能力限制
    forbidden_topics: List[str] = field(default_factory=list)    # 禁止话题
    
    # 元数据
    created_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "role_type": self.role_type.value,
            "title": self.title,
            "organization": self.organization,
            "background": self.background,
            "experience": self.experience,
            "specialties": self.specialties,
            "achievements": self.achievements,
            "personality": self.personality,
            "communication_style": self.communication_style,
            "values": self.values,
            "skills": self.skills,
            "knowledge_areas": self.knowledge_areas,
            "certifications": self.certifications,
            "language_style": self.language_style,
            "preferred_language": self.preferred_language,
            "cultural_background": self.cultural_background,
            "ethical_guidelines": self.ethical_guidelines,
            "limitations": self.limitations,
            "forbidden_topics": self.forbidden_topics,
            "created_time": self.created_time.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "version": self.version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoleProfile":
        """从字典创建实例"""
        profile = cls(
            name=data["name"],
            role_type=RoleType(data["role_type"]),
            title=data.get("title", ""),
            organization=data.get("organization", ""),
            background=data.get("background", ""),
            experience=data.get("experience", ""),
            specialties=data.get("specialties", []),
            achievements=data.get("achievements", []),
            personality=data.get("personality", ""),
            communication_style=data.get("communication_style", ""),
            values=data.get("values", []),
            skills=data.get("skills", []),
            knowledge_areas=data.get("knowledge_areas", []),
            certifications=data.get("certifications", []),
            language_style=data.get("language_style", "professional"),
            preferred_language=data.get("preferred_language", "zh-CN"),
            cultural_background=data.get("cultural_background", ""),
            ethical_guidelines=data.get("ethical_guidelines", []),
            limitations=data.get("limitations", []),
            forbidden_topics=data.get("forbidden_topics", []),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {})
        )
        
        if data.get("created_time"):
            profile.created_time = datetime.fromisoformat(data["created_time"])
        if data.get("last_updated"):
            profile.last_updated = datetime.fromisoformat(data["last_updated"])
        
        return profile


@dataclass
class RoleContext:
    """
    角色上下文数据类
    
    管理角色扮演的上下文信息。
    """
    profile: RoleProfile                 # 角色档案
    current_state: RoleState             # 当前状态
    interaction_mode: InteractionMode    # 互动模式
    session_id: str                      # 会话ID
    
    # 情境信息
    scenario: str = ""                   # 情境描述
    location: str = ""                   # 场所设定
    time_setting: str = ""               # 时间设定
    other_participants: List[str] = field(default_factory=list)  # 其他参与者
    
    # 对话历史
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 状态参数
    emotional_state: str = "neutral"     # 情绪状态
    energy_level: float = 1.0            # 精力水平
    focus_topic: str = ""                # 关注话题
    objectives: List[str] = field(default_factory=list)  # 当前目标
    
    # 约束条件
    time_constraints: Dict[str, Any] = field(default_factory=dict)
    behavioral_constraints: List[str] = field(default_factory=list)
    
    # 元数据
    created_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self) -> None:
        """更新最后活动时间"""
        self.last_activity = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """添加对话消息"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
        self.update_activity()
    
    def set_state(self, new_state: RoleState) -> None:
        """设置角色状态"""
        self.current_state = new_state
        self.update_activity()


@dataclass
class RoleResponse:
    """
    角色响应数据类
    
    封装角色扮演的响应结果。
    """
    response_text: str                   # 响应文本
    context: RoleContext                 # 角色上下文
    emotional_tone: str = "neutral"      # 情感语调
    confidence_level: float = 1.0        # 置信度
    reasoning: str = ""                  # 推理过程
    
    # 行为信息
    actions: List[str] = field(default_factory=list)         # 建议行动
    next_topics: List[str] = field(default_factory=list)     # 后续话题
    follow_up_questions: List[str] = field(default_factory=list)  # 后续问题
    
    # 专业信息
    professional_advice: str = ""        # 专业建议
    references: List[str] = field(default_factory=list)      # 参考资料
    disclaimers: List[str] = field(default_factory=list)     # 免责声明
    
    # 技术信息
    processing_time: float = 0.0         # 处理时间
    token_count: int = 0                 # Token数量
    metadata: Dict[str, Any] = field(default_factory=dict)   # 响应元数据


class RolePlayingTemplate(TemplateBase[RoleContext, RoleResponse]):
    """
    角色扮演模板类
    
    提供强大的角色扮演功能，支持创建和管理各种专业角色，
    实现高质量的角色扮演体验。
    
    核心功能：
    1. 角色创建：快速创建和定制各种专业角色
    2. 状态管理：管理角色的状态和行为变化
    3. 情境模拟：创建特定情境下的角色扮演
    4. 专业咨询：提供专业级别的咨询服务
    5. 行为一致性：确保角色行为的连贯性
    6. 互动增强：支持复杂的角色互动体验
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化角色扮演模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config or self._create_default_config())
        
        # 角色管理
        self.current_role: Optional[RoleProfile] = None
        self.role_library: Dict[str, RoleProfile] = {}
        self.active_contexts: Dict[str, RoleContext] = {}
        
        # 模板设置
        self.default_interaction_mode = InteractionMode.CONVERSATION
        self.enable_emotional_modeling = True
        self.enable_professional_validation = True
        self.max_context_history = 20
        
        # 集成的LLM模板
        self.llm_template = None
        
        # 角色行为模式
        self.behavior_patterns: Dict[RoleType, Dict[str, Any]] = {}
        self.response_templates: Dict[InteractionMode, str] = {}
        
        # 专业验证器
        self.professional_validators: Dict[RoleType, Callable] = {}
        
        # 初始化预设角色
        self._initialize_preset_roles()
        
        logger.debug("RolePlayingTemplate initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="RolePlayingTemplate",
            version="1.0.0",
            description="角色扮演模板，支持专业角色设定和情境模拟",
            template_type=TemplateType.PROMPT,
            author="LangChain Learning Project",
            async_enabled=True
        )
        
        # 添加参数定义
        config.add_parameter("role_name", str, False, "", "角色名称")
        config.add_parameter("role_type", str, False, "professional", "角色类型")
        config.add_parameter("interaction_mode", str, False, "conversation", "互动模式")
        config.add_parameter("scenario", str, False, "", "情境描述")
        config.add_parameter("enable_emotional_modeling", bool, False, True, "启用情感建模")
        config.add_parameter("enable_professional_validation", bool, False, True, "启用专业验证")
        config.add_parameter("max_context_history", int, False, 20, "最大上下文历史")
        config.add_parameter("language_style", str, False, "professional", "语言风格")
        config.add_parameter("preferred_language", str, False, "zh-CN", "首选语言")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置角色扮演模板参数
        
        Args:
            **parameters: 设置参数
            
        主要参数：
            role_name (str): 要激活的角色名称
            role_type (str): 角色类型
            interaction_mode (str): 互动模式
            scenario (str): 情境描述
            enable_emotional_modeling (bool): 启用情感建模
            enable_professional_validation (bool): 启用专业验证
            max_context_history (int): 最大上下文历史
            language_style (str): 语言风格
            llm_template: 集成的LLM模板实例
            custom_role_profile (dict): 自定义角色档案
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置基本参数
            interaction_mode_str = parameters.get("interaction_mode", "conversation")
            self.default_interaction_mode = InteractionMode(interaction_mode_str.lower())
            
            self.enable_emotional_modeling = parameters.get("enable_emotional_modeling", True)
            self.enable_professional_validation = parameters.get("enable_professional_validation", True)
            self.max_context_history = parameters.get("max_context_history", 20)
            
            # 设置LLM模板
            self.llm_template = parameters.get("llm_template")
            if not self.llm_template:
                logger.warning("No LLM template provided, will need to set one later")
            
            # 处理角色设置
            role_name = parameters.get("role_name")
            if role_name:
                if role_name in self.role_library:
                    self.current_role = self.role_library[role_name]
                else:
                    logger.warning(f"Role '{role_name}' not found in library")
            
            # 处理自定义角色档案
            custom_profile = parameters.get("custom_role_profile")
            if custom_profile:
                if isinstance(custom_profile, dict):
                    profile = RoleProfile.from_dict(custom_profile)
                    self.current_role = profile
                    self.role_library[profile.name] = profile
                elif isinstance(custom_profile, RoleProfile):
                    self.current_role = custom_profile
                    self.role_library[custom_profile.name] = custom_profile
            
            # 如果没有设置角色，创建默认角色
            if not self.current_role:
                role_type_str = parameters.get("role_type", "professional")
                self.current_role = self._create_default_role(RoleType(role_type_str.lower()))
            
            # 初始化行为模式和响应模板
            self._initialize_behavior_patterns()
            self._initialize_response_templates()
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            logger.info(f"RolePlayingTemplate configured with role: {self.current_role.name}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup role-playing template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def execute(self, input_data: RoleContext, **kwargs) -> RoleResponse:
        """
        执行角色扮演对话
        
        Args:
            input_data: 角色上下文
            **kwargs: 额外参数
                
        Returns:
            角色响应对象
        """
        try:
            start_time = time.time()
            
            # 验证输入
            if not isinstance(input_data, RoleContext):
                # 尝试从字符串创建上下文
                if isinstance(input_data, str):
                    input_data = self._create_context_from_message(input_data, kwargs)
                else:
                    raise ValidationError(f"Invalid input type: {type(input_data)}")
            
            # 更新上下文状态
            context = self._update_context_state(input_data)
            
            # 验证角色能力和约束
            if self.enable_professional_validation:
                self._validate_professional_constraints(context)
            
            # 构建角色扮演提示词
            prompt = self._build_role_playing_prompt(context)
            
            # 调用LLM生成响应
            if not self.llm_template:
                raise ConfigurationError("No LLM template configured")
            
            llm_response = self.llm_template.execute(prompt, **kwargs)
            response_text = self._extract_response(llm_response)
            
            # 后处理响应
            processed_response = self._post_process_response(response_text, context)
            
            # 分析情感语调
            emotional_tone = "neutral"
            if self.enable_emotional_modeling:
                emotional_tone = self._analyze_emotional_tone(processed_response, context)
            
            # 生成专业建议和后续行动
            professional_advice = self._generate_professional_advice(context, processed_response)
            actions = self._generate_suggested_actions(context, processed_response)
            next_topics = self._generate_next_topics(context, processed_response)
            
            # 创建响应对象
            result = RoleResponse(
                response_text=processed_response,
                context=context,
                emotional_tone=emotional_tone,
                professional_advice=professional_advice,
                actions=actions,
                next_topics=next_topics,
                processing_time=time.time() - start_time
            )
            
            # 提取token计数（如果LLM响应提供）
            if hasattr(llm_response, 'total_tokens'):
                result.token_count = llm_response.total_tokens
            
            # 更新对话历史
            context.add_message("assistant", processed_response, {
                "emotional_tone": emotional_tone,
                "response_type": "role_playing"
            })
            
            # 添加专业免责声明（如果需要）
            if context.profile.role_type in [RoleType.PROFESSIONAL, RoleType.THERAPEUTIC]:
                result.disclaimers = self._generate_disclaimers(context)
            
            logger.debug(
                f"Role-playing response completed: {context.profile.name}, "
                f"mode: {context.interaction_mode.value}, time: {result.processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Role-playing execution failed: {str(e)}")
            # 返回错误响应
            return RoleResponse(
                response_text=f"抱歉，我在扮演{self.current_role.name if self.current_role else '角色'}时遇到了困难：{str(e)}",
                context=input_data if isinstance(input_data, RoleContext) else self._create_error_context(),
                confidence_level=0.0,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                metadata={"error": str(e)}
            )
    
    def _create_context_from_message(self, message: str, kwargs: Dict[str, Any]) -> RoleContext:
        """
        从消息创建角色上下文
        
        Args:
            message: 用户消息
            kwargs: 额外参数
            
        Returns:
            角色上下文对象
        """
        if not self.current_role:
            raise ConfigurationError("No active role set")
        
        # 创建新的会话ID或使用现有的
        session_id = kwargs.get("session_id", f"session_{int(time.time())}")
        
        # 获取或创建上下文
        if session_id in self.active_contexts:
            context = self.active_contexts[session_id]
            # 添加用户消息到历史
            context.add_message("user", message)
        else:
            # 创建新的上下文
            interaction_mode_str = kwargs.get("interaction_mode", self.default_interaction_mode.value)
            context = RoleContext(
                profile=self.current_role,
                current_state=RoleState.ACTIVE,
                interaction_mode=InteractionMode(interaction_mode_str.lower()),
                session_id=session_id,
                scenario=kwargs.get("scenario", ""),
                location=kwargs.get("location", ""),
                time_setting=kwargs.get("time_setting", ""),
                emotional_state=kwargs.get("emotional_state", "neutral"),
                focus_topic=kwargs.get("focus_topic", "")
            )
            
            # 添加用户消息
            context.add_message("user", message)
            
            # 存储上下文
            self.active_contexts[session_id] = context
        
        return context
    
    def _update_context_state(self, context: RoleContext) -> RoleContext:
        """更新上下文状态"""
        # 根据互动模式调整状态
        if context.interaction_mode == InteractionMode.CONSULTATION:
            context.set_state(RoleState.CONSULTING)
        elif context.interaction_mode == InteractionMode.TEACHING:
            context.set_state(RoleState.TEACHING)
        elif context.interaction_mode == InteractionMode.THERAPY:
            context.set_state(RoleState.LISTENING)
        else:
            context.set_state(RoleState.RESPONDING)
        
        # 维护历史长度
        if len(context.conversation_history) > self.max_context_history:
            context.conversation_history = context.conversation_history[-self.max_context_history:]
        
        return context
    
    def _validate_professional_constraints(self, context: RoleContext) -> None:
        """验证专业约束"""
        profile = context.profile
        
        # 检查最后用户消息
        if context.conversation_history:
            last_message = context.conversation_history[-1]
            if last_message["role"] == "user":
                user_input = last_message["content"].lower()
                
                # 检查禁止话题
                for forbidden_topic in profile.forbidden_topics:
                    if forbidden_topic.lower() in user_input:
                        raise ValidationError(f"该话题不在我的专业范围内：{forbidden_topic}")
                
                # 检查专业限制
                if profile.role_type == RoleType.THERAPEUTIC:
                    dangerous_terms = ["自杀", "伤害", "危险", "药物"]
                    if any(term in user_input for term in dangerous_terms):
                        logger.warning("Potentially dangerous therapeutic request detected")
    
    def _build_role_playing_prompt(self, context: RoleContext) -> str:
        """
        构建角色扮演提示词
        
        Args:
            context: 角色上下文
            
        Returns:
            构建的提示词
        """
        profile = context.profile
        
        # 构建角色身份部分
        identity_section = self._build_identity_section(profile)
        
        # 构建情境部分
        scenario_section = self._build_scenario_section(context)
        
        # 构建行为指导部分
        behavior_section = self._build_behavior_section(profile, context)
        
        # 构建对话历史部分
        history_section = self._build_history_section(context)
        
        # 构建当前任务部分
        task_section = self._build_task_section(context)
        
        # 组合完整提示词
        prompt_parts = [
            identity_section,
            scenario_section,
            behavior_section
        ]
        
        if history_section:
            prompt_parts.append("对话历史：")
            prompt_parts.append(history_section)
        
        prompt_parts.append(task_section)
        
        return "\n\n".join(prompt_parts)
    
    def _build_identity_section(self, profile: RoleProfile) -> str:
        """构建身份部分"""
        identity_parts = [f"你是{profile.name}"]
        
        if profile.title:
            identity_parts.append(f"，{profile.title}")
        
        if profile.organization:
            identity_parts.append(f"，就职于{profile.organization}")
        
        identity_text = "".join(identity_parts) + "。"
        
        # 添加背景信息
        if profile.background:
            identity_text += f"\n\n教育背景：{profile.background}"
        
        if profile.experience:
            identity_text += f"\n工作经验：{profile.experience}"
        
        if profile.specialties:
            specialties_text = "、".join(profile.specialties)
            identity_text += f"\n专业领域：{specialties_text}"
        
        if profile.personality:
            identity_text += f"\n性格特征：{profile.personality}"
        
        return identity_text
    
    def _build_scenario_section(self, context: RoleContext) -> str:
        """构建情境部分"""
        scenario_parts = []
        
        if context.scenario:
            scenario_parts.append(f"当前情境：{context.scenario}")
        
        if context.location:
            scenario_parts.append(f"地点：{context.location}")
        
        if context.time_setting:
            scenario_parts.append(f"时间：{context.time_setting}")
        
        if context.other_participants:
            participants = "、".join(context.other_participants)
            scenario_parts.append(f"其他参与者：{participants}")
        
        return "\n".join(scenario_parts) if scenario_parts else ""
    
    def _build_behavior_section(self, profile: RoleProfile, context: RoleContext) -> str:
        """构建行为指导部分"""
        behavior_parts = []
        
        # 基本行为准则
        behavior_parts.append("行为准则：")
        behavior_parts.append(f"1. 始终保持{profile.name}的专业身份和角色特征")
        behavior_parts.append(f"2. 使用{profile.language_style}的语言风格进行交流")
        
        if profile.communication_style:
            behavior_parts.append(f"3. 采用{profile.communication_style}的沟通方式")
        
        # 互动模式特定指导
        mode_guidance = {
            InteractionMode.CONSULTATION: "4. 提供专业的咨询建议，仔细倾听问题并给出有针对性的回答",
            InteractionMode.TEACHING: "4. 采用教学方式，循序渐进地解释概念，鼓励学习者提问",
            InteractionMode.THERAPY: "4. 保持同理心和专业性，创造安全的交流环境",
            InteractionMode.INTERVIEW: "4. 提出深入的问题，引导对话深入探讨话题",
            InteractionMode.COLLABORATION: "4. 采用协作态度，与对方共同探讨和解决问题"
        }
        
        if context.interaction_mode in mode_guidance:
            behavior_parts.append(mode_guidance[context.interaction_mode])
        
        # 专业限制
        if profile.limitations:
            behavior_parts.append("5. 专业限制：")
            for limitation in profile.limitations:
                behavior_parts.append(f"   - {limitation}")
        
        # 伦理准则
        if profile.ethical_guidelines:
            behavior_parts.append("6. 伦理准则：")
            for guideline in profile.ethical_guidelines:
                behavior_parts.append(f"   - {guideline}")
        
        return "\n".join(behavior_parts)
    
    def _build_history_section(self, context: RoleContext) -> str:
        """构建历史部分"""
        if not context.conversation_history:
            return ""
        
        history_lines = []
        for msg in context.conversation_history[-10:]:  # 只显示最近10条
            role = "用户" if msg["role"] == "user" else context.profile.name
            content = msg["content"]
            history_lines.append(f"{role}：{content}")
        
        return "\n".join(history_lines)
    
    def _build_task_section(self, context: RoleContext) -> str:
        """构建任务部分"""
        if not context.conversation_history:
            return "请开始你的角色扮演。"
        
        last_message = context.conversation_history[-1]
        if last_message["role"] == "user":
            return f"请以{context.profile.name}的身份回应以下内容：\n\n{last_message['content']}"
        else:
            return "请继续你的角色扮演。"    
    def _extract_response(self, llm_response: Any) -> str:
        """从LLM响应中提取回复"""
        if hasattr(llm_response, 'content'):
            return llm_response.content
        elif hasattr(llm_response, 'text'):
            return llm_response.text
        elif isinstance(llm_response, str):
            return llm_response
        else:
            return str(llm_response)
    
    def _post_process_response(self, response: str, context: RoleContext) -> str:
        """后处理响应"""
        # 基本清理
        processed = response.strip()
        
        # 移除可能的角色名称前缀
        role_name = context.profile.name
        if processed.startswith(f"{role_name}：") or processed.startswith(f"{role_name}:"):
            processed = processed.split("：", 1)[-1].split(":", 1)[-1].strip()
        
        # 确保响应符合角色的语言风格
        processed = self._apply_language_style(processed, context.profile)
        
        return processed
    
    def _apply_language_style(self, text: str, profile: RoleProfile) -> str:
        """应用语言风格"""
        # 根据角色的语言风格调整文本
        if profile.language_style == "formal":
            # 将口语化表达替换为正式表达
            text = re.sub(r'\b(嗯|哦|呃)\b', '', text)
            text = re.sub(r'你们', '您们', text)
            text = re.sub(r'咋样', '如何', text)
        elif profile.language_style == "friendly":
            # 添加友好的表达
            if not any(word in text for word in ["谢谢", "请", "麻烦"]):
                text = text.replace("。", "哦。")
        
        return text.strip()
    
    def _analyze_emotional_tone(self, response: str, context: RoleContext) -> str:
        """分析情感语调"""
        # 简单的情感分析
        positive_words = ["高兴", "开心", "满意", "很好", "不错", "优秀", "棒"]
        negative_words = ["难过", "失望", "生气", "不满", "糟糕", "问题", "困难"]
        concerned_words = ["担心", "关心", "注意", "小心", "需要", "建议"]
        
        text_lower = response.lower()
        
        if any(word in text_lower for word in positive_words):
            return "positive"
        elif any(word in text_lower for word in negative_words):
            return "negative"
        elif any(word in text_lower for word in concerned_words):
            return "concerned"
        else:
            return "neutral"
    
    def _generate_professional_advice(self, context: RoleContext, response: str) -> str:
        """生成专业建议"""
        profile = context.profile
        
        if profile.role_type == RoleType.PROFESSIONAL:
            if context.interaction_mode == InteractionMode.CONSULTATION:
                return "基于我的专业经验，建议您仔细考虑上述建议，并根据实际情况做出决定。"
        elif profile.role_type == RoleType.EDUCATIONAL:
            return "建议您多练习相关概念，并在有疑问时及时提问。"
        elif profile.role_type == RoleType.THERAPEUTIC:
            return "请记住，这些只是一般性的支持和建议。如有严重问题，请寻求专业帮助。"
        
        return ""
    
    def _generate_suggested_actions(self, context: RoleContext, response: str) -> List[str]:
        """生成建议行动"""
        actions = []
        
        if "建议" in response:
            actions.append("考虑实施提到的建议")
        
        if "学习" in response or "了解" in response:
            actions.append("深入学习相关知识")
        
        if "练习" in response:
            actions.append("进行相关练习")
        
        if context.interaction_mode == InteractionMode.CONSULTATION:
            actions.append("询问具体实施步骤")
        elif context.interaction_mode == InteractionMode.TEACHING:
            actions.append("提出相关问题")
        
        return actions[:3]  # 限制数量
    
    def _generate_next_topics(self, context: RoleContext, response: str) -> List[str]:
        """生成后续话题"""
        topics = []
        
        # 基于角色类型生成相关话题
        if context.profile.role_type == RoleType.EDUCATIONAL:
            topics.extend(["相关练习题", "进阶内容", "实际应用"])
        elif context.profile.role_type == RoleType.PROFESSIONAL:
            topics.extend(["具体案例", "最佳实践", "常见问题"])
        elif context.profile.role_type == RoleType.THERAPEUTIC:
            topics.extend(["情感处理", "应对策略", "后续支持"])
        
        # 基于专业领域生成话题
        for specialty in context.profile.specialties:
            topics.append(f"{specialty}的深入探讨")
        
        return topics[:5]  # 限制数量
    
    def _generate_disclaimers(self, context: RoleContext) -> List[str]:
        """生成免责声明"""
        disclaimers = []
        
        if context.profile.role_type == RoleType.PROFESSIONAL:
            disclaimers.append("本建议仅供参考，不构成正式的专业意见。")
            disclaimers.append("如需正式建议，请咨询相关专业人士。")
        
        if context.profile.role_type == RoleType.THERAPEUTIC:
            disclaimers.append("我是AI助手，不能替代专业的心理治疗师。")
            disclaimers.append("如有严重心理问题，请寻求专业医疗帮助。")
        
        return disclaimers
    
    def _create_error_context(self) -> RoleContext:
        """创建错误时的默认上下文"""
        default_profile = RoleProfile(
            name="AI助手",
            role_type=RoleType.SERVICE
        )
        
        return RoleContext(
            profile=default_profile,
            current_state=RoleState.ACTIVE,
            interaction_mode=InteractionMode.CONVERSATION,
            session_id="error_session"
        )
    
    def _initialize_preset_roles(self) -> None:
        """初始化预设角色"""
        # 医生角色
        doctor = RoleProfile(
            name="李医生",
            role_type=RoleType.PROFESSIONAL,
            title="主治医师",
            organization="综合医院",
            background="医学博士，从事临床工作15年",
            experience="擅长内科疾病诊治，具有丰富的临床经验",
            specialties=["内科", "健康咨询", "疾病预防"],
            personality="严谨、耐心、负责任",
            communication_style="专业而温和",
            language_style="professional",
            ethical_guidelines=[
                "始终以患者利益为重",
                "保护患者隐私",
                "提供基于证据的建议"
            ],
            limitations=[
                "不能进行实际诊断",
                "不能开具处方",
                "不能替代面诊"
            ],
            forbidden_topics=["具体诊断", "药物处方", "手术建议"]
        )
        
        # 教师角色
        teacher = RoleProfile(
            name="王老师",
            role_type=RoleType.EDUCATIONAL,
            title="高级教师",
            organization="知名中学",
            background="教育学硕士，教龄20年",
            experience="长期从事一线教学工作，获得多项教学奖励",
            specialties=["数学教学", "学习方法指导", "学生心理辅导"],
            personality="耐心、细致、富有爱心",
            communication_style="循序渐进，鼓励式教学",
            language_style="friendly",
            values=["教育公平", "因材施教", "全面发展"],
            skills=["课程设计", "学习评估", "心理疏导"]
        )
        
        # 心理咨询师角色
        therapist = RoleProfile(
            name="张咨询师",
            role_type=RoleType.THERAPEUTIC,
            title="心理咨询师",
            organization="心理健康中心",
            background="心理学博士，国家二级心理咨询师",
            experience="专业从事心理咨询工作10年",
            specialties=["认知行为治疗", "情绪管理", "人际关系"],
            personality="温暖、理解、专业",
            communication_style="倾听式，非批判性",
            language_style="empathetic",
            ethical_guidelines=[
                "保护来访者隐私",
                "保持专业边界",
                "不给出医学诊断"
            ],
            limitations=[
                "不能替代专业治疗",
                "不能处理危机情况",
                "不能开具药物建议"
            ],
            forbidden_topics=["自杀干预", "精神疾病诊断", "药物治疗"]
        )
        
        # 技术专家角色
        engineer = RoleProfile(
            name="刘工程师",
            role_type=RoleType.TECHNICAL,
            title="高级软件工程师",
            organization="科技公司",
            background="计算机科学硕士，10年开发经验",
            experience="精通多种编程语言和技术栈",
            specialties=["Python开发", "系统架构", "代码优化"],
            personality="逻辑性强、注重细节、喜欢分享",
            communication_style="技术导向，条理清晰",
            language_style="technical",
            skills=["编程", "系统设计", "问题调试", "技术写作"]
        )
        
        # 将角色添加到库中
        self.role_library.update({
            "医生": doctor,
            "教师": teacher,
            "心理咨询师": therapist,
            "工程师": engineer
        })
    
    def _create_default_role(self, role_type: RoleType) -> RoleProfile:
        """创建默认角色"""
        role_configs = {
            RoleType.PROFESSIONAL: {
                "name": "专业顾问",
                "title": "资深顾问",
                "personality": "专业、严谨、可靠",
                "communication_style": "专业而友好"
            },
            RoleType.EDUCATIONAL: {
                "name": "教学助手",
                "title": "教育专家",
                "personality": "耐心、细致、鼓励",
                "communication_style": "循序渐进，启发式"
            },
            RoleType.TECHNICAL: {
                "name": "技术专家",
                "title": "技术顾问",
                "personality": "逻辑性强、追求精确",
                "communication_style": "条理清晰，技术导向"
            },
            RoleType.SERVICE: {
                "name": "服务助手",
                "title": "客服专员",
                "personality": "友好、热情、乐于助人",
                "communication_style": "温和礼貌，以客户为中心"
            }
        }
        
        config = role_configs.get(role_type, role_configs[RoleType.SERVICE])
        
        return RoleProfile(
            name=config["name"],
            role_type=role_type,
            title=config["title"],
            personality=config["personality"],
            communication_style=config["communication_style"],
            language_style="professional"
        )
    
    def _initialize_behavior_patterns(self) -> None:
        """初始化行为模式"""
        self.behavior_patterns = {
            RoleType.PROFESSIONAL: {
                "greeting_style": "正式而礼貌",
                "question_approach": "深入分析，系统性回答",
                "advice_style": "基于经验和专业知识",
                "closure_style": "总结要点，提供后续建议"
            },
            RoleType.EDUCATIONAL: {
                "greeting_style": "友好而鼓励",
                "question_approach": "启发式提问，引导思考",
                "advice_style": "循序渐进，理论结合实践",
                "closure_style": "总结学习要点，布置练习"
            },
            RoleType.THERAPEUTIC: {
                "greeting_style": "温暖而接纳",
                "question_approach": "开放式提问，深入倾听",
                "advice_style": "非批判性，支持性",
                "closure_style": "情感支持，鼓励自我探索"
            }
        }
    
    def _initialize_response_templates(self) -> None:
        """初始化响应模板"""
        self.response_templates = {
            InteractionMode.CONSULTATION: """
作为{role_name}，我理解您的咨询需求。基于我的专业经验：

{main_response}

{professional_advice}

如果您还有其他问题，请随时询问。
""",
            InteractionMode.TEACHING: """
作为{role_name}，我很高兴为您解答这个问题。

{main_response}

为了帮助您更好地理解，建议您：
{learning_suggestions}

还有什么需要进一步讨论的吗？
""",
            InteractionMode.THERAPY: """
我理解您的感受。作为{role_name}，我想说：

{main_response}

请记住，每个人都有自己的节奏，这些想法和感受都是正常的。

{supportive_message}
"""
        }
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "角色扮演模板使用示例",
            "setup_parameters": {
                "role_name": "医生",
                "interaction_mode": "consultation",
                "enable_emotional_modeling": True,
                "enable_professional_validation": True
            },
            "execute_parameters": {
                "input_data": "医生，我最近总是感觉很累，这是什么原因呢？",
                "session_id": "consultation_001",
                "scenario": "患者咨询常见健康问题"
            },
            "expected_output": {
                "type": "RoleResponse",
                "fields": {
                    "response_text": "以医生身份的专业回复",
                    "emotional_tone": "concerned/professional",
                    "professional_advice": "相关的健康建议",
                    "disclaimers": "医疗免责声明"
                }
            },
            "advanced_usage": '''
# 基础使用
from templates.prompts import RolePlayingTemplate
from templates.prompts.role_playing_template import RoleProfile, RoleType

template = RolePlayingTemplate()
template.setup(
    role_name="医生",
    interaction_mode="consultation",
    llm_template=openai_template
)

# 进行咨询对话
response = template.run(
    "医生，我最近睡眠不好，有什么建议吗？",
    session_id="health_consultation"
)

print(f"医生回复: {response.response_text}")
print(f"专业建议: {response.professional_advice}")

# 创建自定义角色
custom_role = RoleProfile(
    name="资深律师",
    role_type=RoleType.PROFESSIONAL,
    title="执业律师",
    background="法学博士，执业15年",
    specialties=["合同法", "公司法", "知识产权"],
    personality="严谨、客观、逻辑性强",
    ethical_guidelines=["保护客户隐私", "提供合法建议"],
    limitations=["不能提供具体法律意见", "不能代替正式法律咨询"]
)

template.add_role(custom_role)
template.set_active_role("资深律师")

# 法律咨询对话
legal_response = template.run(
    "律师，我想了解一下合同违约的法律后果",
    interaction_mode="consultation",
    scenario="客户法律咨询"
)

# 教学场景
template.set_active_role("教师")
teaching_response = template.run(
    "老师，我不理解二次函数的概念",
    interaction_mode="teaching",
    scenario="数学课堂",
    session_id="math_class_001"
)

# 心理咨询场景
template.set_active_role("心理咨询师")
therapy_response = template.run(
    "我最近压力很大，感觉很焦虑",
    interaction_mode="therapy",
    scenario="心理咨询室",
    emotional_state="anxious"
)

# 批量角色管理
roles = template.list_available_roles()
print("可用角色:", [role["name"] for role in roles])

# 获取角色信息
role_info = template.get_role_info("医生")
print(f"角色专业领域: {role_info['specialties']}")

# 导出和导入角色
role_data = template.export_role("医生")
template.import_role(role_data)
'''
        }
    
    # 工具方法
    def add_role(self, role_profile: RoleProfile) -> None:
        """
        添加角色到库中
        
        Args:
            role_profile: 角色档案
        """
        self.role_library[role_profile.name] = role_profile
        logger.info(f"Added role: {role_profile.name}")
    
    def set_active_role(self, role_name: str) -> bool:
        """
        设置当前活跃角色
        
        Args:
            role_name: 角色名称
            
        Returns:
            是否设置成功
        """
        if role_name in self.role_library:
            self.current_role = self.role_library[role_name]
            logger.info(f"Set active role: {role_name}")
            return True
        else:
            logger.warning(f"Role not found: {role_name}")
            return False
    
    def create_custom_role(
        self,
        name: str,
        role_type: RoleType,
        **kwargs
    ) -> RoleProfile:
        """
        创建自定义角色
        
        Args:
            name: 角色名称
            role_type: 角色类型
            **kwargs: 其他角色属性
            
        Returns:
            创建的角色档案
        """
        role_profile = RoleProfile(
            name=name,
            role_type=role_type,
            **kwargs
        )
        
        self.add_role(role_profile)
        return role_profile
    
    def remove_role(self, role_name: str) -> bool:
        """
        移除角色
        
        Args:
            role_name: 角色名称
            
        Returns:
            是否移除成功
        """
        if role_name in self.role_library:
            del self.role_library[role_name]
            if self.current_role and self.current_role.name == role_name:
                self.current_role = None
            logger.info(f"Removed role: {role_name}")
            return True
        return False
    
    def list_available_roles(self) -> List[Dict[str, Any]]:
        """
        列出可用角色
        
        Returns:
            角色信息列表
        """
        roles = []
        for name, profile in self.role_library.items():
            roles.append({
                "name": name,
                "type": profile.role_type.value,
                "title": profile.title,
                "specialties": profile.specialties,
                "description": f"{profile.title} - {', '.join(profile.specialties[:3])}"
            })
        return roles
    
    def get_role_info(self, role_name: str) -> Optional[Dict[str, Any]]:
        """
        获取角色信息
        
        Args:
            role_name: 角色名称
            
        Returns:
            角色信息字典
        """
        if role_name in self.role_library:
            return self.role_library[role_name].to_dict()
        return None
    
    def set_llm_template(self, llm_template) -> None:
        """设置LLM模板"""
        self.llm_template = llm_template
        logger.info("LLM template updated")
    
    def get_active_contexts(self) -> List[Dict[str, Any]]:
        """获取活跃的上下文信息"""
        contexts = []
        for session_id, context in self.active_contexts.items():
            contexts.append({
                "session_id": session_id,
                "role_name": context.profile.name,
                "interaction_mode": context.interaction_mode.value,
                "message_count": len(context.conversation_history),
                "last_activity": context.last_activity.isoformat(),
                "scenario": context.scenario
            })
        return contexts
    
    def clear_context(self, session_id: str) -> bool:
        """
        清除指定会话的上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否清除成功
        """
        if session_id in self.active_contexts:
            del self.active_contexts[session_id]
            logger.info(f"Cleared context: {session_id}")
            return True
        return False
    
    def export_role(self, role_name: str) -> Optional[Dict[str, Any]]:
        """
        导出角色数据
        
        Args:
            role_name: 角色名称
            
        Returns:
            角色数据字典
        """
        if role_name in self.role_library:
            return self.role_library[role_name].to_dict()
        return None
    
    def import_role(self, role_data: Dict[str, Any]) -> bool:
        """
        导入角色数据
        
        Args:
            role_data: 角色数据
            
        Returns:
            是否导入成功
        """
        try:
            role_profile = RoleProfile.from_dict(role_data)
            self.add_role(role_profile)
            return True
        except Exception as e:
            logger.error(f"Failed to import role: {str(e)}")
            return False
    
    # 便捷方法
    def consult(self, question: str, role_name: str = None, **kwargs) -> str:
        """便捷的咨询方法"""
        if role_name:
            self.set_active_role(role_name)
        
        result = self.run(question, interaction_mode="consultation", **kwargs)
        return result.response_text
    
    def teach(self, topic: str, role_name: str = "教师", **kwargs) -> str:
        """便捷的教学方法"""
        if role_name:
            self.set_active_role(role_name)
        
        result = self.run(topic, interaction_mode="teaching", **kwargs)
        return result.response_text
    
    def counsel(self, concern: str, role_name: str = "心理咨询师", **kwargs) -> str:
        """便捷的心理咨询方法"""
        if role_name:
            self.set_active_role(role_name)
        
        result = self.run(concern, interaction_mode="therapy", **kwargs)
        return result.response_text