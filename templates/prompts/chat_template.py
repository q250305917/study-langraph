"""
对话模板（ChatTemplate）

本模块提供了强大灵活的对话模板系统，专门用于多轮对话场景。
支持角色设定、对话历史管理、动态参数替换等高级功能。

核心特性：
1. 多轮对话：支持复杂的多轮对话管理，保持上下文连贯性
2. 角色设定：支持自定义角色人格、专业背景、对话风格
3. 模板组合：支持多个模板的嵌套和组合使用
4. 动态参数：支持运行时参数替换和条件分支
5. 对话历史：智能管理对话历史，支持摘要和压缩
6. 格式控制：支持多种输出格式和结构化响应

设计原理：
- 状态机模式：管理对话的不同状态和转换
- 策略模式：支持不同的角色和对话策略
- 模板方法模式：定义对话流程的通用结构
- 观察者模式：监控对话状态变化和事件
- 装饰器模式：扩展对话功能和行为

使用场景：
- 客服机器人：专业的客户服务对话
- 教学助手：个性化的教学指导对话
- 角色扮演：沉浸式的角色扮演体验
- 专家咨询：特定领域的专业咨询
- 创意写作：协作式的创意内容生成
"""

import re
import json
import time
import copy
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"      # 系统消息
    USER = "user"          # 用户消息
    ASSISTANT = "assistant" # 助手消息
    FUNCTION = "function"   # 函数调用消息


class ConversationState(Enum):
    """对话状态枚举"""
    INITIALIZED = "initialized"    # 已初始化
    ACTIVE = "active"              # 活跃对话中
    WAITING = "waiting"            # 等待用户输入
    PROCESSING = "processing"       # 处理中
    PAUSED = "paused"              # 已暂停
    COMPLETED = "completed"        # 已完成
    ERROR = "error"                # 错误状态


@dataclass
class Message:
    """
    对话消息数据类
    
    表示对话中的单条消息，包含角色、内容、时间戳等信息。
    """
    role: MessageRole                    # 消息角色
    content: str                         # 消息内容
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)     # 元数据
    
    # 可选字段
    name: Optional[str] = None           # 发送者名称
    function_call: Optional[Dict] = None # 函数调用信息
    tool_calls: Optional[List[Dict]] = None  # 工具调用列表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建实例"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            name=data.get("name"),
            function_call=data.get("function_call"),
            tool_calls=data.get("tool_calls"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationContext:
    """
    对话上下文数据类
    
    管理对话的上下文信息，包括参数、状态、历史等。
    """
    conversation_id: str                 # 对话ID
    state: ConversationState             # 对话状态
    parameters: Dict[str, Any] = field(default_factory=dict)  # 上下文参数
    user_profile: Dict[str, Any] = field(default_factory=dict)  # 用户档案
    session_data: Dict[str, Any] = field(default_factory=dict)  # 会话数据
    created_time: datetime = field(default_factory=datetime.now)  # 创建时间
    last_activity: datetime = field(default_factory=datetime.now)  # 最后活动时间
    
    def update_activity(self) -> None:
        """更新最后活动时间"""
        self.last_activity = datetime.now()
    
    def set_parameter(self, key: str, value: Any) -> None:
        """设置上下文参数"""
        self.parameters[key] = value
        self.update_activity()
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """获取上下文参数"""
        return self.parameters.get(key, default)


@dataclass
class ChatResponse:
    """
    对话响应数据类
    
    封装对话模板的响应结果。
    """
    message: Message                     # 响应消息
    context: ConversationContext         # 对话上下文
    suggested_actions: List[str] = field(default_factory=list)  # 建议操作
    metadata: Dict[str, Any] = field(default_factory=dict)      # 响应元数据
    
    # 统计信息
    processing_time: float = 0.0         # 处理时间
    token_count: int = 0                 # Token数量
    confidence: float = 1.0              # 置信度


class ConversationHistory:
    """
    对话历史管理器
    
    管理对话历史记录，支持存储、检索、摘要、压缩等功能。
    """
    
    def __init__(self, max_messages: int = 100, auto_summarize: bool = True):
        """
        初始化对话历史管理器
        
        Args:
            max_messages: 最大消息数量
            auto_summarize: 是否自动摘要
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.auto_summarize = auto_summarize
        self.summary: Optional[str] = None
        self.summary_up_to_index: int = 0
    
    def add_message(self, message: Message) -> None:
        """
        添加消息到历史记录
        
        Args:
            message: 要添加的消息
        """
        self.messages.append(message)
        
        # 检查是否需要清理历史记录
        if len(self.messages) > self.max_messages:
            self._cleanup_history()
    
    def get_messages(
        self, 
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
        role_filter: Optional[MessageRole] = None
    ) -> List[Message]:
        """
        获取消息列表
        
        Args:
            limit: 限制返回的消息数量
            since: 从指定时间开始的消息
            role_filter: 按角色过滤
            
        Returns:
            符合条件的消息列表
        """
        filtered_messages = self.messages
        
        # 时间过滤
        if since:
            filtered_messages = [msg for msg in filtered_messages if msg.timestamp >= since]
        
        # 角色过滤
        if role_filter:
            filtered_messages = [msg for msg in filtered_messages if msg.role == role_filter]
        
        # 数量限制
        if limit:
            filtered_messages = filtered_messages[-limit:]
        
        return filtered_messages
    
    def get_context_messages(self, context_length: int = 10) -> List[Message]:
        """
        获取用于上下文的消息列表
        
        Args:
            context_length: 上下文长度
            
        Returns:
            上下文消息列表
        """
        # 如果有摘要，包含摘要和最近的消息
        if self.summary and self.summary_up_to_index > 0:
            summary_message = Message(
                role=MessageRole.SYSTEM,
                content=f"对话摘要：{self.summary}",
                metadata={"type": "summary"}
            )
            recent_messages = self.messages[self.summary_up_to_index:][-context_length:]
            return [summary_message] + recent_messages
        else:
            return self.messages[-context_length:]
    
    def _cleanup_history(self) -> None:
        """清理历史记录"""
        if self.auto_summarize and len(self.messages) > self.max_messages // 2:
            # 自动摘要旧的消息
            self._create_summary()
        
        # 保留最近的消息
        keep_count = self.max_messages // 2
        self.messages = self.messages[-keep_count:]
    
    def _create_summary(self) -> None:
        """创建对话摘要"""
        if self.summary_up_to_index >= len(self.messages) - 10:
            return  # 不需要重新摘要
        
        # 获取需要摘要的消息
        messages_to_summarize = self.messages[self.summary_up_to_index:-10]
        
        if not messages_to_summarize:
            return
        
        # 简单的摘要策略（实际应用中可以使用LLM）
        summary_parts = []
        for msg in messages_to_summarize:
            if msg.role == MessageRole.USER:
                summary_parts.append(f"用户询问：{msg.content[:50]}...")
            elif msg.role == MessageRole.ASSISTANT:
                summary_parts.append(f"助手回答：{msg.content[:50]}...")
        
        # 合并现有摘要
        if self.summary:
            self.summary = f"{self.summary}\n\n新增内容：\n" + "\n".join(summary_parts)
        else:
            self.summary = "\n".join(summary_parts)
        
        self.summary_up_to_index = len(self.messages) - 10
        
        logger.debug(f"Created conversation summary up to message {self.summary_up_to_index}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "max_messages": self.max_messages,
            "auto_summarize": self.auto_summarize,
            "summary": self.summary,
            "summary_up_to_index": self.summary_up_to_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """从字典创建实例"""
        history = cls(
            max_messages=data.get("max_messages", 100),
            auto_summarize=data.get("auto_summarize", True)
        )
        history.messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
        history.summary = data.get("summary")
        history.summary_up_to_index = data.get("summary_up_to_index", 0)
        return history


class ChatTemplate(TemplateBase[str, ChatResponse]):
    """
    对话模板类
    
    提供强大的多轮对话功能，支持角色设定、历史管理、动态参数替换等。
    
    核心功能：
    1. 角色设定：定义助手的人格、背景、专业领域
    2. 对话管理：管理多轮对话的状态和历史
    3. 参数替换：支持模板参数的动态替换
    4. 条件分支：根据上下文条件选择不同的响应策略
    5. 历史压缩：智能管理长对话的历史记录
    6. 格式控制：支持结构化和格式化的响应
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化对话模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config or self._create_default_config())
        
        # 模板设置
        self.role_template: str = ""             # 角色设定模板
        self.system_prompt_template: str = ""    # 系统提示词模板
        self.user_prompt_template: str = ""      # 用户提示词模板
        self.response_format: Dict[str, Any] = {}  # 响应格式
        
        # 对话管理
        self.conversations: Dict[str, Tuple[ConversationHistory, ConversationContext]] = {}
        self.default_context: ConversationContext = None
        
        # 参数替换器
        self.parameter_replacers: Dict[str, Callable[[str, Dict[str, Any]], str]] = {}
        self.condition_evaluators: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        
        # 集成的LLM模板
        self.llm_template = None
        
        logger.debug("ChatTemplate initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="ChatTemplate",
            version="1.0.0", 
            description="多轮对话模板，支持角色设定和历史管理",
            template_type=TemplateType.PROMPT,
            author="LangChain Learning Project",
            async_enabled=True
        )
        
        # 添加参数定义
        config.add_parameter("role_name", str, False, "智能助手", "角色名称")
        config.add_parameter("role_description", str, False, "", "角色描述")
        config.add_parameter("personality", str, False, "友好、专业、乐于助人", "角色性格")
        config.add_parameter("expertise", List[str], False, [], "专业领域")
        config.add_parameter("conversation_style", str, False, "自然对话", "对话风格")
        config.add_parameter("max_history_length", int, False, 20, "最大历史长度")
        config.add_parameter("enable_memory", bool, False, True, "是否启用记忆")
        config.add_parameter("response_format", str, False, "text", "响应格式")
        config.add_parameter("language", str, False, "zh-CN", "对话语言")
        config.add_parameter("context_variables", dict, False, {}, "上下文变量")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置对话模板参数
        
        Args:
            **parameters: 设置参数
            
        主要参数：
            role_name (str): 角色名称
            role_description (str): 角色详细描述
            personality (str): 角色性格特征
            expertise (List[str]): 专业领域列表
            conversation_style (str): 对话风格
            max_history_length (int): 最大历史记录长度
            enable_memory (bool): 是否启用对话记忆
            response_format (str): 响应格式（text/json/markdown）
            language (str): 对话语言
            llm_template: 集成的LLM模板实例
            system_prompt_template (str): 自定义系统提示词模板
            user_prompt_template (str): 自定义用户提示词模板
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置基本参数
            self.role_name = parameters.get("role_name", "智能助手")
            self.role_description = parameters.get("role_description", "")
            self.personality = parameters.get("personality", "友好、专业、乐于助人")
            self.expertise = parameters.get("expertise", [])
            self.conversation_style = parameters.get("conversation_style", "自然对话")
            self.max_history_length = parameters.get("max_history_length", 20)
            self.enable_memory = parameters.get("enable_memory", True)
            self.response_format = parameters.get("response_format", {})
            self.language = parameters.get("language", "zh-CN")
            
            # 设置LLM模板
            self.llm_template = parameters.get("llm_template")
            if not self.llm_template:
                logger.warning("No LLM template provided, will need to set one later")
            
            # 构建角色模板
            self._build_role_template()
            
            # 设置提示词模板
            self.system_prompt_template = parameters.get(
                "system_prompt_template",
                self._get_default_system_template()
            )
            self.user_prompt_template = parameters.get(
                "user_prompt_template", 
                self._get_default_user_template()
            )
            
            # 设置上下文变量
            context_variables = parameters.get("context_variables", {})
            self.default_context = ConversationContext(
                conversation_id="default",
                state=ConversationState.INITIALIZED,
                parameters=context_variables
            )
            
            # 注册默认的参数替换器
            self._register_default_replacers()
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            logger.info(f"ChatTemplate configured with role: {self.role_name}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup chat template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def execute(self, input_data: str, **kwargs) -> ChatResponse:
        """
        执行对话生成
        
        Args:
            input_data: 用户输入
            **kwargs: 额外参数
                - conversation_id: 对话ID
                - context_variables: 临时上下文变量
                - reset_conversation: 是否重置对话
                - include_history: 是否包含历史
                
        Returns:
            对话响应对象
        """
        try:
            start_time = time.time()
            
            # 获取或创建对话
            conversation_id = kwargs.get("conversation_id", "default")
            history, context = self._get_or_create_conversation(conversation_id)
            
            # 处理特殊命令
            if kwargs.get("reset_conversation", False):
                self._reset_conversation(conversation_id)
                history, context = self._get_or_create_conversation(conversation_id)
            
            # 更新上下文变量
            context_variables = kwargs.get("context_variables", {})
            for key, value in context_variables.items():
                context.set_parameter(key, value)
            
            # 添加用户消息到历史
            user_message = Message(
                role=MessageRole.USER,
                content=input_data,
                metadata={"conversation_id": conversation_id}
            )
            history.add_message(user_message)
            
            # 构建完整的提示词
            prompt = self._build_prompt(input_data, history, context, kwargs)
            
            # 调用LLM生成响应
            if not self.llm_template:
                raise ConfigurationError("No LLM template configured")
            
            llm_response = self.llm_template.execute(prompt, **kwargs)
            response_content = self._extract_content(llm_response)
            
            # 创建助手消息
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_content,
                metadata={
                    "conversation_id": conversation_id,
                    "processing_time": time.time() - start_time
                }
            )
            history.add_message(assistant_message)
            
            # 更新对话状态
            context.state = ConversationState.ACTIVE
            context.update_activity()
            
            # 生成建议操作
            suggested_actions = self._generate_suggested_actions(context, response_content)
            
            # 创建响应对象
            response = ChatResponse(
                message=assistant_message,
                context=context,
                suggested_actions=suggested_actions,
                processing_time=time.time() - start_time,
                metadata={
                    "conversation_id": conversation_id,
                    "message_count": len(history.messages),
                    "context_variables": context.parameters.copy()
                }
            )
            
            # 提取token计数（如果LLM响应提供）
            if hasattr(llm_response, 'total_tokens'):
                response.token_count = llm_response.total_tokens
            
            return response
            
        except Exception as e:
            logger.error(f"Chat execution failed: {str(e)}")
            # 返回错误响应
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=f"抱歉，处理您的请求时出现了错误：{str(e)}",
                metadata={"error": True}
            )
            return ChatResponse(
                message=error_message,
                context=self.default_context,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                metadata={"error": str(e)}
            )    
    def _get_or_create_conversation(
        self, 
        conversation_id: str
    ) -> Tuple[ConversationHistory, ConversationContext]:
        """
        获取或创建对话
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话历史和上下文的元组
        """
        if conversation_id not in self.conversations:
            history = ConversationHistory(
                max_messages=self.max_history_length * 2,
                auto_summarize=self.enable_memory
            )
            context = ConversationContext(
                conversation_id=conversation_id,
                state=ConversationState.INITIALIZED,
                parameters=self.default_context.parameters.copy() if self.default_context else {}
            )
            self.conversations[conversation_id] = (history, context)
        
        return self.conversations[conversation_id]
    
    def _reset_conversation(self, conversation_id: str) -> None:
        """重置指定对话"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        logger.info(f"Reset conversation: {conversation_id}")
    
    def _build_role_template(self) -> None:
        """构建角色模板"""
        role_parts = [f"你是{self.role_name}"]
        
        if self.role_description:
            role_parts.append(self.role_description)
        
        if self.personality:
            role_parts.append(f"你的性格特征：{self.personality}")
        
        if self.expertise:
            expertise_str = "、".join(self.expertise)
            role_parts.append(f"你的专业领域包括：{expertise_str}")
        
        if self.conversation_style:
            role_parts.append(f"你的对话风格：{self.conversation_style}")
        
        self.role_template = "。".join(role_parts) + "。"
    
    def _get_default_system_template(self) -> str:
        """获取默认系统提示词模板"""
        return """
{role_template}

对话规则：
1. 保持角色一致性，始终按照设定的角色特征进行对话
2. 根据用户的需求提供有用、准确的信息和建议
3. 如果不确定答案，诚实地说明并提供可能的解决方案
4. 保持对话的连贯性，参考之前的对话历史
5. 适当使用用户的语言风格，但保持专业性

{context_info}

请基于以上信息回答用户的问题。
""".strip()
    
    def _get_default_user_template(self) -> str:
        """获取默认用户提示词模板"""
        return "{user_input}"
    
    def _build_prompt(
        self, 
        user_input: str,
        history: ConversationHistory, 
        context: ConversationContext,
        kwargs: Dict[str, Any]
    ) -> str:
        """
        构建完整的提示词
        
        Args:
            user_input: 用户输入
            history: 对话历史
            context: 对话上下文
            kwargs: 额外参数
            
        Returns:
            构建的提示词
        """
        # 准备替换参数
        template_vars = {
            "role_template": self.role_template,
            "user_input": user_input,
            "role_name": self.role_name,
            "language": self.language,
            **context.parameters
        }
        
        # 构建上下文信息
        context_info_parts = []
        
        # 添加用户档案信息
        if context.user_profile:
            profile_info = ", ".join([f"{k}: {v}" for k, v in context.user_profile.items()])
            context_info_parts.append(f"用户信息：{profile_info}")
        
        # 添加对话历史（如果启用且有历史）
        include_history = kwargs.get("include_history", True)
        if include_history and history.messages:
            context_messages = history.get_context_messages(self.max_history_length)
            if context_messages:
                history_text = self._format_history(context_messages)
                context_info_parts.append(f"对话历史：\n{history_text}")
        
        # 添加会话数据
        if context.session_data:
            session_info = ", ".join([f"{k}: {v}" for k, v in context.session_data.items()])
            context_info_parts.append(f"会话信息：{session_info}")
        
        template_vars["context_info"] = "\n\n".join(context_info_parts) if context_info_parts else "这是一个新的对话。"
        
        # 应用参数替换
        system_prompt = self._apply_template_replacements(self.system_prompt_template, template_vars)
        user_prompt = self._apply_template_replacements(self.user_prompt_template, template_vars)
        
        # 组合最终提示词
        if system_prompt and user_prompt:
            return f"{system_prompt}\n\n用户：{user_prompt}"
        elif system_prompt:
            return system_prompt
        else:
            return user_prompt
    
    def _format_history(self, messages: List[Message]) -> str:
        """
        格式化对话历史
        
        Args:
            messages: 消息列表
            
        Returns:
            格式化的历史文本
        """
        formatted_lines = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                if msg.metadata.get("type") == "summary":
                    formatted_lines.append(f"[摘要] {msg.content}")
                else:
                    continue  # 跳过系统消息
            elif msg.role == MessageRole.USER:
                formatted_lines.append(f"用户：{msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                formatted_lines.append(f"{self.role_name}：{msg.content}")
        
        return "\n".join(formatted_lines)
    
    def _apply_template_replacements(self, template: str, variables: Dict[str, Any]) -> str:
        """
        应用模板参数替换
        
        Args:
            template: 模板字符串
            variables: 变量字典
            
        Returns:
            替换后的字符串
        """
        result = template
        
        # 基本变量替换（{variable}格式）
        for key, value in variables.items():
            pattern = "{" + key + "}"
            if pattern in result:
                result = result.replace(pattern, str(value))
        
        # 条件替换（{if condition}content{endif}格式）
        result = self._apply_conditional_replacements(result, variables)
        
        # 应用自定义替换器
        for name, replacer in self.parameter_replacers.items():
            if f"{{{name}:" in result:
                result = replacer(result, variables)
        
        return result
    
    def _apply_conditional_replacements(self, template: str, variables: Dict[str, Any]) -> str:
        """应用条件替换"""
        # 匹配 {if condition}content{endif} 格式
        pattern = r'\{if\s+([^}]+)\}(.*?)\{endif\}'
        
        def replace_condition(match):
            condition = match.group(1).strip()
            content = match.group(2)
            
            # 评估条件
            try:
                # 简单的条件评估（可以扩展）
                if condition in variables:
                    if variables[condition]:
                        return content
                elif "==" in condition:
                    key, value = condition.split("==", 1)
                    key, value = key.strip(), value.strip().strip('"\'')
                    if variables.get(key) == value:
                        return content
                elif "!=" in condition:
                    key, value = condition.split("!=", 1)
                    key, value = key.strip(), value.strip().strip('"\'')
                    if variables.get(key) != value:
                        return content
                
                return ""  # 条件不满足，返回空字符串
                
            except Exception as e:
                logger.warning(f"Failed to evaluate condition '{condition}': {str(e)}")
                return content  # 出错时返回原内容
        
        return re.sub(pattern, replace_condition, template, flags=re.DOTALL)
    
    def _extract_content(self, llm_response: Any) -> str:
        """
        从LLM响应中提取内容
        
        Args:
            llm_response: LLM响应对象
            
        Returns:
            提取的文本内容
        """
        if hasattr(llm_response, 'content'):
            return llm_response.content
        elif hasattr(llm_response, 'text'):
            return llm_response.text
        elif isinstance(llm_response, str):
            return llm_response
        else:
            return str(llm_response)
    
    def _generate_suggested_actions(self, context: ConversationContext, response: str) -> List[str]:
        """
        生成建议操作
        
        Args:
            context: 对话上下文
            response: 响应内容
            
        Returns:
            建议操作列表
        """
        suggestions = []
        
        # 基于专业领域的建议
        if self.expertise:
            if any(expertise in response.lower() for expertise in [e.lower() for e in self.expertise]):
                suggestions.append("深入了解相关专业话题")
        
        # 基于响应类型的建议
        if "?" in response:
            suggestions.append("回答相关问题")
        
        if "例子" in response or "示例" in response:
            suggestions.append("请求更多示例")
        
        if "步骤" in response or "方法" in response:
            suggestions.append("询问具体实施细节")
        
        # 通用建议
        suggestions.extend([
            "继续讨论当前话题",
            "切换到新话题", 
            "总结对话要点"
        ])
        
        return suggestions[:3]  # 限制建议数量
    
    def _register_default_replacers(self) -> None:
        """注册默认的参数替换器"""
        
        def time_replacer(template: str, variables: Dict[str, Any]) -> str:
            """时间相关的替换器"""
            now = datetime.now()
            replacements = {
                "{time:now}": now.strftime("%Y-%m-%d %H:%M:%S"),
                "{time:date}": now.strftime("%Y-%m-%d"),
                "{time:time}": now.strftime("%H:%M:%S"),
                "{time:hour}": str(now.hour),
                "{time:greeting}": self._get_time_greeting(now.hour)
            }
            
            for pattern, replacement in replacements.items():
                template = template.replace(pattern, replacement)
            
            return template
        
        def format_replacer(template: str, variables: Dict[str, Any]) -> str:
            """格式化替换器"""
            # 处理 {format:list:variable} 格式
            pattern = r'\{format:list:([^}]+)\}'
            
            def replace_list(match):
                var_name = match.group(1)
                if var_name in variables and isinstance(variables[var_name], list):
                    items = variables[var_name]
                    return "\n".join([f"- {item}" for item in items])
                return ""
            
            return re.sub(pattern, replace_list, template)
        
        self.parameter_replacers["time"] = time_replacer
        self.parameter_replacers["format"] = format_replacer
    
    def _get_time_greeting(self, hour: int) -> str:
        """根据时间获取问候语"""
        if 5 <= hour < 12:
            return "早上好"
        elif 12 <= hour < 18:
            return "下午好"
        else:
            return "晚上好"
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "多轮对话模板使用示例",
            "setup_parameters": {
                "role_name": "Python编程助手",
                "role_description": "专业的Python编程指导老师，擅长解答编程问题和提供代码示例",
                "personality": "耐心、专业、善于解释复杂概念",
                "expertise": ["Python编程", "算法设计", "代码优化", "调试技巧"],
                "conversation_style": "循序渐进的教学方式",
                "max_history_length": 10,
                "language": "zh-CN"
            },
            "execute_parameters": {
                "input_data": "我想学习Python的列表推导式，可以举个简单的例子吗？",
                "conversation_id": "python_learning_001",
                "context_variables": {
                    "user_level": "初学者",
                    "learning_goal": "掌握Python基础语法"
                }
            },
            "expected_output": {
                "type": "ChatResponse",
                "fields": {
                    "message": "包含Python列表推导式教学内容的回复",
                    "suggested_actions": ["请求更多示例", "询问具体实施细节", "继续讨论当前话题"],
                    "context": "更新的对话上下文"
                }
            },
            "advanced_usage": '''
# 基础对话
template = ChatTemplate()
template.setup(
    role_name="Python编程助手",
    role_description="专业的Python编程指导老师",
    personality="耐心、专业、善于解释",
    expertise=["Python编程", "算法设计"],
    llm_template=openai_template
)

response = template.run("什么是列表推导式？")
print(response.message.content)

# 多轮对话
response1 = template.run(
    "我想学习Python", 
    conversation_id="session_001"
)

response2 = template.run(
    "列表推导式怎么用？", 
    conversation_id="session_001"  # 同一对话
)

# 带上下文变量
response = template.run(
    "给我一些适合的练习题",
    conversation_id="session_001",
    context_variables={
        "user_level": "初学者",
        "current_topic": "列表推导式"
    }
)

# 自定义模板
custom_system_template = """
你是{role_name}。{role_template}

当前用户级别：{user_level}
学习目标：{learning_goal}

{if user_level == "初学者"}
请用简单易懂的语言解释，并提供基础示例。
{endif}

{if user_level == "高级"}
可以提供更复杂的示例和最佳实践。
{endif}

对话历史：
{context_info}
"""

template.setup(
    system_prompt_template=custom_system_template,
    # ... 其他参数
)
''',
            "conversation_management": '''
# 对话管理示例

# 获取对话历史
history, context = template._get_or_create_conversation("session_001")
messages = history.get_messages(limit=5)

# 重置对话
template._reset_conversation("session_001")

# 设置用户档案
context.user_profile = {
    "name": "张三",
    "experience": "初学者",
    "interests": ["Web开发", "数据分析"]
}

# 设置会话数据
context.session_data = {
    "start_time": datetime.now(),
    "topic": "Python基础",
    "progress": 0.3
}
'''
        }
    
    # 工具方法
    def add_custom_replacer(self, name: str, replacer: Callable[[str, Dict[str, Any]], str]) -> None:
        """
        添加自定义参数替换器
        
        Args:
            name: 替换器名称
            replacer: 替换器函数
        """
        self.parameter_replacers[name] = replacer
        logger.info(f"Added custom replacer: {name}")
    
    def set_llm_template(self, llm_template) -> None:
        """
        设置LLM模板
        
        Args:
            llm_template: LLM模板实例
        """
        self.llm_template = llm_template
        logger.info("LLM template updated")
    
    def get_conversation_info(self, conversation_id: str = "default") -> Dict[str, Any]:
        """
        获取对话信息
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话信息字典
        """
        if conversation_id not in self.conversations:
            return {"exists": False}
        
        history, context = self.conversations[conversation_id]
        
        return {
            "exists": True,
            "conversation_id": conversation_id,
            "state": context.state.value,
            "message_count": len(history.messages),
            "created_time": context.created_time.isoformat(),
            "last_activity": context.last_activity.isoformat(),
            "parameters": context.parameters.copy(),
            "user_profile": context.user_profile.copy(),
            "session_data": context.session_data.copy(),
            "has_summary": history.summary is not None
        }
    
    def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        导出对话数据
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            完整的对话数据
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        history, context = self.conversations[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "context": {
                "state": context.state.value,
                "parameters": context.parameters,
                "user_profile": context.user_profile,
                "session_data": context.session_data,
                "created_time": context.created_time.isoformat(),
                "last_activity": context.last_activity.isoformat()
            },
            "history": history.to_dict(),
            "template_config": {
                "role_name": self.role_name,
                "role_template": self.role_template,
                "expertise": self.expertise,
                "conversation_style": self.conversation_style
            }
        }
    
    def import_conversation(self, conversation_data: Dict[str, Any]) -> None:
        """
        导入对话数据
        
        Args:
            conversation_data: 对话数据
        """
        conversation_id = conversation_data["conversation_id"]
        
        # 重建上下文
        context_data = conversation_data["context"]
        context = ConversationContext(
            conversation_id=conversation_id,
            state=ConversationState(context_data["state"]),
            parameters=context_data["parameters"],
            user_profile=context_data["user_profile"],
            session_data=context_data["session_data"],
            created_time=datetime.fromisoformat(context_data["created_time"]),
            last_activity=datetime.fromisoformat(context_data["last_activity"])
        )
        
        # 重建历史
        history = ConversationHistory.from_dict(conversation_data["history"])
        
        # 存储对话
        self.conversations[conversation_id] = (history, context)
        
        logger.info(f"Imported conversation: {conversation_id}")
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        列出所有对话
        
        Returns:
            对话信息列表
        """
        result = []
        for conversation_id in self.conversations:
            info = self.get_conversation_info(conversation_id)
            result.append(info)
        return result
    
    def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """
        清理旧对话
        
        Args:
            max_age_hours: 最大存在时间（小时）
            
        Returns:
            清理的对话数量
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        
        for conversation_id, (history, context) in self.conversations.items():
            if context.last_activity < cutoff_time:
                to_remove.append(conversation_id)
        
        for conversation_id in to_remove:
            del self.conversations[conversation_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old conversations")
        return len(to_remove)