"""
对话Agent模板

专门用于多轮对话场景的Agent实现，支持上下文维护、情感识别、话题跟踪等功能。
"""

import asyncio
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentState, ToolDefinition


@dataclass 
class ConversationTurn:
    """对话轮次数据"""
    user_input: str
    agent_response: str
    emotion: str
    topic: str
    timestamp: float
    confidence: float = 0.0


class ConversationalAgent(BaseAgent):
    """
    对话Agent模板
    
    专门设计用于多轮对话交互，具备以下核心能力：
    1. 情感识别 - 分析用户情感状态，调整回复风格
    2. 话题跟踪 - 检测和维护对话话题，支持话题转换
    3. 上下文维护 - 保持长期对话上下文，提供连贯体验
    4. 个性化回复 - 根据用户特征和历史调整回复策略
    5. 澄清机制 - 主动寻求澄清，避免误解
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 对话历史
        self.conversation_history: Dict[str, List[ConversationTurn]] = {}
        
        # 当前话题
        self.current_topics: Dict[str, str] = {}
        
        # 用户情感状态
        self.user_emotions: Dict[str, str] = {}
        
        # 注册内置工具
        self._register_builtin_tools()
    
    @property
    def default_config(self) -> Dict[str, Any]:
        """默认配置"""
        base_config = super().default_config
        conv_config = {
            "response_style": "friendly",  # friendly, professional, casual
            "max_history_turns": 10,       # 最大保留对话轮次
            "enable_emotion_analysis": True,
            "enable_topic_tracking": True,
            "enable_clarification": True,
            "maintain_context": True,
            "context_window": 5,           # 上下文窗口大小
            "confidence_threshold": 0.7,   # 置信度阈值
        }
        return {**base_config, **conv_config}
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        tools = [
            ToolDefinition(
                name="analyze_emotion",
                description="分析用户输入的情感状态",
                func=self._analyze_emotion
            ),
            ToolDefinition(
                name="detect_topic",
                description="检测对话话题",
                func=self._detect_topic
            ),
            ToolDefinition(
                name="get_conversation_context",
                description="获取对话上下文",
                func=self._get_conversation_context
            ),
            ToolDefinition(
                name="generate_clarification",
                description="生成澄清问题",
                func=self._generate_clarification
            )
        ]
        
        self.register_tools(tools)
    
    async def think(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考过程：分析输入、识别情感、检测话题、制定响应策略
        
        Args:
            input_data: 用户输入
            context: 执行上下文
            
        Returns:
            决策结果包含：情感、话题、响应策略、是否需要澄清等
        """
        session_id = context["session_id"]
        
        # 1. 情感分析
        emotion = "neutral"
        if self.config.get("enable_emotion_analysis", True):
            emotion = await self.call_tool("analyze_emotion", text=input_data)
        
        # 2. 话题检测
        topic = "general"
        if self.config.get("enable_topic_tracking", True):
            topic = await self.call_tool("detect_topic", text=input_data)
        
        # 3. 获取对话上下文
        conv_context = []
        if self.config.get("maintain_context", True):
            conv_context = await self.call_tool(
                "get_conversation_context", 
                session_id=session_id
            )
        
        # 4. 判断是否需要澄清
        needs_clarification = False
        clarification_question = ""
        if self.config.get("enable_clarification", True):
            needs_clarification, clarification_question = await self._check_need_clarification(
                input_data, conv_context
            )
        
        # 5. 制定响应策略
        response_strategy = self._determine_response_strategy(
            emotion, topic, conv_context, needs_clarification
        )
        
        decision = {
            "emotion": emotion,
            "topic": topic,
            "conversation_context": conv_context,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "response_strategy": response_strategy,
            "confidence": self._calculate_confidence(emotion, topic, conv_context)
        }
        
        # 更新状态
        self.user_emotions[session_id] = emotion
        self.current_topics[session_id] = topic
        
        self.logger.debug(f"思考完成 - 情感: {emotion}, 话题: {topic}, 策略: {response_strategy}")
        
        return decision
    
    async def act(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行动作：基于决策生成适当的回复内容
        
        Args:
            decision: 思考阶段的决策结果
            context: 执行上下文
            
        Returns:
            动作结果包含生成的回复内容
        """
        session_id = context["session_id"]
        input_data = context["input"]
        
        # 如果需要澄清，优先返回澄清问题
        if decision["needs_clarification"]:
            response_content = decision["clarification_question"]
        else:
            # 根据响应策略生成回复
            response_content = await self._generate_response(
                input_data=input_data,
                emotion=decision["emotion"],
                topic=decision["topic"],
                strategy=decision["response_strategy"],
                context=decision["conversation_context"]
            )
        
        # 记录对话轮次
        await self._record_conversation_turn(
            session_id=session_id,
            user_input=input_data,
            agent_response=response_content,
            emotion=decision["emotion"],
            topic=decision["topic"],
            confidence=decision["confidence"]
        )
        
        return {
            "response_content": response_content,
            "emotion": decision["emotion"],
            "topic": decision["topic"],
            "confidence": decision["confidence"]
        }
    
    async def respond(self, action_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        生成最终回复：格式化并返回回复文本
        
        Args:
            action_result: 动作执行结果
            context: 执行上下文
            
        Returns:
            最终的回复文本
        """
        return action_result["response_content"]
    
    def _analyze_emotion(self, text: str) -> str:
        """
        分析文本情感
        
        Args:
            text: 待分析文本
            
        Returns:
            情感标签 (happy, sad, angry, surprised, neutral)
        """
        # 简单的关键词情感分析
        emotion_keywords = {
            "happy": ["开心", "高兴", "快乐", "兴奋", "满意", "喜欢", "爱", "棒", "好", "哈哈"],
            "sad": ["难过", "伤心", "失望", "沮丧", "痛苦", "不开心", "郁闷", "唉"],
            "angry": ["生气", "愤怒", "不满", "烦躁", "讨厌", "气死", "火大", "恼火"],
            "surprised": ["惊讶", "震惊", "意外", "没想到", "天哪", "哇", "真的吗"],
            "worried": ["担心", "焦虑", "紧张", "害怕", "忧虑", "不安", "怎么办"]
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        return "neutral"
    
    def _detect_topic(self, text: str) -> str:
        """
        检测对话话题
        
        Args:
            text: 输入文本
            
        Returns:
            话题标签
        """
        # 话题关键词映射
        topic_keywords = {
            "technology": ["技术", "编程", "代码", "开发", "软件", "计算机", "AI", "人工智能"],
            "work": ["工作", "职业", "项目", "任务", "同事", "老板", "公司", "职场"],
            "study": ["学习", "课程", "考试", "书", "知识", "教育", "学校", "专业"],
            "life": ["生活", "家庭", "朋友", "健康", "运动", "旅行", "兴趣", "爱好"],
            "emotion": ["感情", "心情", "情绪", "关系", "友谊", "爱情", "家人"],
            "help": ["帮助", "问题", "困难", "建议", "指导", "解决", "怎么办"]
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        
        return "general"
    
    def _get_conversation_context(self, session_id: str) -> List[ConversationTurn]:
        """
        获取对话上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            最近的对话轮次列表
        """
        if session_id not in self.conversation_history:
            return []
        
        # 获取最近的对话轮次
        context_window = self.config.get("context_window", 5)
        return self.conversation_history[session_id][-context_window:]
    
    async def _check_need_clarification(self, input_data: str, context: List[ConversationTurn]) -> tuple[bool, str]:
        """
        检查是否需要澄清
        
        Args:
            input_data: 用户输入
            context: 对话上下文
            
        Returns:
            (是否需要澄清, 澄清问题)
        """
        # 简单的澄清检测逻辑
        unclear_patterns = [
            r"这个|那个|它|他们",  # 指代不明
            r"^.{1,3}$",          # 过短输入
            r"[？？]{2,}",        # 多个问号
            r"不知道|不清楚|不确定"   # 不确定表达
        ]
        
        for pattern in unclear_patterns:
            if re.search(pattern, input_data):
                clarification = await self.call_tool("generate_clarification", input_data=input_data)
                return True, clarification
        
        return False, ""
    
    def _generate_clarification(self, input_data: str) -> str:
        """
        生成澄清问题
        
        Args:
            input_data: 用户输入
            
        Returns:
            澄清问题
        """
        clarification_templates = [
            "您能更具体地描述一下吗？",
            "我没太理解您的意思，能再解释一下吗？",
            "您是想问关于什么方面的问题呢？",
            "能给我更多的背景信息吗？",
            "您希望我帮您做什么呢？"
        ]
        
        # 简单地返回一个合适的澄清问题
        if len(input_data) <= 3:
            return "您能说得更详细一些吗？"
        elif "这个" in input_data or "那个" in input_data:
            return "您说的是哪个具体的东西呢？"
        else:
            return clarification_templates[0]
    
    def _determine_response_strategy(self, emotion: str, topic: str, context: List[ConversationTurn], needs_clarification: bool) -> str:
        """
        确定响应策略
        
        Args:
            emotion: 情感状态
            topic: 话题
            context: 对话上下文
            needs_clarification: 是否需要澄清
            
        Returns:
            响应策略
        """
        if needs_clarification:
            return "clarification"
        
        # 根据情感和话题确定策略
        if emotion == "sad" or emotion == "worried":
            return "empathetic"  # 同理心回复
        elif emotion == "angry":
            return "calming"     # 安抚回复
        elif emotion == "happy":
            return "enthusiastic" # 热情回复
        elif topic == "help":
            return "helpful"     # 帮助回复
        else:
            return "conversational"  # 常规对话
    
    async def _generate_response(self, input_data: str, emotion: str, topic: str, strategy: str, context: List[ConversationTurn]) -> str:
        """
        生成响应内容
        
        Args:
            input_data: 用户输入
            emotion: 情感状态
            topic: 话题
            strategy: 响应策略
            context: 对话上下文
            
        Returns:
            生成的回复内容
        """
        # 根据策略生成不同风格的回复
        style_templates = {
            "empathetic": "我理解您的感受。{content}",
            "calming": "请不要着急，我们来看看能怎么解决这个问题。{content}",
            "enthusiastic": "太好了！{content}",
            "helpful": "我很乐意帮助您。{content}",
            "conversational": "{content}"
        }
        
        # 基础回复生成（这里可以集成LLM模板）
        base_response = f"关于{topic}的问题，我建议您可以..."  # 简化的回复生成
        
        # 应用风格模板
        template = style_templates.get(strategy, style_templates["conversational"])
        response = template.format(content=base_response)
        
        return response
    
    async def _record_conversation_turn(self, session_id: str, user_input: str, agent_response: str, emotion: str, topic: str, confidence: float):
        """
        记录对话轮次
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            agent_response: Agent回复
            emotion: 情感
            topic: 话题
            confidence: 置信度
        """
        import time
        
        turn = ConversationTurn(
            user_input=user_input,
            agent_response=agent_response,
            emotion=emotion,
            topic=topic,
            timestamp=time.time(),
            confidence=confidence
        )
        
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append(turn)
        
        # 维护历史长度
        max_turns = self.config.get("max_history_turns", 10)
        if len(self.conversation_history[session_id]) > max_turns:
            self.conversation_history[session_id] = self.conversation_history[session_id][-max_turns:]
    
    def _calculate_confidence(self, emotion: str, topic: str, context: List[ConversationTurn]) -> float:
        """
        计算置信度
        
        Args:
            emotion: 情感
            topic: 话题
            context: 上下文
            
        Returns:
            置信度分数 (0.0-1.0)
        """
        base_confidence = 0.7
        
        # 根据情感识别的确定性调整
        if emotion != "neutral":
            base_confidence += 0.1
        
        # 根据话题识别的确定性调整
        if topic != "general":
            base_confidence += 0.1
        
        # 根据上下文长度调整
        if len(context) > 0:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        获取对话摘要
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话摘要信息
        """
        if session_id not in self.conversation_history:
            return {"total_turns": 0}
        
        history = self.conversation_history[session_id]
        emotions = [turn.emotion for turn in history]
        topics = [turn.topic for turn in history]
        
        return {
            "total_turns": len(history),
            "dominant_emotion": max(set(emotions), key=emotions.count) if emotions else "neutral",
            "main_topics": list(set(topics)),
            "avg_confidence": sum(turn.confidence for turn in history) / len(history) if history else 0.0,
            "recent_turns": len([turn for turn in history[-5:]]) if len(history) >= 5 else len(history)
        }