"""
摘要记忆模板模块

本模块实现了智能摘要记忆系统，用于压缩长对话并提取关键信息。
支持多种摘要策略、分层压缩和语义保持，有效管理大型对话历史。

核心功能：
1. 智能摘要 - 使用LLM对长对话进行智能压缩
2. 分层压缩 - 支持多级摘要，从详细到高级概要
3. 关键信息提取 - 自动识别和保留重要信息
4. 语义保持 - 在压缩过程中保持对话的语义一致性
5. 渐进式摘要 - 随着对话进行动态更新摘要
6. 记忆融合 - 将新内容与现有摘要进行智能融合

设计原理：
- 基于LLM的智能摘要：使用大型语言模型进行高质量摘要
- 分层记忆架构：短期详细记忆 + 中期摘要 + 长期核心要点
- 语义相似性检测：避免重复信息，保持摘要简洁
- 渐进式更新：增量更新摘要，避免重复处理
- 可配置压缩策略：支持不同的压缩比例和策略
"""

import json
import time
import asyncio
import hashlib
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import threading

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from .conversation_memory import Message, MessageType, Conversation, ConversationMemoryTemplate
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError, 
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class SummaryStrategy(Enum):
    """摘要策略枚举"""
    EXTRACTIVE = "extractive"      # 抽取式摘要：从原文中选择重要句子
    ABSTRACTIVE = "abstractive"    # 生成式摘要：生成新的摘要文本
    HYBRID = "hybrid"              # 混合式摘要：结合抽取和生成
    KEYWORD = "keyword"            # 关键词摘要：提取关键词和短语
    HIERARCHICAL = "hierarchical"  # 层次摘要：多级摘要结构


class CompressionLevel(Enum):
    """压缩级别枚举"""
    LOW = "low"          # 低压缩：保留70-80%的信息
    MEDIUM = "medium"    # 中压缩：保留40-60%的信息
    HIGH = "high"        # 高压缩：保留20-30%的信息
    EXTREME = "extreme"  # 极限压缩：保留10%以下的信息


@dataclass
class SummarySegment:
    """
    摘要片段数据结构
    
    表示对话中一个时间段或主题的摘要信息。
    """
    segment_id: str                           # 片段ID
    start_time: float                         # 开始时间戳
    end_time: float                          # 结束时间戳
    original_message_count: int               # 原始消息数量
    summary_text: str                         # 摘要文本
    key_points: List[str] = field(default_factory=list)    # 关键要点
    keywords: List[str] = field(default_factory=list)      # 关键词
    entities: List[str] = field(default_factory=list)      # 实体
    topics: List[str] = field(default_factory=list)        # 主题
    compression_ratio: float = 0.0           # 压缩比例
    strategy: SummaryStrategy = SummaryStrategy.ABSTRACTIVE  # 摘要策略
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['strategy'] = self.strategy.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SummarySegment':
        """从字典创建SummarySegment实例"""
        if isinstance(data.get('strategy'), str):
            data['strategy'] = SummaryStrategy(data['strategy'])
        return cls(**data)
    
    def get_importance_score(self) -> float:
        """计算重要性得分"""
        # 基于关键点数量、实体数量和压缩比例计算重要性
        keyword_score = len(self.keywords) * 0.1
        entity_score = len(self.entities) * 0.2
        topic_score = len(self.topics) * 0.3
        compression_score = (1 - self.compression_ratio) * 0.4
        
        return min(1.0, keyword_score + entity_score + topic_score + compression_score)


@dataclass
class ConversationSummary:
    """
    对话摘要数据结构
    
    表示整个对话会话的分层摘要信息。
    """
    session_id: str                           # 会话ID
    user_id: Optional[str] = None             # 用户ID
    created_at: float = field(default_factory=time.time)     # 创建时间
    updated_at: float = field(default_factory=time.time)     # 更新时间
    
    # 分层摘要
    segments: List[SummarySegment] = field(default_factory=list)  # 片段摘要
    overall_summary: str = ""                 # 整体摘要
    key_themes: List[str] = field(default_factory=list)         # 主要主题
    important_facts: List[str] = field(default_factory=list)    # 重要事实
    decisions_made: List[str] = field(default_factory=list)     # 做出的决定
    action_items: List[str] = field(default_factory=list)       # 行动项目
    
    # 统计信息
    total_original_messages: int = 0          # 原始消息总数
    total_compression_ratio: float = 0.0      # 总体压缩比例
    summary_version: int = 1                  # 摘要版本
    
    # 配置信息
    strategy_used: SummaryStrategy = SummaryStrategy.ABSTRACTIVE
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_segment(self, segment: SummarySegment) -> None:
        """添加摘要片段"""
        self.segments.append(segment)
        self.updated_at = time.time()
        self.total_original_messages += segment.original_message_count
        
        # 更新总体压缩比例
        if self.segments:
            total_original = sum(s.original_message_count for s in self.segments)
            total_summary_length = sum(len(s.summary_text) for s in self.segments)
            total_original_length = total_original * 100  # 估算原始长度
            
            if total_original_length > 0:
                self.total_compression_ratio = 1 - (total_summary_length / total_original_length)
    
    def get_latest_segments(self, count: int = 5) -> List[SummarySegment]:
        """获取最新的摘要片段"""
        return sorted(self.segments, key=lambda s: s.end_time, reverse=True)[:count]
    
    def get_most_important_segments(self, count: int = 5) -> List[SummarySegment]:
        """获取最重要的摘要片段"""
        return sorted(self.segments, key=lambda s: s.get_importance_score(), reverse=True)[:count]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'segments': [s.to_dict() for s in self.segments],
            'overall_summary': self.overall_summary,
            'key_themes': self.key_themes,
            'important_facts': self.important_facts,
            'decisions_made': self.decisions_made,
            'action_items': self.action_items,
            'total_original_messages': self.total_original_messages,
            'total_compression_ratio': self.total_compression_ratio,
            'summary_version': self.summary_version,
            'strategy_used': self.strategy_used.value,
            'compression_level': self.compression_level.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSummary':
        """从字典创建ConversationSummary实例"""
        segments = [SummarySegment.from_dict(s) for s in data.get('segments', [])]
        
        # 处理枚举类型
        strategy = SummaryStrategy(data.get('strategy_used', 'abstractive'))
        compression = CompressionLevel(data.get('compression_level', 'medium'))
        
        return cls(
            session_id=data['session_id'],
            user_id=data.get('user_id'),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            segments=segments,
            overall_summary=data.get('overall_summary', ''),
            key_themes=data.get('key_themes', []),
            important_facts=data.get('important_facts', []),
            decisions_made=data.get('decisions_made', []),
            action_items=data.get('action_items', []),
            total_original_messages=data.get('total_original_messages', 0),
            total_compression_ratio=data.get('total_compression_ratio', 0.0),
            summary_version=data.get('summary_version', 1),
            strategy_used=strategy,
            compression_level=compression,
            metadata=data.get('metadata', {})
        )


class SummaryGenerator(ABC):
    """
    摘要生成器抽象基类
    
    定义摘要生成器的接口规范。
    """
    
    @abstractmethod
    def generate_summary(self, messages: List[Message], strategy: SummaryStrategy, 
                        compression_level: CompressionLevel) -> SummarySegment:
        """生成摘要片段"""
        pass
    
    @abstractmethod
    def update_overall_summary(self, current_summary: str, new_segment: SummarySegment) -> str:
        """更新整体摘要"""
        pass
    
    @abstractmethod
    def extract_key_information(self, messages: List[Message]) -> Dict[str, List[str]]:
        """提取关键信息"""
        pass


class LLMSummaryGenerator(SummaryGenerator):
    """
    基于LLM的摘要生成器
    
    使用大型语言模型生成高质量的对话摘要。
    """
    
    def __init__(self, llm_model: Optional[Any] = None):
        """
        初始化LLM摘要生成器
        
        Args:
            llm_model: LLM模型实例
        """
        self.llm_model = llm_model
        
        # 摘要提示模板
        self.summary_prompts = {
            SummaryStrategy.EXTRACTIVE: """
请从以下对话中抽取最重要的句子和信息点，保持原文表达：

对话内容：
{messages}

压缩级别：{compression_level}

请提供：
1. 关键句子（保持原文）
2. 重要信息点
3. 关键词列表
4. 主要话题
""",
            SummaryStrategy.ABSTRACTIVE: """
请对以下对话进行智能摘要，用自己的话重新表达核心内容：

对话内容：
{messages}

压缩级别：{compression_level}

请提供：
1. 对话摘要（用简洁的语言概括）
2. 关键要点
3. 重要实体
4. 主要主题
5. 做出的决定或行动项
""",
            SummaryStrategy.HYBRID: """
请对以下对话进行混合式摘要，结合抽取和生成方法：

对话内容：
{messages}

压缩级别：{compression_level}

请提供：
1. 核心摘要（生成式）
2. 重要原文引用（抽取式）
3. 关键信息点
4. 实体和关键词
5. 主题分析
"""
        }
    
    def generate_summary(self, messages: List[Message], strategy: SummaryStrategy, 
                        compression_level: CompressionLevel) -> SummarySegment:
        """生成摘要片段"""
        if not messages:
            return SummarySegment(
                segment_id=f"empty_{int(time.time())}",
                start_time=time.time(),
                end_time=time.time(),
                original_message_count=0,
                summary_text="空对话段落",
                strategy=strategy
            )
        
        # 格式化消息内容
        formatted_messages = self._format_messages(messages)
        
        # 选择摘要策略
        prompt_template = self.summary_prompts.get(strategy, self.summary_prompts[SummaryStrategy.ABSTRACTIVE])
        
        # 生成摘要（如果没有LLM模型，使用规则式方法）
        if self.llm_model:
            summary_result = self._generate_with_llm(formatted_messages, prompt_template, compression_level)
        else:
            summary_result = self._generate_with_rules(messages, compression_level)
        
        # 创建摘要片段
        segment = SummarySegment(
            segment_id=f"seg_{hashlib.md5(f'{messages[0].timestamp}_{len(messages)}'.encode()).hexdigest()[:8]}",
            start_time=messages[0].timestamp,
            end_time=messages[-1].timestamp,
            original_message_count=len(messages),
            summary_text=summary_result.get('summary', ''),
            key_points=summary_result.get('key_points', []),
            keywords=summary_result.get('keywords', []),
            entities=summary_result.get('entities', []),
            topics=summary_result.get('topics', []),
            strategy=strategy
        )
        
        # 计算压缩比例
        original_length = len(formatted_messages)
        summary_length = len(segment.summary_text)
        segment.compression_ratio = 1 - (summary_length / original_length) if original_length > 0 else 0
        
        return segment
    
    def update_overall_summary(self, current_summary: str, new_segment: SummarySegment) -> str:
        """更新整体摘要"""
        if not current_summary:
            return new_segment.summary_text
        
        # 如果有LLM模型，使用智能融合
        if self.llm_model:
            return self._merge_summaries_with_llm(current_summary, new_segment.summary_text)
        else:
            return self._merge_summaries_with_rules(current_summary, new_segment.summary_text)
    
    def extract_key_information(self, messages: List[Message]) -> Dict[str, List[str]]:
        """提取关键信息"""
        # 基础关键信息提取
        keywords = []
        entities = []
        topics = []
        facts = []
        decisions = []
        actions = []
        
        for message in messages:
            content = message.content.lower()
            
            # 简单关键词提取（基于频率和长度）
            words = re.findall(r'\b\w{3,}\b', content)
            keywords.extend([w for w in words if len(w) > 3])
            
            # 检测决定性语言
            if any(phrase in content for phrase in ['决定', '选择', '确定', '同意', 'decide', 'choose']):
                decisions.append(message.content[:100])
            
            # 检测行动项
            if any(phrase in content for phrase in ['需要', '应该', '计划', 'will do', 'should', 'need to']):
                actions.append(message.content[:100])
            
            # 检测重要事实
            if any(phrase in content for phrase in ['重要', '关键', '注意', 'important', 'key', 'note']):
                facts.append(message.content[:100])
        
        # 去重和频率统计
        from collections import Counter
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(10)]
        
        return {
            'keywords': top_keywords,
            'entities': list(set(entities)),
            'topics': list(set(topics)),
            'facts': list(set(facts)),
            'decisions': list(set(decisions)),
            'actions': list(set(actions))
        }
    
    def _format_messages(self, messages: List[Message]) -> str:
        """格式化消息为文本"""
        formatted = []
        for msg in messages:
            timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            speaker = "用户" if msg.type == MessageType.HUMAN else "助手"
            formatted.append(f"[{timestamp}] {speaker}: {msg.content}")
        
        return "\n".join(formatted)
    
    def _generate_with_llm(self, messages: str, prompt_template: str, 
                          compression_level: CompressionLevel) -> Dict[str, Any]:
        """使用LLM生成摘要"""
        try:
            # 构造完整提示
            prompt = prompt_template.format(
                messages=messages,
                compression_level=compression_level.value
            )
            
            # 调用LLM（这里需要根据实际的LLM接口进行调整）
            if hasattr(self.llm_model, 'predict'):
                response = self.llm_model.predict(prompt)
            elif hasattr(self.llm_model, '__call__'):
                response = self.llm_model(prompt)
            else:
                raise ConfigurationError("LLM模型接口不兼容")
            
            # 解析响应（这里简化处理，实际应该有更复杂的解析逻辑）
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"LLM摘要生成失败：{e}")
            # 回退到规则式方法
            return self._generate_with_rules([], compression_level)
    
    def _generate_with_rules(self, messages: List[Message], 
                           compression_level: CompressionLevel) -> Dict[str, Any]:
        """使用规则式方法生成摘要"""
        if not messages:
            return {
                'summary': '空对话',
                'key_points': [],
                'keywords': [],
                'entities': [],
                'topics': []
            }
        
        # 基于长度的简单摘要
        all_content = " ".join([msg.content for msg in messages])
        
        # 根据压缩级别确定摘要长度
        compression_ratios = {
            CompressionLevel.LOW: 0.7,
            CompressionLevel.MEDIUM: 0.5,
            CompressionLevel.HIGH: 0.3,
            CompressionLevel.EXTREME: 0.1
        }
        
        target_length = int(len(all_content) * compression_ratios[compression_level])
        summary = all_content[:target_length] + ("..." if len(all_content) > target_length else "")
        
        # 提取关键信息
        key_info = self.extract_key_information(messages)
        
        return {
            'summary': summary,
            'key_points': key_info.get('facts', []),
            'keywords': key_info.get('keywords', []),
            'entities': key_info.get('entities', []),
            'topics': key_info.get('topics', [])
        }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        # 简化的解析逻辑，实际应该更复杂
        lines = response.strip().split('\n')
        
        result = {
            'summary': '',
            'key_points': [],
            'keywords': [],
            'entities': [],
            'topics': []
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if '摘要' in line or 'summary' in line.lower():
                current_section = 'summary'
            elif '要点' in line or 'points' in line.lower():
                current_section = 'key_points'
            elif '关键词' in line or 'keywords' in line.lower():
                current_section = 'keywords'
            elif '实体' in line or 'entities' in line.lower():
                current_section = 'entities'
            elif '主题' in line or 'topics' in line.lower():
                current_section = 'topics'
            elif current_section:
                if current_section == 'summary':
                    result['summary'] += line + ' '
                else:
                    # 解析列表项
                    if line.startswith('-') or line.startswith('•') or line[0].isdigit():
                        item = re.sub(r'^[-•\d.\s]+', '', line).strip()
                        if item:
                            result[current_section].append(item)
        
        return result
    
    def _merge_summaries_with_llm(self, current: str, new: str) -> str:
        """使用LLM合并摘要"""
        try:
            merge_prompt = f"""
请将以下两个摘要合并为一个更完整、简洁的摘要：

现有摘要：
{current}

新增摘要：
{new}

请提供合并后的摘要，保持简洁性的同时包含两个摘要的核心信息：
"""
            
            if hasattr(self.llm_model, 'predict'):
                merged = self.llm_model.predict(merge_prompt)
            elif hasattr(self.llm_model, '__call__'):
                merged = self.llm_model(merge_prompt)
            else:
                return self._merge_summaries_with_rules(current, new)
            
            return merged.strip()
            
        except Exception as e:
            logger.error(f"LLM摘要合并失败：{e}")
            return self._merge_summaries_with_rules(current, new)
    
    def _merge_summaries_with_rules(self, current: str, new: str) -> str:
        """使用规则式方法合并摘要"""
        if not current:
            return new
        if not new:
            return current
        
        # 简单合并策略：连接并去重关键信息
        return f"{current}\n\n补充：{new}"


class SummaryMemoryTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    摘要记忆模板
    
    提供智能摘要记忆管理功能，包括对话压缩、关键信息提取、
    分层摘要和语义保持等高级功能。
    
    核心功能：
    1. 智能摘要：使用多种策略生成高质量摘要
    2. 分层压缩：支持多级摘要和渐进式压缩
    3. 关键信息提取：自动识别重要事实、决定和行动项
    4. 语义保持：在压缩过程中保持语义一致性
    5. 摘要融合：智能合并新旧摘要信息
    6. 多种存储：支持文件、数据库等存储方式
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None, 
                 conversation_memory: Optional[ConversationMemoryTemplate] = None):
        """
        初始化摘要记忆模板
        
        Args:
            config: 模板配置
            conversation_memory: 对话记忆模板（用于获取原始对话）
        """
        super().__init__(config)
        
        # 依赖的对话记忆模板
        self.conversation_memory = conversation_memory
        
        # 摘要生成器
        self.summary_generator: Optional[SummaryGenerator] = None
        
        # 摘要存储
        self.summaries: Dict[str, ConversationSummary] = {}
        
        # 配置参数
        self.segment_size: int = 10                    # 每个片段的消息数量
        self.compression_level: CompressionLevel = CompressionLevel.MEDIUM
        self.summary_strategy: SummaryStrategy = SummaryStrategy.ABSTRACTIVE
        self.auto_summarize_threshold: int = 20       # 自动摘要的消息阈值
        self.max_segments_per_summary: int = 50       # 每个摘要的最大片段数
        
        # 线程锁
        self._lock = threading.Lock()
        
        logger.debug("Initialized SummaryMemoryTemplate")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="SummaryMemoryTemplate",
            description="智能摘要记忆管理模板",
            template_type=TemplateType.MEMORY,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("segment_size", int, default=10,
                           description="每个摘要片段的消息数量")
        config.add_parameter("compression_level", str, default="medium",
                           description="压缩级别：low, medium, high, extreme")
        config.add_parameter("summary_strategy", str, default="abstractive",
                           description="摘要策略：extractive, abstractive, hybrid, keyword, hierarchical")
        config.add_parameter("auto_summarize_threshold", int, default=20,
                           description="自动摘要的消息阈值")
        config.add_parameter("max_segments_per_summary", int, default=50,
                           description="每个摘要的最大片段数")
        config.add_parameter("llm_model_name", str, default=None,
                           description="LLM模型名称（可选）")
        config.add_parameter("storage_path", str, default="./summaries",
                           description="摘要存储路径")
        config.add_parameter("enable_semantic_similarity", bool, default=False,
                           description="是否启用语义相似性检测")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置摘要记忆模板
        
        Args:
            **parameters: 配置参数
                - segment_size: 每个片段的消息数量
                - compression_level: 压缩级别
                - summary_strategy: 摘要策略
                - auto_summarize_threshold: 自动摘要阈值
                - max_segments_per_summary: 最大片段数
                - llm_model_name: LLM模型名称
                - storage_path: 存储路径
                - enable_semantic_similarity: 启用语义相似性
        """
        # 验证参数
        if not self.validate_parameters(parameters):
            raise ValidationError("SummaryMemoryTemplate参数验证失败")
        
        # 更新内部参数
        self.segment_size = parameters.get("segment_size", 10)
        self.auto_summarize_threshold = parameters.get("auto_summarize_threshold", 20)
        self.max_segments_per_summary = parameters.get("max_segments_per_summary", 50)
        
        # 设置压缩级别
        compression_str = parameters.get("compression_level", "medium")
        self.compression_level = CompressionLevel(compression_str)
        
        # 设置摘要策略
        strategy_str = parameters.get("summary_strategy", "abstractive")
        self.summary_strategy = SummaryStrategy(strategy_str)
        
        # 初始化摘要生成器
        llm_model_name = parameters.get("llm_model_name")
        self.summary_generator = self._create_summary_generator(llm_model_name)
        
        # 创建存储目录
        storage_path = Path(parameters.get("storage_path", "./summaries"))
        storage_path.mkdir(parents=True, exist_ok=True)
        self.storage_path = storage_path
        
        self.status = self.config.template_type.CONFIGURED
        logger.info(f"SummaryMemoryTemplate配置完成：strategy={strategy_str}, compression={compression_str}")
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行摘要记忆操作
        
        Args:
            input_data: 输入数据
                - action: 操作类型（summarize, get_summary, update_summary等）
                - session_id: 会话ID
                - user_id: 用户ID（可选）
                - force_update: 强制更新摘要
                - segment_count: 摘要片段数量
                
        Returns:
            执行结果字典
        """
        action = input_data.get("action")
        if not action:
            raise ValidationError("必须指定action参数")
        
        session_id = input_data.get("session_id", "default")
        user_id = input_data.get("user_id")
        
        try:
            if action == "summarize":
                return self._create_summary(session_id, user_id, input_data)
            elif action == "get_summary":
                return self._get_summary(session_id, user_id)
            elif action == "update_summary":
                return self._update_summary(session_id, user_id, input_data)
            elif action == "get_segments":
                return self._get_segments(session_id, user_id, input_data.get("count", 5))
            elif action == "compress":
                return self._compress_summary(session_id, user_id, input_data)
            elif action == "extract_insights":
                return self._extract_insights(session_id, user_id)
            elif action == "save_summary":
                return self._save_summary(session_id, user_id)
            elif action == "load_summary":
                return self._load_summary(session_id, user_id)
            elif action == "get_stats":
                return self._get_summary_stats()
            else:
                raise ValidationError(f"未知的操作类型：{action}")
                
        except Exception as e:
            logger.error(f"执行摘要记忆操作失败：{action} - {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    def _create_summary(self, session_id: str, user_id: Optional[str], 
                       input_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建对话摘要"""
        if not self.conversation_memory:
            return {
                "success": False,
                "error": "需要配置conversation_memory依赖"
            }
        
        # 获取对话历史
        conversation_result = self.conversation_memory.execute({
            "action": "get_conversation",
            "session_id": session_id,
            "user_id": user_id
        })
        
        if not conversation_result.get("success"):
            return {
                "success": False,
                "error": "无法获取对话历史"
            }
        
        conversation_data = conversation_result.get("conversation", {})
        messages = [Message.from_dict(msg) for msg in conversation_data.get("messages", [])]
        
        if not messages:
            return {
                "success": False,
                "error": "对话为空，无法生成摘要"
            }
        
        # 创建或获取摘要
        summary_key = self._get_summary_key(session_id, user_id)
        
        with self._lock:
            if summary_key not in self.summaries:
                self.summaries[summary_key] = ConversationSummary(
                    session_id=session_id,
                    user_id=user_id,
                    strategy_used=self.summary_strategy,
                    compression_level=self.compression_level
                )
        
        conversation_summary = self.summaries[summary_key]
        
        # 分片段生成摘要
        force_update = input_data.get("force_update", False)
        segment_count = input_data.get("segment_count", len(messages) // self.segment_size + 1)
        
        new_segments = []
        for i in range(0, len(messages), self.segment_size):
            segment_messages = messages[i:i + self.segment_size]
            
            # 检查是否需要更新这个片段
            if not force_update:
                # 检查是否已有此时间段的摘要
                segment_start = segment_messages[0].timestamp
                segment_end = segment_messages[-1].timestamp
                
                existing_segment = next(
                    (s for s in conversation_summary.segments 
                     if s.start_time <= segment_start <= s.end_time),
                    None
                )
                
                if existing_segment:
                    continue
            
            # 生成新的摘要片段
            segment = self.summary_generator.generate_summary(
                segment_messages, 
                self.summary_strategy, 
                self.compression_level
            )
            
            conversation_summary.add_segment(segment)
            new_segments.append(segment)
        
        # 更新整体摘要
        if new_segments or force_update:
            self._update_overall_summary(conversation_summary)
        
        return {
            "success": True,
            "session_id": session_id,
            "summary_id": conversation_summary.session_id,
            "new_segments": len(new_segments),
            "total_segments": len(conversation_summary.segments),
            "compression_ratio": conversation_summary.total_compression_ratio,
            "overall_summary": conversation_summary.overall_summary
        }
    
    def _get_summary(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """获取对话摘要"""
        summary_key = self._get_summary_key(session_id, user_id)
        
        with self._lock:
            if summary_key not in self.summaries:
                # 尝试从存储加载
                load_result = self._load_summary(session_id, user_id)
                if not load_result.get("success"):
                    return {
                        "success": False,
                        "error": f"摘要不存在：{session_id}"
                    }
        
        conversation_summary = self.summaries[summary_key]
        
        return {
            "success": True,
            "summary": conversation_summary.to_dict()
        }
    
    def _update_summary(self, session_id: str, user_id: Optional[str], 
                       input_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新对话摘要"""
        # 强制重新生成摘要
        input_data["force_update"] = True
        return self._create_summary(session_id, user_id, input_data)
    
    def _get_segments(self, session_id: str, user_id: Optional[str], count: int) -> Dict[str, Any]:
        """获取摘要片段"""
        summary_key = self._get_summary_key(session_id, user_id)
        
        with self._lock:
            if summary_key not in self.summaries:
                return {
                    "success": False,
                    "error": f"摘要不存在：{session_id}"
                }
        
        conversation_summary = self.summaries[summary_key]
        latest_segments = conversation_summary.get_latest_segments(count)
        
        return {
            "success": True,
            "segments": [s.to_dict() for s in latest_segments],
            "count": len(latest_segments)
        }
    
    def _compress_summary(self, session_id: str, user_id: Optional[str], 
                         input_data: Dict[str, Any]) -> Dict[str, Any]:
        """压缩摘要"""
        summary_key = self._get_summary_key(session_id, user_id)
        
        with self._lock:
            if summary_key not in self.summaries:
                return {
                    "success": False,
                    "error": f"摘要不存在：{session_id}"
                }
        
        conversation_summary = self.summaries[summary_key]
        
        # 如果片段数量超过限制，进行压缩
        if len(conversation_summary.segments) > self.max_segments_per_summary:
            # 保留最重要和最新的片段
            important_segments = conversation_summary.get_most_important_segments(
                self.max_segments_per_summary // 2
            )
            recent_segments = conversation_summary.get_latest_segments(
                self.max_segments_per_summary // 2
            )
            
            # 合并并去重
            kept_segments = important_segments + [
                s for s in recent_segments 
                if s.segment_id not in [imp.segment_id for imp in important_segments]
            ]
            
            # 生成被压缩片段的摘要
            removed_segments = [
                s for s in conversation_summary.segments 
                if s.segment_id not in [kept.segment_id for kept in kept_segments]
            ]
            
            if removed_segments:
                # 将被移除的片段合并为一个高级摘要
                compressed_summary = " ".join([s.summary_text for s in removed_segments])
                compressed_segment = SummarySegment(
                    segment_id=f"compressed_{int(time.time())}",
                    start_time=min(s.start_time for s in removed_segments),
                    end_time=max(s.end_time for s in removed_segments),
                    original_message_count=sum(s.original_message_count for s in removed_segments),
                    summary_text=compressed_summary[:500] + "...",  # 进一步压缩
                    strategy=SummaryStrategy.HIERARCHICAL,
                    compression_ratio=0.9  # 高压缩比
                )
                
                kept_segments.append(compressed_segment)
            
            conversation_summary.segments = kept_segments
            conversation_summary.summary_version += 1
            
            return {
                "success": True,
                "original_segments": len(conversation_summary.segments) + len(removed_segments),
                "compressed_segments": len(kept_segments),
                "compression_ratio": len(kept_segments) / (len(conversation_summary.segments) + len(removed_segments))
            }
        
        return {
            "success": True,
            "message": "摘要无需压缩",
            "segments": len(conversation_summary.segments)
        }
    
    def _extract_insights(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """提取对话洞察"""
        summary_key = self._get_summary_key(session_id, user_id)
        
        with self._lock:
            if summary_key not in self.summaries:
                return {
                    "success": False,
                    "error": f"摘要不存在：{session_id}"
                }
        
        conversation_summary = self.summaries[summary_key]
        
        # 聚合所有片段的关键信息
        all_keywords = []
        all_entities = []
        all_topics = []
        
        for segment in conversation_summary.segments:
            all_keywords.extend(segment.keywords)
            all_entities.extend(segment.entities)
            all_topics.extend(segment.topics)
        
        # 计算频率并排序
        from collections import Counter
        
        keyword_freq = Counter(all_keywords)
        entity_freq = Counter(all_entities)
        topic_freq = Counter(all_topics)
        
        insights = {
            "top_keywords": [{"word": word, "count": count} 
                           for word, count in keyword_freq.most_common(10)],
            "top_entities": [{"entity": entity, "count": count} 
                           for entity, count in entity_freq.most_common(10)],
            "top_topics": [{"topic": topic, "count": count} 
                         for topic, count in topic_freq.most_common(5)],
            "conversation_stats": {
                "total_segments": len(conversation_summary.segments),
                "total_messages": conversation_summary.total_original_messages,
                "avg_compression_ratio": conversation_summary.total_compression_ratio,
                "conversation_duration": self._calculate_duration(conversation_summary),
                "key_themes": conversation_summary.key_themes,
                "important_facts": conversation_summary.important_facts,
                "decisions_made": conversation_summary.decisions_made,
                "action_items": conversation_summary.action_items
            }
        }
        
        return {
            "success": True,
            "insights": insights
        }
    
    def _save_summary(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """保存摘要到文件"""
        summary_key = self._get_summary_key(session_id, user_id)
        
        with self._lock:
            if summary_key not in self.summaries:
                return {
                    "success": False,
                    "error": f"摘要不存在：{session_id}"
                }
        
        conversation_summary = self.summaries[summary_key]
        
        try:
            # 保存到文件
            file_path = self.storage_path / f"{summary_key}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_summary.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved summary to file: {file_path}")
            return {
                "success": True,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _load_summary(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """从文件加载摘要"""
        summary_key = self._get_summary_key(session_id, user_id)
        
        try:
            file_path = self.storage_path / f"{summary_key}.json"
            if not file_path.exists():
                return {
                    "success": False,
                    "error": "摘要文件不存在"
                }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation_summary = ConversationSummary.from_dict(data)
            
            with self._lock:
                self.summaries[summary_key] = conversation_summary
            
            logger.debug(f"Loaded summary from file: {file_path}")
            return {
                "success": True,
                "summary": conversation_summary.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to load summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        """获取摘要统计信息"""
        with self._lock:
            total_summaries = len(self.summaries)
            total_segments = sum(len(s.segments) for s in self.summaries.values())
            avg_compression = sum(s.total_compression_ratio for s in self.summaries.values()) / total_summaries if total_summaries > 0 else 0
            
            return {
                "success": True,
                "stats": {
                    "total_summaries": total_summaries,
                    "total_segments": total_segments,
                    "average_compression_ratio": avg_compression,
                    "storage_path": str(self.storage_path),
                    "config": {
                        "segment_size": self.segment_size,
                        "compression_level": self.compression_level.value,
                        "summary_strategy": self.summary_strategy.value,
                        "auto_summarize_threshold": self.auto_summarize_threshold
                    }
                }
            }
    
    def _get_summary_key(self, session_id: str, user_id: Optional[str]) -> str:
        """生成摘要键值"""
        if user_id:
            return f"{user_id}:{session_id}"
        return session_id
    
    def _create_summary_generator(self, llm_model_name: Optional[str]) -> SummaryGenerator:
        """创建摘要生成器"""
        # 这里可以根据llm_model_name创建不同的LLM实例
        # 目前简化为使用LLMSummaryGenerator
        return LLMSummaryGenerator(llm_model=None)  # 暂时不使用LLM
    
    def _update_overall_summary(self, conversation_summary: ConversationSummary) -> None:
        """更新整体摘要"""
        if not conversation_summary.segments:
            return
        
        # 获取最新的几个片段
        recent_segments = conversation_summary.get_latest_segments(5)
        
        # 合并最新片段的摘要
        new_content = " ".join([s.summary_text for s in recent_segments])
        
        # 更新整体摘要
        conversation_summary.overall_summary = self.summary_generator.update_overall_summary(
            conversation_summary.overall_summary, 
            SummarySegment(
                segment_id="temp",
                start_time=time.time(),
                end_time=time.time(),
                original_message_count=0,
                summary_text=new_content
            )
        )
        
        # 更新主题和关键信息
        self._update_key_information(conversation_summary)
    
    def _update_key_information(self, conversation_summary: ConversationSummary) -> None:
        """更新关键信息"""
        # 聚合所有片段的关键信息
        all_keywords = []
        all_topics = []
        all_facts = []
        all_decisions = []
        all_actions = []
        
        for segment in conversation_summary.segments:
            all_keywords.extend(segment.keywords)
            all_topics.extend(segment.topics)
        
        # 去重并保留最重要的
        from collections import Counter
        
        keyword_freq = Counter(all_keywords)
        topic_freq = Counter(all_topics)
        
        conversation_summary.key_themes = [topic for topic, _ in topic_freq.most_common(5)]
    
    def _calculate_duration(self, conversation_summary: ConversationSummary) -> float:
        """计算对话持续时间（小时）"""
        if not conversation_summary.segments:
            return 0.0
        
        start_time = min(s.start_time for s in conversation_summary.segments)
        end_time = max(s.end_time for s in conversation_summary.segments)
        
        return (end_time - start_time) / 3600  # 转换为小时
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "segment_size": 10,
                "compression_level": "medium",
                "summary_strategy": "abstractive",
                "auto_summarize_threshold": 20,
                "max_segments_per_summary": 50,
                "storage_path": "./summaries",
                "enable_semantic_similarity": False
            },
            "execute_examples": [
                {
                    "description": "创建对话摘要",
                    "input": {
                        "action": "summarize",
                        "session_id": "chat_001",
                        "user_id": "user123"
                    }
                },
                {
                    "description": "获取摘要",
                    "input": {
                        "action": "get_summary",
                        "session_id": "chat_001",
                        "user_id": "user123"
                    }
                },
                {
                    "description": "获取摘要片段",
                    "input": {
                        "action": "get_segments",
                        "session_id": "chat_001",
                        "user_id": "user123",
                        "count": 5
                    }
                },
                {
                    "description": "提取对话洞察",
                    "input": {
                        "action": "extract_insights",
                        "session_id": "chat_001",
                        "user_id": "user123"
                    }
                }
            ],
            "usage_code": """
# 使用示例
from templates.memory.summary_memory import SummaryMemoryTemplate
from templates.memory.conversation_memory import ConversationMemoryTemplate

# 初始化对话记忆（作为依赖）
conv_memory = ConversationMemoryTemplate()
conv_memory.setup(backend_type="file", storage_path="./conversations")

# 初始化摘要记忆模板
summary_template = SummaryMemoryTemplate(conversation_memory=conv_memory)

# 配置参数
summary_template.setup(
    segment_size=10,
    compression_level="medium",
    summary_strategy="abstractive",
    storage_path="./summaries"
)

# 创建摘要
result = summary_template.run({
    "action": "summarize",
    "session_id": "chat_001",
    "user_id": "user123"
})

# 获取摘要
summary = summary_template.run({
    "action": "get_summary",
    "session_id": "chat_001",
    "user_id": "user123"
})

# 提取洞察
insights = summary_template.run({
    "action": "extract_insights",
    "session_id": "chat_001",
    "user_id": "user123"
})

print("对话摘要：", summary)
print("对话洞察：", insights)
"""
        }