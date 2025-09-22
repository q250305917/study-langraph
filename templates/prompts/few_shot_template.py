"""
少样本学习模板（FewShotTemplate）

本模块提供了强大的少样本学习（Few-Shot Learning）模板系统，通过少量示例实现高效的任务学习。
支持智能示例选择、动态示例生成、相似度匹配等高级功能。

核心特性：
1. 智能示例选择：基于相似度自动选择最相关的示例
2. 动态示例管理：运行时添加、更新、删除示例
3. 多种匹配策略：支持语义、关键词、结构等多种匹配方式
4. 示例质量控制：自动评估和过滤低质量示例
5. 格式化输出：统一的示例格式化和展示
6. 增量学习：支持在线学习和示例库扩展

设计原理：
- 策略模式：支持不同的示例选择和匹配策略
- 模板方法模式：定义标准的少样本学习流程
- 工厂模式：动态创建不同类型的示例选择器
- 观察者模式：监控示例库变化和性能指标
- 装饰器模式：增强示例处理和优化功能

使用场景：
- 文本分类：基于少量标记样本进行分类
- 情感分析：通过示例学习情感识别
- 命名实体识别：识别文本中的特定实体
- 代码生成：基于示例生成相似代码
- 问答系统：通过问答示例学习回答模式
- 文本摘要：学习摘要生成的模式
"""

import re
import json
import math
import time
import random
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ErrorCodes
)

logger = get_logger(__name__)


class ExampleType(Enum):
    """示例类型枚举"""
    CLASSIFICATION = "classification"      # 分类任务
    QA = "qa"                             # 问答任务
    GENERATION = "generation"             # 生成任务
    TRANSLATION = "translation"           # 翻译任务
    COMPLETION = "completion"             # 补全任务
    EXTRACTION = "extraction"             # 抽取任务
    REWRITING = "rewriting"               # 重写任务
    REASONING = "reasoning"               # 推理任务


class SelectionStrategy(Enum):
    """示例选择策略枚举"""
    SIMILARITY = "similarity"             # 相似度选择
    RANDOM = "random"                     # 随机选择
    DIVERSE = "diverse"                   # 多样性选择
    RECENT = "recent"                     # 最新优先
    HIGH_QUALITY = "high_quality"         # 高质量优先
    KEYWORD_MATCH = "keyword_match"       # 关键词匹配
    SEMANTIC = "semantic"                 # 语义匹配
    ADAPTIVE = "adaptive"                 # 自适应选择


@dataclass
class Example:
    """
    示例数据类
    
    表示少样本学习中的单个示例。
    """
    input_text: str                       # 输入文本
    output_text: str                      # 期望输出
    example_type: ExampleType             # 示例类型
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    # 质量指标
    quality_score: float = 1.0            # 质量分数
    usage_count: int = 0                  # 使用次数
    success_rate: float = 1.0             # 成功率
    
    # 时间信息
    created_time: datetime = field(default_factory=datetime.now)  # 创建时间
    last_used: Optional[datetime] = None  # 最后使用时间
    
    # 特征信息
    keywords: List[str] = field(default_factory=list)       # 关键词
    categories: List[str] = field(default_factory=list)     # 类别
    difficulty: float = 0.5               # 难度系数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "example_type": self.example_type.value,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_time": self.created_time.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "keywords": self.keywords,
            "categories": self.categories,
            "difficulty": self.difficulty
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Example":
        """从字典创建实例"""
        example = cls(
            input_text=data["input_text"],
            output_text=data["output_text"],
            example_type=ExampleType(data["example_type"]),
            metadata=data.get("metadata", {}),
            quality_score=data.get("quality_score", 1.0),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 1.0),
            keywords=data.get("keywords", []),
            categories=data.get("categories", []),
            difficulty=data.get("difficulty", 0.5)
        )
        
        if data.get("created_time"):
            example.created_time = datetime.fromisoformat(data["created_time"])
        if data.get("last_used"):
            example.last_used = datetime.fromisoformat(data["last_used"])
        
        return example
    
    def update_usage(self, success: bool = True) -> None:
        """更新使用统计"""
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # 更新成功率（使用指数移动平均）
        alpha = 0.1
        if success:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 1.0
        else:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 0.0


@dataclass
class FewShotContext:
    """
    少样本学习上下文数据类
    
    管理少样本学习任务的上下文信息。
    """
    query: str                            # 查询文本
    example_type: ExampleType             # 示例类型
    selection_strategy: SelectionStrategy # 选择策略
    max_examples: int = 5                 # 最大示例数
    min_examples: int = 1                 # 最小示例数
    similarity_threshold: float = 0.3     # 相似度阈值
    diversity_weight: float = 0.3         # 多样性权重
    quality_weight: float = 0.7           # 质量权重
    task_description: str = ""            # 任务描述
    output_format: str = ""               # 输出格式要求
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件
    metadata: Dict[str, Any] = field(default_factory=dict)     # 元数据


@dataclass
class FewShotResult:
    """
    少样本学习结果数据类
    
    封装少样本学习的结果和相关信息。
    """
    prediction: str                       # 预测结果
    selected_examples: List[Example]      # 选中的示例
    context: FewShotContext              # 原始上下文
    confidence: float = 1.0               # 置信度
    reasoning: str = ""                   # 推理过程
    processing_time: float = 0.0         # 处理时间
    token_count: int = 0                  # Token数量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 结果元数据


class ExampleSelector:
    """
    示例选择器基类
    
    定义示例选择的通用接口。
    """
    
    def __init__(self, strategy: SelectionStrategy):
        """
        初始化示例选择器
        
        Args:
            strategy: 选择策略
        """
        self.strategy = strategy
    
    def select_examples(
        self, 
        query: str, 
        examples: List[Example], 
        context: FewShotContext
    ) -> List[Example]:
        """
        选择示例
        
        Args:
            query: 查询文本
            examples: 候选示例列表
            context: 少样本上下文
            
        Returns:
            选中的示例列表
        """
        raise NotImplementedError


class SimilaritySelector(ExampleSelector):
    """基于相似度的示例选择器"""
    
    def select_examples(
        self, 
        query: str, 
        examples: List[Example], 
        context: FewShotContext
    ) -> List[Example]:
        """基于相似度选择示例"""
        if not examples:
            return []
        
        # 计算相似度
        scored_examples = []
        for example in examples:
            similarity = self._calculate_similarity(query, example.input_text)
            if similarity >= context.similarity_threshold:
                scored_examples.append((similarity, example))
        
        # 按相似度排序
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # 选择前N个
        selected = [example for _, example in scored_examples[:context.max_examples]]
        
        return selected
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        # 简单的词汇重叠相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


class DiverseSelector(ExampleSelector):
    """基于多样性的示例选择器"""
    
    def select_examples(
        self, 
        query: str, 
        examples: List[Example], 
        context: FewShotContext
    ) -> List[Example]:
        """基于多样性选择示例"""
        if not examples:
            return []
        
        selected = []
        remaining = examples.copy()
        
        # 首先选择最相似的示例
        if remaining:
            similarities = [(self._calculate_similarity(query, ex.input_text), ex) for ex in remaining]
            similarities.sort(key=lambda x: x[0], reverse=True)
            selected.append(similarities[0][1])
            remaining.remove(similarities[0][1])
        
        # 然后选择多样性最大的示例
        while len(selected) < context.max_examples and remaining:
            best_candidate = None
            best_diversity = -1
            
            for candidate in remaining:
                diversity = self._calculate_diversity(candidate, selected)
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        return len(words1.intersection(words2)) / len(words1.union(words2))
    
    def _calculate_diversity(self, candidate: Example, selected: List[Example]) -> float:
        """计算候选示例与已选示例的多样性"""
        if not selected:
            return 1.0
        
        similarities = [
            self._calculate_similarity(candidate.input_text, ex.input_text)
            for ex in selected
        ]
        
        # 多样性 = 1 - 平均相似度
        return 1.0 - (sum(similarities) / len(similarities))


class AdaptiveSelector(ExampleSelector):
    """自适应示例选择器"""
    
    def select_examples(
        self, 
        query: str, 
        examples: List[Example], 
        context: FewShotContext
    ) -> List[Example]:
        """自适应选择示例"""
        if not examples:
            return []
        
        # 综合考虑相似度、质量、多样性等因素
        scored_examples = []
        
        for example in examples:
            similarity = self._calculate_similarity(query, example.input_text)
            quality = example.quality_score
            recency = self._calculate_recency_score(example)
            
            # 综合评分
            score = (
                similarity * 0.4 +
                quality * context.quality_weight * 0.3 +
                recency * 0.2 +
                (1.0 / (example.usage_count + 1)) * 0.1  # 避免过度使用
            )
            
            scored_examples.append((score, example))
        
        # 按综合评分排序
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # 选择前N个，但考虑多样性
        selected = []
        for score, example in scored_examples:
            if len(selected) >= context.max_examples:
                break
            
            # 检查多样性
            if not selected or self._is_diverse_enough(example, selected):
                selected.append(example)
        
        return selected
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        return len(words1.intersection(words2)) / len(words1.union(words2))
    
    def _calculate_recency_score(self, example: Example) -> float:
        """计算时间新鲜度分数"""
        if not example.last_used:
            return 0.5
        
        days_since_used = (datetime.now() - example.last_used).days
        return max(0.0, 1.0 - days_since_used / 30.0)  # 30天内为满分
    
    def _is_diverse_enough(self, candidate: Example, selected: List[Example]) -> bool:
        """检查候选示例是否足够多样"""
        if not selected:
            return True
        
        similarities = [
            self._calculate_similarity(candidate.input_text, ex.input_text)
            for ex in selected
        ]
        
        max_similarity = max(similarities)
        return max_similarity < 0.8  # 相似度阈值


class ExampleDatabase:
    """
    示例数据库
    
    管理少样本学习的示例集合。
    """
    
    def __init__(self):
        """初始化示例数据库"""
        self.examples: Dict[ExampleType, List[Example]] = defaultdict(list)
        self.total_examples = 0
        self.indices = {}  # 可以添加索引以提高查询效率
    
    def add_example(self, example: Example) -> None:
        """
        添加示例
        
        Args:
            example: 要添加的示例
        """
        self.examples[example.example_type].append(example)
        self.total_examples += 1
        
        # 提取关键词（如果没有）
        if not example.keywords:
            example.keywords = self._extract_keywords(example.input_text)
        
        logger.debug(f"Added example for {example.example_type.value}: {example.input_text[:50]}...")
    
    def get_examples(
        self, 
        example_type: Optional[ExampleType] = None,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_quality: float = 0.0
    ) -> List[Example]:
        """
        获取示例列表
        
        Args:
            example_type: 示例类型过滤
            keywords: 关键词过滤
            categories: 类别过滤
            min_quality: 最小质量阈值
            
        Returns:
            符合条件的示例列表
        """
        if example_type:
            candidates = self.examples[example_type]
        else:
            candidates = []
            for type_examples in self.examples.values():
                candidates.extend(type_examples)
        
        # 应用过滤条件
        filtered = []
        for example in candidates:
            # 质量过滤
            if example.quality_score < min_quality:
                continue
            
            # 关键词过滤
            if keywords and not any(kw in example.keywords for kw in keywords):
                continue
            
            # 类别过滤
            if categories and not any(cat in example.categories for cat in categories):
                continue
            
            filtered.append(example)
        
        return filtered
    
    def update_example_quality(self, example: Example, success: bool) -> None:
        """
        更新示例质量
        
        Args:
            example: 要更新的示例
            success: 是否成功使用
        """
        example.update_usage(success)
        
        # 基于成功率调整质量分数
        if example.usage_count >= 5:  # 有足够的使用数据
            example.quality_score = example.success_rate
    
    def remove_low_quality_examples(self, min_quality: float = 0.3) -> int:
        """
        移除低质量示例
        
        Args:
            min_quality: 最小质量阈值
            
        Returns:
            移除的示例数量
        """
        removed_count = 0
        
        for example_type in list(self.examples.keys()):
            filtered_examples = []
            for example in self.examples[example_type]:
                if example.quality_score >= min_quality or example.usage_count < 3:
                    filtered_examples.append(example)
                else:
                    removed_count += 1
            
            self.examples[example_type] = filtered_examples
        
        self.total_examples -= removed_count
        logger.info(f"Removed {removed_count} low-quality examples")
        return removed_count
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # 过滤停用词
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     '的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那', '有', '没'}
        
        keywords = [word for word in words if word not in stop_words]
        
        # 返回频率最高的词
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(5)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {
            "total_examples": self.total_examples,
            "examples_by_type": {
                etype.value: len(examples) 
                for etype, examples in self.examples.items()
            },
            "average_quality": 0.0,
            "total_usage": 0
        }
        
        if self.total_examples > 0:
            all_examples = []
            for type_examples in self.examples.values():
                all_examples.extend(type_examples)
            
            stats["average_quality"] = sum(ex.quality_score for ex in all_examples) / len(all_examples)
            stats["total_usage"] = sum(ex.usage_count for ex in all_examples)
        
        return stats
    
    def export_examples(self) -> Dict[str, Any]:
        """导出示例数据"""
        export_data = {}
        for example_type, examples in self.examples.items():
            export_data[example_type.value] = [ex.to_dict() for ex in examples]
        
        return export_data
    
    def import_examples(self, data: Dict[str, Any]) -> int:
        """
        导入示例数据
        
        Args:
            data: 示例数据
            
        Returns:
            导入的示例数量
        """
        imported_count = 0
        
        for type_name, examples_data in data.items():
            try:
                example_type = ExampleType(type_name)
                for example_data in examples_data:
                    example = Example.from_dict(example_data)
                    self.add_example(example)
                    imported_count += 1
            except ValueError as e:
                logger.warning(f"Unknown example type: {type_name}")
        
        logger.info(f"Imported {imported_count} examples")
        return imported_count
class FewShotTemplate(TemplateBase[FewShotContext, FewShotResult]):
    """
    少样本学习模板类
    
    提供强大的少样本学习功能，通过少量示例实现高效的任务学习。
    
    核心功能：
    1. 示例管理：维护和管理示例数据库
    2. 智能选择：基于多种策略选择最佳示例
    3. 格式化生成：将示例格式化为有效的提示词
    4. 质量控制：评估和维护示例质量
    5. 增量学习：支持在线添加和更新示例
    6. 性能优化：高效的示例检索和匹配
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化少样本学习模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config or self._create_default_config())
        
        # 示例数据库
        self.example_db = ExampleDatabase()
        
        # 示例选择器
        self.selectors = {
            SelectionStrategy.SIMILARITY: SimilaritySelector(SelectionStrategy.SIMILARITY),
            SelectionStrategy.DIVERSE: DiverseSelector(SelectionStrategy.DIVERSE),
            SelectionStrategy.ADAPTIVE: AdaptiveSelector(SelectionStrategy.ADAPTIVE),
            SelectionStrategy.RANDOM: self._create_random_selector(),
            SelectionStrategy.HIGH_QUALITY: self._create_quality_selector()
        }
        
        # 默认配置
        self.default_strategy = SelectionStrategy.ADAPTIVE
        self.default_max_examples = 5
        self.default_example_type = ExampleType.GENERATION
        self.format_template = ""
        self.instruction_template = ""
        
        # 集成的LLM模板
        self.llm_template = None
        
        # 性能统计
        self.usage_stats = {
            "total_queries": 0,
            "successful_predictions": 0,
            "average_examples_used": 0.0,
            "strategy_usage": defaultdict(int)
        }
        
        logger.debug("FewShotTemplate initialized")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="FewShotTemplate",
            version="1.0.0",
            description="少样本学习模板，支持智能示例选择和任务学习",
            template_type=TemplateType.PROMPT,
            author="LangChain Learning Project",
            async_enabled=True
        )
        
        # 添加参数定义
        config.add_parameter("example_type", str, False, "generation", "示例类型")
        config.add_parameter("selection_strategy", str, False, "adaptive", "选择策略")
        config.add_parameter("max_examples", int, False, 5, "最大示例数")
        config.add_parameter("min_examples", int, False, 1, "最小示例数")
        config.add_parameter("similarity_threshold", float, False, 0.3, "相似度阈值")
        config.add_parameter("quality_threshold", float, False, 0.5, "质量阈值")
        config.add_parameter("enable_quality_control", bool, False, True, "启用质量控制")
        config.add_parameter("auto_update_quality", bool, False, True, "自动更新质量")
        config.add_parameter("instruction_template", str, False, "", "指令模板")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置少样本学习模板参数
        
        Args:
            **parameters: 设置参数
            
        主要参数：
            example_type (str): 默认示例类型
            selection_strategy (str): 默认选择策略
            max_examples (int): 最大示例数
            min_examples (int): 最小示例数
            similarity_threshold (float): 相似度阈值
            quality_threshold (float): 质量阈值
            enable_quality_control (bool): 启用质量控制
            auto_update_quality (bool): 自动更新质量
            llm_template: 集成的LLM模板实例
            instruction_template (str): 自定义指令模板
            format_template (str): 自定义格式模板
            
        Raises:
            ValidationError: 参数验证失败
            ConfigurationError: 配置错误
        """
        try:
            # 验证参数
            if not self.validate_parameters(parameters):
                raise ValidationError("Parameters validation failed")
            
            # 设置基本参数
            example_type_str = parameters.get("example_type", "generation")
            self.default_example_type = ExampleType(example_type_str.lower())
            
            strategy_str = parameters.get("selection_strategy", "adaptive")
            self.default_strategy = SelectionStrategy(strategy_str.lower())
            
            self.default_max_examples = parameters.get("max_examples", 5)
            self.default_min_examples = parameters.get("min_examples", 1)
            self.similarity_threshold = parameters.get("similarity_threshold", 0.3)
            self.quality_threshold = parameters.get("quality_threshold", 0.5)
            self.enable_quality_control = parameters.get("enable_quality_control", True)
            self.auto_update_quality = parameters.get("auto_update_quality", True)
            
            # 设置LLM模板
            self.llm_template = parameters.get("llm_template")
            if not self.llm_template:
                logger.warning("No LLM template provided, will need to set one later")
            
            # 设置模板
            self.instruction_template = parameters.get(
                "instruction_template",
                self._get_default_instruction_template()
            )
            self.format_template = parameters.get(
                "format_template",
                self._get_default_format_template()
            )
            
            # 保存设置参数
            self._setup_parameters = parameters.copy()
            
            logger.info(f"FewShotTemplate configured with strategy: {self.default_strategy.value}")
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to setup few-shot template: {str(e)}",
                error_code=ErrorCodes.CONFIG_VALIDATION_ERROR,
                cause=e
            )
    
    def execute(self, input_data: FewShotContext, **kwargs) -> FewShotResult:
        """
        执行少样本学习推理
        
        Args:
            input_data: 少样本学习上下文
            **kwargs: 额外参数
                
        Returns:
            少样本学习结果对象
        """
        try:
            start_time = time.time()
            
            # 验证输入
            if not isinstance(input_data, FewShotContext):
                # 尝试从字符串创建上下文
                if isinstance(input_data, str):
                    input_data = self._create_context_from_query(input_data, kwargs)
                else:
                    raise ValidationError(f"Invalid input type: {type(input_data)}")
            
            # 获取候选示例
            candidate_examples = self._get_candidate_examples(input_data)
            
            if not candidate_examples:
                logger.warning("No candidate examples found")
                # 创建无示例的结果
                return self._create_no_examples_result(input_data, start_time)
            
            # 选择最佳示例
            selected_examples = self._select_examples(input_data, candidate_examples)
            
            if not selected_examples:
                logger.warning("No examples selected")
                return self._create_no_examples_result(input_data, start_time)
            
            # 构建少样本提示词
            prompt = self._build_few_shot_prompt(input_data, selected_examples)
            
            # 调用LLM进行推理
            if not self.llm_template:
                raise ConfigurationError("No LLM template configured")
            
            llm_response = self.llm_template.execute(prompt, **kwargs)
            prediction = self._extract_prediction(llm_response)
            
            # 后处理预测结果
            processed_prediction = self._post_process_prediction(prediction, input_data)
            
            # 创建结果对象
            result = FewShotResult(
                prediction=processed_prediction,
                selected_examples=selected_examples,
                context=input_data,
                processing_time=time.time() - start_time
            )
            
            # 提取token计数（如果LLM响应提供）
            if hasattr(llm_response, 'total_tokens'):
                result.token_count = llm_response.total_tokens
            
            # 更新统计信息
            self._update_usage_stats(input_data, result)
            
            # 更新示例使用情况（如果启用自动更新）
            if self.auto_update_quality:
                self._update_example_usage(selected_examples, True)  # 假设成功
            
            logger.debug(
                f"Few-shot prediction completed: {len(selected_examples)} examples used, "
                f"time: {result.processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Few-shot execution failed: {str(e)}")
            # 返回错误结果
            return FewShotResult(
                prediction=f"推理失败：{str(e)}",
                selected_examples=[],
                context=input_data if isinstance(input_data, FewShotContext) else FewShotContext(
                    query=str(input_data),
                    example_type=self.default_example_type,
                    selection_strategy=self.default_strategy
                ),
                confidence=0.0,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0.0,
                metadata={"error": str(e)}
            )
    
    def _create_context_from_query(self, query: str, kwargs: Dict[str, Any]) -> FewShotContext:
        """
        从查询创建少样本学习上下文
        
        Args:
            query: 查询文本
            kwargs: 额外参数
            
        Returns:
            少样本学习上下文对象
        """
        example_type_str = kwargs.get("example_type", self.default_example_type.value)
        strategy_str = kwargs.get("selection_strategy", self.default_strategy.value)
        
        return FewShotContext(
            query=query,
            example_type=ExampleType(example_type_str.lower()),
            selection_strategy=SelectionStrategy(strategy_str.lower()),
            max_examples=kwargs.get("max_examples", self.default_max_examples),
            min_examples=kwargs.get("min_examples", self.default_min_examples),
            similarity_threshold=kwargs.get("similarity_threshold", self.similarity_threshold),
            task_description=kwargs.get("task_description", ""),
            output_format=kwargs.get("output_format", ""),
            constraints=kwargs.get("constraints", {})
        )
    
    def _get_candidate_examples(self, context: FewShotContext) -> List[Example]:
        """
        获取候选示例
        
        Args:
            context: 少样本学习上下文
            
        Returns:
            候选示例列表
        """
        # 从数据库获取相应类型的示例
        candidates = self.example_db.get_examples(
            example_type=context.example_type,
            min_quality=self.quality_threshold if self.enable_quality_control else 0.0
        )
        
        # 可以根据查询进一步过滤
        if context.constraints.get("keywords"):
            keywords = context.constraints["keywords"]
            candidates = [
                ex for ex in candidates
                if any(kw.lower() in ex.input_text.lower() for kw in keywords)
            ]
        
        return candidates
    
    def _select_examples(self, context: FewShotContext, candidates: List[Example]) -> List[Example]:
        """
        选择最佳示例
        
        Args:
            context: 少样本学习上下文
            candidates: 候选示例列表
            
        Returns:
            选中的示例列表
        """
        if not candidates:
            return []
        
        # 获取相应的选择器
        selector = self.selectors.get(context.selection_strategy)
        if not selector:
            logger.warning(f"Unknown selection strategy: {context.selection_strategy.value}")
            selector = self.selectors[SelectionStrategy.ADAPTIVE]
        
        # 选择示例
        selected = selector.select_examples(context.query, candidates, context)
        
        # 确保数量在限制范围内
        if len(selected) < context.min_examples:
            # 如果选中的示例太少，随机补充
            remaining = [ex for ex in candidates if ex not in selected]
            needed = context.min_examples - len(selected)
            if remaining:
                additional = random.sample(remaining, min(needed, len(remaining)))
                selected.extend(additional)
        
        return selected[:context.max_examples]
    
    def _build_few_shot_prompt(self, context: FewShotContext, examples: List[Example]) -> str:
        """
        构建少样本提示词
        
        Args:
            context: 少样本学习上下文
            examples: 选中的示例
            
        Returns:
            构建的提示词
        """
        # 构建指令部分
        instruction = self._build_instruction(context)
        
        # 构建示例部分
        examples_text = self._format_examples(examples, context)
        
        # 构建查询部分
        query_text = self._format_query(context)
        
        # 组合完整提示词
        prompt_parts = [instruction]
        
        if examples_text:
            prompt_parts.append("示例：")
            prompt_parts.append(examples_text)
        
        prompt_parts.append("现在，请处理以下输入：")
        prompt_parts.append(query_text)
        
        return "\n\n".join(prompt_parts)
    
    def _build_instruction(self, context: FewShotContext) -> str:
        """构建指令部分"""
        if context.task_description:
            instruction = context.task_description
        else:
            # 基于示例类型生成默认指令
            type_instructions = {
                ExampleType.CLASSIFICATION: "请根据以下示例，对输入文本进行分类。",
                ExampleType.QA: "请根据以下问答示例，回答问题。",
                ExampleType.GENERATION: "请根据以下示例，生成相应的输出。",
                ExampleType.TRANSLATION: "请根据以下翻译示例，翻译输入文本。",
                ExampleType.COMPLETION: "请根据以下示例，完成输入内容。",
                ExampleType.EXTRACTION: "请根据以下示例，从输入中提取相关信息。",
                ExampleType.REWRITING: "请根据以下示例，重写输入文本。",
                ExampleType.REASONING: "请根据以下推理示例，分析并回答问题。"
            }
            instruction = type_instructions.get(
                context.example_type,
                "请根据以下示例，处理输入内容。"
            )
        
        # 添加输出格式要求
        if context.output_format:
            instruction += f"\n\n输出格式要求：{context.output_format}"
        
        # 添加约束条件
        if context.constraints:
            constraints_text = self._format_constraints(context.constraints)
            instruction += f"\n\n约束条件：\n{constraints_text}"
        
        return instruction
    
    def _format_examples(self, examples: List[Example], context: FewShotContext) -> str:
        """格式化示例"""
        if not examples:
            return ""
        
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            if context.example_type == ExampleType.QA:
                formatted_example = f"示例 {i}:\n问题: {example.input_text}\n答案: {example.output_text}"
            elif context.example_type == ExampleType.CLASSIFICATION:
                formatted_example = f"示例 {i}:\n输入: {example.input_text}\n类别: {example.output_text}"
            elif context.example_type == ExampleType.TRANSLATION:
                formatted_example = f"示例 {i}:\n原文: {example.input_text}\n译文: {example.output_text}"
            else:
                formatted_example = f"示例 {i}:\n输入: {example.input_text}\n输出: {example.output_text}"
            
            formatted_examples.append(formatted_example)
        
        return "\n\n".join(formatted_examples)
    
    def _format_query(self, context: FewShotContext) -> str:
        """格式化查询"""
        if context.example_type == ExampleType.QA:
            return f"问题: {context.query}\n答案:"
        elif context.example_type == ExampleType.CLASSIFICATION:
            return f"输入: {context.query}\n类别:"
        elif context.example_type == ExampleType.TRANSLATION:
            return f"原文: {context.query}\n译文:"
        else:
            return f"输入: {context.query}\n输出:"
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """格式化约束条件"""
        constraint_parts = []
        
        if constraints.get("max_length"):
            constraint_parts.append(f"- 最大长度：{constraints['max_length']} 字符")
        
        if constraints.get("required_keywords"):
            keywords = ", ".join(constraints["required_keywords"])
            constraint_parts.append(f"- 必须包含关键词：{keywords}")
        
        if constraints.get("forbidden_words"):
            words = ", ".join(constraints["forbidden_words"])
            constraint_parts.append(f"- 禁止使用词汇：{words}")
        
        if constraints.get("style"):
            constraint_parts.append(f"- 风格要求：{constraints['style']}")
        
        return "\n".join(constraint_parts)
    
    def _extract_prediction(self, llm_response: Any) -> str:
        """从LLM响应中提取预测结果"""
        if hasattr(llm_response, 'content'):
            return llm_response.content
        elif hasattr(llm_response, 'text'):
            return llm_response.text
        elif isinstance(llm_response, str):
            return llm_response
        else:
            return str(llm_response)
    
    def _post_process_prediction(self, prediction: str, context: FewShotContext) -> str:
        """后处理预测结果"""
        # 基本清理
        processed = prediction.strip()
        
        # 移除可能的提示词残留
        prefixes_to_remove = ["输出:", "答案:", "类别:", "译文:", "结果:"]
        for prefix in prefixes_to_remove:
            if processed.startswith(prefix):
                processed = processed[len(prefix):].strip()
        
        # 应用约束条件
        if context.constraints.get("max_length"):
            max_len = context.constraints["max_length"]
            if len(processed) > max_len:
                processed = processed[:max_len].rstrip()
        
        return processed
    
    def _create_no_examples_result(self, context: FewShotContext, start_time: float) -> FewShotResult:
        """创建无示例的结果"""
        return FewShotResult(
            prediction="无法找到相关示例进行推理，请添加示例后重试。",
            selected_examples=[],
            context=context,
            confidence=0.0,
            processing_time=time.time() - start_time,
            metadata={"no_examples": True}
        )
    
    def _update_usage_stats(self, context: FewShotContext, result: FewShotResult) -> None:
        """更新使用统计"""
        self.usage_stats["total_queries"] += 1
        self.usage_stats["strategy_usage"][context.selection_strategy.value] += 1
        
        if result.selected_examples:
            # 更新平均示例使用数
            current_avg = self.usage_stats["average_examples_used"]
            total_queries = self.usage_stats["total_queries"]
            new_avg = ((current_avg * (total_queries - 1)) + len(result.selected_examples)) / total_queries
            self.usage_stats["average_examples_used"] = new_avg
    
    def _update_example_usage(self, examples: List[Example], success: bool) -> None:
        """更新示例使用情况"""
        for example in examples:
            self.example_db.update_example_quality(example, success)
    
    def _get_default_instruction_template(self) -> str:
        """获取默认指令模板"""
        return """
根据以下示例，学习输入和输出之间的模式，然后处理新的输入。

{task_description}

{format_requirements}

{constraints}
""".strip()
    
    def _get_default_format_template(self) -> str:
        """获取默认格式模板"""
        return """
示例 {index}:
输入: {input}
输出: {output}
""".strip()
    
    def _create_random_selector(self) -> ExampleSelector:
        """创建随机选择器"""
        class RandomSelector(ExampleSelector):
            def select_examples(self, query: str, examples: List[Example], context: FewShotContext) -> List[Example]:
                if not examples:
                    return []
                return random.sample(examples, min(context.max_examples, len(examples)))
        
        return RandomSelector(SelectionStrategy.RANDOM)
    
    def _create_quality_selector(self) -> ExampleSelector:
        """创建质量优先选择器"""
        class QualitySelector(ExampleSelector):
            def select_examples(self, query: str, examples: List[Example], context: FewShotContext) -> List[Example]:
                if not examples:
                    return []
                
                # 按质量分数排序
                sorted_examples = sorted(examples, key=lambda x: x.quality_score, reverse=True)
                return sorted_examples[:context.max_examples]
        
        return QualitySelector(SelectionStrategy.HIGH_QUALITY)
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "description": "少样本学习模板使用示例",
            "setup_parameters": {
                "example_type": "classification",
                "selection_strategy": "adaptive",
                "max_examples": 3,
                "similarity_threshold": 0.3,
                "enable_quality_control": True
            },
            "execute_parameters": {
                "input_data": "这部电影真的很棒，演员表演出色，剧情引人入胜。",
                "example_type": "classification",
                "task_description": "根据文本内容判断情感倾向",
                "output_format": "positive/negative/neutral"
            },
            "expected_output": {
                "type": "FewShotResult",
                "fields": {
                    "prediction": "positive",
                    "selected_examples": "用于推理的示例列表",
                    "confidence": "0.8-1.0之间的置信度"
                }
            },
            "usage_examples": '''
# 基础用法
from templates.prompts import FewShotTemplate
from templates.prompts.few_shot_template import FewShotContext, Example, ExampleType

template = FewShotTemplate()
template.setup(
    example_type="classification",
    selection_strategy="adaptive",
    llm_template=openai_template
)

# 添加示例
examples = [
    Example(
        input_text="这个产品质量很好，我很满意。",
        output_text="positive",
        example_type=ExampleType.CLASSIFICATION
    ),
    Example(
        input_text="服务态度太差了，完全不推荐。",
        output_text="negative", 
        example_type=ExampleType.CLASSIFICATION
    ),
    Example(
        input_text="价格还可以，没什么特别的。",
        output_text="neutral",
        example_type=ExampleType.CLASSIFICATION
    )
]

for example in examples:
    template.add_example(example)

# 进行推理
result = template.run("这家餐厅的菜品味道不错！")
print(f"预测结果: {result.prediction}")
print(f"使用的示例数量: {len(result.selected_examples)}")

# 使用上下文对象
context = FewShotContext(
    query="这个电影剧情很无聊。",
    example_type=ExampleType.CLASSIFICATION,
    selection_strategy=SelectionStrategy.SIMILARITY,
    max_examples=2,
    task_description="情感分类",
    output_format="positive/negative/neutral"
)

result = template.run(context)

# 问答任务
qa_examples = [
    Example(
        input_text="什么是机器学习？",
        output_text="机器学习是人工智能的一个分支，它让计算机能够从数据中自动学习和改进。",
        example_type=ExampleType.QA
    ),
    Example(
        input_text="Python有什么优势？",
        output_text="Python语法简洁，库丰富，适合快速开发和数据分析。",
        example_type=ExampleType.QA
    )
]

for example in qa_examples:
    template.add_example(example)

qa_result = template.run(
    "什么是深度学习？",
    example_type="qa",
    max_examples=2
)

# 批量添加示例
template.bulk_add_examples([
    {"input": "输入1", "output": "输出1", "type": "generation"},
    {"input": "输入2", "output": "输出2", "type": "generation"}
])

# 获取统计信息
stats = template.get_statistics()
print(f"总示例数: {stats['total_examples']}")
print(f"平均质量: {stats['average_quality']:.2f}")
'''
        }
    
    # 工具方法
    def add_example(self, example: Example) -> None:
        """
        添加示例
        
        Args:
            example: 要添加的示例
        """
        self.example_db.add_example(example)
        logger.info(f"Added example for {example.example_type.value}")
    
    def add_examples(self, examples: List[Example]) -> None:
        """
        批量添加示例
        
        Args:
            examples: 示例列表
        """
        for example in examples:
            self.example_db.add_example(example)
        logger.info(f"Added {len(examples)} examples")
    
    def bulk_add_examples(self, examples_data: List[Dict[str, Any]]) -> int:
        """
        批量添加示例（从字典数据）
        
        Args:
            examples_data: 示例数据列表
            
        Returns:
            成功添加的示例数量
        """
        added_count = 0
        for data in examples_data:
            try:
                example = Example(
                    input_text=data["input"],
                    output_text=data["output"],
                    example_type=ExampleType(data.get("type", "generation").lower()),
                    keywords=data.get("keywords", []),
                    categories=data.get("categories", []),
                    quality_score=data.get("quality", 1.0)
                )
                self.add_example(example)
                added_count += 1
            except Exception as e:
                logger.warning(f"Failed to add example: {str(e)}")
        
        return added_count
    
    def remove_example(self, example: Example) -> bool:
        """
        移除示例
        
        Args:
            example: 要移除的示例
            
        Returns:
            是否成功移除
        """
        try:
            examples_list = self.example_db.examples[example.example_type]
            if example in examples_list:
                examples_list.remove(example)
                self.example_db.total_examples -= 1
                return True
        except Exception as e:
            logger.error(f"Failed to remove example: {str(e)}")
        
        return False
    
    def clear_examples(self, example_type: Optional[ExampleType] = None) -> int:
        """
        清空示例
        
        Args:
            example_type: 要清空的示例类型，None表示清空所有
            
        Returns:
            清空的示例数量
        """
        if example_type:
            count = len(self.example_db.examples[example_type])
            self.example_db.examples[example_type].clear()
            self.example_db.total_examples -= count
            return count
        else:
            count = self.example_db.total_examples
            self.example_db.examples.clear()
            self.example_db.total_examples = 0
            return count
    
    def set_llm_template(self, llm_template) -> None:
        """设置LLM模板"""
        self.llm_template = llm_template
        logger.info("LLM template updated")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        db_stats = self.example_db.get_statistics()
        
        return {
            **db_stats,
            "usage_stats": self.usage_stats.copy(),
            "available_strategies": [s.value for s in SelectionStrategy],
            "available_types": [t.value for t in ExampleType]
        }
    
    def export_examples(self) -> Dict[str, Any]:
        """导出示例数据"""
        return self.example_db.export_examples()
    
    def import_examples(self, data: Dict[str, Any]) -> int:
        """导入示例数据"""
        return self.example_db.import_examples(data)
    
    def cleanup_low_quality_examples(self, min_quality: float = 0.3) -> int:
        """清理低质量示例"""
        return self.example_db.remove_low_quality_examples(min_quality)
    
    # 便捷方法
    def classify(self, text: str, max_examples: int = 3, **kwargs) -> str:
        """便捷的分类方法"""
        result = self.run(text, example_type="classification", max_examples=max_examples, **kwargs)
        return result.prediction
    
    def answer_question(self, question: str, max_examples: int = 3, **kwargs) -> str:
        """便捷的问答方法"""
        result = self.run(question, example_type="qa", max_examples=max_examples, **kwargs)
        return result.prediction
    
    def generate_text(self, prompt: str, max_examples: int = 3, **kwargs) -> str:
        """便捷的文本生成方法"""
        result = self.run(prompt, example_type="generation", max_examples=max_examples, **kwargs)
        return result.prediction