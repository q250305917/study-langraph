"""
准确性评估模板模块

本模块实现了完整的准确性评估系统，用于评估LangChain应用输出的质量和准确性。
支持多种评估指标、对比分析和自动化质量检测。

核心功能：
1. 输出质量评估 - 评估模型输出的准确性、相关性和完整性
2. 对比分析 - 比较不同模型或配置的性能差异
3. 参考答案匹配 - 与标准答案进行对比评估
4. 语义相似性检测 - 基于语义理解的准确性评估
5. A/B测试支持 - 支持多版本对比测试
6. 自动化评估 - 批量评估和持续监控

设计原理：
- 多维度评估：从准确性、相关性、完整性等多个维度评估
- 灵活的评估指标：支持自定义评估标准和权重
- 智能对比分析：自动检测和分析性能差异
- 结果可视化：生成详细的评估报告和图表
- 可扩展架构：支持添加新的评估指标和方法
"""

import json
import time
import asyncio
import hashlib
import statistics
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum
import threading
import re

try:
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    accuracy_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError, 
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class EvaluationMetric(Enum):
    """评估指标枚举"""
    ACCURACY = "accuracy"                    # 准确性
    PRECISION = "precision"                  # 精确度
    RECALL = "recall"                        # 召回率
    F1_SCORE = "f1_score"                   # F1分数
    SEMANTIC_SIMILARITY = "semantic_similarity"  # 语义相似性
    COSINE_SIMILARITY = "cosine_similarity"  # 余弦相似性
    BLEU_SCORE = "bleu_score"               # BLEU分数
    ROUGE_SCORE = "rouge_score"             # ROUGE分数
    PERPLEXITY = "perplexity"               # 困惑度
    COHERENCE = "coherence"                 # 连贯性
    RELEVANCE = "relevance"                 # 相关性
    COMPLETENESS = "completeness"           # 完整性
    CUSTOM = "custom"                       # 自定义指标


class EvaluationType(Enum):
    """评估类型枚举"""
    SINGLE = "single"                       # 单一输出评估
    COMPARISON = "comparison"               # 对比评估
    AB_TEST = "ab_test"                     # A/B测试
    BATCH = "batch"                         # 批量评估
    CONTINUOUS = "continuous"               # 持续评估


class ScoreLevel(Enum):
    """评分级别枚举"""
    EXCELLENT = "excellent"                 # 优秀 (90-100)
    GOOD = "good"                          # 良好 (80-89)
    AVERAGE = "average"                    # 一般 (70-79)
    POOR = "poor"                          # 较差 (60-69)
    VERY_POOR = "very_poor"                # 很差 (0-59)


@dataclass
class EvaluationResult:
    """
    评估结果数据结构
    
    表示一次评估的完整结果信息。
    """
    evaluation_id: str                      # 评估ID
    timestamp: float = field(default_factory=time.time)    # 评估时间戳
    evaluation_type: EvaluationType = EvaluationType.SINGLE
    
    # 输入数据
    input_text: str = ""                    # 输入文本
    actual_output: str = ""                 # 实际输出
    expected_output: Optional[str] = None   # 期望输出
    reference_answers: List[str] = field(default_factory=list)  # 参考答案
    
    # 评估结果
    scores: Dict[str, float] = field(default_factory=dict)      # 各项指标得分
    overall_score: float = 0.0              # 总体得分
    score_level: ScoreLevel = ScoreLevel.AVERAGE               # 评分级别
    
    # 详细分析
    strengths: List[str] = field(default_factory=list)         # 优点
    weaknesses: List[str] = field(default_factory=list)        # 缺点
    suggestions: List[str] = field(default_factory=list)       # 改进建议
    
    # 元数据
    model_name: Optional[str] = None        # 模型名称
    model_version: Optional[str] = None     # 模型版本
    config_hash: Optional[str] = None       # 配置哈希
    evaluation_duration: float = 0.0       # 评估耗时
    metadata: Dict[str, Any] = field(default_factory=dict)     # 附加元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['evaluation_type'] = self.evaluation_type.value
        data['score_level'] = self.score_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """从字典创建EvaluationResult实例"""
        if isinstance(data.get('evaluation_type'), str):
            data['evaluation_type'] = EvaluationType(data['evaluation_type'])
        if isinstance(data.get('score_level'), str):
            data['score_level'] = ScoreLevel(data['score_level'])
        return cls(**data)
    
    def get_score_summary(self) -> Dict[str, Any]:
        """获取得分摘要"""
        return {
            "overall_score": self.overall_score,
            "score_level": self.score_level.value,
            "top_scores": sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:3],
            "low_scores": sorted(self.scores.items(), key=lambda x: x[1])[:3]
        }
    
    def is_passed(self, threshold: float = 70.0) -> bool:
        """检查是否通过评估"""
        return self.overall_score >= threshold


@dataclass
class ComparisonResult:
    """
    对比评估结果数据结构
    
    表示两个或多个输出的对比评估结果。
    """
    comparison_id: str                      # 对比ID
    timestamp: float = field(default_factory=time.time)
    
    # 对比的输出
    outputs: List[Dict[str, Any]] = field(default_factory=list)  # 对比的输出列表
    
    # 对比结果
    winner: Optional[str] = None            # 获胜者ID
    scores_comparison: Dict[str, List[float]] = field(default_factory=dict)  # 分数对比
    statistical_significance: Dict[str, float] = field(default_factory=dict)  # 统计显著性
    
    # 分析结果
    key_differences: List[str] = field(default_factory=list)    # 关键差异
    performance_gaps: Dict[str, float] = field(default_factory=dict)  # 性能差距
    recommendations: List[str] = field(default_factory=list)   # 建议
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparisonResult':
        """从字典创建ComparisonResult实例"""
        return cls(**data)


class MetricCalculator(ABC):
    """
    指标计算器抽象基类
    
    定义各种评估指标的计算接口。
    """
    
    @abstractmethod
    def calculate(self, actual: str, expected: str, **kwargs) -> float:
        """计算指标得分"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取指标名称"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """获取指标描述"""
        pass


class SemanticSimilarityCalculator(MetricCalculator):
    """语义相似性计算器"""
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        """
        初始化语义相似性计算器
        
        Args:
            model_name: 句子转换模型名称
        """
        self.model_name = model_name
        self.model = None
        
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
                logger.debug(f"Loaded sentence transformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
    
    def calculate(self, actual: str, expected: str, **kwargs) -> float:
        """计算语义相似性得分"""
        if not self.model:
            # 回退到简单的词汇相似性
            return self._calculate_simple_similarity(actual, expected)
        
        try:
            # 使用句子转换器计算语义相似性
            embeddings = self.model.encode([actual, expected])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity) * 100  # 转换为百分制
            
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return self._calculate_simple_similarity(actual, expected)
    
    def _calculate_simple_similarity(self, actual: str, expected: str) -> float:
        """计算简单的词汇相似性"""
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        intersection = actual_words.intersection(expected_words)
        union = actual_words.union(expected_words)
        
        if not union:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity * 100
    
    def get_name(self) -> str:
        return "semantic_similarity"
    
    def get_description(self) -> str:
        return "语义相似性：使用句子转换器计算文本的语义相似度"


class CosineSimilarityCalculator(MetricCalculator):
    """余弦相似性计算器"""
    
    def __init__(self):
        """初始化余弦相似性计算器"""
        self.vectorizer = TfidfVectorizer() if TfidfVectorizer else None
    
    def calculate(self, actual: str, expected: str, **kwargs) -> float:
        """计算余弦相似性得分"""
        if not self.vectorizer:
            # 简单的字符相似性
            return self._calculate_char_similarity(actual, expected)
        
        try:
            # 使用TF-IDF向量化
            texts = [actual, expected]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity) * 100
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return self._calculate_char_similarity(actual, expected)
    
    def _calculate_char_similarity(self, actual: str, expected: str) -> float:
        """计算字符级相似性"""
        if not expected:
            return 0.0
        
        # 计算编辑距离的相似性
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(actual), len(expected))
        if max_len == 0:
            return 100.0
        
        distance = levenshtein_distance(actual, expected)
        similarity = (1 - distance / max_len) * 100
        return max(0.0, similarity)
    
    def get_name(self) -> str:
        return "cosine_similarity"
    
    def get_description(self) -> str:
        return "余弦相似性：使用TF-IDF向量的余弦相似度"


class RelevanceCalculator(MetricCalculator):
    """相关性计算器"""
    
    def calculate(self, actual: str, expected: str, **kwargs) -> float:
        """计算相关性得分"""
        input_text = kwargs.get("input_text", "")
        
        # 检查输出是否回答了输入问题
        actual_lower = actual.lower()
        input_lower = input_text.lower()
        expected_lower = expected.lower() if expected else ""
        
        # 关键词匹配得分
        input_keywords = self._extract_keywords(input_lower)
        actual_keywords = self._extract_keywords(actual_lower)
        expected_keywords = self._extract_keywords(expected_lower) if expected else set()
        
        # 计算与输入的相关性
        input_relevance = len(input_keywords.intersection(actual_keywords)) / len(input_keywords) if input_keywords else 0
        
        # 计算与期望输出的相关性
        expected_relevance = len(expected_keywords.intersection(actual_keywords)) / len(expected_keywords) if expected_keywords else 1
        
        # 综合得分
        relevance_score = (input_relevance * 0.6 + expected_relevance * 0.4) * 100
        
        return min(100.0, relevance_score)
    
    def _extract_keywords(self, text: str) -> set:
        """提取关键词"""
        # 移除停用词并提取重要词汇
        stop_words = {'是', '的', '在', '有', '和', '与', '或', '但', '如果', '那么', '这', '那', '什么', '怎么', '为什么',
                     'is', 'the', 'in', 'and', 'or', 'but', 'if', 'then', 'this', 'that', 'what', 'how', 'why', 'a', 'an'}
        
        words = re.findall(r'\b\w{2,}\b', text.lower())
        keywords = {word for word in words if word not in stop_words and len(word) > 2}
        
        return keywords
    
    def get_name(self) -> str:
        return "relevance"
    
    def get_description(self) -> str:
        return "相关性：评估输出与输入问题和期望答案的相关程度"


class CompletenessCalculator(MetricCalculator):
    """完整性计算器"""
    
    def calculate(self, actual: str, expected: str, **kwargs) -> float:
        """计算完整性得分"""
        if not expected:
            # 如果没有期望输出，基于输出长度和结构评估完整性
            return self._assess_structural_completeness(actual)
        
        # 检查实际输出是否包含期望输出的主要内容
        expected_points = self._extract_key_points(expected)
        actual_points = self._extract_key_points(actual)
        
        if not expected_points:
            return 100.0
        
        covered_points = 0
        for expected_point in expected_points:
            for actual_point in actual_points:
                if self._points_similar(expected_point, actual_point):
                    covered_points += 1
                    break
        
        completeness_score = (covered_points / len(expected_points)) * 100
        return completeness_score
    
    def _extract_key_points(self, text: str) -> List[str]:
        """提取关键要点"""
        # 简单的要点提取：基于句子分割
        sentences = re.split(r'[.!?。！？]', text)
        points = [sent.strip() for sent in sentences if len(sent.strip()) > 10]
        return points
    
    def _points_similar(self, point1: str, point2: str, threshold: float = 0.5) -> bool:
        """检查两个要点是否相似"""
        words1 = set(point1.lower().split())
        words2 = set(point2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold
    
    def _assess_structural_completeness(self, text: str) -> float:
        """评估结构完整性"""
        # 基于文本长度、句子数量、段落结构等评估完整性
        
        sentences = re.split(r'[.!?。！？]', text)
        sentence_count = len([s for s in sentences if len(s.strip()) > 5])
        
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if len(p.strip()) > 20])
        
        word_count = len(text.split())
        
        # 评估得分
        structure_score = 0
        
        # 句子数量得分 (20分)
        if sentence_count >= 3:
            structure_score += 20
        elif sentence_count >= 1:
            structure_score += 10
        
        # 段落结构得分 (20分)
        if paragraph_count >= 2:
            structure_score += 20
        elif paragraph_count >= 1:
            structure_score += 15
        
        # 词汇丰富度得分 (30分)
        if word_count >= 100:
            structure_score += 30
        elif word_count >= 50:
            structure_score += 20
        elif word_count >= 20:
            structure_score += 10
        
        # 内容组织得分 (30分)
        has_introduction = any(word in text.lower() for word in ['首先', '开始', '介绍', 'first', 'introduction'])
        has_conclusion = any(word in text.lower() for word in ['总结', '结论', '最后', 'conclusion', 'summary', 'finally'])
        has_examples = any(word in text.lower() for word in ['例如', '比如', '举例', 'example', 'for instance'])
        
        organization_score = 0
        if has_introduction:
            organization_score += 10
        if has_conclusion:
            organization_score += 10
        if has_examples:
            organization_score += 10
        
        structure_score += organization_score
        
        return min(100.0, structure_score)
    
    def get_name(self) -> str:
        return "completeness"
    
    def get_description(self) -> str:
        return "完整性：评估输出内容的完整程度和结构组织"


class AccuracyEvalTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    准确性评估模板
    
    提供全面的准确性评估功能，支持多种评估指标、对比分析和自动化质量检测。
    
    核心功能：
    1. 多指标评估：支持语义相似性、相关性、完整性等多种指标
    2. 对比分析：比较不同模型或配置的性能差异
    3. A/B测试：支持多版本对比测试
    4. 批量评估：大规模数据集的自动化评估
    5. 持续监控：实时性能监控和报警
    6. 详细报告：生成可视化的评估报告
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化准确性评估模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config)
        
        # 指标计算器
        self.metric_calculators: Dict[str, MetricCalculator] = {}
        
        # 评估结果存储
        self.evaluation_results: List[EvaluationResult] = []
        self.comparison_results: List[ComparisonResult] = []
        
        # 配置参数
        self.default_metrics: List[str] = ["semantic_similarity", "relevance", "completeness"]
        self.metric_weights: Dict[str, float] = {
            "semantic_similarity": 0.4,
            "relevance": 0.3,
            "completeness": 0.3
        }
        self.pass_threshold: float = 70.0
        self.storage_path: Optional[Path] = None
        
        # 线程锁
        self._lock = threading.Lock()
        
        logger.debug("Initialized AccuracyEvalTemplate")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="AccuracyEvalTemplate",
            description="准确性评估模板",
            template_type=TemplateType.EVALUATION,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("default_metrics", list, 
                           default=["semantic_similarity", "relevance", "completeness"],
                           description="默认评估指标")
        config.add_parameter("metric_weights", dict, 
                           default={"semantic_similarity": 0.4, "relevance": 0.3, "completeness": 0.3},
                           description="指标权重")
        config.add_parameter("pass_threshold", float, default=70.0,
                           description="通过阈值")
        config.add_parameter("storage_path", str, default="./evaluations",
                           description="评估结果存储路径")
        config.add_parameter("enable_visualization", bool, default=True,
                           description="是否启用可视化")
        config.add_parameter("semantic_model", str, default="paraphrase-MiniLM-L6-v2",
                           description="语义相似性模型")
        config.add_parameter("auto_save", bool, default=True,
                           description="是否自动保存结果")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置准确性评估模板
        
        Args:
            **parameters: 配置参数
                - default_metrics: 默认评估指标
                - metric_weights: 指标权重
                - pass_threshold: 通过阈值
                - storage_path: 存储路径
                - enable_visualization: 启用可视化
                - semantic_model: 语义模型
                - auto_save: 自动保存
        """
        # 验证参数
        if not self.validate_parameters(parameters):
            raise ValidationError("AccuracyEvalTemplate参数验证失败")
        
        # 更新内部参数
        self.default_metrics = parameters.get("default_metrics", 
                                             ["semantic_similarity", "relevance", "completeness"])
        self.metric_weights = parameters.get("metric_weights", {
            "semantic_similarity": 0.4,
            "relevance": 0.3,
            "completeness": 0.3
        })
        self.pass_threshold = parameters.get("pass_threshold", 70.0)
        
        # 设置存储路径
        storage_path = Path(parameters.get("storage_path", "./evaluations"))
        storage_path.mkdir(parents=True, exist_ok=True)
        self.storage_path = storage_path
        
        # 初始化指标计算器
        self._initialize_calculators(parameters)
        
        self.status = self.config.template_type.CONFIGURED
        logger.info(f"AccuracyEvalTemplate配置完成：metrics={self.default_metrics}")
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行准确性评估
        
        Args:
            input_data: 输入数据
                - action: 操作类型（evaluate, compare, batch_evaluate等）
                - input_text: 输入文本
                - actual_output: 实际输出
                - expected_output: 期望输出（可选）
                - reference_answers: 参考答案列表（可选）
                - metrics: 使用的评估指标（可选）
                - model_name: 模型名称（可选）
                
        Returns:
            评估结果字典
        """
        action = input_data.get("action", "evaluate")
        
        try:
            if action == "evaluate":
                return self._evaluate_single(input_data)
            elif action == "compare":
                return self._compare_outputs(input_data)
            elif action == "batch_evaluate":
                return self._batch_evaluate(input_data)
            elif action == "ab_test":
                return self._ab_test(input_data)
            elif action == "get_results":
                return self._get_results(input_data)
            elif action == "get_stats":
                return self._get_statistics()
            elif action == "generate_report":
                return self._generate_report(input_data)
            elif action == "save_results":
                return self._save_results()
            elif action == "load_results":
                return self._load_results()
            else:
                raise ValidationError(f"未知的操作类型：{action}")
                
        except Exception as e:
            logger.error(f"执行准确性评估失败：{action} - {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    def _evaluate_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """单一输出评估"""
        # 提取必需的输入数据
        input_text = input_data.get("input_text", "")
        actual_output = input_data.get("actual_output", "")
        expected_output = input_data.get("expected_output")
        reference_answers = input_data.get("reference_answers", [])
        
        if not actual_output:
            return {
                "success": False,
                "error": "实际输出不能为空"
            }
        
        # 选择评估指标
        metrics = input_data.get("metrics", self.default_metrics)
        
        # 生成评估ID
        evaluation_id = hashlib.md5(f"{input_text}_{actual_output}_{time.time()}".encode()).hexdigest()[:8]
        
        # 开始评估
        start_time = time.time()
        scores = {}
        
        # 计算各项指标
        for metric_name in metrics:
            if metric_name in self.metric_calculators:
                calculator = self.metric_calculators[metric_name]
                try:
                    # 选择最佳的期望输出
                    best_expected = self._select_best_reference(expected_output, reference_answers)
                    
                    score = calculator.calculate(
                        actual=actual_output,
                        expected=best_expected or "",
                        input_text=input_text
                    )
                    scores[metric_name] = score
                    logger.debug(f"Calculated {metric_name}: {score}")
                    
                except Exception as e:
                    logger.error(f"Failed to calculate {metric_name}: {e}")
                    scores[metric_name] = 0.0
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                scores[metric_name] = 0.0
        
        # 计算总体得分
        overall_score = self._calculate_overall_score(scores)
        
        # 确定得分级别
        score_level = self._determine_score_level(overall_score)
        
        # 生成分析结果
        strengths, weaknesses, suggestions = self._analyze_output(
            actual_output, expected_output or "", scores
        )
        
        # 创建评估结果
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_type=EvaluationType.SINGLE,
            input_text=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            reference_answers=reference_answers,
            scores=scores,
            overall_score=overall_score,
            score_level=score_level,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            model_name=input_data.get("model_name"),
            model_version=input_data.get("model_version"),
            evaluation_duration=time.time() - start_time
        )
        
        # 存储结果
        with self._lock:
            self.evaluation_results.append(result)
        
        # 自动保存
        if input_data.get("auto_save", True):
            self._save_single_result(result)
        
        return {
            "success": True,
            "evaluation_id": evaluation_id,
            "result": result.to_dict(),
            "passed": result.is_passed(self.pass_threshold)
        }
    
    def _compare_outputs(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """对比多个输出"""
        outputs = input_data.get("outputs", [])
        if len(outputs) < 2:
            return {
                "success": False,
                "error": "至少需要两个输出进行对比"
            }
        
        input_text = input_data.get("input_text", "")
        expected_output = input_data.get("expected_output")
        metrics = input_data.get("metrics", self.default_metrics)
        
        # 生成对比ID
        comparison_id = hashlib.md5(f"compare_{time.time()}".encode()).hexdigest()[:8]
        
        # 评估每个输出
        evaluated_outputs = []
        for i, output_data in enumerate(outputs):
            output_text = output_data.get("text", "")
            output_id = output_data.get("id", f"output_{i}")
            model_name = output_data.get("model_name", f"model_{i}")
            
            # 单独评估
            eval_result = self._evaluate_single({
                "input_text": input_text,
                "actual_output": output_text,
                "expected_output": expected_output,
                "metrics": metrics,
                "model_name": model_name,
                "auto_save": False
            })
            
            if eval_result.get("success"):
                evaluated_outputs.append({
                    "id": output_id,
                    "model_name": model_name,
                    "text": output_text,
                    "evaluation": eval_result["result"]
                })
        
        if not evaluated_outputs:
            return {
                "success": False,
                "error": "所有输出评估失败"
            }
        
        # 分析对比结果
        scores_comparison = {}
        for metric in metrics:
            scores_comparison[metric] = [
                output["evaluation"]["scores"].get(metric, 0.0)
                for output in evaluated_outputs
            ]
        
        # 确定获胜者
        overall_scores = [output["evaluation"]["overall_score"] for output in evaluated_outputs]
        winner_index = overall_scores.index(max(overall_scores))
        winner = evaluated_outputs[winner_index]["id"]
        
        # 计算性能差距
        performance_gaps = {}
        best_score = max(overall_scores)
        for i, output in enumerate(evaluated_outputs):
            gap = best_score - overall_scores[i]
            performance_gaps[output["id"]] = gap
        
        # 分析关键差异
        key_differences = self._analyze_differences(evaluated_outputs)
        
        # 生成建议
        recommendations = self._generate_comparison_recommendations(evaluated_outputs, key_differences)
        
        # 创建对比结果
        comparison_result = ComparisonResult(
            comparison_id=comparison_id,
            outputs=evaluated_outputs,
            winner=winner,
            scores_comparison=scores_comparison,
            key_differences=key_differences,
            performance_gaps=performance_gaps,
            recommendations=recommendations
        )
        
        # 存储结果
        with self._lock:
            self.comparison_results.append(comparison_result)
        
        return {
            "success": True,
            "comparison_id": comparison_id,
            "result": comparison_result.to_dict(),
            "winner": winner,
            "summary": {
                "total_outputs": len(evaluated_outputs),
                "best_score": best_score,
                "score_range": max(overall_scores) - min(overall_scores),
                "top_metric": max(scores_comparison, key=lambda k: max(scores_comparison[k]))
            }
        }
    
    def _batch_evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """批量评估"""
        test_cases = input_data.get("test_cases", [])
        if not test_cases:
            return {
                "success": False,
                "error": "测试用例不能为空"
            }
        
        metrics = input_data.get("metrics", self.default_metrics)
        model_name = input_data.get("model_name", "unknown")
        
        # 批量评估
        batch_results = []
        failed_cases = []
        
        for i, test_case in enumerate(test_cases):
            try:
                eval_input = {
                    "input_text": test_case.get("input", ""),
                    "actual_output": test_case.get("output", ""),
                    "expected_output": test_case.get("expected"),
                    "reference_answers": test_case.get("references", []),
                    "metrics": metrics,
                    "model_name": model_name,
                    "auto_save": False
                }
                
                result = self._evaluate_single(eval_input)
                if result.get("success"):
                    batch_results.append(result["result"])
                else:
                    failed_cases.append({"index": i, "error": result.get("error")})
                    
            except Exception as e:
                failed_cases.append({"index": i, "error": str(e)})
                logger.error(f"Batch evaluation failed for case {i}: {e}")
        
        if not batch_results:
            return {
                "success": False,
                "error": "所有测试用例评估失败",
                "failed_cases": failed_cases
            }
        
        # 计算批量统计
        batch_stats = self._calculate_batch_statistics(batch_results)
        
        return {
            "success": True,
            "total_cases": len(test_cases),
            "successful_cases": len(batch_results),
            "failed_cases": len(failed_cases),
            "results": batch_results,
            "statistics": batch_stats,
            "failures": failed_cases
        }
    
    def _ab_test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """A/B测试"""
        version_a = input_data.get("version_a", {})
        version_b = input_data.get("version_b", {})
        test_cases = input_data.get("test_cases", [])
        
        if not version_a or not version_b or not test_cases:
            return {
                "success": False,
                "error": "A/B测试需要两个版本和测试用例"
            }
        
        # 分别评估两个版本
        results_a = []
        results_b = []
        
        for test_case in test_cases:
            # 评估版本A
            if "outputs_a" in test_case:
                eval_a = self._evaluate_single({
                    "input_text": test_case.get("input", ""),
                    "actual_output": test_case["outputs_a"],
                    "expected_output": test_case.get("expected"),
                    "model_name": version_a.get("name", "version_a"),
                    "auto_save": False
                })
                if eval_a.get("success"):
                    results_a.append(eval_a["result"])
            
            # 评估版本B
            if "outputs_b" in test_case:
                eval_b = self._evaluate_single({
                    "input_text": test_case.get("input", ""),
                    "actual_output": test_case["outputs_b"],
                    "expected_output": test_case.get("expected"),
                    "model_name": version_b.get("name", "version_b"),
                    "auto_save": False
                })
                if eval_b.get("success"):
                    results_b.append(eval_b["result"])
        
        # 统计分析
        if not results_a or not results_b:
            return {
                "success": False,
                "error": "A/B测试数据不足"
            }
        
        stats_a = self._calculate_batch_statistics(results_a)
        stats_b = self._calculate_batch_statistics(results_b)
        
        # 显著性检验
        significance_test = self._perform_significance_test(results_a, results_b)
        
        # 确定获胜者
        winner = "version_a" if stats_a["average_score"] > stats_b["average_score"] else "version_b"
        confidence_level = significance_test.get("confidence_level", 0.0)
        
        return {
            "success": True,
            "winner": winner,
            "confidence_level": confidence_level,
            "version_a_stats": stats_a,
            "version_b_stats": stats_b,
            "significance_test": significance_test,
            "recommendation": self._generate_ab_test_recommendation(
                winner, confidence_level, stats_a, stats_b
            )
        }
    
    def _get_results(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取评估结果"""
        limit = input_data.get("limit", 100)
        model_name = input_data.get("model_name")
        start_time = input_data.get("start_time")
        end_time = input_data.get("end_time")
        
        # 过滤结果
        filtered_results = []
        
        with self._lock:
            for result in self.evaluation_results[-limit:]:
                # 模型名称过滤
                if model_name and result.model_name != model_name:
                    continue
                
                # 时间范围过滤
                if start_time and result.timestamp < start_time:
                    continue
                if end_time and result.timestamp > end_time:
                    continue
                
                filtered_results.append(result.to_dict())
        
        return {
            "success": True,
            "results": filtered_results,
            "total_count": len(filtered_results)
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            if not self.evaluation_results:
                return {
                    "success": True,
                    "statistics": {
                        "total_evaluations": 0,
                        "total_comparisons": 0
                    }
                }
            
            # 计算基本统计
            total_evaluations = len(self.evaluation_results)
            total_comparisons = len(self.comparison_results)
            
            # 得分统计
            all_scores = [result.overall_score for result in self.evaluation_results]
            avg_score = statistics.mean(all_scores)
            median_score = statistics.median(all_scores)
            score_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
            
            # 通过率
            passed_count = sum(1 for result in self.evaluation_results 
                             if result.is_passed(self.pass_threshold))
            pass_rate = (passed_count / total_evaluations) * 100 if total_evaluations > 0 else 0
            
            # 指标统计
            metric_stats = {}
            for metric in self.default_metrics:
                metric_scores = [
                    result.scores.get(metric, 0) 
                    for result in self.evaluation_results 
                    if metric in result.scores
                ]
                if metric_scores:
                    metric_stats[metric] = {
                        "average": statistics.mean(metric_scores),
                        "median": statistics.median(metric_scores),
                        "min": min(metric_scores),
                        "max": max(metric_scores),
                        "std": statistics.stdev(metric_scores) if len(metric_scores) > 1 else 0.0
                    }
            
            return {
                "success": True,
                "statistics": {
                    "total_evaluations": total_evaluations,
                    "total_comparisons": total_comparisons,
                    "average_score": avg_score,
                    "median_score": median_score,
                    "score_std": score_std,
                    "pass_rate": pass_rate,
                    "pass_threshold": self.pass_threshold,
                    "metric_statistics": metric_stats
                }
            }
    
    def _select_best_reference(self, expected: Optional[str], references: List[str]) -> Optional[str]:
        """选择最佳参考答案"""
        if expected:
            return expected
        
        if not references:
            return None
        
        if len(references) == 1:
            return references[0]
        
        # 选择最长的参考答案（假设更详细）
        return max(references, key=len)
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """计算总体得分"""
        if not scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = self.metric_weights.get(metric, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_score_level(self, score: float) -> ScoreLevel:
        """确定得分级别"""
        if score >= 90:
            return ScoreLevel.EXCELLENT
        elif score >= 80:
            return ScoreLevel.GOOD
        elif score >= 70:
            return ScoreLevel.AVERAGE
        elif score >= 60:
            return ScoreLevel.POOR
        else:
            return ScoreLevel.VERY_POOR
    
    def _analyze_output(self, actual: str, expected: str, scores: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]:
        """分析输出质量"""
        strengths = []
        weaknesses = []
        suggestions = []
        
        # 基于得分分析
        for metric, score in scores.items():
            if score >= 85:
                strengths.append(f"{metric}表现优秀（{score:.1f}分）")
            elif score < 60:
                weaknesses.append(f"{metric}需要改进（{score:.1f}分）")
                
                # 提供具体建议
                if metric == "semantic_similarity":
                    suggestions.append("建议提高输出与期望答案的语义相似性")
                elif metric == "relevance":
                    suggestions.append("建议确保输出更好地回答输入问题")
                elif metric == "completeness":
                    suggestions.append("建议提供更完整和详细的回答")
        
        # 内容分析
        if len(actual) < 50:
            weaknesses.append("输出过于简短")
            suggestions.append("建议提供更详细的回答")
        
        if len(actual.split()) < 10:
            weaknesses.append("词汇量不足")
            suggestions.append("建议使用更丰富的词汇表达")
        
        # 结构分析
        sentences = len([s for s in actual.split('.') if len(s.strip()) > 5])
        if sentences < 2:
            weaknesses.append("句子结构过于简单")
            suggestions.append("建议使用多个句子组织内容")
        
        return strengths, weaknesses, suggestions
    
    def _analyze_differences(self, evaluated_outputs: List[Dict[str, Any]]) -> List[str]:
        """分析输出差异"""
        differences = []
        
        if len(evaluated_outputs) < 2:
            return differences
        
        # 得分差异分析
        scores = [output["evaluation"]["overall_score"] for output in evaluated_outputs]
        score_range = max(scores) - min(scores)
        
        if score_range > 20:
            differences.append(f"整体得分差异较大（差距{score_range:.1f}分）")
        
        # 长度差异分析
        lengths = [len(output["text"]) for output in evaluated_outputs]
        length_range = max(lengths) - min(lengths)
        
        if length_range > 100:
            differences.append(f"输出长度差异明显（差距{length_range}字符）")
        
        # 指标差异分析
        for metric in self.default_metrics:
            metric_scores = [
                output["evaluation"]["scores"].get(metric, 0)
                for output in evaluated_outputs
            ]
            metric_range = max(metric_scores) - min(metric_scores)
            
            if metric_range > 15:
                differences.append(f"{metric}指标差异显著（差距{metric_range:.1f}分）")
        
        return differences
    
    def _generate_comparison_recommendations(self, evaluated_outputs: List[Dict[str, Any]], 
                                           differences: List[str]) -> List[str]:
        """生成对比建议"""
        recommendations = []
        
        if not evaluated_outputs:
            return recommendations
        
        # 找出最佳输出
        best_output = max(evaluated_outputs, key=lambda x: x["evaluation"]["overall_score"])
        worst_output = min(evaluated_outputs, key=lambda x: x["evaluation"]["overall_score"])
        
        score_gap = best_output["evaluation"]["overall_score"] - worst_output["evaluation"]["overall_score"]
        
        if score_gap > 10:
            recommendations.append(f"建议采用{best_output['model_name']}的方法，其表现显著优于其他选项")
        
        # 基于具体指标的建议
        for metric in self.default_metrics:
            metric_scores = [(output["id"], output["evaluation"]["scores"].get(metric, 0)) 
                           for output in evaluated_outputs]
            best_metric = max(metric_scores, key=lambda x: x[1])
            
            if best_metric[1] > 80:
                recommendations.append(f"在{metric}方面，{best_metric[0]}表现最佳，可作为参考")
        
        return recommendations
    
    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算批量统计"""
        if not results:
            return {}
        
        scores = [result["overall_score"] for result in results]
        
        stats = {
            "count": len(results),
            "average_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "pass_rate": sum(1 for score in scores if score >= self.pass_threshold) / len(scores) * 100
        }
        
        # 指标统计
        for metric in self.default_metrics:
            metric_scores = [result["scores"].get(metric, 0) for result in results if metric in result["scores"]]
            if metric_scores:
                stats[f"{metric}_avg"] = statistics.mean(metric_scores)
                stats[f"{metric}_std"] = statistics.stdev(metric_scores) if len(metric_scores) > 1 else 0.0
        
        return stats
    
    def _perform_significance_test(self, results_a: List[Dict[str, Any]], 
                                 results_b: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行显著性检验"""
        scores_a = [result["overall_score"] for result in results_a]
        scores_b = [result["overall_score"] for result in results_b]
        
        # 简单的t检验近似
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0
        
        # 计算效应大小
        pooled_std = ((std_a ** 2 + std_b ** 2) / 2) ** 0.5
        effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # 简化的置信度估计
        if effect_size > 0.8:
            confidence_level = 95.0
        elif effect_size > 0.5:
            confidence_level = 80.0
        elif effect_size > 0.2:
            confidence_level = 60.0
        else:
            confidence_level = 30.0
        
        return {
            "mean_difference": mean_a - mean_b,
            "effect_size": effect_size,
            "confidence_level": confidence_level,
            "sample_size_a": len(scores_a),
            "sample_size_b": len(scores_b)
        }
    
    def _generate_ab_test_recommendation(self, winner: str, confidence: float, 
                                       stats_a: Dict[str, Any], stats_b: Dict[str, Any]) -> str:
        """生成A/B测试建议"""
        if confidence > 90:
            return f"强烈建议采用{winner}，置信度为{confidence:.1f}%"
        elif confidence > 70:
            return f"建议采用{winner}，但需要更多数据验证（置信度{confidence:.1f}%）"
        else:
            return f"两个版本差异不显著，建议增加测试样本（置信度{confidence:.1f}%）"
    
    def _generate_report(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成评估报告"""
        # 这里可以实现详细的报告生成逻辑
        # 包括图表、统计分析、趋势分析等
        pass
    
    def _save_results(self) -> Dict[str, Any]:
        """保存所有结果"""
        if not self.storage_path:
            return {
                "success": False,
                "error": "未配置存储路径"
            }
        
        try:
            # 保存评估结果
            eval_file = self.storage_path / "evaluation_results.json"
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump([result.to_dict() for result in self.evaluation_results], 
                         f, ensure_ascii=False, indent=2)
            
            # 保存对比结果
            comp_file = self.storage_path / "comparison_results.json"
            with open(comp_file, 'w', encoding='utf-8') as f:
                json.dump([result.to_dict() for result in self.comparison_results], 
                         f, ensure_ascii=False, indent=2)
            
            return {
                "success": True,
                "evaluation_file": str(eval_file),
                "comparison_file": str(comp_file),
                "evaluation_count": len(self.evaluation_results),
                "comparison_count": len(self.comparison_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_single_result(self, result: EvaluationResult) -> None:
        """保存单个评估结果"""
        if not self.storage_path:
            return
        
        try:
            result_file = self.storage_path / f"eval_{result.evaluation_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save single result: {e}")
    
    def _load_results(self) -> Dict[str, Any]:
        """加载保存的结果"""
        if not self.storage_path:
            return {
                "success": False,
                "error": "未配置存储路径"
            }
        
        try:
            loaded_count = 0
            
            # 加载评估结果
            eval_file = self.storage_path / "evaluation_results.json"
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
                
                with self._lock:
                    self.evaluation_results = [EvaluationResult.from_dict(data) for data in eval_data]
                    loaded_count += len(self.evaluation_results)
            
            # 加载对比结果
            comp_file = self.storage_path / "comparison_results.json"
            if comp_file.exists():
                with open(comp_file, 'r', encoding='utf-8') as f:
                    comp_data = json.load(f)
                
                with self._lock:
                    self.comparison_results = [ComparisonResult.from_dict(data) for data in comp_data]
                    loaded_count += len(self.comparison_results)
            
            return {
                "success": True,
                "loaded_count": loaded_count,
                "evaluation_count": len(self.evaluation_results),
                "comparison_count": len(self.comparison_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _initialize_calculators(self, parameters: Dict[str, Any]) -> None:
        """初始化指标计算器"""
        semantic_model = parameters.get("semantic_model", "paraphrase-MiniLM-L6-v2")
        
        # 初始化各种计算器
        self.metric_calculators = {
            "semantic_similarity": SemanticSimilarityCalculator(semantic_model),
            "cosine_similarity": CosineSimilarityCalculator(),
            "relevance": RelevanceCalculator(),
            "completeness": CompletenessCalculator()
        }
        
        logger.debug(f"Initialized {len(self.metric_calculators)} metric calculators")
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "default_metrics": ["semantic_similarity", "relevance", "completeness"],
                "metric_weights": {
                    "semantic_similarity": 0.4,
                    "relevance": 0.3,
                    "completeness": 0.3
                },
                "pass_threshold": 70.0,
                "storage_path": "./evaluations",
                "enable_visualization": True,
                "semantic_model": "paraphrase-MiniLM-L6-v2",
                "auto_save": True
            },
            "execute_examples": [
                {
                    "description": "单一输出评估",
                    "input": {
                        "action": "evaluate",
                        "input_text": "什么是机器学习？",
                        "actual_output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。",
                        "expected_output": "机器学习是一种人工智能技术，通过算法让计算机从数据中学习并做出预测。",
                        "model_name": "gpt-3.5-turbo"
                    }
                },
                {
                    "description": "对比评估",
                    "input": {
                        "action": "compare",
                        "input_text": "解释深度学习的概念",
                        "outputs": [
                            {
                                "id": "model_a",
                                "text": "深度学习是机器学习的子集，使用神经网络。",
                                "model_name": "model_a"
                            },
                            {
                                "id": "model_b", 
                                "text": "深度学习是一种模仿人脑神经网络结构的机器学习方法，通过多层神经网络进行特征学习和模式识别。",
                                "model_name": "model_b"
                            }
                        ],
                        "expected_output": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂模式。"
                    }
                },
                {
                    "description": "批量评估",
                    "input": {
                        "action": "batch_evaluate",
                        "test_cases": [
                            {
                                "input": "什么是Python？",
                                "output": "Python是一种编程语言。",
                                "expected": "Python是一种高级编程语言，以其简洁和可读性著称。"
                            },
                            {
                                "input": "解释变量的概念",
                                "output": "变量是存储数据的容器。",
                                "expected": "变量是编程中用于存储和引用数据值的标识符。"
                            }
                        ],
                        "model_name": "test_model"
                    }
                }
            ],
            "usage_code": """
# 使用示例
from templates.evaluation.accuracy_eval import AccuracyEvalTemplate

# 初始化评估模板
eval_template = AccuracyEvalTemplate()

# 配置参数
eval_template.setup(
    default_metrics=["semantic_similarity", "relevance", "completeness"],
    metric_weights={
        "semantic_similarity": 0.4,
        "relevance": 0.3,
        "completeness": 0.3
    },
    pass_threshold=70.0,
    storage_path="./evaluations",
    auto_save=True
)

# 单一评估
result = eval_template.run({
    "action": "evaluate",
    "input_text": "什么是机器学习？",
    "actual_output": "机器学习是人工智能的一个分支...",
    "expected_output": "机器学习是一种人工智能技术...",
    "model_name": "gpt-3.5-turbo"
})

# 对比评估
comparison = eval_template.run({
    "action": "compare",
    "input_text": "解释深度学习",
    "outputs": [
        {"id": "model_a", "text": "深度学习是...", "model_name": "model_a"},
        {"id": "model_b", "text": "深度学习是一种...", "model_name": "model_b"}
    ]
})

print("评估结果：", result)
print("对比结果：", comparison)
"""
        }