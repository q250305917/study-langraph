"""
准确性评估模板测试

测试AccuracyEvalTemplate的各项功能，包括：
1. 基本配置和初始化
2. 单一输出评估
3. 对比评估
4. 批量评估
5. A/B测试
6. 各种指标计算器
7. 错误处理
"""

import unittest
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from templates.evaluation.accuracy_eval import (
    AccuracyEvalTemplate,
    EvaluationResult,
    ComparisonResult,
    EvaluationMetric,
    EvaluationType,
    ScoreLevel,
    SemanticSimilarityCalculator,
    CosineSimilarityCalculator,
    RelevanceCalculator,
    CompletenessCalculator
)
from templates.base.template_base import TemplateConfig, TemplateType


class TestAccuracyEvalTemplate(unittest.TestCase):
    """准确性评估模板测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.template = AccuracyEvalTemplate()
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_template_initialization(self):
        """测试模板初始化"""
        # 测试默认初始化
        template = AccuracyEvalTemplate()
        self.assertIsNotNone(template.config)
        self.assertEqual(template.config.name, "AccuracyEvalTemplate")
        self.assertEqual(template.config.template_type, TemplateType.EVALUATION)
        
        # 测试自定义配置初始化
        config = TemplateConfig(
            name="CustomAccuracyEval",
            description="自定义准确性评估",
            template_type=TemplateType.EVALUATION
        )
        template = AccuracyEvalTemplate(config)
        self.assertEqual(template.config.name, "CustomAccuracyEval")
    
    def test_template_setup(self):
        """测试模板设置"""
        storage_path = Path(self.temp_dir) / "evaluations"
        
        self.template.setup(
            default_metrics=["semantic_similarity", "relevance"],
            metric_weights={"semantic_similarity": 0.6, "relevance": 0.4},
            pass_threshold=80.0,
            storage_path=str(storage_path),
            auto_save=False
        )
        
        self.assertEqual(self.template.default_metrics, ["semantic_similarity", "relevance"])
        self.assertEqual(self.template.metric_weights["semantic_similarity"], 0.6)
        self.assertEqual(self.template.pass_threshold, 80.0)
        self.assertTrue(storage_path.exists())
        self.assertGreater(len(self.template.metric_calculators), 0)
    
    def test_single_evaluation_basic(self):
        """测试基本单一评估"""
        self.template.setup(auto_save=False)
        
        result = self.template.execute({
            "action": "evaluate",
            "input_text": "什么是机器学习？",
            "actual_output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。",
            "expected_output": "机器学习是一种人工智能技术，通过算法让计算机从数据中学习并做出预测。",
            "model_name": "test_model"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("evaluation_id", result)
        self.assertIn("result", result)
        self.assertIn("passed", result)
        
        # 验证评估结果结构
        eval_result = result["result"]
        self.assertIn("scores", eval_result)
        self.assertIn("overall_score", eval_result)
        self.assertIn("score_level", eval_result)
        self.assertIn("strengths", eval_result)
        self.assertIn("weaknesses", eval_result)
        self.assertIn("suggestions", eval_result)
    
    def test_single_evaluation_without_expected(self):
        """测试没有期望输出的单一评估"""
        self.template.setup(auto_save=False)
        
        result = self.template.execute({
            "action": "evaluate",
            "input_text": "解释深度学习",
            "actual_output": "深度学习是机器学习的一个子集，使用神经网络。"
        })
        
        self.assertTrue(result["success"])
        # 即使没有期望输出，也应该能进行评估
        eval_result = result["result"]
        self.assertGreaterEqual(eval_result["overall_score"], 0)
    
    def test_single_evaluation_with_reference_answers(self):
        """测试带参考答案的单一评估"""
        self.template.setup(auto_save=False)
        
        result = self.template.execute({
            "action": "evaluate",
            "input_text": "什么是Python？",
            "actual_output": "Python是一种编程语言。",
            "reference_answers": [
                "Python是一种高级编程语言。",
                "Python是一种解释型编程语言，以其简洁和可读性著称。"
            ]
        })
        
        self.assertTrue(result["success"])
        eval_result = result["result"]
        self.assertIn("reference_answers", eval_result)
        self.assertEqual(len(eval_result["reference_answers"]), 2)
    
    def test_comparison_evaluation(self):
        """测试对比评估"""
        self.template.setup(auto_save=False)
        
        result = self.template.execute({
            "action": "compare",
            "input_text": "解释机器学习的概念",
            "outputs": [
                {
                    "id": "output_a",
                    "text": "机器学习是AI的一部分，让计算机学习。",
                    "model_name": "model_a"
                },
                {
                    "id": "output_b",
                    "text": "机器学习是人工智能的一个分支，它使计算机能够通过数据学习并做出预测，而无需明确编程每个任务。",
                    "model_name": "model_b"
                }
            ],
            "expected_output": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("comparison_id", result)
        self.assertIn("result", result)
        self.assertIn("winner", result)
        self.assertIn("summary", result)
        
        # 验证对比结果结构
        comparison_result = result["result"]
        self.assertIn("outputs", comparison_result)
        self.assertIn("scores_comparison", comparison_result)
        self.assertIn("key_differences", comparison_result)
        self.assertIn("recommendations", comparison_result)
        
        # 验证获胜者
        self.assertIn(result["winner"], ["output_a", "output_b"])
    
    def test_batch_evaluation(self):
        """测试批量评估"""
        self.template.setup(auto_save=False)
        
        test_cases = [
            {
                "input": "什么是Python？",
                "output": "Python是一种编程语言。",
                "expected": "Python是一种高级编程语言。"
            },
            {
                "input": "解释变量的概念",
                "output": "变量是存储数据的容器。",
                "expected": "变量是编程中用于存储和引用数据值的标识符。"
            },
            {
                "input": "什么是函数？",
                "output": "函数是可重用的代码块。",
                "expected": "函数是执行特定任务的可重用代码块。"
            }
        ]
        
        result = self.template.execute({
            "action": "batch_evaluate",
            "test_cases": test_cases,
            "model_name": "batch_test_model"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["total_cases"], 3)
        self.assertEqual(result["successful_cases"], 3)
        self.assertEqual(result["failed_cases"], 0)
        self.assertIn("results", result)
        self.assertIn("statistics", result)
        
        # 验证统计信息
        stats = result["statistics"]
        self.assertIn("average_score", stats)
        self.assertIn("pass_rate", stats)
    
    def test_ab_test(self):
        """测试A/B测试"""
        self.template.setup(auto_save=False)
        
        test_cases = [
            {
                "input": "什么是AI？",
                "outputs_a": "AI是人工智能。",
                "outputs_b": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "expected": "人工智能是模拟人类智能的计算机系统。"
            },
            {
                "input": "解释机器学习",
                "outputs_a": "机器学习让计算机学习。",
                "outputs_b": "机器学习是人工智能的一个子集，它使计算机能够从数据中学习并改进性能。",
                "expected": "机器学习是AI的一部分，让计算机从数据中学习。"
            }
        ]
        
        result = self.template.execute({
            "action": "ab_test",
            "version_a": {"name": "version_a"},
            "version_b": {"name": "version_b"},
            "test_cases": test_cases
        })
        
        self.assertTrue(result["success"])
        self.assertIn("winner", result)
        self.assertIn("confidence_level", result)
        self.assertIn("version_a_stats", result)
        self.assertIn("version_b_stats", result)
        self.assertIn("recommendation", result)
        
        # 验证获胜者
        self.assertIn(result["winner"], ["version_a", "version_b"])
        self.assertGreaterEqual(result["confidence_level"], 0)
        self.assertLessEqual(result["confidence_level"], 100)
    
    def test_get_results(self):
        """测试获取评估结果"""
        self.template.setup(auto_save=False)
        
        # 先进行一些评估
        for i in range(3):
            self.template.execute({
                "action": "evaluate",
                "input_text": f"测试问题{i}",
                "actual_output": f"测试答案{i}",
                "model_name": "test_model"
            })
        
        # 获取结果
        result = self.template.execute({
            "action": "get_results",
            "limit": 2
        })
        
        self.assertTrue(result["success"])
        self.assertIn("results", result)
        self.assertLessEqual(len(result["results"]), 2)
        self.assertEqual(result["total_count"], len(result["results"]))
    
    def test_get_results_with_filters(self):
        """测试带过滤条件的结果获取"""
        self.template.setup(auto_save=False)
        
        # 添加不同模型的评估
        models = ["model_a", "model_b"]
        for model in models:
            self.template.execute({
                "action": "evaluate",
                "input_text": f"测试{model}",
                "actual_output": f"答案{model}",
                "model_name": model
            })
        
        # 过滤特定模型的结果
        result = self.template.execute({
            "action": "get_results",
            "model_name": "model_a"
        })
        
        self.assertTrue(result["success"])
        # 验证过滤效果
        for eval_result in result["results"]:
            self.assertEqual(eval_result["model_name"], "model_a")
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        self.template.setup(auto_save=False)
        
        # 先进行一些评估
        for i in range(5):
            self.template.execute({
                "action": "evaluate",
                "input_text": f"问题{i}",
                "actual_output": f"答案{i}",
                "expected_output": f"期望答案{i}"
            })
        
        result = self.template.execute({
            "action": "get_statistics"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("statistics", result)
        stats = result["statistics"]
        
        self.assertEqual(stats["total_evaluations"], 5)
        self.assertIn("average_score", stats)
        self.assertIn("median_score", stats)
        self.assertIn("pass_rate", stats)
        self.assertIn("metric_statistics", stats)
    
    def test_save_and_load_results(self):
        """测试保存和加载结果"""
        storage_path = Path(self.temp_dir) / "eval_storage"
        self.template.setup(storage_path=str(storage_path), auto_save=False)
        
        # 进行评估
        self.template.execute({
            "action": "evaluate",
            "input_text": "测试保存",
            "actual_output": "测试输出"
        })
        
        # 保存结果
        save_result = self.template.execute({
            "action": "save_results"
        })
        
        self.assertTrue(save_result["success"])
        self.assertIn("evaluation_file", save_result)
        self.assertGreater(save_result["evaluation_count"], 0)
        
        # 验证文件存在
        eval_file = Path(save_result["evaluation_file"])
        self.assertTrue(eval_file.exists())
        
        # 清除内存数据
        self.template.evaluation_results.clear()
        
        # 加载结果
        load_result = self.template.execute({
            "action": "load_results"
        })
        
        self.assertTrue(load_result["success"])
        self.assertGreater(load_result["loaded_count"], 0)
        self.assertGreater(load_result["evaluation_count"], 0)
    
    def test_error_handling_invalid_action(self):
        """测试无效操作的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "invalid_action"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("未知的操作类型", result["error"])
    
    def test_error_handling_missing_output(self):
        """测试缺少输出的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "evaluate",
            "input_text": "测试问题"
            # 缺少actual_output
        })
        
        self.assertFalse(result["success"])
        self.assertIn("实际输出不能为空", result["error"])
    
    def test_error_handling_insufficient_outputs_for_comparison(self):
        """测试对比评估输出不足的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "compare",
            "input_text": "测试",
            "outputs": [
                {"id": "only_one", "text": "只有一个输出"}
            ]
        })
        
        self.assertFalse(result["success"])
        self.assertIn("至少需要两个输出进行对比", result["error"])
    
    def test_different_metrics(self):
        """测试不同评估指标"""
        metrics = ["semantic_similarity", "cosine_similarity", "relevance", "completeness"]
        
        for metric in metrics:
            with self.subTest(metric=metric):
                template = AccuracyEvalTemplate()
                template.setup(default_metrics=[metric], auto_save=False)
                
                result = template.execute({
                    "action": "evaluate",
                    "input_text": "测试问题",
                    "actual_output": "这是一个测试答案，用来验证不同的评估指标。",
                    "expected_output": "这是期望的答案。"
                })
                
                self.assertTrue(result["success"])
                eval_result = result["result"]
                self.assertIn(metric, eval_result["scores"])
                self.assertGreaterEqual(eval_result["scores"][metric], 0)
                self.assertLessEqual(eval_result["scores"][metric], 100)
    
    def test_custom_metric_weights(self):
        """测试自定义指标权重"""
        # 设置不同的权重
        weights = {"semantic_similarity": 0.8, "relevance": 0.2}
        
        self.template.setup(
            default_metrics=list(weights.keys()),
            metric_weights=weights,
            auto_save=False
        )
        
        result = self.template.execute({
            "action": "evaluate",
            "input_text": "测试权重",
            "actual_output": "测试输出",
            "expected_output": "期望输出"
        })
        
        self.assertTrue(result["success"])
        # 验证权重被正确应用
        self.assertEqual(self.template.metric_weights, weights)
    
    def test_score_level_determination(self):
        """测试得分级别判定"""
        # 创建模拟的评估结果来测试不同得分级别
        from templates.evaluation.accuracy_eval import EvaluationResult, ScoreLevel
        
        test_scores = [
            (95, ScoreLevel.EXCELLENT),
            (85, ScoreLevel.GOOD),
            (75, ScoreLevel.AVERAGE),
            (65, ScoreLevel.POOR),
            (45, ScoreLevel.VERY_POOR)
        ]
        
        for score, expected_level in test_scores:
            with self.subTest(score=score):
                level = self.template._determine_score_level(score)
                self.assertEqual(level, expected_level)


class TestEvaluationResult(unittest.TestCase):
    """评估结果测试类"""
    
    def test_evaluation_result_creation(self):
        """测试评估结果创建"""
        result = EvaluationResult(
            evaluation_id="test_eval",
            input_text="测试输入",
            actual_output="实际输出",
            expected_output="期望输出",
            scores={"similarity": 85.0, "relevance": 90.0},
            overall_score=87.5
        )
        
        self.assertEqual(result.evaluation_id, "test_eval")
        self.assertEqual(result.input_text, "测试输入")
        self.assertEqual(result.scores["similarity"], 85.0)
        self.assertEqual(result.overall_score, 87.5)
    
    def test_score_summary(self):
        """测试得分摘要"""
        result = EvaluationResult(
            evaluation_id="summary_test",
            scores={
                "metric_a": 95.0,
                "metric_b": 75.0,
                "metric_c": 85.0,
                "metric_d": 65.0
            },
            overall_score=80.0,
            score_level=ScoreLevel.GOOD
        )
        
        summary = result.get_score_summary()
        
        self.assertEqual(summary["overall_score"], 80.0)
        self.assertEqual(summary["score_level"], "good")
        self.assertEqual(len(summary["top_scores"]), 3)
        self.assertEqual(len(summary["low_scores"]), 3)
        
        # 验证排序
        self.assertEqual(summary["top_scores"][0][0], "metric_a")  # 最高分
        self.assertEqual(summary["low_scores"][0][0], "metric_d")  # 最低分
    
    def test_is_passed(self):
        """测试通过判定"""
        result = EvaluationResult(
            evaluation_id="pass_test",
            overall_score=75.0
        )
        
        self.assertTrue(result.is_passed(70.0))   # 阈值70，得分75，应该通过
        self.assertFalse(result.is_passed(80.0))  # 阈值80，得分75，应该不通过
    
    def test_serialization(self):
        """测试序列化和反序列化"""
        result = EvaluationResult(
            evaluation_id="serialize_test",
            evaluation_type=EvaluationType.COMPARISON,
            scores={"test_metric": 80.0},
            overall_score=80.0,
            score_level=ScoreLevel.GOOD
        )
        
        # 序列化
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["evaluation_id"], "serialize_test")
        self.assertEqual(result_dict["evaluation_type"], "comparison")
        self.assertEqual(result_dict["score_level"], "good")
        
        # 反序列化
        restored_result = EvaluationResult.from_dict(result_dict)
        self.assertEqual(restored_result.evaluation_id, result.evaluation_id)
        self.assertEqual(restored_result.evaluation_type, result.evaluation_type)
        self.assertEqual(restored_result.score_level, result.score_level)


class TestMetricCalculators(unittest.TestCase):
    """指标计算器测试类"""
    
    def test_semantic_similarity_calculator(self):
        """测试语义相似性计算器"""
        calculator = SemanticSimilarityCalculator()
        
        # 测试相似文本
        score1 = calculator.calculate(
            "机器学习是人工智能的一个分支",
            "机器学习是AI的一部分"
        )
        
        # 测试不相似文本
        score2 = calculator.calculate(
            "机器学习是人工智能的一个分支",
            "今天天气很好"
        )
        
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 100)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 100)
        self.assertGreater(score1, score2)  # 相似文本得分应该更高
        
        # 测试计算器信息
        self.assertEqual(calculator.get_name(), "semantic_similarity")
        self.assertIsInstance(calculator.get_description(), str)
    
    def test_cosine_similarity_calculator(self):
        """测试余弦相似性计算器"""
        calculator = CosineSimilarityCalculator()
        
        score = calculator.calculate(
            "Python是一种编程语言",
            "Python是编程语言"
        )
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertEqual(calculator.get_name(), "cosine_similarity")
    
    def test_relevance_calculator(self):
        """测试相关性计算器"""
        calculator = RelevanceCalculator()
        
        # 测试相关的输入输出
        score1 = calculator.calculate(
            actual="Python是一种编程语言，适合初学者学习。",
            expected="Python是编程语言。",
            input_text="什么是Python？"
        )
        
        # 测试不相关的输入输出
        score2 = calculator.calculate(
            actual="今天天气很好。",
            expected="Python是编程语言。",
            input_text="什么是Python？"
        )
        
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 100)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 100)
        self.assertGreater(score1, score2)  # 相关输出得分应该更高
        
        self.assertEqual(calculator.get_name(), "relevance")
    
    def test_completeness_calculator(self):
        """测试完整性计算器"""
        calculator = CompletenessCalculator()
        
        # 测试完整的输出
        score1 = calculator.calculate(
            actual="Python是一种高级编程语言。它有简洁的语法和丰富的库。适合初学者学习。",
            expected="Python是编程语言，有简洁语法，适合初学者。"
        )
        
        # 测试不完整的输出
        score2 = calculator.calculate(
            actual="Python是语言。",
            expected="Python是编程语言，有简洁语法，适合初学者。"
        )
        
        self.assertGreaterEqual(score1, 0)
        self.assertLessEqual(score1, 100)
        self.assertGreaterEqual(score2, 0)
        self.assertLessEqual(score2, 100)
        self.assertGreater(score1, score2)  # 完整输出得分应该更高
        
        self.assertEqual(calculator.get_name(), "completeness")
    
    def test_completeness_without_expected(self):
        """测试无期望输出的完整性计算"""
        calculator = CompletenessCalculator()
        
        # 测试结构化输出
        score1 = calculator.calculate(
            actual="首先，Python是一种编程语言。其次，它有简洁的语法。最后，它适合初学者。举例来说，print('Hello')就是一个简单的例子。",
            expected=""
        )
        
        # 测试简单输出
        score2 = calculator.calculate(
            actual="Python是语言。",
            expected=""
        )
        
        self.assertGreater(score1, score2)  # 结构化输出应该得分更高


class TestComparisonResult(unittest.TestCase):
    """对比结果测试类"""
    
    def test_comparison_result_creation(self):
        """测试对比结果创建"""
        result = ComparisonResult(
            comparison_id="test_comparison",
            outputs=[
                {"id": "output_a", "text": "输出A"},
                {"id": "output_b", "text": "输出B"}
            ],
            winner="output_b",
            scores_comparison={"similarity": [80.0, 90.0]},
            key_differences=["长度差异", "详细程度不同"],
            recommendations=["建议使用输出B"]
        )
        
        self.assertEqual(result.comparison_id, "test_comparison")
        self.assertEqual(len(result.outputs), 2)
        self.assertEqual(result.winner, "output_b")
        self.assertEqual(len(result.key_differences), 2)
        self.assertEqual(len(result.recommendations), 1)
    
    def test_serialization(self):
        """测试序列化和反序列化"""
        result = ComparisonResult(
            comparison_id="serialize_test",
            outputs=[{"id": "test", "text": "测试"}],
            winner="test"
        )
        
        # 序列化
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["comparison_id"], "serialize_test")
        
        # 反序列化
        restored_result = ComparisonResult.from_dict(result_dict)
        self.assertEqual(restored_result.comparison_id, result.comparison_id)
        self.assertEqual(restored_result.winner, result.winner)


if __name__ == "__main__":
    unittest.main()