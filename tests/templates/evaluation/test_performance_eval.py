"""
性能评估模板测试

测试PerformanceEvalTemplate的各项功能，包括：
1. 基本配置和初始化
2. 性能测试启动和结束
3. 请求性能记录
4. 资源监控
5. 基准线对比
6. 统计分析
7. 错误处理
"""

import unittest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from templates.evaluation.performance_eval import (
    PerformanceEvalTemplate,
    PerformanceMetrics,
    ResourceSnapshot,
    ResourceMonitor,
    PerformanceProfiler,
    PerformanceMetric,
    MonitoringLevel,
    PerformanceStatus
)
from templates.base.template_base import TemplateConfig, TemplateType


class TestPerformanceEvalTemplate(unittest.TestCase):
    """性能评估模板测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.template = PerformanceEvalTemplate()
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_template_initialization(self):
        """测试模板初始化"""
        # 测试默认初始化
        template = PerformanceEvalTemplate()
        self.assertIsNotNone(template.config)
        self.assertEqual(template.config.name, "PerformanceEvalTemplate")
        self.assertEqual(template.config.template_type, TemplateType.EVALUATION)
        
        # 测试自定义配置初始化
        config = TemplateConfig(
            name="CustomPerformanceEval",
            description="自定义性能评估",
            template_type=TemplateType.EVALUATION
        )
        template = PerformanceEvalTemplate(config)
        self.assertEqual(template.config.name, "CustomPerformanceEval")
    
    def test_template_setup(self):
        """测试模板设置"""
        storage_path = Path(self.temp_dir) / "performance"
        
        self.template.setup(
            monitoring_level="advanced",
            auto_baseline=False,
            alert_thresholds={
                "max_response_time": 5.0,
                "max_memory_usage": 512.0,
                "max_cpu_usage": 80.0
            },
            storage_path=str(storage_path),
            history_limit=500
        )
        
        self.assertEqual(self.template.monitoring_level, MonitoringLevel.ADVANCED)
        self.assertFalse(self.template.auto_baseline)
        self.assertEqual(self.template.alert_thresholds["max_response_time"], 5.0)
        self.assertTrue(storage_path.exists())
        self.assertEqual(self.template.history_limit, 500)
    
    def test_start_performance_test(self):
        """测试开始性能测试"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "start_test",
            "test_id": "test_001",
            "config": {
                "description": "基础性能测试",
                "expected_duration": 10
            }
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["test_id"], "test_001")
        self.assertIn("started_at", result)
        self.assertIn("monitoring_level", result)
    
    def test_start_test_with_auto_id(self):
        """测试自动生成测试ID"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "start_test",
            "config": {"description": "自动ID测试"}
        })
        
        self.assertTrue(result["success"])
        self.assertIn("test_id", result)
        self.assertTrue(result["test_id"].startswith("test_"))
    
    def test_record_request_performance(self):
        """测试记录请求性能"""
        self.template.setup()
        
        # 先开始测试
        start_result = self.template.execute({
            "action": "start_test",
            "test_id": "record_test"
        })
        self.assertTrue(start_result["success"])
        
        # 记录请求性能
        result = self.template.execute({
            "action": "record_request",
            "test_id": "record_test",
            "duration": 2.5,
            "tokens": 150,
            "api_calls": 1,
            "cache_hit": True,
            "quality_score": 85.0,
            "error": False
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["test_id"], "record_test")
        self.assertIn("recorded_at", result)
    
    def test_end_performance_test(self):
        """测试结束性能测试"""
        self.template.setup()
        
        test_id = "end_test"
        
        # 开始测试
        self.template.execute({
            "action": "start_test",
            "test_id": test_id
        })
        
        # 记录一些请求
        for i in range(3):
            self.template.execute({
                "action": "record_request",
                "test_id": test_id,
                "duration": 1.0 + i * 0.5,
                "tokens": 100 + i * 50,
                "api_calls": 1,
                "quality_score": 80.0 + i * 5
            })
        
        # 短暂等待以确保有性能数据
        time.sleep(0.1)
        
        # 结束测试
        result = self.template.execute({
            "action": "end_test",
            "test_id": test_id
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["test_id"], test_id)
        self.assertIn("metrics", result)
        self.assertIn("performance_status", result)
        self.assertIn("efficiency_score", result)
        self.assertIn("throughput", result)
        self.assertIn("alerts", result)
        
        # 验证指标结构
        metrics = result["metrics"]
        self.assertIn("test_id", metrics)
        self.assertIn("duration", metrics)
        self.assertIn("requests_processed", metrics)
        self.assertIn("tokens_consumed", metrics)
        self.assertEqual(metrics["requests_processed"], 3)
        self.assertEqual(metrics["tokens_consumed"], 300)  # 100+150+200
    
    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        self.template.setup()
        
        # 执行完整的测试流程
        test_id = "metrics_test"
        self.template.execute({"action": "start_test", "test_id": test_id})
        self.template.execute({
            "action": "record_request",
            "test_id": test_id,
            "duration": 1.5,
            "tokens": 100
        })
        self.template.execute({"action": "end_test", "test_id": test_id})
        
        # 获取指标
        result = self.template.execute({
            "action": "get_metrics",
            "test_id": test_id
        })
        
        self.assertTrue(result["success"])
        self.assertIn("metrics", result)
        self.assertEqual(result["count"], 1)
        
        # 获取所有指标
        result_all = self.template.execute({
            "action": "get_metrics",
            "limit": 10
        })
        
        self.assertTrue(result_all["success"])
        self.assertGreaterEqual(len(result_all["metrics"]), 1)
    
    def test_set_and_compare_baseline(self):
        """测试设置和对比基准线"""
        self.template.setup()
        
        # 执行测试
        test_id = "baseline_test"
        self.template.execute({"action": "start_test", "test_id": test_id})
        self.template.execute({
            "action": "record_request",
            "test_id": test_id,
            "duration": 2.0,
            "tokens": 200
        })
        self.template.execute({"action": "end_test", "test_id": test_id})
        
        # 设置基准线
        baseline_result = self.template.execute({
            "action": "set_baseline",
            "test_id": test_id,
            "baseline_id": "test_baseline"
        })
        
        self.assertTrue(baseline_result["success"])
        self.assertEqual(baseline_result["baseline_id"], "test_baseline")
        
        # 执行另一个测试
        test_id2 = "comparison_test"
        self.template.execute({"action": "start_test", "test_id": test_id2})
        self.template.execute({
            "action": "record_request",
            "test_id": test_id2,
            "duration": 1.8,  # 稍快一些
            "tokens": 180
        })
        self.template.execute({"action": "end_test", "test_id": test_id2})
        
        # 与基准线对比
        compare_result = self.template.execute({
            "action": "compare_with_baseline",
            "test_id": test_id2,
            "baseline_id": "test_baseline"
        })
        
        self.assertTrue(compare_result["success"])
        self.assertIn("current_metrics", compare_result)
        self.assertIn("baseline_metrics", compare_result)
        self.assertIn("comparison", compare_result)
        
        # 验证对比结果
        comparison = compare_result["comparison"]
        self.assertIn("duration_change", comparison)
        self.assertIn("performance_verdict", comparison)
        self.assertIn("overall_improvement", comparison)
    
    def test_analyze_performance_trends(self):
        """测试性能趋势分析"""
        self.template.setup()
        
        # 执行多个测试以建立趋势数据
        for i in range(5):
            test_id = f"trend_test_{i}"
            self.template.execute({"action": "start_test", "test_id": test_id})
            self.template.execute({
                "action": "record_request",
                "test_id": test_id,
                "duration": 1.0 + i * 0.1,  # 递增的执行时间
                "tokens": 100 + i * 10
            })
            self.template.execute({"action": "end_test", "test_id": test_id})
            time.sleep(0.01)  # 确保时间戳不同
        
        # 分析趋势
        result = self.template.execute({
            "action": "analyze_trends",
            "days": 1,  # 分析最近1天
            "metric": "duration"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["metric"], "duration")
        self.assertGreaterEqual(result["data_points"], 5)
        self.assertIn("trend_analysis", result)
        
        # 验证趋势分析结果
        trend = result["trend_analysis"]
        self.assertIn("trend_direction", trend)
        self.assertIn("min_value", trend)
        self.assertIn("max_value", trend)
        self.assertIn("mean_value", trend)
        self.assertEqual(trend["trend_direction"], "increasing")  # 应该检测到递增趋势
    
    def test_get_optimization_recommendations(self):
        """测试获取优化建议"""
        self.template.setup(alert_thresholds={
            "max_response_time": 1.0,  # 设置较低的阈值以触发建议
            "max_memory_usage": 100.0,
            "max_cpu_usage": 50.0
        })
        
        # 执行一个较慢的测试
        test_id = "slow_test"
        self.template.execute({"action": "start_test", "test_id": test_id})
        self.template.execute({
            "action": "record_request",
            "test_id": test_id,
            "duration": 5.0,  # 超过阈值的慢速响应
            "tokens": 1000,
            "quality_score": 70.0
        })
        self.template.execute({"action": "end_test", "test_id": test_id})
        
        # 获取优化建议
        result = self.template.execute({
            "action": "get_recommendations",
            "test_id": test_id
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["test_id"], test_id)
        self.assertIn("performance_status", result)
        self.assertIn("recommendations", result)
        
        # 验证建议结构
        recommendations = result["recommendations"]
        self.assertIsInstance(recommendations, list)
        if recommendations:  # 如果有建议
            rec = recommendations[0]
            self.assertIn("category", rec)
            self.assertIn("priority", rec)
            self.assertIn("title", rec)
            self.assertIn("description", rec)
            self.assertIn("suggestions", rec)
    
    def test_check_performance_alerts(self):
        """测试性能告警检查"""
        self.template.setup(alert_thresholds={
            "max_response_time": 2.0,
            "max_memory_usage": 200.0,
            "min_success_rate": 90.0
        })
        
        # 执行一个触发告警的测试
        test_id = "alert_test"
        self.template.execute({"action": "start_test", "test_id": test_id})
        
        # 记录包含错误的请求
        self.template.execute({
            "action": "record_request",
            "test_id": test_id,
            "duration": 3.0,  # 超过阈值
            "tokens": 100,
            "error": False
        })
        self.template.execute({
            "action": "record_request",
            "test_id": test_id,
            "duration": 1.0,
            "tokens": 100,
            "error": True  # 错误请求
        })
        
        self.template.execute({"action": "end_test", "test_id": test_id})
        
        # 检查告警
        result = self.template.execute({
            "action": "check_alerts",
            "test_id": test_id
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["test_id"], test_id)
        self.assertIn("alerts", result)
        self.assertIn("alert_count", result)
        
        # 应该有告警（响应时间和成功率）
        alerts = result["alerts"]
        self.assertGreater(len(alerts), 0)
        
        # 验证告警结构
        if alerts:
            alert = alerts[0]
            self.assertIn("type", alert)
            self.assertIn("severity", alert)
            self.assertIn("message", alert)
            self.assertIn("value", alert)
            self.assertIn("threshold", alert)
    
    def test_get_performance_statistics(self):
        """测试获取性能统计"""
        self.template.setup()
        
        # 执行多个测试
        for i in range(3):
            test_id = f"stats_test_{i}"
            self.template.execute({"action": "start_test", "test_id": test_id})
            self.template.execute({
                "action": "record_request",
                "test_id": test_id,
                "duration": 1.0 + i * 0.5,
                "tokens": 100 + i * 50
            })
            self.template.execute({"action": "end_test", "test_id": test_id})
        
        # 获取统计信息
        result = self.template.execute({
            "action": "get_statistics"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("statistics", result)
        stats = result["statistics"]
        
        self.assertEqual(stats["total_tests"], 3)
        self.assertIn("duration_stats", stats)
        self.assertIn("cpu_usage_stats", stats)
        self.assertIn("memory_usage_stats", stats)
        self.assertIn("throughput_stats", stats)
        self.assertIn("efficiency_stats", stats)
        
        # 验证统计结构
        duration_stats = stats["duration_stats"]
        self.assertIn("mean", duration_stats)
        self.assertIn("median", duration_stats)
        self.assertIn("min", duration_stats)
        self.assertIn("max", duration_stats)
    
    def test_error_handling_invalid_action(self):
        """测试无效操作的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "invalid_action"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("未知的操作类型", result["error"])
    
    def test_error_handling_missing_test_id(self):
        """测试缺少测试ID的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "end_test"
            # 缺少test_id
        })
        
        self.assertFalse(result["success"])
        self.assertIn("必须提供test_id", result["error"])
    
    def test_error_handling_nonexistent_test(self):
        """测试不存在测试的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "end_test",
            "test_id": "nonexistent_test"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_concurrent_performance_tests(self):
        """测试并发性能测试"""
        self.template.setup()
        
        results = []
        errors = []
        
        def run_test(thread_id):
            """线程函数：运行性能测试"""
            try:
                test_id = f"concurrent_test_{thread_id}"
                
                # 开始测试
                start_result = self.template.execute({
                    "action": "start_test",
                    "test_id": test_id
                })
                results.append(start_result)
                
                # 记录请求
                record_result = self.template.execute({
                    "action": "record_request",
                    "test_id": test_id,
                    "duration": 1.0,
                    "tokens": 100
                })
                results.append(record_result)
                
                # 结束测试
                end_result = self.template.execute({
                    "action": "end_test",
                    "test_id": test_id
                })
                results.append(end_result)
                
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_test, args=(i,))
            threads.append(thread)
        
        # 启动线程
        for thread in threads:
            thread.start()
        
        # 等待线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(errors), 0)  # 不应该有错误
        self.assertEqual(len(results), 9)  # 3个线程 × 3个操作
        
        # 验证所有操作都成功
        for result in results:
            self.assertTrue(result["success"])
    
    def test_monitoring_levels(self):
        """测试不同监控级别"""
        levels = [
            MonitoringLevel.BASIC,
            MonitoringLevel.STANDARD,
            MonitoringLevel.ADVANCED,
            MonitoringLevel.COMPREHENSIVE
        ]
        
        for level in levels:
            with self.subTest(level=level):
                template = PerformanceEvalTemplate()
                template.setup(monitoring_level=level.value)
                
                self.assertEqual(template.monitoring_level, level)
                
                # 执行简单测试验证不同级别都能工作
                test_id = f"level_test_{level.value}"
                result = template.execute({
                    "action": "start_test",
                    "test_id": test_id
                })
                self.assertTrue(result["success"])


class TestPerformanceMetrics(unittest.TestCase):
    """性能指标测试类"""
    
    def test_performance_metrics_creation(self):
        """测试性能指标创建"""
        metrics = PerformanceMetrics(
            test_id="test_metrics",
            duration=2.5,
            requests_processed=10,
            tokens_consumed=500,
            cpu_usage_avg=45.0,
            memory_usage_avg=256.0
        )
        
        self.assertEqual(metrics.test_id, "test_metrics")
        self.assertEqual(metrics.duration, 2.5)
        self.assertEqual(metrics.requests_processed, 10)
        self.assertEqual(metrics.tokens_consumed, 500)
        self.assertEqual(metrics.cpu_usage_avg, 45.0)
        self.assertEqual(metrics.memory_usage_avg, 256.0)
    
    def test_calculate_throughput(self):
        """测试吞吐量计算"""
        metrics = PerformanceMetrics(
            test_id="throughput_test",
            duration=5.0,
            requests_processed=20
        )
        
        throughput = metrics.calculate_throughput()
        self.assertEqual(throughput, 4.0)  # 20请求 / 5秒 = 4请求/秒
        
        # 测试零除法保护
        metrics_zero = PerformanceMetrics(
            test_id="zero_test",
            duration=0.0,
            requests_processed=10
        )
        self.assertEqual(metrics_zero.calculate_throughput(), 0.0)
    
    def test_calculate_efficiency_score(self):
        """测试效率得分计算"""
        metrics = PerformanceMetrics(
            test_id="efficiency_test",
            duration=1.0,          # 快速执行
            cpu_usage_avg=30.0,    # 低CPU使用
            memory_usage_avg=100.0, # 低内存使用
            average_quality_score=90.0  # 高质量
        )
        
        score = metrics.calculate_efficiency_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        self.assertGreater(score, 50.0)  # 应该是较高的效率得分
    
    def test_get_performance_status(self):
        """测试性能状态判定"""
        # 优秀性能
        excellent_metrics = PerformanceMetrics(
            test_id="excellent_test",
            duration=0.5,
            cpu_usage_avg=10.0,
            memory_usage_avg=50.0,
            average_quality_score=95.0
        )
        self.assertEqual(excellent_metrics.get_performance_status(), PerformanceStatus.EXCELLENT)
        
        # 较差性能
        poor_metrics = PerformanceMetrics(
            test_id="poor_test",
            duration=10.0,         # 很慢
            cpu_usage_avg=90.0,    # 高CPU使用
            memory_usage_avg=800.0, # 高内存使用
            average_quality_score=60.0  # 低质量
        )
        status = poor_metrics.get_performance_status()
        self.assertIn(status, [PerformanceStatus.POOR, PerformanceStatus.CRITICAL])
    
    def test_serialization(self):
        """测试序列化和反序列化"""
        # 创建包含ResourceSnapshot的指标
        snapshot = ResourceSnapshot(
            cpu_percent=50.0,
            memory_used=1024*1024*256,  # 256MB
            memory_total=1024*1024*1024  # 1GB
        )
        
        metrics = PerformanceMetrics(
            test_id="serialize_test",
            duration=3.0,
            initial_snapshot=snapshot,
            final_snapshot=snapshot
        )
        
        # 序列化
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict["test_id"], "serialize_test")
        self.assertIsInstance(metrics_dict["initial_snapshot"], dict)
        
        # 反序列化
        restored_metrics = PerformanceMetrics.from_dict(metrics_dict)
        self.assertEqual(restored_metrics.test_id, metrics.test_id)
        self.assertEqual(restored_metrics.duration, metrics.duration)
        self.assertIsInstance(restored_metrics.initial_snapshot, ResourceSnapshot)


class TestResourceSnapshot(unittest.TestCase):
    """资源快照测试类"""
    
    def test_resource_snapshot_creation(self):
        """测试资源快照创建"""
        snapshot = ResourceSnapshot(
            cpu_percent=75.0,
            memory_used=1024*1024*512,  # 512MB
            memory_total=1024*1024*1024, # 1GB
            process_memory_rss=1024*1024*128  # 128MB
        )
        
        self.assertEqual(snapshot.cpu_percent, 75.0)
        self.assertEqual(snapshot.memory_used, 1024*1024*512)
        self.assertEqual(snapshot.memory_total, 1024*1024*1024)
        self.assertIsNotNone(snapshot.timestamp)
    
    def test_memory_conversion_methods(self):
        """测试内存转换方法"""
        snapshot = ResourceSnapshot(
            memory_used=1024*1024*256,  # 256MB
            process_memory_rss=1024*1024*128  # 128MB
        )
        
        self.assertEqual(snapshot.get_memory_usage_mb(), 256.0)
        self.assertEqual(snapshot.get_process_memory_mb(), 128.0)
    
    def test_serialization(self):
        """测试序列化和反序列化"""
        snapshot = ResourceSnapshot(
            cpu_percent=60.0,
            memory_used=1024*1024*300,
            load_average=(1.0, 1.5, 2.0)
        )
        
        # 序列化
        snapshot_dict = snapshot.to_dict()
        self.assertIsInstance(snapshot_dict, dict)
        self.assertEqual(snapshot_dict["cpu_percent"], 60.0)
        
        # 反序列化
        restored_snapshot = ResourceSnapshot.from_dict(snapshot_dict)
        self.assertEqual(restored_snapshot.cpu_percent, snapshot.cpu_percent)
        self.assertEqual(restored_snapshot.memory_used, snapshot.memory_used)
        self.assertEqual(restored_snapshot.load_average, snapshot.load_average)


class TestResourceMonitor(unittest.TestCase):
    """资源监控器测试类"""
    
    def test_resource_monitor_initialization(self):
        """测试资源监控器初始化"""
        monitor = ResourceMonitor(MonitoringLevel.STANDARD)
        
        self.assertEqual(monitor.monitoring_level, MonitoringLevel.STANDARD)
        self.assertFalse(monitor.is_monitoring)
        self.assertEqual(len(monitor.snapshots), 0)
    
    def test_get_current_snapshot(self):
        """测试获取当前资源快照"""
        monitor = ResourceMonitor()
        
        snapshot = monitor.get_current_snapshot()
        
        self.assertIsInstance(snapshot, ResourceSnapshot)
        self.assertIsNotNone(snapshot.timestamp)
        # 即使psutil不可用，也应该返回一个有效的快照对象
    
    @patch('templates.evaluation.performance_eval.psutil')
    def test_monitoring_with_psutil(self, mock_psutil):
        """测试带psutil的监控"""
        # 模拟psutil
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.cpu_count.return_value = 4
        
        mock_memory = MagicMock()
        mock_memory.used = 1024*1024*512
        mock_memory.total = 1024*1024*1024
        mock_memory.percent = 50.0
        mock_memory.available = 1024*1024*512
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 1024*1024*128
        mock_memory_info.vms = 1024*1024*256
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 12.5
        mock_psutil.Process.return_value = mock_process
        
        monitor = ResourceMonitor()
        snapshot = monitor.get_current_snapshot()
        
        self.assertEqual(snapshot.cpu_percent, 50.0)
        self.assertEqual(snapshot.memory_used, 1024*1024*512)
        self.assertEqual(snapshot.process_memory_rss, 1024*1024*128)
    
    def test_start_stop_monitoring(self):
        """测试开始和停止监控"""
        monitor = ResourceMonitor()
        
        # 开始监控
        monitor.start_monitoring()
        self.assertTrue(monitor.is_monitoring)
        
        # 等待一小段时间收集数据
        time.sleep(0.2)
        
        # 停止监控
        monitor.stop_monitoring()
        self.assertFalse(monitor.is_monitoring)
        
        # 应该收集到一些快照
        self.assertGreater(len(monitor.snapshots), 0)
    
    def test_get_average_metrics(self):
        """测试获取平均指标"""
        monitor = ResourceMonitor()
        
        # 手动添加一些快照
        for i in range(3):
            snapshot = ResourceSnapshot(
                cpu_percent=50.0 + i * 10,
                memory_used=1024*1024*(100 + i * 50)
            )
            monitor.snapshots.append(snapshot)
        
        avg_metrics = monitor.get_average_metrics()
        
        self.assertIn("cpu_usage_avg", avg_metrics)
        self.assertIn("memory_usage_avg", avg_metrics)
        self.assertEqual(avg_metrics["cpu_usage_avg"], 60.0)  # (50+60+70)/3
        self.assertEqual(avg_metrics["memory_usage_avg"], 150.0)  # (100+150+200)/3


class TestPerformanceProfiler(unittest.TestCase):
    """性能分析器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.profiler = PerformanceProfiler()
    
    def test_start_end_test(self):
        """测试开始和结束测试"""
        test_id = "profiler_test"
        
        # 开始测试
        actual_test_id = self.profiler.start_test(test_id, {"description": "测试"})
        self.assertEqual(actual_test_id, test_id)
        self.assertIn(test_id, self.profiler.active_tests)
        
        # 记录一些请求
        self.profiler.record_request(
            test_id=test_id,
            duration=1.5,
            tokens=100,
            api_calls=1,
            quality_score=85.0
        )
        
        # 结束测试
        metrics = self.profiler.end_test(test_id)
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.test_id, test_id)
        self.assertEqual(metrics.requests_processed, 1)
        self.assertEqual(metrics.tokens_consumed, 100)
        self.assertGreater(metrics.duration, 0)
        
        # 测试应该已从活跃列表中移除
        self.assertNotIn(test_id, self.profiler.active_tests)
    
    def test_duplicate_test_error(self):
        """测试重复测试ID错误"""
        test_id = "duplicate_test"
        
        # 第一次开始测试
        self.profiler.start_test(test_id)
        
        # 第二次开始相同ID的测试应该抛出错误
        with self.assertRaises(Exception):
            self.profiler.start_test(test_id)
    
    def test_record_multiple_requests(self):
        """测试记录多个请求"""
        test_id = "multi_request_test"
        
        self.profiler.start_test(test_id)
        
        # 记录多个请求
        request_data = [
            (1.0, 100, 1, False, 80.0, False),
            (1.5, 150, 1, True, 85.0, False),
            (2.0, 200, 2, False, 90.0, True)  # 包含错误
        ]
        
        for duration, tokens, api_calls, cache_hit, quality, error in request_data:
            self.profiler.record_request(
                test_id=test_id,
                duration=duration,
                tokens=tokens,
                api_calls=api_calls,
                cache_hit=cache_hit,
                quality_score=quality,
                error=error
            )
        
        # 结束测试并验证统计
        metrics = self.profiler.end_test(test_id)
        
        self.assertEqual(metrics.requests_processed, 3)
        self.assertEqual(metrics.tokens_consumed, 450)  # 100+150+200
        self.assertEqual(metrics.api_calls_made, 4)     # 1+1+2
        self.assertEqual(metrics.cache_hits, 1)
        self.assertEqual(metrics.errors_occurred, 1)
        self.assertEqual(metrics.average_quality_score, 85.0)  # (80+85+90)/3
        self.assertEqual(metrics.success_rate, 66.67)  # 2/3成功，约66.67%
    
    def test_latency_calculations(self):
        """测试延迟计算"""
        test_id = "latency_test"
        
        self.profiler.start_test(test_id)
        
        # 记录一系列延迟值
        latencies = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for latency in latencies:
            self.profiler.record_request(test_id=test_id, duration=latency)
        
        metrics = self.profiler.end_test(test_id)
        
        # 验证延迟百分位数
        self.assertAlmostEqual(metrics.latency_p50, 2.75, places=1)  # 50%分位数
        self.assertAlmostEqual(metrics.latency_p95, 4.75, places=1)  # 95%分位数
        self.assertAlmostEqual(metrics.latency_p99, 4.95, places=1)  # 99%分位数
    
    def test_nonexistent_test_error(self):
        """测试不存在测试的错误处理"""
        # 尝试结束不存在的测试
        with self.assertRaises(Exception):
            self.profiler.end_test("nonexistent_test")
        
        # 尝试记录不存在测试的请求
        self.profiler.record_request(
            test_id="nonexistent_test",
            duration=1.0
        )
        # record_request应该静默失败而不抛出异常


if __name__ == "__main__":
    unittest.main()