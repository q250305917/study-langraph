"""
性能评估模板模块

本模块实现了全面的性能评估系统，用于监控LangChain应用的执行速度、资源使用和成本效益。
支持实时监控、历史分析和性能优化建议。

核心功能：
1. 执行时间监控 - 精确测量各组件的执行时间和响应延迟
2. 资源使用监控 - 监控CPU、内存、网络等系统资源使用情况
3. 成本分析 - 计算API调用成本、计算资源成本等
4. 吞吐量测试 - 测试系统在不同负载下的处理能力
5. 性能基准对比 - 与历史数据或基准线进行对比分析
6. 瓶颈识别 - 自动识别性能瓶颈和优化建议

设计原理：
- 多维度监控：时间、资源、成本、质量的综合评估
- 实时性能跟踪：支持实时监控和历史趋势分析
- 自动化基准测试：定期执行性能基准测试
- 智能优化建议：基于性能数据提供优化建议
- 可扩展的指标体系：支持自定义性能指标
"""

import json
import time
import asyncio
import psutil
import threading
import statistics
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import tracemalloc
import gc
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    np = None
    plt = None
    sns = None

try:
    import psutil
    import GPUtil
except ImportError:
    psutil = None
    GPUtil = None

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError, 
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class PerformanceMetric(Enum):
    """性能指标枚举"""
    RESPONSE_TIME = "response_time"           # 响应时间
    THROUGHPUT = "throughput"                 # 吞吐量
    CPU_USAGE = "cpu_usage"                   # CPU使用率
    MEMORY_USAGE = "memory_usage"             # 内存使用量
    MEMORY_PEAK = "memory_peak"               # 内存峰值
    DISK_IO = "disk_io"                       # 磁盘I/O
    NETWORK_IO = "network_io"                 # 网络I/O
    API_COST = "api_cost"                     # API成本
    TOKEN_USAGE = "token_usage"               # Token使用量
    CACHE_HIT_RATE = "cache_hit_rate"         # 缓存命中率
    ERROR_RATE = "error_rate"                 # 错误率
    LATENCY_P50 = "latency_p50"              # 50%延迟
    LATENCY_P95 = "latency_p95"              # 95%延迟
    LATENCY_P99 = "latency_p99"              # 99%延迟


class MonitoringLevel(Enum):
    """监控级别枚举"""
    BASIC = "basic"                          # 基础监控：时间、内存
    STANDARD = "standard"                    # 标准监控：+CPU、网络
    ADVANCED = "advanced"                    # 高级监控：+详细资源、成本
    COMPREHENSIVE = "comprehensive"          # 全面监控：所有指标


class PerformanceStatus(Enum):
    """性能状态枚举"""
    EXCELLENT = "excellent"                  # 优秀
    GOOD = "good"                           # 良好
    ACCEPTABLE = "acceptable"               # 可接受
    POOR = "poor"                           # 较差
    CRITICAL = "critical"                   # 严重


@dataclass
class ResourceSnapshot:
    """
    资源快照数据结构
    
    记录特定时刻的系统资源使用情况。
    """
    timestamp: float = field(default_factory=time.time)
    
    # CPU信息
    cpu_percent: float = 0.0                 # CPU使用率
    cpu_count: int = 0                       # CPU核心数
    load_average: Optional[Tuple[float, float, float]] = None  # 负载平均值
    
    # 内存信息
    memory_used: int = 0                     # 已用内存（字节）
    memory_total: int = 0                    # 总内存（字节）
    memory_percent: float = 0.0              # 内存使用率
    memory_available: int = 0                # 可用内存（字节）
    
    # 进程内存信息
    process_memory_rss: int = 0              # 常驻集大小
    process_memory_vms: int = 0              # 虚拟内存大小
    process_memory_percent: float = 0.0      # 进程内存使用率
    
    # 磁盘I/O
    disk_read_bytes: int = 0                 # 磁盘读取字节
    disk_write_bytes: int = 0                # 磁盘写入字节
    disk_read_count: int = 0                 # 磁盘读取次数
    disk_write_count: int = 0                # 磁盘写入次数
    
    # 网络I/O
    network_sent_bytes: int = 0              # 网络发送字节
    network_recv_bytes: int = 0              # 网络接收字节
    network_sent_packets: int = 0            # 网络发送包数
    network_recv_packets: int = 0            # 网络接收包数
    
    # GPU信息（如果可用）
    gpu_usage: Optional[float] = None        # GPU使用率
    gpu_memory_used: Optional[int] = None    # GPU内存使用
    gpu_memory_total: Optional[int] = None   # GPU内存总量
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceSnapshot':
        """从字典创建ResourceSnapshot实例"""
        return cls(**data)
    
    def get_memory_usage_mb(self) -> float:
        """获取内存使用量（MB）"""
        return self.memory_used / (1024 * 1024)
    
    def get_process_memory_mb(self) -> float:
        """获取进程内存使用量（MB）"""
        return self.process_memory_rss / (1024 * 1024)


@dataclass
class PerformanceMetrics:
    """
    性能指标数据结构
    
    记录一次性能测试的完整指标数据。
    """
    test_id: str                             # 测试ID
    timestamp: float = field(default_factory=time.time)
    
    # 基础性能指标
    start_time: float = 0.0                  # 开始时间
    end_time: float = 0.0                    # 结束时间
    duration: float = 0.0                    # 执行时间（秒）
    
    # 资源使用指标
    initial_snapshot: Optional[ResourceSnapshot] = None  # 初始资源快照
    final_snapshot: Optional[ResourceSnapshot] = None    # 最终资源快照
    peak_snapshot: Optional[ResourceSnapshot] = None     # 峰值资源快照
    
    # 统计指标
    cpu_usage_avg: float = 0.0               # 平均CPU使用率
    cpu_usage_max: float = 0.0               # 最大CPU使用率
    memory_usage_avg: float = 0.0            # 平均内存使用量（MB）
    memory_usage_max: float = 0.0            # 最大内存使用量（MB）
    memory_delta: float = 0.0                # 内存增量（MB）
    
    # 业务指标
    requests_processed: int = 0              # 处理请求数
    tokens_consumed: int = 0                 # 消耗token数
    api_calls_made: int = 0                  # API调用次数
    cache_hits: int = 0                      # 缓存命中次数
    errors_occurred: int = 0                 # 发生错误次数
    
    # 成本指标
    estimated_cost: float = 0.0              # 估算成本
    cost_per_request: float = 0.0            # 每请求成本
    cost_per_token: float = 0.0              # 每token成本
    
    # 质量指标
    success_rate: float = 100.0              # 成功率
    average_quality_score: float = 0.0       # 平均质量得分
    
    # 延迟指标
    latency_p50: float = 0.0                 # 50%延迟
    latency_p95: float = 0.0                 # 95%延迟
    latency_p99: float = 0.0                 # 99%延迟
    
    # 元数据
    test_config: Dict[str, Any] = field(default_factory=dict)  # 测试配置
    environment: Dict[str, Any] = field(default_factory=dict)  # 环境信息
    metadata: Dict[str, Any] = field(default_factory=dict)     # 附加元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 处理可能的None值
        for key, value in data.items():
            if isinstance(value, ResourceSnapshot):
                data[key] = value.to_dict() if value else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """从字典创建PerformanceMetrics实例"""
        # 处理ResourceSnapshot字段
        for field_name in ['initial_snapshot', 'final_snapshot', 'peak_snapshot']:
            if data.get(field_name):
                data[field_name] = ResourceSnapshot.from_dict(data[field_name])
        return cls(**data)
    
    def calculate_throughput(self) -> float:
        """计算吞吐量（请求/秒）"""
        if self.duration > 0:
            return self.requests_processed / self.duration
        return 0.0
    
    def calculate_efficiency_score(self) -> float:
        """计算效率得分"""
        # 综合考虑时间、资源使用和质量的效率得分
        time_score = min(100, 100 / max(1, self.duration))  # 时间越短得分越高
        resource_score = max(0, 100 - self.cpu_usage_avg - self.memory_usage_avg / 10)  # 资源使用越少得分越高
        quality_score = self.average_quality_score
        
        return (time_score * 0.3 + resource_score * 0.3 + quality_score * 0.4)
    
    def get_performance_status(self) -> PerformanceStatus:
        """获取性能状态"""
        efficiency = self.calculate_efficiency_score()
        
        if efficiency >= 90:
            return PerformanceStatus.EXCELLENT
        elif efficiency >= 75:
            return PerformanceStatus.GOOD
        elif efficiency >= 60:
            return PerformanceStatus.ACCEPTABLE
        elif efficiency >= 40:
            return PerformanceStatus.POOR
        else:
            return PerformanceStatus.CRITICAL


class ResourceMonitor:
    """
    资源监控器
    
    负责实时监控系统和进程资源使用情况。
    """
    
    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD):
        """
        初始化资源监控器
        
        Args:
            monitoring_level: 监控级别
        """
        self.monitoring_level = monitoring_level
        self.is_monitoring = False
        self.snapshots: deque = deque(maxlen=1000)  # 最多保存1000个快照
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 0.1  # 监控间隔（秒）
        self._lock = threading.Lock()
        
        # 获取当前进程
        self.current_process = psutil.Process() if psutil else None
        
        logger.debug(f"Initialized ResourceMonitor with level: {monitoring_level}")
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.snapshots.clear()
        
        if self.current_process:
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.debug("Started resource monitoring")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.debug("Stopped resource monitoring")
    
    def get_current_snapshot(self) -> ResourceSnapshot:
        """获取当前资源快照"""
        snapshot = ResourceSnapshot()
        
        if not psutil:
            return snapshot
        
        try:
            # 系统信息
            snapshot.cpu_percent = psutil.cpu_percent()
            snapshot.cpu_count = psutil.cpu_count()
            
            # 负载平均值（仅Unix系统）
            if hasattr(psutil, 'getloadavg'):
                try:
                    snapshot.load_average = psutil.getloadavg()
                except (AttributeError, OSError):
                    pass
            
            # 内存信息
            memory = psutil.virtual_memory()
            snapshot.memory_used = memory.used
            snapshot.memory_total = memory.total
            snapshot.memory_percent = memory.percent
            snapshot.memory_available = memory.available
            
            # 进程信息
            if self.current_process:
                process_memory = self.current_process.memory_info()
                snapshot.process_memory_rss = process_memory.rss
                snapshot.process_memory_vms = process_memory.vms
                snapshot.process_memory_percent = self.current_process.memory_percent()
                
                # 磁盘I/O（如果支持）
                if self.monitoring_level in [MonitoringLevel.ADVANCED, MonitoringLevel.COMPREHENSIVE]:
                    try:
                        io_counters = self.current_process.io_counters()
                        snapshot.disk_read_bytes = io_counters.read_bytes
                        snapshot.disk_write_bytes = io_counters.write_bytes
                        snapshot.disk_read_count = io_counters.read_count
                        snapshot.disk_write_count = io_counters.write_count
                    except (AttributeError, psutil.AccessDenied):
                        pass
            
            # 网络I/O
            if self.monitoring_level in [MonitoringLevel.STANDARD, MonitoringLevel.ADVANCED, MonitoringLevel.COMPREHENSIVE]:
                try:
                    net_io = psutil.net_io_counters()
                    if net_io:
                        snapshot.network_sent_bytes = net_io.bytes_sent
                        snapshot.network_recv_bytes = net_io.bytes_recv
                        snapshot.network_sent_packets = net_io.packets_sent
                        snapshot.network_recv_packets = net_io.packets_recv
                except (AttributeError, psutil.AccessDenied):
                    pass
            
            # GPU信息
            if self.monitoring_level == MonitoringLevel.COMPREHENSIVE and GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一个GPU
                        snapshot.gpu_usage = gpu.load * 100
                        snapshot.gpu_memory_used = int(gpu.memoryUsed * 1024 * 1024)  # 转换为字节
                        snapshot.gpu_memory_total = int(gpu.memoryTotal * 1024 * 1024)
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to collect resource snapshot: {e}")
        
        return snapshot
    
    def get_peak_snapshot(self) -> Optional[ResourceSnapshot]:
        """获取峰值资源快照"""
        with self._lock:
            if not self.snapshots:
                return None
            
            # 找到内存使用峰值的快照
            peak_snapshot = max(self.snapshots, key=lambda s: s.memory_used)
            return peak_snapshot
    
    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        with self._lock:
            if not self.snapshots:
                return {}
            
            cpu_values = [s.cpu_percent for s in self.snapshots if s.cpu_percent > 0]
            memory_values = [s.get_memory_usage_mb() for s in self.snapshots]
            process_memory_values = [s.get_process_memory_mb() for s in self.snapshots]
            
            return {
                "cpu_usage_avg": statistics.mean(cpu_values) if cpu_values else 0.0,
                "cpu_usage_max": max(cpu_values) if cpu_values else 0.0,
                "memory_usage_avg": statistics.mean(memory_values) if memory_values else 0.0,
                "memory_usage_max": max(memory_values) if memory_values else 0.0,
                "process_memory_avg": statistics.mean(process_memory_values) if process_memory_values else 0.0,
                "process_memory_max": max(process_memory_values) if process_memory_values else 0.0
            }
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                snapshot = self.get_current_snapshot()
                with self._lock:
                    self.snapshots.append(snapshot)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitor_interval)


class PerformanceProfiler:
    """
    性能分析器
    
    用于分析和收集详细的性能数据。
    """
    
    def __init__(self):
        """初始化性能分析器"""
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.latency_records: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # 内存跟踪
        self.memory_tracking_enabled = False
        
        logger.debug("Initialized PerformanceProfiler")
    
    def start_test(self, test_id: str, config: Dict[str, Any] = None) -> str:
        """开始性能测试"""
        with self._lock:
            if test_id in self.active_tests:
                raise ValidationError(f"测试 {test_id} 已在进行中")
            
            # 启用内存跟踪
            if not self.memory_tracking_enabled:
                tracemalloc.start()
                self.memory_tracking_enabled = True
            
            test_data = {
                "test_id": test_id,
                "start_time": time.time(),
                "config": config or {},
                "resource_monitor": ResourceMonitor(),
                "request_count": 0,
                "error_count": 0,
                "token_count": 0,
                "api_calls": 0,
                "cache_hits": 0,
                "quality_scores": []
            }
            
            # 开始资源监控
            test_data["resource_monitor"].start_monitoring()
            
            self.active_tests[test_id] = test_data
            
        logger.debug(f"Started performance test: {test_id}")
        return test_id
    
    def record_request(self, test_id: str, duration: float, tokens: int = 0, 
                      api_calls: int = 0, cache_hit: bool = False, 
                      quality_score: float = 0.0, error: bool = False) -> None:
        """记录请求性能数据"""
        with self._lock:
            if test_id not in self.active_tests:
                return
            
            test_data = self.active_tests[test_id]
            
            # 更新计数
            test_data["request_count"] += 1
            test_data["token_count"] += tokens
            test_data["api_calls"] += api_calls
            if cache_hit:
                test_data["cache_hits"] += 1
            if error:
                test_data["error_count"] += 1
            if quality_score > 0:
                test_data["quality_scores"].append(quality_score)
            
            # 记录延迟
            self.latency_records[test_id].append(duration)
    
    def end_test(self, test_id: str) -> PerformanceMetrics:
        """结束性能测试并返回指标"""
        with self._lock:
            if test_id not in self.active_tests:
                raise ValidationError(f"测试 {test_id} 不存在")
            
            test_data = self.active_tests[test_id]
            resource_monitor = test_data["resource_monitor"]
            
            # 停止资源监控
            resource_monitor.stop_monitoring()
            
            # 收集最终数据
            end_time = time.time()
            duration = end_time - test_data["start_time"]
            
            # 创建性能指标
            metrics = PerformanceMetrics(
                test_id=test_id,
                start_time=test_data["start_time"],
                end_time=end_time,
                duration=duration,
                requests_processed=test_data["request_count"],
                tokens_consumed=test_data["token_count"],
                api_calls_made=test_data["api_calls"],
                cache_hits=test_data["cache_hits"],
                errors_occurred=test_data["error_count"],
                test_config=test_data["config"]
            )
            
            # 计算资源使用指标
            avg_metrics = resource_monitor.get_average_metrics()
            metrics.cpu_usage_avg = avg_metrics.get("cpu_usage_avg", 0.0)
            metrics.cpu_usage_max = avg_metrics.get("cpu_usage_max", 0.0)
            metrics.memory_usage_avg = avg_metrics.get("memory_usage_avg", 0.0)
            metrics.memory_usage_max = avg_metrics.get("memory_usage_max", 0.0)
            
            # 获取资源快照
            metrics.peak_snapshot = resource_monitor.get_peak_snapshot()
            if resource_monitor.snapshots:
                metrics.initial_snapshot = resource_monitor.snapshots[0]
                metrics.final_snapshot = resource_monitor.snapshots[-1]
                
                if metrics.initial_snapshot and metrics.final_snapshot:
                    initial_memory = metrics.initial_snapshot.get_process_memory_mb()
                    final_memory = metrics.final_snapshot.get_process_memory_mb()
                    metrics.memory_delta = final_memory - initial_memory
            
            # 计算延迟指标
            latencies = self.latency_records.get(test_id, [])
            if latencies:
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                metrics.latency_p50 = sorted_latencies[int(n * 0.5)]
                metrics.latency_p95 = sorted_latencies[int(n * 0.95)]
                metrics.latency_p99 = sorted_latencies[int(n * 0.99)]
            
            # 计算质量指标
            if test_data["quality_scores"]:
                metrics.average_quality_score = statistics.mean(test_data["quality_scores"])
            
            # 计算成功率
            total_requests = test_data["request_count"]
            if total_requests > 0:
                metrics.success_rate = ((total_requests - test_data["error_count"]) / total_requests) * 100
            
            # 估算成本（简化版本）
            metrics.estimated_cost = self._estimate_cost(metrics)
            if total_requests > 0:
                metrics.cost_per_request = metrics.estimated_cost / total_requests
            if metrics.tokens_consumed > 0:
                metrics.cost_per_token = metrics.estimated_cost / metrics.tokens_consumed
            
            # 环境信息
            metrics.environment = self._collect_environment_info()
            
            # 清理
            del self.active_tests[test_id]
            if test_id in self.latency_records:
                del self.latency_records[test_id]
            
        logger.debug(f"Ended performance test: {test_id}")
        return metrics
    
    def _estimate_cost(self, metrics: PerformanceMetrics) -> float:
        """估算成本"""
        # 这里可以根据实际的API pricing来计算
        # 目前使用简化的成本模型
        
        # API调用成本（假设每次调用0.001美元）
        api_cost = metrics.api_calls_made * 0.001
        
        # Token成本（假设每1K token 0.002美元）
        token_cost = (metrics.tokens_consumed / 1000) * 0.002
        
        # 计算资源成本（基于CPU和内存使用）
        compute_cost = (metrics.cpu_usage_avg / 100) * metrics.duration * 0.0001  # 假设每CPU小时0.0001美元
        memory_cost = (metrics.memory_usage_avg / 1000) * metrics.duration * 0.00001  # 假设每GB小时0.00001美元
        
        return api_cost + token_cost + compute_cost + memory_cost
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """收集环境信息"""
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": time.time()
        }
        
        if psutil:
            try:
                env_info.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "disk_usage": psutil.disk_usage('/').percent if sys.platform != 'win32' else None
                })
            except Exception:
                pass
        
        return env_info


class PerformanceEvalTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    性能评估模板
    
    提供全面的性能评估和监控功能，包括执行时间、资源使用、
    成本分析和性能优化建议。
    
    核心功能：
    1. 实时性能监控：监控CPU、内存、网络等资源使用
    2. 执行时间分析：精确测量各组件的执行时间
    3. 成本效益分析：计算API成本和资源成本
    4. 性能基准测试：与历史数据和基准线对比
    5. 瓶颈识别：自动识别性能瓶颈和优化建议
    6. 负载测试：测试不同负载下的系统性能
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化性能评估模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config)
        
        # 性能分析器
        self.profiler = PerformanceProfiler()
        
        # 历史性能数据
        self.performance_history: List[PerformanceMetrics] = []
        self.baseline_metrics: Dict[str, PerformanceMetrics] = {}
        
        # 配置参数
        self.monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
        self.auto_baseline: bool = True
        self.alert_thresholds: Dict[str, float] = {
            "max_response_time": 10.0,      # 最大响应时间（秒）
            "max_memory_usage": 1024.0,     # 最大内存使用（MB）
            "max_cpu_usage": 90.0,          # 最大CPU使用率（%）
            "min_success_rate": 95.0,       # 最小成功率（%）
            "max_cost_per_request": 0.01    # 最大单请求成本（美元）
        }
        self.storage_path: Optional[Path] = None
        
        # 线程锁
        self._lock = threading.Lock()
        
        logger.debug("Initialized PerformanceEvalTemplate")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="PerformanceEvalTemplate",
            description="性能评估模板",
            template_type=TemplateType.EVALUATION,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("monitoring_level", str, default="standard",
                           description="监控级别：basic, standard, advanced, comprehensive")
        config.add_parameter("auto_baseline", bool, default=True,
                           description="是否自动设置基准线")
        config.add_parameter("alert_thresholds", dict, 
                           default={
                               "max_response_time": 10.0,
                               "max_memory_usage": 1024.0,
                               "max_cpu_usage": 90.0,
                               "min_success_rate": 95.0,
                               "max_cost_per_request": 0.01
                           },
                           description="告警阈值")
        config.add_parameter("storage_path", str, default="./performance",
                           description="性能数据存储路径")
        config.add_parameter("history_limit", int, default=1000,
                           description="历史数据保存数量限制")
        config.add_parameter("enable_detailed_profiling", bool, default=False,
                           description="是否启用详细性能分析")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置性能评估模板
        
        Args:
            **parameters: 配置参数
                - monitoring_level: 监控级别
                - auto_baseline: 自动基准线
                - alert_thresholds: 告警阈值
                - storage_path: 存储路径
                - history_limit: 历史限制
                - enable_detailed_profiling: 详细分析
        """
        # 验证参数
        if not self.validate_parameters(parameters):
            raise ValidationError("PerformanceEvalTemplate参数验证失败")
        
        # 更新内部参数
        monitoring_level_str = parameters.get("monitoring_level", "standard")
        self.monitoring_level = MonitoringLevel(monitoring_level_str)
        
        self.auto_baseline = parameters.get("auto_baseline", True)
        self.alert_thresholds = parameters.get("alert_thresholds", {
            "max_response_time": 10.0,
            "max_memory_usage": 1024.0,
            "max_cpu_usage": 90.0,
            "min_success_rate": 95.0,
            "max_cost_per_request": 0.01
        })
        
        # 设置存储路径
        storage_path = Path(parameters.get("storage_path", "./performance"))
        storage_path.mkdir(parents=True, exist_ok=True)
        self.storage_path = storage_path
        
        # 历史数据限制
        self.history_limit = parameters.get("history_limit", 1000)
        
        # 加载历史数据
        self._load_performance_history()
        
        self.status = self.config.template_type.CONFIGURED
        logger.info(f"PerformanceEvalTemplate配置完成：monitoring_level={monitoring_level_str}")
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行性能评估
        
        Args:
            input_data: 输入数据
                - action: 操作类型（start_test, end_test, record_request等）
                - test_id: 测试ID
                - duration: 执行时间
                - tokens: token使用量
                - api_calls: API调用次数
                - config: 测试配置
                
        Returns:
            执行结果字典
        """
        action = input_data.get("action", "start_test")
        
        try:
            if action == "start_test":
                return self._start_performance_test(input_data)
            elif action == "end_test":
                return self._end_performance_test(input_data)
            elif action == "record_request":
                return self._record_request_performance(input_data)
            elif action == "get_metrics":
                return self._get_performance_metrics(input_data)
            elif action == "compare_with_baseline":
                return self._compare_with_baseline(input_data)
            elif action == "set_baseline":
                return self._set_baseline(input_data)
            elif action == "analyze_trends":
                return self._analyze_performance_trends(input_data)
            elif action == "load_test":
                return self._run_load_test(input_data)
            elif action == "get_recommendations":
                return self._get_optimization_recommendations(input_data)
            elif action == "check_alerts":
                return self._check_performance_alerts(input_data)
            elif action == "get_statistics":
                return self._get_performance_statistics()
            elif action == "export_report":
                return self._export_performance_report(input_data)
            else:
                raise ValidationError(f"未知的操作类型：{action}")
                
        except Exception as e:
            logger.error(f"执行性能评估失败：{action} - {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    def _start_performance_test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """开始性能测试"""
        test_id = input_data.get("test_id")
        if not test_id:
            test_id = f"test_{int(time.time())}"
        
        config = input_data.get("config", {})
        config["monitoring_level"] = self.monitoring_level.value
        
        try:
            actual_test_id = self.profiler.start_test(test_id, config)
            
            return {
                "success": True,
                "test_id": actual_test_id,
                "started_at": time.time(),
                "monitoring_level": self.monitoring_level.value
            }
            
        except Exception as e:
            logger.error(f"Failed to start performance test: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _end_performance_test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """结束性能测试"""
        test_id = input_data.get("test_id")
        if not test_id:
            return {
                "success": False,
                "error": "必须提供test_id"
            }
        
        try:
            metrics = self.profiler.end_test(test_id)
            
            # 保存到历史记录
            with self._lock:
                self.performance_history.append(metrics)
                
                # 限制历史记录数量
                if len(self.performance_history) > self.history_limit:
                    self.performance_history = self.performance_history[-self.history_limit:]
            
            # 自动设置基准线
            if self.auto_baseline and test_id not in self.baseline_metrics:
                self._auto_set_baseline(test_id, metrics)
            
            # 检查告警
            alerts = self._check_alerts(metrics)
            
            # 保存结果
            self._save_performance_metrics(metrics)
            
            return {
                "success": True,
                "test_id": test_id,
                "metrics": metrics.to_dict(),
                "performance_status": metrics.get_performance_status().value,
                "efficiency_score": metrics.calculate_efficiency_score(),
                "throughput": metrics.calculate_throughput(),
                "alerts": alerts
            }
            
        except Exception as e:
            logger.error(f"Failed to end performance test: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _record_request_performance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录请求性能"""
        test_id = input_data.get("test_id")
        duration = input_data.get("duration", 0.0)
        
        if not test_id:
            return {
                "success": False,
                "error": "必须提供test_id"
            }
        
        try:
            self.profiler.record_request(
                test_id=test_id,
                duration=duration,
                tokens=input_data.get("tokens", 0),
                api_calls=input_data.get("api_calls", 0),
                cache_hit=input_data.get("cache_hit", False),
                quality_score=input_data.get("quality_score", 0.0),
                error=input_data.get("error", False)
            )
            
            return {
                "success": True,
                "test_id": test_id,
                "recorded_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to record request performance: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_performance_metrics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取性能指标"""
        test_id = input_data.get("test_id")
        limit = input_data.get("limit", 10)
        
        with self._lock:
            if test_id:
                # 获取特定测试的指标
                metrics = [m for m in self.performance_history if m.test_id == test_id]
            else:
                # 获取最近的指标
                metrics = self.performance_history[-limit:]
        
        return {
            "success": True,
            "metrics": [m.to_dict() for m in metrics],
            "count": len(metrics)
        }
    
    def _compare_with_baseline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """与基准线对比"""
        test_id = input_data.get("test_id")
        baseline_id = input_data.get("baseline_id", test_id)
        
        if not test_id:
            return {
                "success": False,
                "error": "必须提供test_id"
            }
        
        # 获取当前指标
        with self._lock:
            current_metrics = next((m for m in self.performance_history if m.test_id == test_id), None)
            baseline_metrics = self.baseline_metrics.get(baseline_id)
        
        if not current_metrics:
            return {
                "success": False,
                "error": f"未找到测试 {test_id} 的性能数据"
            }
        
        if not baseline_metrics:
            return {
                "success": False,
                "error": f"未找到基准线 {baseline_id}"
            }
        
        # 计算对比结果
        comparison = self._calculate_performance_comparison(current_metrics, baseline_metrics)
        
        return {
            "success": True,
            "current_metrics": current_metrics.to_dict(),
            "baseline_metrics": baseline_metrics.to_dict(),
            "comparison": comparison
        }
    
    def _set_baseline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """设置基准线"""
        test_id = input_data.get("test_id")
        baseline_id = input_data.get("baseline_id", test_id)
        
        if not test_id:
            return {
                "success": False,
                "error": "必须提供test_id"
            }
        
        with self._lock:
            metrics = next((m for m in self.performance_history if m.test_id == test_id), None)
        
        if not metrics:
            return {
                "success": False,
                "error": f"未找到测试 {test_id} 的性能数据"
            }
        
        with self._lock:
            self.baseline_metrics[baseline_id] = metrics
        
        # 保存基准线
        self._save_baseline_metrics()
        
        return {
            "success": True,
            "baseline_id": baseline_id,
            "baseline_metrics": metrics.to_dict()
        }
    
    def _analyze_performance_trends(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能趋势"""
        days = input_data.get("days", 7)
        metric_name = input_data.get("metric", "duration")
        
        # 获取指定时间范围内的数据
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with self._lock:
            recent_metrics = [
                m for m in self.performance_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {
                "success": False,
                "error": f"没有找到最近 {days} 天的性能数据"
            }
        
        # 分析趋势
        trend_analysis = self._calculate_trend_analysis(recent_metrics, metric_name)
        
        return {
            "success": True,
            "period_days": days,
            "metric": metric_name,
            "data_points": len(recent_metrics),
            "trend_analysis": trend_analysis
        }
    
    def _run_load_test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行负载测试"""
        # 这里可以实现具体的负载测试逻辑
        # 目前返回模拟结果
        
        concurrent_users = input_data.get("concurrent_users", 10)
        test_duration = input_data.get("test_duration", 60)
        
        return {
            "success": True,
            "message": "负载测试功能正在开发中",
            "config": {
                "concurrent_users": concurrent_users,
                "test_duration": test_duration
            }
        }
    
    def _get_optimization_recommendations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取优化建议"""
        test_id = input_data.get("test_id")
        
        with self._lock:
            if test_id:
                metrics = next((m for m in self.performance_history if m.test_id == test_id), None)
            else:
                metrics = self.performance_history[-1] if self.performance_history else None
        
        if not metrics:
            return {
                "success": False,
                "error": "没有可用的性能数据"
            }
        
        recommendations = self._generate_optimization_recommendations(metrics)
        
        return {
            "success": True,
            "test_id": metrics.test_id,
            "performance_status": metrics.get_performance_status().value,
            "recommendations": recommendations
        }
    
    def _check_performance_alerts(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查性能告警"""
        test_id = input_data.get("test_id")
        
        with self._lock:
            if test_id:
                metrics = next((m for m in self.performance_history if m.test_id == test_id), None)
            else:
                metrics = self.performance_history[-1] if self.performance_history else None
        
        if not metrics:
            return {
                "success": False,
                "error": "没有可用的性能数据"
            }
        
        alerts = self._check_alerts(metrics)
        
        return {
            "success": True,
            "test_id": metrics.test_id,
            "alerts": alerts,
            "alert_count": len(alerts)
        }
    
    def _get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._lock:
            if not self.performance_history:
                return {
                    "success": True,
                    "statistics": {
                        "total_tests": 0
                    }
                }
            
            # 计算统计数据
            durations = [m.duration for m in self.performance_history]
            cpu_usages = [m.cpu_usage_avg for m in self.performance_history]
            memory_usages = [m.memory_usage_avg for m in self.performance_history]
            throughputs = [m.calculate_throughput() for m in self.performance_history]
            efficiency_scores = [m.calculate_efficiency_score() for m in self.performance_history]
            
            stats = {
                "total_tests": len(self.performance_history),
                "duration_stats": {
                    "mean": statistics.mean(durations),
                    "median": statistics.median(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "std": statistics.stdev(durations) if len(durations) > 1 else 0.0
                },
                "cpu_usage_stats": {
                    "mean": statistics.mean(cpu_usages),
                    "median": statistics.median(cpu_usages),
                    "min": min(cpu_usages),
                    "max": max(cpu_usages)
                },
                "memory_usage_stats": {
                    "mean": statistics.mean(memory_usages),
                    "median": statistics.median(memory_usages),
                    "min": min(memory_usages),
                    "max": max(memory_usages)
                },
                "throughput_stats": {
                    "mean": statistics.mean(throughputs),
                    "median": statistics.median(throughputs),
                    "min": min(throughputs),
                    "max": max(throughputs)
                },
                "efficiency_stats": {
                    "mean": statistics.mean(efficiency_scores),
                    "median": statistics.median(efficiency_scores),
                    "min": min(efficiency_scores),
                    "max": max(efficiency_scores)
                }
            }
            
            return {
                "success": True,
                "statistics": stats
            }
    
    def _export_performance_report(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """导出性能报告"""
        # 这里可以实现详细的报告生成逻辑
        # 包括图表、分析等
        return {
            "success": True,
            "message": "性能报告导出功能正在开发中"
        }
    
    def _auto_set_baseline(self, test_id: str, metrics: PerformanceMetrics) -> None:
        """自动设置基准线"""
        baseline_id = f"{test_id}_baseline"
        with self._lock:
            self.baseline_metrics[baseline_id] = metrics
        logger.debug(f"Auto-set baseline: {baseline_id}")
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """检查告警"""
        alerts = []
        
        # 响应时间告警
        if metrics.duration > self.alert_thresholds.get("max_response_time", 10.0):
            alerts.append({
                "type": "response_time",
                "severity": "warning",
                "message": f"响应时间过长：{metrics.duration:.2f}秒",
                "value": metrics.duration,
                "threshold": self.alert_thresholds["max_response_time"]
            })
        
        # 内存使用告警
        if metrics.memory_usage_max > self.alert_thresholds.get("max_memory_usage", 1024.0):
            alerts.append({
                "type": "memory_usage",
                "severity": "warning",
                "message": f"内存使用过高：{metrics.memory_usage_max:.2f}MB",
                "value": metrics.memory_usage_max,
                "threshold": self.alert_thresholds["max_memory_usage"]
            })
        
        # CPU使用告警
        if metrics.cpu_usage_max > self.alert_thresholds.get("max_cpu_usage", 90.0):
            alerts.append({
                "type": "cpu_usage",
                "severity": "warning",
                "message": f"CPU使用率过高：{metrics.cpu_usage_max:.1f}%",
                "value": metrics.cpu_usage_max,
                "threshold": self.alert_thresholds["max_cpu_usage"]
            })
        
        # 成功率告警
        if metrics.success_rate < self.alert_thresholds.get("min_success_rate", 95.0):
            alerts.append({
                "type": "success_rate",
                "severity": "error",
                "message": f"成功率过低：{metrics.success_rate:.1f}%",
                "value": metrics.success_rate,
                "threshold": self.alert_thresholds["min_success_rate"]
            })
        
        # 成本告警
        if metrics.cost_per_request > self.alert_thresholds.get("max_cost_per_request", 0.01):
            alerts.append({
                "type": "cost_per_request",
                "severity": "warning",
                "message": f"单请求成本过高：${metrics.cost_per_request:.4f}",
                "value": metrics.cost_per_request,
                "threshold": self.alert_thresholds["max_cost_per_request"]
            })
        
        return alerts
    
    def _calculate_performance_comparison(self, current: PerformanceMetrics, 
                                        baseline: PerformanceMetrics) -> Dict[str, Any]:
        """计算性能对比"""
        def calculate_change(current_val, baseline_val):
            if baseline_val == 0:
                return 0.0
            return ((current_val - baseline_val) / baseline_val) * 100
        
        comparison = {
            "duration_change": calculate_change(current.duration, baseline.duration),
            "throughput_change": calculate_change(current.calculate_throughput(), baseline.calculate_throughput()),
            "cpu_usage_change": calculate_change(current.cpu_usage_avg, baseline.cpu_usage_avg),
            "memory_usage_change": calculate_change(current.memory_usage_avg, baseline.memory_usage_avg),
            "efficiency_change": calculate_change(current.calculate_efficiency_score(), baseline.calculate_efficiency_score()),
            "cost_change": calculate_change(current.estimated_cost, baseline.estimated_cost)
        }
        
        # 判断性能改善或退化
        improvement_score = 0
        if comparison["duration_change"] < 0:  # 时间减少是好的
            improvement_score += 1
        if comparison["throughput_change"] > 0:  # 吞吐量增加是好的
            improvement_score += 1
        if comparison["cpu_usage_change"] < 0:  # CPU使用减少是好的
            improvement_score += 1
        if comparison["memory_usage_change"] < 0:  # 内存使用减少是好的
            improvement_score += 1
        if comparison["efficiency_change"] > 0:  # 效率提升是好的
            improvement_score += 1
        
        comparison["overall_improvement"] = improvement_score / 5 * 100  # 转换为百分比
        comparison["performance_verdict"] = "improved" if improvement_score >= 3 else "degraded"
        
        return comparison
    
    def _calculate_trend_analysis(self, metrics_list: List[PerformanceMetrics], 
                                metric_name: str) -> Dict[str, Any]:
        """计算趋势分析"""
        if len(metrics_list) < 2:
            return {"error": "数据点不足，无法分析趋势"}
        
        # 提取指标值
        values = []
        timestamps = []
        
        for metrics in sorted(metrics_list, key=lambda m: m.timestamp):
            if metric_name == "duration":
                values.append(metrics.duration)
            elif metric_name == "throughput":
                values.append(metrics.calculate_throughput())
            elif metric_name == "cpu_usage":
                values.append(metrics.cpu_usage_avg)
            elif metric_name == "memory_usage":
                values.append(metrics.memory_usage_avg)
            elif metric_name == "efficiency":
                values.append(metrics.calculate_efficiency_score())
            else:
                values.append(0.0)
            
            timestamps.append(metrics.timestamp)
        
        # 计算趋势
        if len(values) > 1:
            # 简单线性趋势
            x = list(range(len(values)))
            y = values
            
            # 计算斜率
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
            
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            
            return {
                "metric": metric_name,
                "data_points": len(values),
                "trend_direction": trend_direction,
                "trend_slope": slope,
                "min_value": min(values),
                "max_value": max(values),
                "mean_value": statistics.mean(values),
                "std_value": statistics.stdev(values) if len(values) > 1 else 0.0,
                "latest_value": values[-1],
                "change_from_first": values[-1] - values[0],
                "percent_change": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0.0
            }
        
        return {"error": "无法计算趋势"}
    
    def _generate_optimization_recommendations(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []
        
        # 响应时间优化建议
        if metrics.duration > 5.0:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "title": "响应时间优化",
                "description": f"当前响应时间为{metrics.duration:.2f}秒，建议优化",
                "suggestions": [
                    "检查是否有不必要的同步操作",
                    "考虑使用缓存减少重复计算",
                    "优化数据库查询",
                    "使用异步处理"
                ]
            })
        
        # 内存使用优化建议
        if metrics.memory_usage_max > 512.0:
            recommendations.append({
                "category": "memory",
                "priority": "medium",
                "title": "内存使用优化",
                "description": f"峰值内存使用{metrics.memory_usage_max:.1f}MB，可能过高",
                "suggestions": [
                    "检查是否有内存泄漏",
                    "优化数据结构选择",
                    "及时释放不需要的对象",
                    "考虑分批处理大量数据"
                ]
            })
        
        # CPU使用优化建议
        if metrics.cpu_usage_max > 80.0:
            recommendations.append({
                "category": "cpu",
                "priority": "medium",
                "title": "CPU使用优化",
                "description": f"CPU使用率达到{metrics.cpu_usage_max:.1f}%",
                "suggestions": [
                    "优化算法复杂度",
                    "避免不必要的循环",
                    "使用多线程或多进程",
                    "缓存计算结果"
                ]
            })
        
        # 成本优化建议
        if metrics.cost_per_request > 0.005:
            recommendations.append({
                "category": "cost",
                "priority": "low",
                "title": "成本优化",
                "description": f"单请求成本${metrics.cost_per_request:.4f}",
                "suggestions": [
                    "减少不必要的API调用",
                    "优化提示词以减少token使用",
                    "使用更经济的模型",
                    "实施智能缓存策略"
                ]
            })
        
        # 如果没有明显问题，给出一般性建议
        if not recommendations:
            recommendations.append({
                "category": "general",
                "priority": "low",
                "title": "常规优化建议",
                "description": "性能表现良好，可以考虑进一步优化",
                "suggestions": [
                    "定期监控性能指标",
                    "建立性能基准线",
                    "实施持续性能测试",
                    "优化用户体验"
                ]
            })
        
        return recommendations
    
    def _save_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """保存性能指标"""
        if not self.storage_path:
            return
        
        try:
            metrics_file = self.storage_path / f"metrics_{metrics.test_id}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    def _save_baseline_metrics(self) -> None:
        """保存基准线指标"""
        if not self.storage_path:
            return
        
        try:
            baseline_file = self.storage_path / "baselines.json"
            baseline_data = {
                baseline_id: metrics.to_dict() 
                for baseline_id, metrics in self.baseline_metrics.items()
            }
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save baseline metrics: {e}")
    
    def _load_performance_history(self) -> None:
        """加载性能历史数据"""
        if not self.storage_path:
            return
        
        try:
            # 加载基准线数据
            baseline_file = self.storage_path / "baselines.json"
            if baseline_file.exists():
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                
                with self._lock:
                    self.baseline_metrics = {
                        baseline_id: PerformanceMetrics.from_dict(data)
                        for baseline_id, data in baseline_data.items()
                    }
                
                logger.debug(f"Loaded {len(self.baseline_metrics)} baseline metrics")
            
            # 加载历史指标数据
            metrics_files = list(self.storage_path.glob("metrics_*.json"))
            loaded_metrics = []
            
            for metrics_file in metrics_files[-self.history_limit:]:  # 只加载最近的文件
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics_data = json.load(f)
                    
                    metrics = PerformanceMetrics.from_dict(metrics_data)
                    loaded_metrics.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"Failed to load metrics file {metrics_file}: {e}")
            
            # 按时间戳排序
            loaded_metrics.sort(key=lambda m: m.timestamp)
            
            with self._lock:
                self.performance_history = loaded_metrics
            
            logger.debug(f"Loaded {len(loaded_metrics)} performance metrics from history")
            
        except Exception as e:
            logger.error(f"Failed to load performance history: {e}")
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "monitoring_level": "standard",
                "auto_baseline": True,
                "alert_thresholds": {
                    "max_response_time": 10.0,
                    "max_memory_usage": 1024.0,
                    "max_cpu_usage": 90.0,
                    "min_success_rate": 95.0,
                    "max_cost_per_request": 0.01
                },
                "storage_path": "./performance",
                "history_limit": 1000,
                "enable_detailed_profiling": False
            },
            "execute_examples": [
                {
                    "description": "开始性能测试",
                    "input": {
                        "action": "start_test",
                        "test_id": "llm_test_001",
                        "config": {
                            "model": "gpt-3.5-turbo",
                            "max_tokens": 1000
                        }
                    }
                },
                {
                    "description": "记录请求性能",
                    "input": {
                        "action": "record_request",
                        "test_id": "llm_test_001",
                        "duration": 2.5,
                        "tokens": 150,
                        "api_calls": 1,
                        "cache_hit": False,
                        "quality_score": 85.0,
                        "error": False
                    }
                },
                {
                    "description": "结束性能测试",
                    "input": {
                        "action": "end_test",
                        "test_id": "llm_test_001"
                    }
                },
                {
                    "description": "获取优化建议",
                    "input": {
                        "action": "get_recommendations",
                        "test_id": "llm_test_001"
                    }
                }
            ],
            "usage_code": """
# 使用示例
from templates.evaluation.performance_eval import PerformanceEvalTemplate

# 初始化性能评估模板
perf_template = PerformanceEvalTemplate()

# 配置参数
perf_template.setup(
    monitoring_level="standard",
    auto_baseline=True,
    alert_thresholds={
        "max_response_time": 10.0,
        "max_memory_usage": 1024.0,
        "max_cpu_usage": 90.0,
        "min_success_rate": 95.0
    },
    storage_path="./performance"
)

# 开始性能测试
start_result = perf_template.run({
    "action": "start_test",
    "test_id": "my_test",
    "config": {"model": "gpt-3.5-turbo"}
})

# 记录请求性能（在实际使用中多次调用）
perf_template.run({
    "action": "record_request",
    "test_id": "my_test",
    "duration": 2.5,
    "tokens": 150,
    "api_calls": 1,
    "quality_score": 85.0
})

# 结束测试并获取结果
end_result = perf_template.run({
    "action": "end_test",
    "test_id": "my_test"
})

# 获取优化建议
recommendations = perf_template.run({
    "action": "get_recommendations",
    "test_id": "my_test"
})

print("性能结果：", end_result)
print("优化建议：", recommendations)
"""
        }