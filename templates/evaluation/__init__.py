"""
评估系统模板模块

本模块提供了完整的评估系统模板，用于评估LangChain应用的性能和质量：

模板类型：
- AccuracyEvalTemplate: 准确性评估模板，提供质量评估和对比分析
- PerformanceEvalTemplate: 性能评估模板，监控速度、成本和资源使用

核心功能：
1. 准确性评估 - 评估模型输出的质量和准确性
2. 性能监控 - 监控执行速度、内存使用和成本
3. 对比分析 - 比较不同模型或配置的性能
4. A/B测试 - 支持多个版本的对比测试
5. 指标收集 - 收集和分析各种性能指标
6. 报告生成 - 生成详细的评估报告

使用场景：
- 模型性能评估
- 系统优化分析
- 成本效益分析
- 质量控制检查
- A/B测试对比
- 生产环境监控

设计原理：
- 支持多种评估指标和标准
- 灵活的评估策略配置
- 实时性能监控
- 详细的报告和可视化
- 可扩展的评估框架
"""

from .accuracy_eval import AccuracyEvalTemplate
from .performance_eval import PerformanceEvalTemplate

__all__ = [
    "AccuracyEvalTemplate", 
    "PerformanceEvalTemplate",
]

# 模块版本信息
__version__ = "1.0.0"
__author__ = "LangChain Learning Team"
__description__ = "评估系统模板模块"