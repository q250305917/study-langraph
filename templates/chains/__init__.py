"""
Chain模板模块

本模块提供了完整的链组合模板系统，支持多种链类型的组合和编排。
这是模板系统的核心组件之一，用于实现复杂的工作流和业务逻辑。

核心特性：
1. 顺序链：步骤依次执行，适用于流水线处理
2. 并行链：多个分支同时执行，提高处理效率
3. 条件链：基于条件动态选择执行路径
4. 管道链：复杂工作流编排，支持多种模式组合
5. 链嵌套：支持链的嵌套组合，构建复杂系统
6. 错误处理：完善的错误处理和恢复机制

设计原理：
- 组合模式：将不同类型的链组合成更复杂的结构
- 策略模式：支持不同的执行策略和模式
- 观察者模式：监控链的执行状态和进度
- 责任链模式：将处理请求传递给链中的处理者
- 状态机模式：管理链的执行状态转换

使用场景：
- 数据处理流水线：ETL、数据清洗、特征工程
- 业务流程自动化：订单处理、审批流程、工作流
- 机器学习管道：数据预处理→训练→验证→部署
- 内容处理：文档解析→分析→转换→发布
- 系统集成：多系统间的数据流转和处理

模块结构：
- sequential_chain.py: 顺序链模板
- parallel_chain.py: 并行链模板
- conditional_chain.py: 条件链模板
- pipeline_chain.py: 管道链模板

作者: Claude Code Assistant
版本: 1.0.0
创建时间: 2024-09-21
"""

from typing import Dict, Any, Optional, List, Union, Type
import logging

# 导入链模板类
from .sequential_chain import (
    SequentialChainTemplate,
    StepConfig,
    StepResult,
    ChainContext,
    StepStatus,
    ErrorHandlingStrategy
)

from .parallel_chain import (
    ParallelChainTemplate,
    BranchConfig as ParallelBranchConfig,
    BranchResult as ParallelBranchResult,
    ParallelExecutor,
    BranchStatus,
    ExecutionMode,
    AggregationStrategy
)

from .conditional_chain import (
    ConditionalChainTemplate,
    ConditionConfig,
    BranchConfig as ConditionalBranchConfig,
    BranchResult as ConditionalBranchResult,
    ConditionEvaluator,
    ConditionType,
    LogicalOperator,
    ComparisonOperator,
    BranchStatus as ConditionalBranchStatus
)

from .pipeline_chain import (
    PipelineChainTemplate,
    StageConfig,
    StageResult,
    PipelineContext,
    StageType,
    StageStatus,
    DataFlowMode,
    RecoveryStrategy
)

# 导入基础模板类
from ..base.template_base import TemplateBase, TemplateConfig, TemplateType

# 配置日志
logger = logging.getLogger(__name__)

# 模块版本信息
__version__ = "1.0.0"
__author__ = "Claude Code Assistant"
__email__ = "claude@anthropic.com"
__description__ = "Chain模板系统 - 支持多种链组合模式的工作流编排"

# 导出的主要类
__all__ = [
    # === 主要模板类 ===
    "SequentialChainTemplate",
    "ParallelChainTemplate", 
    "ConditionalChainTemplate",
    "PipelineChainTemplate",
    
    # === 顺序链相关 ===
    "StepConfig",
    "StepResult",
    "ChainContext",
    "StepStatus",
    "ErrorHandlingStrategy",
    
    # === 并行链相关 ===
    "ParallelBranchConfig",
    "ParallelBranchResult",
    "ParallelExecutor",
    "BranchStatus",
    "ExecutionMode",
    "AggregationStrategy",
    
    # === 条件链相关 ===
    "ConditionConfig",
    "ConditionalBranchConfig",
    "ConditionalBranchResult",
    "ConditionEvaluator",
    "ConditionType",
    "LogicalOperator",
    "ComparisonOperator",
    "ConditionalBranchStatus",
    
    # === 管道链相关 ===
    "StageConfig",
    "StageResult",
    "PipelineContext",
    "StageType",
    "StageStatus",
    "DataFlowMode",
    "RecoveryStrategy",
    
    # === 工具函数 ===
    "create_chain",
    "get_available_chain_types",
    "validate_chain_config",
    "ChainFactory",
]


class ChainFactory:
    """
    链工厂类
    
    提供统一的链模板创建和管理接口，支持动态创建不同类型的链模板。
    """
    
    # 注册的链类型
    _chain_types: Dict[str, Type[TemplateBase]] = {
        "sequential": SequentialChainTemplate,
        "parallel": ParallelChainTemplate,
        "conditional": ConditionalChainTemplate,
        "pipeline": PipelineChainTemplate,
    }
    
    @classmethod
    def create_chain(
        cls, 
        chain_type: str, 
        config: Optional[TemplateConfig] = None,
        **kwargs
    ) -> TemplateBase:
        """
        创建链模板实例
        
        Args:
            chain_type: 链类型 ('sequential', 'parallel', 'conditional', 'pipeline')
            config: 模板配置
            **kwargs: 额外的配置参数
        
        Returns:
            链模板实例
        
        Raises:
            ValueError: 不支持的链类型
            
        Examples:
            >>> # 创建顺序链
            >>> chain = ChainFactory.create_chain('sequential')
            >>> 
            >>> # 创建并行链
            >>> chain = ChainFactory.create_chain('parallel', max_workers=8)
            >>> 
            >>> # 创建条件链
            >>> chain = ChainFactory.create_chain('conditional')
            >>> 
            >>> # 创建管道链
            >>> chain = ChainFactory.create_chain('pipeline')
        """
        if chain_type not in cls._chain_types:
            available_types = list(cls._chain_types.keys())
            raise ValueError(
                f"Unsupported chain type: {chain_type}. "
                f"Available types: {available_types}"
            )
        
        chain_class = cls._chain_types[chain_type]
        chain_instance = chain_class(config)
        
        # 如果有额外的配置参数，进行设置
        if kwargs:
            chain_instance.setup(**kwargs)
        
        logger.info(f"Created {chain_type} chain: {chain_instance.config.name}")
        return chain_instance
    
    @classmethod
    def register_chain_type(cls, name: str, chain_class: Type[TemplateBase]) -> None:
        """
        注册新的链类型
        
        Args:
            name: 链类型名称
            chain_class: 链模板类
        """
        cls._chain_types[name] = chain_class
        logger.info(f"Registered chain type: {name}")
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """
        获取可用的链类型列表
        
        Returns:
            链类型名称列表
        """
        return list(cls._chain_types.keys())
    
    @classmethod
    def get_chain_info(cls, chain_type: str) -> Dict[str, Any]:
        """
        获取链类型信息
        
        Args:
            chain_type: 链类型名称
        
        Returns:
            链类型信息字典
        """
        if chain_type not in cls._chain_types:
            raise ValueError(f"Unknown chain type: {chain_type}")
        
        chain_class = cls._chain_types[chain_type]
        
        return {
            "name": chain_type,
            "class": chain_class.__name__,
            "module": chain_class.__module__,
            "description": chain_class.__doc__.split('\n')[0] if chain_class.__doc__ else "",
            "template_type": TemplateType.CHAIN
        }


def create_chain(
    chain_type: str, 
    config: Optional[TemplateConfig] = None,
    **kwargs
) -> TemplateBase:
    """
    创建链模板实例的便捷函数
    
    Args:
        chain_type: 链类型
        config: 模板配置
        **kwargs: 额外的配置参数
    
    Returns:
        链模板实例
    
    Examples:
        >>> # 创建顺序链
        >>> chain = create_chain('sequential')
        >>> 
        >>> # 创建并行链并设置参数
        >>> chain = create_chain('parallel', max_workers=8)
    """
    return ChainFactory.create_chain(chain_type, config, **kwargs)


def get_available_chain_types() -> List[str]:
    """
    获取可用的链类型列表
    
    Returns:
        链类型名称列表
    
    Examples:
        >>> types = get_available_chain_types()
        >>> print(types)  # ['sequential', 'parallel', 'conditional', 'pipeline']
    """
    return ChainFactory.get_available_types()


def validate_chain_config(chain_type: str, config: Dict[str, Any]) -> bool:
    """
    验证链配置的有效性
    
    Args:
        chain_type: 链类型
        config: 链配置字典
    
    Returns:
        True如果配置有效
    
    Raises:
        ValueError: 配置无效
    
    Examples:
        >>> config = {"steps": [{"name": "step1", "executor": lambda x: x}]}
        >>> is_valid = validate_chain_config('sequential', config)
    """
    try:
        # 创建临时实例进行验证
        chain = create_chain(chain_type)
        chain.validate_parameters(config)
        return True
    except Exception as e:
        logger.error(f"Chain config validation failed: {str(e)}")
        raise ValueError(f"Invalid chain config: {str(e)}")


# 模块初始化
def _initialize_module():
    """初始化模块，注册所有链模板到全局工厂"""
    try:
        from ..base.template_base import get_template_factory
        
        factory = get_template_factory()
        
        # 注册所有链模板
        factory.register_template("sequential_chain", SequentialChainTemplate)
        factory.register_template("parallel_chain", ParallelChainTemplate)
        factory.register_template("conditional_chain", ConditionalChainTemplate)
        factory.register_template("pipeline_chain", PipelineChainTemplate)
        
        logger.info("Initialized chains module and registered all chain templates")
        
    except Exception as e:
        logger.error(f"Failed to initialize chains module: {str(e)}")


# 自动初始化模块
_initialize_module()


# 模块使用示例
def _get_usage_examples() -> Dict[str, str]:
    """获取模块使用示例"""
    return {
        "sequential_chain": '''
# 顺序链使用示例
from templates.chains import create_chain

# 创建顺序链
chain = create_chain('sequential')

# 配置步骤
chain.setup(
    steps=[
        {"name": "数据清洗", "executor": clean_data_func},
        {"name": "特征提取", "executor": extract_features_func},
        {"name": "模型预测", "executor": predict_func}
    ]
)

# 执行链
result = chain.run({"data": "input_data"})
''',
        
        "parallel_chain": '''
# 并行链使用示例
from templates.chains import create_chain

# 创建并行链
chain = create_chain('parallel', max_workers=4)

# 配置分支
chain.setup(
    branches=[
        {"name": "模型A", "executor": model_a_predict},
        {"name": "模型B", "executor": model_b_predict},
        {"name": "规则引擎", "executor": rule_engine}
    ]
)

# 执行链
result = chain.run({"text": "input_text"})
''',
        
        "conditional_chain": '''
# 条件链使用示例
from templates.chains import create_chain

# 创建条件链
chain = create_chain('conditional')

# 配置条件分支
chain.setup(
    branches=[
        {
            "name": "高价值用户",
            "condition": {
                "type": "value",
                "field_path": "user.score",
                "operator": "gt",
                "value": 80
            },
            "executor": handle_vip_user
        }
    ],
    default_branch={"name": "普通用户", "executor": handle_normal_user}
)

# 执行链
result = chain.run({"user": {"score": 85}})
''',
        
        "pipeline_chain": '''
# 管道链使用示例
from templates.chains import create_chain

# 创建管道链
chain = create_chain('pipeline')

# 配置阶段
chain.setup(
    stages=[
        {
            "name": "预处理阶段",
            "stage_type": "sequential",
            "template_config": {
                "steps": [
                    {"name": "清洗", "executor": clean_func},
                    {"name": "验证", "executor": validate_func}
                ]
            }
        },
        {
            "name": "并行处理阶段",
            "stage_type": "parallel", 
            "template_config": {
                "branches": [
                    {"name": "路径A", "executor": process_a},
                    {"name": "路径B", "executor": process_b}
                ]
            },
            "dependencies": ["stage_0"]
        }
    ]
)

# 执行管道
result = chain.run({"raw_data": "input"})
'''
    }


# 导出使用示例
USAGE_EXAMPLES = _get_usage_examples()


if __name__ == "__main__":
    # 模块测试代码
    print(f"Chain模板模块 v{__version__}")
    print(f"可用的链类型: {get_available_chain_types()}")
    
    # 创建测试链
    for chain_type in get_available_chain_types():
        try:
            chain = create_chain(chain_type)
            print(f"✓ 成功创建 {chain_type} 链: {chain.config.name}")
        except Exception as e:
            print(f"✗ 创建 {chain_type} 链失败: {str(e)}")