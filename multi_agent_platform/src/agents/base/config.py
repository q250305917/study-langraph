"""
智能体配置管理模块

提供智能体的配置管理功能，支持：
- 基础配置定义
- 配置验证和加载
- 动态配置更新
- 配置模板支持
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
from abc import ABC, abstractmethod


@dataclass
class AgentConfig:
    """智能体基础配置类"""
    
    # 基础信息
    agent_id: str = ""
    agent_name: str = ""
    agent_type: str = "base"
    description: str = ""
    version: str = "1.0.0"
    
    # 性能配置
    max_concurrent_tasks: int = 5
    task_timeout: float = 300.0  # 5分钟超时
    memory_limit: int = 1024  # MB
    cpu_limit: float = 1.0  # CPU核心数
    
    # LLM配置
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    llm_timeout: float = 30.0
    
    # 工具配置
    enabled_tools: List[str] = field(default_factory=list)
    tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 记忆配置
    memory_type: str = "redis"  # redis, sqlite, memory
    memory_ttl: int = 3600  # 秒
    max_memory_items: int = 1000
    
    # 通信配置
    communication_protocol: str = "async"
    message_queue_size: int = 100
    heartbeat_interval: float = 30.0
    
    # 监控配置
    enable_monitoring: bool = True
    metrics_interval: float = 10.0
    log_level: str = "INFO"
    
    # 扩展配置
    custom_configs: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if not self.agent_id:
            raise ValueError("agent_id不能为空")
        
        if not self.agent_name:
            self.agent_name = self.agent_id
            
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks必须大于0")
            
        if self.task_timeout <= 0:
            raise ValueError("task_timeout必须大于0")
            
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("llm_temperature必须在0-2之间")
            
        return True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """从字典创建配置"""
        # 过滤掉不存在的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        config = cls(**filtered_dict)
        config.validate()
        return config
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'AgentConfig':
        """从文件加载配置"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        suffix = config_path.suffix.lower()
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix in ['.yml', '.yaml']:
                config_dict = yaml.safe_load(f)
            elif suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'description': self.description,
            'version': self.version,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'task_timeout': self.task_timeout,
            'memory_limit': self.memory_limit,
            'cpu_limit': self.cpu_limit,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'llm_temperature': self.llm_temperature,
            'llm_max_tokens': self.llm_max_tokens,
            'llm_timeout': self.llm_timeout,
            'enabled_tools': self.enabled_tools,
            'tool_configs': self.tool_configs,
            'memory_type': self.memory_type,
            'memory_ttl': self.memory_ttl,
            'max_memory_items': self.max_memory_items,
            'communication_protocol': self.communication_protocol,
            'message_queue_size': self.message_queue_size,
            'heartbeat_interval': self.heartbeat_interval,
            'enable_monitoring': self.enable_monitoring,
            'metrics_interval': self.metrics_interval,
            'log_level': self.log_level,
            'custom_configs': self.custom_configs
        }
    
    def save_to_file(self, config_path: Path) -> None:
        """保存配置到文件"""
        config_dict = self.to_dict()
        
        suffix = config_path.suffix.lower()
        with open(config_path, 'w', encoding='utf-8') as f:
            if suffix in ['.yml', '.yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif suffix == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {suffix}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 添加到自定义配置中
                self.custom_configs[key] = value
        
        self.validate()
    
    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """获取自定义配置"""
        return self.custom_configs.get(key, default)


class ConfigTemplate:
    """配置模板管理器"""
    
    @staticmethod
    def create_data_analyst_config(agent_id: str) -> AgentConfig:
        """创建数据分析师智能体配置"""
        return AgentConfig(
            agent_id=agent_id,
            agent_name=f"数据分析师-{agent_id}",
            agent_type="data_analyst",
            description="专业的数据分析智能体，擅长数据处理、统计分析和可视化",
            enabled_tools=[
                "pandas_tool",
                "numpy_tool", 
                "plotly_tool",
                "statistics_tool",
                "sql_tool"
            ],
            llm_model="gpt-4",
            llm_temperature=0.1,
            max_concurrent_tasks=3,
            custom_configs={
                "supported_formats": ["csv", "xlsx", "json", "parquet"],
                "max_dataset_size": "100MB",
                "default_chart_type": "plotly"
            }
        )
    
    @staticmethod
    def create_code_generator_config(agent_id: str) -> AgentConfig:
        """创建代码生成器智能体配置"""
        return AgentConfig(
            agent_id=agent_id,
            agent_name=f"代码生成器-{agent_id}",
            agent_type="code_generator",
            description="专业的代码生成智能体，支持多种编程语言和框架",
            enabled_tools=[
                "code_template_tool",
                "syntax_validator_tool",
                "test_generator_tool",
                "documentation_tool"
            ],
            llm_model="gpt-4",
            llm_temperature=0.2,
            max_concurrent_tasks=2,
            custom_configs={
                "supported_languages": ["python", "javascript", "java", "go"],
                "code_style": "pep8",
                "include_tests": True,
                "include_docs": True
            }
        )
    
    @staticmethod
    def create_coordinator_config(agent_id: str) -> AgentConfig:
        """创建协调器智能体配置"""
        return AgentConfig(
            agent_id=agent_id,
            agent_name=f"协调器-{agent_id}",
            agent_type="coordinator",
            description="负责任务分解、智能体调度和结果整合的协调智能体",
            enabled_tools=[
                "task_decomposer_tool",
                "agent_registry_tool",
                "result_integrator_tool",
                "workflow_manager_tool"
            ],
            llm_model="gpt-4",
            llm_temperature=0.1,
            max_concurrent_tasks=10,
            custom_configs={
                "max_subtasks": 20,
                "scheduling_strategy": "load_balanced",
                "result_validation": True
            }
        )


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, AgentConfig] = {}
    
    def load_config(self, agent_id: str) -> Optional[AgentConfig]:
        """加载智能体配置"""
        if agent_id in self._configs:
            return self._configs[agent_id]
        
        config_file = self.config_dir / f"{agent_id}.yaml"
        if config_file.exists():
            config = AgentConfig.from_file(config_file)
            self._configs[agent_id] = config
            return config
        
        return None
    
    def save_config(self, config: AgentConfig) -> None:
        """保存智能体配置"""
        config_file = self.config_dir / f"{config.agent_id}.yaml"
        config.save_to_file(config_file)
        self._configs[config.agent_id] = config
    
    def list_configs(self) -> List[str]:
        """列出所有配置"""
        config_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in config_files]
    
    def delete_config(self, agent_id: str) -> bool:
        """删除智能体配置"""
        config_file = self.config_dir / f"{agent_id}.yaml"
        if config_file.exists():
            config_file.unlink()
            self._configs.pop(agent_id, None)
            return True
        return False