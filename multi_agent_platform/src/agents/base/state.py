"""
智能体状态管理模块

提供智能体的状态管理功能，包括：
- 智能体运行状态跟踪
- 任务执行状态管理
- 性能指标收集
- 状态持久化
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid


class AgentStatus(Enum):
    """智能体状态枚举"""
    INITIALIZING = "initializing"    # 初始化中
    IDLE = "idle"                    # 空闲
    BUSY = "busy"                    # 忙碌
    ERROR = "error"                  # 错误
    SHUTDOWN = "shutdown"            # 关闭
    MAINTENANCE = "maintenance"      # 维护中


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"              # 等待中
    RUNNING = "running"              # 执行中
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 失败
    CANCELLED = "cancelled"          # 已取消
    TIMEOUT = "timeout"              # 超时


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task_type: str
    status: TaskStatus
    created_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status.value,
            'created_time': self.created_time.isoformat(),
            'started_time': self.started_time.isoformat() if self.started_time else None,
            'completed_time': self.completed_time.isoformat() if self.completed_time else None,
            'duration': self.duration,
            'result': self.result,
            'error': self.error,
            'metadata': self.metadata
        }


@dataclass 
class PerformanceMetrics:
    """性能指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # 任务/秒
    last_activity: Optional[datetime] = None
    
    def update_task_metrics(self, task_info: TaskInfo) -> None:
        """更新任务指标"""
        self.total_tasks += 1
        
        if task_info.status == TaskStatus.COMPLETED:
            self.completed_tasks += 1
            if task_info.duration:
                # 更新平均执行时间
                total_duration = self.avg_task_duration * (self.completed_tasks - 1) + task_info.duration
                self.avg_task_duration = total_duration / self.completed_tasks
        elif task_info.status == TaskStatus.FAILED:
            self.failed_tasks += 1
        
        # 更新错误率
        if self.total_tasks > 0:
            self.error_rate = self.failed_tasks / self.total_tasks
        
        self.last_activity = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'avg_task_duration': self.avg_task_duration,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'error_rate': self.error_rate,
            'throughput': self.throughput,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


class AgentState:
    """智能体状态管理器"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.status = AgentStatus.INITIALIZING
        self.start_time = datetime.now()
        self.last_heartbeat = datetime.now()
        
        # 任务管理
        self.current_tasks: Dict[str, TaskInfo] = {}
        self.task_history: List[TaskInfo] = []
        self.max_history_size = 1000
        
        # 性能指标
        self.metrics = PerformanceMetrics()
        
        # 自定义状态数据
        self.custom_data: Dict[str, Any] = {}
        
        # 锁保护并发访问
        self._lock = threading.Lock()
        
        # 状态监听器
        self._listeners: List[callable] = []
    
    def set_status(self, status: AgentStatus, message: str = "") -> None:
        """设置智能体状态"""
        with self._lock:
            old_status = self.status
            self.status = status
            self.last_heartbeat = datetime.now()
            
            # 通知监听器
            self._notify_listeners('status_changed', {
                'old_status': old_status.value,
                'new_status': status.value,
                'message': message,
                'timestamp': self.last_heartbeat
            })
    
    def add_task(self, task_id: str, task_type: str, metadata: Dict[str, Any] = None) -> TaskInfo:
        """添加新任务"""
        with self._lock:
            task_info = TaskInfo(
                task_id=task_id,
                task_type=task_type,
                status=TaskStatus.PENDING,
                created_time=datetime.now(),
                metadata=metadata or {}
            )
            
            self.current_tasks[task_id] = task_info
            
            # 如果当前空闲，设置为忙碌状态
            if self.status == AgentStatus.IDLE:
                self.set_status(AgentStatus.BUSY)
            
            self._notify_listeners('task_added', task_info.to_dict())
            return task_info
    
    def start_task(self, task_id: str) -> bool:
        """开始执行任务"""
        with self._lock:
            if task_id not in self.current_tasks:
                return False
            
            task_info = self.current_tasks[task_id]
            task_info.status = TaskStatus.RUNNING
            task_info.started_time = datetime.now()
            
            self._notify_listeners('task_started', task_info.to_dict())
            return True
    
    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """完成任务"""
        with self._lock:
            if task_id not in self.current_tasks:
                return False
            
            task_info = self.current_tasks[task_id]
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_time = datetime.now()
            task_info.result = result
            
            if task_info.started_time:
                task_info.duration = (task_info.completed_time - task_info.started_time).total_seconds()
            
            # 更新性能指标
            self.metrics.update_task_metrics(task_info)
            
            # 移到历史记录
            self._move_to_history(task_id)
            
            # 检查是否所有任务都完成
            if not self.current_tasks and self.status == AgentStatus.BUSY:
                self.set_status(AgentStatus.IDLE)
            
            self._notify_listeners('task_completed', task_info.to_dict())
            return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """任务失败"""
        with self._lock:
            if task_id not in self.current_tasks:
                return False
            
            task_info = self.current_tasks[task_id]
            task_info.status = TaskStatus.FAILED
            task_info.completed_time = datetime.now()
            task_info.error = error
            
            if task_info.started_time:
                task_info.duration = (task_info.completed_time - task_info.started_time).total_seconds()
            
            # 更新性能指标
            self.metrics.update_task_metrics(task_info)
            
            # 移到历史记录
            self._move_to_history(task_id)
            
            # 检查是否所有任务都完成
            if not self.current_tasks and self.status == AgentStatus.BUSY:
                self.set_status(AgentStatus.IDLE)
            
            self._notify_listeners('task_failed', task_info.to_dict())
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            if task_id not in self.current_tasks:
                return False
            
            task_info = self.current_tasks[task_id]
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_time = datetime.now()
            
            # 移到历史记录
            self._move_to_history(task_id)
            
            # 检查是否所有任务都完成
            if not self.current_tasks and self.status == AgentStatus.BUSY:
                self.set_status(AgentStatus.IDLE)
            
            self._notify_listeners('task_cancelled', task_info.to_dict())
            return True
    
    def _move_to_history(self, task_id: str) -> None:
        """将任务移到历史记录"""
        if task_id in self.current_tasks:
            task_info = self.current_tasks.pop(task_id)
            self.task_history.append(task_info)
            
            # 限制历史记录大小
            if len(self.task_history) > self.max_history_size:
                self.task_history = self.task_history[-self.max_history_size:]
    
    def get_current_tasks(self) -> List[TaskInfo]:
        """获取当前任务列表"""
        with self._lock:
            return list(self.current_tasks.values())
    
    def get_task_history(self, limit: int = None) -> List[TaskInfo]:
        """获取任务历史"""
        with self._lock:
            if limit:
                return self.task_history[-limit:]
            return self.task_history.copy()
    
    def update_custom_data(self, key: str, value: Any) -> None:
        """更新自定义数据"""
        with self._lock:
            self.custom_data[key] = value
            self._notify_listeners('custom_data_updated', {
                'key': key,
                'value': value,
                'timestamp': datetime.now()
            })
    
    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """获取自定义数据"""
        with self._lock:
            return self.custom_data.get(key, default)
    
    def heartbeat(self) -> None:
        """发送心跳"""
        with self._lock:
            self.last_heartbeat = datetime.now()
    
    def is_healthy(self, timeout: float = 60.0) -> bool:
        """检查智能体是否健康"""
        with self._lock:
            if self.status in [AgentStatus.ERROR, AgentStatus.SHUTDOWN]:
                return False
            
            # 检查心跳超时
            time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
            return time_since_heartbeat <= timeout
    
    def get_uptime(self) -> float:
        """获取运行时间（秒）"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def add_listener(self, listener: callable) -> None:
        """添加状态监听器"""
        with self._lock:
            self._listeners.append(listener)
    
    def remove_listener(self, listener: callable) -> None:
        """移除状态监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def _notify_listeners(self, event_type: str, data: Any) -> None:
        """通知状态监听器"""
        for listener in self._listeners:
            try:
                listener(self.agent_id, event_type, data)
            except Exception as e:
                # 监听器异常不应影响状态管理
                pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        with self._lock:
            return {
                'agent_id': self.agent_id,
                'status': self.status.value,
                'start_time': self.start_time.isoformat(),
                'last_heartbeat': self.last_heartbeat.isoformat(),
                'uptime': self.get_uptime(),
                'current_tasks': [task.to_dict() for task in self.current_tasks.values()],
                'task_count': len(self.current_tasks),
                'metrics': self.metrics.to_dict(),
                'custom_data': self.custom_data,
                'is_healthy': self.is_healthy()
            }
    
    def save_state(self, file_path: str) -> None:
        """保存状态到文件"""
        state_dict = self.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_state(cls, agent_id: str, file_path: str) -> 'AgentState':
        """从文件加载状态"""
        with open(file_path, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        state = cls(agent_id)
        
        # 恢复基本信息
        state.status = AgentStatus(state_dict['status'])
        state.start_time = datetime.fromisoformat(state_dict['start_time'])
        state.last_heartbeat = datetime.fromisoformat(state_dict['last_heartbeat'])
        state.custom_data = state_dict.get('custom_data', {})
        
        # 恢复性能指标
        metrics_dict = state_dict.get('metrics', {})
        state.metrics = PerformanceMetrics(
            total_tasks=metrics_dict.get('total_tasks', 0),
            completed_tasks=metrics_dict.get('completed_tasks', 0),
            failed_tasks=metrics_dict.get('failed_tasks', 0),
            avg_task_duration=metrics_dict.get('avg_task_duration', 0.0),
            cpu_usage=metrics_dict.get('cpu_usage', 0.0),
            memory_usage=metrics_dict.get('memory_usage', 0.0),
            error_rate=metrics_dict.get('error_rate', 0.0),
            throughput=metrics_dict.get('throughput', 0.0),
            last_activity=datetime.fromisoformat(metrics_dict['last_activity']) 
                         if metrics_dict.get('last_activity') else None
        )
        
        return state


class StateManager:
    """状态管理器 - 管理多个智能体状态"""
    
    def __init__(self):
        self._states: Dict[str, AgentState] = {}
        self._lock = threading.Lock()
        self._global_listeners: List[callable] = []
    
    def create_agent_state(self, agent_id: str) -> AgentState:
        """创建智能体状态"""
        with self._lock:
            if agent_id in self._states:
                raise ValueError(f"智能体状态已存在: {agent_id}")
            
            state = AgentState(agent_id)
            # 添加全局监听器
            for listener in self._global_listeners:
                state.add_listener(listener)
            
            self._states[agent_id] = state
            return state
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """获取智能体状态"""
        with self._lock:
            return self._states.get(agent_id)
    
    def remove_agent_state(self, agent_id: str) -> bool:
        """移除智能体状态"""
        with self._lock:
            if agent_id in self._states:
                del self._states[agent_id]
                return True
            return False
    
    def list_agents(self) -> List[str]:
        """列出所有智能体ID"""
        with self._lock:
            return list(self._states.keys())
    
    def get_all_states(self) -> Dict[str, AgentState]:
        """获取所有智能体状态"""
        with self._lock:
            return self._states.copy()
    
    def add_global_listener(self, listener: callable) -> None:
        """添加全局状态监听器"""
        with self._lock:
            self._global_listeners.append(listener)
            # 为现有状态添加监听器
            for state in self._states.values():
                state.add_listener(listener)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        with self._lock:
            total_agents = len(self._states)
            active_agents = sum(1 for state in self._states.values() 
                               if state.status not in [AgentStatus.SHUTDOWN, AgentStatus.ERROR])
            
            total_tasks = sum(len(state.current_tasks) for state in self._states.values())
            
            return {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'total_current_tasks': total_tasks,
                'agents_by_status': {
                    status.value: sum(1 for state in self._states.values() 
                                     if state.status == status)
                    for status in AgentStatus
                },
                'timestamp': datetime.now().isoformat()
            }