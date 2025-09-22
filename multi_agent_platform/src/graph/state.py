"""
工作流状态管理模块

定义和管理LangGraph工作流的状态，包括：
- 全局工作流状态
- 智能体工作状态
- 状态同步和持久化
- 状态变更追踪
"""

from typing import Dict, Any, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
import copy
from abc import ABC, abstractmethod

T = TypeVar('T')


class WorkflowStatus(Enum):
    """工作流状态枚举"""
    PENDING = "pending"          # 等待中
    RUNNING = "running"          # 运行中
    PAUSED = "paused"           # 暂停
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"      # 已取消


class StateOperation(Enum):
    """状态操作类型"""
    SET = "set"                 # 设置值
    UPDATE = "update"           # 更新值
    APPEND = "append"           # 追加到列表
    REMOVE = "remove"           # 从列表移除
    MERGE = "merge"             # 合并字典


@dataclass
class StateChange:
    """状态变更记录"""
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    node_id: str = ""
    operation: StateOperation = StateOperation.SET
    key: str = ""
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'change_id': self.change_id,
            'timestamp': self.timestamp.isoformat(),
            'node_id': self.node_id,
            'operation': self.operation.value,
            'key': self.key,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'metadata': self.metadata
        }


class BaseState(ABC):
    """状态基类"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._history: List[StateChange] = []
        self._version = 0
        self._lock = False  # 状态锁定标志
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any, node_id: str = "", metadata: Dict[str, Any] = None) -> None:
        """设置状态值"""
        if self._lock:
            raise RuntimeError("状态已锁定，无法修改")
        
        old_value = self._data.get(key)
        self._data[key] = value
        
        # 记录变更
        change = StateChange(
            node_id=node_id,
            operation=StateOperation.SET,
            key=key,
            old_value=old_value,
            new_value=value,
            metadata=metadata or {}
        )
        self._history.append(change)
        self._version += 1
    
    def update(self, updates: Dict[str, Any], node_id: str = "", metadata: Dict[str, Any] = None) -> None:
        """批量更新状态"""
        if self._lock:
            raise RuntimeError("状态已锁定，无法修改")
        
        for key, value in updates.items():
            old_value = self._data.get(key)
            self._data[key] = value
            
            # 记录变更
            change = StateChange(
                node_id=node_id,
                operation=StateOperation.UPDATE,
                key=key,
                old_value=old_value,
                new_value=value,
                metadata=metadata or {}
            )
            self._history.append(change)
        
        self._version += 1
    
    def append(self, key: str, value: Any, node_id: str = "", metadata: Dict[str, Any] = None) -> None:
        """向列表追加值"""
        if self._lock:
            raise RuntimeError("状态已锁定，无法修改")
        
        if key not in self._data:
            self._data[key] = []
        
        if not isinstance(self._data[key], list):
            raise TypeError(f"键 {key} 的值不是列表类型")
        
        old_value = copy.deepcopy(self._data[key])
        self._data[key].append(value)
        
        # 记录变更
        change = StateChange(
            node_id=node_id,
            operation=StateOperation.APPEND,
            key=key,
            old_value=old_value,
            new_value=copy.deepcopy(self._data[key]),
            metadata=metadata or {}
        )
        self._history.append(change)
        self._version += 1
    
    def remove(self, key: str, value: Any, node_id: str = "", metadata: Dict[str, Any] = None) -> bool:
        """从列表移除值"""
        if self._lock:
            raise RuntimeError("状态已锁定，无法修改")
        
        if key not in self._data or not isinstance(self._data[key], list):
            return False
        
        if value not in self._data[key]:
            return False
        
        old_value = copy.deepcopy(self._data[key])
        self._data[key].remove(value)
        
        # 记录变更
        change = StateChange(
            node_id=node_id,
            operation=StateOperation.REMOVE,
            key=key,
            old_value=old_value,
            new_value=copy.deepcopy(self._data[key]),
            metadata=metadata or {}
        )
        self._history.append(change)
        self._version += 1
        return True
    
    def merge(self, key: str, value: Dict[str, Any], node_id: str = "", metadata: Dict[str, Any] = None) -> None:
        """合并字典值"""
        if self._lock:
            raise RuntimeError("状态已锁定，无法修改")
        
        if key not in self._data:
            self._data[key] = {}
        
        if not isinstance(self._data[key], dict):
            raise TypeError(f"键 {key} 的值不是字典类型")
        
        old_value = copy.deepcopy(self._data[key])
        self._data[key].update(value)
        
        # 记录变更
        change = StateChange(
            node_id=node_id,
            operation=StateOperation.MERGE,
            key=key,
            old_value=old_value,
            new_value=copy.deepcopy(self._data[key]),
            metadata=metadata or {}
        )
        self._history.append(change)
        self._version += 1
    
    def delete(self, key: str, node_id: str = "", metadata: Dict[str, Any] = None) -> bool:
        """删除状态值"""
        if self._lock:
            raise RuntimeError("状态已锁定，无法修改")
        
        if key not in self._data:
            return False
        
        old_value = self._data.pop(key)
        
        # 记录变更
        change = StateChange(
            node_id=node_id,
            operation=StateOperation.REMOVE,
            key=key,
            old_value=old_value,
            new_value=None,
            metadata=metadata or {}
        )
        self._history.append(change)
        self._version += 1
        return True
    
    def contains(self, key: str) -> bool:
        """检查是否包含键"""
        return key in self._data
    
    def keys(self) -> List[str]:
        """获取所有键"""
        return list(self._data.keys())
    
    def values(self) -> List[Any]:
        """获取所有值"""
        return list(self._data.values())
    
    def items(self) -> List[tuple]:
        """获取所有键值对"""
        return list(self._data.items())
    
    def lock(self) -> None:
        """锁定状态（只读）"""
        self._lock = True
    
    def unlock(self) -> None:
        """解锁状态"""
        self._lock = False
    
    def is_locked(self) -> bool:
        """检查是否锁定"""
        return self._lock
    
    def get_version(self) -> int:
        """获取版本号"""
        return self._version
    
    def get_history(self, limit: Optional[int] = None) -> List[StateChange]:
        """获取变更历史"""
        if limit:
            return self._history[-limit:]
        return self._history.copy()
    
    def clear_history(self) -> None:
        """清空变更历史"""
        self._history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return copy.deepcopy(self._data)
    
    def from_dict(self, data: Dict[str, Any], node_id: str = "") -> None:
        """从字典加载状态"""
        if self._lock:
            raise RuntimeError("状态已锁定，无法修改")
        
        old_data = copy.deepcopy(self._data)
        self._data = copy.deepcopy(data)
        
        # 记录变更
        change = StateChange(
            node_id=node_id,
            operation=StateOperation.SET,
            key="__all__",
            old_value=old_data,
            new_value=copy.deepcopy(self._data),
            metadata={'operation': 'from_dict'}
        )
        self._history.append(change)
        self._version += 1
    
    def clone(self) -> 'BaseState':
        """克隆状态"""
        new_state = type(self)()
        new_state._data = copy.deepcopy(self._data)
        new_state._history = copy.deepcopy(self._history)
        new_state._version = self._version
        new_state._lock = False  # 新状态不继承锁定状态
        return new_state


class WorkflowState(BaseState):
    """工作流状态"""
    
    def __init__(self, workflow_id: str = None):
        super().__init__()
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.status = WorkflowStatus.PENDING
        self.current_node = ""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        
        # 智能体状态映射
        self.agent_states: Dict[str, 'AgentWorkflowState'] = {}
        
        # 工作流结果
        self.results: Dict[str, Any] = {}
        
        # 执行上下文
        self.context: Dict[str, Any] = {}
        
        # 初始化基础状态
        self._initialize_base_state()
    
    def _initialize_base_state(self) -> None:
        """初始化基础状态"""
        self.set('workflow_id', self.workflow_id)
        self.set('status', self.status.value)
        self.set('current_node', self.current_node)
        self.set('results', self.results)
        self.set('context', self.context)
    
    def set_status(self, status: WorkflowStatus, node_id: str = "") -> None:
        """设置工作流状态"""
        old_status = self.status
        self.status = status
        self.set('status', status.value, node_id, {'old_status': old_status.value})
        
        # 记录时间戳
        if status == WorkflowStatus.RUNNING and not self.start_time:
            self.start_time = datetime.now()
            self.set('start_time', self.start_time.isoformat(), node_id)
        elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            self.end_time = datetime.now()
            self.set('end_time', self.end_time.isoformat(), node_id)
    
    def set_current_node(self, node_id: str, metadata: Dict[str, Any] = None) -> None:
        """设置当前节点"""
        old_node = self.current_node
        self.current_node = node_id
        self.set('current_node', node_id, node_id, 
                {'old_node': old_node, **(metadata or {})})
    
    def add_agent_state(self, agent_id: str, agent_state: 'AgentWorkflowState') -> None:
        """添加智能体状态"""
        self.agent_states[agent_id] = agent_state
        self.set(f'agent_states.{agent_id}', agent_state.to_dict())
    
    def get_agent_state(self, agent_id: str) -> Optional['AgentWorkflowState']:
        """获取智能体状态"""
        return self.agent_states.get(agent_id)
    
    def set_result(self, key: str, value: Any, node_id: str = "") -> None:
        """设置结果"""
        self.results[key] = value
        self.set(f'results.{key}', value, node_id)
    
    def get_result(self, key: str, default: Any = None) -> Any:
        """获取结果"""
        return self.results.get(key, default)
    
    def set_context(self, key: str, value: Any, node_id: str = "") -> None:
        """设置上下文"""
        self.context[key] = value
        self.set(f'context.{key}', value, node_id)
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文"""
        return self.context.get(key, default)
    
    def set_error(self, error_message: str, node_id: str = "") -> None:
        """设置错误信息"""
        self.error_message = error_message
        self.set('error_message', error_message, node_id)
        self.set_status(WorkflowStatus.FAILED, node_id)
    
    def is_completed(self) -> bool:
        """检查是否完成"""
        return self.status == WorkflowStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """检查是否失败"""
        return self.status == WorkflowStatus.FAILED
    
    def is_running(self) -> bool:
        """检查是否运行中"""
        return self.status == WorkflowStatus.RUNNING
    
    def get_duration(self) -> Optional[float]:
        """获取执行时长（秒）"""
        if not self.start_time:
            return None
        
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        
        return {
            **base_dict,
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'current_node': self.current_node,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.get_duration(),
            'error_message': self.error_message,
            'agent_states': {k: v.to_dict() for k, v in self.agent_states.items()},
            'results': self.results,
            'context': self.context,
            'version': self._version,
            'history_count': len(self._history)
        }


class AgentWorkflowState(BaseState):
    """智能体工作流状态"""
    
    def __init__(self, agent_id: str, workflow_id: str):
        super().__init__()
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.status = WorkflowStatus.PENDING
        self.current_task = ""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # 任务相关
        self.tasks: List[Dict[str, Any]] = []
        self.current_task_index = -1
        
        # 智能体特定状态
        self.agent_data: Dict[str, Any] = {}
        
        # 初始化基础状态
        self._initialize_base_state()
    
    def _initialize_base_state(self) -> None:
        """初始化基础状态"""
        self.set('agent_id', self.agent_id)
        self.set('workflow_id', self.workflow_id)
        self.set('status', self.status.value)
        self.set('current_task', self.current_task)
        self.set('tasks', self.tasks)
        self.set('agent_data', self.agent_data)
    
    def set_status(self, status: WorkflowStatus) -> None:
        """设置状态"""
        old_status = self.status
        self.status = status
        self.set('status', status.value, self.agent_id, {'old_status': old_status.value})
        
        # 记录时间戳
        if status == WorkflowStatus.RUNNING and not self.start_time:
            self.start_time = datetime.now()
            self.set('start_time', self.start_time.isoformat(), self.agent_id)
        elif status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            self.end_time = datetime.now()
            self.set('end_time', self.end_time.isoformat(), self.agent_id)
    
    def add_task(self, task: Dict[str, Any]) -> None:
        """添加任务"""
        self.tasks.append(task)
        self.set('tasks', self.tasks, self.agent_id)
    
    def start_task(self, task_id: str) -> bool:
        """开始任务"""
        for i, task in enumerate(self.tasks):
            if task.get('task_id') == task_id:
                self.current_task_index = i
                self.current_task = task_id
                task['status'] = 'running'
                task['start_time'] = datetime.now().isoformat()
                
                self.set('current_task', task_id, self.agent_id)
                self.set('tasks', self.tasks, self.agent_id)
                return True
        return False
    
    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """完成任务"""
        for task in self.tasks:
            if task.get('task_id') == task_id:
                task['status'] = 'completed'
                task['end_time'] = datetime.now().isoformat()
                task['result'] = result
                
                self.set('tasks', self.tasks, self.agent_id)
                return True
        return False
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """任务失败"""
        for task in self.tasks:
            if task.get('task_id') == task_id:
                task['status'] = 'failed'
                task['end_time'] = datetime.now().isoformat()
                task['error'] = error
                
                self.set('tasks', self.tasks, self.agent_id)
                return True
        return False
    
    def set_agent_data(self, key: str, value: Any) -> None:
        """设置智能体数据"""
        self.agent_data[key] = value
        self.set(f'agent_data.{key}', value, self.agent_id)
    
    def get_agent_data(self, key: str, default: Any = None) -> Any:
        """获取智能体数据"""
        return self.agent_data.get(key, default)
    
    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """获取当前任务"""
        if 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None
    
    def get_completed_tasks(self) -> List[Dict[str, Any]]:
        """获取已完成任务"""
        return [task for task in self.tasks if task.get('status') == 'completed']
    
    def get_failed_tasks(self) -> List[Dict[str, Any]]:
        """获取失败任务"""
        return [task for task in self.tasks if task.get('status') == 'failed']
    
    def get_duration(self) -> Optional[float]:
        """获取执行时长（秒）"""
        if not self.start_time:
            return None
        
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        
        return {
            **base_dict,
            'agent_id': self.agent_id,
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'current_task': self.current_task,
            'current_task_index': self.current_task_index,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.get_duration(),
            'tasks': self.tasks,
            'agent_data': self.agent_data,
            'version': self._version
        }


class StateManager:
    """状态管理器"""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowState] = {}
        self.state_snapshots: Dict[str, List[Dict[str, Any]]] = {}
        self.max_snapshots = 10
    
    def create_workflow_state(self, workflow_id: str = None) -> WorkflowState:
        """创建工作流状态"""
        workflow_state = WorkflowState(workflow_id)
        self.workflows[workflow_state.workflow_id] = workflow_state
        return workflow_state
    
    def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """获取工作流状态"""
        return self.workflows.get(workflow_id)
    
    def remove_workflow_state(self, workflow_id: str) -> bool:
        """移除工作流状态"""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            self.state_snapshots.pop(workflow_id, None)
            return True
        return False
    
    def create_snapshot(self, workflow_id: str) -> bool:
        """创建状态快照"""
        workflow_state = self.workflows.get(workflow_id)
        if not workflow_state:
            return False
        
        if workflow_id not in self.state_snapshots:
            self.state_snapshots[workflow_id] = []
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'state': workflow_state.to_dict()
        }
        
        self.state_snapshots[workflow_id].append(snapshot)
        
        # 限制快照数量
        if len(self.state_snapshots[workflow_id]) > self.max_snapshots:
            self.state_snapshots[workflow_id].pop(0)
        
        return True
    
    def restore_snapshot(self, workflow_id: str, snapshot_index: int = -1) -> bool:
        """恢复状态快照"""
        if workflow_id not in self.state_snapshots:
            return False
        
        snapshots = self.state_snapshots[workflow_id]
        if not snapshots or abs(snapshot_index) > len(snapshots):
            return False
        
        snapshot = snapshots[snapshot_index]
        workflow_state = self.workflows.get(workflow_id)
        
        if not workflow_state:
            # 重新创建工作流状态
            workflow_state = WorkflowState(workflow_id)
            self.workflows[workflow_id] = workflow_state
        
        # 恢复状态数据
        workflow_state.from_dict(snapshot['state'])
        
        return True
    
    def list_workflows(self) -> List[str]:
        """列出所有工作流"""
        return list(self.workflows.keys())
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """获取工作流摘要"""
        summary = {
            'total_workflows': len(self.workflows),
            'by_status': {},
            'workflows': []
        }
        
        for workflow_id, state in self.workflows.items():
            # 按状态统计
            status = state.status.value
            summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
            
            # 工作流信息
            summary['workflows'].append({
                'workflow_id': workflow_id,
                'status': status,
                'current_node': state.current_node,
                'duration': state.get_duration(),
                'agent_count': len(state.agent_states)
            })
        
        return summary