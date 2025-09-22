"""
通信协议定义模块

定义智能体间通信的消息格式和协议，包括：
- 消息类型定义
- 消息和响应数据结构
- 协议版本管理
- 消息验证机制
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json


class MessageType(Enum):
    """消息类型枚举"""
    # 基础消息
    PING = "ping"                           # 心跳检测
    PONG = "pong"                          # 心跳响应
    STATUS = "status"                       # 状态查询
    
    # 任务消息
    TASK_REQUEST = "task_request"           # 任务请求
    TASK_RESPONSE = "task_response"         # 任务响应
    TASK_UPDATE = "task_update"             # 任务更新
    TASK_CANCEL = "task_cancel"             # 任务取消
    
    # 协作消息
    COLLABORATE = "collaborate"             # 协作请求
    DELEGATE = "delegate"                   # 任务委托
    RESULT_SHARE = "result_share"           # 结果分享
    KNOWLEDGE_SHARE = "knowledge_share"     # 知识分享
    
    # 控制消息
    START = "start"                         # 启动指令
    STOP = "stop"                          # 停止指令
    RESTART = "restart"                     # 重启指令
    CONFIG_UPDATE = "config_update"         # 配置更新
    
    # 监控消息
    METRICS = "metrics"                     # 性能指标
    LOG = "log"                            # 日志消息
    ALERT = "alert"                        # 告警消息
    
    # 自定义消息
    CUSTOM = "custom"                       # 自定义类型


class Priority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Message:
    """智能体消息"""
    
    # 基础字段
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.CUSTOM
    sender_id: str = ""
    recipient_id: str = ""
    content: Any = None
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    priority: Priority = Priority.NORMAL
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None  # 关联ID，用于请求-响应配对
    
    # 路由信息
    routing_key: Optional[str] = None
    broadcast: bool = False  # 是否广播消息
    
    # 附加数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: Dict[str, Any] = field(default_factory=dict)
    
    # 协议版本
    protocol_version: str = "1.0"
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保message_type是枚举类型
        if isinstance(self.message_type, str):
            try:
                self.message_type = MessageType(self.message_type)
            except ValueError:
                self.message_type = MessageType.CUSTOM
        
        # 确保priority是枚举类型
        if isinstance(self.priority, (int, str)):
            if isinstance(self.priority, int):
                priority_values = [p.value for p in Priority]
                if self.priority in priority_values:
                    self.priority = Priority(self.priority)
                else:
                    self.priority = Priority.NORMAL
            else:
                try:
                    self.priority = Priority[self.priority.upper()]
                except (KeyError, AttributeError):
                    self.priority = Priority.NORMAL
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)
    
    def add_attachment(self, name: str, data: Any) -> None:
        """添加附件"""
        self.attachments[name] = data
    
    def get_attachment(self, name: str, default: Any = None) -> Any:
        """获取附件"""
        return self.attachments.get(name, default)
    
    def create_response(self, content: Any = None, success: bool = True,
                       error: str = None) -> 'Response':
        """创建响应消息"""
        return Response(
            message_id=str(uuid.uuid4()),
            correlation_id=self.message_id,
            sender_id="",  # 应由发送者设置
            recipient_id=self.sender_id,
            success=success,
            content=content,
            error=error,
            timestamp=datetime.now()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'correlation_id': self.correlation_id,
            'routing_key': self.routing_key,
            'broadcast': self.broadcast,
            'metadata': self.metadata,
            'attachments': self.attachments,
            'protocol_version': self.protocol_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        # 处理时间字段
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        expires_at = data.get('expires_at')
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            message_type=data.get('message_type', MessageType.CUSTOM),
            sender_id=data.get('sender_id', ''),
            recipient_id=data.get('recipient_id', ''),
            content=data.get('content'),
            timestamp=timestamp or datetime.now(),
            priority=data.get('priority', Priority.NORMAL),
            expires_at=expires_at,
            correlation_id=data.get('correlation_id'),
            routing_key=data.get('routing_key'),
            broadcast=data.get('broadcast', False),
            metadata=data.get('metadata', {}),
            attachments=data.get('attachments', {}),
            protocol_version=data.get('protocol_version', '1.0')
        )


@dataclass
class Response:
    """响应消息"""
    
    # 基础字段
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""  # 对应的请求消息ID
    sender_id: str = ""
    recipient_id: str = ""
    
    # 响应内容
    success: bool = True
    content: Any = None
    error: Optional[str] = None
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None  # 处理耗时（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 协议版本
    protocol_version: str = "1.0"
    
    def add_metadata(self, key: str, value: Any) -> None:
        """添加元数据"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'message_id': self.message_id,
            'correlation_id': self.correlation_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'success': self.success,
            'content': self.content,
            'error': self.error,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'protocol_version': self.protocol_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Response':
        """从字典创建响应"""
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            correlation_id=data.get('correlation_id', ''),
            sender_id=data.get('sender_id', ''),
            recipient_id=data.get('recipient_id', ''),
            success=data.get('success', True),
            content=data.get('content'),
            error=data.get('error'),
            timestamp=timestamp or datetime.now(),
            processing_time=data.get('processing_time'),
            metadata=data.get('metadata', {}),
            protocol_version=data.get('protocol_version', '1.0')
        )


class MessageBuilder:
    """消息构建器"""
    
    @staticmethod
    def create_ping(sender_id: str, recipient_id: str = "") -> Message:
        """创建ping消息"""
        return Message(
            message_type=MessageType.PING,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content="ping"
        )
    
    @staticmethod
    def create_pong(sender_id: str, recipient_id: str, correlation_id: str) -> Response:
        """创建pong响应"""
        return Response(
            correlation_id=correlation_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            success=True,
            content="pong"
        )
    
    @staticmethod
    def create_task_request(sender_id: str, recipient_id: str, 
                           task_type: str, task_data: Any,
                           priority: Priority = Priority.NORMAL) -> Message:
        """创建任务请求"""
        return Message(
            message_type=MessageType.TASK_REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content={
                'task_type': task_type,
                'task_data': task_data
            },
            priority=priority
        )
    
    @staticmethod
    def create_task_response(sender_id: str, recipient_id: str,
                            correlation_id: str, success: bool,
                            result: Any = None, error: str = None) -> Response:
        """创建任务响应"""
        return Response(
            correlation_id=correlation_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            success=success,
            content=result,
            error=error
        )
    
    @staticmethod
    def create_collaborate_request(sender_id: str, recipient_id: str,
                                  collaboration_type: str, data: Any,
                                  priority: Priority = Priority.NORMAL) -> Message:
        """创建协作请求"""
        return Message(
            message_type=MessageType.COLLABORATE,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content={
                'collaboration_type': collaboration_type,
                'data': data
            },
            priority=priority
        )
    
    @staticmethod
    def create_broadcast(sender_id: str, message_type: MessageType,
                        content: Any, routing_key: str = None) -> Message:
        """创建广播消息"""
        return Message(
            message_type=message_type,
            sender_id=sender_id,
            recipient_id="",  # 广播消息无特定接收者
            content=content,
            broadcast=True,
            routing_key=routing_key
        )
    
    @staticmethod
    def create_status_request(sender_id: str, recipient_id: str) -> Message:
        """创建状态查询"""
        return Message(
            message_type=MessageType.STATUS,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content="status_request"
        )
    
    @staticmethod
    def create_delegate_request(sender_id: str, recipient_id: str,
                               original_task: Dict[str, Any],
                               delegation_reason: str = "") -> Message:
        """创建任务委托请求"""
        return Message(
            message_type=MessageType.DELEGATE,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content={
                'original_task': original_task,
                'delegation_reason': delegation_reason,
                'delegated_at': datetime.now().isoformat()
            },
            priority=Priority.HIGH
        )
    
    @staticmethod
    def create_knowledge_share(sender_id: str, recipient_id: str,
                              knowledge_type: str, knowledge_data: Any,
                              context: str = "") -> Message:
        """创建知识分享消息"""
        return Message(
            message_type=MessageType.KNOWLEDGE_SHARE,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content={
                'knowledge_type': knowledge_type,
                'knowledge_data': knowledge_data,
                'context': context,
                'shared_at': datetime.now().isoformat()
            }
        )


class MessageValidator:
    """消息验证器"""
    
    @staticmethod
    def validate_message(message: Message) -> tuple[bool, Optional[str]]:
        """验证消息格式"""
        # 检查必需字段
        if not message.message_id:
            return False, "消息ID不能为空"
        
        if not message.sender_id:
            return False, "发送者ID不能为空"
        
        if not message.message_type:
            return False, "消息类型不能为空"
        
        # 检查消息是否过期
        if message.is_expired():
            return False, "消息已过期"
        
        # 检查协议版本
        if not message.protocol_version:
            return False, "协议版本不能为空"
        
        # 检查广播消息
        if message.broadcast and message.recipient_id:
            return False, "广播消息不应指定特定接收者"
        
        return True, None
    
    @staticmethod
    def validate_response(response: Response) -> tuple[bool, Optional[str]]:
        """验证响应格式"""
        # 检查必需字段
        if not response.message_id:
            return False, "响应ID不能为空"
        
        if not response.correlation_id:
            return False, "关联ID不能为空"
        
        if not response.sender_id:
            return False, "发送者ID不能为空"
        
        # 检查错误响应
        if not response.success and not response.error:
            return False, "失败响应必须包含错误信息"
        
        return True, None


# 预定义消息模板
class MessageTemplates:
    """常用消息模板"""
    
    @staticmethod
    def heartbeat(sender_id: str, recipient_id: str = "") -> Message:
        """心跳消息"""
        return MessageBuilder.create_ping(sender_id, recipient_id)
    
    @staticmethod
    def shutdown_notice(sender_id: str) -> Message:
        """关闭通知"""
        return MessageBuilder.create_broadcast(
            sender_id=sender_id,
            message_type=MessageType.STOP,
            content="智能体即将关闭",
            routing_key="system.shutdown"
        )
    
    @staticmethod
    def error_alert(sender_id: str, error_message: str, 
                   error_level: str = "ERROR") -> Message:
        """错误告警"""
        return MessageBuilder.create_broadcast(
            sender_id=sender_id,
            message_type=MessageType.ALERT,
            content={
                'level': error_level,
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            },
            routing_key=f"alert.{error_level.lower()}"
        )
    
    @staticmethod
    def metrics_report(sender_id: str, metrics_data: Dict[str, Any]) -> Message:
        """性能指标报告"""
        return MessageBuilder.create_broadcast(
            sender_id=sender_id,
            message_type=MessageType.METRICS,
            content=metrics_data,
            routing_key="metrics.performance"
        )