"""
智能体通信模块

提供智能体间通信的基础设施，包括：
- 消息定义和序列化
- 通信协议实现
- 消息路由和分发
- 通信通道管理
"""

from .protocols import Message, Response, MessageType
from .channels import CommunicationChannel, MemoryChannel, RedisChannel
from .router import MessageRouter
from .serialization import MessageSerializer

__all__ = [
    "Message",
    "Response", 
    "MessageType",
    "CommunicationChannel",
    "MemoryChannel",
    "RedisChannel",
    "MessageRouter",
    "MessageSerializer"
]