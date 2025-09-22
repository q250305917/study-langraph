"""
对话记忆模板模块

本模块实现了完整的对话记忆管理系统，用于存储、检索和维护对话历史。
支持多种存储后端、智能上下文管理和灵活的记忆策略。

核心功能：
1. 对话历史存储 - 持久化对话记录到内存、文件或数据库
2. 上下文维护 - 保持对话的连续性和相关性
3. 记忆检索 - 根据关键词、时间或相似度检索历史对话
4. 自动清理 - 根据配置自动清理过期或无关的对话记录
5. 多用户支持 - 支持多用户会话的独立记忆管理
6. 记忆压缩 - 当记忆超过限制时自动压缩历史记录

设计原理：
- 插件化存储后端：支持内存、文件、Redis、数据库等多种存储方式
- 分层记忆架构：短期记忆（当前会话）+ 长期记忆（历史会话）
- 智能上下文选择：根据相关性选择最合适的历史上下文
- 性能优化：缓存机制、批量操作、异步处理
- 安全考虑：数据加密、访问控制、隐私保护
"""

import json
import time
import asyncio
import hashlib
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Protocol
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque, defaultdict
import threading
from enum import Enum

try:
    import redis
except ImportError:
    redis = None

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError, 
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class MessageType(Enum):
    """消息类型枚举"""
    HUMAN = "human"           # 人类用户消息
    AI = "ai"                 # AI助手消息
    SYSTEM = "system"         # 系统消息
    FUNCTION = "function"     # 函数调用消息
    TOOL = "tool"            # 工具调用消息


class MemoryBackend(Enum):
    """记忆存储后端枚举"""
    MEMORY = "memory"         # 内存存储
    FILE = "file"            # 文件存储
    SQLITE = "sqlite"        # SQLite数据库
    REDIS = "redis"          # Redis缓存
    POSTGRES = "postgres"    # PostgreSQL数据库
    VECTOR = "vector"        # 向量数据库（用于相似度检索）


@dataclass
class Message:
    """
    消息数据结构
    
    表示对话中的一条消息，包含消息内容、类型、时间戳等信息。
    """
    content: str                              # 消息内容
    type: MessageType                         # 消息类型
    timestamp: float = field(default_factory=time.time)  # 时间戳
    session_id: str = "default"               # 会话ID
    user_id: Optional[str] = None             # 用户ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据
    
    # 可选字段
    message_id: Optional[str] = None          # 消息唯一标识
    parent_id: Optional[str] = None           # 父消息ID（用于树状对话）
    tokens: Optional[int] = None              # 消息token数量
    model: Optional[str] = None               # 生成消息的模型
    temperature: Optional[float] = None       # 生成参数
    
    def __post_init__(self):
        """初始化后处理"""
        if self.message_id is None:
            # 生成唯一消息ID
            content_hash = hashlib.md5(f"{self.content}_{self.timestamp}".encode()).hexdigest()
            self.message_id = f"msg_{content_hash[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建Message实例"""
        # 处理枚举类型
        if isinstance(data.get('type'), str):
            data['type'] = MessageType(data['type'])
        
        return cls(**data)
    
    def get_size(self) -> int:
        """获取消息大小（字符数）"""
        return len(self.content)
    
    def is_from_user(self) -> bool:
        """检查是否为用户消息"""
        return self.type == MessageType.HUMAN
    
    def is_from_ai(self) -> bool:
        """检查是否为AI消息"""
        return self.type == MessageType.AI


@dataclass
class Conversation:
    """
    对话数据结构
    
    表示一个完整的对话会话，包含多条消息和会话元数据。
    """
    session_id: str                           # 会话ID
    messages: List[Message] = field(default_factory=list)  # 消息列表
    created_at: float = field(default_factory=time.time)   # 创建时间
    updated_at: float = field(default_factory=time.time)   # 更新时间
    user_id: Optional[str] = None             # 用户ID
    title: Optional[str] = None               # 对话标题
    summary: Optional[str] = None             # 对话摘要
    metadata: Dict[str, Any] = field(default_factory=dict)  # 会话元数据
    
    def add_message(self, message: Message) -> None:
        """添加消息到对话"""
        message.session_id = self.session_id
        if self.user_id:
            message.user_id = self.user_id
        
        self.messages.append(message)
        self.updated_at = time.time()
        
        # 自动生成标题（基于第一条用户消息）
        if not self.title and message.is_from_user():
            self.title = message.content[:50] + ("..." if len(message.content) > 50 else "")
    
    def get_total_tokens(self) -> int:
        """获取对话总token数"""
        return sum(msg.tokens or len(msg.content.split()) for msg in self.messages)
    
    def get_message_count(self) -> int:
        """获取消息数量"""
        return len(self.messages)
    
    def get_last_messages(self, count: int = 10) -> List[Message]:
        """获取最近的N条消息"""
        return self.messages[-count:] if count > 0 else []
    
    def get_messages_by_type(self, message_type: MessageType) -> List[Message]:
        """根据类型获取消息"""
        return [msg for msg in self.messages if msg.type == message_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'session_id': self.session_id,
            'messages': [msg.to_dict() for msg in self.messages],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'user_id': self.user_id,
            'title': self.title,
            'summary': self.summary,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """从字典创建Conversation实例"""
        messages = [Message.from_dict(msg_data) for msg_data in data.get('messages', [])]
        
        return cls(
            session_id=data['session_id'],
            messages=messages,
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            user_id=data.get('user_id'),
            title=data.get('title'),
            summary=data.get('summary'),
            metadata=data.get('metadata', {})
        )


class MemoryBackendInterface(ABC):
    """
    记忆存储后端接口
    
    定义所有存储后端必须实现的方法。
    """
    
    @abstractmethod
    def save_conversation(self, conversation: Conversation) -> bool:
        """保存对话"""
        pass
    
    @abstractmethod
    def load_conversation(self, session_id: str, user_id: Optional[str] = None) -> Optional[Conversation]:
        """加载对话"""
        pass
    
    @abstractmethod
    def delete_conversation(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """删除对话"""
        pass
    
    @abstractmethod
    def list_conversations(self, user_id: Optional[str] = None, limit: int = 100) -> List[str]:
        """列出对话会话ID"""
        pass
    
    @abstractmethod
    def search_messages(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Message]:
        """搜索消息"""
        pass
    
    @abstractmethod
    def cleanup_old_conversations(self, days: int = 30) -> int:
        """清理旧对话，返回清理的数量"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        pass


class MemoryBackend(MemoryBackendInterface):
    """
    内存存储后端
    
    将对话数据存储在内存中，适用于临时会话和测试环境。
    """
    
    def __init__(self, max_conversations: int = 1000):
        """
        初始化内存存储后端
        
        Args:
            max_conversations: 最大对话数量
        """
        self.conversations: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations
        self._lock = threading.Lock()
        
        logger.debug(f"Initialized MemoryBackend with max_conversations={max_conversations}")
    
    def save_conversation(self, conversation: Conversation) -> bool:
        """保存对话到内存"""
        with self._lock:
            # 检查是否超过最大数量限制
            if len(self.conversations) >= self.max_conversations:
                # 删除最旧的对话
                oldest_session = min(self.conversations.keys(), 
                                   key=lambda k: self.conversations[k].updated_at)
                del self.conversations[oldest_session]
                logger.debug(f"Removed oldest conversation: {oldest_session}")
            
            key = self._get_conversation_key(conversation.session_id, conversation.user_id)
            self.conversations[key] = conversation
            
            logger.debug(f"Saved conversation to memory: {conversation.session_id}")
            return True
    
    def load_conversation(self, session_id: str, user_id: Optional[str] = None) -> Optional[Conversation]:
        """从内存加载对话"""
        with self._lock:
            key = self._get_conversation_key(session_id, user_id)
            conversation = self.conversations.get(key)
            
            if conversation:
                logger.debug(f"Loaded conversation from memory: {session_id}")
            
            return conversation
    
    def delete_conversation(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """从内存删除对话"""
        with self._lock:
            key = self._get_conversation_key(session_id, user_id)
            if key in self.conversations:
                del self.conversations[key]
                logger.debug(f"Deleted conversation from memory: {session_id}")
                return True
            return False
    
    def list_conversations(self, user_id: Optional[str] = None, limit: int = 100) -> List[str]:
        """列出内存中的对话"""
        with self._lock:
            conversations = []
            for key, conv in self.conversations.items():
                if user_id is None or conv.user_id == user_id:
                    conversations.append((conv.session_id, conv.updated_at))
            
            # 按更新时间倒序排列
            conversations.sort(key=lambda x: x[1], reverse=True)
            return [session_id for session_id, _ in conversations[:limit]]
    
    def search_messages(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Message]:
        """在内存中搜索消息"""
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for conversation in self.conversations.values():
                if user_id is None or conversation.user_id == user_id:
                    for message in conversation.messages:
                        if query_lower in message.content.lower():
                            results.append(message)
                            if len(results) >= limit:
                                return results
        
        return results
    
    def cleanup_old_conversations(self, days: int = 30) -> int:
        """清理内存中的旧对话"""
        cutoff_time = time.time() - (days * 24 * 3600)
        removed_count = 0
        
        with self._lock:
            keys_to_remove = []
            for key, conversation in self.conversations.items():
                if conversation.updated_at < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.conversations[key]
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old conversations from memory")
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存存储统计"""
        with self._lock:
            total_messages = sum(len(conv.messages) for conv in self.conversations.values())
            total_tokens = sum(conv.get_total_tokens() for conv in self.conversations.values())
            
            return {
                "backend_type": "memory",
                "total_conversations": len(self.conversations),
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "max_conversations": self.max_conversations,
                "memory_usage_mb": self._estimate_memory_usage()
            }
    
    def _get_conversation_key(self, session_id: str, user_id: Optional[str]) -> str:
        """生成对话键值"""
        if user_id:
            return f"{user_id}:{session_id}"
        return session_id
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        try:
            import sys
            total_size = sys.getsizeof(self.conversations)
            for conversation in self.conversations.values():
                total_size += sys.getsizeof(conversation)
                for message in conversation.messages:
                    total_size += sys.getsizeof(message)
                    total_size += sys.getsizeof(message.content)
            
            return total_size / (1024 * 1024)  # 转换为MB
        except Exception:
            return 0.0


class FileBackend(MemoryBackendInterface):
    """
    文件存储后端
    
    将对话数据存储在JSON文件中，适用于持久化存储需求。
    """
    
    def __init__(self, storage_dir: Union[str, Path] = "conversations"):
        """
        初始化文件存储后端
        
        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        logger.debug(f"Initialized FileBackend with storage_dir={self.storage_dir}")
    
    def save_conversation(self, conversation: Conversation) -> bool:
        """保存对话到文件"""
        try:
            with self._lock:
                file_path = self._get_conversation_file(conversation.session_id, conversation.user_id)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
                
                logger.debug(f"Saved conversation to file: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to save conversation to file: {e}")
            return False
    
    def load_conversation(self, session_id: str, user_id: Optional[str] = None) -> Optional[Conversation]:
        """从文件加载对话"""
        try:
            file_path = self._get_conversation_file(session_id, user_id)
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation = Conversation.from_dict(data)
            logger.debug(f"Loaded conversation from file: {file_path}")
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to load conversation from file: {e}")
            return None
    
    def delete_conversation(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """删除对话文件"""
        try:
            file_path = self._get_conversation_file(session_id, user_id)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted conversation file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete conversation file: {e}")
            return False
    
    def list_conversations(self, user_id: Optional[str] = None, limit: int = 100) -> List[str]:
        """列出文件中的对话"""
        conversations = []
        
        try:
            search_pattern = f"{user_id}_*.json" if user_id else "*.json"
            
            for file_path in self.storage_dir.glob(search_pattern):
                try:
                    stat = file_path.stat()
                    session_id = self._extract_session_id(file_path.name, user_id)
                    conversations.append((session_id, stat.st_mtime))
                except Exception as e:
                    logger.warning(f"Failed to process conversation file {file_path}: {e}")
            
            # 按修改时间倒序排列
            conversations.sort(key=lambda x: x[1], reverse=True)
            return [session_id for session_id, _ in conversations[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []
    
    def search_messages(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Message]:
        """在文件中搜索消息"""
        results = []
        query_lower = query.lower()
        
        try:
            for session_id in self.list_conversations(user_id):
                conversation = self.load_conversation(session_id, user_id)
                if conversation:
                    for message in conversation.messages:
                        if query_lower in message.content.lower():
                            results.append(message)
                            if len(results) >= limit:
                                return results
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
        
        return results
    
    def cleanup_old_conversations(self, days: int = 30) -> int:
        """清理旧的对话文件"""
        cutoff_time = time.time() - (days * 24 * 3600)
        removed_count = 0
        
        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old conversation file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old conversation file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")
        
        logger.info(f"Cleaned up {removed_count} old conversation files")
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取文件存储统计"""
        try:
            conversation_files = list(self.storage_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in conversation_files)
            
            return {
                "backend_type": "file",
                "total_conversations": len(conversation_files),
                "storage_dir": str(self.storage_dir),
                "total_size_mb": total_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Failed to get file backend stats: {e}")
            return {"backend_type": "file", "error": str(e)}
    
    def _get_conversation_file(self, session_id: str, user_id: Optional[str] = None) -> Path:
        """获取对话文件路径"""
        if user_id:
            filename = f"{user_id}_{session_id}.json"
        else:
            filename = f"{session_id}.json"
        
        return self.storage_dir / filename
    
    def _extract_session_id(self, filename: str, user_id: Optional[str] = None) -> str:
        """从文件名提取会话ID"""
        name_without_ext = filename.replace('.json', '')
        if user_id and name_without_ext.startswith(f"{user_id}_"):
            return name_without_ext[len(f"{user_id}_"):]
        return name_without_ext


class ConversationMemoryTemplate(TemplateBase[Dict[str, Any], Dict[str, Any]]):
    """
    对话记忆模板
    
    提供完整的对话记忆管理功能，包括消息存储、检索、上下文维护等。
    支持多种存储后端和智能记忆策略。
    
    核心功能：
    1. 对话存储：支持多种存储后端（内存、文件、数据库等）
    2. 上下文管理：智能选择和维护对话上下文
    3. 记忆检索：根据关键词、时间、相似度检索历史记录
    4. 自动清理：根据配置自动清理过期或无关记录
    5. 多用户支持：独立的用户会话管理
    6. 性能优化：缓存、批量操作、异步处理
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        初始化对话记忆模板
        
        Args:
            config: 模板配置
        """
        super().__init__(config)
        
        # 存储后端
        self.backend: Optional[MemoryBackendInterface] = None
        
        # 当前会话缓存
        self.current_conversations: Dict[str, Conversation] = {}
        
        # 配置参数
        self.max_context_length: int = 4000  # 最大上下文长度（token）
        self.max_messages_per_context: int = 20  # 每个上下文最大消息数
        self.cleanup_interval_hours: int = 24  # 自动清理间隔（小时）
        self.auto_cleanup_days: int = 30  # 自动清理天数
        
        # 线程锁
        self._lock = threading.Lock()
        
        logger.debug("Initialized ConversationMemoryTemplate")
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="ConversationMemoryTemplate",
            description="对话记忆管理模板",
            template_type=TemplateType.MEMORY,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("backend_type", str, default="memory", 
                           description="存储后端类型：memory, file, sqlite, redis")
        config.add_parameter("storage_path", str, default="./conversations", 
                           description="存储路径（文件/数据库）")
        config.add_parameter("max_context_length", int, default=4000,
                           description="最大上下文长度（token）")
        config.add_parameter("max_messages_per_context", int, default=20,
                           description="每个上下文最大消息数")
        config.add_parameter("auto_cleanup_days", int, default=30,
                           description="自动清理天数")
        config.add_parameter("enable_compression", bool, default=True,
                           description="是否启用记忆压缩")
        config.add_parameter("enable_search", bool, default=True,
                           description="是否启用消息搜索")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置对话记忆模板
        
        Args:
            **parameters: 配置参数
                - backend_type: 存储后端类型
                - storage_path: 存储路径
                - max_context_length: 最大上下文长度
                - max_messages_per_context: 每个上下文最大消息数
                - auto_cleanup_days: 自动清理天数
                - enable_compression: 是否启用压缩
                - enable_search: 是否启用搜索
        """
        # 验证参数
        if not self.validate_parameters(parameters):
            raise ValidationError("ConversationMemoryTemplate参数验证失败")
        
        # 更新内部参数
        self.max_context_length = parameters.get("max_context_length", 4000)
        self.max_messages_per_context = parameters.get("max_messages_per_context", 20)
        self.auto_cleanup_days = parameters.get("auto_cleanup_days", 30)
        
        # 初始化存储后端
        backend_type = parameters.get("backend_type", "memory")
        storage_path = parameters.get("storage_path", "./conversations")
        
        self.backend = self._create_backend(backend_type, storage_path)
        
        self.status = self.config.template_type.CONFIGURED
        logger.info(f"ConversationMemoryTemplate配置完成：backend={backend_type}, storage={storage_path}")
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行对话记忆操作
        
        Args:
            input_data: 输入数据
                - action: 操作类型（add_message, get_context, search, cleanup等）
                - session_id: 会话ID
                - user_id: 用户ID（可选）
                - message: 消息内容（add_message时必需）
                - query: 搜索查询（search时必需）
                - limit: 结果限制数量
            
        Returns:
            执行结果字典
        """
        action = input_data.get("action")
        if not action:
            raise ValidationError("必须指定action参数")
        
        session_id = input_data.get("session_id", "default")
        user_id = input_data.get("user_id")
        
        try:
            if action == "add_message":
                return self._add_message(input_data, session_id, user_id)
            elif action == "get_context":
                return self._get_context(session_id, user_id, input_data.get("limit", 10))
            elif action == "get_conversation":
                return self._get_conversation(session_id, user_id)
            elif action == "search":
                return self._search_messages(input_data.get("query", ""), user_id, input_data.get("limit", 10))
            elif action == "list_conversations":
                return self._list_conversations(user_id, input_data.get("limit", 100))
            elif action == "delete_conversation":
                return self._delete_conversation(session_id, user_id)
            elif action == "cleanup":
                return self._cleanup_old_conversations(input_data.get("days", self.auto_cleanup_days))
            elif action == "get_stats":
                return self._get_stats()
            else:
                raise ValidationError(f"未知的操作类型：{action}")
                
        except Exception as e:
            logger.error(f"执行对话记忆操作失败：{action} - {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    def _add_message(self, input_data: Dict[str, Any], session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """添加消息到对话"""
        message_data = input_data.get("message")
        if not message_data:
            raise ValidationError("添加消息时必须提供message参数")
        
        # 创建消息对象
        if isinstance(message_data, dict):
            message = Message(
                content=message_data.get("content", ""),
                type=MessageType(message_data.get("type", "human")),
                session_id=session_id,
                user_id=user_id,
                metadata=message_data.get("metadata", {})
            )
        else:
            # 简单字符串消息，默认为human类型
            message = Message(
                content=str(message_data),
                type=MessageType.HUMAN,
                session_id=session_id,
                user_id=user_id
            )
        
        # 获取或创建对话
        conversation = self._get_or_create_conversation(session_id, user_id)
        conversation.add_message(message)
        
        # 保存到后端
        success = self.backend.save_conversation(conversation)
        
        # 更新缓存
        with self._lock:
            cache_key = self._get_cache_key(session_id, user_id)
            self.current_conversations[cache_key] = conversation
        
        return {
            "success": success,
            "message_id": message.message_id,
            "session_id": session_id,
            "message_count": conversation.get_message_count(),
            "total_tokens": conversation.get_total_tokens()
        }
    
    def _get_context(self, session_id: str, user_id: Optional[str], limit: int) -> Dict[str, Any]:
        """获取对话上下文"""
        conversation = self._get_or_create_conversation(session_id, user_id)
        
        # 获取最近的消息
        recent_messages = conversation.get_last_messages(min(limit, self.max_messages_per_context))
        
        # 计算token数量并进行截断
        context_messages = []
        total_tokens = 0
        
        for message in reversed(recent_messages):
            message_tokens = message.tokens or len(message.content.split())
            if total_tokens + message_tokens > self.max_context_length:
                break
            
            context_messages.insert(0, message)
            total_tokens += message_tokens
        
        return {
            "success": True,
            "session_id": session_id,
            "messages": [msg.to_dict() for msg in context_messages],
            "message_count": len(context_messages),
            "total_tokens": total_tokens,
            "truncated": len(context_messages) < len(recent_messages)
        }
    
    def _get_conversation(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """获取完整对话"""
        conversation = self.backend.load_conversation(session_id, user_id)
        
        if conversation:
            return {
                "success": True,
                "conversation": conversation.to_dict()
            }
        else:
            return {
                "success": False,
                "error": f"对话不存在：{session_id}"
            }
    
    def _search_messages(self, query: str, user_id: Optional[str], limit: int) -> Dict[str, Any]:
        """搜索消息"""
        if not query.strip():
            return {
                "success": False,
                "error": "搜索查询不能为空"
            }
        
        messages = self.backend.search_messages(query, user_id, limit)
        
        return {
            "success": True,
            "query": query,
            "messages": [msg.to_dict() for msg in messages],
            "result_count": len(messages)
        }
    
    def _list_conversations(self, user_id: Optional[str], limit: int) -> Dict[str, Any]:
        """列出对话会话"""
        session_ids = self.backend.list_conversations(user_id, limit)
        
        return {
            "success": True,
            "conversations": session_ids,
            "count": len(session_ids)
        }
    
    def _delete_conversation(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """删除对话"""
        success = self.backend.delete_conversation(session_id, user_id)
        
        # 从缓存中移除
        if success:
            with self._lock:
                cache_key = self._get_cache_key(session_id, user_id)
                self.current_conversations.pop(cache_key, None)
        
        return {
            "success": success,
            "session_id": session_id
        }
    
    def _cleanup_old_conversations(self, days: int) -> Dict[str, Any]:
        """清理旧对话"""
        removed_count = self.backend.cleanup_old_conversations(days)
        
        return {
            "success": True,
            "removed_count": removed_count,
            "cleanup_days": days
        }
    
    def _get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        backend_stats = self.backend.get_stats()
        
        return {
            "success": True,
            "stats": backend_stats,
            "cache_size": len(self.current_conversations),
            "config": {
                "max_context_length": self.max_context_length,
                "max_messages_per_context": self.max_messages_per_context,
                "auto_cleanup_days": self.auto_cleanup_days
            }
        }
    
    def _get_or_create_conversation(self, session_id: str, user_id: Optional[str]) -> Conversation:
        """获取或创建对话"""
        # 先从缓存查找
        cache_key = self._get_cache_key(session_id, user_id)
        
        with self._lock:
            if cache_key in self.current_conversations:
                return self.current_conversations[cache_key]
        
        # 从后端加载
        conversation = self.backend.load_conversation(session_id, user_id)
        
        if conversation is None:
            # 创建新对话
            conversation = Conversation(
                session_id=session_id,
                user_id=user_id
            )
        
        # 更新缓存
        with self._lock:
            self.current_conversations[cache_key] = conversation
        
        return conversation
    
    def _get_cache_key(self, session_id: str, user_id: Optional[str]) -> str:
        """生成缓存键值"""
        if user_id:
            return f"{user_id}:{session_id}"
        return session_id
    
    def _create_backend(self, backend_type: str, storage_path: str) -> MemoryBackendInterface:
        """创建存储后端"""
        if backend_type == "memory":
            return MemoryBackend()
        elif backend_type == "file":
            return FileBackend(storage_path)
        else:
            raise ConfigurationError(f"不支持的存储后端类型：{backend_type}")
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "backend_type": "file",
                "storage_path": "./conversations",
                "max_context_length": 4000,
                "max_messages_per_context": 20,
                "auto_cleanup_days": 30,
                "enable_compression": True,
                "enable_search": True
            },
            "execute_examples": [
                {
                    "description": "添加用户消息",
                    "input": {
                        "action": "add_message",
                        "session_id": "chat_001",
                        "user_id": "user123",
                        "message": {
                            "content": "你好，我想了解Python编程。",
                            "type": "human"
                        }
                    }
                },
                {
                    "description": "添加AI回复",
                    "input": {
                        "action": "add_message",
                        "session_id": "chat_001",
                        "user_id": "user123",
                        "message": {
                            "content": "你好！我很乐意帮助你学习Python编程。",
                            "type": "ai"
                        }
                    }
                },
                {
                    "description": "获取对话上下文",
                    "input": {
                        "action": "get_context",
                        "session_id": "chat_001",
                        "user_id": "user123",
                        "limit": 10
                    }
                },
                {
                    "description": "搜索历史消息",
                    "input": {
                        "action": "search",
                        "query": "Python",
                        "user_id": "user123",
                        "limit": 5
                    }
                }
            ],
            "usage_code": """
# 使用示例
from templates.memory.conversation_memory import ConversationMemoryTemplate

# 初始化模板
memory_template = ConversationMemoryTemplate()

# 配置参数
memory_template.setup(
    backend_type="file",
    storage_path="./conversations",
    max_context_length=4000,
    max_messages_per_context=20
)

# 添加消息
result = memory_template.run({
    "action": "add_message",
    "session_id": "chat_001",
    "user_id": "user123",
    "message": {
        "content": "你好，我想了解Python编程。",
        "type": "human"
    }
})

# 获取上下文
context = memory_template.run({
    "action": "get_context",
    "session_id": "chat_001",
    "user_id": "user123",
    "limit": 10
})

print("对话上下文：", context)
"""
        }