"""
智能体记忆系统模块

提供智能体的记忆功能，包括：
- 短期记忆（工作记忆）
- 长期记忆（持久化存储）
- 对话历史管理
- 知识库集成
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import hashlib
import asyncio
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """记忆项"""
    id: str
    content: Any
    content_type: str  # text, json, binary等
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: datetime = field(default_factory=datetime.now)
    accessed_time: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5  # 重要性评分 0-1
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'content': self.content,
            'content_type': self.content_type,
            'metadata': self.metadata,
            'created_time': self.created_time.isoformat(),
            'accessed_time': self.accessed_time.isoformat(),
            'access_count': self.access_count,
            'importance': self.importance,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """从字典创建"""
        return cls(
            id=data['id'],
            content=data['content'],
            content_type=data['content_type'],
            metadata=data.get('metadata', {}),
            created_time=datetime.fromisoformat(data['created_time']),
            accessed_time=datetime.fromisoformat(data['accessed_time']),
            access_count=data.get('access_count', 0),
            importance=data.get('importance', 0.5),
            tags=data.get('tags', [])
        )


class MemoryBackend(ABC):
    """记忆后端抽象接口"""
    
    @abstractmethod
    async def store(self, key: str, item: MemoryItem) -> bool:
        """存储记忆项"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[MemoryItem]:
        """检索记忆项"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除记忆项"""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """搜索记忆项"""
        pass
    
    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """列出键"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """清空记忆"""
        pass


class InMemoryBackend(MemoryBackend):
    """内存记忆后端"""
    
    def __init__(self):
        self._storage: Dict[str, MemoryItem] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, key: str, item: MemoryItem) -> bool:
        """存储记忆项"""
        async with self._lock:
            self._storage[key] = item
            return True
    
    async def retrieve(self, key: str) -> Optional[MemoryItem]:
        """检索记忆项"""
        async with self._lock:
            item = self._storage.get(key)
            if item:
                item.accessed_time = datetime.now()
                item.access_count += 1
            return item
    
    async def delete(self, key: str) -> bool:
        """删除记忆项"""
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """搜索记忆项"""
        async with self._lock:
            results = []
            query_lower = query.lower()
            
            for item in self._storage.values():
                # 简单的文本匹配搜索
                content_str = str(item.content).lower()
                if (query_lower in content_str or 
                    any(query_lower in tag.lower() for tag in item.tags) or
                    any(query_lower in str(v).lower() for v in item.metadata.values())):
                    results.append(item)
                    
                    if len(results) >= limit:
                        break
            
            # 按重要性和访问频率排序
            results.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
            return results
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """列出键"""
        async with self._lock:
            if pattern == "*":
                return list(self._storage.keys())
            
            # 简单的通配符匹配
            import fnmatch
            return [key for key in self._storage.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def clear(self) -> bool:
        """清空记忆"""
        async with self._lock:
            self._storage.clear()
            return True


class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, max_turns: int = 100):
        self.max_turns = max_turns
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_turns))
        self._lock = asyncio.Lock()
    
    async def add_message(self, conversation_id: str, role: str, content: str, 
                         metadata: Dict[str, Any] = None) -> None:
        """添加对话消息"""
        async with self._lock:
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.conversations[conversation_id].append(message)
    
    async def get_conversation(self, conversation_id: str, 
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取对话历史"""
        async with self._lock:
            messages = list(self.conversations[conversation_id])
            if limit:
                messages = messages[-limit:]
            return messages
    
    async def clear_conversation(self, conversation_id: str) -> None:
        """清空对话"""
        async with self._lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id].clear()
    
    async def get_recent_context(self, conversation_id: str, 
                                context_turns: int = 5) -> str:
        """获取最近的对话上下文"""
        messages = await self.get_conversation(conversation_id, context_turns * 2)
        
        context_parts = []
        for msg in messages:
            context_parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context_parts)


class WorkingMemory:
    """工作记忆 - 短期记忆管理"""
    
    def __init__(self, capacity: int = 50, ttl: int = 3600):
        self.capacity = capacity
        self.ttl = ttl  # 生存时间（秒）
        self._memory: deque = deque(maxlen=capacity)
        self._index: Dict[str, MemoryItem] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, key: str, content: Any, content_type: str = "text",
                   importance: float = 0.5, tags: List[str] = None) -> None:
        """存储到工作记忆"""
        async with self._lock:
            # 创建记忆项
            item = MemoryItem(
                id=key,
                content=content,
                content_type=content_type,
                importance=importance,
                tags=tags or []
            )
            
            # 如果已存在，先移除
            if key in self._index:
                old_item = self._index[key]
                try:
                    self._memory.remove(old_item)
                except ValueError:
                    pass
            
            # 添加新项
            self._memory.append(item)
            self._index[key] = item
            
            # 清理过期项
            await self._cleanup_expired()
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """从工作记忆检索"""
        async with self._lock:
            item = self._index.get(key)
            if item and not self._is_expired(item):
                item.accessed_time = datetime.now()
                item.access_count += 1
                return item.content
            return None
    
    async def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any]]:
        """搜索工作记忆"""
        async with self._lock:
            results = []
            query_lower = query.lower()
            
            for item in self._memory:
                if self._is_expired(item):
                    continue
                
                content_str = str(item.content).lower()
                if (query_lower in content_str or 
                    any(query_lower in tag.lower() for tag in item.tags)):
                    results.append((item.id, item.content))
                    
                    if len(results) >= limit:
                        break
            
            return results
    
    def _is_expired(self, item: MemoryItem) -> bool:
        """检查项是否过期"""
        age = (datetime.now() - item.created_time).total_seconds()
        return age > self.ttl
    
    async def _cleanup_expired(self) -> None:
        """清理过期项"""
        now = datetime.now()
        expired_keys = []
        
        for key, item in self._index.items():
            if self._is_expired(item):
                expired_keys.append(key)
        
        for key in expired_keys:
            item = self._index.pop(key)
            try:
                self._memory.remove(item)
            except ValueError:
                pass
    
    async def clear(self) -> None:
        """清空工作记忆"""
        async with self._lock:
            self._memory.clear()
            self._index.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        async with self._lock:
            total_items = len(self._memory)
            expired_count = sum(1 for item in self._memory if self._is_expired(item))
            
            return {
                'total_items': total_items,
                'active_items': total_items - expired_count,
                'expired_items': expired_count,
                'capacity': self.capacity,
                'utilization': total_items / self.capacity if self.capacity > 0 else 0
            }


class AgentMemory:
    """智能体记忆系统主类"""
    
    def __init__(self, agent_id: str, backend: Optional[MemoryBackend] = None,
                 working_memory_capacity: int = 50, conversation_memory_turns: int = 100):
        self.agent_id = agent_id
        self.backend = backend or InMemoryBackend()
        
        # 短期记忆
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        
        # 对话记忆
        self.conversation_memory = ConversationMemory(max_turns=conversation_memory_turns)
        
        # 知识缓存
        self._knowledge_cache: Dict[str, Any] = {}
        self._cache_lock = asyncio.Lock()
    
    async def store_short_term(self, key: str, content: Any, 
                              importance: float = 0.5, tags: List[str] = None) -> None:
        """存储短期记忆"""
        await self.working_memory.store(key, content, importance=importance, tags=tags)
    
    async def store_long_term(self, key: str, content: Any, content_type: str = "text",
                             importance: float = 0.5, tags: List[str] = None,
                             metadata: Dict[str, Any] = None) -> bool:
        """存储长期记忆"""
        item = MemoryItem(
            id=key,
            content=content,
            content_type=content_type,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        return await self.backend.store(f"{self.agent_id}:{key}", item)
    
    async def retrieve(self, key: str, check_long_term: bool = True) -> Optional[Any]:
        """检索记忆（先短期，后长期）"""
        # 先检查短期记忆
        result = await self.working_memory.retrieve(key)
        if result is not None:
            return result
        
        # 检查长期记忆
        if check_long_term:
            item = await self.backend.retrieve(f"{self.agent_id}:{key}")
            if item:
                # 升级到短期记忆
                await self.working_memory.store(key, item.content, 
                                               importance=item.importance, tags=item.tags)
                return item.content
        
        return None
    
    async def search_memory(self, query: str, limit: int = 10, 
                           include_long_term: bool = True) -> List[Tuple[str, Any]]:
        """搜索记忆"""
        results = []
        
        # 搜索短期记忆
        short_term_results = await self.working_memory.search(query, limit)
        results.extend(short_term_results)
        
        # 搜索长期记忆
        if include_long_term and len(results) < limit:
            long_term_items = await self.backend.search(query, limit - len(results))
            for item in long_term_items:
                if item.id.startswith(f"{self.agent_id}:"):
                    key = item.id[len(f"{self.agent_id}:"):]
                    results.append((key, item.content))
        
        return results[:limit]
    
    async def add_conversation_turn(self, conversation_id: str, role: str, 
                                  content: str, metadata: Dict[str, Any] = None) -> None:
        """添加对话轮次"""
        await self.conversation_memory.add_message(conversation_id, role, content, metadata)
    
    async def get_conversation_history(self, conversation_id: str, 
                                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return await self.conversation_memory.get_conversation(conversation_id, limit)
    
    async def get_conversation_context(self, conversation_id: str, 
                                     context_turns: int = 5) -> str:
        """获取对话上下文"""
        return await self.conversation_memory.get_recent_context(conversation_id, context_turns)
    
    async def cache_knowledge(self, key: str, value: Any, ttl: int = 3600) -> None:
        """缓存知识"""
        async with self._cache_lock:
            expire_time = datetime.now() + timedelta(seconds=ttl)
            self._knowledge_cache[key] = {
                'value': value,
                'expire_time': expire_time
            }
    
    async def get_cached_knowledge(self, key: str) -> Optional[Any]:
        """获取缓存的知识"""
        async with self._cache_lock:
            cache_item = self._knowledge_cache.get(key)
            if cache_item:
                if datetime.now() < cache_item['expire_time']:
                    return cache_item['value']
                else:
                    # 过期，删除缓存
                    del self._knowledge_cache[key]
            return None
    
    async def forget(self, key: str, forget_long_term: bool = False) -> bool:
        """遗忘记忆"""
        # 从短期记忆中删除
        await self.working_memory.retrieve(key)  # 这会触发清理
        
        # 从长期记忆中删除
        if forget_long_term:
            return await self.backend.delete(f"{self.agent_id}:{key}")
        
        return True
    
    async def clear_all_memory(self, include_long_term: bool = False) -> None:
        """清空所有记忆"""
        await self.working_memory.clear()
        
        if include_long_term:
            # 删除该智能体的所有长期记忆
            keys = await self.backend.list_keys(f"{self.agent_id}:*")
            for key in keys:
                await self.backend.delete(key)
        
        # 清空知识缓存
        async with self._cache_lock:
            self._knowledge_cache.clear()
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        working_stats = await self.working_memory.get_stats()
        
        # 统计长期记忆
        long_term_keys = await self.backend.list_keys(f"{self.agent_id}:*")
        
        # 统计对话记忆
        total_conversations = len(self.conversation_memory.conversations)
        total_messages = sum(len(conv) for conv in self.conversation_memory.conversations.values())
        
        return {
            'agent_id': self.agent_id,
            'working_memory': working_stats,
            'long_term_memory': {
                'total_items': len(long_term_keys)
            },
            'conversation_memory': {
                'total_conversations': total_conversations,
                'total_messages': total_messages
            },
            'knowledge_cache': {
                'total_items': len(self._knowledge_cache)
            }
        }
    
    async def export_memory(self, include_conversations: bool = True) -> Dict[str, Any]:
        """导出记忆数据"""
        export_data = {
            'agent_id': self.agent_id,
            'export_time': datetime.now().isoformat(),
            'long_term_memory': {}
        }
        
        # 导出长期记忆
        long_term_keys = await self.backend.list_keys(f"{self.agent_id}:*")
        for key in long_term_keys:
            item = await self.backend.retrieve(key)
            if item:
                clean_key = key[len(f"{self.agent_id}:"):]
                export_data['long_term_memory'][clean_key] = item.to_dict()
        
        # 导出对话记忆
        if include_conversations:
            export_data['conversations'] = {}
            for conv_id, messages in self.conversation_memory.conversations.items():
                export_data['conversations'][conv_id] = list(messages)
        
        return export_data
    
    async def import_memory(self, import_data: Dict[str, Any]) -> bool:
        """导入记忆数据"""
        try:
            # 导入长期记忆
            long_term_data = import_data.get('long_term_memory', {})
            for key, item_dict in long_term_data.items():
                item = MemoryItem.from_dict(item_dict)
                await self.backend.store(f"{self.agent_id}:{key}", item)
            
            # 导入对话记忆
            conversations_data = import_data.get('conversations', {})
            for conv_id, messages in conversations_data.items():
                self.conversation_memory.conversations[conv_id].clear()
                for msg in messages:
                    await self.conversation_memory.add_message(
                        conv_id, msg['role'], msg['content'], msg.get('metadata', {})
                    )
            
            logger.info(f"成功导入智能体 {self.agent_id} 的记忆数据")
            return True
            
        except Exception as e:
            logger.error(f"导入记忆数据失败: {e}")
            return False