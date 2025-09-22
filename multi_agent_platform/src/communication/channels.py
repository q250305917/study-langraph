"""
通信通道实现模块

提供不同类型的通信通道实现，包括：
- 内存通道（单进程）
- Redis通道（分布式）
- WebSocket通道（实时）
- 消息队列通道（异步）
"""

from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import asyncio
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid

from .protocols import Message, Response, MessageType, Priority
from .serialization import MessageSerializer

logger = logging.getLogger(__name__)


class CommunicationChannel(ABC):
    """通信通道抽象基类"""
    
    def __init__(self, channel_id: str = None):
        self.channel_id = channel_id or str(uuid.uuid4())
        self.is_connected = False
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.serializer = MessageSerializer()
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接通道"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开通道"""
        pass
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        pass
    
    @abstractmethod
    async def send_response(self, response: Response) -> bool:
        """发送响应"""
        pass
    
    @abstractmethod
    async def receive_messages(self, timeout: float = None) -> List[Union[Message, Response]]:
        """接收消息"""
        pass
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """注册消息处理器"""
        type_key = message_type.value if isinstance(message_type, MessageType) else message_type
        self.message_handlers[type_key].append(handler)
    
    def unregister_handler(self, message_type: MessageType, handler: Callable) -> None:
        """注销消息处理器"""
        type_key = message_type.value if isinstance(message_type, MessageType) else message_type
        if handler in self.message_handlers[type_key]:
            self.message_handlers[type_key].remove(handler)
    
    async def process_message(self, message: Union[Message, Response]) -> None:
        """处理接收到的消息"""
        if isinstance(message, Message):
            type_key = message.message_type.value
        else:
            type_key = "response"
        
        handlers = self.message_handlers.get(type_key, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"处理消息时发生错误: {e}")


class MemoryChannel(CommunicationChannel):
    """内存通信通道（单进程内通信）"""
    
    # 全局消息总线
    _message_bus: Dict[str, 'MemoryChannel'] = {}
    _global_message_queue: deque = deque()
    _subscribers: Dict[str, List['MemoryChannel']] = defaultdict(list)
    
    def __init__(self, agent_id: str, max_queue_size: int = 1000):
        super().__init__(f"memory_{agent_id}")
        self.agent_id = agent_id
        self.max_queue_size = max_queue_size
        self.message_queue: deque = deque(maxlen=max_queue_size)
        self._lock = asyncio.Lock()
    
    async def connect(self) -> bool:
        """连接到内存总线"""
        try:
            MemoryChannel._message_bus[self.agent_id] = self
            self.is_connected = True
            logger.info(f"智能体 {self.agent_id} 连接到内存通道")
            return True
        except Exception as e:
            logger.error(f"连接内存通道失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开内存总线连接"""
        try:
            if self.agent_id in MemoryChannel._message_bus:
                del MemoryChannel._message_bus[self.agent_id]
            
            # 从订阅列表中移除
            for subscribers in MemoryChannel._subscribers.values():
                if self in subscribers:
                    subscribers.remove(self)
            
            self.is_connected = False
            logger.info(f"智能体 {self.agent_id} 断开内存通道")
            return True
        except Exception as e:
            logger.error(f"断开内存通道失败: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        if not self.is_connected:
            return False
        
        try:
            message.sender_id = self.agent_id
            
            if message.broadcast:
                # 广播消息
                await self._broadcast_message(message)
            else:
                # 点对点消息
                await self._send_direct_message(message)
            
            return True
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    async def send_response(self, response: Response) -> bool:
        """发送响应"""
        if not self.is_connected:
            return False
        
        try:
            response.sender_id = self.agent_id
            
            # 响应总是点对点的
            if response.recipient_id in MemoryChannel._message_bus:
                target_channel = MemoryChannel._message_bus[response.recipient_id]
                async with target_channel._lock:
                    target_channel.message_queue.append(response)
            
            return True
        except Exception as e:
            logger.error(f"发送响应失败: {e}")
            return False
    
    async def _send_direct_message(self, message: Message) -> None:
        """发送点对点消息"""
        if message.recipient_id in MemoryChannel._message_bus:
            target_channel = MemoryChannel._message_bus[message.recipient_id]
            async with target_channel._lock:
                target_channel.message_queue.append(message)
        else:
            logger.warning(f"目标智能体 {message.recipient_id} 不在线")
    
    async def _broadcast_message(self, message: Message) -> None:
        """广播消息"""
        routing_key = message.routing_key or "broadcast"
        
        # 发送给所有订阅者
        if routing_key in MemoryChannel._subscribers:
            for channel in MemoryChannel._subscribers[routing_key]:
                if channel.agent_id != self.agent_id:  # 不发送给自己
                    async with channel._lock:
                        channel.message_queue.append(message)
        
        # 发送给所有连接的智能体（如果没有订阅者）
        if not MemoryChannel._subscribers[routing_key]:
            for agent_id, channel in MemoryChannel._message_bus.items():
                if agent_id != self.agent_id:
                    async with channel._lock:
                        channel.message_queue.append(message)
    
    async def receive_messages(self, timeout: float = None) -> List[Union[Message, Response]]:
        """接收消息"""
        if not self.is_connected:
            return []
        
        messages = []
        start_time = datetime.now()
        
        while True:
            async with self._lock:
                if self.message_queue:
                    messages.append(self.message_queue.popleft())
                    break
            
            # 检查超时
            if timeout is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    break
            
            await asyncio.sleep(0.01)  # 短暂等待
        
        return messages
    
    def subscribe(self, routing_key: str) -> None:
        """订阅特定路由的消息"""
        if self not in MemoryChannel._subscribers[routing_key]:
            MemoryChannel._subscribers[routing_key].append(self)
            logger.info(f"智能体 {self.agent_id} 订阅路由: {routing_key}")
    
    def unsubscribe(self, routing_key: str) -> None:
        """取消订阅"""
        if self in MemoryChannel._subscribers[routing_key]:
            MemoryChannel._subscribers[routing_key].remove(self)
            logger.info(f"智能体 {self.agent_id} 取消订阅路由: {routing_key}")
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return len(self.message_queue)
    
    @classmethod
    def get_online_agents(cls) -> List[str]:
        """获取在线智能体列表"""
        return list(cls._message_bus.keys())
    
    @classmethod
    def get_channel_stats(cls) -> Dict[str, Any]:
        """获取通道统计信息"""
        return {
            'total_channels': len(cls._message_bus),
            'online_agents': list(cls._message_bus.keys()),
            'global_queue_size': len(cls._global_message_queue),
            'subscriptions': {
                routing_key: len(subscribers) 
                for routing_key, subscribers in cls._subscribers.items()
            }
        }


class RedisChannel(CommunicationChannel):
    """Redis通信通道（分布式通信）"""
    
    def __init__(self, agent_id: str, redis_url: str = "redis://localhost:6379",
                 db: int = 0, channel_prefix: str = "agent"):
        super().__init__(f"redis_{agent_id}")
        self.agent_id = agent_id
        self.redis_url = redis_url
        self.db = db
        self.channel_prefix = channel_prefix
        self.redis_client = None
        self.pubsub = None
        self.subscribe_task = None
        
        # 消息队列
        self.inbox_key = f"{channel_prefix}:inbox:{agent_id}"
        self.outbox_key = f"{channel_prefix}:outbox:{agent_id}"
    
    async def connect(self) -> bool:
        """连接到Redis"""
        try:
            import redis.asyncio as redis
            
            self.redis_client = redis.from_url(
                self.redis_url,
                db=self.db,
                decode_responses=True
            )
            
            # 测试连接
            await self.redis_client.ping()
            
            # 设置发布/订阅
            self.pubsub = self.redis_client.pubsub()
            
            # 订阅智能体专用频道
            await self.pubsub.subscribe(f"{self.channel_prefix}:direct:{self.agent_id}")
            await self.pubsub.subscribe(f"{self.channel_prefix}:broadcast")
            
            # 启动订阅任务
            self.subscribe_task = asyncio.create_task(self._listen_for_messages())
            
            self.is_connected = True
            logger.info(f"智能体 {self.agent_id} 连接到Redis通道")
            return True
            
        except Exception as e:
            logger.error(f"连接Redis失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开Redis连接"""
        try:
            if self.subscribe_task:
                self.subscribe_task.cancel()
                try:
                    await self.subscribe_task
                except asyncio.CancelledError:
                    pass
            
            if self.pubsub:
                await self.pubsub.unsubscribe()
                await self.pubsub.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.is_connected = False
            logger.info(f"智能体 {self.agent_id} 断开Redis通道")
            return True
            
        except Exception as e:
            logger.error(f"断开Redis连接失败: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        if not self.is_connected:
            return False
        
        try:
            message.sender_id = self.agent_id
            message_data = self.serializer.serialize_message(message)
            
            if message.broadcast:
                # 广播消息
                await self.redis_client.publish(
                    f"{self.channel_prefix}:broadcast",
                    message_data
                )
            else:
                # 点对点消息
                await self.redis_client.publish(
                    f"{self.channel_prefix}:direct:{message.recipient_id}",
                    message_data
                )
            
            return True
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    async def send_response(self, response: Response) -> bool:
        """发送响应"""
        if not self.is_connected:
            return False
        
        try:
            response.sender_id = self.agent_id
            response_data = self.serializer.serialize_response(response)
            
            await self.redis_client.publish(
                f"{self.channel_prefix}:direct:{response.recipient_id}",
                response_data
            )
            
            return True
        except Exception as e:
            logger.error(f"发送响应失败: {e}")
            return False
    
    async def receive_messages(self, timeout: float = None) -> List[Union[Message, Response]]:
        """接收消息（Redis通道通过订阅自动接收）"""
        # Redis通道通过pubsub自动接收消息，这里返回空列表
        # 实际的消息处理在_listen_for_messages中进行
        return []
    
    async def _listen_for_messages(self) -> None:
        """监听Redis消息"""
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        # 反序列化消息
                        data = message['data']
                        if data.startswith('MSG:'):
                            msg = self.serializer.deserialize_message(data)
                            await self.process_message(msg)
                        elif data.startswith('RSP:'):
                            rsp = self.serializer.deserialize_response(data)
                            await self.process_message(rsp)
                    except Exception as e:
                        logger.error(f"处理Redis消息失败: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis消息监听异常: {e}")
    
    async def subscribe_topic(self, topic: str) -> None:
        """订阅特定主题"""
        if self.pubsub:
            await self.pubsub.subscribe(f"{self.channel_prefix}:topic:{topic}")
            logger.info(f"智能体 {self.agent_id} 订阅主题: {topic}")
    
    async def unsubscribe_topic(self, topic: str) -> None:
        """取消订阅主题"""
        if self.pubsub:
            await self.pubsub.unsubscribe(f"{self.channel_prefix}:topic:{topic}")
            logger.info(f"智能体 {self.agent_id} 取消订阅主题: {topic}")
    
    async def get_online_agents(self) -> List[str]:
        """获取在线智能体列表"""
        try:
            # 通过检查活跃的频道来判断在线智能体
            channels = await self.redis_client.pubsub_channels(
                f"{self.channel_prefix}:direct:*"
            )
            agents = []
            for channel in channels:
                # 提取智能体ID
                if channel.startswith(f"{self.channel_prefix}:direct:"):
                    agent_id = channel[len(f"{self.channel_prefix}:direct:"):]
                    agents.append(agent_id)
            return agents
        except Exception as e:
            logger.error(f"获取在线智能体失败: {e}")
            return []


class WebSocketChannel(CommunicationChannel):
    """WebSocket通信通道（实时Web通信）"""
    
    def __init__(self, agent_id: str, websocket_url: str = "ws://localhost:8765"):
        super().__init__(f"websocket_{agent_id}")
        self.agent_id = agent_id
        self.websocket_url = websocket_url
        self.websocket = None
        self.receive_task = None
    
    async def connect(self) -> bool:
        """连接到WebSocket服务器"""
        try:
            import websockets
            
            self.websocket = await websockets.connect(self.websocket_url)
            
            # 发送注册消息
            register_msg = {
                'type': 'register',
                'agent_id': self.agent_id
            }
            await self.websocket.send(json.dumps(register_msg))
            
            # 启动接收任务
            self.receive_task = asyncio.create_task(self._receive_loop())
            
            self.is_connected = True
            logger.info(f"智能体 {self.agent_id} 连接到WebSocket通道")
            return True
            
        except Exception as e:
            logger.error(f"连接WebSocket失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开WebSocket连接"""
        try:
            if self.receive_task:
                self.receive_task.cancel()
                try:
                    await self.receive_task
                except asyncio.CancelledError:
                    pass
            
            if self.websocket:
                await self.websocket.close()
            
            self.is_connected = False
            logger.info(f"智能体 {self.agent_id} 断开WebSocket通道")
            return True
            
        except Exception as e:
            logger.error(f"断开WebSocket连接失败: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        if not self.is_connected or not self.websocket:
            return False
        
        try:
            message.sender_id = self.agent_id
            message_data = self.serializer.serialize_message(message)
            
            await self.websocket.send(message_data)
            return True
        except Exception as e:
            logger.error(f"发送WebSocket消息失败: {e}")
            return False
    
    async def send_response(self, response: Response) -> bool:
        """发送响应"""
        if not self.is_connected or not self.websocket:
            return False
        
        try:
            response.sender_id = self.agent_id
            response_data = self.serializer.serialize_response(response)
            
            await self.websocket.send(response_data)
            return True
        except Exception as e:
            logger.error(f"发送WebSocket响应失败: {e}")
            return False
    
    async def receive_messages(self, timeout: float = None) -> List[Union[Message, Response]]:
        """接收消息（WebSocket通过循环自动接收）"""
        return []
    
    async def _receive_loop(self) -> None:
        """WebSocket接收循环"""
        try:
            async for raw_message in self.websocket:
                try:
                    if raw_message.startswith('MSG:'):
                        msg = self.serializer.deserialize_message(raw_message)
                        await self.process_message(msg)
                    elif raw_message.startswith('RSP:'):
                        rsp = self.serializer.deserialize_response(raw_message)
                        await self.process_message(rsp)
                except Exception as e:
                    logger.error(f"处理WebSocket消息失败: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket接收循环异常: {e}")


class ChannelManager:
    """通道管理器"""
    
    def __init__(self):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.default_channel: Optional[CommunicationChannel] = None
    
    def add_channel(self, name: str, channel: CommunicationChannel,
                   set_as_default: bool = False) -> None:
        """添加通道"""
        self.channels[name] = channel
        if set_as_default or self.default_channel is None:
            self.default_channel = channel
        logger.info(f"添加通信通道: {name}")
    
    def remove_channel(self, name: str) -> bool:
        """移除通道"""
        if name in self.channels:
            channel = self.channels.pop(name)
            if channel == self.default_channel:
                self.default_channel = None
            logger.info(f"移除通信通道: {name}")
            return True
        return False
    
    def get_channel(self, name: str) -> Optional[CommunicationChannel]:
        """获取通道"""
        return self.channels.get(name)
    
    def get_default_channel(self) -> Optional[CommunicationChannel]:
        """获取默认通道"""
        return self.default_channel
    
    async def connect_all(self) -> bool:
        """连接所有通道"""
        results = []
        for name, channel in self.channels.items():
            result = await channel.connect()
            results.append(result)
            if not result:
                logger.error(f"通道 {name} 连接失败")
        return all(results)
    
    async def disconnect_all(self) -> bool:
        """断开所有通道"""
        results = []
        for name, channel in self.channels.items():
            result = await channel.disconnect()
            results.append(result)
            if not result:
                logger.error(f"通道 {name} 断开失败")
        return all(results)
    
    def list_channels(self) -> List[str]:
        """列出所有通道"""
        return list(self.channels.keys())
    
    def get_channel_stats(self) -> Dict[str, Any]:
        """获取通道统计"""
        stats = {}
        for name, channel in self.channels.items():
            stats[name] = {
                'channel_id': channel.channel_id,
                'is_connected': channel.is_connected,
                'type': type(channel).__name__
            }
        return stats