"""
消息路由模块

提供智能体间消息路由功能，包括：
- 消息路由策略
- 负载均衡
- 故障转移
- 消息广播
"""

from typing import Dict, Any, List, Optional, Callable, Set
import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import fnmatch

from .protocols import Message, Response, MessageType, Priority
from .channels import CommunicationChannel

logger = logging.getLogger(__name__)


class RoutingStrategy(ABC):
    """路由策略抽象基类"""
    
    @abstractmethod
    def select_target(self, message: Message, available_agents: List[str]) -> Optional[str]:
        """选择目标智能体"""
        pass


class RoundRobinStrategy(RoutingStrategy):
    """轮询路由策略"""
    
    def __init__(self):
        self.current_index = 0
    
    def select_target(self, message: Message, available_agents: List[str]) -> Optional[str]:
        """轮询选择目标"""
        if not available_agents:
            return None
        
        target = available_agents[self.current_index % len(available_agents)]
        self.current_index += 1
        return target


class LoadBalancedStrategy(RoutingStrategy):
    """负载均衡路由策略"""
    
    def __init__(self):
        self.agent_loads: Dict[str, int] = defaultdict(int)
    
    def select_target(self, message: Message, available_agents: List[str]) -> Optional[str]:
        """选择负载最低的智能体"""
        if not available_agents:
            return None
        
        # 找到负载最低的智能体
        min_load = min(self.agent_loads.get(agent, 0) for agent in available_agents)
        candidates = [agent for agent in available_agents 
                     if self.agent_loads.get(agent, 0) == min_load]
        
        # 如果有多个候选者，选择第一个
        target = candidates[0]
        self.agent_loads[target] += 1
        return target
    
    def update_load(self, agent_id: str, load_delta: int) -> None:
        """更新智能体负载"""
        self.agent_loads[agent_id] = max(0, self.agent_loads[agent_id] + load_delta)


class CapabilityBasedStrategy(RoutingStrategy):
    """基于能力的路由策略"""
    
    def __init__(self):
        self.agent_capabilities: Dict[str, Set[str]] = {}
    
    def register_capabilities(self, agent_id: str, capabilities: List[str]) -> None:
        """注册智能体能力"""
        self.agent_capabilities[agent_id] = set(capabilities)
    
    def select_target(self, message: Message, available_agents: List[str]) -> Optional[str]:
        """根据能力选择目标"""
        if not available_agents:
            return None
        
        # 获取消息要求的能力
        required_capability = message.get_metadata("required_capability")
        if not required_capability:
            # 如果没有特定要求，使用轮询
            return available_agents[0]
        
        # 找到具有所需能力的智能体
        capable_agents = []
        for agent in available_agents:
            capabilities = self.agent_capabilities.get(agent, set())
            if required_capability in capabilities:
                capable_agents.append(agent)
        
        return capable_agents[0] if capable_agents else None


class MessageRouter:
    """消息路由器"""
    
    def __init__(self, default_strategy: RoutingStrategy = None):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.agent_registry: Dict[str, str] = {}  # agent_id -> channel_name
        self.routing_table: Dict[str, List[str]] = defaultdict(list)  # routing_key -> agent_ids
        self.routing_strategy = default_strategy or RoundRobinStrategy()
        
        # 路由统计
        self.message_count = 0
        self.error_count = 0
        self.routing_stats: Dict[str, int] = defaultdict(int)
        
        # 消息缓存和重试
        self.pending_messages: Dict[str, Message] = {}
        self.retry_queue: deque = deque()
        self.max_retries = 3
        self.retry_delay = 5.0  # 秒
        
        # 广播订阅
        self.broadcast_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # 故障检测
        self.failed_agents: Set[str] = set()
        self.heartbeat_timestamps: Dict[str, datetime] = {}
        self.heartbeat_timeout = 60.0  # 秒
    
    def register_channel(self, channel_name: str, channel: CommunicationChannel) -> None:
        """注册通信通道"""
        self.channels[channel_name] = channel
        logger.info(f"注册通信通道: {channel_name}")
    
    def register_agent(self, agent_id: str, channel_name: str) -> bool:
        """注册智能体到通道"""
        if channel_name not in self.channels:
            logger.error(f"通道 {channel_name} 不存在")
            return False
        
        self.agent_registry[agent_id] = channel_name
        self.heartbeat_timestamps[agent_id] = datetime.now()
        
        # 从失败列表中移除
        self.failed_agents.discard(agent_id)
        
        logger.info(f"智能体 {agent_id} 注册到通道 {channel_name}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        if agent_id in self.agent_registry:
            channel_name = self.agent_registry.pop(agent_id)
            
            # 从路由表中移除
            for routing_key, agents in self.routing_table.items():
                if agent_id in agents:
                    agents.remove(agent_id)
            
            # 从广播订阅中移除
            for subscribers in self.broadcast_subscribers.values():
                subscribers.discard(agent_id)
            
            # 清理心跳记录
            self.heartbeat_timestamps.pop(agent_id, None)
            
            logger.info(f"智能体 {agent_id} 从通道 {channel_name} 注销")
            return True
        return False
    
    def subscribe_routing_key(self, agent_id: str, routing_key: str) -> None:
        """订阅路由键"""
        if agent_id not in self.agent_registry:
            logger.error(f"智能体 {agent_id} 未注册")
            return
        
        if agent_id not in self.routing_table[routing_key]:
            self.routing_table[routing_key].append(agent_id)
            logger.info(f"智能体 {agent_id} 订阅路由键: {routing_key}")
    
    def unsubscribe_routing_key(self, agent_id: str, routing_key: str) -> None:
        """取消订阅路由键"""
        if agent_id in self.routing_table[routing_key]:
            self.routing_table[routing_key].remove(agent_id)
            logger.info(f"智能体 {agent_id} 取消订阅路由键: {routing_key}")
    
    def subscribe_broadcast(self, agent_id: str, pattern: str = "*") -> None:
        """订阅广播消息"""
        if agent_id not in self.agent_registry:
            logger.error(f"智能体 {agent_id} 未注册")
            return
        
        self.broadcast_subscribers[pattern].add(agent_id)
        logger.info(f"智能体 {agent_id} 订阅广播模式: {pattern}")
    
    def unsubscribe_broadcast(self, agent_id: str, pattern: str = "*") -> None:
        """取消订阅广播消息"""
        if agent_id in self.broadcast_subscribers[pattern]:
            self.broadcast_subscribers[pattern].remove(agent_id)
            logger.info(f"智能体 {agent_id} 取消订阅广播模式: {pattern}")
    
    async def route_message(self, message: Message) -> bool:
        """路由消息"""
        try:
            self.message_count += 1
            
            # 更新发送者心跳
            if message.sender_id:
                self.heartbeat_timestamps[message.sender_id] = datetime.now()
            
            if message.broadcast:
                return await self._route_broadcast_message(message)
            else:
                return await self._route_direct_message(message)
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"路由消息失败: {e}")
            return False
    
    async def route_response(self, response: Response) -> bool:
        """路由响应"""
        try:
            # 响应总是点对点的
            if not response.recipient_id:
                logger.error("响应缺少接收者ID")
                return False
            
            return await self._send_to_agent(response.recipient_id, response)
            
        except Exception as e:
            logger.error(f"路由响应失败: {e}")
            return False
    
    async def _route_direct_message(self, message: Message) -> bool:
        """路由点对点消息"""
        if message.recipient_id:
            # 指定接收者
            return await self._send_to_agent(message.recipient_id, message)
        
        elif message.routing_key:
            # 根据路由键选择接收者
            return await self._route_by_key(message)
        
        else:
            logger.error("消息缺少接收者或路由键")
            return False
    
    async def _route_broadcast_message(self, message: Message) -> bool:
        """路由广播消息"""
        routing_key = message.routing_key or "*"
        recipients = set()
        
        # 根据模式匹配查找订阅者
        for pattern, subscribers in self.broadcast_subscribers.items():
            if fnmatch.fnmatch(routing_key, pattern):
                recipients.update(subscribers)
        
        # 如果没有特定订阅者，发送给所有在线智能体
        if not recipients:
            recipients = set(self.get_online_agents())
            # 不发送给自己
            recipients.discard(message.sender_id)
        
        # 发送给所有接收者
        success_count = 0
        for recipient in recipients:
            if await self._send_to_agent(recipient, message):
                success_count += 1
        
        # 记录统计
        self.routing_stats[f"broadcast_{routing_key}"] += success_count
        
        return success_count > 0
    
    async def _route_by_key(self, message: Message) -> bool:
        """根据路由键路由消息"""
        routing_key = message.routing_key
        available_agents = self.routing_table.get(routing_key, [])
        
        # 过滤掉失败的智能体
        available_agents = [agent for agent in available_agents 
                           if agent not in self.failed_agents]
        
        if not available_agents:
            logger.warning(f"路由键 {routing_key} 没有可用的智能体")
            await self._handle_undeliverable_message(message, "no_available_agents")
            return False
        
        # 使用路由策略选择目标
        target_agent = self.routing_strategy.select_target(message, available_agents)
        if not target_agent:
            logger.error("路由策略无法选择目标智能体")
            return False
        
        # 发送消息
        success = await self._send_to_agent(target_agent, message)
        
        # 更新统计
        self.routing_stats[f"key_{routing_key}"] += 1
        
        return success
    
    async def _send_to_agent(self, agent_id: str, 
                           message_or_response: Union[Message, Response]) -> bool:
        """发送消息/响应给指定智能体"""
        # 检查智能体是否注册
        if agent_id not in self.agent_registry:
            logger.error(f"智能体 {agent_id} 未注册")
            return False
        
        # 检查智能体是否失败
        if agent_id in self.failed_agents:
            logger.warning(f"智能体 {agent_id} 处于失败状态")
            if isinstance(message_or_response, Message):
                await self._handle_undeliverable_message(message_or_response, "agent_failed")
            return False
        
        # 获取通道
        channel_name = self.agent_registry[agent_id]
        channel = self.channels.get(channel_name)
        
        if not channel or not channel.is_connected:
            logger.error(f"通道 {channel_name} 不可用")
            await self._mark_agent_failed(agent_id)
            return False
        
        try:
            # 发送消息或响应
            if isinstance(message_or_response, Message):
                success = await channel.send_message(message_or_response)
            else:
                success = await channel.send_response(message_or_response)
            
            if not success:
                logger.error(f"发送到智能体 {agent_id} 失败")
                await self._mark_agent_failed(agent_id)
            
            return success
            
        except Exception as e:
            logger.error(f"发送到智能体 {agent_id} 异常: {e}")
            await self._mark_agent_failed(agent_id)
            return False
    
    async def _mark_agent_failed(self, agent_id: str) -> None:
        """标记智能体失败"""
        self.failed_agents.add(agent_id)
        logger.warning(f"智能体 {agent_id} 被标记为失败")
        
        # 可以在这里实现故障通知机制
        await self._notify_agent_failure(agent_id)
    
    async def _notify_agent_failure(self, failed_agent_id: str) -> None:
        """通知智能体失败"""
        # 创建失败通知消息
        from .protocols import MessageBuilder, MessageType
        
        failure_message = MessageBuilder.create_broadcast(
            sender_id="router",
            message_type=MessageType.ALERT,
            content={
                'alert_type': 'agent_failure',
                'failed_agent': failed_agent_id,
                'timestamp': datetime.now().isoformat()
            },
            routing_key="alert.agent_failure"
        )
        
        # 广播失败通知
        await self.route_message(failure_message)
    
    async def _handle_undeliverable_message(self, message: Message, reason: str) -> None:
        """处理无法投递的消息"""
        # 检查重试次数
        retry_count = message.get_metadata("retry_count", 0)
        
        if retry_count < self.max_retries:
            # 增加重试次数
            message.add_metadata("retry_count", retry_count + 1)
            message.add_metadata("retry_reason", reason)
            
            # 添加到重试队列
            retry_time = datetime.now() + timedelta(seconds=self.retry_delay)
            self.retry_queue.append((retry_time, message))
            
            logger.info(f"消息 {message.message_id} 将在 {self.retry_delay} 秒后重试")
        else:
            logger.error(f"消息 {message.message_id} 超过最大重试次数，丢弃")
            
            # 发送失败通知
            await self._notify_message_failed(message, reason)
    
    async def _notify_message_failed(self, message: Message, reason: str) -> None:
        """通知消息失败"""
        if message.sender_id:
            from .protocols import MessageBuilder, MessageType
            
            failure_response = Response(
                correlation_id=message.message_id,
                sender_id="router",
                recipient_id=message.sender_id,
                success=False,
                error=f"消息投递失败: {reason}"
            )
            
            await self.route_response(failure_response)
    
    async def process_retry_queue(self) -> None:
        """处理重试队列"""
        now = datetime.now()
        
        while self.retry_queue:
            retry_time, message = self.retry_queue[0]
            
            if retry_time <= now:
                self.retry_queue.popleft()
                logger.info(f"重试消息 {message.message_id}")
                await self.route_message(message)
            else:
                break
    
    def update_heartbeat(self, agent_id: str) -> None:
        """更新智能体心跳"""
        self.heartbeat_timestamps[agent_id] = datetime.now()
        
        # 如果智能体之前失败，现在恢复了
        if agent_id in self.failed_agents:
            self.failed_agents.remove(agent_id)
            logger.info(f"智能体 {agent_id} 恢复正常")
    
    def check_agent_health(self) -> None:
        """检查智能体健康状态"""
        now = datetime.now()
        timeout_threshold = now - timedelta(seconds=self.heartbeat_timeout)
        
        for agent_id, last_heartbeat in self.heartbeat_timestamps.items():
            if last_heartbeat < timeout_threshold and agent_id not in self.failed_agents:
                asyncio.create_task(self._mark_agent_failed(agent_id))
    
    def get_online_agents(self) -> List[str]:
        """获取在线智能体列表"""
        return [agent_id for agent_id in self.agent_registry.keys() 
                if agent_id not in self.failed_agents]
    
    def get_failed_agents(self) -> List[str]:
        """获取失败智能体列表"""
        return list(self.failed_agents)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        return {
            'total_messages': self.message_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / self.message_count if self.message_count > 0 else 0,
            'routing_stats': dict(self.routing_stats),
            'online_agents': len(self.get_online_agents()),
            'failed_agents': len(self.failed_agents),
            'pending_retries': len(self.retry_queue),
            'registered_agents': len(self.agent_registry),
            'routing_keys': len(self.routing_table),
            'broadcast_patterns': len(self.broadcast_subscribers)
        }
    
    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        """设置路由策略"""
        self.routing_strategy = strategy
        logger.info(f"设置路由策略: {type(strategy).__name__}")
    
    async def start_background_tasks(self) -> None:
        """启动后台任务"""
        # 启动重试队列处理
        asyncio.create_task(self._retry_queue_processor())
        
        # 启动健康检查
        asyncio.create_task(self._health_checker())
    
    async def _retry_queue_processor(self) -> None:
        """重试队列处理器"""
        while True:
            try:
                await self.process_retry_queue()
                await asyncio.sleep(1.0)  # 每秒检查一次
            except Exception as e:
                logger.error(f"重试队列处理器异常: {e}")
                await asyncio.sleep(5.0)
    
    async def _health_checker(self) -> None:
        """健康检查器"""
        while True:
            try:
                self.check_agent_health()
                await asyncio.sleep(30.0)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"健康检查器异常: {e}")
                await asyncio.sleep(60.0)