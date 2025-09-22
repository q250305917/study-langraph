"""
基础智能体抽象类

定义智能体的核心接口和基础实现，包括：
- 智能体生命周期管理
- 任务处理机制
- 消息通信接口
- 状态和记忆管理
"""

from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import asyncio
import uuid
import logging
from datetime import datetime

from .config import AgentConfig
from .state import AgentState, AgentStatus, TaskStatus
from .memory import AgentMemory
from .tools import AgentTools, ToolResult
from ..communication import Message, Response

logger = logging.getLogger(__name__)


class Task:
    """任务定义"""
    
    def __init__(self, task_id: str = None, task_type: str = "generic",
                 content: Any = None, parameters: Dict[str, Any] = None,
                 priority: int = 0, timeout: float = 300.0):
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type
        self.content = content
        self.parameters = parameters or {}
        self.priority = priority
        self.timeout = timeout
        self.created_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'content': self.content,
            'parameters': self.parameters,
            'priority': self.priority,
            'timeout': self.timeout,
            'created_time': self.created_time.isoformat()
        }


class TaskResult:
    """任务结果"""
    
    def __init__(self, task_id: str, success: bool = True, 
                 result: Any = None, error: str = None,
                 metadata: Dict[str, Any] = None):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.metadata = metadata or {}
        self.completed_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'metadata': self.metadata,
            'completed_time': self.completed_time.isoformat()
        }


class BaseAgent(ABC):
    """智能体基础抽象类"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.agent_name = config.agent_name
        self.agent_type = config.agent_type
        
        # 初始化核心组件
        self.state = AgentState(self.agent_id)
        self.memory = AgentMemory(self.agent_id)
        self.tools = AgentTools(self.agent_id)
        
        # 任务管理
        self.task_queue = asyncio.Queue(maxsize=config.message_queue_size)
        self.current_tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent_tasks = config.max_concurrent_tasks
        
        # 生命周期管理
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._task_processor = None
        self._heartbeat_task = None
        
        # 通信管理
        self.message_handlers: Dict[str, Callable] = {}
        self.collaboration_partners: List[str] = []
        
        # 监控和日志
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # 注册默认消息处理器
        self._register_default_handlers()
    
    async def start(self) -> None:
        """启动智能体"""
        if self._is_running:
            self.logger.warning("智能体已经在运行")
            return
        
        try:
            self.logger.info(f"启动智能体 {self.agent_id}")
            self.state.set_status(AgentStatus.INITIALIZING)
            
            # 初始化组件
            await self._initialize()
            
            # 启动任务处理器
            self._task_processor = asyncio.create_task(self._process_tasks())
            
            # 启动心跳
            if self.config.heartbeat_interval > 0:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # 设置为运行状态
            self._is_running = True
            self.state.set_status(AgentStatus.IDLE)
            
            self.logger.info(f"智能体 {self.agent_id} 启动成功")
            
        except Exception as e:
            self.logger.error(f"智能体启动失败: {e}")
            self.state.set_status(AgentStatus.ERROR, str(e))
            raise
    
    async def stop(self) -> None:
        """停止智能体"""
        if not self._is_running:
            self.logger.warning("智能体已经停止")
            return
        
        try:
            self.logger.info(f"停止智能体 {self.agent_id}")
            self.state.set_status(AgentStatus.SHUTDOWN)
            
            # 设置停止标志
            self._is_running = False
            self._shutdown_event.set()
            
            # 取消所有当前任务
            for task_id, task in self.current_tasks.items():
                if not task.done():
                    task.cancel()
                    self.state.cancel_task(task_id)
            
            # 停止任务处理器
            if self._task_processor:
                self._task_processor.cancel()
                try:
                    await self._task_processor
                except asyncio.CancelledError:
                    pass
            
            # 停止心跳
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # 清理资源
            await self._cleanup()
            
            self.logger.info(f"智能体 {self.agent_id} 停止成功")
            
        except Exception as e:
            self.logger.error(f"智能体停止时发生错误: {e}")
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        if not self._is_running:
            raise RuntimeError("智能体未运行")
        
        if self.task_queue.full():
            raise RuntimeError("任务队列已满")
        
        # 添加任务到状态管理
        self.state.add_task(task.task_id, task.task_type, task.parameters)
        
        # 添加到队列
        await self.task_queue.put(task)
        
        self.logger.info(f"任务 {task.task_id} 已提交")
        return task.task_id
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """获取任务结果（等待任务完成）"""
        if task_id in self.current_tasks:
            try:
                # 等待任务完成
                await asyncio.wait_for(self.current_tasks[task_id], timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"等待任务 {task_id} 结果超时")
                return None
        
        # 从历史记录中查找结果
        task_history = self.state.get_task_history()
        for task_info in task_history:
            if task_info.task_id == task_id:
                return TaskResult(
                    task_id=task_id,
                    success=task_info.status == TaskStatus.COMPLETED,
                    result=task_info.result,
                    error=task_info.error
                )
        
        return None
    
    async def send_message(self, target_agent: str, message: Message) -> Optional[Response]:
        """发送消息给其他智能体"""
        # 这里需要实现消息路由，暂时返回None
        self.logger.info(f"向 {target_agent} 发送消息: {message.content}")
        return None
    
    async def handle_message(self, message: Message) -> Response:
        """处理接收到的消息"""
        message_type = message.message_type
        
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            try:
                return await handler(message)
            except Exception as e:
                self.logger.error(f"处理消息时发生错误: {e}")
                return Response(
                    success=False,
                    content=f"消息处理失败: {e}",
                    sender_id=self.agent_id
                )
        else:
            # 使用默认处理器
            return await self._default_message_handler(message)
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"注册消息处理器: {message_type}")
    
    def add_collaboration_partner(self, agent_id: str) -> None:
        """添加协作伙伴"""
        if agent_id not in self.collaboration_partners:
            self.collaboration_partners.append(agent_id)
            self.logger.info(f"添加协作伙伴: {agent_id}")
    
    def remove_collaboration_partner(self, agent_id: str) -> None:
        """移除协作伙伴"""
        if agent_id in self.collaboration_partners:
            self.collaboration_partners.remove(agent_id)
            self.logger.info(f"移除协作伙伴: {agent_id}")
    
    async def get_capabilities(self) -> List[str]:
        """获取智能体能力"""
        capabilities = [
            f"任务处理: {self.agent_type}",
            f"工具数量: {len(self.tools.enabled_tools)}",
            f"最大并发: {self.max_concurrent_tasks}"
        ]
        
        # 添加具体工具能力
        for tool_name in self.tools.enabled_tools:
            capabilities.append(f"工具: {tool_name}")
        
        return capabilities
    
    async def get_status_info(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'agent_info': {
                'id': self.agent_id,
                'name': self.agent_name,
                'type': self.agent_type,
                'version': self.config.version
            },
            'status': self.state.to_dict(),
            'memory': await self.memory.get_memory_summary(),
            'tools': self.tools.get_tool_usage_stats(),
            'capabilities': await self.get_capabilities(),
            'collaboration_partners': self.collaboration_partners
        }
    
    # 抽象方法 - 子类必须实现
    @abstractmethod
    async def process_task(self, task: Task) -> TaskResult:
        """处理任务（子类必须实现）"""
        pass
    
    @abstractmethod
    async def collaborate(self, message: Message) -> Response:
        """协作处理（子类必须实现）"""
        pass
    
    # 可选重写的方法
    async def _initialize(self) -> None:
        """初始化（子类可重写）"""
        # 启用配置中的工具
        for tool_name in self.config.enabled_tools:
            tool_config = self.config.tool_configs.get(tool_name, {})
            self.tools.enable_tool(tool_name, tool_config)
        
        self.logger.info(f"智能体初始化完成，启用工具: {self.config.enabled_tools}")
    
    async def _cleanup(self) -> None:
        """清理资源（子类可重写）"""
        pass
    
    # 私有方法
    async def _process_tasks(self) -> None:
        """任务处理循环"""
        while self._is_running:
            try:
                # 检查是否可以处理新任务
                if len(self.current_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                
                # 等待新任务
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # 启动任务处理
                task_coroutine = self._handle_task(task)
                task_future = asyncio.create_task(task_coroutine)
                self.current_tasks[task.task_id] = task_future
                
                # 设置任务完成回调
                task_future.add_done_callback(
                    lambda t, tid=task.task_id: self._on_task_complete(tid, t)
                )
                
            except Exception as e:
                self.logger.error(f"任务处理循环异常: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_task(self, task: Task) -> None:
        """处理单个任务"""
        self.logger.info(f"开始处理任务 {task.task_id}")
        
        # 更新任务状态
        self.state.start_task(task.task_id)
        
        try:
            # 设置任务超时
            result = await asyncio.wait_for(
                self.process_task(task),
                timeout=task.timeout
            )
            
            # 任务成功完成
            self.state.complete_task(task.task_id, result.result)
            self.logger.info(f"任务 {task.task_id} 完成")
            
        except asyncio.TimeoutError:
            # 任务超时
            error_msg = f"任务执行超时 ({task.timeout}秒)"
            self.state.fail_task(task.task_id, error_msg)
            self.logger.warning(f"任务 {task.task_id} 超时")
            
        except Exception as e:
            # 任务执行失败
            error_msg = str(e)
            self.state.fail_task(task.task_id, error_msg)
            self.logger.error(f"任务 {task.task_id} 失败: {e}")
    
    def _on_task_complete(self, task_id: str, task_future: asyncio.Task) -> None:
        """任务完成回调"""
        if task_id in self.current_tasks:
            del self.current_tasks[task_id]
        
        # 如果没有当前任务，设置为空闲状态
        if not self.current_tasks and self.state.status == AgentStatus.BUSY:
            self.state.set_status(AgentStatus.IDLE)
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self._is_running:
            try:
                self.state.heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"心跳异常: {e}")
                await asyncio.sleep(5.0)
    
    def _register_default_handlers(self) -> None:
        """注册默认消息处理器"""
        self.register_message_handler("ping", self._handle_ping)
        self.register_message_handler("status", self._handle_status_request)
        self.register_message_handler("collaborate", self._handle_collaboration)
    
    async def _handle_ping(self, message: Message) -> Response:
        """处理ping消息"""
        return Response(
            success=True,
            content="pong",
            sender_id=self.agent_id
        )
    
    async def _handle_status_request(self, message: Message) -> Response:
        """处理状态查询"""
        status_info = await self.get_status_info()
        return Response(
            success=True,
            content=status_info,
            sender_id=self.agent_id
        )
    
    async def _handle_collaboration(self, message: Message) -> Response:
        """处理协作消息"""
        return await self.collaborate(message)
    
    async def _default_message_handler(self, message: Message) -> Response:
        """默认消息处理器"""
        return Response(
            success=False,
            content=f"不支持的消息类型: {message.message_type}",
            sender_id=self.agent_id
        )