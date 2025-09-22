"""
协作Agent模板

专门用于多Agent协作的实现，支持任务分解、Agent管理、协作执行和结果汇总。
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .base_agent import BaseAgent, ToolDefinition, ExecutionMetrics


class CollaborationStrategy(Enum):
    """协作策略枚举"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"          # 并行执行  
    HIERARCHICAL = "hierarchical"  # 层次化执行
    PIPELINE = "pipeline"          # 流水线执行


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SubTask:
    """子任务定义"""
    task_id: str
    description: str
    agent_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentResult:
    """Agent执行结果"""
    agent_id: str
    task_id: str
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationContext:
    """协作上下文"""
    session_id: str
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    start_time: float = 0.0
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)


class SubAgentInterface(ABC):
    """子Agent接口"""
    
    @abstractmethod
    async def execute_task(self, task: SubTask, context: CollaborationContext) -> AgentResult:
        """
        执行任务
        
        Args:
            task: 子任务
            context: 协作上下文
            
        Returns:
            执行结果
        """
        pass


class MockSubAgent(SubAgentInterface):
    """模拟子Agent（用于演示）"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
    
    async def execute_task(self, task: SubTask, context: CollaborationContext) -> AgentResult:
        """模拟任务执行"""
        start_time = time.time()
        
        # 模拟执行时间
        await asyncio.sleep(0.1 + (task.priority.value * 0.05))
        
        execution_time = time.time() - start_time
        
        # 简单的成功/失败模拟
        success = True
        result = f"任务 {task.task_id} 由 {self.agent_type} 代理完成"
        error = None
        
        # 模拟小概率失败
        import random
        if random.random() < 0.1:  # 10% 失败率
            success = False
            result = None
            error = f"模拟执行失败: {task.task_id}"
        
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
            metadata={"agent_type": self.agent_type, "priority": task.priority.value}
        )


class CollaborativeAgent(BaseAgent):
    """
    协作Agent模板
    
    专门设计用于多Agent协作场景，具备以下核心能力：
    1. 任务分解 - 智能分解复杂任务为可并行的子任务
    2. Agent管理 - 动态注册、管理和调度多个子Agent
    3. 协作策略 - 支持多种协作模式（顺序、并行、层次化等）
    4. 结果汇总 - 收集和整合多Agent执行结果
    5. 冲突解决 - 处理Agent间的依赖关系和资源冲突
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 子Agent管理
        self.sub_agents: Dict[str, SubAgentInterface] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # 任务分解规则
        self.decomposition_rules: Dict[str, Callable[[str], List[SubTask]]] = {}
        
        # 协作历史
        self.collaboration_history: List[CollaborationContext] = []
        
        # 注册内置工具
        self._register_builtin_tools()
        
        # 设置默认分解规则
        self._setup_default_decomposition_rules()
    
    @property
    def default_config(self) -> Dict[str, Any]:
        """默认配置"""
        base_config = super().default_config
        collab_config = {
            "collaboration_strategy": "hierarchical",  # sequential, parallel, hierarchical, pipeline
            "max_parallel_agents": 5,                  # 最大并行Agent数
            "task_timeout": 30.0,                      # 子任务超时时间
            "result_aggregation": "comprehensive",     # simple, comprehensive, prioritized
            "enable_dependency_resolution": True,      # 启用依赖解析
            "enable_load_balancing": True,             # 启用负载均衡
            "retry_failed_tasks": True,                # 重试失败任务
            "conflict_resolution": "priority_based",   # priority_based, first_come_first_serve
        }
        return {**base_config, **collab_config}
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        tools = [
            ToolDefinition(
                name="decompose_task",
                description="分解复杂任务为子任务",
                func=self._decompose_task,
                parameters={
                    "task_content": {"type": "str", "required": True},
                    "task_type": {"type": "str", "default": "default"}
                }
            ),
            ToolDefinition(
                name="assign_agents",
                description="为子任务分配Agent",
                func=self._assign_agents,
                parameters={
                    "subtasks": {"type": "list", "required": True},
                    "strategy": {"type": "str", "default": "capability_based"}
                }
            ),
            ToolDefinition(
                name="execute_collaboration",
                description="执行协作任务",
                func=self._execute_collaboration,
                async_func=True,
                parameters={
                    "subtasks": {"type": "list", "required": True},
                    "strategy": {"type": "str", "required": True},
                    "context": {"type": "dict", "required": True}
                }
            ),
            ToolDefinition(
                name="aggregate_results",
                description="汇总多Agent执行结果",
                func=self._aggregate_results,
                parameters={
                    "results": {"type": "list", "required": True},
                    "aggregation_type": {"type": "str", "default": "comprehensive"}
                }
            ),
            ToolDefinition(
                name="resolve_dependencies",
                description="解析任务依赖关系",
                func=self._resolve_dependencies,
                parameters={
                    "subtasks": {"type": "list", "required": True}
                }
            )
        ]
        
        self.register_tools(tools)
    
    def _setup_default_decomposition_rules(self):
        """设置默认任务分解规则"""
        self.decomposition_rules = {
            "default": self._default_decomposition,
            "research": self._research_decomposition,
            "analysis": self._analysis_decomposition,
            "generation": self._generation_decomposition,
            "processing": self._processing_decomposition
        }
    
    async def think(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考过程：分析任务、制定协作策略、规划执行方案
        
        Args:
            input_data: 输入任务描述
            context: 执行上下文
            
        Returns:
            决策结果包含协作策略和执行计划
        """
        # 1. 任务复杂度评估
        task_complexity = await self._assess_task_complexity(input_data)
        
        # 2. 任务类型识别
        task_type = await self._identify_task_type(input_data)
        
        # 3. 分解任务
        subtasks = await self.call_tool(
            "decompose_task",
            task_content=input_data,
            task_type=task_type
        )
        
        # 4. 解析依赖关系
        if self.config.get("enable_dependency_resolution", True):
            resolved_subtasks = await self.call_tool(
                "resolve_dependencies",
                subtasks=subtasks
            )
        else:
            resolved_subtasks = subtasks
        
        # 5. 确定协作策略
        collaboration_strategy = self._determine_collaboration_strategy(
            task_complexity, len(resolved_subtasks)
        )
        
        # 6. 分配Agent
        agent_assignments = await self.call_tool(
            "assign_agents",
            subtasks=resolved_subtasks,
            strategy="capability_based"
        )
        
        decision = {
            "task_complexity": task_complexity,
            "task_type": task_type,
            "subtasks": resolved_subtasks,
            "collaboration_strategy": collaboration_strategy,
            "agent_assignments": agent_assignments,
            "estimated_duration": self._estimate_execution_duration(resolved_subtasks, collaboration_strategy),
            "resource_requirements": self._calculate_resource_requirements(resolved_subtasks)
        }
        
        self.logger.info(f"任务分解完成 - {len(resolved_subtasks)} 个子任务，策略: {collaboration_strategy}")
        
        return decision
    
    async def act(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行动作：协调多Agent执行子任务
        
        Args:
            decision: 思考阶段的决策结果
            context: 执行上下文
            
        Returns:
            协作执行结果
        """
        session_id = context["session_id"]
        subtasks = decision["subtasks"]
        strategy = decision["collaboration_strategy"]
        
        # 创建协作上下文
        collab_context = CollaborationContext(
            session_id=session_id,
            total_tasks=len(subtasks),
            start_time=time.time()
        )
        
        # 执行协作任务
        execution_results = await self.call_tool(
            "execute_collaboration",
            subtasks=subtasks,
            strategy=strategy,
            context=collab_context
        )
        
        # 处理失败任务的重试
        if self.config.get("retry_failed_tasks", True):
            execution_results = await self._handle_failed_tasks(execution_results, collab_context)
        
        # 汇总结果
        aggregated_result = await self.call_tool(
            "aggregate_results",
            results=execution_results,
            aggregation_type=self.config.get("result_aggregation", "comprehensive")
        )
        
        # 记录协作历史
        collab_context.completed_tasks = len([r for r in execution_results if r.success])
        collab_context.failed_tasks = len([r for r in execution_results if not r.success])
        self.collaboration_history.append(collab_context)
        
        return {
            "execution_results": execution_results,
            "aggregated_result": aggregated_result,
            "collaboration_context": collab_context,
            "performance_metrics": self._calculate_performance_metrics(execution_results, collab_context)
        }
    
    async def respond(self, action_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        生成最终回复：整合协作结果，生成综合回复
        
        Args:
            action_result: 动作执行结果
            context: 执行上下文
            
        Returns:
            最终回复文本
        """
        aggregated_result = action_result["aggregated_result"]
        performance_metrics = action_result["performance_metrics"]
        collab_context = action_result["collaboration_context"]
        
        # 根据聚合类型生成不同风格的回复
        aggregation_type = self.config.get("result_aggregation", "comprehensive")
        
        if aggregation_type == "simple":
            return self._generate_simple_response(aggregated_result)
        elif aggregation_type == "prioritized":
            return self._generate_prioritized_response(aggregated_result, performance_metrics)
        else:  # comprehensive
            return self._generate_comprehensive_response(
                aggregated_result, performance_metrics, collab_context
            )
    
    async def _assess_task_complexity(self, task_content: str) -> str:
        """
        评估任务复杂度
        
        Args:
            task_content: 任务内容
            
        Returns:
            复杂度等级 (low, medium, high, very_high)
        """
        # 简单的复杂度评估逻辑
        complexity_indicators = {
            "high": ["分析", "研究", "比较", "评估", "设计", "优化"],
            "medium": ["处理", "转换", "整理", "汇总", "计算", "查询"],
            "low": ["获取", "读取", "显示", "复制", "保存", "删除"]
        }
        
        task_lower = task_content.lower()
        scores = {"high": 0, "medium": 0, "low": 0}
        
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in task_lower:
                    scores[level] += 1
        
        # 根据任务长度和关键词数量调整
        word_count = len(task_content.split())
        if word_count > 50:
            scores["high"] += 2
        elif word_count > 20:
            scores["medium"] += 1
        
        # 返回得分最高的复杂度等级
        if scores["high"] >= 2:
            return "very_high"
        elif scores["high"] >= 1:
            return "high"
        elif scores["medium"] >= 2:
            return "medium"
        else:
            return "low"
    
    async def _identify_task_type(self, task_content: str) -> str:
        """
        识别任务类型
        
        Args:
            task_content: 任务内容
            
        Returns:
            任务类型
        """
        type_keywords = {
            "research": ["研究", "调查", "查找", "搜索", "了解", "学习"],
            "analysis": ["分析", "评估", "比较", "判断", "检查", "审核"],
            "generation": ["生成", "创建", "编写", "制作", "设计", "开发"],
            "processing": ["处理", "转换", "整理", "清理", "格式化", "计算"]
        }
        
        task_lower = task_content.lower()
        
        for task_type, keywords in type_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return task_type
        
        return "default"
    
    def _decompose_task(self, task_content: str, task_type: str = "default") -> List[SubTask]:
        """
        分解任务为子任务
        
        Args:
            task_content: 任务内容
            task_type: 任务类型
            
        Returns:
            子任务列表
        """
        decomposer = self.decomposition_rules.get(task_type, self.decomposition_rules["default"])
        return decomposer(task_content)
    
    def _default_decomposition(self, task_content: str) -> List[SubTask]:
        """默认任务分解规则"""
        # 简化的分解逻辑
        subtasks = [
            SubTask(
                task_id="subtask_1",
                description=f"分析任务: {task_content[:50]}...",
                agent_type="analysis_agent",
                priority=TaskPriority.HIGH
            ),
            SubTask(
                task_id="subtask_2", 
                description=f"执行任务: {task_content[:50]}...",
                agent_type="execution_agent",
                dependencies=["subtask_1"],
                priority=TaskPriority.MEDIUM
            ),
            SubTask(
                task_id="subtask_3",
                description=f"总结结果: {task_content[:50]}...",
                agent_type="summary_agent",
                dependencies=["subtask_2"],
                priority=TaskPriority.LOW
            )
        ]
        return subtasks
    
    def _research_decomposition(self, task_content: str) -> List[SubTask]:
        """研究类任务分解"""
        return [
            SubTask(
                task_id="research_1",
                description="收集相关资料和信息",
                agent_type="research_agent",
                priority=TaskPriority.HIGH
            ),
            SubTask(
                task_id="research_2",
                description="分析和验证收集的信息",
                agent_type="analysis_agent",
                dependencies=["research_1"],
                priority=TaskPriority.MEDIUM
            ),
            SubTask(
                task_id="research_3",
                description="整理研究发现和结论",
                agent_type="summary_agent",
                dependencies=["research_2"],
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def _analysis_decomposition(self, task_content: str) -> List[SubTask]:
        """分析类任务分解"""
        return [
            SubTask(
                task_id="analysis_1",
                description="数据预处理和清理",
                agent_type="data_agent",
                priority=TaskPriority.HIGH
            ),
            SubTask(
                task_id="analysis_2",
                description="执行核心分析逻辑",
                agent_type="analysis_agent",
                dependencies=["analysis_1"],
                priority=TaskPriority.CRITICAL
            ),
            SubTask(
                task_id="analysis_3",
                description="结果可视化和报告生成",
                agent_type="visualization_agent",
                dependencies=["analysis_2"],
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def _generation_decomposition(self, task_content: str) -> List[SubTask]:
        """生成类任务分解"""
        return [
            SubTask(
                task_id="gen_1",
                description="规划和设计阶段",
                agent_type="planning_agent",
                priority=TaskPriority.HIGH
            ),
            SubTask(
                task_id="gen_2",
                description="内容生成阶段",
                agent_type="generation_agent", 
                dependencies=["gen_1"],
                priority=TaskPriority.CRITICAL
            ),
            SubTask(
                task_id="gen_3",
                description="质量检查和优化",
                agent_type="review_agent",
                dependencies=["gen_2"],
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def _processing_decomposition(self, task_content: str) -> List[SubTask]:
        """处理类任务分解"""
        return [
            SubTask(
                task_id="proc_1",
                description="输入验证和预处理",
                agent_type="validation_agent",
                priority=TaskPriority.HIGH
            ),
            SubTask(
                task_id="proc_2",
                description="核心处理逻辑",
                agent_type="processing_agent",
                dependencies=["proc_1"],
                priority=TaskPriority.CRITICAL
            ),
            SubTask(
                task_id="proc_3",
                description="结果后处理和输出",
                agent_type="output_agent",
                dependencies=["proc_2"],
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def _resolve_dependencies(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        解析和优化任务依赖关系
        
        Args:
            subtasks: 原始子任务列表
            
        Returns:
            优化后的子任务列表
        """
        # 简单的依赖验证和拓扑排序
        task_map = {task.task_id: task for task in subtasks}
        
        # 验证依赖关系的有效性
        for task in subtasks:
            for dep in task.dependencies:
                if dep not in task_map:
                    self.logger.warning(f"任务 {task.task_id} 的依赖 {dep} 不存在")
                    task.dependencies.remove(dep)
        
        # 检测循环依赖（简化版）
        def has_cycle(task_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep in task.dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        # 检查循环依赖
        visited = set()
        for task in subtasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id, visited, set()):
                    self.logger.error(f"检测到循环依赖，涉及任务: {task.task_id}")
                    # 简单处理：移除有问题的依赖
                    task.dependencies.clear()
        
        return subtasks
    
    def _determine_collaboration_strategy(self, complexity: str, subtask_count: int) -> str:
        """
        确定协作策略
        
        Args:
            complexity: 任务复杂度
            subtask_count: 子任务数量
            
        Returns:
            协作策略
        """
        configured_strategy = self.config.get("collaboration_strategy", "hierarchical")
        
        # 根据复杂度和任务数量调整策略
        if complexity in ["very_high", "high"] and subtask_count > 5:
            return "hierarchical"
        elif subtask_count <= 2:
            return "sequential"
        elif subtask_count <= 5 and complexity in ["low", "medium"]:
            return "parallel"
        else:
            return configured_strategy
    
    def _assign_agents(self, subtasks: List[SubTask], strategy: str = "capability_based") -> Dict[str, str]:
        """
        为子任务分配Agent
        
        Args:
            subtasks: 子任务列表
            strategy: 分配策略
            
        Returns:
            任务ID到Agent ID的映射
        """
        assignments = {}
        
        for task in subtasks:
            # 查找合适的Agent
            available_agents = [
                agent_id for agent_id, capabilities in self.agent_capabilities.items()
                if task.agent_type in capabilities
            ]
            
            if available_agents:
                # 简单的负载均衡
                assigned_agent = min(available_agents, key=lambda x: len([
                    t for t, a in assignments.items() if a == x
                ]))
            else:
                # 创建新的模拟Agent
                agent_id = f"{task.agent_type}_{len(self.sub_agents) + 1}"
                self.sub_agents[agent_id] = MockSubAgent(agent_id, task.agent_type)
                self.agent_capabilities[agent_id] = [task.agent_type]
                assigned_agent = agent_id
            
            assignments[task.task_id] = assigned_agent
        
        return assignments
    
    async def _execute_collaboration(self, subtasks: List[SubTask], strategy: str, context: CollaborationContext) -> List[AgentResult]:
        """
        执行协作任务
        
        Args:
            subtasks: 子任务列表
            strategy: 协作策略
            context: 协作上下文
            
        Returns:
            执行结果列表
        """
        if strategy == "sequential":
            return await self._execute_sequential(subtasks, context)
        elif strategy == "parallel":
            return await self._execute_parallel(subtasks, context)
        elif strategy == "hierarchical":
            return await self._execute_hierarchical(subtasks, context)
        elif strategy == "pipeline":
            return await self._execute_pipeline(subtasks, context)
        else:
            # 默认使用顺序执行
            return await self._execute_sequential(subtasks, context)
    
    async def _execute_sequential(self, subtasks: List[SubTask], context: CollaborationContext) -> List[AgentResult]:
        """顺序执行子任务"""
        results = []
        
        # 按依赖关系排序任务
        sorted_tasks = self._topological_sort(subtasks)
        
        for task in sorted_tasks:
            agent_id = self._get_assigned_agent(task.task_id)
            if agent_id and agent_id in self.sub_agents:
                agent = self.sub_agents[agent_id]
                result = await agent.execute_task(task, context)
                results.append(result)
                context.agent_results[task.task_id] = result
                
                # 如果是关键任务失败，可能需要终止
                if not result.success and task.priority == TaskPriority.CRITICAL:
                    self.logger.warning(f"关键任务 {task.task_id} 失败，终止后续执行")
                    break
        
        return results
    
    async def _execute_parallel(self, subtasks: List[SubTask], context: CollaborationContext) -> List[AgentResult]:
        """并行执行子任务"""
        max_parallel = self.config.get("max_parallel_agents", 5)
        
        # 按依赖关系分组
        independent_tasks = [task for task in subtasks if not task.dependencies]
        dependent_tasks = [task for task in subtasks if task.dependencies]
        
        results = []
        
        # 首先执行独立任务
        if independent_tasks:
            parallel_results = await self._execute_batch_parallel(independent_tasks, context, max_parallel)
            results.extend(parallel_results)
            
            # 更新上下文
            for result in parallel_results:
                context.agent_results[result.task_id] = result
        
        # 然后执行有依赖的任务
        for task in dependent_tasks:
            # 检查依赖是否完成
            dependencies_completed = all(
                dep in context.agent_results and context.agent_results[dep].success
                for dep in task.dependencies
            )
            
            if dependencies_completed:
                agent_id = self._get_assigned_agent(task.task_id)
                if agent_id and agent_id in self.sub_agents:
                    agent = self.sub_agents[agent_id]
                    result = await agent.execute_task(task, context)
                    results.append(result)
                    context.agent_results[task.task_id] = result
        
        return results
    
    async def _execute_hierarchical(self, subtasks: List[SubTask], context: CollaborationContext) -> List[AgentResult]:
        """层次化执行子任务"""
        # 按优先级分层
        priority_groups = {}
        for task in subtasks:
            priority = task.priority.value
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(task)
        
        results = []
        
        # 按优先级从高到低执行
        for priority in sorted(priority_groups.keys(), reverse=True):
            group_tasks = priority_groups[priority]
            
            # 每个优先级组内并行执行
            group_results = await self._execute_batch_parallel(group_tasks, context)
            results.extend(group_results)
            
            # 更新上下文
            for result in group_results:
                context.agent_results[result.task_id] = result
        
        return results
    
    async def _execute_pipeline(self, subtasks: List[SubTask], context: CollaborationContext) -> List[AgentResult]:
        """流水线执行子任务"""
        # 简化的流水线实现：按依赖关系顺序执行，但允许重叠
        sorted_tasks = self._topological_sort(subtasks)
        results = []
        
        # 使用异步队列实现流水线
        task_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        
        # 添加任务到队列
        for task in sorted_tasks:
            await task_queue.put(task)
        
        # 启动工作协程
        async def worker():
            while True:
                try:
                    task = await asyncio.wait_for(task_queue.get(), timeout=1.0)
                    agent_id = self._get_assigned_agent(task.task_id)
                    if agent_id and agent_id in self.sub_agents:
                        agent = self.sub_agents[agent_id]
                        result = await agent.execute_task(task, context)
                        await result_queue.put(result)
                    task_queue.task_done()
                except asyncio.TimeoutError:
                    break
        
        # 启动多个工作协程
        workers = [asyncio.create_task(worker()) for _ in range(min(3, len(sorted_tasks)))]
        
        # 收集结果
        completed_count = 0
        while completed_count < len(sorted_tasks):
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=5.0)
                results.append(result)
                context.agent_results[result.task_id] = result
                completed_count += 1
            except asyncio.TimeoutError:
                break
        
        # 清理工作协程
        for worker in workers:
            worker.cancel()
        
        return results
    
    async def _execute_batch_parallel(self, tasks: List[SubTask], context: CollaborationContext, max_parallel: int = None) -> List[AgentResult]:
        """并行执行一批任务"""
        if max_parallel is None:
            max_parallel = self.config.get("max_parallel_agents", 5)
        
        if len(tasks) <= max_parallel:
            # 直接并行执行所有任务
            tasks_to_execute = []
            for task in tasks:
                agent_id = self._get_assigned_agent(task.task_id)
                if agent_id and agent_id in self.sub_agents:
                    agent = self.sub_agents[agent_id]
                    task_coroutine = agent.execute_task(task, context)
                    tasks_to_execute.append(task_coroutine)
            
            results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = AgentResult(
                        agent_id="unknown",
                        task_id=tasks[i].task_id,
                        success=False,
                        error=str(result)
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
        else:
            # 分批执行
            all_results = []
            for i in range(0, len(tasks), max_parallel):
                batch = tasks[i:i + max_parallel]
                batch_results = await self._execute_batch_parallel(batch, context, max_parallel)
                all_results.extend(batch_results)
            return all_results
    
    def _topological_sort(self, tasks: List[SubTask]) -> List[SubTask]:
        """拓扑排序任务"""
        # 简化的拓扑排序实现
        task_map = {task.task_id: task for task in tasks}
        result = []
        visited = set()
        
        def visit(task_id: str):
            if task_id in visited:
                return
            
            visited.add(task_id)
            task = task_map.get(task_id)
            if task:
                # 先访问所有依赖
                for dep in task.dependencies:
                    visit(dep)
                result.append(task)
        
        for task in tasks:
            visit(task.task_id)
        
        return result
    
    def _get_assigned_agent(self, task_id: str) -> Optional[str]:
        """获取任务分配的Agent"""
        # 简化实现：直接查找或创建
        return f"agent_{task_id}"
    
    async def _handle_failed_tasks(self, results: List[AgentResult], context: CollaborationContext) -> List[AgentResult]:
        """处理失败任务的重试"""
        failed_results = [r for r in results if not r.success]
        
        if not failed_results:
            return results
        
        retried_results = []
        for failed_result in failed_results:
            task_id = failed_result.task_id
            
            # 查找原始任务
            original_task = None
            for task in context.shared_data.get("original_tasks", []):
                if hasattr(task, 'task_id') and task.task_id == task_id:
                    original_task = task
                    break
            
            if original_task and original_task.retry_count < original_task.max_retries:
                original_task.retry_count += 1
                
                # 重试任务
                agent_id = self._get_assigned_agent(task_id)
                if agent_id and agent_id in self.sub_agents:
                    agent = self.sub_agents[agent_id]
                    retry_result = await agent.execute_task(original_task, context)
                    retried_results.append(retry_result)
                    
                    self.logger.info(f"任务 {task_id} 重试完成，结果: {retry_result.success}")
        
        # 合并原始结果和重试结果
        final_results = []
        failed_task_ids = {r.task_id for r in failed_results}
        
        for result in results:
            if result.task_id not in failed_task_ids or result.success:
                final_results.append(result)
        
        final_results.extend(retried_results)
        return final_results
    
    def _aggregate_results(self, results: List[AgentResult], aggregation_type: str = "comprehensive") -> Dict[str, Any]:
        """
        汇总多Agent执行结果
        
        Args:
            results: Agent执行结果列表
            aggregation_type: 汇总类型
            
        Returns:
            汇总后的结果
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        base_aggregation = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "total_execution_time": sum(r.execution_time for r in results),
            "average_execution_time": sum(r.execution_time for r in results) / len(results) if results else 0
        }
        
        if aggregation_type == "simple":
            base_aggregation["summary"] = "任务执行完成" if base_aggregation["success_rate"] > 0.8 else "部分任务执行失败"
            return base_aggregation
        
        elif aggregation_type == "prioritized":
            # 优先显示高优先级任务的结果
            high_priority_results = [r for r in successful_results if r.metadata.get("priority", 2) >= 3]
            base_aggregation["high_priority_results"] = [r.result for r in high_priority_results]
            base_aggregation["other_results"] = [r.result for r in successful_results if r not in high_priority_results]
            return base_aggregation
        
        else:  # comprehensive
            base_aggregation.update({
                "detailed_results": {r.task_id: r.result for r in successful_results},
                "error_details": {r.task_id: r.error for r in failed_results},
                "agent_performance": self._calculate_agent_performance(results),
                "execution_timeline": sorted(results, key=lambda r: r.execution_time)
            })
            return base_aggregation
    
    def _calculate_performance_metrics(self, results: List[AgentResult], context: CollaborationContext) -> Dict[str, Any]:
        """计算性能指标"""
        total_time = time.time() - context.start_time
        
        return {
            "total_collaboration_time": total_time,
            "task_completion_rate": context.completed_tasks / context.total_tasks if context.total_tasks > 0 else 0,
            "average_task_time": sum(r.execution_time for r in results) / len(results) if results else 0,
            "parallel_efficiency": context.total_tasks / total_time if total_time > 0 else 0,
            "agent_utilization": len(set(r.agent_id for r in results)) / len(self.sub_agents) if self.sub_agents else 0
        }
    
    def _calculate_agent_performance(self, results: List[AgentResult]) -> Dict[str, Dict[str, Any]]:
        """计算各Agent性能"""
        agent_stats = {}
        
        for result in results:
            agent_id = result.agent_id
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "total_time": 0.0,
                    "average_time": 0.0
                }
            
            stats = agent_stats[agent_id]
            stats["total_tasks"] += 1
            stats["total_time"] += result.execution_time
            
            if result.success:
                stats["successful_tasks"] += 1
            
            stats["average_time"] = stats["total_time"] / stats["total_tasks"]
            stats["success_rate"] = stats["successful_tasks"] / stats["total_tasks"]
        
        return agent_stats
    
    def _estimate_execution_duration(self, subtasks: List[SubTask], strategy: str) -> float:
        """估算执行持续时间"""
        if strategy == "parallel":
            # 并行执行：取最长任务时间
            return max(task.timeout for task in subtasks) if subtasks else 0.0
        elif strategy == "sequential":
            # 顺序执行：累加所有任务时间
            return sum(task.timeout for task in subtasks)
        else:
            # 其他策略：取中间值
            return sum(task.timeout for task in subtasks) * 0.6
    
    def _calculate_resource_requirements(self, subtasks: List[SubTask]) -> Dict[str, Any]:
        """计算资源需求"""
        agent_types = set(task.agent_type for task in subtasks)
        high_priority_tasks = len([task for task in subtasks if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]])
        
        return {
            "required_agent_types": list(agent_types),
            "total_agents_needed": len(agent_types),
            "high_priority_tasks": high_priority_tasks,
            "estimated_memory_usage": len(subtasks) * 0.1,  # 简化的内存估算
            "estimated_cpu_usage": high_priority_tasks * 0.2  # 简化的CPU估算
        }
    
    def _generate_simple_response(self, aggregated_result: Dict[str, Any]) -> str:
        """生成简单回复"""
        success_rate = aggregated_result.get("success_rate", 0)
        if success_rate >= 0.8:
            return "协作任务执行成功！"
        elif success_rate >= 0.5:
            return f"协作任务部分完成，成功率 {success_rate:.1%}"
        else:
            return "协作任务执行遇到较多问题，请检查错误信息。"
    
    def _generate_prioritized_response(self, aggregated_result: Dict[str, Any], performance_metrics: Dict[str, Any]) -> str:
        """生成优先级回复"""
        high_priority_results = aggregated_result.get("high_priority_results", [])
        
        response_parts = []
        response_parts.append(f"协作执行完成，{len(high_priority_results)} 个重要任务成功完成。")
        
        if high_priority_results:
            response_parts.append("重要结果：")
            for i, result in enumerate(high_priority_results[:3], 1):
                response_parts.append(f"{i}. {result}")
        
        completion_rate = performance_metrics.get("task_completion_rate", 0)
        response_parts.append(f"总体完成率：{completion_rate:.1%}")
        
        return "\n".join(response_parts)
    
    def _generate_comprehensive_response(self, aggregated_result: Dict[str, Any], performance_metrics: Dict[str, Any], context: CollaborationContext) -> str:
        """生成详细回复"""
        response_parts = []
        
        # 执行概要
        total_tasks = aggregated_result["total_tasks"]
        successful_tasks = aggregated_result["successful_tasks"]
        success_rate = aggregated_result["success_rate"]
        
        response_parts.append(f"协作任务执行完成：{successful_tasks}/{total_tasks} 个任务成功，成功率 {success_rate:.1%}")
        
        # 性能指标
        total_time = performance_metrics.get("total_collaboration_time", 0)
        response_parts.append(f"总执行时间：{total_time:.2f} 秒")
        
        # 详细结果
        detailed_results = aggregated_result.get("detailed_results", {})
        if detailed_results:
            response_parts.append("\n详细结果：")
            for task_id, result in list(detailed_results.items())[:5]:  # 限制显示数量
                response_parts.append(f"- {task_id}: {result}")
        
        # 错误信息
        error_details = aggregated_result.get("error_details", {})
        if error_details:
            response_parts.append("\n错误信息：")
            for task_id, error in error_details.items():
                response_parts.append(f"- {task_id}: {error}")
        
        return "\n".join(response_parts)
    
    def register_sub_agent(self, agent_id: str, agent: SubAgentInterface, capabilities: List[str]):
        """
        注册子Agent
        
        Args:
            agent_id: Agent ID
            agent: Agent实例
            capabilities: Agent能力列表
        """
        self.sub_agents[agent_id] = agent
        self.agent_capabilities[agent_id] = capabilities
        self.logger.info(f"已注册子Agent: {agent_id}，能力: {capabilities}")
    
    def add_decomposition_rule(self, task_type: str, decomposer: Callable[[str], List[SubTask]]):
        """
        添加任务分解规则
        
        Args:
            task_type: 任务类型
            decomposer: 分解函数
        """
        self.decomposition_rules[task_type] = decomposer
        self.logger.info(f"已添加任务分解规则: {task_type}")
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """
        获取协作统计信息
        
        Returns:
            协作统计数据
        """
        if not self.collaboration_history:
            return {"total_collaborations": 0}
        
        total_collabs = len(self.collaboration_history)
        total_tasks = sum(ctx.total_tasks for ctx in self.collaboration_history)
        completed_tasks = sum(ctx.completed_tasks for ctx in self.collaboration_history)
        
        return {
            "total_collaborations": total_collabs,
            "total_tasks_executed": total_tasks,
            "total_tasks_completed": completed_tasks,
            "overall_success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "registered_agents": len(self.sub_agents),
            "available_capabilities": list(set().union(*self.agent_capabilities.values())) if self.agent_capabilities else [],
            "decomposition_rules": list(self.decomposition_rules.keys())
        }