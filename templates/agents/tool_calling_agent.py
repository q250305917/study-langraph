"""
工具调用Agent模板

专门用于工具调用和外部服务集成的Agent实现，支持智能工具选择、并行调用、错误恢复等功能。
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, ToolDefinition


class ToolCallResult:
    """工具调用结果类"""
    def __init__(self, tool_name: str, success: bool, result: Any = None, error: str = None, execution_time: float = 0.0):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.timestamp = time.time()


class ToolCallingAgent(BaseAgent):
    """
    工具调用Agent模板
    
    专门设计用于工具调用和外部服务集成，具备以下核心能力：
    1. 智能工具选择 - 根据用户需求自动选择最合适的工具
    2. 并行执行 - 支持多个工具的并行调用，提高执行效率
    3. 错误恢复 - 工具失败时的重试机制和备选策略
    4. 调用统计 - 详细的工具使用统计和性能监控
    5. 高级工具 - 内置常用工具如HTTP请求、文件操作等
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 工具调用历史
        self.tool_call_history: List[ToolCallResult] = []
        
        # 工具使用统计
        self.tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # 注册内置高级工具
        self._register_advanced_tools()
    
    @property
    def default_config(self) -> Dict[str, Any]:
        """默认配置"""
        base_config = super().default_config
        tool_config = {
            "max_parallel_tools": 5,      # 最大并行工具数
            "tool_timeout": 10.0,         # 工具执行超时时间
            "retry_attempts": 3,          # 重试次数
            "retry_delay": 1.0,           # 重试延迟
            "enable_parallel_execution": True,
            "tool_selection_strategy": "best_match",  # best_match, all_applicable
            "result_aggregation": "comprehensive",    # simple, comprehensive
        }
        return {**base_config, **tool_config}
    
    def _register_advanced_tools(self):
        """注册高级内置工具"""
        tools = [
            ToolDefinition(
                name="http_request",
                description="发送HTTP请求",
                func=self._http_request,
                async_func=True,
                parameters={
                    "url": {"type": "str", "required": True},
                    "method": {"type": "str", "default": "GET"},
                    "headers": {"type": "dict", "default": {}},
                    "data": {"type": "dict", "default": None}
                }
            ),
            ToolDefinition(
                name="file_operation",
                description="文件操作（读取、写入、删除）",
                func=self._file_operation,
                parameters={
                    "operation": {"type": "str", "required": True},  # read, write, delete
                    "filepath": {"type": "str", "required": True},
                    "content": {"type": "str", "default": None}
                }
            ),
            ToolDefinition(
                name="json_parser",
                description="解析和处理JSON数据",
                func=self._json_parser,
                parameters={
                    "json_string": {"type": "str", "required": True},
                    "operation": {"type": "str", "default": "parse"}  # parse, extract, validate
                }
            ),
            ToolDefinition(
                name="text_processor",
                description="文本处理工具",
                func=self._text_processor,
                parameters={
                    "text": {"type": "str", "required": True},
                    "operation": {"type": "str", "required": True}  # clean, extract, format
                }
            ),
            ToolDefinition(
                name="calculator",
                description="数学计算工具",
                func=self._calculator,
                parameters={
                    "expression": {"type": "str", "required": True}
                }
            )
        ]
        
        self.register_tools(tools)
    
    async def think(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考过程：分析需求、选择工具、制定执行计划
        
        Args:
            input_data: 用户输入
            context: 执行上下文
            
        Returns:
            决策结果包含工具选择和执行计划
        """
        # 1. 分析用户需求
        requirements = await self._analyze_requirements(input_data)
        
        # 2. 选择合适的工具
        selected_tools = await self._select_tools(requirements)
        
        # 3. 制定执行计划
        execution_plan = await self._create_execution_plan(selected_tools, requirements)
        
        # 4. 评估执行复杂度
        complexity = self._assess_complexity(execution_plan)
        
        decision = {
            "requirements": requirements,
            "selected_tools": selected_tools,
            "execution_plan": execution_plan,
            "complexity": complexity,
            "parallel_execution": len(selected_tools) > 1 and self.config.get("enable_parallel_execution", True)
        }
        
        self.logger.info(f"已选择 {len(selected_tools)} 个工具: {[tool['name'] for tool in selected_tools]}")
        
        return decision
    
    async def act(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行动作：按照计划执行工具调用
        
        Args:
            decision: 思考阶段的决策结果
            context: 执行上下文
            
        Returns:
            工具执行结果
        """
        execution_plan = decision["execution_plan"]
        
        if decision["parallel_execution"]:
            # 并行执行
            results = await self._execute_tools_parallel(execution_plan)
        else:
            # 顺序执行
            results = await self._execute_tools_sequential(execution_plan)
        
        # 处理执行结果
        processed_results = await self._process_results(results, decision["requirements"])
        
        # 更新统计信息
        self._update_tool_stats(results)
        
        return {
            "tool_results": results,
            "processed_results": processed_results,
            "execution_summary": self._create_execution_summary(results)
        }
    
    async def respond(self, action_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        生成最终回复：整合工具执行结果，生成用户友好的回复
        
        Args:
            action_result: 动作执行结果
            context: 执行上下文
            
        Returns:
            最终回复文本
        """
        results = action_result["tool_results"]
        processed_results = action_result["processed_results"]
        summary = action_result["execution_summary"]
        
        # 根据配置选择回复风格
        aggregation_style = self.config.get("result_aggregation", "comprehensive")
        
        if aggregation_style == "simple":
            return self._generate_simple_response(processed_results)
        else:
            return self._generate_comprehensive_response(results, processed_results, summary)
    
    async def _analyze_requirements(self, input_data: str) -> Dict[str, Any]:
        """
        分析用户需求
        
        Args:
            input_data: 用户输入
            
        Returns:
            需求分析结果
        """
        # 简单的需求分析逻辑（可以集成NLP模型）
        requirements = {
            "action_type": "unknown",
            "entities": [],
            "parameters": {},
            "urgency": "normal"
        }
        
        input_lower = input_data.lower()
        
        # 检测动作类型
        if any(keyword in input_lower for keyword in ["计算", "算", "数学"]):
            requirements["action_type"] = "calculation"
        elif any(keyword in input_lower for keyword in ["请求", "获取", "下载", "url", "http"]):
            requirements["action_type"] = "http_request"
        elif any(keyword in input_lower for keyword in ["文件", "读取", "写入", "保存"]):
            requirements["action_type"] = "file_operation"
        elif any(keyword in input_lower for keyword in ["json", "解析", "格式化"]):
            requirements["action_type"] = "json_processing"
        elif any(keyword in input_lower for keyword in ["文本", "处理", "清理", "提取"]):
            requirements["action_type"] = "text_processing"
        
        # 提取实体和参数（简化版）
        import re
        
        # 提取URL
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', input_data)
        if urls:
            requirements["entities"].extend(urls)
            requirements["parameters"]["url"] = urls[0]
        
        # 提取数学表达式
        math_expressions = re.findall(r'[\d+\-*/().\s]+', input_data)
        if math_expressions and requirements["action_type"] == "calculation":
            requirements["parameters"]["expression"] = max(math_expressions, key=len).strip()
        
        return requirements
    
    async def _select_tools(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        选择合适的工具
        
        Args:
            requirements: 需求分析结果
            
        Returns:
            选中的工具列表
        """
        action_type = requirements["action_type"]
        
        # 工具映射
        tool_mapping = {
            "calculation": ["calculator"],
            "http_request": ["http_request"],
            "file_operation": ["file_operation"],
            "json_processing": ["json_parser"],
            "text_processing": ["text_processor"]
        }
        
        selected_tool_names = tool_mapping.get(action_type, [])
        
        # 构建工具选择结果
        selected_tools = []
        for tool_name in selected_tool_names:
            if tool_name in self.tools:
                tool_config = {
                    "name": tool_name,
                    "parameters": self._extract_tool_parameters(tool_name, requirements),
                    "priority": 1,
                    "required": True
                }
                selected_tools.append(tool_config)
        
        return selected_tools
    
    def _extract_tool_parameters(self, tool_name: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        为工具提取参数
        
        Args:
            tool_name: 工具名称
            requirements: 需求分析结果
            
        Returns:
            工具参数字典
        """
        base_params = requirements.get("parameters", {})
        
        # 根据工具类型调整参数
        if tool_name == "http_request":
            return {
                "url": base_params.get("url", ""),
                "method": base_params.get("method", "GET"),
                "headers": base_params.get("headers", {}),
                "data": base_params.get("data")
            }
        elif tool_name == "calculator":
            return {
                "expression": base_params.get("expression", "")
            }
        elif tool_name == "file_operation":
            return {
                "operation": base_params.get("operation", "read"),
                "filepath": base_params.get("filepath", ""),
                "content": base_params.get("content")
            }
        
        return base_params
    
    async def _create_execution_plan(self, selected_tools: List[Dict[str, Any]], requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        创建执行计划
        
        Args:
            selected_tools: 选中的工具
            requirements: 需求分析
            
        Returns:
            执行计划列表
        """
        plan = []
        
        for i, tool in enumerate(selected_tools):
            step = {
                "step_id": i + 1,
                "tool_name": tool["name"],
                "parameters": tool["parameters"],
                "dependencies": [],  # 简化版，不处理复杂依赖
                "timeout": self.config.get("tool_timeout", 10.0),
                "retry_attempts": self.config.get("retry_attempts", 3)
            }
            plan.append(step)
        
        return plan
    
    def _assess_complexity(self, execution_plan: List[Dict[str, Any]]) -> str:
        """
        评估执行复杂度
        
        Args:
            execution_plan: 执行计划
            
        Returns:
            复杂度等级 (low, medium, high)
        """
        if len(execution_plan) == 1:
            return "low"
        elif len(execution_plan) <= 3:
            return "medium"
        else:
            return "high"
    
    async def _execute_tools_parallel(self, execution_plan: List[Dict[str, Any]]) -> List[ToolCallResult]:
        """
        并行执行工具
        
        Args:
            execution_plan: 执行计划
            
        Returns:
            工具执行结果列表
        """
        max_parallel = self.config.get("max_parallel_tools", 5)
        
        # 限制并行数量
        if len(execution_plan) > max_parallel:
            # 分批执行
            results = []
            for i in range(0, len(execution_plan), max_parallel):
                batch = execution_plan[i:i + max_parallel]
                batch_results = await self._execute_batch_parallel(batch)
                results.extend(batch_results)
            return results
        else:
            return await self._execute_batch_parallel(execution_plan)
    
    async def _execute_batch_parallel(self, batch: List[Dict[str, Any]]) -> List[ToolCallResult]:
        """
        并行执行一批工具
        
        Args:
            batch: 工具批次
            
        Returns:
            执行结果列表
        """
        tasks = []
        for step in batch:
            task = asyncio.create_task(
                self._execute_single_tool(step["tool_name"], step["parameters"])
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ToolCallResult(
                    tool_name=batch[i]["tool_name"],
                    success=False,
                    error=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_tools_sequential(self, execution_plan: List[Dict[str, Any]]) -> List[ToolCallResult]:
        """
        顺序执行工具
        
        Args:
            execution_plan: 执行计划
            
        Returns:
            工具执行结果列表
        """
        results = []
        
        for step in execution_plan:
            result = await self._execute_single_tool(step["tool_name"], step["parameters"])
            results.append(result)
            
            # 如果是关键工具失败，可以选择终止执行
            if not result.success and step.get("required", False):
                self.logger.warning(f"关键工具 {step['tool_name']} 执行失败，终止后续执行")
                break
        
        return results
    
    async def _execute_single_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolCallResult:
        """
        执行单个工具
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            工具执行结果
        """
        start_time = time.time()
        
        try:
            result = await self.call_tool(tool_name, **parameters)
            execution_time = time.time() - start_time
            
            tool_result = ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
            self.tool_call_history.append(tool_result)
            return tool_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            tool_result = ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
            
            self.tool_call_history.append(tool_result)
            return tool_result
    
    async def _process_results(self, results: List[ToolCallResult], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理工具执行结果
        
        Args:
            results: 工具执行结果
            requirements: 原始需求
            
        Returns:
            处理后的结果
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        processed = {
            "total_tools": len(results),
            "successful_tools": len(successful_results),
            "failed_tools": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "results": {},
            "errors": {},
            "aggregated_result": None
        }
        
        # 收集成功结果
        for result in successful_results:
            processed["results"][result.tool_name] = result.result
        
        # 收集错误信息
        for result in failed_results:
            processed["errors"][result.tool_name] = result.error
        
        # 聚合结果（根据需求类型）
        if successful_results:
            processed["aggregated_result"] = self._aggregate_results(successful_results, requirements)
        
        return processed
    
    def _aggregate_results(self, results: List[ToolCallResult], requirements: Dict[str, Any]) -> Any:
        """
        聚合工具结果
        
        Args:
            results: 成功的工具结果
            requirements: 原始需求
            
        Returns:
            聚合后的结果
        """
        if len(results) == 1:
            return results[0].result
        
        # 根据需求类型进行不同的聚合策略
        action_type = requirements.get("action_type", "unknown")
        
        if action_type == "calculation":
            # 对于计算，返回最后一个结果
            return results[-1].result
        elif action_type == "http_request":
            # 对于HTTP请求，合并所有结果
            return [r.result for r in results]
        else:
            # 默认策略：返回所有结果
            return {r.tool_name: r.result for r in results}
    
    def _create_execution_summary(self, results: List[ToolCallResult]) -> Dict[str, Any]:
        """
        创建执行摘要
        
        Args:
            results: 工具执行结果
            
        Returns:
            执行摘要
        """
        total_time = sum(r.execution_time for r in results)
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return {
            "total_execution_time": total_time,
            "total_tools_executed": len(results),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "fastest_tool": min(results, key=lambda r: r.execution_time).tool_name if results else None,
            "slowest_tool": max(results, key=lambda r: r.execution_time).tool_name if results else None
        }
    
    def _update_tool_stats(self, results: List[ToolCallResult]):
        """
        更新工具使用统计
        
        Args:
            results: 工具执行结果
        """
        for result in results:
            tool_name = result.tool_name
            
            if tool_name not in self.tool_usage_stats:
                self.tool_usage_stats[tool_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_execution_time": 0.0,
                    "average_execution_time": 0.0
                }
            
            stats = self.tool_usage_stats[tool_name]
            stats["total_calls"] += 1
            stats["total_execution_time"] += result.execution_time
            
            if result.success:
                stats["successful_calls"] += 1
            else:
                stats["failed_calls"] += 1
            
            stats["average_execution_time"] = stats["total_execution_time"] / stats["total_calls"]
    
    def _generate_simple_response(self, processed_results: Dict[str, Any]) -> str:
        """
        生成简单回复
        
        Args:
            processed_results: 处理后的结果
            
        Returns:
            简单回复文本
        """
        if processed_results["aggregated_result"] is not None:
            return f"执行结果：{processed_results['aggregated_result']}"
        elif processed_results["failed_tools"] > 0:
            return f"执行失败，{processed_results['failed_tools']} 个工具出现错误。"
        else:
            return "执行完成，但没有返回结果。"
    
    def _generate_comprehensive_response(self, results: List[ToolCallResult], processed_results: Dict[str, Any], summary: Dict[str, Any]) -> str:
        """
        生成详细回复
        
        Args:
            results: 原始工具结果
            processed_results: 处理后的结果
            summary: 执行摘要
            
        Returns:
            详细回复文本
        """
        response_parts = []
        
        # 添加执行摘要
        response_parts.append(f"执行了 {summary['total_tools_executed']} 个工具，成功率 {summary['success_rate']:.1%}")
        
        # 添加主要结果
        if processed_results["aggregated_result"] is not None:
            response_parts.append(f"主要结果：{processed_results['aggregated_result']}")
        
        # 添加各工具详细结果
        if processed_results["results"]:
            response_parts.append("详细结果：")
            for tool_name, result in processed_results["results"].items():
                response_parts.append(f"- {tool_name}: {result}")
        
        # 添加错误信息
        if processed_results["errors"]:
            response_parts.append("错误信息：")
            for tool_name, error in processed_results["errors"].items():
                response_parts.append(f"- {tool_name}: {error}")
        
        return "\n".join(response_parts)
    
    # 内置工具实现
    
    async def _http_request(self, url: str, method: str = "GET", headers: Dict = None, data: Dict = None) -> Dict[str, Any]:
        """HTTP请求工具"""
        # 简化的HTTP请求实现
        return {
            "status": "simulated",
            "url": url,
            "method": method,
            "message": f"模拟 {method} 请求到 {url}"
        }
    
    def _file_operation(self, operation: str, filepath: str, content: str = None) -> Dict[str, Any]:
        """文件操作工具"""
        if operation == "read":
            return {"operation": "read", "filepath": filepath, "content": "模拟文件内容"}
        elif operation == "write":
            return {"operation": "write", "filepath": filepath, "status": "success"}
        elif operation == "delete":
            return {"operation": "delete", "filepath": filepath, "status": "success"}
        else:
            raise ValueError(f"不支持的文件操作: {operation}")
    
    def _json_parser(self, json_string: str, operation: str = "parse") -> Dict[str, Any]:
        """JSON解析工具"""
        try:
            if operation == "parse":
                parsed_data = json.loads(json_string)
                return {"operation": "parse", "result": parsed_data, "status": "success"}
            else:
                return {"operation": operation, "status": "not_implemented"}
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {e}")
    
    def _text_processor(self, text: str, operation: str) -> Dict[str, Any]:
        """文本处理工具"""
        if operation == "clean":
            cleaned_text = text.strip().replace("\n", " ")
            return {"operation": "clean", "result": cleaned_text}
        elif operation == "extract":
            # 简化的提取逻辑
            words = text.split()
            return {"operation": "extract", "word_count": len(words), "words": words[:10]}
        elif operation == "format":
            formatted_text = text.title()
            return {"operation": "format", "result": formatted_text}
        else:
            raise ValueError(f"不支持的文本操作: {operation}")
    
    def _calculator(self, expression: str) -> Dict[str, Any]:
        """计算器工具"""
        try:
            # 安全的数学表达式求值（简化版）
            import re
            
            # 只允许数字、运算符和括号
            if re.match(r'^[\d+\-*/().\s]+$', expression):
                result = eval(expression)  # 注意：实际使用中应该用更安全的求值方法
                return {"expression": expression, "result": result, "status": "success"}
            else:
                raise ValueError("表达式包含不允许的字符")
        except Exception as e:
            raise ValueError(f"计算失败: {e}")
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """
        获取工具使用统计
        
        Returns:
            工具使用统计信息
        """
        return {
            "registered_tools": len(self.tools),
            "total_calls": len(self.tool_call_history),
            "successful_calls": len([r for r in self.tool_call_history if r.success]),
            "failed_calls": len([r for r in self.tool_call_history if not r.success]),
            "tool_stats": self.tool_usage_stats.copy()
        }