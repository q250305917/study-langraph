"""
RAG (检索增强生成) Agent模板

专门用于检索增强生成场景的Agent实现，集成知识库查询、文档检索和答案生成功能。
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .base_agent import BaseAgent, ToolDefinition


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class KnowledgeSource:
    """知识源配置"""
    name: str
    type: str  # vector_db, file_system, web_api
    config: Dict[str, Any]
    enabled: bool = True


class BaseRetriever(ABC):
    """基础检索器抽象类"""
    
    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        pass


class MockRetriever(BaseRetriever):
    """模拟检索器（用于演示）"""
    
    def __init__(self):
        # 模拟知识库
        self.knowledge_base = [
            {
                "content": "Python是一种高级编程语言，以其简单易学和功能强大而闻名。",
                "source": "python_intro.txt",
                "metadata": {"topic": "programming", "language": "python"}
            },
            {
                "content": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
                "source": "ml_basics.txt", 
                "metadata": {"topic": "ai", "category": "machine_learning"}
            },
            {
                "content": "LangChain是一个用于构建LLM应用程序的框架，提供了丰富的工具和组件。",
                "source": "langchain_docs.txt",
                "metadata": {"topic": "framework", "technology": "langchain"}
            }
        ]
    
    async def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """模拟检索过程"""
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_base:
            # 简单的相关性计算
            content_lower = item["content"].lower()
            score = 0.0
            
            # 计算关键词匹配度
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 1.0
            
            # 归一化分数
            score = score / len(query_words) if query_words else 0.0
            
            if score > 0:
                result = RetrievalResult(
                    content=item["content"],
                    source=item["source"],
                    score=score,
                    metadata=item["metadata"],
                    timestamp=time.time()
                )
                results.append(result)
        
        # 按分数排序并返回前k个结果
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]


class RAGAgent(BaseAgent):
    """
    RAG (检索增强生成) Agent模板
    
    专门设计用于检索增强生成场景，具备以下核心能力：
    1. 智能检索 - 多策略文档检索（语义检索、关键词检索等）
    2. 相关性排序 - 基于多种因素的结果排序和过滤
    3. 答案合成 - 基于检索内容生成准确、相关的答案
    4. 缓存机制 - 检索结果缓存，提高响应速度
    5. 多源支持 - 支持多种知识源的配置和管理
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 检索器
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.default_retriever: Optional[BaseRetriever] = None
        
        # 检索缓存
        self.retrieval_cache: Dict[str, List[RetrievalResult]] = {}
        
        # 知识源配置
        self.knowledge_sources: List[KnowledgeSource] = []
        
        # 检索历史
        self.retrieval_history: List[Dict[str, Any]] = []
        
        # 初始化默认检索器
        self._setup_default_retriever()
        
        # 注册内置工具
        self._register_builtin_tools()
    
    @property
    def default_config(self) -> Dict[str, Any]:
        """默认配置"""
        base_config = super().default_config
        rag_config = {
            "retrieval_strategy": "hybrid",  # semantic, keyword, hybrid
            "max_retrieved_docs": 5,         # 最大检索文档数
            "relevance_threshold": 0.3,      # 相关性阈值
            "enable_caching": True,          # 启用缓存
            "cache_ttl": 3600,              # 缓存时间（秒）
            "answer_generation_style": "comprehensive",  # concise, comprehensive, detailed
            "include_sources": True,         # 在答案中包含来源
            "max_context_length": 2000,      # 最大上下文长度
            "rerank_results": True,          # 重新排序结果
        }
        return {**base_config, **rag_config}
    
    def _setup_default_retriever(self):
        """设置默认检索器"""
        self.default_retriever = MockRetriever()
        self.retrievers["default"] = self.default_retriever
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        tools = [
            ToolDefinition(
                name="retrieve_documents",
                description="检索相关文档",
                func=self._retrieve_documents,
                async_func=True,
                parameters={
                    "query": {"type": "str", "required": True},
                    "k": {"type": "int", "default": 5},
                    "retriever_name": {"type": "str", "default": "default"}
                }
            ),
            ToolDefinition(
                name="rerank_results",
                description="重新排序检索结果",
                func=self._rerank_results,
                parameters={
                    "results": {"type": "list", "required": True},
                    "query": {"type": "str", "required": True}
                }
            ),
            ToolDefinition(
                name="filter_by_relevance",
                description="按相关性过滤结果",
                func=self._filter_by_relevance,
                parameters={
                    "results": {"type": "list", "required": True},
                    "threshold": {"type": "float", "default": 0.3}
                }
            ),
            ToolDefinition(
                name="extract_context",
                description="提取上下文信息",
                func=self._extract_context,
                parameters={
                    "results": {"type": "list", "required": True},
                    "max_length": {"type": "int", "default": 2000}
                }
            ),
            ToolDefinition(
                name="generate_answer",
                description="基于检索结果生成答案",
                func=self._generate_answer,
                async_func=True,
                parameters={
                    "query": {"type": "str", "required": True},
                    "context": {"type": "str", "required": True},
                    "style": {"type": "str", "default": "comprehensive"}
                }
            )
        ]
        
        self.register_tools(tools)
    
    async def think(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考过程：分析查询、制定检索策略、评估检索需求
        
        Args:
            input_data: 用户查询
            context: 执行上下文
            
        Returns:
            决策结果包含检索策略和参数
        """
        # 1. 查询分析
        query_analysis = await self._analyze_query(input_data)
        
        # 2. 制定检索策略
        retrieval_strategy = self._determine_retrieval_strategy(query_analysis)
        
        # 3. 配置检索参数
        retrieval_params = self._configure_retrieval_params(query_analysis, retrieval_strategy)
        
        # 4. 评估缓存
        cache_key = self._generate_cache_key(input_data, retrieval_params)
        use_cache = self.config.get("enable_caching", True) and cache_key in self.retrieval_cache
        
        decision = {
            "query_analysis": query_analysis,
            "retrieval_strategy": retrieval_strategy,
            "retrieval_params": retrieval_params,
            "cache_key": cache_key,
            "use_cache": use_cache,
            "expected_sources": self._identify_relevant_sources(query_analysis)
        }
        
        self.logger.info(f"查询分析完成 - 策略: {retrieval_strategy}, 参数: {retrieval_params}")
        
        return decision
    
    async def act(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行动作：执行文档检索、重排序、过滤等操作
        
        Args:
            decision: 思考阶段的决策结果
            context: 执行上下文
            
        Returns:
            检索和处理结果
        """
        query = context["input"]
        
        # 1. 检索文档
        if decision["use_cache"]:
            retrieved_docs = self.retrieval_cache[decision["cache_key"]]
            self.logger.info(f"使用缓存结果，共 {len(retrieved_docs)} 个文档")
        else:
            retrieved_docs = await self.call_tool(
                "retrieve_documents",
                query=query,
                k=decision["retrieval_params"]["max_docs"],
                retriever_name=decision["retrieval_params"]["retriever"]
            )
            
            # 缓存结果
            if self.config.get("enable_caching", True):
                self.retrieval_cache[decision["cache_key"]] = retrieved_docs
        
        # 2. 过滤相关性
        filtered_docs = await self.call_tool(
            "filter_by_relevance",
            results=retrieved_docs,
            threshold=decision["retrieval_params"]["relevance_threshold"]
        )
        
        # 3. 重新排序（如果启用）
        if self.config.get("rerank_results", True) and len(filtered_docs) > 1:
            reranked_docs = await self.call_tool(
                "rerank_results",
                results=filtered_docs,
                query=query
            )
        else:
            reranked_docs = filtered_docs
        
        # 4. 提取上下文
        context_text = await self.call_tool(
            "extract_context",
            results=reranked_docs,
            max_length=self.config.get("max_context_length", 2000)
        )
        
        # 5. 记录检索历史
        await self._record_retrieval(query, retrieved_docs, filtered_docs, reranked_docs)
        
        return {
            "retrieved_docs": retrieved_docs,
            "filtered_docs": filtered_docs,
            "reranked_docs": reranked_docs,
            "context_text": context_text,
            "retrieval_stats": {
                "total_retrieved": len(retrieved_docs),
                "after_filtering": len(filtered_docs),
                "final_count": len(reranked_docs),
                "used_cache": decision["use_cache"]
            }
        }
    
    async def respond(self, action_result: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        生成最终回复：基于检索结果生成答案
        
        Args:
            action_result: 动作执行结果
            context: 执行上下文
            
        Returns:
            生成的答案文本
        """
        query = context["input"]
        context_text = action_result["context_text"]
        reranked_docs = action_result["reranked_docs"]
        
        # 生成答案
        answer = await self.call_tool(
            "generate_answer",
            query=query,
            context=context_text,
            style=self.config.get("answer_generation_style", "comprehensive")
        )
        
        # 添加来源信息（如果启用）
        if self.config.get("include_sources", True) and reranked_docs:
            sources = self._format_sources(reranked_docs)
            answer += f"\n\n**参考来源：**\n{sources}"
        
        return answer
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析用户查询
        
        Args:
            query: 用户查询
            
        Returns:
            查询分析结果
        """
        query_lower = query.lower()
        
        analysis = {
            "query_type": "factual",  # factual, conceptual, procedural
            "complexity": "medium",   # low, medium, high
            "domain": "general",      # general, technical, academic
            "keywords": [],
            "intent": "information_seeking"  # information_seeking, problem_solving, explanation
        }
        
        # 简单的查询类型判断
        if any(word in query_lower for word in ["如何", "怎么", "步骤", "方法"]):
            analysis["query_type"] = "procedural"
            analysis["intent"] = "problem_solving"
        elif any(word in query_lower for word in ["什么是", "定义", "概念", "原理"]):
            analysis["query_type"] = "conceptual"
            analysis["intent"] = "explanation"
        
        # 提取关键词（简化版）
        import re
        words = re.findall(r'\b\w+\b', query)
        analysis["keywords"] = [word for word in words if len(word) > 2]
        
        # 判断复杂度
        if len(analysis["keywords"]) > 5:
            analysis["complexity"] = "high"
        elif len(analysis["keywords"]) < 3:
            analysis["complexity"] = "low"
        
        return analysis
    
    def _determine_retrieval_strategy(self, query_analysis: Dict[str, Any]) -> str:
        """
        确定检索策略
        
        Args:
            query_analysis: 查询分析结果
            
        Returns:
            检索策略
        """
        configured_strategy = self.config.get("retrieval_strategy", "hybrid")
        
        # 根据查询特征调整策略
        if query_analysis["query_type"] == "factual" and query_analysis["complexity"] == "low":
            return "keyword"
        elif query_analysis["query_type"] == "conceptual":
            return "semantic"
        else:
            return configured_strategy
    
    def _configure_retrieval_params(self, query_analysis: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """
        配置检索参数
        
        Args:
            query_analysis: 查询分析
            strategy: 检索策略
            
        Returns:
            检索参数配置
        """
        base_params = {
            "max_docs": self.config.get("max_retrieved_docs", 5),
            "relevance_threshold": self.config.get("relevance_threshold", 0.3),
            "retriever": "default"
        }
        
        # 根据复杂度调整参数
        if query_analysis["complexity"] == "high":
            base_params["max_docs"] = min(base_params["max_docs"] + 3, 10)
        elif query_analysis["complexity"] == "low":
            base_params["max_docs"] = max(base_params["max_docs"] - 2, 3)
        
        # 根据策略调整相关性阈值
        if strategy == "semantic":
            base_params["relevance_threshold"] *= 0.8  # 语义检索容忍度更高
        elif strategy == "keyword":
            base_params["relevance_threshold"] *= 1.2  # 关键词检索要求更严格
        
        return base_params
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            query: 查询文本
            params: 检索参数
            
        Returns:
            缓存键
        """
        import hashlib
        
        key_data = f"{query}_{params['max_docs']}_{params['relevance_threshold']}_{params['retriever']}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _identify_relevant_sources(self, query_analysis: Dict[str, Any]) -> List[str]:
        """
        识别相关知识源
        
        Args:
            query_analysis: 查询分析
            
        Returns:
            相关知识源列表
        """
        # 简化版实现
        return ["default"]
    
    async def _retrieve_documents(self, query: str, k: int = 5, retriever_name: str = "default") -> List[RetrievalResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            retriever_name: 检索器名称
            
        Returns:
            检索结果列表
        """
        # 检查缓存
        cache_key = self._generate_cache_key(query, {"max_docs": k, "relevance_threshold": 0.0, "retriever": retriever_name})
        if self.config.get("enable_caching", True) and cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        # 执行检索
        if retriever_name in self.retrievers:
            retriever = self.retrievers[retriever_name]
        else:
            retriever = self.default_retriever
        
        if retriever:
            results = await retriever.retrieve(query, k)
            
            # 缓存结果
            if self.config.get("enable_caching", True):
                self.retrieval_cache[cache_key] = results
            
            return results
        
        return []
    
    def _rerank_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        重新排序检索结果
        
        Args:
            results: 原始检索结果
            query: 查询文本
            
        Returns:
            重新排序后的结果
        """
        # 简化的重排序逻辑
        query_words = set(query.lower().split())
        
        def calculate_enhanced_score(result: RetrievalResult) -> float:
            # 基础相关性分数
            base_score = result.score
            
            # 计算额外因素
            content_words = set(result.content.lower().split())
            word_overlap = len(query_words.intersection(content_words))
            overlap_bonus = word_overlap * 0.1
            
            # 时效性加权（较新的文档得分更高）
            time_bonus = 0.05 if result.timestamp > time.time() - 86400 else 0  # 24小时内
            
            return base_score + overlap_bonus + time_bonus
        
        # 重新计算分数并排序
        for result in results:
            result.score = calculate_enhanced_score(result)
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _filter_by_relevance(self, results: List[RetrievalResult], threshold: float = 0.3) -> List[RetrievalResult]:
        """
        按相关性过滤结果
        
        Args:
            results: 检索结果
            threshold: 相关性阈值
            
        Returns:
            过滤后的结果
        """
        return [result for result in results if result.score >= threshold]
    
    def _extract_context(self, results: List[RetrievalResult], max_length: int = 2000) -> str:
        """
        提取上下文信息
        
        Args:
            results: 检索结果
            max_length: 最大长度
            
        Returns:
            上下文文本
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result.content
            if current_length + len(content) <= max_length:
                context_parts.append(f"[来源: {result.source}] {content}")
                current_length += len(content) + len(result.source) + 10  # 额外格式字符
            else:
                # 截取部分内容
                remaining_length = max_length - current_length
                if remaining_length > 50:  # 确保有足够空间显示有意义的内容
                    truncated_content = content[:remaining_length-3] + "..."
                    context_parts.append(f"[来源: {result.source}] {truncated_content}")
                break
        
        return "\n\n".join(context_parts)
    
    async def _generate_answer(self, query: str, context: str, style: str = "comprehensive") -> str:
        """
        基于检索结果生成答案
        
        Args:
            query: 用户查询
            context: 上下文信息
            style: 生成风格
            
        Returns:
            生成的答案
        """
        # 简化的答案生成逻辑（实际应用中会集成LLM）
        if not context:
            return "抱歉，我没有找到相关的信息来回答您的问题。"
        
        # 根据风格调整答案
        if style == "concise":
            answer_template = "根据相关资料，{answer}"
        elif style == "detailed":
            answer_template = "基于多个来源的信息分析，{answer}\n\n详细说明：{context}"
        else:  # comprehensive
            answer_template = "根据检索到的相关资料，{answer}"
        
        # 简单的答案提取（实际应用中会使用更复杂的NLG技术）
        first_sentence = context.split('.')[0] if '.' in context else context[:100]
        answer = first_sentence.strip()
        
        if style == "detailed":
            return answer_template.format(answer=answer, context=context[:500])
        else:
            return answer_template.format(answer=answer)
    
    def _format_sources(self, results: List[RetrievalResult]) -> str:
        """
        格式化来源信息
        
        Args:
            results: 检索结果
            
        Returns:
            格式化的来源文本
        """
        sources = []
        for i, result in enumerate(results[:3], 1):  # 只显示前3个来源
            sources.append(f"{i}. {result.source} (相关度: {result.score:.2f})")
        
        return "\n".join(sources)
    
    async def _record_retrieval(self, query: str, retrieved: List[RetrievalResult], filtered: List[RetrievalResult], final: List[RetrievalResult]):
        """
        记录检索历史
        
        Args:
            query: 查询文本
            retrieved: 原始检索结果
            filtered: 过滤后结果
            final: 最终结果
        """
        record = {
            "timestamp": time.time(),
            "query": query,
            "retrieved_count": len(retrieved),
            "filtered_count": len(filtered),
            "final_count": len(final),
            "avg_score": sum(r.score for r in final) / len(final) if final else 0.0,
            "sources": [r.source for r in final]
        }
        
        self.retrieval_history.append(record)
        
        # 保持历史记录大小
        if len(self.retrieval_history) > 100:
            self.retrieval_history = self.retrieval_history[-100:]
    
    def add_knowledge_source(self, source: KnowledgeSource):
        """
        添加知识源
        
        Args:
            source: 知识源配置
        """
        self.knowledge_sources.append(source)
        self.logger.info(f"已添加知识源: {source.name}")
    
    def add_retriever(self, name: str, retriever: BaseRetriever):
        """
        添加检索器
        
        Args:
            name: 检索器名称
            retriever: 检索器实例
        """
        self.retrievers[name] = retriever
        self.logger.info(f"已添加检索器: {name}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Returns:
            检索统计数据
        """
        if not self.retrieval_history:
            return {"total_retrievals": 0}
        
        total_retrievals = len(self.retrieval_history)
        avg_retrieved = sum(r["retrieved_count"] for r in self.retrieval_history) / total_retrievals
        avg_final = sum(r["final_count"] for r in self.retrieval_history) / total_retrievals
        avg_score = sum(r["avg_score"] for r in self.retrieval_history) / total_retrievals
        
        return {
            "total_retrievals": total_retrievals,
            "avg_retrieved_per_query": avg_retrieved,
            "avg_final_results_per_query": avg_final,
            "avg_relevance_score": avg_score,
            "cache_hit_rate": len(self.retrieval_cache) / total_retrievals if total_retrievals > 0 else 0,
            "registered_retrievers": len(self.retrievers),
            "knowledge_sources": len(self.knowledge_sources)
        }