"""
RAG生成链模块

实现基于检索增强生成的核心链条，包括基础RAG链和对话式RAG链。
支持上下文管理、提示词模板和回答质量控制。
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

# LangChain导入
from langchain.schema import Document, BaseMessage, HumanMessage, AIMessage
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel

# 本地导入
from .prompts import PromptManager, create_rag_prompt
from .context import ContextManager

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG响应结果"""
    answer: str
    source_documents: List[Document]
    context_used: str
    confidence_score: float = 0.0
    generation_time: float = 0.0
    token_usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class RAGChain:
    """RAG生成链
    
    核心的检索增强生成链，结合检索器和LLM生成回答。
    """
    
    def __init__(self,
                 llm: Union[BaseLLM, BaseChatModel],
                 retriever,
                 prompt_manager: Optional[PromptManager] = None,
                 context_manager: Optional[ContextManager] = None,
                 max_context_length: int = 4000,
                 include_sources: bool = True,
                 return_source_documents: bool = True):
        """
        初始化RAG链
        
        Args:
            llm: 语言模型
            retriever: 检索器
            prompt_manager: 提示词管理器
            context_manager: 上下文管理器
            max_context_length: 最大上下文长度
            include_sources: 是否在回答中包含来源
            return_source_documents: 是否返回源文档
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_manager = prompt_manager or PromptManager()
        self.context_manager = context_manager or ContextManager()
        self.max_context_length = max_context_length
        self.include_sources = include_sources
        self.return_source_documents = return_source_documents
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_tokens_used": 0,
            "total_generation_time": 0.0,
            "avg_confidence": 0.0
        }
        
        logger.info("初始化RAG生成链")
    
    def run(self, 
            query: str,
            k: int = 5,
            **kwargs) -> RAGResponse:
        """
        执行RAG查询
        
        Args:
            query: 用户查询
            k: 检索文档数量
            **kwargs: 额外参数
            
        Returns:
            RAGResponse对象
        """
        start_time = time.time()
        
        try:
            self.stats["total_queries"] += 1
            
            # 1. 检索相关文档
            logger.debug(f"检索文档: {query[:50]}")
            retrieved_docs = self.retriever.retrieve(query, k=k)
            
            if hasattr(retrieved_docs, 'documents'):
                # 如果是RetrievalResult对象
                documents = retrieved_docs.documents
                scores = retrieved_docs.scores
            else:
                # 如果是Document列表
                documents = retrieved_docs
                scores = None
            
            if not documents:
                logger.warning("未找到相关文档")
                return RAGResponse(
                    answer="抱歉，我没有找到相关信息来回答您的问题。",
                    source_documents=[],
                    context_used="",
                    generation_time=time.time() - start_time
                )
            
            # 2. 构建上下文
            context = self.context_manager.build_context(
                documents,
                query,
                max_length=self.max_context_length
            )
            
            # 3. 生成提示词
            prompt = self.prompt_manager.format_rag_prompt(
                query=query,
                context=context,
                include_sources=self.include_sources
            )
            
            # 4. 生成回答
            logger.debug("生成回答")
            if isinstance(self.llm, BaseChatModel):
                # 聊天模型
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                answer = response.content
                token_usage = getattr(response, 'usage_metadata', None)
            else:
                # 基础语言模型
                response = self.llm.invoke(prompt)
                answer = response
                token_usage = None
            
            # 5. 后处理回答
            processed_answer = self._post_process_answer(answer, documents)
            
            # 6. 计算置信度
            confidence = self._calculate_confidence(query, processed_answer, documents, scores)
            
            # 7. 构建响应
            response = RAGResponse(
                answer=processed_answer,
                source_documents=documents if self.return_source_documents else [],
                context_used=context,
                confidence_score=confidence,
                generation_time=time.time() - start_time,
                token_usage=token_usage,
                metadata={
                    "retrieved_docs_count": len(documents),
                    "context_length": len(context),
                    "prompt_template": self.prompt_manager.current_template_name
                }
            )
            
            # 更新统计
            self._update_stats(response, success=True)
            
            logger.info(f"RAG查询完成，置信度: {confidence:.2f}")
            return response
        
        except Exception as e:
            logger.error(f"RAG查询失败: {e}")
            self.stats["failed_queries"] += 1
            
            # 返回错误响应
            return RAGResponse(
                answer=f"抱歉，处理您的查询时出现错误: {str(e)}",
                source_documents=[],
                context_used="",
                generation_time=time.time() - start_time
            )
    
    def _post_process_answer(self, answer: str, documents: List[Document]) -> str:
        """后处理回答"""
        # 清理回答
        answer = answer.strip()
        
        # 添加来源引用（如果启用）
        if self.include_sources and documents:
            sources = []
            for i, doc in enumerate(documents[:3]):  # 只显示前3个来源
                source = doc.metadata.get('source', f'文档{i+1}')
                sources.append(f"[{i+1}] {source}")
            
            if sources:
                answer += f"\n\n参考来源:\n" + "\n".join(sources)
        
        return answer
    
    def _calculate_confidence(self, 
                            query: str, 
                            answer: str, 
                            documents: List[Document],
                            scores: Optional[List[float]]) -> float:
        """计算回答的置信度"""
        try:
            confidence_factors = []
            
            # 因子1: 检索文档的平均分数
            if scores:
                avg_retrieval_score = sum(scores) / len(scores)
                confidence_factors.append(min(avg_retrieval_score, 1.0))
            else:
                confidence_factors.append(0.5)
            
            # 因子2: 回答长度（太短或太长都降低置信度）
            answer_length = len(answer.split())
            if 10 <= answer_length <= 200:
                length_score = 1.0
            elif answer_length < 10:
                length_score = answer_length / 10.0
            else:
                length_score = max(0.5, 200.0 / answer_length)
            confidence_factors.append(length_score)
            
            # 因子3: 检索文档数量
            doc_count_score = min(len(documents) / 5.0, 1.0)
            confidence_factors.append(doc_count_score)
            
            # 因子4: 关键词匹配度
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            if query_words:
                keyword_overlap = len(query_words & answer_words) / len(query_words)
                confidence_factors.append(keyword_overlap)
            
            # 综合置信度
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
            
            return round(overall_confidence, 3)
        
        except Exception as e:
            logger.warning(f"计算置信度失败: {e}")
            return 0.5
    
    def _update_stats(self, response: RAGResponse, success: bool):
        """更新统计信息"""
        if success:
            self.stats["successful_queries"] += 1
            self.stats["total_generation_time"] += response.generation_time
            
            # 更新平均置信度
            total_successful = self.stats["successful_queries"]
            current_avg = self.stats["avg_confidence"]
            new_avg = (current_avg * (total_successful - 1) + response.confidence_score) / total_successful
            self.stats["avg_confidence"] = new_avg
            
            # 更新token使用量
            if response.token_usage:
                total_tokens = response.token_usage.get('total_tokens', 0)
                self.stats["total_tokens_used"] += total_tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
        
        # 计算平均生成时间
        if stats["successful_queries"] > 0:
            stats["avg_generation_time"] = stats["total_generation_time"] / stats["successful_queries"]
        
        return stats


class ConversationalRAGChain(RAGChain):
    """对话式RAG链
    
    支持多轮对话的RAG系统，维护对话历史和上下文。
    """
    
    def __init__(self,
                 llm: Union[BaseLLM, BaseChatModel],
                 retriever,
                 prompt_manager: Optional[PromptManager] = None,
                 context_manager: Optional[ContextManager] = None,
                 max_context_length: int = 4000,
                 max_history_length: int = 5,
                 include_sources: bool = True,
                 return_source_documents: bool = True):
        """
        初始化对话式RAG链
        
        Args:
            llm: 语言模型
            retriever: 检索器
            prompt_manager: 提示词管理器
            context_manager: 上下文管理器
            max_context_length: 最大上下文长度
            max_history_length: 最大历史轮数
            include_sources: 是否包含来源
            return_source_documents: 是否返回源文档
        """
        super().__init__(
            llm=llm,
            retriever=retriever,
            prompt_manager=prompt_manager,
            context_manager=context_manager,
            max_context_length=max_context_length,
            include_sources=include_sources,
            return_source_documents=return_source_documents
        )
        
        self.max_history_length = max_history_length
        self.conversation_history: List[Tuple[str, str]] = []  # (question, answer)
        self.session_id = None
        
        logger.info("初始化对话式RAG链")
    
    def run_conversation(self,
                        query: str,
                        session_id: Optional[str] = None,
                        k: int = 5,
                        **kwargs) -> RAGResponse:
        """
        执行对话式RAG查询
        
        Args:
            query: 用户查询
            session_id: 会话ID
            k: 检索文档数量
            **kwargs: 额外参数
            
        Returns:
            RAGResponse对象
        """
        # 管理会话
        if session_id != self.session_id:
            self.session_id = session_id
            self.conversation_history = []
            logger.info(f"开始新对话会话: {session_id}")
        
        # 构建带历史的查询
        enhanced_query = self._build_contextual_query(query)
        
        # 执行RAG
        response = self.run(enhanced_query, k=k, **kwargs)
        
        # 更新对话历史
        self._update_conversation_history(query, response.answer)
        
        # 添加对话元数据
        if response.metadata:
            response.metadata.update({
                "session_id": session_id,
                "history_length": len(self.conversation_history),
                "enhanced_query": enhanced_query != query
            })
        
        return response
    
    def _build_contextual_query(self, current_query: str) -> str:
        """构建带历史上下文的查询"""
        if not self.conversation_history:
            return current_query
        
        try:
            # 构建上下文信息
            context_parts = []
            
            # 添加最近的对话历史
            recent_history = self.conversation_history[-2:]  # 最近2轮
            for i, (prev_q, prev_a) in enumerate(recent_history):
                context_parts.append(f"Q{i+1}: {prev_q}")
                context_parts.append(f"A{i+1}: {prev_a[:200]}")  # 限制长度
            
            context_str = "\n".join(context_parts)
            
            # 构建增强查询
            enhanced_query = f"""根据以下对话历史回答新问题：

对话历史:
{context_str}

新问题: {current_query}

请基于对话历史和新问题提供准确回答。"""
            
            return enhanced_query
        
        except Exception as e:
            logger.warning(f"构建上下文查询失败: {e}")
            return current_query
    
    def _update_conversation_history(self, question: str, answer: str):
        """更新对话历史"""
        self.conversation_history.append((question, answer))
        
        # 限制历史长度
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        self.session_id = None
        logger.info("对话历史已清空")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            "session_id": self.session_id,
            "history_length": len(self.conversation_history),
            "conversation_history": self.conversation_history,
            "stats": self.get_stats()
        }


# 便捷函数
def create_rag_chain(llm: Union[BaseLLM, BaseChatModel],
                    retriever,
                    conversational: bool = False,
                    **kwargs) -> Union[RAGChain, ConversationalRAGChain]:
    """
    创建RAG链的便捷函数
    
    Args:
        llm: 语言模型
        retriever: 检索器
        conversational: 是否创建对话式链
        **kwargs: 额外参数
        
    Returns:
        RAG链实例
    """
    if conversational:
        return ConversationalRAGChain(llm, retriever, **kwargs)
    else:
        return RAGChain(llm, retriever, **kwargs)