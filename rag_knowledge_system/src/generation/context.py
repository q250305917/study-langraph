"""
上下文管理模块

提供智能的上下文构建和压缩功能，确保在有限的上下文窗口内
提供最相关的信息给LLM。
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LangChain导入
from langchain.schema import Document

# 可选依赖
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """上下文配置"""
    max_length: int = 4000
    overlap_threshold: float = 0.8
    enable_compression: bool = True
    preserve_metadata: bool = True
    include_source_info: bool = True


class ContextManager:
    """上下文管理器"""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        初始化上下文管理器
        
        Args:
            config: 上下文配置
        """
        self.config = config or ContextConfig()
        
        # 初始化token计数器
        if HAS_TIKTOKEN:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except:
                self.encoding = None
                logger.warning("tiktoken初始化失败，使用简单字符计数")
        else:
            self.encoding = None
            logger.warning("tiktoken未安装，使用简单字符计数")
    
    def build_context(self,
                     documents: List[Document],
                     query: str,
                     max_length: Optional[int] = None) -> str:
        """
        构建上下文
        
        Args:
            documents: 文档列表
            query: 用户查询
            max_length: 最大长度
            
        Returns:
            构建的上下文字符串
        """
        max_length = max_length or self.config.max_length
        
        if not documents:
            return ""
        
        try:
            # 1. 去重
            unique_docs = self._deduplicate_documents(documents)
            
            # 2. 排序（按相关性）
            sorted_docs = self._sort_by_relevance(unique_docs, query)
            
            # 3. 构建上下文
            context_parts = []
            current_length = 0
            
            for i, doc in enumerate(sorted_docs):
                # 准备文档片段
                doc_text = self._prepare_document_text(doc, i + 1)
                doc_length = self._count_tokens(doc_text)
                
                # 检查是否超出长度限制
                if current_length + doc_length > max_length:
                    # 尝试压缩或截断
                    remaining_length = max_length - current_length
                    if remaining_length > 100:  # 至少保留100个token
                        compressed_text = self._compress_text(doc_text, remaining_length)
                        if compressed_text:
                            context_parts.append(compressed_text)
                    break
                
                context_parts.append(doc_text)
                current_length += doc_length
            
            # 4. 组合最终上下文
            final_context = "\n\n".join(context_parts)
            
            logger.debug(f"构建上下文完成，长度: {self._count_tokens(final_context)}")
            return final_context
        
        except Exception as e:
            logger.error(f"构建上下文失败: {e}")
            # 返回简单的文档连接
            return "\n\n".join([doc.page_content for doc in documents[:3]])
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """去除重复文档"""
        unique_docs = []
        seen_hashes = set()
        
        for doc in documents:
            # 使用内容哈希检测重复
            content_hash = hash(doc.page_content)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                logger.debug("发现重复文档，已跳过")
        
        return unique_docs
    
    def _sort_by_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """按相关性排序文档"""
        try:
            # 简单的关键词匹配评分
            query_words = set(query.lower().split())
            
            def relevance_score(doc: Document) -> float:
                doc_words = set(doc.page_content.lower().split())
                
                # 计算关键词重叠度
                if query_words:
                    overlap = len(query_words & doc_words) / len(query_words)
                else:
                    overlap = 0.0
                
                # 考虑文档长度（中等长度优先）
                doc_length = len(doc.page_content)
                length_score = min(doc_length / 1000, 1.0) if doc_length < 2000 else 2000 / doc_length
                
                return 0.7 * overlap + 0.3 * length_score
            
            return sorted(documents, key=relevance_score, reverse=True)
        
        except Exception as e:
            logger.warning(f"文档排序失败: {e}")
            return documents
    
    def _prepare_document_text(self, doc: Document, index: int) -> str:
        """准备文档文本"""
        parts = []
        
        # 添加文档索引
        parts.append(f"文档 {index}:")
        
        # 添加来源信息（如果启用）
        if self.config.include_source_info and doc.metadata:
            source = doc.metadata.get('source', '')
            title = doc.metadata.get('title', '')
            
            if title:
                parts.append(f"标题: {title}")
            if source:
                parts.append(f"来源: {source}")
        
        # 添加文档内容
        parts.append(doc.page_content)
        
        return "\n".join(parts)
    
    def _compress_text(self, text: str, max_length: int) -> str:
        """压缩文本到指定长度"""
        if not self.config.enable_compression:
            return text[:max_length * 4]  # 简单截断
        
        try:
            current_length = self._count_tokens(text)
            
            if current_length <= max_length:
                return text
            
            # 计算压缩比例
            compression_ratio = max_length / current_length
            
            # 按句子分割
            sentences = text.split('。')
            
            # 选择最重要的句子
            if len(sentences) > 1:
                # 保留前半部分和最后一部分
                keep_count = max(1, int(len(sentences) * compression_ratio))
                
                if keep_count < len(sentences):
                    # 选择前几句和最后一句
                    selected_sentences = (
                        sentences[:keep_count-1] + 
                        ['...'] + 
                        sentences[-1:]
                    )
                    compressed = '。'.join(selected_sentences)
                    
                    # 再次检查长度
                    if self._count_tokens(compressed) <= max_length:
                        return compressed
            
            # 如果句子级压缩不够，进行字符级截断
            char_limit = int(max_length * 4)  # 粗略估算
            return text[:char_limit] + "..."
        
        except Exception as e:
            logger.warning(f"文本压缩失败: {e}")
            return text[:max_length * 4]
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        
        # 备用计算：粗略估算
        return len(text.split()) + len(text) // 4


class ContextCompressor:
    """上下文压缩器
    
    专门用于压缩长文档和上下文的工具。
    """
    
    def __init__(self, llm=None):
        """
        初始化上下文压缩器
        
        Args:
            llm: 可选的LLM，用于智能压缩
        """
        self.llm = llm
    
    def compress_documents(self,
                          documents: List[Document],
                          query: str,
                          target_length: int = 2000) -> List[Document]:
        """
        压缩文档列表
        
        Args:
            documents: 文档列表
            query: 查询文本
            target_length: 目标总长度
            
        Returns:
            压缩后的文档列表
        """
        if not documents:
            return documents
        
        try:
            compressed_docs = []
            
            for doc in documents:
                if len(doc.page_content) > target_length // len(documents):
                    # 需要压缩
                    compressed_content = self._compress_single_document(
                        doc.page_content,
                        query,
                        target_length // len(documents)
                    )
                    
                    compressed_doc = Document(
                        page_content=compressed_content,
                        metadata={
                            **doc.metadata,
                            "compressed": True,
                            "original_length": len(doc.page_content)
                        }
                    )
                    compressed_docs.append(compressed_doc)
                else:
                    compressed_docs.append(doc)
            
            return compressed_docs
        
        except Exception as e:
            logger.error(f"文档压缩失败: {e}")
            return documents
    
    def _compress_single_document(self,
                                 content: str,
                                 query: str,
                                 target_length: int) -> str:
        """压缩单个文档"""
        if self.llm and len(content) > target_length * 2:
            # 使用LLM进行智能压缩
            return self._llm_compress(content, query, target_length)
        else:
            # 使用简单的文本压缩
            return self._simple_compress(content, target_length)
    
    def _llm_compress(self, content: str, query: str, target_length: int) -> str:
        """使用LLM进行智能压缩"""
        try:
            prompt = f"""请将以下文档内容压缩到大约{target_length}个字符，保留与查询"{query}"最相关的信息：

原文档：
{content}

压缩后的文档："""
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                compressed = response.content
            else:
                compressed = str(response)
            
            return compressed.strip()
        
        except Exception as e:
            logger.warning(f"LLM压缩失败: {e}")
            return self._simple_compress(content, target_length)
    
    def _simple_compress(self, content: str, target_length: int) -> str:
        """简单的文本压缩"""
        if len(content) <= target_length:
            return content
        
        # 按段落分割
        paragraphs = content.split('\n\n')
        
        if len(paragraphs) > 1:
            # 保留前几个段落
            compressed_paragraphs = []
            current_length = 0
            
            for para in paragraphs:
                if current_length + len(para) > target_length:
                    # 部分添加最后一个段落
                    remaining = target_length - current_length
                    if remaining > 50:
                        compressed_paragraphs.append(para[:remaining] + "...")
                    break
                
                compressed_paragraphs.append(para)
                current_length += len(para) + 2  # +2 for \n\n
            
            return '\n\n'.join(compressed_paragraphs)
        else:
            # 直接截断
            return content[:target_length] + "..."


# 便捷函数
def build_context(documents: List[Document],
                 query: str,
                 max_length: int = 4000) -> str:
    """
    构建上下文的便捷函数
    
    Args:
        documents: 文档列表
        query: 查询文本
        max_length: 最大长度
        
    Returns:
        构建的上下文字符串
    """
    manager = ContextManager()
    return manager.build_context(documents, query, max_length)