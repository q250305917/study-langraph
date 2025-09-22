"""
文档分割器模块

提供多种文档分割策略，将长文档分割为合适大小的文本块，
便于向量化和检索。支持递归字符分割、语义分割等策略。
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

# LangChain导入
from langchain.schema import Document
from langchain.text_splitter import TextSplitter

# 可选依赖
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)


@dataclass
class SplitterConfig:
    """分割器配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    length_function: Optional[Callable] = None
    separators: Optional[List[str]] = None
    keep_separator: bool = True
    add_start_index: bool = False
    strip_whitespace: bool = True


class BaseDocumentSplitter(TextSplitter, ABC):
    """文档分割器基类"""
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        """
        初始化分割器
        
        Args:
            config: 分割器配置
        """
        self.config = config or SplitterConfig()
        
        # 初始化父类
        super().__init__(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=self.config.length_function or len,
            keep_separator=self.config.keep_separator,
            add_start_index=self.config.add_start_index,
            strip_whitespace=self.config.strip_whitespace
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档列表
        """
        split_docs = []
        
        for doc in documents:
            # 分割文档内容
            chunks = self.split_text(doc.page_content)
            
            # 为每个块创建新的Document
            for i, chunk in enumerate(chunks):
                # 复制原文档的元数据
                metadata = doc.metadata.copy()
                
                # 添加分割相关的元数据
                metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "splitter_type": self.__class__.__name__
                })
                
                split_docs.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
        
        return split_docs
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """分割文本（子类实现）"""
        pass


class RecursiveCharacterTextSplitter(BaseDocumentSplitter):
    """递归字符文本分割器
    
    按照优先级尝试不同的分隔符，递归分割文本直到达到目标大小。
    这是最常用的分割策略，能较好地保持文本的语义连贯性。
    """
    
    def __init__(self, 
                 config: Optional[SplitterConfig] = None,
                 separators: Optional[List[str]] = None):
        """
        初始化递归字符分割器
        
        Args:
            config: 分割器配置
            separators: 分隔符列表（优先级从高到低）
        """
        super().__init__(config)
        
        # 默认分隔符（按优先级排序）
        self.separators = separators or [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            " ",     # 空格分隔
            ""       # 字符级分隔
        ]
    
    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本实现"""
        final_chunks = []
        
        # 选择当前分隔符
        separator = separators[-1] if separators else ""
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # 使用当前分隔符分割
        splits = self._split_by_separator(text, separator)
        
        # 合并小块并递归处理大块
        good_splits = []
        for split in splits:
            if self._length_function(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                
                # 递归处理过大的块
                if not new_separators:
                    final_chunks.append(split)
                else:
                    other_info = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(other_info)
        
        # 处理剩余的小块
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        
        return final_chunks
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """使用指定分隔符分割文本"""
        if separator:
            if self._keep_separator:
                # 保留分隔符的分割
                parts = text.split(separator)
                splits = []
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        splits.append(part + separator)
                    else:
                        splits.append(part)
                return [s for s in splits if s]
            else:
                return [s for s in text.split(separator) if s]
        else:
            return list(text)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并分割的文本块"""
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            _len = self._length_function(split)
            
            if total + _len + (len(current_doc) * len(separator) if separator else 0) > self._chunk_size:
                if current_doc:
                    doc = separator.join(current_doc) if separator else "".join(current_doc)
                    if doc:
                        docs.append(doc)
                    
                    # 处理重叠
                    while (total > self._chunk_overlap and 
                           len(current_doc) > 0 and 
                           total + _len + (len(current_doc) * len(separator) if separator else 0) > self._chunk_size):
                        total -= self._length_function(current_doc[0]) + (len(separator) if separator else 0)
                        current_doc = current_doc[1:]
            
            current_doc.append(split)
            total += _len + (len(separator) if separator else 0)
        
        # 添加最后一个文档
        if current_doc:
            doc = separator.join(current_doc) if separator else "".join(current_doc)
            if doc:
                docs.append(doc)
        
        return docs


class CharacterTextSplitter(BaseDocumentSplitter):
    """字符文本分割器
    
    使用单一分隔符进行简单分割，适用于格式化文本。
    """
    
    def __init__(self,
                 config: Optional[SplitterConfig] = None,
                 separator: str = "\n\n"):
        """
        初始化字符分割器
        
        Args:
            config: 分割器配置
            separator: 分隔符
        """
        super().__init__(config)
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """使用单一分隔符分割文本"""
        # 首先用分隔符分割
        splits = text.split(self.separator)
        
        # 过滤空字符串
        splits = [s for s in splits if s.strip()]
        
        # 合并小块
        return self._merge_splits(splits)
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """合并文本块"""
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            _len = self._length_function(split)
            
            if total + _len + len(current_doc) * len(self.separator) > self._chunk_size:
                if current_doc:
                    doc = self.separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    
                    # 处理重叠
                    while (total > self._chunk_overlap and 
                           len(current_doc) > 0 and 
                           total + _len + len(current_doc) * len(self.separator) > self._chunk_size):
                        total -= self._length_function(current_doc[0]) + len(self.separator)
                        current_doc = current_doc[1:]
            
            current_doc.append(split)
            total += _len + len(self.separator)
        
        # 添加最后一个文档
        if current_doc:
            doc = self.separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
        
        return docs


class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Markdown文本分割器
    
    专门用于Markdown格式文档的分割，保持Markdown结构的完整性。
    """
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        """初始化Markdown分割器"""
        # Markdown特定的分隔符
        markdown_separators = [
            "\n#{1,6} ",  # 标题
            "```\n",      # 代码块
            "\n\n",       # 段落
            "\n",         # 行
            " ",          # 空格
            ""            # 字符
        ]
        
        super().__init__(config, markdown_separators)
    
    def split_text(self, text: str) -> List[str]:
        """分割Markdown文本"""
        # 预处理：标记代码块
        text = self._mark_code_blocks(text)
        
        # 使用递归分割
        chunks = super().split_text(text)
        
        # 后处理：恢复代码块标记
        return [self._restore_code_blocks(chunk) for chunk in chunks]
    
    def _mark_code_blocks(self, text: str) -> str:
        """标记代码块，防止被分割"""
        # 使用特殊标记替换代码块分隔符
        text = re.sub(r'```(.*?)```', r'__CODE_BLOCK_START__\1__CODE_BLOCK_END__', text, flags=re.DOTALL)
        return text
    
    def _restore_code_blocks(self, text: str) -> str:
        """恢复代码块标记"""
        text = text.replace('__CODE_BLOCK_START__', '```')
        text = text.replace('__CODE_BLOCK_END__', '```')
        return text


class SemanticTextSplitter(BaseDocumentSplitter):
    """语义文本分割器
    
    基于句子嵌入的语义相似度进行文本分割，
    确保相关内容被分在同一个块中。
    """
    
    def __init__(self,
                 config: Optional[SplitterConfig] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7):
        """
        初始化语义分割器
        
        Args:
            config: 分割器配置
            model_name: 句子嵌入模型名称
            similarity_threshold: 语义相似度阈值
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("需要安装sentence-transformers: pip install sentence-transformers")
        
        super().__init__(config)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        
        # 加载模型
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"加载语义分割模型: {model_name}")
        except Exception as e:
            logger.error(f"加载语义分割模型失败: {e}")
            raise
    
    def split_text(self, text: str) -> List[str]:
        """基于语义相似度分割文本"""
        # 首先按句子分割
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # 计算句子嵌入
        embeddings = self._compute_embeddings(sentences)
        
        # 基于语义相似度分组
        groups = self._group_by_similarity(sentences, embeddings)
        
        # 合并成最终的文本块
        chunks = self._merge_groups(groups)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 使用spaCy进行更好的句子分割
        if HAS_SPACY:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except:
                pass
        
        # 备用的简单句子分割
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """计算句子嵌入"""
        try:
            embeddings = self.model.encode(sentences)
            return embeddings
        except Exception as e:
            logger.error(f"计算嵌入失败: {e}")
            raise
    
    def _group_by_similarity(self, sentences: List[str], embeddings: np.ndarray) -> List[List[str]]:
        """基于语义相似度分组句子"""
        groups = []
        current_group = [sentences[0]]
        current_embeddings = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            # 计算与当前组的平均相似度
            current_avg = np.mean(current_embeddings, axis=0)
            similarity = cosine_similarity([embeddings[i]], [current_avg])[0][0]
            
            # 检查是否应该开始新组
            if similarity < self.similarity_threshold or len(' '.join(current_group)) > self._chunk_size:
                groups.append(current_group)
                current_group = [sentences[i]]
                current_embeddings = [embeddings[i]]
            else:
                current_group.append(sentences[i])
                current_embeddings.append(embeddings[i])
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _merge_groups(self, groups: List[List[str]]) -> List[str]:
        """将句子组合并成文本块"""
        chunks = []
        
        for group in groups:
            chunk = ' '.join(group)
            
            # 如果块太大，进一步分割
            if len(chunk) > self._chunk_size:
                # 使用简单的递归分割器作为备用
                backup_splitter = RecursiveCharacterTextSplitter(self.config)
                sub_chunks = backup_splitter.split_text(chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk)
        
        return chunks


class TokenTextSplitter(BaseDocumentSplitter):
    """基于Token的文本分割器
    
    使用tokenizer计算文本长度，确保分割后的块
    不超过模型的token限制。
    """
    
    def __init__(self,
                 config: Optional[SplitterConfig] = None,
                 encoding_name: str = "cl100k_base",
                 model_name: Optional[str] = None):
        """
        初始化Token分割器
        
        Args:
            config: 分割器配置
            encoding_name: tiktoken编码名称
            model_name: 模型名称（如果提供，自动选择编码）
        """
        if not HAS_TIKTOKEN:
            raise ImportError("需要安装tiktoken: pip install tiktoken")
        
        # 设置token长度函数
        if config is None:
            config = SplitterConfig()
        
        try:
            if model_name:
                self.encoding = tiktoken.encoding_for_model(model_name)
            else:
                self.encoding = tiktoken.get_encoding(encoding_name)
            
            # 使用token计数作为长度函数
            config.length_function = self._token_length
            
        except Exception as e:
            logger.error(f"初始化tiktoken失败: {e}")
            # 备用：使用字符长度
            config.length_function = len
        
        super().__init__(config)
    
    def _token_length(self, text: str) -> int:
        """计算文本的token长度"""
        try:
            return len(self.encoding.encode(text))
        except:
            # 备用估算：平均4个字符为1个token
            return len(text) // 4
    
    def split_text(self, text: str) -> List[str]:
        """基于token长度分割文本"""
        # 使用递归字符分割器，但基于token长度
        recursive_splitter = RecursiveCharacterTextSplitter(self.config)
        return recursive_splitter.split_text(text)


# 工厂函数
def get_splitter(splitter_type: str,
                config: Optional[SplitterConfig] = None,
                **kwargs) -> BaseDocumentSplitter:
    """
    根据类型创建文档分割器
    
    Args:
        splitter_type: 分割器类型
        config: 分割器配置
        **kwargs: 额外参数
        
    Returns:
        文档分割器实例
    """
    splitter_map = {
        "recursive_character": RecursiveCharacterTextSplitter,
        "character": CharacterTextSplitter,
        "markdown": MarkdownTextSplitter,
        "semantic": SemanticTextSplitter,
        "token": TokenTextSplitter
    }
    
    if splitter_type not in splitter_map:
        raise ValueError(f"不支持的分割器类型: {splitter_type}")
    
    splitter_class = splitter_map[splitter_type]
    return splitter_class(config, **kwargs)


# 便捷函数
def split_documents(documents: List[Document],
                   splitter_type: str = "recursive_character",
                   config: Optional[SplitterConfig] = None,
                   **kwargs) -> List[Document]:
    """
    分割文档列表的便捷函数
    
    Args:
        documents: 文档列表
        splitter_type: 分割器类型
        config: 分割器配置
        **kwargs: 额外参数
        
    Returns:
        分割后的文档列表
    """
    splitter = get_splitter(splitter_type, config, **kwargs)
    return splitter.split_documents(documents)


def split_text(text: str,
              splitter_type: str = "recursive_character",
              config: Optional[SplitterConfig] = None,
              **kwargs) -> List[str]:
    """
    分割文本的便捷函数
    
    Args:
        text: 要分割的文本
        splitter_type: 分割器类型
        config: 分割器配置
        **kwargs: 额外参数
        
    Returns:
        分割后的文本列表
    """
    splitter = get_splitter(splitter_type, config, **kwargs)
    return splitter.split_text(text)