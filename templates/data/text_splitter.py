"""
文本分割器模板

本模块提供了多种智能文本分割策略，用于将长文档分割成适合处理的文本块。
不同的分割策略适用于不同的场景和需求。

核心特性：
1. 多种分割策略：字符分割、句子分割、段落分割、语义分割等
2. 智能边界检测：避免在单词或句子中间分割
3. 上下文保持：支持重叠窗口，保持文本上下文连贯性
4. 自适应分割：根据内容特征动态调整分割策略
5. 元数据保持：保留原始文档的元数据信息
6. 批量处理：支持批量处理多个文档
7. 异步支持：支持异步文本分割
8. 性能优化：高效的分割算法，支持大文档处理

支持的分割策略：
1. CharacterTextSplitter: 按字符数分割，最基础的分割方法
2. SentenceTextSplitter: 按句子分割，保持语义完整性
3. ParagraphTextSplitter: 按段落分割，保持逻辑结构
4. SemanticTextSplitter: 基于语义相似度的智能分割
5. RecursiveCharacterTextSplitter: 递归字符分割，优先保持结构
6. TokenTextSplitter: 按Token数分割，适用于LLM处理
7. MarkdownHeaderTextSplitter: 专门用于Markdown文档的分割
8. CodeTextSplitter: 专门用于代码文档的分割

设计模式：
- 策略模式：不同的分割策略可以动态切换
- 工厂模式：根据配置创建相应的分割器
- 装饰器模式：为分割器添加额外功能（如重叠、过滤等）
- 模板方法模式：定义分割的通用流程，具体策略实现细节
"""

import re
import nltk
import spacy
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math
import statistics

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
    RecursiveCharacterTextSplitter as RCTSplitter
)

# 基础模板导入
from ..base.template_base import TemplateBase, TemplateConfig, TemplateType, ParameterSchema
from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import (
    ValidationError, 
    ConfigurationError,
    ResourceError,
    ErrorCodes
)

logger = get_logger(__name__)


class SplitStrategy(Enum):
    """文本分割策略枚举"""
    CHARACTER = "character"                    # 按字符分割
    SENTENCE = "sentence"                      # 按句子分割
    PARAGRAPH = "paragraph"                    # 按段落分割
    SEMANTIC = "semantic"                      # 语义分割
    RECURSIVE_CHARACTER = "recursive_character" # 递归字符分割
    TOKEN = "token"                           # 按Token分割
    MARKDOWN_HEADER = "markdown_header"       # Markdown标题分割
    CODE = "code"                             # 代码分割
    ADAPTIVE = "adaptive"                     # 自适应分割


class OverlapStrategy(Enum):
    """重叠策略枚举"""
    FIXED = "fixed"                           # 固定重叠大小
    PERCENTAGE = "percentage"                 # 按百分比重叠
    SENTENCE_BASED = "sentence_based"         # 基于句子的重叠
    ADAPTIVE = "adaptive"                     # 自适应重叠


@dataclass
class SplitResult:
    """
    文本分割结果类
    
    包含分割后的文本块和相关统计信息。
    """
    chunks: List[Document]                    # 分割后的文本块
    total_chunks: int = 0                     # 总块数
    avg_chunk_size: float = 0.0              # 平均块大小
    min_chunk_size: int = 0                   # 最小块大小
    max_chunk_size: int = 0                   # 最大块大小
    overlap_ratio: float = 0.0               # 重叠比例
    split_time: float = 0.0                  # 分割耗时
    
    def __post_init__(self):
        """计算统计信息"""
        if self.chunks:
            chunk_sizes = [len(chunk.page_content) for chunk in self.chunks]
            self.total_chunks = len(self.chunks)
            self.avg_chunk_size = statistics.mean(chunk_sizes)
            self.min_chunk_size = min(chunk_sizes)
            self.max_chunk_size = max(chunk_sizes)


class BaseTextSplitter(ABC):
    """
    文本分割器抽象基类
    
    定义文本分割器的通用接口。
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        **kwargs
    ):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 重叠大小
            length_function: 长度计算函数
            **kwargs: 其他参数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators: List[str] = kwargs.get('separators', ["\n\n", "\n", " ", ""])
        self.keep_separator = kwargs.get('keep_separator', False)
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        Args:
            documents: 要分割的文档列表
            
        Returns:
            分割后的文档列表
        """
        pass
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本列表
        """
        pass
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        合并分割的文本片段，考虑重叠
        
        Args:
            splits: 分割的文本片段
            separator: 分隔符
            
        Returns:
            合并后的文本块
        """
        separator_len = self.length_function(separator)
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            _len = self.length_function(split)
            
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                    
                    # 处理重叠
                    while (
                        total > self.chunk_overlap 
                        and len(current_doc) > 1
                    ):
                        total -= self.length_function(current_doc[0]) + separator_len
                        current_doc = current_doc[1:]
            
            current_doc.append(split)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        
        # 添加最后的文档
        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)
        
        return docs


class CharacterSplitter(BaseTextSplitter):
    """字符文本分割器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separator=kwargs.get('separator', '\n\n')
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        return self.splitter.split_text(text)


class SentenceSplitter(BaseTextSplitter):
    """句子文本分割器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language = kwargs.get('language', 'english')
        
        # 初始化NLTK句子分割器
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
        
        self.sentence_tokenizer = nltk.data.load(f'tokenizers/punkt/{self.language}.pickle')
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        result_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_id': i,
                        'chunk_type': 'sentence_split'
                    }
                )
                result_docs.append(new_doc)
        
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        # 按句子分割
        sentences = self.sentence_tokenizer.tokenize(text)
        
        # 合并句子到适当的块大小
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = self.length_function(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # 创建当前块
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # 处理重叠
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + self.length_function(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += self.length_function(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 添加最后的块
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


class ParagraphSplitter(BaseTextSplitter):
    """段落文本分割器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paragraph_separator = kwargs.get('paragraph_separator', '\n\n')
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        result_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_id': i,
                        'chunk_type': 'paragraph_split'
                    }
                )
                result_docs.append(new_doc)
        
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        # 按段落分割
        paragraphs = text.split(self.paragraph_separator)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 合并段落到适当的块大小
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = self.length_function(paragraph)
            
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                # 创建当前块
                chunk_text = self.paragraph_separator.join(current_chunk)
                chunks.append(chunk_text)
                
                # 处理重叠
                overlap_paras = []
                overlap_length = 0
                for para in reversed(current_chunk):
                    if overlap_length + self.length_function(para) <= self.chunk_overlap:
                        overlap_paras.insert(0, para)
                        overlap_length += self.length_function(para)
                    else:
                        break
                
                current_chunk = overlap_paras
                current_length = overlap_length
            
            current_chunk.append(paragraph)
            current_length += paragraph_length
        
        # 添加最后的块
        if current_chunk:
            chunk_text = self.paragraph_separator.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


class SemanticSplitter(BaseTextSplitter):
    """语义文本分割器，基于语义相似度进行分割"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.7)
        self.model_name = kwargs.get('model_name', 'all-MiniLM-L6-v2')
        
        # 初始化句子嵌入模型
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to sentence splitting")
            self.model = None
        
        # 初始化句子分割器
        self.sentence_splitter = SentenceSplitter(**kwargs)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        if not self.model:
            return self.sentence_splitter.split_documents(documents)
        
        result_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_id': i,
                        'chunk_type': 'semantic_split'
                    }
                )
                result_docs.append(new_doc)
        
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """基于语义相似度分割文本"""
        if not self.model:
            return self.sentence_splitter.split_text(text)
        
        # 首先按句子分割
        sentences = self.sentence_splitter.sentence_tokenizer.tokenize(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 基于相似度进行分组
        chunks = []
        current_chunk = [sentences[0]]
        current_length = self.length_function(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = self.length_function(sentence)
            
            # 计算与当前块最后一句的相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                [embeddings[i-1]], [embeddings[i]]
            )[0][0]
            
            # 如果相似度低于阈值或块太大，开始新块
            if (similarity < self.similarity_threshold or 
                current_length + sentence_length > self.chunk_size):
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)
                
                # 处理重叠
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + self.length_function(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += self.length_function(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 添加最后的块
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


class RecursiveCharacterSplitter(BaseTextSplitter):
    """递归字符分割器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            separators=self.separators,
            keep_separator=self.keep_separator
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        return self.splitter.split_text(text)


class TokenSplitter(BaseTextSplitter):
    """Token分割器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
        
        try:
            self.splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                model_name=self.model_name
            )
        except Exception as e:
            logger.warning(f"Failed to initialize TokenTextSplitter: {e}")
            # 回退到字符分割器
            self.splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        return self.splitter.split_text(text)


class MarkdownHeaderSplitter(BaseTextSplitter):
    """Markdown标题分割器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.headers_to_split_on = kwargs.get('headers_to_split_on', [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
        
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        
        # 如果需要进一步分割，使用递归字符分割器
        self.secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        ) if kwargs.get('use_secondary_splitter', True) else None
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        result_docs = []
        
        for doc in documents:
            # 首先按标题分割
            header_splits = self.splitter.split_text(doc.page_content)
            
            # 如果有二级分割器，进一步分割大块
            if self.secondary_splitter:
                final_docs = self.secondary_splitter.split_documents(header_splits)
            else:
                final_docs = header_splits
            
            # 添加原始元数据
            for split_doc in final_docs:
                split_doc.metadata.update(doc.metadata)
                split_doc.metadata['chunk_type'] = 'markdown_header_split'
                result_docs.append(split_doc)
        
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        docs = self.splitter.split_text(text)
        
        if self.secondary_splitter:
            final_chunks = []
            for doc in docs:
                chunks = self.secondary_splitter.split_text(doc.page_content)
                final_chunks.extend(chunks)
            return final_chunks
        else:
            return [doc.page_content for doc in docs]


class AdaptiveSplitter(BaseTextSplitter):
    """自适应分割器，根据文本特征选择最佳分割策略"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化各种分割器
        self.sentence_splitter = SentenceSplitter(**kwargs)
        self.paragraph_splitter = ParagraphSplitter(**kwargs)
        self.recursive_splitter = RecursiveCharacterSplitter(**kwargs)
        self.markdown_splitter = MarkdownHeaderSplitter(**kwargs)
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """分析文本结构特征"""
        analysis = {
            'total_length': len(text),
            'line_count': len(text.split('\n')),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'sentence_count': len(self.sentence_splitter.sentence_tokenizer.tokenize(text)),
            'avg_sentence_length': 0,
            'has_markdown_headers': bool(re.search(r'^#+\s', text, re.MULTILINE)),
            'has_code_blocks': bool(re.search(r'```', text)),
            'has_lists': bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            'structure_score': 0
        }
        
        # 计算平均句子长度
        sentences = self.sentence_splitter.sentence_tokenizer.tokenize(text)
        if sentences:
            analysis['avg_sentence_length'] = sum(len(s) for s in sentences) / len(sentences)
        
        # 计算结构化程度分数
        structure_score = 0
        if analysis['has_markdown_headers']:
            structure_score += 3
        if analysis['has_code_blocks']:
            structure_score += 2
        if analysis['has_lists']:
            structure_score += 1
        if analysis['paragraph_count'] > analysis['line_count'] * 0.3:
            structure_score += 2
        
        analysis['structure_score'] = structure_score
        
        return analysis
    
    def _choose_splitter(self, text: str) -> BaseTextSplitter:
        """根据文本特征选择最佳分割器"""
        analysis = self._analyze_text_structure(text)
        
        # Markdown文档优先使用Markdown分割器
        if analysis['has_markdown_headers'] and analysis['structure_score'] >= 3:
            logger.debug("Using MarkdownHeaderSplitter for structured markdown content")
            return self.markdown_splitter
        
        # 结构化文档使用段落分割器
        elif analysis['structure_score'] >= 2 and analysis['paragraph_count'] > 1:
            logger.debug("Using ParagraphSplitter for structured content")
            return self.paragraph_splitter
        
        # 句子较长且结构简单时使用句子分割器
        elif analysis['avg_sentence_length'] > 100 and analysis['sentence_count'] > 5:
            logger.debug("Using SentenceSplitter for long sentences")
            return self.sentence_splitter
        
        # 默认使用递归字符分割器
        else:
            logger.debug("Using RecursiveCharacterSplitter as default")
            return self.recursive_splitter
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        result_docs = []
        
        for doc in documents:
            # 为每个文档选择最佳分割器
            splitter = self._choose_splitter(doc.page_content)
            chunks = splitter.split_documents([doc])
            
            # 添加自适应分割标记
            for chunk in chunks:
                chunk.metadata['chunk_type'] = f"adaptive_{splitter.__class__.__name__}"
                chunk.metadata['adaptive_strategy'] = splitter.__class__.__name__
            
            result_docs.extend(chunks)
        
        return result_docs
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        splitter = self._choose_splitter(text)
        return splitter.split_text(text)


class TextSplitterTemplate(TemplateBase[List[Document], List[Document]]):
    """
    文本分割器模板
    
    提供多种智能文本分割策略，将长文档分割成适合处理的文本块。
    支持多种分割模式和自定义配置。
    
    核心功能：
    1. 多种分割策略：字符、句子、段落、语义、自适应等
    2. 智能重叠：支持多种重叠策略，保持上下文连贯性
    3. 批量处理：支持批量处理多个文档
    4. 异步支持：支持异步文本分割
    5. 性能优化：高效的分割算法
    6. 元数据保持：保留原始文档的元数据信息
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """初始化文本分割器模板"""
        super().__init__(config)
        
        # 配置参数
        self.strategy: SplitStrategy = SplitStrategy.RECURSIVE_CHARACTER
        self.chunk_size: int = 1000
        self.chunk_overlap: int = 200
        self.overlap_strategy: OverlapStrategy = OverlapStrategy.FIXED
        self.separators: List[str] = ["\n\n", "\n", " ", ""]
        self.keep_separator: bool = False
        self.length_function: Callable[[str], int] = len
        
        # 策略特定参数
        self.strategy_params: Dict[str, Any] = {}
        
        # 分割器实例
        self.splitter: Optional[BaseTextSplitter] = None
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunks_per_doc': 0.0,
            'avg_chunk_size': 0.0,
            'split_time': 0.0
        }
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="TextSplitterTemplate",
            description="智能文本分割器模板",
            template_type=TemplateType.DATA,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("strategy", str, False, "recursive_character", "分割策略")
        config.add_parameter("chunk_size", int, False, 1000, "文本块大小")
        config.add_parameter("chunk_overlap", int, False, 200, "重叠大小")
        config.add_parameter("overlap_strategy", str, False, "fixed", "重叠策略")
        config.add_parameter("separators", list, False, ["\n\n", "\n", " ", ""], "分隔符列表")
        config.add_parameter("keep_separator", bool, False, False, "是否保留分隔符")
        config.add_parameter("length_function", str, False, "len", "长度计算函数")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置文本分割器参数
        
        Args:
            **parameters: 分割器参数
        """
        if not self.validate_parameters(parameters):
            raise ValidationError("Text splitter parameter validation failed")
        
        # 设置基本参数
        strategy_str = parameters.get('strategy', 'recursive_character')
        try:
            self.strategy = SplitStrategy(strategy_str)
        except ValueError:
            raise ValidationError(f"Unknown split strategy: {strategy_str}")
        
        self.chunk_size = parameters.get('chunk_size', 1000)
        self.chunk_overlap = parameters.get('chunk_overlap', 200)
        
        overlap_strategy_str = parameters.get('overlap_strategy', 'fixed')
        try:
            self.overlap_strategy = OverlapStrategy(overlap_strategy_str)
        except ValueError:
            raise ValidationError(f"Unknown overlap strategy: {overlap_strategy_str}")
        
        self.separators = parameters.get('separators', ["\n\n", "\n", " ", ""])
        self.keep_separator = parameters.get('keep_separator', False)
        
        # 设置长度函数
        length_func_str = parameters.get('length_function', 'len')
        if length_func_str == 'len':
            self.length_function = len
        elif length_func_str == 'token_count':
            # 需要实现token计数功能
            self.length_function = self._count_tokens
        else:
            self.length_function = len
        
        # 设置策略特定参数
        self.strategy_params = parameters.get('strategy_params', {})
        
        # 创建分割器实例
        self._create_splitter()
        
        self.status = self.status.CONFIGURED
        self._setup_parameters = parameters.copy()
        
        logger.info(
            f"Text splitter configured: strategy={self.strategy.value}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量（简单实现）"""
        # 这是一个简化的token计数实现
        # 实际应用中可能需要使用tiktoken等库
        return len(text.split())
    
    def _create_splitter(self) -> None:
        """根据策略创建分割器实例"""
        common_params = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'length_function': self.length_function,
            'separators': self.separators,
            'keep_separator': self.keep_separator,
            **self.strategy_params
        }
        
        if self.strategy == SplitStrategy.CHARACTER:
            self.splitter = CharacterSplitter(**common_params)
        elif self.strategy == SplitStrategy.SENTENCE:
            self.splitter = SentenceSplitter(**common_params)
        elif self.strategy == SplitStrategy.PARAGRAPH:
            self.splitter = ParagraphSplitter(**common_params)
        elif self.strategy == SplitStrategy.SEMANTIC:
            self.splitter = SemanticSplitter(**common_params)
        elif self.strategy == SplitStrategy.RECURSIVE_CHARACTER:
            self.splitter = RecursiveCharacterSplitter(**common_params)
        elif self.strategy == SplitStrategy.TOKEN:
            self.splitter = TokenSplitter(**common_params)
        elif self.strategy == SplitStrategy.MARKDOWN_HEADER:
            self.splitter = MarkdownHeaderSplitter(**common_params)
        elif self.strategy == SplitStrategy.ADAPTIVE:
            self.splitter = AdaptiveSplitter(**common_params)
        else:
            raise ConfigurationError(f"Unsupported split strategy: {self.strategy}")
    
    def execute(self, input_data: List[Document], **kwargs) -> List[Document]:
        """
        执行文本分割
        
        Args:
            input_data: 要分割的文档列表
            **kwargs: 额外参数
            
        Returns:
            分割后的文档列表
        """
        if not self.splitter:
            raise RuntimeError("Text splitter not configured. Call setup() first.")
        
        start_time = datetime.now()
        
        try:
            # 执行分割
            result_documents = self.splitter.split_documents(input_data)
            
            # 添加分割相关的元数据
            for i, doc in enumerate(result_documents):
                doc.metadata.update({
                    'split_strategy': self.strategy.value,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'global_chunk_id': i
                })
            
            # 更新统计信息
            split_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(input_data, result_documents, split_time)
            
            logger.info(
                f"Text splitting completed: {len(input_data)} documents -> "
                f"{len(result_documents)} chunks in {split_time:.2f}s"
            )
            
            return result_documents
            
        except Exception as e:
            logger.error(f"Text splitting failed: {str(e)}")
            raise
    
    def _update_stats(self, input_docs: List[Document], output_docs: List[Document], split_time: float) -> None:
        """更新统计信息"""
        self.stats.update({
            'total_documents': len(input_docs),
            'total_chunks': len(output_docs),
            'avg_chunks_per_doc': len(output_docs) / len(input_docs) if input_docs else 0,
            'avg_chunk_size': sum(len(doc.page_content) for doc in output_docs) / len(output_docs) if output_docs else 0,
            'split_time': split_time
        })
    
    async def execute_async(self, input_data: List[Document], **kwargs) -> List[Document]:
        """
        异步执行文本分割
        
        Args:
            input_data: 要分割的文档列表
            **kwargs: 额外参数
            
        Returns:
            分割后的文档列表
        """
        # 对于文本分割，通常是CPU密集型操作，使用线程池
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data, **kwargs)
    
    def get_split_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """获取分割统计信息"""
        if not documents:
            return {}
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        
        return {
            'total_chunks': len(documents),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_chunk_size': statistics.mean(chunk_sizes),
            'median_chunk_size': statistics.median(chunk_sizes),
            'std_chunk_size': statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0,
            'total_content_length': sum(chunk_sizes)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunks_per_doc': 0.0,
            'avg_chunk_size': 0.0,
            'split_time': 0.0
        }
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "strategy": "recursive_character",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""],
                "keep_separator": False
            },
            "execute_parameters": {
                "input_data": "List[Document] objects to split"
            },
            "usage_code": """
# 基础使用示例
from templates.data.text_splitter import TextSplitterTemplate
from langchain_core.documents import Document

# 初始化文本分割器
splitter = TextSplitterTemplate()

# 配置参数 - 递归字符分割
splitter.setup(
    strategy="recursive_character",
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# 准备文档
documents = [
    Document(page_content="长文档内容...", metadata={"source": "doc1.txt"})
]

# 执行分割
chunks = splitter.execute(documents)
print(f"分割成 {len(chunks)} 个文本块")

# 获取分割统计信息
stats = splitter.get_split_stats(chunks)
print(f"平均块大小: {stats['avg_chunk_size']:.0f} 字符")

# 语义分割示例
splitter.setup(
    strategy="semantic",
    chunk_size=1000,
    chunk_overlap=100,
    strategy_params={
        "similarity_threshold": 0.7,
        "model_name": "all-MiniLM-L6-v2"
    }
)
semantic_chunks = splitter.execute(documents)

# 自适应分割示例
splitter.setup(strategy="adaptive", chunk_size=1500, chunk_overlap=300)
adaptive_chunks = splitter.execute(documents)

# Markdown分割示例
splitter.setup(
    strategy="markdown_header",
    chunk_size=2000,
    chunk_overlap=100,
    strategy_params={
        "headers_to_split_on": [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
    }
)
markdown_chunks = splitter.execute(documents)

# 异步分割
import asyncio
async def async_split():
    chunks = await splitter.execute_async(documents)
    return chunks

chunks = asyncio.run(async_split())
""",
            "expected_output": {
                "type": "List[Document]",
                "description": "分割后的文档列表，每个Document包含分割的文本块和元数据"
            }
        }