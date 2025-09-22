"""
元数据提取模块

提供文档元数据提取和管理功能，包括文件信息、内容摘要、
关键词提取等，增强文档的可搜索性和可理解性。
"""

import os
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import logging

# LangChain导入
from langchain.schema import Document

# 可选依赖
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

try:
    from langchain.llms.base import BaseLLM
    from langchain.chains.summarize import load_summarize_chain
    HAS_SUMMARIZE = True
except ImportError:
    HAS_SUMMARIZE = False

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """文档元数据结构"""
    
    # 基本文件信息
    source: str = ""
    filename: str = ""
    file_type: str = ""
    file_size: int = 0
    file_hash: str = ""
    mime_type: str = ""
    
    # 时间信息
    created_time: Optional[datetime] = None
    modified_time: Optional[datetime] = None
    processed_time: Optional[datetime] = None
    
    # 内容统计
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    line_count: int = 0
    
    # 内容分析
    language: str = "unknown"
    encoding: str = "utf-8"
    title: str = ""
    summary: str = ""
    keywords: List[str] = None
    topics: List[str] = None
    
    # 可读性指标
    reading_ease: Optional[float] = None
    reading_grade: Optional[float] = None
    
    # 自定义字段
    custom_fields: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.keywords is None:
            self.keywords = []
        if self.topics is None:
            self.topics = []
        if self.custom_fields is None:
            self.custom_fields = {}
        if self.processed_time is None:
            self.processed_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        
        # 处理datetime对象
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """从字典创建实例"""
        # 处理datetime字段
        datetime_fields = ['created_time', 'modified_time', 'processed_time']
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except:
                    data[field] = None
        
        return cls(**data)


class MetadataExtractor:
    """元数据提取器"""
    
    def __init__(self, 
                 llm: Optional[BaseLLM] = None,
                 enable_summarization: bool = True,
                 enable_keyword_extraction: bool = True,
                 enable_readability: bool = True):
        """
        初始化元数据提取器
        
        Args:
            llm: 用于摘要生成的LLM
            enable_summarization: 是否启用摘要生成
            enable_keyword_extraction: 是否启用关键词提取
            enable_readability: 是否启用可读性分析
        """
        self.llm = llm
        self.enable_summarization = enable_summarization and HAS_SUMMARIZE and llm is not None
        self.enable_keyword_extraction = enable_keyword_extraction and HAS_NLTK
        self.enable_readability = enable_readability and HAS_TEXTSTAT
        
        # 初始化NLTK资源
        if self.enable_keyword_extraction:
            self._init_nltk()
        
        # 初始化词干提取器
        if HAS_NLTK:
            self.stemmer = PorterStemmer()
    
    def _init_nltk(self):
        """初始化NLTK所需资源"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("下载NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("下载NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def extract_from_file(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """
        从文件路径提取元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            DocumentMetadata对象
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 提取文件基本信息
        metadata = self._extract_file_info(file_path)
        
        # 读取文件内容并分析
        try:
            # 检测编码
            encoding = self._detect_encoding(file_path)
            metadata.encoding = encoding
            
            # 读取内容
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 分析内容
            self._analyze_content(content, metadata)
            
        except Exception as e:
            logger.error(f"分析文件内容失败 {file_path}: {e}")
        
        return metadata
    
    def extract_from_document(self, document: Document) -> DocumentMetadata:
        """
        从Document对象提取元数据
        
        Args:
            document: LangChain Document对象
            
        Returns:
            DocumentMetadata对象
        """
        # 从现有元数据创建基础信息
        existing_meta = document.metadata or {}
        
        metadata = DocumentMetadata(
            source=existing_meta.get('source', ''),
            filename=existing_meta.get('filename', ''),
            file_type=existing_meta.get('file_type', ''),
            file_size=existing_meta.get('file_size', 0),
            encoding=existing_meta.get('encoding', 'utf-8')
        )
        
        # 分析文档内容
        self._analyze_content(document.page_content, metadata)
        
        return metadata
    
    def _extract_file_info(self, file_path: Path) -> DocumentMetadata:
        """提取文件基本信息"""
        stat = file_path.stat()
        
        # 计算文件哈希
        file_hash = self._calculate_file_hash(file_path)
        
        # 获取MIME类型
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        metadata = DocumentMetadata(
            source=str(file_path),
            filename=file_path.name,
            file_type=file_path.suffix.lower(),
            file_size=stat.st_size,
            file_hash=file_hash,
            mime_type=mime_type or "unknown",
            created_time=datetime.fromtimestamp(stat.st_ctime),
            modified_time=datetime.fromtimestamp(stat.st_mtime)
        )
        
        return metadata
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"计算文件哈希失败: {e}")
            return ""
    
    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        try:
            import chardet
            
            with open(file_path, 'rb') as f:
                raw_data = f.read(32768)  # 读取前32KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        
        except ImportError:
            return 'utf-8'
        except Exception as e:
            logger.warning(f"编码检测失败: {e}")
            return 'utf-8'
    
    def _analyze_content(self, content: str, metadata: DocumentMetadata):
        """分析文档内容"""
        # 基本统计
        metadata.char_count = len(content)
        metadata.word_count = len(content.split())
        metadata.line_count = len(content.split('\n'))
        metadata.paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # 句子统计
        if HAS_NLTK:
            try:
                sentences = sent_tokenize(content)
                metadata.sentence_count = len(sentences)
            except:
                # 备用方法
                metadata.sentence_count = len([s for s in content.split('.') if s.strip()])
        else:
            metadata.sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # 语言检测
        metadata.language = self._detect_language(content)
        
        # 提取标题
        metadata.title = self._extract_title(content)
        
        # 关键词提取
        if self.enable_keyword_extraction:
            metadata.keywords = self._extract_keywords(content)
        
        # 生成摘要
        if self.enable_summarization:
            metadata.summary = self._generate_summary(content)
        
        # 可读性分析
        if self.enable_readability:
            metadata.reading_ease = self._calculate_reading_ease(content)
            metadata.reading_grade = self._calculate_reading_grade(content)
    
    def _detect_language(self, content: str) -> str:
        """检测文档语言"""
        try:
            # 简单的语言检测：基于字符特征
            chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
            total_chars = len([c for c in content if c.isalpha()])
            
            if total_chars > 0 and chinese_chars / total_chars > 0.3:
                return "zh"
            else:
                return "en"
        
        except Exception:
            return "unknown"
    
    def _extract_title(self, content: str) -> str:
        """提取文档标题"""
        lines = content.split('\n')
        
        for line in lines[:10]:  # 检查前10行
            line = line.strip()
            if line:
                # Markdown标题
                if line.startswith('#'):
                    return line.lstrip('#').strip()
                
                # 如果是短行（可能是标题）
                if len(line) < 100 and not line.endswith('.'):
                    return line
        
        # 使用第一个非空行作为标题
        for line in lines:
            if line.strip():
                return line.strip()[:100]  # 限制长度
        
        return ""
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """提取关键词"""
        if not HAS_NLTK:
            return []
        
        try:
            # 获取停用词
            stop_words = set(stopwords.words('english'))
            
            # 分词
            words = word_tokenize(content.lower())
            
            # 过滤和处理
            filtered_words = []
            for word in words:
                if (word.isalpha() and 
                    len(word) > 2 and 
                    word not in stop_words):
                    # 词干提取
                    stemmed = self.stemmer.stem(word)
                    filtered_words.append(stemmed)
            
            # 计算词频
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 获取高频词
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:max_keywords]]
            
            return keywords
        
        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
            return []
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """生成文档摘要"""
        if not self.enable_summarization or not self.llm:
            # 简单摘要：返回前几句话
            sentences = content.split('. ')
            summary = '. '.join(sentences[:3])
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        
        try:
            # 使用LangChain的摘要链
            from langchain.text_splitter import CharacterTextSplitter
            from langchain.schema import Document
            
            # 如果内容太长，先分割
            if len(content) > 4000:
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_text(content)
                docs = [Document(page_content=t) for t in texts]
            else:
                docs = [Document(page_content=content)]
            
            # 生成摘要
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            summary = chain.run(docs)
            
            # 限制长度
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary.strip()
        
        except Exception as e:
            logger.warning(f"LLM摘要生成失败: {e}")
            # 备用简单摘要
            sentences = content.split('. ')
            summary = '. '.join(sentences[:3])
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
    
    def _calculate_reading_ease(self, content: str) -> Optional[float]:
        """计算阅读容易度"""
        if not HAS_TEXTSTAT:
            return None
        
        try:
            return flesch_reading_ease(content)
        except Exception as e:
            logger.warning(f"计算阅读容易度失败: {e}")
            return None
    
    def _calculate_reading_grade(self, content: str) -> Optional[float]:
        """计算阅读年级水平"""
        if not HAS_TEXTSTAT:
            return None
        
        try:
            return flesch_kincaid_grade(content)
        except Exception as e:
            logger.warning(f"计算阅读年级失败: {e}")
            return None


# 便捷函数
def extract_file_metadata(file_path: Union[str, Path],
                         llm: Optional[BaseLLM] = None,
                         **kwargs) -> DocumentMetadata:
    """
    从文件提取元数据的便捷函数
    
    Args:
        file_path: 文件路径
        llm: 用于摘要生成的LLM
        **kwargs: 额外参数
        
    Returns:
        DocumentMetadata对象
    """
    extractor = MetadataExtractor(llm=llm, **kwargs)
    return extractor.extract_from_file(file_path)


def extract_document_metadata(document: Document,
                            llm: Optional[BaseLLM] = None,
                            **kwargs) -> DocumentMetadata:
    """
    从Document提取元数据的便捷函数
    
    Args:
        document: LangChain Document对象
        llm: 用于摘要生成的LLM
        **kwargs: 额外参数
        
    Returns:
        DocumentMetadata对象
    """
    extractor = MetadataExtractor(llm=llm, **kwargs)
    return extractor.extract_from_document(document)


def generate_document_summary(content: str,
                            llm: BaseLLM,
                            max_length: int = 200) -> str:
    """
    生成文档摘要的便捷函数
    
    Args:
        content: 文档内容
        llm: LLM实例
        max_length: 摘要最大长度
        
    Returns:
        文档摘要
    """
    extractor = MetadataExtractor(llm=llm)
    return extractor._generate_summary(content, max_length)


def enrich_documents_metadata(documents: List[Document],
                            llm: Optional[BaseLLM] = None,
                            **kwargs) -> List[Document]:
    """
    为文档列表批量添加元数据
    
    Args:
        documents: Document列表
        llm: 用于摘要生成的LLM
        **kwargs: 额外参数
        
    Returns:
        添加了元数据的Document列表
    """
    extractor = MetadataExtractor(llm=llm, **kwargs)
    enriched_docs = []
    
    for doc in documents:
        try:
            # 提取元数据
            metadata = extractor.extract_from_document(doc)
            
            # 合并元数据
            combined_metadata = doc.metadata.copy() if doc.metadata else {}
            combined_metadata.update(metadata.to_dict())
            
            # 创建新的Document
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=combined_metadata
            )
            
            enriched_docs.append(enriched_doc)
            
        except Exception as e:
            logger.error(f"为文档添加元数据失败: {e}")
            # 保持原文档
            enriched_docs.append(doc)
    
    return enriched_docs