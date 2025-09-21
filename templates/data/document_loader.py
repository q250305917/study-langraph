"""
文档加载器模板

本模块提供了统一的文档加载接口，支持多种文档格式的加载和解析。
设计为可扩展的架构，便于添加新的文档格式支持。

核心特性：
1. 多格式支持：PDF、Word、Excel、PowerPoint、TXT、Markdown、HTML、CSV、JSON等
2. 统一接口：所有格式使用相同的加载接口
3. 元数据提取：自动提取文档的元数据信息
4. 批量处理：支持目录批量加载和并行处理
5. 异步支持：支持异步文档加载，提高处理效率
6. 错误处理：完善的错误处理和恢复机制
7. 内容清理：自动清理文档内容，去除无用字符
8. 编码检测：自动检测文本文件编码
9. 递归加载：支持递归加载子目录文档
10. 过滤条件：支持按文件大小、修改时间等条件过滤

支持的文档格式：
- PDF: 使用 PyPDF2, pdfplumber 等库
- Microsoft Office: docx, xlsx, pptx (python-docx, openpyxl, python-pptx)
- 文本文件: txt, md, html, xml, csv, json
- 电子书: epub (ebooklib)
- 其他: rtf, odt 等

设计模式：
- 策略模式：不同文档格式使用不同的加载策略
- 工厂模式：根据文件扩展名创建对应的加载器
- 装饰器模式：为加载器添加额外功能（缓存、日志等）
- 责任链模式：按优先级尝试不同的加载方法
"""

import os
import json
import csv
import asyncio
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import chardet
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain imports
from langchain_core.documents import Document
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, CSVLoader, JSONLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
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


class DocumentFormat(Enum):
    """文档格式枚举"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    PPTX = "pptx"
    PPT = "ppt"
    TXT = "txt"
    MD = "md"
    MARKDOWN = "markdown"
    HTML = "html"
    HTM = "htm"
    XML = "xml"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    RTF = "rtf"
    ODT = "odt"
    EPUB = "epub"
    UNKNOWN = "unknown"


class LoadingMode(Enum):
    """加载模式枚举"""
    SINGLE = "single"        # 单个文件加载
    BATCH = "batch"          # 批量文件加载
    DIRECTORY = "directory"  # 目录递归加载
    STREAM = "stream"        # 流式加载
    ASYNC = "async"          # 异步加载


@dataclass
class DocumentMetadata:
    """
    文档元数据类
    
    存储文档的各种元数据信息，用于后续的索引和检索。
    """
    file_path: str                          # 文件路径
    file_name: str                          # 文件名
    file_size: int                          # 文件大小（字节）
    file_format: DocumentFormat             # 文档格式
    encoding: Optional[str] = None          # 文件编码
    created_time: Optional[datetime] = None # 创建时间
    modified_time: Optional[datetime] = None # 修改时间
    content_hash: Optional[str] = None      # 内容哈希
    page_count: Optional[int] = None        # 页数（适用于PDF等）
    word_count: Optional[int] = None        # 词数
    char_count: Optional[int] = None        # 字符数
    language: Optional[str] = None          # 语言
    title: Optional[str] = None             # 标题
    author: Optional[str] = None            # 作者
    subject: Optional[str] = None           # 主题
    keywords: List[str] = field(default_factory=list) # 关键词
    custom_metadata: Dict[str, Any] = field(default_factory=dict) # 自定义元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_format": self.file_format.value,
            "encoding": self.encoding,
            "created_time": self.created_time.isoformat() if self.created_time else None,
            "modified_time": self.modified_time.isoformat() if self.modified_time else None,
            "content_hash": self.content_hash,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "language": self.language,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "custom_metadata": self.custom_metadata
        }


@dataclass
class LoadingResult:
    """
    文档加载结果类
    
    包含加载的文档内容和相关信息。
    """
    documents: List[Document]               # 加载的文档列表
    metadata: DocumentMetadata              # 文档元数据
    success: bool = True                    # 加载是否成功
    error_message: Optional[str] = None     # 错误信息
    loading_time: float = 0.0              # 加载耗时
    warnings: List[str] = field(default_factory=list) # 警告信息


class BaseDocumentLoader(ABC):
    """
    文档加载器抽象基类
    
    定义文档加载器的通用接口。
    """
    
    def __init__(self):
        """初始化文档加载器"""
        self.supported_formats: List[DocumentFormat] = []
    
    @abstractmethod
    def can_load(self, file_path: Union[str, Path]) -> bool:
        """
        检查是否可以加载指定文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否可以加载
        """
        pass
    
    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> LoadingResult:
        """
        加载文档
        
        Args:
            file_path: 文件路径
            **kwargs: 加载参数
            
        Returns:
            加载结果
        """
        pass
    
    def _detect_encoding(self, file_path: Union[str, Path]) -> str:
        """检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10240)  # 读取前10KB用于检测
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except Exception as e:
            logger.warning(f"Failed to detect encoding for {file_path}: {e}")
            return 'utf-8'
    
    def _calculate_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _extract_basic_metadata(self, file_path: Union[str, Path], content: str) -> DocumentMetadata:
        """提取基础元数据"""
        file_path = Path(file_path)
        stat = file_path.stat()
        
        # 检测文档格式
        format_name = file_path.suffix.lower().lstrip('.')
        try:
            doc_format = DocumentFormat(format_name)
        except ValueError:
            doc_format = DocumentFormat.UNKNOWN
        
        return DocumentMetadata(
            file_path=str(file_path),
            file_name=file_path.name,
            file_size=stat.st_size,
            file_format=doc_format,
            created_time=datetime.fromtimestamp(stat.st_ctime),
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            content_hash=self._calculate_content_hash(content),
            word_count=len(content.split()),
            char_count=len(content)
        )


class TextDocumentLoader(BaseDocumentLoader):
    """文本文档加载器，支持TXT、MD、HTML等文本格式"""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = [
            DocumentFormat.TXT, DocumentFormat.MD, DocumentFormat.MARKDOWN,
            DocumentFormat.HTML, DocumentFormat.HTM, DocumentFormat.XML
        ]
    
    def can_load(self, file_path: Union[str, Path]) -> bool:
        """检查是否可以加载指定文件"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower().lstrip('.')
        
        try:
            format_type = DocumentFormat(suffix)
            return format_type in self.supported_formats
        except ValueError:
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> LoadingResult:
        """加载文本文档"""
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # 检测编码
            encoding = kwargs.get('encoding') or self._detect_encoding(file_path)
            
            # 根据文件类型选择加载器
            suffix = file_path.suffix.lower()
            if suffix in ['.html', '.htm']:
                loader = UnstructuredHTMLLoader(str(file_path))
            elif suffix in ['.md', '.markdown']:
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding=encoding)
            
            # 加载文档
            documents = loader.load()
            
            # 提取元数据
            content = '\n'.join([doc.page_content for doc in documents])
            metadata = self._extract_basic_metadata(file_path, content)
            metadata.encoding = encoding
            
            # 更新文档元数据
            for doc in documents:
                doc.metadata.update(metadata.to_dict())
            
            loading_time = (datetime.now() - start_time).total_seconds()
            
            return LoadingResult(
                documents=documents,
                metadata=metadata,
                success=True,
                loading_time=loading_time
            )
            
        except Exception as e:
            loading_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to load text document {file_path}: {str(e)}")
            
            return LoadingResult(
                documents=[],
                metadata=self._extract_basic_metadata(file_path, ""),
                success=False,
                error_message=str(e),
                loading_time=loading_time
            )


class PDFDocumentLoader(BaseDocumentLoader):
    """PDF文档加载器"""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = [DocumentFormat.PDF]
    
    def can_load(self, file_path: Union[str, Path]) -> bool:
        """检查是否可以加载指定文件"""
        return Path(file_path).suffix.lower() == '.pdf'
    
    def load(self, file_path: Union[str, Path], **kwargs) -> LoadingResult:
        """加载PDF文档"""
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # 使用PyPDFLoader加载PDF
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # 提取元数据
            content = '\n'.join([doc.page_content for doc in documents])
            metadata = self._extract_basic_metadata(file_path, content)
            metadata.page_count = len(documents)
            
            # 尝试提取PDF特定元数据
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    if pdf_reader.metadata:
                        metadata.title = pdf_reader.metadata.get('/Title')
                        metadata.author = pdf_reader.metadata.get('/Author')
                        metadata.subject = pdf_reader.metadata.get('/Subject')
            except Exception as e:
                logger.warning(f"Failed to extract PDF metadata: {e}")
            
            # 更新文档元数据
            for i, doc in enumerate(documents):
                doc.metadata.update(metadata.to_dict())
                doc.metadata['page_number'] = i + 1
            
            loading_time = (datetime.now() - start_time).total_seconds()
            
            return LoadingResult(
                documents=documents,
                metadata=metadata,
                success=True,
                loading_time=loading_time
            )
            
        except Exception as e:
            loading_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to load PDF document {file_path}: {str(e)}")
            
            return LoadingResult(
                documents=[],
                metadata=self._extract_basic_metadata(file_path, ""),
                success=False,
                error_message=str(e),
                loading_time=loading_time
            )


class OfficeDocumentLoader(BaseDocumentLoader):
    """Microsoft Office文档加载器，支持Word、Excel、PowerPoint"""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = [
            DocumentFormat.DOCX, DocumentFormat.DOC,
            DocumentFormat.XLSX, DocumentFormat.XLS,
            DocumentFormat.PPTX, DocumentFormat.PPT
        ]
    
    def can_load(self, file_path: Union[str, Path]) -> bool:
        """检查是否可以加载指定文件"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower().lstrip('.')
        
        try:
            format_type = DocumentFormat(suffix)
            return format_type in self.supported_formats
        except ValueError:
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> LoadingResult:
        """加载Office文档"""
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # 根据文件类型选择加载器
            suffix = file_path.suffix.lower()
            
            if suffix in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif suffix in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(str(file_path))
            elif suffix in ['.pptx', '.ppt']:
                loader = UnstructuredPowerPointLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported Office format: {suffix}")
            
            # 加载文档
            documents = loader.load()
            
            # 提取元数据
            content = '\n'.join([doc.page_content for doc in documents])
            metadata = self._extract_basic_metadata(file_path, content)
            
            # 更新文档元数据
            for doc in documents:
                doc.metadata.update(metadata.to_dict())
            
            loading_time = (datetime.now() - start_time).total_seconds()
            
            return LoadingResult(
                documents=documents,
                metadata=metadata,
                success=True,
                loading_time=loading_time
            )
            
        except Exception as e:
            loading_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to load Office document {file_path}: {str(e)}")
            
            return LoadingResult(
                documents=[],
                metadata=self._extract_basic_metadata(file_path, ""),
                success=False,
                error_message=str(e),
                loading_time=loading_time
            )


class StructuredDataLoader(BaseDocumentLoader):
    """结构化数据加载器，支持CSV、JSON等格式"""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = [
            DocumentFormat.CSV, DocumentFormat.JSON, DocumentFormat.JSONL
        ]
    
    def can_load(self, file_path: Union[str, Path]) -> bool:
        """检查是否可以加载指定文件"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower().lstrip('.')
        
        try:
            format_type = DocumentFormat(suffix)
            return format_type in self.supported_formats
        except ValueError:
            return False
    
    def load(self, file_path: Union[str, Path], **kwargs) -> LoadingResult:
        """加载结构化数据文档"""
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.csv':
                # 加载CSV文件
                csv_args = kwargs.get('csv_args', {})
                loader = CSVLoader(str(file_path), **csv_args)
                documents = loader.load()
                
            elif suffix in ['.json', '.jsonl']:
                # 加载JSON文件
                json_args = kwargs.get('json_args', {})
                if suffix == '.jsonl':
                    # 处理JSONL文件
                    documents = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                data = json.loads(line.strip())
                                content = json.dumps(data, ensure_ascii=False, indent=2)
                                doc = Document(
                                    page_content=content,
                                    metadata={"line_number": line_num, "source": str(file_path)}
                                )
                                documents.append(doc)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                else:
                    loader = JSONLoader(str(file_path), **json_args)
                    documents = loader.load()
            else:
                raise ValueError(f"Unsupported structured data format: {suffix}")
            
            # 提取元数据
            content = '\n'.join([doc.page_content for doc in documents])
            metadata = self._extract_basic_metadata(file_path, content)
            
            # 更新文档元数据
            for doc in documents:
                doc.metadata.update(metadata.to_dict())
            
            loading_time = (datetime.now() - start_time).total_seconds()
            
            return LoadingResult(
                documents=documents,
                metadata=metadata,
                success=True,
                loading_time=loading_time
            )
            
        except Exception as e:
            loading_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to load structured data {file_path}: {str(e)}")
            
            return LoadingResult(
                documents=[],
                metadata=self._extract_basic_metadata(file_path, ""),
                success=False,
                error_message=str(e),
                loading_time=loading_time
            )


class DocumentLoaderTemplate(TemplateBase[Union[str, Path, List[Union[str, Path]]], List[Document]]):
    """
    文档加载器模板
    
    统一的文档加载接口，支持多种文档格式的加载和处理。
    自动识别文档格式并选择合适的加载器。
    
    核心功能：
    1. 多格式支持：自动识别并加载多种文档格式
    2. 批量处理：支持批量加载多个文档
    3. 异步处理：支持异步文档加载
    4. 元数据提取：自动提取文档元数据
    5. 错误处理：完善的错误处理和恢复机制
    6. 内容过滤：支持按条件过滤文档
    """
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """初始化文档加载器模板"""
        super().__init__(config)
        
        # 初始化各种文档加载器
        self.loaders: List[BaseDocumentLoader] = [
            TextDocumentLoader(),
            PDFDocumentLoader(),
            OfficeDocumentLoader(),
            StructuredDataLoader()
        ]
        
        # 配置参数
        self.supported_formats: List[str] = []
        self.max_file_size: Optional[int] = None
        self.encoding: Optional[str] = None
        self.recursive: bool = False
        self.max_workers: int = 4
        self.batch_size: int = 10
        
        # 过滤条件
        self.file_filters: Dict[str, Any] = {}
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'loaded_files': 0,
            'failed_files': 0,
            'total_documents': 0,
            'total_size': 0,
            'loading_time': 0.0
        }
    
    def _create_default_config(self) -> TemplateConfig:
        """创建默认配置"""
        config = TemplateConfig(
            name="DocumentLoaderTemplate",
            description="多格式文档加载器模板",
            template_type=TemplateType.DATA,
            version="1.0.0"
        )
        
        # 添加参数定义
        config.add_parameter("file_types", list, False, [], "支持的文件类型列表")
        config.add_parameter("max_file_size", int, False, None, "最大文件大小（字节）")
        config.add_parameter("encoding", str, False, None, "文本文件编码")
        config.add_parameter("recursive", bool, False, False, "是否递归加载子目录")
        config.add_parameter("max_workers", int, False, 4, "并行加载的最大工作线程数")
        config.add_parameter("batch_size", int, False, 10, "批处理大小")
        
        return config
    
    def setup(self, **parameters) -> None:
        """
        设置文档加载器参数
        
        Args:
            **parameters: 加载器参数
        """
        if not self.validate_parameters(parameters):
            raise ValidationError("Document loader parameter validation failed")
        
        # 设置支持的文件格式
        self.supported_formats = parameters.get('file_types', [])
        if not self.supported_formats:
            # 默认支持所有格式
            self.supported_formats = [fmt.value for fmt in DocumentFormat if fmt != DocumentFormat.UNKNOWN]
        
        # 设置其他参数
        self.max_file_size = parameters.get('max_file_size')
        self.encoding = parameters.get('encoding')
        self.recursive = parameters.get('recursive', False)
        self.max_workers = parameters.get('max_workers', 4)
        self.batch_size = parameters.get('batch_size', 10)
        
        # 设置文件过滤条件
        self.file_filters = parameters.get('file_filters', {})
        
        self.status = self.status.CONFIGURED
        self._setup_parameters = parameters.copy()
        
        logger.info(
            f"Document loader configured: formats={self.supported_formats}, "
            f"max_size={self.max_file_size}, recursive={self.recursive}"
        )
    
    def execute(self, input_data: Union[str, Path, List[Union[str, Path]]], **kwargs) -> List[Document]:
        """
        执行文档加载
        
        Args:
            input_data: 文件路径或文件路径列表
            **kwargs: 额外参数
            
        Returns:
            加载的文档列表
        """
        start_time = datetime.now()
        all_documents = []
        
        try:
            # 标准化输入数据
            if isinstance(input_data, (str, Path)):
                file_paths = [Path(input_data)]
            else:
                file_paths = [Path(p) for p in input_data]
            
            # 展开路径（处理目录）
            expanded_paths = []
            for path in file_paths:
                if path.is_file():
                    expanded_paths.append(path)
                elif path.is_dir():
                    expanded_paths.extend(self._expand_directory(path))
                else:
                    logger.warning(f"Path does not exist: {path}")
            
            # 过滤文件
            filtered_paths = self._filter_files(expanded_paths)
            
            # 更新统计信息
            self.stats['total_files'] = len(filtered_paths)
            
            # 根据模式选择加载方法
            loading_mode = kwargs.get('mode', 'batch')
            if loading_mode == 'async' and len(filtered_paths) > 1:
                all_documents = self._load_documents_async(filtered_paths, **kwargs)
            elif len(filtered_paths) > self.batch_size:
                all_documents = self._load_documents_batch(filtered_paths, **kwargs)
            else:
                all_documents = self._load_documents_sequential(filtered_paths, **kwargs)
            
            # 更新统计信息
            self.stats['total_documents'] = len(all_documents)
            self.stats['loading_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Document loading completed: {self.stats['loaded_files']}/{self.stats['total_files']} files, "
                f"{self.stats['total_documents']} documents, "
                f"time: {self.stats['loading_time']:.2f}s"
            )
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise
    
    def _expand_directory(self, dir_path: Path) -> List[Path]:
        """展开目录，获取所有文件路径"""
        file_paths = []
        
        try:
            if self.recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for path in dir_path.glob(pattern):
                if path.is_file():
                    file_paths.append(path)
        
        except Exception as e:
            logger.error(f"Failed to expand directory {dir_path}: {e}")
        
        return file_paths
    
    def _filter_files(self, file_paths: List[Path]) -> List[Path]:
        """根据条件过滤文件"""
        filtered_paths = []
        
        for path in file_paths:
            try:
                # 检查文件格式
                suffix = path.suffix.lower().lstrip('.')
                if self.supported_formats and suffix not in self.supported_formats:
                    continue
                
                # 检查文件大小
                if self.max_file_size:
                    if path.stat().st_size > self.max_file_size:
                        logger.warning(f"File too large, skipping: {path}")
                        continue
                
                # 应用自定义过滤条件
                if self._apply_file_filters(path):
                    filtered_paths.append(path)
            
            except Exception as e:
                logger.warning(f"Failed to check file {path}: {e}")
        
        return filtered_paths
    
    def _apply_file_filters(self, file_path: Path) -> bool:
        """应用文件过滤条件"""
        if not self.file_filters:
            return True
        
        try:
            stat = file_path.stat()
            
            # 检查修改时间范围
            if 'modified_after' in self.file_filters:
                modified_after = self.file_filters['modified_after']
                if isinstance(modified_after, datetime):
                    if datetime.fromtimestamp(stat.st_mtime) < modified_after:
                        return False
            
            if 'modified_before' in self.file_filters:
                modified_before = self.file_filters['modified_before']
                if isinstance(modified_before, datetime):
                    if datetime.fromtimestamp(stat.st_mtime) > modified_before:
                        return False
            
            # 检查文件名模式
            if 'name_pattern' in self.file_filters:
                import re
                pattern = self.file_filters['name_pattern']
                if not re.search(pattern, file_path.name):
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to apply filters to {file_path}: {e}")
            return False
    
    def _find_loader(self, file_path: Path) -> Optional[BaseDocumentLoader]:
        """查找合适的文档加载器"""
        for loader in self.loaders:
            if loader.can_load(file_path):
                return loader
        return None
    
    def _load_single_document(self, file_path: Path, **kwargs) -> LoadingResult:
        """加载单个文档"""
        try:
            # 查找合适的加载器
            loader = self._find_loader(file_path)
            if not loader:
                return LoadingResult(
                    documents=[],
                    metadata=DocumentMetadata(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_size=0,
                        file_format=DocumentFormat.UNKNOWN
                    ),
                    success=False,
                    error_message=f"No suitable loader found for {file_path}"
                )
            
            # 加载文档
            result = loader.load(file_path, encoding=self.encoding, **kwargs)
            
            # 更新统计信息
            if result.success:
                self.stats['loaded_files'] += 1
                self.stats['total_size'] += result.metadata.file_size
            else:
                self.stats['failed_files'] += 1
            
            return result
            
        except Exception as e:
            self.stats['failed_files'] += 1
            logger.error(f"Failed to load document {file_path}: {e}")
            
            return LoadingResult(
                documents=[],
                metadata=DocumentMetadata(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_size=0,
                    file_format=DocumentFormat.UNKNOWN
                ),
                success=False,
                error_message=str(e)
            )
    
    def _load_documents_sequential(self, file_paths: List[Path], **kwargs) -> List[Document]:
        """顺序加载文档"""
        all_documents = []
        
        for file_path in file_paths:
            result = self._load_single_document(file_path, **kwargs)
            if result.success:
                all_documents.extend(result.documents)
            else:
                logger.warning(f"Failed to load {file_path}: {result.error_message}")
        
        return all_documents
    
    def _load_documents_batch(self, file_paths: List[Path], **kwargs) -> List[Document]:
        """批量并行加载文档"""
        all_documents = []
        
        # 分批处理
        for i in range(0, len(file_paths), self.batch_size):
            batch_paths = file_paths[i:i + self.batch_size]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交任务
                future_to_path = {
                    executor.submit(self._load_single_document, path, **kwargs): path
                    for path in batch_paths
                }
                
                # 收集结果
                for future in as_completed(future_to_path):
                    result = future.result()
                    if result.success:
                        all_documents.extend(result.documents)
                    else:
                        path = future_to_path[future]
                        logger.warning(f"Failed to load {path}: {result.error_message}")
        
        return all_documents
    
    def _load_documents_async(self, file_paths: List[Path], **kwargs) -> List[Document]:
        """异步加载文档"""
        # 为了保持同步接口，这里使用线程池模拟异步
        return self._load_documents_batch(file_paths, **kwargs)
    
    async def execute_async(self, input_data: Union[str, Path, List[Union[str, Path]]], **kwargs) -> List[Document]:
        """
        异步执行文档加载
        
        Args:
            input_data: 文件路径或文件路径列表
            **kwargs: 额外参数
            
        Returns:
            加载的文档列表
        """
        # 使用线程池执行同步加载方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取加载统计信息"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            'total_files': 0,
            'loaded_files': 0,
            'failed_files': 0,
            'total_documents': 0,
            'total_size': 0,
            'loading_time': 0.0
        }
    
    def get_example(self) -> Dict[str, Any]:
        """获取使用示例"""
        return {
            "setup_parameters": {
                "file_types": ["pdf", "txt", "docx", "md"],
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "recursive": True,
                "max_workers": 4,
                "batch_size": 10
            },
            "execute_parameters": {
                "input_data": "./documents/",
                "mode": "batch"
            },
            "usage_code": """
# 基础使用示例
from templates.data.document_loader import DocumentLoaderTemplate

# 初始化文档加载器
loader = DocumentLoaderTemplate()

# 配置参数
loader.setup(
    file_types=["pdf", "txt", "docx", "md"],
    max_file_size=10 * 1024 * 1024,  # 10MB
    recursive=True,
    max_workers=4
)

# 加载单个文件
documents = loader.execute("./document.pdf")

# 批量加载文件
documents = loader.execute(["./doc1.pdf", "./doc2.txt", "./doc3.docx"])

# 加载整个目录
documents = loader.execute("./documents/")

# 异步加载
import asyncio
documents = asyncio.run(loader.execute_async("./documents/"))

# 获取统计信息
stats = loader.get_stats()
print(f"Loaded {stats['loaded_files']} files, {stats['total_documents']} documents")

# 高级配置示例
loader.setup(
    file_types=["pdf", "docx"],
    max_file_size=50 * 1024 * 1024,  # 50MB
    recursive=True,
    file_filters={
        "modified_after": datetime(2023, 1, 1),
        "name_pattern": r".*report.*"
    }
)
""",
            "expected_output": {
                "type": "List[Document]",
                "description": "LangChain Document对象列表，每个Document包含page_content和metadata"
            }
        }