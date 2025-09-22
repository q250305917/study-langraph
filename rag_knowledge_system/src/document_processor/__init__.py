"""
文档处理模块

提供多格式文档加载、智能分割和元数据提取功能。
支持PDF、Word、文本、Markdown等格式的处理。
"""

from .processor import DocumentProcessor
from .loaders import (
    BaseDocumentLoader,
    PDFLoader,
    DocxLoader,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    load_document,
    get_loader_for_file
)
from .splitters import (
    BaseDocumentSplitter,
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    SemanticTextSplitter,
    get_splitter
)
from .metadata import (
    MetadataExtractor,
    DocumentMetadata,
    extract_file_metadata,
    generate_document_summary
)

__all__ = [
    # 核心处理器
    "DocumentProcessor",
    
    # 加载器
    "BaseDocumentLoader",
    "PDFLoader",
    "DocxLoader", 
    "TextLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "load_document",
    "get_loader_for_file",
    
    # 分割器
    "BaseDocumentSplitter",
    "RecursiveCharacterTextSplitter",
    "CharacterTextSplitter",
    "MarkdownTextSplitter",
    "SemanticTextSplitter",
    "get_splitter",
    
    # 元数据
    "MetadataExtractor",
    "DocumentMetadata",
    "extract_file_metadata",
    "generate_document_summary"
]