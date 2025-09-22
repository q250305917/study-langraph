"""
文档处理器主模块

整合文档加载、分割和元数据提取功能，提供完整的文档处理管道。
支持批量处理、增量更新和并发处理。
"""

import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime

# LangChain导入
from langchain.schema import Document

# 本地导入
from .loaders import (
    BaseDocumentLoader, 
    get_loader_for_file, 
    load_documents_from_directory,
    LoaderConfig
)
from .splitters import (
    BaseDocumentSplitter,
    get_splitter,
    SplitterConfig
)
from .metadata import (
    MetadataExtractor,
    DocumentMetadata,
    enrich_documents_metadata
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """文档处理配置"""
    
    # 加载器配置
    loader_config: Optional[LoaderConfig] = None
    
    # 分割器配置
    splitter_config: Optional[SplitterConfig] = None
    splitter_type: str = "recursive_character"
    
    # 元数据配置
    extract_metadata: bool = True
    generate_summary: bool = True
    extract_keywords: bool = True
    
    # 并发配置
    max_workers: int = 4
    batch_size: int = 10
    
    # 过滤配置
    min_content_length: int = 50
    max_content_length: int = 50000
    supported_formats: Optional[List[str]] = None
    
    # 输出配置
    save_processed: bool = False
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.loader_config is None:
            self.loader_config = LoaderConfig()
        
        if self.splitter_config is None:
            self.splitter_config = SplitterConfig()
        
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.txt', '.md', '.html']


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool = False
    documents: List[Document] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    source_file: str = ""
    chunk_count: int = 0
    
    def __post_init__(self):
        """初始化后处理"""
        if self.documents is None:
            self.documents = []
        if self.documents:
            self.chunk_count = len(self.documents)


class DocumentProcessor:
    """文档处理器主类
    
    提供完整的文档处理管道，包括：
    1. 文档加载（支持多种格式）
    2. 文档分割（智能分块）
    3. 元数据提取（内容分析）
    4. 批量处理（并发优化）
    """
    
    def __init__(self, 
                 config: Optional[ProcessingConfig] = None,
                 llm=None):
        """
        初始化文档处理器
        
        Args:
            config: 处理配置
            llm: 用于摘要生成的LLM
        """
        self.config = config or ProcessingConfig()
        self.llm = llm
        
        # 初始化组件
        self.splitter = get_splitter(
            self.config.splitter_type,
            self.config.splitter_config
        )
        
        self.metadata_extractor = MetadataExtractor(
            llm=self.llm,
            enable_summarization=self.config.generate_summary,
            enable_keyword_extraction=self.config.extract_keywords
        ) if self.config.extract_metadata else None
        
        # 处理统计
        self.stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "errors": []
        }
        
        logger.info(f"初始化文档处理器: {self.config.splitter_type} 分割器")
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            ProcessingResult对象
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        result = ProcessingResult(source_file=str(file_path))
        
        try:
            # 验证文件
            if not self._validate_file(file_path):
                result.error = f"文件验证失败: {file_path}"
                return result
            
            # 加载文档
            logger.info(f"开始处理文件: {file_path}")
            loader = get_loader_for_file(file_path, self.config.loader_config)
            documents = loader.load()
            
            if not documents:
                result.error = "文档加载失败，没有内容"
                return result
            
            # 分割文档
            if self.splitter:
                documents = self.splitter.split_documents(documents)
                logger.info(f"文档分割完成，共 {len(documents)} 个块")
            
            # 过滤文档
            documents = self._filter_documents(documents)
            
            # 提取元数据
            if self.metadata_extractor:
                documents = enrich_documents_metadata(documents, self.llm)
                logger.info("元数据提取完成")
            
            # 保存处理结果
            if self.config.save_processed and self.config.output_dir:
                self._save_processed_documents(documents, file_path)
            
            # 更新结果
            result.success = True
            result.documents = documents
            result.chunk_count = len(documents)
            
            logger.info(f"文件处理完成: {file_path}, {len(documents)} 个块")
            
        except Exception as e:
            error_msg = f"处理文件失败 {file_path}: {str(e)}"
            logger.error(error_msg)
            result.error = error_msg
            self.stats["errors"].append(error_msg)
        
        finally:
            # 计算处理时间
            end_time = datetime.now()
            result.processing_time = (end_time - start_time).total_seconds()
            
            # 更新统计
            self._update_stats(result)
        
        return result
    
    def process_directory(self, 
                         directory: Union[str, Path],
                         recursive: bool = True,
                         parallel: bool = True) -> List[ProcessingResult]:
        """
        批量处理目录中的文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归搜索子目录
            parallel: 是否并行处理
            
        Returns:
            ProcessingResult列表
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"目录不存在: {directory}")
        
        # 搜索文件
        files = self._find_files(directory, recursive)
        logger.info(f"找到 {len(files)} 个文件待处理")
        
        if not files:
            return []
        
        # 处理文件
        if parallel and len(files) > 1:
            return self._process_files_parallel(files)
        else:
            return self._process_files_sequential(files)
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        处理已有的Document列表
        
        Args:
            documents: Document列表
            
        Returns:
            处理后的Document列表
        """
        try:
            # 分割文档
            if self.splitter:
                documents = self.splitter.split_documents(documents)
                logger.info(f"文档分割完成，共 {len(documents)} 个块")
            
            # 过滤文档
            documents = self._filter_documents(documents)
            
            # 提取元数据
            if self.metadata_extractor:
                documents = enrich_documents_metadata(documents, self.llm)
                logger.info("元数据提取完成")
            
            return documents
            
        except Exception as e:
            logger.error(f"处理文档列表失败: {e}")
            raise
    
    def _validate_file(self, file_path: Path) -> bool:
        """验证文件是否可处理"""
        # 检查文件存在
        if not file_path.exists():
            return False
        
        # 检查文件大小
        if file_path.stat().st_size == 0:
            return False
        
        # 检查文件格式
        if file_path.suffix.lower() not in self.config.supported_formats:
            return False
        
        return True
    
    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        """过滤文档"""
        filtered = []
        
        for doc in documents:
            content = doc.page_content.strip()
            
            # 长度过滤
            if (len(content) < self.config.min_content_length or 
                len(content) > self.config.max_content_length):
                continue
            
            # 内容质量过滤
            if self._is_low_quality_content(content):
                continue
            
            filtered.append(doc)
        
        logger.info(f"文档过滤完成: {len(documents)} -> {len(filtered)}")
        return filtered
    
    def _is_low_quality_content(self, content: str) -> bool:
        """检查是否为低质量内容"""
        # 空内容或只有空白字符
        if not content.strip():
            return True
        
        # 重复字符过多
        if len(set(content)) < len(content) * 0.1:
            return True
        
        # 非文本内容（如二进制数据）
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            return True
        
        return False
    
    def _find_files(self, directory: Path, recursive: bool) -> List[Path]:
        """查找待处理的文件"""
        files = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.config.supported_formats):
                files.append(file_path)
        
        return sorted(files)
    
    def _process_files_sequential(self, files: List[Path]) -> List[ProcessingResult]:
        """顺序处理文件"""
        results = []
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"处理文件 {i}/{len(files)}: {file_path.name}")
            result = self.process_file(file_path)
            results.append(result)
        
        return results
    
    def _process_files_parallel(self, files: List[Path]) -> List[ProcessingResult]:
        """并行处理文件"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in files
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"并行处理完成: {file_path.name}")
                except Exception as e:
                    error_result = ProcessingResult(
                        source_file=str(file_path),
                        error=f"并行处理异常: {e}"
                    )
                    results.append(error_result)
                    logger.error(f"并行处理失败 {file_path}: {e}")
        
        return results
    
    def _save_processed_documents(self, documents: List[Document], source_file: Path):
        """保存处理后的文档"""
        if not self.config.output_dir:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        base_name = source_file.stem
        output_file = output_dir / f"{base_name}_processed.json"
        
        try:
            import json
            
            # 准备数据
            data = {
                "source_file": str(source_file),
                "processed_time": datetime.now().isoformat(),
                "chunk_count": len(documents),
                "documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
            }
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"处理结果已保存: {output_file}")
            
        except Exception as e:
            logger.error(f"保存处理结果失败: {e}")
    
    def _update_stats(self, result: ProcessingResult):
        """更新处理统计"""
        self.stats["total_files"] += 1
        self.stats["total_processing_time"] += result.processing_time
        
        if result.success:
            self.stats["successful_files"] += 1
            self.stats["total_chunks"] += result.chunk_count
        else:
            self.stats["failed_files"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        
        # 计算平均值
        if stats["total_files"] > 0:
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["total_files"]
            stats["success_rate"] = stats["successful_files"] / stats["total_files"]
        else:
            stats["avg_processing_time"] = 0.0
            stats["success_rate"] = 0.0
        
        if stats["successful_files"] > 0:
            stats["avg_chunks_per_file"] = stats["total_chunks"] / stats["successful_files"]
        else:
            stats["avg_chunks_per_file"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "errors": []
        }
    
    def configure_splitter(self, splitter_type: str, config: Optional[SplitterConfig] = None):
        """重新配置分割器"""
        self.config.splitter_type = splitter_type
        if config:
            self.config.splitter_config = config
        
        self.splitter = get_splitter(splitter_type, self.config.splitter_config)
        logger.info(f"分割器已更新: {splitter_type}")
    
    def configure_metadata_extractor(self, llm=None, **kwargs):
        """重新配置元数据提取器"""
        if llm:
            self.llm = llm
        
        self.metadata_extractor = MetadataExtractor(
            llm=self.llm,
            **kwargs
        )
        logger.info("元数据提取器已更新")


# 便捷函数
def process_single_file(file_path: Union[str, Path],
                       config: Optional[ProcessingConfig] = None,
                       llm=None) -> List[Document]:
    """
    处理单个文件的便捷函数
    
    Args:
        file_path: 文件路径
        config: 处理配置
        llm: LLM实例
        
    Returns:
        处理后的Document列表
    """
    processor = DocumentProcessor(config, llm)
    result = processor.process_file(file_path)
    
    if result.success:
        return result.documents
    else:
        raise Exception(result.error)


def process_directory_batch(directory: Union[str, Path],
                          config: Optional[ProcessingConfig] = None,
                          llm=None,
                          recursive: bool = True,
                          parallel: bool = True) -> List[Document]:
    """
    批量处理目录的便捷函数
    
    Args:
        directory: 目录路径
        config: 处理配置
        llm: LLM实例
        recursive: 是否递归搜索
        parallel: 是否并行处理
        
    Returns:
        所有处理后的Document列表
    """
    processor = DocumentProcessor(config, llm)
    results = processor.process_directory(directory, recursive, parallel)
    
    # 合并所有成功的结果
    all_documents = []
    for result in results:
        if result.success:
            all_documents.extend(result.documents)
    
    return all_documents


async def process_files_async(file_paths: List[Union[str, Path]],
                            config: Optional[ProcessingConfig] = None,
                            llm=None) -> List[ProcessingResult]:
    """
    异步处理文件列表
    
    Args:
        file_paths: 文件路径列表
        config: 处理配置
        llm: LLM实例
        
    Returns:
        ProcessingResult列表
    """
    processor = DocumentProcessor(config, llm)
    
    # 创建异步任务
    async def process_file_async(file_path):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, processor.process_file, file_path)
    
    # 并发执行
    tasks = [process_file_async(file_path) for file_path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_result = ProcessingResult(
                source_file=str(file_paths[i]),
                error=str(result)
            )
            processed_results.append(error_result)
        else:
            processed_results.append(result)
    
    return processed_results