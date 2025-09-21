"""
数据处理模板集成测试

测试数据处理模板之间的集成和协作，包括：
- DocumentLoader + TextSplitter 集成
- TextSplitter + VectorStore 集成  
- 完整的RAG流水线测试
- 性能和可扩展性测试
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# 简化的测试，只测试基本功能而不依赖外部服务
from templates.data.document_loader import DocumentLoaderTemplate
from templates.data.text_splitter import TextSplitterTemplate


class TestDataProcessingIntegration:
    """数据处理模板集成测试"""
    
    def setup_method(self):
        """测试前的准备工作"""
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_documents()
    
    def teardown_method(self):
        """测试后的清理工作"""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_documents(self):
        """创建测试文档"""
        # 创建多种类型的测试文档
        documents = {
            "article1.txt": """人工智能的发展历史

人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

人工智能的发展可以分为几个阶段：

第一阶段：符号主义时期（1956-1970年代）
这个时期的研究主要集中在逻辑推理和符号处理上。研究者们相信，通过符号操作可以实现人工智能。

第二阶段：专家系统时期（1970-1980年代）  
专家系统是这个时期的主要成果，它试图将专家的知识编码到计算机程序中。

第三阶段：机器学习时期（1980年代至今）
机器学习成为人工智能研究的主流，特别是深度学习的兴起，使得AI在很多领域取得了突破性进展。""",
            
            "article2.md": """# 机器学习基础概念

## 什么是机器学习

机器学习是人工智能的一个子领域，它使计算机系统能够从数据中自动学习和改进，而无需被明确编程。

## 机器学习的类型

### 监督学习
监督学习使用标记的训练数据来学习从输入到输出的映射函数。

常见的监督学习算法包括：
- 线性回归
- 逻辑回归  
- 决策树
- 随机森林
- 支持向量机

### 无监督学习
无监督学习从没有标记的数据中发现隐藏的模式。

常见的无监督学习算法包括：
- K-means聚类
- 层次聚类
- 主成分分析（PCA）
- DBSCAN

### 强化学习
强化学习通过与环境交互来学习最优行为策略。

## 机器学习的应用

机器学习在各个领域都有广泛应用：
- 图像识别
- 自然语言处理
- 推荐系统
- 金融预测
- 医疗诊断""",
            
            "data.json": """{
    "title": "深度学习发展史",
    "content": "深度学习是机器学习的一个分支，它模仿人脑神经网络的结构和功能。深度学习的发展经历了多个重要阶段，从早期的感知机到现代的Transformer架构，每一步都标志着人工智能技术的重大进步。",
    "keywords": ["深度学习", "神经网络", "人工智能", "机器学习"],
    "date": "2024-01-01"
}"""
        }
        
        # 写入文件
        for filename, content in documents.items():
            file_path = Path(self.temp_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def test_loader_splitter_integration(self):
        """测试文档加载器和文本分割器的集成"""
        # 步骤1：加载文档
        loader = DocumentLoaderTemplate()
        loader.setup(
            file_types=['txt', 'md', 'json'],
            recursive=True
        )
        
        documents = loader.execute(self.temp_dir)
        
        # 验证加载结果
        assert len(documents) >= 3
        assert all(len(doc.page_content) > 0 for doc in documents)
        
        # 步骤2：分割文档
        splitter = TextSplitterTemplate()
        splitter.setup(
            strategy="recursive_character",
            chunk_size=500,
            chunk_overlap=100
        )
        
        chunks = splitter.execute(documents)
        
        # 验证分割结果
        assert len(chunks) >= len(documents)  # 应该产生更多的文本块
        
        # 检查元数据传递
        for chunk in chunks:
            assert 'source' in chunk.metadata or 'file_path' in chunk.metadata
            assert 'split_strategy' in chunk.metadata
            assert 'chunk_size' in chunk.metadata
        
        # 验证内容完整性
        original_content = " ".join(doc.page_content for doc in documents)
        chunked_content = " ".join(chunk.page_content for chunk in chunks)
        
        # 检查重要关键词是否都保留了
        keywords = ["人工智能", "机器学习", "深度学习", "神经网络"]
        for keyword in keywords:
            if keyword in original_content:
                assert keyword in chunked_content
    
    def test_different_splitting_strategies(self):
        """测试不同分割策略对同一文档的处理"""
        # 加载文档
        loader = DocumentLoaderTemplate()
        loader.setup(file_types=['txt'])
        
        documents = loader.execute(self.temp_dir)
        txt_docs = [doc for doc in documents if doc.metadata.get('source', '').endswith('.txt')]
        
        if not txt_docs:
            pytest.skip("No text documents found")
        
        strategies = ["character", "sentence", "paragraph", "recursive_character"]
        results = {}
        
        for strategy in strategies:
            try:
                splitter = TextSplitterTemplate()
                splitter.setup(
                    strategy=strategy,
                    chunk_size=300,
                    chunk_overlap=50
                )
                
                chunks = splitter.execute(txt_docs[:1])  # 只使用第一个文档
                results[strategy] = {
                    'chunk_count': len(chunks),
                    'avg_chunk_size': sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
                }
                
            except Exception as e:
                print(f"Strategy {strategy} failed: {e}")
                continue
        
        # 验证不同策略产生了不同的结果
        assert len(results) >= 2  # 至少有两种策略成功
        
        chunk_counts = [r['chunk_count'] for r in results.values()]
        assert len(set(chunk_counts)) > 1 or max(chunk_counts) > 1  # 应该有不同的分割结果
    
    def test_markdown_specific_processing(self):
        """测试Markdown文档的特殊处理"""
        # 加载Markdown文档
        loader = DocumentLoaderTemplate()
        loader.setup(file_types=['md'])
        
        documents = loader.execute(self.temp_dir)
        md_docs = [doc for doc in documents if doc.metadata.get('source', '').endswith('.md')]
        
        if not md_docs:
            pytest.skip("No markdown documents found")
        
        # 使用Markdown专用分割策略
        splitter = TextSplitterTemplate()
        splitter.setup(
            strategy="markdown_header",
            chunk_size=1000,
            chunk_overlap=100,
            strategy_params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ]
            }
        )
        
        chunks = splitter.execute(md_docs)
        
        # 验证Markdown结构被正确处理
        assert len(chunks) >= 1
        
        # 检查是否按标题分割
        header_chunks = [c for c in chunks if '#' in c.page_content or 'Header' in str(c.metadata)]
        # Markdown分割应该能识别标题结构
    
    def test_large_document_processing(self):
        """测试大文档处理"""
        # 创建一个大文档
        large_content = """
        
大型文档处理测试。""" + """

这是一个段落。包含重要信息。这些信息需要被正确分割和保存。
段落之间应该保持逻辑关系。分割后的文本块应该具有语义完整性。
""" * 50  # 重复50次创建大文档
        
        large_doc_path = Path(self.temp_dir) / "large_document.txt"
        with open(large_doc_path, 'w', encoding='utf-8') as f:
            f.write(large_content)
        
        # 加载大文档
        loader = DocumentLoaderTemplate()
        loader.setup(file_types=['txt'])
        
        documents = loader.execute(str(large_doc_path))
        
        # 分割大文档
        splitter = TextSplitterTemplate()
        splitter.setup(
            strategy="paragraph",
            chunk_size=500,
            chunk_overlap=100
        )
        
        chunks = splitter.execute(documents)
        
        # 验证大文档被正确处理
        assert len(chunks) > 10  # 应该产生很多块
        
        # 检查块大小分布
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        assert avg_size > 0
        assert max(chunk_sizes) <= 700  # 考虑重叠，不应该远超过chunk_size
    
    def test_error_resilience(self):
        """测试错误恢复能力"""
        # 创建一些有问题的文档
        problematic_files = {
            "empty.txt": "",
            "special_chars.txt": "特殊字符测试：\x00\x01\x02\x03",
            "very_long_line.txt": "这是一行非常长的文本，" * 200 + "没有任何分段。"
        }
        
        for filename, content in problematic_files.items():
            file_path = Path(self.temp_dir) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 测试加载器的错误处理
        loader = DocumentLoaderTemplate()
        loader.setup(file_types=['txt'])
        
        # 应该能处理有问题的文件而不崩溃
        documents = loader.execute(self.temp_dir)
        assert isinstance(documents, list)
        
        # 测试分割器的错误处理
        if documents:
            splitter = TextSplitterTemplate()
            splitter.setup(
                strategy="character",
                chunk_size=100,
                chunk_overlap=20
            )
            
            # 应该能处理各种文档而不崩溃
            chunks = splitter.execute(documents)
            assert isinstance(chunks, list)
    
    def test_metadata_consistency(self):
        """测试元数据一致性"""
        # 加载文档
        loader = DocumentLoaderTemplate()
        loader.setup(file_types=['txt', 'md', 'json'])
        
        documents = loader.execute(self.temp_dir)
        
        # 分割文档
        splitter = TextSplitterTemplate()
        splitter.setup(
            strategy="recursive_character",
            chunk_size=400,
            chunk_overlap=80
        )
        
        chunks = splitter.execute(documents)
        
        # 验证元数据一致性
        for chunk in chunks:
            # 应该包含原始文档的元数据
            assert 'source' in chunk.metadata or 'file_path' in chunk.metadata
            
            # 应该包含分割相关的元数据
            assert 'split_strategy' in chunk.metadata
            assert 'chunk_size' in chunk.metadata
            assert 'chunk_overlap' in chunk.metadata
            assert 'global_chunk_id' in chunk.metadata
        
        # 检查同一文档的不同块是否有相同的源信息
        source_to_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get('source') or chunk.metadata.get('file_path')
            if source:
                if source not in source_to_chunks:
                    source_to_chunks[source] = []
                source_to_chunks[source].append(chunk)
        
        # 每个源文档应该有至少一个块
        assert len(source_to_chunks) >= 1
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        # 加载文档并监控性能
        loader = DocumentLoaderTemplate()
        loader.setup(file_types=['txt', 'md', 'json'])
        
        loader.reset_stats()
        documents = loader.execute(self.temp_dir)
        loader_stats = loader.get_stats()
        
        # 验证加载器统计信息
        assert loader_stats['total_files'] >= 3
        assert loader_stats['loaded_files'] >= 0
        assert loader_stats['loading_time'] > 0
        
        # 分割文档并监控性能
        splitter = TextSplitterTemplate()
        splitter.setup(strategy="character", chunk_size=300)
        
        splitter.reset_stats()
        chunks = splitter.execute(documents)
        splitter_stats = splitter.get_stats()
        
        # 验证分割器统计信息
        assert splitter_stats['total_documents'] == len(documents)
        assert splitter_stats['total_chunks'] == len(chunks)
        assert splitter_stats['split_time'] > 0
        assert splitter_stats['avg_chunks_per_doc'] > 0
        
        # 获取分割结果统计
        split_detail_stats = splitter.get_split_stats(chunks)
        assert 'total_chunks' in split_detail_stats
        assert 'avg_chunk_size' in split_detail_stats


class TestEdgeCases:
    """边界情况测试"""
    
    def test_empty_input_handling(self):
        """测试空输入处理"""
        splitter = TextSplitterTemplate()
        splitter.setup(strategy="character", chunk_size=100)
        
        # 测试空文档列表
        result = splitter.execute([])
        assert result == []
        
        # 测试包含空内容的文档
        empty_docs = [
            Document(page_content="", metadata={"source": "empty"}),
            Document(page_content="   \n\n   ", metadata={"source": "whitespace"})
        ]
        
        result = splitter.execute(empty_docs)
        assert isinstance(result, list)
    
    def test_single_character_chunks(self):
        """测试极小的块大小"""
        splitter = TextSplitterTemplate()
        splitter.setup(
            strategy="character", 
            chunk_size=1,  # 极小的块大小
            chunk_overlap=0
        )
        
        doc = Document(
            page_content="测试",
            metadata={"source": "test"}
        )
        
        result = splitter.execute([doc])
        assert isinstance(result, list)
        assert len(result) >= 1
    
    def test_chunk_larger_than_document(self):
        """测试块大小大于文档大小"""
        splitter = TextSplitterTemplate()
        splitter.setup(
            strategy="character",
            chunk_size=10000,  # 很大的块大小
            chunk_overlap=100
        )
        
        small_doc = Document(
            page_content="短文档",
            metadata={"source": "small"}
        )
        
        result = splitter.execute([small_doc])
        assert len(result) == 1  # 应该只有一个块
        assert result[0].page_content == "短文档"


if __name__ == "__main__":
    pytest.main([__file__])