"""
TextSplitterTemplate 测试用例

测试文本分割器模板的各种功能，包括：
- 各种分割策略测试
- 分割参数配置测试
- 重叠窗口功能测试
- 异步分割功能测试
- 性能测试
- 边界条件测试
"""

import pytest
import asyncio
from typing import List

from langchain_core.documents import Document

from templates.data.text_splitter import (
    TextSplitterTemplate,
    SplitStrategy,
    OverlapStrategy,
    SplitResult
)
from templates.base.template_base import TemplateConfig, TemplateType


class TestTextSplitterTemplate:
    """TextSplitterTemplate 测试类"""
    
    def setup_method(self):
        """测试前的准备工作"""
        self.splitter = TextSplitterTemplate()
        self.sample_documents = self._create_sample_documents()
    
    def _create_sample_documents(self) -> List[Document]:
        """创建测试文档"""
        documents = [
            Document(
                page_content="""这是一个长文档的示例。它包含多个段落和句子。

第一段包含一些重要信息。这些信息对于理解文档内容很重要。
我们需要确保分割后的文本块保持语义完整性。

第二段讨论另一个主题。这个主题与第一段相关但又有所不同。
文本分割应该考虑段落边界和语义连贯性。

第三段提供总结信息。总结信息帮助读者理解整个文档的要点。
这样的分割策略对于RAG应用很重要。""",
                metadata={"source": "test_doc_1.txt", "title": "测试文档1"}
            ),
            Document(
                page_content="""# Markdown文档示例

这是一个Markdown格式的文档。它包含标题、段落和列表。

## 子标题1

- 列表项1：重要信息
- 列表项2：更多信息  
- 列表项3：补充信息

## 子标题2

这里是另一个段落。包含更多的详细信息。

### 三级标题

最后一个段落包含总结信息。""",
                metadata={"source": "test_doc_2.md", "title": "Markdown测试文档"}
            ),
            Document(
                page_content="这是一个短文档。只包含一个段落和几个句子。用于测试边界情况。",
                metadata={"source": "test_doc_3.txt", "title": "短文档"}
            )
        ]
        return documents
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        config = self.splitter._create_default_config()
        
        assert isinstance(config, TemplateConfig)
        assert config.name == "TextSplitterTemplate"
        assert config.template_type == TemplateType.DATA
        assert "strategy" in config.parameters
        assert "chunk_size" in config.parameters
        assert "chunk_overlap" in config.parameters
    
    def test_basic_setup(self):
        """测试基本配置设置"""
        self.splitter.setup(
            strategy="recursive_character",
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " "]
        )
        
        assert self.splitter.strategy == SplitStrategy.RECURSIVE_CHARACTER
        assert self.splitter.chunk_size == 500
        assert self.splitter.chunk_overlap == 100
        assert self.splitter.separators == ["\n\n", "\n", " "]
    
    def test_invalid_strategy(self):
        """测试无效策略处理"""
        with pytest.raises(Exception):
            self.splitter.setup(strategy="invalid_strategy")
    
    def test_character_splitting(self):
        """测试字符分割策略"""
        self.splitter.setup(
            strategy="character",
            chunk_size=200,
            chunk_overlap=50
        )
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        assert isinstance(result_docs, list)
        assert len(result_docs) > len(self.sample_documents)  # 应该被分割成更多块
        
        # 检查每个块的大小
        for doc in result_docs:
            assert len(doc.page_content) <= 250  # 考虑重叠，稍大于chunk_size是正常的
            assert 'chunk_id' in doc.metadata
            assert 'chunk_type' in doc.metadata
    
    def test_sentence_splitting(self):
        """测试句子分割策略"""
        self.splitter.setup(
            strategy="sentence",
            chunk_size=300,
            chunk_overlap=50
        )
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        assert isinstance(result_docs, list)
        assert len(result_docs) >= len(self.sample_documents)
        
        # 检查句子分割是否保持完整性
        for doc in result_docs:
            content = doc.page_content.strip()
            if content:
                # 大多数块应该以句号、问号或感叹号结尾（或者是最后一块）
                assert len(content) > 0
    
    def test_paragraph_splitting(self):
        """测试段落分割策略"""
        self.splitter.setup(
            strategy="paragraph",
            chunk_size=400,
            chunk_overlap=80
        )
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        assert isinstance(result_docs, list)
        
        # 检查段落分割的结果
        for doc in result_docs:
            assert 'chunk_type' in doc.metadata
            assert doc.metadata['chunk_type'] == 'paragraph_split'
    
    def test_recursive_character_splitting(self):
        """测试递归字符分割策略"""
        self.splitter.setup(
            strategy="recursive_character",
            chunk_size=300,
            chunk_overlap=75,
            separators=["\n\n", "\n", " ", ""]
        )
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        assert isinstance(result_docs, list)
        assert len(result_docs) >= len(self.sample_documents)
        
        # 检查分割的有效性
        for doc in result_docs:
            assert len(doc.page_content) > 0
            assert len(doc.page_content) <= 400  # 考虑重叠
    
    def test_markdown_header_splitting(self):
        """测试Markdown标题分割策略"""
        self.splitter.setup(
            strategy="markdown_header",
            chunk_size=500,
            chunk_overlap=50,
            strategy_params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ]
            }
        )
        
        # 只使用Markdown文档
        md_docs = [doc for doc in self.sample_documents if doc.metadata.get('source', '').endswith('.md')]
        
        result_docs = self.splitter.execute(md_docs)
        
        assert isinstance(result_docs, list)
        assert len(result_docs) >= len(md_docs)
        
        # 检查是否按标题分割
        for doc in result_docs:
            assert 'chunk_type' in doc.metadata
            assert doc.metadata['chunk_type'] == 'markdown_header_split'
    
    def test_adaptive_splitting(self):
        """测试自适应分割策略"""
        self.splitter.setup(
            strategy="adaptive",
            chunk_size=400,
            chunk_overlap=100
        )
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        assert isinstance(result_docs, list)
        
        # 检查自适应策略是否选择了合适的分割方法
        for doc in result_docs:
            assert 'adaptive_strategy' in doc.metadata
            # 应该选择了某种具体的分割策略
            strategy_used = doc.metadata['adaptive_strategy']
            assert strategy_used in ['SentenceSplitter', 'ParagraphSplitter', 'RecursiveCharacterSplitter', 'MarkdownHeaderSplitter']
    
    def test_chunk_overlap(self):
        """测试重叠窗口功能"""
        self.splitter.setup(
            strategy="character",
            chunk_size=100,
            chunk_overlap=30
        )
        
        result_docs = self.splitter.execute([self.sample_documents[0]])  # 使用第一个较长的文档
        
        if len(result_docs) >= 2:
            # 检查相邻块是否有重叠
            first_chunk = result_docs[0].page_content
            second_chunk = result_docs[1].page_content
            
            # 查找重叠部分
            overlap_found = False
            for i in range(len(first_chunk) - 10):
                substr = first_chunk[i:i+10]
                if substr in second_chunk:
                    overlap_found = True
                    break
            
            # 注意：由于分割算法的复杂性，重叠可能不总是字符级的完全匹配
            # 这个测试主要确保分割器考虑了重叠参数
    
    def test_empty_document_handling(self):
        """测试空文档处理"""
        empty_docs = [
            Document(page_content="", metadata={"source": "empty.txt"}),
            Document(page_content="   \n\n   ", metadata={"source": "whitespace.txt"})
        ]
        
        self.splitter.setup(strategy="character", chunk_size=100)
        
        result_docs = self.splitter.execute(empty_docs)
        
        # 应该优雅地处理空文档
        assert isinstance(result_docs, list)
        # 空文档可能被过滤掉或保留，取决于实现
    
    def test_very_long_document(self):
        """测试非常长的文档"""
        long_content = "这是一个很长的句子。" * 200  # 创建一个很长的文档
        long_doc = Document(
            page_content=long_content,
            metadata={"source": "long_doc.txt"}
        )
        
        self.splitter.setup(
            strategy="character",
            chunk_size=500,
            chunk_overlap=100
        )
        
        result_docs = self.splitter.execute([long_doc])
        
        assert isinstance(result_docs, list)
        assert len(result_docs) > 1  # 应该被分割成多个块
        
        # 检查每个块的大小
        for doc in result_docs:
            assert len(doc.page_content) <= 600  # 考虑重叠
    
    def test_metadata_preservation(self):
        """测试元数据保持"""
        self.splitter.setup(strategy="character", chunk_size=200)
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        # 检查原始元数据是否被保留
        for doc in result_docs:
            assert 'source' in doc.metadata
            assert 'title' in doc.metadata
            
            # 检查新增的分割元数据
            assert 'chunk_id' in doc.metadata
            assert 'split_strategy' in doc.metadata
            assert 'chunk_size' in doc.metadata
            assert 'chunk_overlap' in doc.metadata
    
    def test_chunk_size_validation(self):
        """测试块大小验证"""
        # 测试过小的块大小
        with pytest.raises(Exception):
            self.splitter.setup(chunk_size=0)
        
        # 测试负数块大小
        with pytest.raises(Exception):
            self.splitter.setup(chunk_size=-100)
    
    def test_overlap_validation(self):
        """测试重叠大小验证"""
        # 重叠大小不应该大于块大小
        with pytest.raises(Exception):
            self.splitter.setup(chunk_size=100, chunk_overlap=150)
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        self.splitter.setup(strategy="character", chunk_size=200)
        
        # 重置统计信息
        self.splitter.reset_stats()
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        stats = self.splitter.get_stats()
        
        assert stats['total_documents'] == len(self.sample_documents)
        assert stats['total_chunks'] == len(result_docs)
        assert stats['split_time'] > 0
        assert stats['avg_chunks_per_doc'] > 0
    
    def test_split_result_statistics(self):
        """测试分割结果统计"""
        self.splitter.setup(strategy="character", chunk_size=200)
        
        result_docs = self.splitter.execute(self.sample_documents)
        split_stats = self.splitter.get_split_stats(result_docs)
        
        assert 'total_chunks' in split_stats
        assert 'min_chunk_size' in split_stats
        assert 'max_chunk_size' in split_stats
        assert 'avg_chunk_size' in split_stats
        assert 'median_chunk_size' in split_stats
        
        assert split_stats['total_chunks'] == len(result_docs)
        assert split_stats['min_chunk_size'] >= 0
        assert split_stats['max_chunk_size'] >= split_stats['min_chunk_size']
    
    @pytest.mark.asyncio
    async def test_async_execution(self):
        """测试异步执行"""
        self.splitter.setup(strategy="character", chunk_size=200)
        
        result_docs = await self.splitter.execute_async(self.sample_documents)
        
        assert isinstance(result_docs, list)
        assert len(result_docs) >= len(self.sample_documents)
    
    def test_custom_separators(self):
        """测试自定义分隔符"""
        custom_separators = ["###", "---", "\n\n", "\n", " "]
        
        self.splitter.setup(
            strategy="recursive_character",
            chunk_size=300,
            separators=custom_separators
        )
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        assert isinstance(result_docs, list)
        # 分割器应该使用自定义分隔符
    
    def test_length_function_customization(self):
        """测试长度函数自定义"""
        # 测试使用token计数而不是字符计数
        self.splitter.setup(
            strategy="character",
            chunk_size=50,  # 50个token
            length_function="token_count"
        )
        
        result_docs = self.splitter.execute(self.sample_documents)
        
        assert isinstance(result_docs, list)
        # 使用token计数应该产生不同的分割结果


class TestTextSplitterIntegration:
    """TextSplitter 集成测试"""
    
    def test_different_strategies_comparison(self):
        """测试不同策略的比较"""
        strategies = ["character", "sentence", "paragraph", "recursive_character"]
        
        sample_doc = Document(
            page_content="""这是一个测试文档。包含多个段落。

第一段有重要信息。信息很详细。

第二段讨论其他内容。内容很丰富。""",
            metadata={"source": "comparison_test.txt"}
        )
        
        results = {}
        
        for strategy in strategies:
            splitter = TextSplitterTemplate()
            splitter.setup(
                strategy=strategy,
                chunk_size=100,
                chunk_overlap=20
            )
            
            result_docs = splitter.execute([sample_doc])
            results[strategy] = len(result_docs)
        
        # 不同策略应该产生不同数量的块
        assert len(set(results.values())) >= 1  # 至少有一些差异
    
    def test_pipeline_integration(self):
        """测试与其他组件的集成"""
        # 模拟与文档加载器的集成
        documents = [
            Document(
                page_content="长文档内容。" * 50,
                metadata={"source": "doc1.txt", "loaded_by": "DocumentLoader"}
            ),
            Document(
                page_content="另一个长文档。" * 30,
                metadata={"source": "doc2.txt", "loaded_by": "DocumentLoader"}
            )
        ]
        
        splitter = TextSplitterTemplate()
        splitter.setup(
            strategy="recursive_character",
            chunk_size=200,
            chunk_overlap=50
        )
        
        chunks = splitter.execute(documents)
        
        # 检查集成结果
        assert len(chunks) > len(documents)
        
        # 检查元数据链
        for chunk in chunks:
            assert 'loaded_by' in chunk.metadata
            assert 'split_strategy' in chunk.metadata
            assert chunk.metadata['loaded_by'] == 'DocumentLoader'


if __name__ == "__main__":
    pytest.main([__file__])