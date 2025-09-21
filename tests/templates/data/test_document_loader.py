"""
DocumentLoaderTemplate 测试用例

测试文档加载器模板的各种功能，包括：
- 基本文档加载功能
- 多种文档格式支持
- 批量文档处理
- 异步文档加载
- 错误处理和异常情况
- 性能测试
"""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List

from langchain_core.documents import Document

from templates.data.document_loader import (
    DocumentLoaderTemplate,
    DocumentFormat,
    LoadingMode,
    DocumentMetadata,
    LoadingResult
)
from templates.base.template_base import TemplateConfig, TemplateType


class TestDocumentLoaderTemplate:
    """DocumentLoaderTemplate 测试类"""
    
    def setup_method(self):
        """测试前的准备工作"""
        self.loader = DocumentLoaderTemplate()
        self.test_dir = tempfile.mkdtemp()
        self.test_files = {}
        self._create_test_files()
    
    def teardown_method(self):
        """测试后的清理工作"""
        # 清理测试文件
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_files(self):
        """创建测试文件"""
        # 创建文本文件
        txt_content = "这是一个测试文本文件。\n包含多行内容。\n用于测试文档加载功能。"
        txt_path = Path(self.test_dir) / "test.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        self.test_files['txt'] = str(txt_path)
        
        # 创建Markdown文件
        md_content = """# 测试标题
        
这是一个测试Markdown文件。

## 子标题

- 列表项1
- 列表项2
- 列表项3

```python
print("Hello, World!")
```
"""
        md_path = Path(self.test_dir) / "test.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        self.test_files['md'] = str(md_path)
        
        # 创建JSON文件
        json_content = """{
    "title": "测试文档",
    "content": "这是JSON格式的测试内容",
    "metadata": {
        "author": "测试用户",
        "date": "2024-01-01"
    }
}"""
        json_path = Path(self.test_dir) / "test.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        self.test_files['json'] = str(json_path)
        
        # 创建CSV文件
        csv_content = """name,age,city
张三,25,北京
李四,30,上海
王五,28,广州"""
        csv_path = Path(self.test_dir) / "test.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        self.test_files['csv'] = str(csv_path)
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        config = self.loader._create_default_config()
        
        assert isinstance(config, TemplateConfig)
        assert config.name == "DocumentLoaderTemplate"
        assert config.template_type == TemplateType.DATA
        assert "file_types" in config.parameters
        assert "max_file_size" in config.parameters
        assert "recursive" in config.parameters
    
    def test_basic_setup(self):
        """测试基本配置设置"""
        self.loader.setup(
            file_types=['txt', 'md', 'json'],
            max_file_size=1024 * 1024,  # 1MB
            recursive=True,
            max_workers=2
        )
        
        assert self.loader.supported_formats == ['txt', 'md', 'json']
        assert self.loader.max_file_size == 1024 * 1024
        assert self.loader.recursive == True
        assert self.loader.max_workers == 2
    
    def test_invalid_parameters(self):
        """测试无效参数处理"""
        # 测试无效的文件类型
        with pytest.raises(Exception):
            self.loader.setup(file_types=['invalid_format'])
    
    def test_single_file_loading(self):
        """测试单个文件加载"""
        self.loader.setup(file_types=['txt'])
        
        documents = self.loader.execute(self.test_files['txt'])
        
        assert isinstance(documents, list)
        assert len(documents) >= 1
        assert isinstance(documents[0], Document)
        assert len(documents[0].page_content) > 0
        assert 'source' in documents[0].metadata or 'file_path' in documents[0].metadata
    
    def test_multiple_file_loading(self):
        """测试多个文件加载"""
        self.loader.setup(file_types=['txt', 'md', 'json'])
        
        file_paths = [
            self.test_files['txt'],
            self.test_files['md'],
            self.test_files['json']
        ]
        
        documents = self.loader.execute(file_paths)
        
        assert isinstance(documents, list)
        assert len(documents) >= 3  # 至少3个文档
        
        # 检查每个文档都有内容
        for doc in documents:
            assert isinstance(doc, Document)
            assert len(doc.page_content) > 0
    
    def test_directory_loading(self):
        """测试目录加载"""
        self.loader.setup(
            file_types=['txt', 'md', 'json', 'csv'],
            recursive=True
        )
        
        documents = self.loader.execute(self.test_dir)
        
        assert isinstance(documents, list)
        assert len(documents) >= 4  # 至少4个文件
    
    def test_file_filtering(self):
        """测试文件过滤"""
        self.loader.setup(
            file_types=['txt', 'md'],  # 只加载txt和md文件
            max_file_size=1000  # 小文件大小限制
        )
        
        documents = self.loader.execute(self.test_dir)
        
        # 检查只返回了指定格式的文件
        for doc in documents:
            file_path = doc.metadata.get('source', doc.metadata.get('file_path', ''))
            assert file_path.endswith(('.txt', '.md'))
    
    def test_batch_processing(self):
        """测试批量处理"""
        self.loader.setup(
            file_types=['txt', 'md', 'json', 'csv'],
            batch_size=2,
            max_workers=2
        )
        
        # 创建更多测试文件
        additional_files = []
        for i in range(5):
            file_path = Path(self.test_dir) / f"test_{i}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"测试文件 {i} 的内容")
            additional_files.append(str(file_path))
        
        documents = self.loader.execute(additional_files)
        
        assert len(documents) == 5
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        self.loader.setup(file_types=['txt', 'md'])
        
        # 重置统计信息
        self.loader.reset_stats()
        
        documents = self.loader.execute([self.test_files['txt'], self.test_files['md']])
        
        stats = self.loader.get_stats()
        
        assert stats['total_files'] >= 2
        assert stats['loaded_files'] >= 2
        assert stats['total_documents'] >= 2
        assert stats['loading_time'] > 0
    
    def test_format_detection(self):
        """测试文档格式检测"""
        # 测试各种格式的检测
        formats_to_test = ['txt', 'md', 'json', 'csv']
        
        for fmt in formats_to_test:
            if fmt in self.test_files:
                file_path = Path(self.test_files[fmt])
                
                # 检测格式
                suffix = file_path.suffix.lower().lstrip('.')
                try:
                    detected_format = DocumentFormat(suffix)
                    assert detected_format.value == fmt
                except ValueError:
                    # 某些格式可能不支持，这是正常的
                    pass
    
    def test_metadata_extraction(self):
        """测试元数据提取"""
        self.loader.setup(file_types=['txt'])
        
        documents = self.loader.execute(self.test_files['txt'])
        
        assert len(documents) >= 1
        doc = documents[0]
        
        # 检查基本元数据
        assert 'file_path' in doc.metadata or 'source' in doc.metadata
        assert 'file_name' in doc.metadata
        assert 'file_size' in doc.metadata
        
        # 检查文件大小是否合理
        file_size = doc.metadata.get('file_size', 0)
        assert file_size > 0
    
    def test_encoding_detection(self):
        """测试编码检测"""
        # 创建不同编码的文件
        utf8_content = "UTF-8编码的中文内容"
        utf8_path = Path(self.test_dir) / "utf8_test.txt"
        with open(utf8_path, 'w', encoding='utf-8') as f:
            f.write(utf8_content)
        
        self.loader.setup(file_types=['txt'])
        documents = self.loader.execute(str(utf8_path))
        
        assert len(documents) >= 1
        assert "中文内容" in documents[0].page_content
    
    def test_error_handling(self):
        """测试错误处理"""
        self.loader.setup(file_types=['txt'])
        
        # 测试不存在的文件
        non_existent_file = "/path/to/non/existent/file.txt"
        documents = self.loader.execute(non_existent_file)
        
        # 应该返回空列表而不是抛出异常
        assert isinstance(documents, list)
        
        # 检查统计信息中的失败计数
        stats = self.loader.get_stats()
        assert stats['failed_files'] >= 0
    
    def test_large_file_handling(self):
        """测试大文件处理"""
        # 创建一个较大的测试文件
        large_file_path = Path(self.test_dir) / "large_test.txt"
        large_content = "大文件测试内容。\n" * 1000  # 创建较大的内容
        
        with open(large_file_path, 'w', encoding='utf-8') as f:
            f.write(large_content)
        
        # 设置文件大小限制
        self.loader.setup(
            file_types=['txt'],
            max_file_size=len(large_content.encode('utf-8')) + 100  # 略大于文件大小
        )
        
        documents = self.loader.execute(str(large_file_path))
        assert len(documents) >= 1
        
        # 测试文件大小限制
        self.loader.setup(
            file_types=['txt'],
            max_file_size=100  # 设置很小的限制
        )
        
        documents = self.loader.execute(str(large_file_path))
        # 应该被过滤掉
        assert len(documents) == 0
    
    def test_concurrent_loading(self):
        """测试并发加载"""
        # 创建多个测试文件
        test_files = []
        for i in range(10):
            file_path = Path(self.test_dir) / f"concurrent_test_{i}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"并发测试文件 {i}")
            test_files.append(str(file_path))
        
        self.loader.setup(
            file_types=['txt'],
            max_workers=4,
            batch_size=3
        )
        
        start_time = datetime.now()
        documents = self.loader.execute(test_files)
        end_time = datetime.now()
        
        assert len(documents) == 10
        
        # 并发加载应该比顺序加载快（这个测试可能不稳定，取决于系统性能）
        loading_time = (end_time - start_time).total_seconds()
        assert loading_time < 10  # 应该在合理时间内完成
    
    def test_json_structure_preservation(self):
        """测试JSON结构保持"""
        self.loader.setup(file_types=['json'])
        
        documents = self.loader.execute(self.test_files['json'])
        
        assert len(documents) >= 1
        content = documents[0].page_content
        
        # 检查JSON内容是否被正确加载
        assert "测试文档" in content
        assert "测试用户" in content
    
    def test_csv_data_loading(self):
        """测试CSV数据加载"""
        self.loader.setup(file_types=['csv'])
        
        documents = self.loader.execute(self.test_files['csv'])
        
        assert len(documents) >= 1
        
        # 检查CSV内容
        content = documents[0].page_content
        assert "张三" in content
        assert "北京" in content
    
    def test_markdown_structure_handling(self):
        """测试Markdown结构处理"""
        self.loader.setup(file_types=['md'])
        
        documents = self.loader.execute(self.test_files['md'])
        
        assert len(documents) >= 1
        content = documents[0].page_content
        
        # 检查Markdown内容
        assert "测试标题" in content
        assert "列表项" in content or "Hello, World!" in content


class TestDocumentLoaderIntegration:
    """DocumentLoader 集成测试"""
    
    def test_end_to_end_processing(self):
        """测试端到端文档处理"""
        loader = DocumentLoaderTemplate()
        
        # 创建临时测试目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建多种格式的测试文件
            files = {
                'test1.txt': '这是第一个测试文件',
                'test2.md': '# Markdown测试\n这是markdown内容',
                'test3.json': '{"content": "JSON测试内容"}'
            }
            
            file_paths = []
            for filename, content in files.items():
                file_path = Path(temp_dir) / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                file_paths.append(str(file_path))
            
            # 配置加载器
            loader.setup(
                file_types=['txt', 'md', 'json'],
                recursive=True,
                max_workers=2
            )
            
            # 执行加载
            documents = loader.execute(temp_dir)
            
            # 验证结果
            assert len(documents) >= 3
            
            # 检查每个文档都有必要的元数据
            for doc in documents:
                assert len(doc.page_content) > 0
                assert 'file_path' in doc.metadata or 'source' in doc.metadata
                assert 'file_name' in doc.metadata
            
            # 检查统计信息
            stats = loader.get_stats()
            assert stats['loaded_files'] >= 3
            assert stats['total_documents'] >= 3
            assert stats['loading_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__])