#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模板基础使用示例

本示例演示如何使用数据处理模板进行文档处理，包括：
1. 文档加载和解析
2. 文本分割和预处理
3. 向量化和存储
4. 语义搜索和检索

作者: LangChain Learning Project
版本: 1.0.0
"""

import os
import sys
import tempfile
from typing import List, Dict, Any

# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from templates.data.document_loader import DocumentLoaderTemplate
from templates.data.text_splitter import TextSplitterTemplate
from templates.data.vectorstore_template import VectorStoreTemplate
from templates.data.retrieval_template import RetrievalTemplate


def create_sample_documents():
    """创建示例文档用于测试"""
    print("=== 创建示例文档 ===")
    
    sample_texts = [
        """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        
        机器学习是人工智能的一个子领域，它使计算机具有学习能力，
        而不需要明确编程。机器学习专注于计算机程序的开发，
        这些程序可以访问数据并使用它来为自己学习。
        """,
        
        """
        深度学习是机器学习的一个子集，它模仿人脑的工作方式来处理数据并创建用于决策的模式。
        深度学习使用人工神经网络，这些网络具有多个层，可以学习数据的复杂模式。
        
        神经网络由互连的节点组成，这些节点模仿人脑中的神经元。
        每个连接都有一个权重，在学习过程中会调整这些权重。
        深度学习在图像识别、语音识别和自然语言处理等领域取得了重大突破。
        """,
        
        """
        自然语言处理（Natural Language Processing，NLP）是人工智能的一个重要分支，
        它涉及计算机和人类语言之间的交互。NLP的目标是让计算机能够理解、
        解释和生成人类语言。
        
        NLP的应用包括机器翻译、文本摘要、情感分析、问答系统、
        聊天机器人等。现代NLP技术广泛使用深度学习方法，
        特别是Transformer模型，如BERT、GPT等。
        """,
        
        """
        LangChain是一个用于开发由语言模型驱动的应用程序的框架。
        它提供了一套工具和组件，帮助开发者构建复杂的LLM应用。
        
        LangChain的核心概念包括：
        1. LLMs和Prompts - 与语言模型交互的接口
        2. Chains - 将多个组件链接在一起
        3. Agents - 使用LLM作为推理引擎的实体
        4. Memory - 在链或代理调用之间保持状态
        5. Document Loaders - 从各种源加载文档
        """
    ]
    
    # 创建临时文档文件
    temp_dir = tempfile.mkdtemp()
    doc_paths = []
    
    for i, text in enumerate(sample_texts, 1):
        file_path = os.path.join(temp_dir, f"document_{i}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        doc_paths.append(file_path)
        print(f"创建文档: {file_path}")
    
    return doc_paths, temp_dir


def document_loading_example(doc_paths: List[str]):
    """文档加载示例"""
    print("\n=== 文档加载示例 ===")
    
    # 创建文档加载器模板
    loader = DocumentLoaderTemplate()
    
    # 配置加载器参数
    loader.setup(
        file_paths=doc_paths,
        file_type="txt",
        encoding="utf-8",
        batch_size=2,
        parallel_processing=True
    )
    
    try:
        # 执行文档加载
        documents = loader.run()
        
        print(f"成功加载 {len(documents)} 个文档")
        
        # 显示加载的文档信息
        for i, doc in enumerate(documents, 1):
            print(f"\n--- 文档 {i} ---")
            print(f"来源: {doc.metadata.get('source', 'Unknown')}")
            print(f"内容长度: {len(doc.page_content)} 字符")
            print(f"内容预览: {doc.page_content[:100]}...")
        
        return documents
        
    except Exception as e:
        print(f"文档加载失败: {str(e)}")
        return []


def text_splitting_example(documents: List[Any]):
    """文本分割示例"""
    print("\n=== 文本分割示例 ===")
    
    if not documents:
        print("没有文档可供分割")
        return []
    
    # 创建文本分割器模板
    splitter = TextSplitterTemplate()
    
    # 配置分割器参数
    splitter.setup(
        splitter_type="recursive",
        chunk_size=300,  # 较小的块大小用于演示
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " ", ""],
        length_function="len"
    )
    
    try:
        # 执行文本分割
        chunks = splitter.run(documents)
        
        print(f"文档分割完成，共生成 {len(chunks)} 个文本块")
        
        # 显示分割结果
        for i, chunk in enumerate(chunks[:5], 1):  # 只显示前5个块
            print(f"\n--- 文本块 {i} ---")
            print(f"来源: {chunk.metadata.get('source', 'Unknown')}")
            print(f"块ID: {chunk.metadata.get('chunk_id', 'Unknown')}")
            print(f"长度: {len(chunk.page_content)} 字符")
            print(f"内容: {chunk.page_content}")
        
        if len(chunks) > 5:
            print(f"\n... 还有 {len(chunks) - 5} 个文本块")
        
        return chunks
        
    except Exception as e:
        print(f"文本分割失败: {str(e)}")
        return []


def vectorstore_example(chunks: List[Any]):
    """向量存储示例"""
    print("\n=== 向量存储示例 ===")
    
    if not chunks:
        print("没有文本块可供向量化")
        return None
    
    # 创建向量存储模板
    vectorstore = VectorStoreTemplate()
    
    # 配置向量存储参数
    vectorstore.setup(
        vectorstore_type="chroma",  # 使用本地Chroma数据库
        collection_name="ai_knowledge_base",
        embedding_model="sentence-transformers",  # 使用本地嵌入模型
        embedding_config={
            "model_name": "all-MiniLM-L6-v2",
            "device": "cpu"
        },
        persist_directory="./data/chroma_db",
        metadata_keys=["source", "chunk_id"]
    )
    
    try:
        # 执行向量化和存储
        result = vectorstore.run(chunks)
        
        print("向量存储创建成功")
        print(f"存储位置: {vectorstore.persist_directory}")
        print(f"集合名称: {vectorstore.collection_name}")
        print(f"向量维度: {result.get('dimension', 'Unknown')}")
        print(f"存储的文档数量: {result.get('count', len(chunks))}")
        
        return vectorstore
        
    except Exception as e:
        print(f"向量存储失败: {str(e)}")
        return None


def retrieval_example(vectorstore: Any):
    """检索示例"""
    print("\n=== 语义检索示例 ===")
    
    if not vectorstore:
        print("没有可用的向量存储")
        return
    
    # 创建检索模板
    retriever = RetrievalTemplate()
    
    # 配置检索参数
    retriever.setup(
        vectorstore=vectorstore,
        search_type="similarity",
        k=3,  # 返回最相关的3个结果
        similarity_threshold=0.5,
        include_metadata=True
    )
    
    # 测试查询
    queries = [
        "什么是机器学习？",
        "深度学习的应用领域",
        "LangChain的核心概念",
        "自然语言处理技术"
    ]
    
    for query in queries:
        print(f"\n--- 查询: {query} ---")
        
        try:
            # 执行检索
            results = retriever.run(query)
            
            if results:
                print(f"找到 {len(results)} 个相关结果:")
                
                for i, result in enumerate(results, 1):
                    print(f"\n结果 {i}:")
                    print(f"相似度分数: {result.get('score', 'N/A')}")
                    print(f"来源: {result['metadata'].get('source', 'Unknown')}")
                    print(f"内容: {result['content'][:150]}...")
            else:
                print("未找到相关结果")
                
        except Exception as e:
            print(f"检索失败: {str(e)}")


def batch_processing_example():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建更多示例文档
    additional_texts = [
        "机器人技术是人工智能的一个重要应用领域...",
        "计算机视觉使计算机能够理解和解释视觉信息...",
        "强化学习是机器学习的一个分支，专注于智能体如何在环境中学习..."
    ]
    
    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    batch_paths = []
    
    for i, text in enumerate(additional_texts, 1):
        file_path = os.path.join(temp_dir, f"batch_doc_{i}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        batch_paths.append(file_path)
    
    # 创建批量处理管道
    pipeline_steps = [
        {
            "name": "loading",
            "template": DocumentLoaderTemplate(),
            "config": {
                "file_paths": batch_paths,
                "batch_size": 10,
                "parallel_processing": True
            }
        },
        {
            "name": "splitting", 
            "template": TextSplitterTemplate(),
            "config": {
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        },
        {
            "name": "vectorizing",
            "template": VectorStoreTemplate(),
            "config": {
                "vectorstore_type": "faiss",  # 使用FAISS进行批量处理
                "collection_name": "batch_collection"
            }
        }
    ]
    
    print("开始批量处理管道...")
    
    # 执行批量处理
    current_data = None
    for step in pipeline_steps:
        print(f"执行步骤: {step['name']}")
        
        try:
            template = step["template"]
            template.setup(**step["config"])
            
            if current_data is None:
                current_data = template.run()
            else:
                current_data = template.run(current_data)
                
            print(f"步骤 {step['name']} 完成")
            
        except Exception as e:
            print(f"步骤 {step['name']} 失败: {str(e)}")
            break
    
    print("批量处理完成")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)


def performance_analysis_example():
    """性能分析示例"""
    print("\n=== 性能分析示例 ===")
    
    # 创建性能监控的文档加载器
    loader = DocumentLoaderTemplate()
    loader.setup(
        enable_metrics=True,
        profiling_enabled=True
    )
    
    # 创建测试文档
    test_docs, temp_dir = create_sample_documents()
    
    try:
        # 测试不同的配置参数
        configs = [
            {"batch_size": 1, "parallel_processing": False},
            {"batch_size": 2, "parallel_processing": False},
            {"batch_size": 2, "parallel_processing": True}
        ]
        
        for i, config in enumerate(configs, 1):
            print(f"\n--- 配置 {i}: {config} ---")
            
            loader.setup(
                file_paths=test_docs,
                **config,
                enable_metrics=True
            )
            
            # 执行并测量性能
            start_time = time.time()
            documents = loader.run()
            end_time = time.time()
            
            # 获取性能指标
            metrics = loader.get_metrics()
            
            print(f"执行时间: {end_time - start_time:.2f}秒")
            print(f"加载文档数: {len(documents)}")
            print(f"平均每文档时间: {(end_time - start_time) / len(documents):.3f}秒")
            
            if metrics:
                print(f"内存使用: {metrics.get('memory_usage', 'N/A')}")
                print(f"CPU使用: {metrics.get('cpu_usage', 'N/A')}")
    
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """主函数"""
    print("LangChain Learning - 数据处理模板基础使用示例")
    print("=" * 50)
    
    try:
        # 1. 创建示例文档
        doc_paths, temp_dir = create_sample_documents()
        
        # 2. 文档加载示例
        documents = document_loading_example(doc_paths)
        
        # 3. 文本分割示例
        chunks = text_splitting_example(documents)
        
        # 4. 向量存储示例
        vectorstore = vectorstore_example(chunks)
        
        # 5. 检索示例
        retrieval_example(vectorstore)
        
        # 6. 批量处理示例
        batch_processing_example()
        
        # 7. 性能分析示例
        import time
        performance_analysis_example()
        
        print("\n所有数据处理示例执行完成！")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        print("临时文件已清理")
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
        
    except Exception as e:
        print(f"示例执行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()