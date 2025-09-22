"""
RAG知识库系统演示脚本

展示完整的RAG系统工作流程，包括文档处理、向量存储、检索和生成。
"""

import os
import logging
from pathlib import Path
from typing import List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 本地导入
from src.document_processor import DocumentProcessor, ProcessingConfig
from src.vectorstore import VectorStoreManager, EmbeddingConfig
from src.retrieval import create_hybrid_retriever, RetrievalConfig
from src.generation import RAGChain, PromptManager


def setup_demo_environment():
    """设置演示环境"""
    # 创建演示数据目录
    demo_dir = Path("./demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # 创建示例文档
    sample_docs = [
        {
            "title": "人工智能简介",
            "content": """
            人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
            这些任务包括学习、推理、感知、语言理解和问题解决。
            
            AI的主要技术包括机器学习、深度学习、自然语言处理和计算机视觉。
            机器学习让计算机能够从数据中学习，而无需明确编程。
            深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。
            """
        },
        {
            "title": "机器学习基础",
            "content": """
            机器学习是AI的核心技术之一，它使计算机能够自动学习和改进性能。
            主要有三种类型的机器学习：监督学习、无监督学习和强化学习。
            
            监督学习使用标记的训练数据来学习输入和输出之间的映射关系。
            无监督学习从未标记的数据中发现隐藏的模式。
            强化学习通过与环境的交互来学习最优行为策略。
            
            常见的机器学习算法包括线性回归、决策树、随机森林、支持向量机和神经网络。
            """
        },
        {
            "title": "深度学习应用",
            "content": """
            深度学习在许多领域都有重要应用。在计算机视觉中，卷积神经网络（CNN）
            用于图像识别、物体检测和图像分类。
            
            在自然语言处理中，循环神经网络（RNN）和Transformer模型用于
            机器翻译、文本生成和情感分析。
            
            深度学习还在语音识别、推荐系统、医学诊断和自动驾驶等领域
            发挥着重要作用。最近的突破包括GPT系列、BERT和其他大型语言模型。
            """
        },
        {
            "title": "RAG技术详解",
            "content": """
            检索增强生成（RAG）是一种结合信息检索和文本生成的AI技术。
            它首先从知识库中检索相关信息，然后使用这些信息来生成更准确的回答。
            
            RAG系统通常包含三个主要组件：文档索引器、检索器和生成器。
            文档索引器将知识库转换为可搜索的向量表示。
            检索器根据查询找到最相关的文档片段。
            生成器使用检索到的信息来生成最终答案。
            
            RAG的优势包括减少幻觉、提供可追溯的答案来源、支持知识更新。
            """
        }
    ]
    
    # 保存示例文档
    for i, doc in enumerate(sample_docs):
        doc_path = demo_dir / f"doc_{i+1}.txt"
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(f"# {doc['title']}\n\n{doc['content']}")
    
    logger.info(f"演示环境准备完成，创建了 {len(sample_docs)} 个示例文档")
    return demo_dir


def demo_document_processing(demo_dir: Path):
    """演示文档处理功能"""
    logger.info("=== 文档处理演示 ===")
    
    # 配置文档处理器
    config = ProcessingConfig(
        chunk_size=500,
        chunk_overlap=50,
        extract_metadata=True,
        generate_summary=True
    )
    
    # 创建处理器
    processor = DocumentProcessor(config)
    
    # 处理文档目录
    results = processor.process_directory(demo_dir, parallel=False)
    
    # 收集所有处理后的文档
    all_documents = []
    for result in results:
        if result.success:
            all_documents.extend(result.documents)
    
    logger.info(f"文档处理完成，共处理 {len(all_documents)} 个文档块")
    
    # 显示处理统计
    stats = processor.get_stats()
    logger.info(f"处理统计: {stats}")
    
    return all_documents


def demo_vector_storage(documents):
    """演示向量存储功能"""
    logger.info("=== 向量存储演示 ===")
    
    # 配置嵌入模型（使用HuggingFace作为演示）
    embedding_config = EmbeddingConfig(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        batch_size=32
    )
    
    # 创建向量存储管理器
    vector_manager = VectorStoreManager(
        collection_name="demo_knowledge_base",
        store_type="chroma",
        embedding_config=embedding_config,
        persist_directory="./demo_data/vectorstore"
    )
    
    # 添加文档
    doc_ids = vector_manager.add_documents(documents)
    logger.info(f"向量存储完成，添加了 {len(doc_ids)} 个文档")
    
    # 测试搜索
    test_query = "什么是深度学习？"
    search_results = vector_manager.search(test_query, k=3)
    
    logger.info(f"测试搜索: '{test_query}'")
    logger.info(f"找到 {len(search_results)} 个相关文档")
    
    return vector_manager


def demo_hybrid_retrieval(vector_manager, documents):
    """演示混合检索功能"""
    logger.info("=== 混合检索演示 ===")
    
    # 创建混合检索器
    retrieval_config = RetrievalConfig(
        k=5,
        similarity_threshold=0.3
    )
    
    hybrid_retriever = create_hybrid_retriever(
        vector_manager,
        documents,
        config=retrieval_config,
        vector_weight=0.7
    )
    
    # 测试检索
    test_queries = [
        "人工智能的主要技术有哪些？",
        "机器学习的类型",
        "RAG技术的优势",
        "深度学习在哪些领域应用？"
    ]
    
    for query in test_queries:
        logger.info(f"\n查询: {query}")
        
        result = hybrid_retriever.retrieve(query, k=3)
        
        logger.info(f"检索到 {len(result.documents)} 个文档")
        for i, doc in enumerate(result.documents[:2]):  # 只显示前2个
            logger.info(f"  文档 {i+1}: {doc.page_content[:100]}...")
    
    return hybrid_retriever


def demo_mock_llm():
    """创建模拟LLM用于演示"""
    class MockLLM:
        def invoke(self, prompt):
            # 简单的模拟响应
            if "什么是" in prompt or "What is" in prompt:
                return MockResponse("这是一个基于检索信息生成的回答。根据提供的上下文信息，...")
            elif "类型" in prompt or "types" in prompt:
                return MockResponse("根据检索到的信息，主要有以下几种类型：1. ... 2. ... 3. ...")
            elif "优势" in prompt or "advantages" in prompt:
                return MockResponse("基于检索到的文档，主要优势包括：...")
            else:
                return MockResponse("根据检索到的相关信息，我可以为您提供以下回答：...")
    
    class MockResponse:
        def __init__(self, content):
            self.content = content
    
    return MockLLM()


def demo_rag_generation(retriever):
    """演示RAG生成功能"""
    logger.info("=== RAG生成演示 ===")
    
    # 创建模拟LLM
    mock_llm = demo_mock_llm()
    
    # 创建RAG链
    rag_chain = RAGChain(
        llm=mock_llm,
        retriever=retriever,
        max_context_length=2000,
        include_sources=True
    )
    
    # 测试问答
    test_questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "RAG技术有什么优势？",
        "深度学习在哪些领域有应用？"
    ]
    
    for question in test_questions:
        logger.info(f"\n问题: {question}")
        
        response = rag_chain.run(question, k=3)
        
        logger.info(f"回答: {response.answer}")
        logger.info(f"置信度: {response.confidence_score:.2f}")
        logger.info(f"响应时间: {response.generation_time:.2f}秒")
        logger.info(f"使用了 {len(response.source_documents)} 个参考文档")
    
    # 显示统计信息
    stats = rag_chain.get_stats()
    logger.info(f"\nRAG统计: {stats}")


def main():
    """主演示函数"""
    logger.info("🚀 开始RAG知识库系统演示")
    
    try:
        # 1. 设置环境
        demo_dir = setup_demo_environment()
        
        # 2. 文档处理
        documents = demo_document_processing(demo_dir)
        
        # 3. 向量存储
        vector_manager = demo_vector_storage(documents)
        
        # 4. 混合检索
        hybrid_retriever = demo_hybrid_retrieval(vector_manager, documents)
        
        # 5. RAG生成
        demo_rag_generation(hybrid_retriever)
        
        logger.info("\n✅ RAG知识库系统演示完成！")
        
        # 系统信息
        logger.info("\n=== 系统信息 ===")
        collection_info = vector_manager.get_collection_info()
        logger.info(f"集合信息: {collection_info}")
        
        retrieval_stats = hybrid_retriever.get_stats()
        logger.info(f"检索统计: {retrieval_stats}")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()