# LangChain Learning 入门教程

本教程将帮助你快速上手LangChain Learning模板系统，从零开始构建你的第一个LLM应用。

## 📚 学习目标

完成本教程后，你将能够：
- 理解模板系统的基本概念和架构
- 配置和使用LLM模板
- 处理文档和构建知识库
- 创建智能对话系统
- 监控和优化应用性能

## 🔧 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install langchain>=0.1.0
pip install langchain-openai>=0.1.0
pip install langchain-anthropic>=0.1.0

# 数据处理依赖
pip install chromadb>=0.4.0
pip install sentence-transformers>=2.2.0
pip install pypdf>=3.0.0

# 可选依赖
pip install faiss-cpu>=1.7.4  # 高性能向量搜索
pip install tiktoken>=0.5.0   # OpenAI token计算
```

### 2. 设置API密钥

```bash
# 设置OpenAI API密钥
export OPENAI_API_KEY="sk-your-openai-api-key"

# 设置Anthropic API密钥（可选）
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"

# 或者创建.env文件
echo "OPENAI_API_KEY=sk-your-openai-api-key" > .env
echo "ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key" >> .env
```

### 3. 验证安装

```python
# test_installation.py
import os
from templates.llm.openai_template import OpenAITemplate

# 检查API密钥
assert os.getenv("OPENAI_API_KEY"), "请设置OPENAI_API_KEY环境变量"

# 测试模板导入
template = OpenAITemplate()
print("✅ 安装成功！模板系统已就绪")
```

## 🚀 第一步：你的第一个LLM调用

让我们从最简单的LLM调用开始：

```python
# step1_first_llm_call.py
from templates.llm.openai_template import OpenAITemplate

# 1. 创建LLM模板实例
llm = OpenAITemplate()

# 2. 配置参数
llm.setup(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)

# 3. 执行调用
response = llm.run("请介绍一下Python编程语言的特点")

# 4. 查看结果
print("AI回复:", response.content)
print("执行时间:", llm.get_status()["execution_time"], "秒")
```

### 💡 理解关键概念

**模板(Template)**: 封装了特定功能的可复用代码单元
- `setup()`: 配置模板参数
- `run()`: 执行核心功能
- `get_status()`: 获取执行状态

**配置参数**:
- `model_name`: 使用的模型名称
- `temperature`: 控制输出随机性 (0.0-2.0)
- `max_tokens`: 最大输出长度

## 📝 第二步：提示工程和参数调优

学习如何优化提示词和调整参数：

```python
# step2_prompt_engineering.py
from templates.llm.openai_template import OpenAITemplate

def compare_temperatures():
    """比较不同温度参数的效果"""
    
    prompt = "请为一家科技公司写一个创意广告语"
    temperatures = [0.1, 0.7, 1.3]
    
    for temp in temperatures:
        print(f"\n=== 温度参数: {temp} ===")
        
        llm = OpenAITemplate()
        llm.setup(
            model_name="gpt-3.5-turbo",
            temperature=temp,
            max_tokens=100
        )
        
        response = llm.run(prompt)
        print("输出:", response.content)

def system_prompt_example():
    """使用系统提示词定义AI角色"""
    
    llm = OpenAITemplate()
    llm.setup(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        system_prompt="你是一个专业的Python编程导师，擅长用简单易懂的方式解释复杂概念。"
    )
    
    response = llm.run("什么是装饰器？请举个例子")
    print("专业导师回复:", response.content)

# 运行示例
compare_temperatures()
system_prompt_example()
```

### 🎯 提示工程最佳实践

1. **明确指令**: 使用清晰、具体的指令
2. **提供上下文**: 给AI足够的背景信息
3. **指定格式**: 明确期望的输出格式
4. **使用示例**: 通过示例展示期望的行为

```python
# 好的提示词示例
good_prompt = """
作为一个专业的数据分析师，请分析以下销售数据：

数据: [1月: 100万, 2月: 120万, 3月: 95万]

请按以下格式输出分析结果：
1. 趋势分析
2. 关键发现
3. 改进建议

使用专业但易懂的语言。
"""
```

## 📄 第三步：文档处理和知识库构建

学习如何处理文档并构建可搜索的知识库：

```python
# step3_document_processing.py
from templates.data.document_loader import DocumentLoaderTemplate
from templates.data.text_splitter import TextSplitterTemplate
from templates.data.vectorstore_template import VectorStoreTemplate

def build_knowledge_base():
    """构建知识库的完整流程"""
    
    # 1. 加载文档
    print("📚 加载文档...")
    loader = DocumentLoaderTemplate()
    loader.setup(
        file_paths=["./docs/company_handbook.pdf", "./docs/policies.txt"],
        file_type="auto",  # 自动检测文件类型
        encoding="utf-8"
    )
    documents = loader.run()
    print(f"✅ 加载了 {len(documents)} 个文档")
    
    # 2. 分割文本
    print("✂️ 分割文本...")
    splitter = TextSplitterTemplate()
    splitter.setup(
        splitter_type="recursive",
        chunk_size=1000,      # 每个块1000字符
        chunk_overlap=100     # 块之间重叠100字符
    )
    chunks = splitter.run(documents)
    print(f"✅ 生成了 {len(chunks)} 个文本块")
    
    # 3. 创建向量存储
    print("🔍 创建向量存储...")
    vectorstore = VectorStoreTemplate()
    vectorstore.setup(
        vectorstore_type="chroma",
        collection_name="company_knowledge",
        embedding_model="text-embedding-ada-002",
        persist_directory="./data/knowledge_base"
    )
    result = vectorstore.run(chunks)
    print(f"✅ 向量存储创建完成，存储了 {result['count']} 个向量")
    
    return vectorstore

def search_knowledge_base(vectorstore):
    """搜索知识库"""
    from templates.data.retrieval_template import RetrievalTemplate
    
    retriever = RetrievalTemplate()
    retriever.setup(
        vectorstore=vectorstore,
        k=3,  # 返回最相关的3个结果
        similarity_threshold=0.7
    )
    
    # 测试搜索
    queries = [
        "公司的休假政策是什么？",
        "如何申请出差？",
        "员工福利有哪些？"
    ]
    
    for query in queries:
        print(f"\n❓ 查询: {query}")
        results = retriever.run(query)
        
        for i, result in enumerate(results, 1):
            print(f"📄 结果 {i}: {result['content'][:100]}...")

# 运行示例
vectorstore = build_knowledge_base()
search_knowledge_base(vectorstore)
```

### 📋 文档处理要点

**支持的文件格式**:
- 文本文件: `.txt`, `.md`
- PDF文档: `.pdf`
- Word文档: `.docx`
- 网页文件: `.html`

**文本分割策略**:
- `recursive`: 递归分割，保持语义完整性
- `character`: 按字符数分割
- `token`: 按token数分割

**向量化选择**:
- OpenAI Embeddings: 高质量，需要API调用
- SentenceTransformers: 本地运行，免费使用

## 🤖 第四步：构建智能对话系统

结合LLM和知识库创建智能问答系统：

```python
# step4_qa_system.py
from templates.llm.openai_template import OpenAITemplate
from templates.chains.sequential_chain import SequentialChainTemplate
from templates.data.retrieval_template import RetrievalTemplate

class IntelligentQASystem:
    """智能问答系统"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self._setup_components()
    
    def _setup_components(self):
        """设置系统组件"""
        
        # 1. 设置检索器
        self.retriever = RetrievalTemplate()
        self.retriever.setup(
            vectorstore=self.vectorstore,
            k=3,
            similarity_threshold=0.6
        )
        
        # 2. 设置LLM
        self.llm = OpenAITemplate()
        self.llm.setup(
            model_name="gpt-3.5-turbo",
            temperature=0.1,  # 低温度确保准确性
            system_prompt="""
            你是一个专业的企业助手。基于提供的文档内容回答用户问题。
            如果文档中没有相关信息，请诚实地说"我在现有文档中没有找到相关信息"。
            回答要准确、简洁、有用。
            """
        )
        
        # 3. 设置问答链
        self.qa_chain = SequentialChainTemplate()
        self.qa_chain.setup(
            chain_type="sequential",
            llm=self.llm.llm,
            steps=[
                {
                    "name": "context_retrieval",
                    "prompt": "基于以下文档内容回答问题：\n\n文档内容：\n{context}\n\n问题：{question}\n\n答案：",
                    "output_key": "answer"
                }
            ],
            input_variables=["context", "question"]
        )
    
    def ask(self, question: str) -> dict:
        """处理用户问题"""
        
        # 1. 检索相关文档
        relevant_docs = self.retriever.run(question)
        
        if not relevant_docs:
            return {
                "answer": "抱歉，我在知识库中没有找到相关信息。",
                "sources": [],
                "confidence": 0.0
            }
        
        # 2. 准备上下文
        context = "\n".join([doc["content"] for doc in relevant_docs])
        
        # 3. 生成回答
        result = self.qa_chain.run({
            "context": context,
            "question": question
        })
        
        return {
            "answer": result["answer"],
            "sources": [doc["metadata"]["source"] for doc in relevant_docs],
            "confidence": sum(doc.get("score", 0) for doc in relevant_docs) / len(relevant_docs)
        }

def interactive_chat(qa_system):
    """交互式聊天界面"""
    print("\n🤖 智能问答系统已启动！输入'quit'退出")
    print("-" * 50)
    
    while True:
        question = input("\n❓ 你的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        if not question:
            continue
        
        # 获取回答
        result = qa_system.ask(question)
        
        print(f"\n🤖 回答: {result['answer']}")
        print(f"📚 信息来源: {', '.join(result['sources'])}")
        print(f"🎯 置信度: {result['confidence']:.2f}")

# 使用示例
if __name__ == "__main__":
    # 假设你已经有了vectorstore（从第三步获得）
    qa_system = IntelligentQASystem(vectorstore)
    
    # 测试几个问题
    test_questions = [
        "公司的工作时间是什么？",
        "如何申请年假？",
        "公司有哪些培训机会？"
    ]
    
    for question in test_questions:
        result = qa_system.ask(question)
        print(f"\n问题: {question}")
        print(f"回答: {result['answer']}")
    
    # 启动交互式聊天
    interactive_chat(qa_system)
```

## 🔄 第五步：使用链组合复杂工作流

学习如何组合多个步骤创建复杂的工作流：

```python
# step5_advanced_chains.py
from templates.chains.sequential_chain import SequentialChainTemplate
from templates.chains.parallel_chain import ParallelChainTemplate
from templates.llm.openai_template import OpenAITemplate

def content_creation_pipeline():
    """内容创作流水线"""
    
    # 创建LLM实例
    llm = OpenAITemplate()
    llm.setup(model_name="gpt-3.5-turbo", temperature=0.8)
    
    # 创建顺序链：大纲 -> 内容 -> 修改
    chain = SequentialChainTemplate()
    chain.setup(
        chain_type="sequential",
        llm=llm.llm,
        steps=[
            {
                "name": "outline_generation",
                "prompt": "为主题'{topic}'创建详细的文章大纲，包含3-5个主要部分",
                "output_key": "outline"
            },
            {
                "name": "content_writing",
                "prompt": "基于以下大纲写一篇专业的文章：\n{outline}",
                "output_key": "draft"
            },
            {
                "name": "content_polish",
                "prompt": "改进以下文章，使其更加生动有趣：\n{draft}",
                "output_key": "final_article"
            }
        ],
        input_variables=["topic"]
    )
    
    # 测试流水线
    topic = "人工智能在教育中的应用"
    result = chain.run({"topic": topic})
    
    print("📝 大纲:")
    print(result["outline"])
    print("\n📄 最终文章:")
    print(result["final_article"][:500] + "...")

def multi_perspective_analysis():
    """多角度分析（并行处理）"""
    
    llm = OpenAITemplate()
    llm.setup(model_name="gpt-3.5-turbo", temperature=0.6)
    
    # 创建并行链：同时进行技术、商业、社会影响分析
    chain = ParallelChainTemplate()
    chain.setup(
        chain_type="parallel",
        llm=llm.llm,
        steps=[
            {
                "name": "technical_analysis",
                "prompt": "从技术角度分析'{topic}'，重点关注技术原理和实现方式",
                "output_key": "tech_analysis"
            },
            {
                "name": "business_analysis",
                "prompt": "从商业角度分析'{topic}'，重点关注市场机会和商业模式",
                "output_key": "business_analysis"
            },
            {
                "name": "social_analysis",
                "prompt": "从社会影响角度分析'{topic}'，重点关注对社会的积极和消极影响",
                "output_key": "social_analysis"
            }
        ],
        input_variables=["topic"],
        max_workers=3,  # 并行执行
        timeout=60.0
    )
    
    # 测试并行分析
    topic = "区块链技术"
    result = chain.run({"topic": topic})
    
    print("🔧 技术分析:")
    print(result["tech_analysis"][:200] + "...")
    print("\n💼 商业分析:")
    print(result["business_analysis"][:200] + "...")
    print("\n🌍 社会影响分析:")
    print(result["social_analysis"][:200] + "...")

# 运行示例
content_creation_pipeline()
print("\n" + "="*50 + "\n")
multi_perspective_analysis()
```

## 📊 第六步：性能监控和优化

学习如何监控和优化你的应用：

```python
# step6_monitoring.py
from templates.evaluation.performance_eval import PerformanceEvalTemplate
from templates.evaluation.cost_analysis import CostAnalysisTemplate
import time

def performance_monitoring_example():
    """性能监控示例"""
    
    # 创建性能评估器
    perf_eval = PerformanceEvalTemplate()
    perf_eval.setup(
        benchmark_enabled=True,
        profiling_enabled=True,
        memory_profiling=True
    )
    
    # 创建成本分析器
    cost_analyzer = CostAnalysisTemplate()
    cost_analyzer.setup(
        track_api_calls=True,
        cost_per_token=0.0015,  # GPT-3.5-turbo价格
        daily_budget=10.0
    )
    
    # 测试不同配置的性能
    llm = OpenAITemplate()
    
    test_configs = [
        {"model": "gpt-3.5-turbo", "temp": 0.3, "max_tokens": 500},
        {"model": "gpt-3.5-turbo", "temp": 0.7, "max_tokens": 1000},
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n📈 测试配置 {i}: {config}")
        
        llm.setup(
            model_name=config["model"],
            temperature=config["temp"],
            max_tokens=config["max_tokens"]
        )
        
        # 性能测试
        with perf_eval.measure(f"config_{i}"):
            with cost_analyzer.track_cost():
                start_time = time.time()
                
                # 执行多次调用
                for j in range(3):
                    result = llm.run(f"解释什么是云计算 - 测试 {j+1}")
                
                end_time = time.time()
        
        # 显示结果
        print(f"⏱️ 总执行时间: {end_time - start_time:.2f}秒")
        print(f"💰 预估成本: ${cost_analyzer.get_total_cost():.4f}")
        
        # 获取性能报告
        perf_report = perf_eval.get_report(f"config_{i}")
        print(f"🧠 内存峰值: {perf_report.get('peak_memory', 'N/A')} MB")

def optimization_tips():
    """优化建议"""
    print("\n🚀 性能优化建议:")
    print("1. 模型选择: 根据任务复杂度选择合适的模型")
    print("2. 参数调优: 调整temperature和max_tokens平衡质量和成本")
    print("3. 缓存策略: 对相似查询启用缓存")
    print("4. 批处理: 使用批处理减少API调用开销")
    print("5. 异步处理: 对独立任务使用异步执行")

# 运行监控示例
performance_monitoring_example()
optimization_tips()
```

## 🎓 总结和下一步

恭喜！你已经完成了LangChain Learning的入门教程。现在你应该能够：

### ✅ 你学会了什么

1. **基础操作**:
   - 配置和使用LLM模板
   - 处理不同类型的文档
   - 构建和搜索向量数据库

2. **高级功能**:
   - 创建复杂的处理链
   - 构建智能问答系统
   - 监控应用性能

3. **最佳实践**:
   - 提示工程技巧
   - 错误处理策略
   - 性能优化方法

### 🚀 下一步学习

1. **深入学习**:
   - [智能代理教程](02_agents_deep_dive.md)
   - [高级链组合](03_advanced_chains.md)
   - [自定义模板开发](04_custom_templates.md)

2. **实战项目**:
   - 构建客服机器人
   - 创建代码助手
   - 开发内容生成工具

3. **社区资源**:
   - [GitHub讨论区](https://github.com/your-repo/discussions)
   - [示例项目库](../examples/)
   - [最佳实践指南](../best_practices/)

### 🔧 常用命令速查

```bash
# 快速测试安装
python -c "from templates.llm.openai_template import OpenAITemplate; print('✅ OK')"

# 查看模板列表
ls templates/*/

# 运行基础示例
python templates/examples/basic_examples/01_llm_basic_usage.py

# 检查配置
python -c "from templates.base.config_loader import ConfigLoader; print(ConfigLoader().get_config())"
```

### 💡 提示和技巧

- **开发时**: 使用`development.yaml`配置，启用调试模式
- **生产时**: 使用`production.yaml`配置，关注性能和安全
- **测试时**: 使用较小的模型和数据集快速迭代
- **调试时**: 启用`verbose=True`查看详细执行过程

---

**需要帮助？**
- 查看 [FAQ文档](../docs/faq.md)
- 访问 [故障排除指南](../docs/troubleshooting.md)
- 在 [GitHub Issues](https://github.com/your-repo/issues) 提问

祝你在LangChain Learning的学习旅程中收获满满！🎉