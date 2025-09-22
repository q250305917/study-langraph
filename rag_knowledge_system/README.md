# RAG知识库系统

一个基于LangChain构建的完整RAG（检索增强生成）知识库系统，支持多种文档格式、向量数据库和智能检索策略。

## 🚀 核心特性

### 📄 文档处理
- **多格式支持**: PDF、Word、文本、Markdown、HTML
- **智能分割**: 递归字符分割、语义分割、Markdown结构保持
- **元数据提取**: 自动提取文件信息、内容摘要、关键词
- **批量处理**: 支持并发处理和增量更新

### 🔍 向量存储
- **多种后端**: Chroma、FAISS、Pinecone、Elasticsearch
- **嵌入模型**: OpenAI、HuggingFace、SentenceTransformers
- **索引优化**: 自动索引管理、增量更新、性能监控
- **缓存机制**: 智能缓存提升查询性能

### 🔎 智能检索
- **混合检索**: 向量检索 + 关键词检索（BM25）
- **重排序**: CrossEncoder、多样性优化、分数融合
- **多查询扩展**: LLM驱动的查询改写和扩展
- **过滤器**: 元数据过滤、相似度阈值、时间范围

### 🤖 生成模块
- **提示词管理**: 多场景模板、动态参数、版本控制
- **上下文压缩**: 智能截断、语义压缩、token管理
- **对话支持**: 会话历史、上下文记忆、多轮对话
- **质量评估**: 置信度计算、来源标注、回答评估

### 🌐 API接口
- **RESTful API**: 完整的HTTP接口
- **文档管理**: 上传、删除、更新、批量操作
- **检索查询**: 搜索、RAG问答、对话接口
- **系统监控**: 健康检查、性能指标、日志管理

## 📦 安装部署

### 环境要求
- Python 3.8+
- 16GB+ RAM（推荐）
- 5GB+ 磁盘空间

### 快速安装
```bash
# 克隆项目
git clone <repository-url>
cd rag_knowledge_system

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
cp .env.example .env
# 编辑.env文件，添加API密钥

# 运行演示
python demo.py
```

### Docker部署
```bash
# 构建镜像
docker build -t rag-system .

# 运行容器
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -v ./data:/app/data \
  rag-system
```

## 🎯 快速开始

### 基础用法

```python
from rag_knowledge_system import (
    DocumentProcessor, VectorStoreManager, 
    RAGChain, create_hybrid_retriever
)

# 1. 处理文档
processor = DocumentProcessor()
documents = processor.process_directory("./documents")

# 2. 创建向量存储
vector_manager = VectorStoreManager(
    collection_name="my_knowledge_base",
    store_type="chroma"
)
vector_manager.add_documents(documents)

# 3. 创建检索器
retriever = create_hybrid_retriever(vector_manager, documents)

# 4. 创建RAG链
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.1)

rag_chain = RAGChain(llm=llm, retriever=retriever)

# 5. 进行问答
response = rag_chain.run("什么是人工智能？")
print(f"回答: {response.answer}")
print(f"置信度: {response.confidence_score}")
```

### API使用

```bash
# 启动API服务
python -m src.api.app

# 上传文档
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "这是一个测试文档内容",
    "metadata": {"title": "测试文档"}
  }'

# RAG问答
curl -X POST "http://localhost:8000/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是机器学习？",
    "k": 5
  }'
```

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档处理模块   │    │   向量存储模块   │    │   检索模块      │
│                │    │                │    │                │
│ • 多格式加载     │ ─► │ • 嵌入管理      │ ─► │ • 向量检索      │
│ • 智能分割      │    │ • 向量数据库     │    │ • 关键词检索    │
│ • 元数据提取     │    │ • 索引管理      │    │ • 混合策略      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐            │
│   API接口模块   │    │   生成模块      │ ◄──────────┘
│                │    │                │
│ • RESTful API  │ ◄─ │ • RAG链        │
│ • 文档管理      │    │ • 提示词管理     │
│ • 监控面板      │    │ • 上下文压缩     │
└─────────────────┘    └─────────────────┘
```

## 📊 性能特性

### 处理能力
- **文档处理**: 1000+文档/分钟
- **向量检索**: <100ms响应时间
- **RAG生成**: <3秒端到端响应
- **并发支持**: 100+并发请求

### 扩展性
- **文档规模**: 支持百万级文档
- **存储后端**: 可扩展到云端向量数据库
- **计算资源**: 支持GPU加速
- **分布式**: 支持集群部署

## 🔧 配置说明

### 主要配置文件

```yaml
# config/development.yaml
app:
  name: "RAG Knowledge System"
  environment: "development"
  debug: true

document_processor:
  chunk_size: 1000
  chunk_overlap: 200
  supported_formats: [".pdf", ".docx", ".txt", ".md"]

vectorstore:
  type: "chroma"
  embedding_provider: "openai"
  embedding_model: "text-embedding-ada-002"

retrieval:
  default_k: 5
  similarity_threshold: 0.7
  enable_hybrid: true

generation:
  llm_provider: "openai"
  llm_model: "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 1000
```

### 环境变量

```bash
# .env
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
PINECONE_API_KEY=your_pinecone_api_key

RAG_ENVIRONMENT=development
RAG_LOG_LEVEL=INFO
RAG_DATA_DIR=./data
RAG_CACHE_DIR=./cache
```

## 🧪 测试与评估

### 运行测试
```bash
# 单元测试
pytest tests/unit/ -v

# 集成测试
pytest tests/integration/ -v

# 性能测试
pytest tests/performance/ -v

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 评估指标
- **检索质量**: Precision@K, Recall@K, MRR
- **生成质量**: BLEU, ROUGE, BERTScore
- **系统性能**: 响应时间, 吞吐量, 资源使用
- **用户体验**: 答案相关性, 可读性, 满意度

## 🔍 监控运维

### 系统监控
- **性能指标**: Prometheus + Grafana
- **日志管理**: ELK Stack
- **健康检查**: /health 端点
- **告警系统**: 异常检测和通知

### 故障排除
```bash
# 查看系统状态
curl http://localhost:8000/health

# 检查日志
tail -f logs/rag_system.log

# 性能分析
python -m cProfile -o profile.stats demo.py
```

## 🤝 贡献指南

### 开发流程
1. Fork 项目
2. 创建特性分支
3. 提交代码变更
4. 编写测试用例
5. 提交Pull Request

### 代码规范
- 遵循PEP 8
- 添加类型注解
- 编写文档字符串
- 单元测试覆盖率 > 80%

## 📝 更新日志

### v0.1.0 (2024-01-XX)
- ✨ 初始版本发布
- 🔧 文档处理模块
- 💾 向量存储集成
- 🔍 混合检索实现
- 🤖 RAG生成链
- 🌐 RESTful API

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙋‍♂️ 支持

- 📖 [文档](docs/)
- 🐛 [问题反馈](issues/)
- 💬 [讨论区](discussions/)
- 📧 Email: support@example.com

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**