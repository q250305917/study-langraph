# LangChain Learning 模板系统

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![LangChain Version](https://img.shields.io/badge/langchain-0.1+-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

这是一个完整的LangChain学习模板系统，提供了参数化的示例代码模板，覆盖LangChain的所有核心应用场景。通过这些模板，学习者可以快速理解和应用不同的LangChain功能，同时获得可复用的代码结构。

## 📋 目录

- [系统概览](#系统概览)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [模板类型](#模板类型)
- [配置系统](#配置系统)
- [使用示例](#使用示例)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)

## 🎯 系统概览

### 核心特性

- **📝 参数化模板**: 所有模板都支持灵活的参数配置
- **🔄 统一接口**: 一致的setup、execute、get_example方法
- **⚡ 高性能**: 支持异步执行和缓存机制
- **🛡️ 类型安全**: 完整的类型注解和参数验证
- **📊 监控支持**: 内置性能指标收集和执行历史
- **🔧 易于扩展**: 模块化设计，便于添加新模板

### 支持的LangChain功能

- **LLM调用**: OpenAI、Anthropic、本地模型等
- **提示工程**: 各种场景的提示词模板
- **链组合**: 顺序链、并行链、条件链等
- **智能代理**: ReAct、工具调用、规划代理等
- **数据处理**: 文档加载、文本分割、向量化
- **记忆系统**: 对话记忆、摘要记忆、向量记忆
- **评估工具**: 准确性评估、性能分析、成本控制

## 🚀 快速开始

### 环境要求

```bash
# Python版本要求
python >= 3.8

# 核心依赖
pip install langchain>=0.1.0
pip install langchain-openai>=0.1.0
pip install langchain-anthropic>=0.1.0
```

### 基础安装

```bash
# 克隆项目
git clone <repository-url>
cd study_langraph

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 第一个示例

```python
from templates.llm.openai_template import OpenAITemplate

# 创建LLM模板实例
template = OpenAITemplate()

# 配置参数
template.setup(
    api_key="your-openai-api-key",
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# 执行调用
result = template.run("介绍一下LangChain的主要特点")
print(result.content)
```

## 📁 目录结构

```
templates/
├── README.md                    # 本文档
├── __init__.py                  # 包初始化
├── base/                        # 基础架构
│   ├── __init__.py
│   ├── template_base.py         # 模板基类
│   ├── config_loader.py         # 配置加载器
│   └── parameter_validator.py   # 参数验证器
├── configs/                     # 配置文件
│   ├── default.yaml             # 默认配置
│   ├── development.yaml         # 开发环境配置
│   ├── production.yaml          # 生产环境配置
│   └── template_configs/        # 模板专用配置
│       ├── llm_template.yaml
│       ├── data_template.yaml
│       ├── chain_template.yaml
│       └── agent_template.yaml
├── llm/                         # LLM模板
│   ├── __init__.py
│   ├── openai_template.py       # OpenAI模板
│   ├── anthropic_template.py    # Anthropic模板
│   ├── local_llm_template.py    # 本地模型模板
│   └── multi_model_template.py  # 多模型对比模板
├── prompts/                     # 提示词模板
│   ├── __init__.py
│   ├── chat_template.py         # 对话模板
│   ├── completion_template.py   # 补全模板
│   ├── few_shot_template.py     # 少样本学习模板
│   └── role_playing_template.py # 角色扮演模板
├── chains/                      # 链组合模板
│   ├── __init__.py
│   ├── sequential_chain.py      # 顺序链模板
│   ├── parallel_chain.py        # 并行链模板
│   ├── conditional_chain.py     # 条件链模板
│   └── pipeline_chain.py        # 管道链模板
├── agents/                      # 代理模板
│   ├── __init__.py
│   ├── react_agent.py          # ReAct代理模板
│   ├── tool_calling_agent.py   # 工具调用代理模板
│   ├── planning_agent.py       # 规划代理模板
│   └── multi_agent_template.py # 多代理协作模板
├── data/                        # 数据处理模板
│   ├── __init__.py
│   ├── document_loader.py       # 文档加载模板
│   ├── text_splitter.py        # 文本分割模板
│   ├── vectorstore_template.py # 向量存储模板
│   └── retrieval_template.py   # 检索模板
├── memory/                      # 记忆系统模板
│   ├── __init__.py
│   ├── conversation_memory.py   # 对话记忆模板
│   ├── summary_memory.py       # 摘要记忆模板
│   ├── vector_memory.py        # 向量记忆模板
│   └── knowledge_base.py       # 知识库模板
├── evaluation/                  # 评估模板
│   ├── __init__.py
│   ├── accuracy_eval.py        # 准确性评估模板
│   ├── performance_eval.py     # 性能评估模板
│   ├── cost_analysis.py        # 成本分析模板
│   └── ab_testing.py          # A/B测试模板
└── examples/                    # 使用示例
    ├── basic_examples/          # 基础示例
    ├── advanced_examples/       # 高级示例
    ├── tutorials/               # 教程
    └── best_practices/          # 最佳实践
```

## 🔧 模板类型

### 1. LLM模板 (`templates/llm/`)

用于调用各种大语言模型的模板。

**支持的模型**:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude系列)
- 本地模型 (Llama, ChatGLM等)

**核心功能**:
- 统一的模型调用接口
- 参数自动验证
- 错误重试机制
- 成本监控

**使用示例**:
```python
from templates.llm.openai_template import OpenAITemplate

template = OpenAITemplate()
template.setup(
    model_name="gpt-4",
    temperature=0.5,
    max_tokens=2000
)
result = template.run("解释量子计算的基本原理")
```

### 2. 提示词模板 (`templates/prompts/`)

管理各种应用场景的提示词模板。

**模板类型**:
- 对话模板: 多轮对话场景
- 补全模板: 文本补全任务
- 少样本模板: Few-shot学习
- 角色扮演: 特定角色设定

**特性**:
- 动态参数替换
- 模板继承和组合
- 上下文管理
- 格式验证

### 3. 链组合模板 (`templates/chains/`)

实现复杂的工作流组合。

**链类型**:
- **顺序链**: 步骤依次执行
- **并行链**: 同时执行多个步骤
- **条件链**: 根据条件选择执行路径
- **管道链**: 数据流式处理

**使用场景**:
- 文档处理流水线
- 多步骤推理任务
- 内容生成工作流
- 数据分析管道

### 4. 代理模板 (`templates/agents/`)

智能代理实现，能够自主决策和使用工具。

**代理类型**:
- **ReAct代理**: 推理-行动循环
- **工具调用代理**: 直接工具调用
- **规划代理**: 先规划后执行
- **多代理协作**: 代理间协作

**核心能力**:
- 自主推理决策
- 工具选择和使用
- 任务分解执行
- 错误处理恢复

### 5. 数据处理模板 (`templates/data/`)

处理各种数据源和格式。

**功能模块**:
- **文档加载器**: 支持PDF、Word、HTML等
- **文本分割器**: 智能文本分块
- **向量存储**: 多种向量数据库
- **检索模板**: 语义搜索和过滤

**支持格式**:
- 文档: PDF, DOCX, TXT, HTML, MD
- 数据: CSV, JSON, XML
- 网页: HTML, XML, RSS

### 6. 记忆系统模板 (`templates/memory/`)

管理对话历史和知识状态。

**记忆类型**:
- **缓冲记忆**: 完整对话历史
- **摘要记忆**: 压缩历史信息
- **向量记忆**: 语义相似性检索
- **知识库**: 结构化知识存储

### 7. 评估模板 (`templates/evaluation/`)

评估系统性能和质量。

**评估维度**:
- **准确性评估**: 答案正确性
- **性能评估**: 执行速度和资源使用
- **成本分析**: API调用成本
- **A/B测试**: 对比实验

## ⚙️ 配置系统

### 配置文件层级

1. **全局默认配置** (`configs/default.yaml`)
2. **环境特定配置** (`configs/development.yaml`, `configs/production.yaml`)
3. **模板专用配置** (`configs/template_configs/`)

### 配置优先级

```
模板专用配置 > 环境配置 > 全局默认配置
```

### 环境变量支持

配置文件支持环境变量引用：

```yaml
openai:
  api_key: "${OPENAI_API_KEY}"
  organization: "${OPENAI_ORG_ID}"
```

### 开发环境配置示例

```yaml
# development.yaml 特点
global:
  debug_mode: true
  log_level: "DEBUG"
  cache:
    enabled: false  # 开发时禁用缓存
llm:
  openai:
    model: "gpt-3.5-turbo"  # 使用便宜的模型
data:
  chunk_size: 500  # 较小的块便于调试
```

### 生产环境配置示例

```yaml
# production.yaml 特点
global:
  debug_mode: false
  log_level: "INFO"
  cache:
    enabled: true
    backend: "redis"  # 使用Redis缓存
llm:
  openai:
    model: "gpt-4"  # 使用更强的模型
security:
  api_key_required: true
  rate_limiting: true
```

## 📚 使用示例

### 示例1: 文档问答系统

```python
from templates.data.document_loader import DocumentLoaderTemplate
from templates.data.vectorstore_template import VectorStoreTemplate
from templates.chains.sequential_chain import SequentialChainTemplate
from templates.llm.openai_template import OpenAITemplate

# 1. 加载文档
doc_loader = DocumentLoaderTemplate()
doc_loader.setup(
    file_path="./documents/company_handbook.pdf",
    chunk_size=1000,
    chunk_overlap=100
)
documents = doc_loader.run()

# 2. 创建向量存储
vectorstore = VectorStoreTemplate()
vectorstore.setup(
    documents=documents,
    embedding_model="text-embedding-ada-002",
    vectorstore_type="chroma",
    collection_name="company_docs"
)
vectorstore.run()

# 3. 创建问答链
llm = OpenAITemplate()
llm.setup(model_name="gpt-4", temperature=0.1)

qa_chain = SequentialChainTemplate()
qa_chain.setup(
    llm=llm.llm,
    steps=[
        {
            "name": "retrieval",
            "prompt": "基于文档内容回答问题: {question}",
            "output_key": "answer"
        }
    ],
    input_variables=["question"]
)

# 4. 使用问答系统
answer = qa_chain.run("公司的休假政策是什么？")
print(answer)
```

### 示例2: 多模型对比分析

```python
from templates.llm.multi_model_template import MultiModelTemplate

# 创建多模型对比模板
multi_model = MultiModelTemplate()
multi_model.setup(
    models=[
        {
            "name": "gpt-4",
            "provider": "openai",
            "config": {"temperature": 0.7}
        },
        {
            "name": "claude-3-sonnet",
            "provider": "anthropic",
            "config": {"temperature": 0.7}
        }
    ]
)

# 对比分析
prompt = "分析人工智能对教育行业的影响"
results = multi_model.run(prompt)

for model_name, result in results.items():
    print(f"=== {model_name} ===")
    print(result.content)
    print(f"执行时间: {result.execution_time}秒")
    print(f"Token使用: {result.token_usage}")
    print()
```

### 示例3: 智能代理工作流

```python
from templates.agents.react_agent import ReactAgentTemplate
from langchain.tools import Calculator, Wikipedia

# 创建工具列表
tools = [
    Calculator(),
    Wikipedia()
]

# 创建ReAct代理
agent = ReactAgentTemplate()
agent.setup(
    tools=tools,
    llm_config={
        "model_name": "gpt-4",
        "temperature": 0.1
    },
    max_iterations=10,
    verbose=True
)

# 执行复杂任务
task = """
帮我分析一下：
1. 查询爱因斯坦的出生年份
2. 计算他如果还活着现在多少岁
3. 分析他对现代物理学的主要贡献
"""

result = agent.run(task)
print(result)
```

### 示例4: 批量数据处理

```python
from templates.data.text_splitter import TextSplitterTemplate
from templates.evaluation.performance_eval import PerformanceEvalTemplate
import asyncio

async def batch_process_documents():
    # 创建文本分割模板
    splitter = TextSplitterTemplate()
    splitter.setup(
        splitter_type="recursive",
        chunk_size=1000,
        chunk_overlap=100,
        parallel_processing=True,
        max_workers=4
    )
    
    # 批量处理文档
    document_paths = [
        "./docs/doc1.pdf",
        "./docs/doc2.pdf", 
        "./docs/doc3.pdf"
    ]
    
    # 性能评估
    perf_eval = PerformanceEvalTemplate()
    perf_eval.setup(benchmark_enabled=True)
    
    with perf_eval.measure("batch_processing"):
        results = await splitter.run_async(document_paths)
    
    # 输出性能报告
    report = perf_eval.generate_report()
    print(report)
    
    return results

# 运行异步处理
results = asyncio.run(batch_process_documents())
```

## 🎯 最佳实践

### 1. 模板选择指南

**选择LLM模板时**:
- 开发阶段: 使用`gpt-3.5-turbo`降低成本
- 生产环境: 使用`gpt-4`保证质量  
- 本地部署: 使用`local_llm_template`

**选择链类型时**:
- 简单任务: 使用单个LLM模板
- 顺序依赖: 使用`sequential_chain`
- 独立并行: 使用`parallel_chain`
- 条件分支: 使用`conditional_chain`

**选择代理类型时**:
- 需要推理: 使用`react_agent`
- 效率优先: 使用`tool_calling_agent`
- 复杂任务: 使用`planning_agent`

### 2. 性能优化

**缓存策略**:
```python
# 启用模板级别缓存
template.setup(
    cache_enabled=True,
    cache_ttl=1800  # 30分钟
)

# 全局缓存配置
from templates.base.config_loader import ConfigLoader
config = ConfigLoader()
config.enable_global_cache("redis")
```

**异步执行**:
```python
# 使用异步执行提高并发性能
results = await template.run_async(input_data)

# 批量异步处理
tasks = [template.run_async(data) for data in batch_data]
results = await asyncio.gather(*tasks)
```

**资源管理**:
```python
# 设置资源限制
template.setup(
    max_memory_usage=1024,  # 1GB内存限制
    timeout=60.0,           # 60秒超时
    max_workers=4           # 最大并发数
)
```

### 3. 错误处理

**重试机制**:
```python
template.setup(
    retry_count=3,
    retry_delay=1.0,
    error_handling="graceful"
)
```

**监控和日志**:
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 性能监控
template.setup(enable_metrics=True)
metrics = template.get_metrics()
print(f"成功率: {metrics['success_rate']:.2%}")
print(f"平均执行时间: {metrics['avg_execution_time']:.2f}秒")
```

### 4. 安全最佳实践

**API密钥管理**:
```bash
# 使用环境变量
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# 或使用.env文件
echo "OPENAI_API_KEY=sk-..." > .env
```

**输入验证**:
```python
# 启用严格参数验证
template.setup(
    parameter_validation="strict",
    sanitize_input=True,
    max_input_length=10000
)
```

**权限控制**:
```python
# 代理权限限制
agent.setup(
    allowed_tools=["calculator", "search"],  # 白名单
    forbidden_tools=["file_delete"],         # 黑名单
    sandbox_mode=True                        # 沙箱模式
)
```

### 5. 测试策略

**单元测试**:
```python
import unittest
from templates.llm.openai_template import OpenAITemplate

class TestOpenAITemplate(unittest.TestCase):
    def setUp(self):
        self.template = OpenAITemplate()
        
    def test_basic_setup(self):
        self.template.setup(
            model_name="gpt-3.5-turbo",
            temperature=0.5
        )
        self.assertEqual(self.template.config.model_name, "gpt-3.5-turbo")
        
    def test_example_execution(self):
        example = self.template.get_example()
        # 使用示例参数进行测试
        self.template.setup(**example["setup_parameters"])
        result = self.template.run(example["execute_parameters"]["input"])
        self.assertIsNotNone(result)
```

**集成测试**:
```python
# 端到端测试
def test_document_qa_pipeline():
    # 测试完整的文档问答流程
    doc_loader = DocumentLoaderTemplate()
    vectorstore = VectorStoreTemplate()
    qa_chain = SequentialChainTemplate()
    
    # 执行完整流程
    documents = doc_loader.run("test_document.pdf")
    vectorstore.run(documents)
    answer = qa_chain.run("测试问题")
    
    assert len(answer) > 0
```

## 🔍 故障排除

### 常见问题及解决方案

#### 1. API密钥错误

**错误信息**: `AuthenticationError: Invalid API key`

**解决方案**:
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 重新设置API密钥
export OPENAI_API_KEY="your-correct-api-key"

# 验证密钥格式
python -c "import os; print(len(os.getenv('OPENAI_API_KEY', '')))"
```

#### 2. 内存不足错误

**错误信息**: `MemoryError: Unable to allocate memory`

**解决方案**:
```python
# 减少批处理大小
template.setup(
    batch_size=10,          # 减少批次大小
    chunk_size=500,         # 减少块大小
    max_memory_usage=512    # 限制内存使用
)

# 启用流式处理
template.setup(streaming=True)
```

#### 3. 超时错误

**错误信息**: `TimeoutError: Request timed out`

**解决方案**:
```python
# 增加超时时间
template.setup(timeout=120.0)

# 启用重试机制
template.setup(
    retry_count=3,
    retry_delay=2.0
)

# 检查网络连接
import requests
response = requests.get("https://api.openai.com/v1/models", timeout=10)
print(response.status_code)
```

#### 4. 依赖包冲突

**错误信息**: `ImportError: No module named 'xxx'`

**解决方案**:
```bash
# 重新安装依赖
pip install --upgrade langchain langchain-openai langchain-anthropic

# 检查版本兼容性
pip list | grep langchain

# 使用虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

#### 5. 配置文件错误

**错误信息**: `ConfigurationError: Invalid configuration`

**解决方案**:
```python
# 验证配置文件语法
import yaml
with open("configs/development.yaml", "r") as f:
    config = yaml.safe_load(f)
    print("配置文件语法正确")

# 使用配置验证工具
from templates.base.config_loader import ConfigLoader
loader = ConfigLoader()
is_valid = loader.validate_config("configs/development.yaml")
print(f"配置有效性: {is_valid}")
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 启用模板调试模式
template.setup(
    debug_mode=True,
    verbose=True,
    log_intermediate_steps=True
)
```

#### 2. 使用性能分析器

```python
from templates.evaluation.performance_eval import PerformanceEvalTemplate

perf_eval = PerformanceEvalTemplate()
perf_eval.setup(
    profiling_enabled=True,
    memory_profiling=True
)

with perf_eval.profile("template_execution"):
    result = template.run(input_data)

# 查看性能报告
print(perf_eval.get_profile_report())
```

#### 3. 执行历史分析

```python
# 查看执行历史
history = template.execution_history
for record in history[-5:]:  # 最近5次执行
    print(f"执行ID: {record['execution_id']}")
    print(f"执行时间: {record['execution_time']:.2f}秒")
    print(f"成功状态: {record['success']}")
    if not record['success']:
        print(f"错误信息: {record.get('error', 'Unknown')}")
```

### 性能调优指南

#### 1. 监控关键指标

```python
# 获取性能指标
metrics = template.get_metrics()
print(f"总执行次数: {metrics['total_executions']}")
print(f"成功率: {metrics['success_rate']:.2%}")
print(f"平均执行时间: {metrics['avg_execution_time']:.2f}秒")
print(f"最大执行时间: {metrics['max_execution_time']:.2f}秒")
```

#### 2. 资源使用优化

```python
# CPU密集型任务优化
template.setup(
    max_workers=4,              # 基于CPU核心数
    parallel_processing=True,
    batch_size=50
)

# 内存密集型任务优化
template.setup(
    max_memory_usage=1024,      # 1GB限制
    enable_streaming=True,
    chunk_processing=True
)

# I/O密集型任务优化
template.setup(
    async_enabled=True,
    connection_pool_size=10,
    request_timeout=30.0
)
```

#### 3. 缓存策略优化

```python
# 智能缓存配置
template.setup(
    cache_enabled=True,
    cache_strategy="lru",       # LRU策略
    cache_size=1000,           # 缓存项数量
    cache_ttl=3600,            # 1小时过期
    cache_compression=True     # 启用压缩
)

# 缓存命中率监控
cache_stats = template.get_cache_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
```

## 🤝 贡献指南

### 开发环境设置

```bash
# 1. Fork并克隆仓库
git clone https://github.com/your-username/study-langraph.git
cd study-langraph

# 2. 创建开发分支
git checkout -b feature/new-template

# 3. 安装开发依赖
pip install -r requirements-dev.txt

# 4. 安装pre-commit hooks
pre-commit install
```

### 添加新模板

1. **创建模板文件**:
```python
# templates/custom/my_template.py
from templates.base.template_base import TemplateBase
from typing import Dict, Any

class MyTemplate(TemplateBase):
    def setup(self, **parameters) -> None:
        # 实现参数设置逻辑
        pass
    
    def execute(self, input_data, **kwargs):
        # 实现核心执行逻辑
        pass
    
    def get_example(self) -> Dict[str, Any]:
        # 返回使用示例
        return {
            "setup_parameters": {},
            "execute_parameters": {},
            "expected_output": {}
        }
```

2. **创建配置文件**:
```yaml
# templates/configs/template_configs/my_template.yaml
name: "MyTemplate"
version: "1.0.0"
description: "我的自定义模板"
template_type: "custom"

parameters:
  my_param:
    type: "str"
    required: true
    description: "示例参数"
```

3. **编写测试**:
```python
# tests/templates/custom/test_my_template.py
import unittest
from templates.custom.my_template import MyTemplate

class TestMyTemplate(unittest.TestCase):
    def test_basic_functionality(self):
        template = MyTemplate()
        template.setup(my_param="test_value")
        result = template.run("test_input")
        self.assertIsNotNone(result)
```

4. **更新文档**:
```markdown
# 在README.md中添加新模板说明
### 自定义模板 (`templates/custom/`)
...
```

### 代码规范

- **类型注解**: 所有函数都应有完整的类型注解
- **文档字符串**: 使用Google风格的docstring
- **错误处理**: 使用项目定义的异常类型
- **日志记录**: 使用项目的日志系统
- **测试覆盖**: 新代码应有>=90%的测试覆盖率

### 提交规范

```bash
# 提交信息格式
git commit -m "feat(templates): 添加新的自定义模板

- 实现MyTemplate类
- 添加配置文件和测试
- 更新文档

closes #123"
```

**提交类型**:
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### Pull Request流程

1. 确保所有测试通过
2. 更新相关文档
3. 遵循代码规范
4. 提供清晰的PR描述
5. 响应代码审查反馈

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的LLM应用开发框架
- [OpenAI](https://openai.com/) - GPT模型提供商
- [Anthropic](https://www.anthropic.com/) - Claude模型提供商
- 所有贡献者和社区成员

## 📞 支持与反馈

- **Issue跟踪**: [GitHub Issues](https://github.com/your-username/study-langraph/issues)
- **讨论论坛**: [GitHub Discussions](https://github.com/your-username/study-langraph/discussions)
- **邮箱联系**: langchain-learning@example.com

---

**快速链接**:
- [安装指南](#快速开始)
- [API文档](docs/api_reference.md)
- [教程示例](examples/tutorials/)
- [最佳实践](examples/best_practices/)
- [FAQ](docs/faq.md)