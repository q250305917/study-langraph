# 性能优化最佳实践指南

本指南提供了优化LangChain Learning模板系统性能的实用建议和技巧。

## 🎯 优化目标

- **响应速度**: 减少用户等待时间
- **资源效率**: 优化内存和CPU使用
- **成本控制**: 降低API调用费用
- **系统稳定性**: 提高系统可靠性

## 🚀 LLM性能优化

### 1. 模型选择策略

```python
# 根据任务复杂度选择模型
def choose_model_by_task(task_type: str) -> str:
    """根据任务类型选择最适合的模型"""
    
    model_mapping = {
        # 简单任务 - 快速且便宜
        "translation": "gpt-3.5-turbo",
        "summarization": "gpt-3.5-turbo", 
        "classification": "gpt-3.5-turbo",
        
        # 复杂任务 - 质量优先
        "reasoning": "gpt-4",
        "creative_writing": "gpt-4",
        "code_generation": "gpt-4",
        
        # 对话任务 - 平衡性能和成本
        "chat": "gpt-3.5-turbo",
        "customer_service": "gpt-3.5-turbo"
    }
    
    return model_mapping.get(task_type, "gpt-3.5-turbo")

# 示例使用
llm_template = OpenAITemplate()
llm_template.setup(
    model_name=choose_model_by_task("translation"),
    temperature=0.3  # 翻译任务使用较低温度
)
```

### 2. 参数优化

```python
# 针对不同场景的参数配置
class OptimizedConfigs:
    """优化的配置集合"""
    
    @staticmethod
    def fast_response_config():
        """快速响应配置 - 牺牲一些质量换取速度"""
        return {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": 500,
            "top_p": 0.8,
            "frequency_penalty": 0.1
        }
    
    @staticmethod
    def high_quality_config():
        """高质量配置 - 重视输出质量"""
        return {
            "model_name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.95,
            "frequency_penalty": 0.0
        }
    
    @staticmethod
    def cost_effective_config():
        """成本效益配置 - 平衡成本和质量"""
        return {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.5,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.2
        }

# 根据场景选择配置
def setup_llm_for_scenario(scenario: str):
    llm = OpenAITemplate()
    
    if scenario == "real_time_chat":
        config = OptimizedConfigs.fast_response_config()
    elif scenario == "content_creation":
        config = OptimizedConfigs.high_quality_config()
    else:
        config = OptimizedConfigs.cost_effective_config()
    
    llm.setup(**config)
    return llm
```

### 3. 缓存策略

```python
from functools import lru_cache
import hashlib
import pickle
import time

class SmartCache:
    """智能缓存系统"""
    
    def __init__(self, ttl=3600, max_size=1000):
        self.cache = {}
        self.ttl = ttl
        self.max_size = max_size
        self.access_times = {}
    
    def _generate_key(self, prompt: str, config: dict) -> str:
        """生成缓存键"""
        content = f"{prompt}_{str(sorted(config.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, config: dict):
        """获取缓存结果"""
        key = self._generate_key(prompt, config)
        
        if key in self.cache:
            cached_time, result = self.cache[key]
            
            # 检查是否过期
            if time.time() - cached_time < self.ttl:
                self.access_times[key] = time.time()
                return result
            else:
                # 清理过期缓存
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def set(self, prompt: str, config: dict, result):
        """设置缓存"""
        key = self._generate_key(prompt, config)
        
        # 如果缓存已满，删除最久未访问的项
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = (time.time(), result)
        self.access_times[key] = time.time()

# 使用缓存的LLM模板
class CachedLLMTemplate(OpenAITemplate):
    """带缓存的LLM模板"""
    
    def __init__(self, cache_ttl=3600):
        super().__init__()
        self.cache = SmartCache(ttl=cache_ttl)
    
    def run(self, input_data: str, **kwargs):
        """带缓存的执行方法"""
        
        # 生成配置键
        config_key = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # 尝试从缓存获取
        cached_result = self.cache.get(input_data, config_key)
        if cached_result:
            return cached_result
        
        # 执行LLM调用
        result = super().run(input_data, **kwargs)
        
        # 保存到缓存
        self.cache.set(input_data, config_key, result)
        
        return result
```

### 4. 批处理优化

```python
import asyncio
from typing import List, Dict, Any

class BatchProcessor:
    """批处理器 - 优化多个请求的处理"""
    
    def __init__(self, llm_template, batch_size=5, delay=0.1):
        self.llm_template = llm_template
        self.batch_size = batch_size
        self.delay = delay  # 请求间延迟，避免速率限制
    
    async def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """异步批处理"""
        results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            
            # 创建异步任务
            tasks = []
            for prompt in batch:
                task = self._process_single(prompt)
                tasks.append(task)
            
            # 等待批次完成
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "input": batch[j],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    results.append({
                        "input": batch[j],
                        "success": True,
                        "output": result
                    })
            
            # 添加延迟避免速率限制
            if i + self.batch_size < len(prompts):
                await asyncio.sleep(self.delay)
        
        return results
    
    async def _process_single(self, prompt: str):
        """处理单个请求"""
        return await self.llm_template.run_async(prompt)

# 使用示例
async def batch_processing_example():
    llm = OpenAITemplate()
    llm.setup(model_name="gpt-3.5-turbo", async_enabled=True)
    
    processor = BatchProcessor(llm, batch_size=3)
    
    prompts = [
        "介绍Python的特点",
        "解释机器学习概念", 
        "描述云计算优势",
        "分析人工智能发展",
        "讨论区块链技术"
    ]
    
    results = await processor.process_batch(prompts)
    
    for result in results:
        if result["success"]:
            print(f"✅ {result['input'][:20]}... -> 成功")
        else:
            print(f"❌ {result['input'][:20]}... -> {result['error']}")
```

## 📊 数据处理优化

### 1. 文档加载优化

```python
class OptimizedDocumentLoader:
    """优化的文档加载器"""
    
    def __init__(self):
        self.file_size_limits = {
            "txt": 50 * 1024 * 1024,   # 50MB
            "pdf": 20 * 1024 * 1024,   # 20MB
            "docx": 10 * 1024 * 1024   # 10MB
        }
    
    def should_process_file(self, file_path: str) -> bool:
        """检查文件是否应该处理"""
        import os
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1][1:].lower()
        
        max_size = self.file_size_limits.get(file_ext, 5 * 1024 * 1024)
        
        if file_size > max_size:
            print(f"⚠️ 跳过大文件: {file_path} ({file_size / 1024 / 1024:.1f}MB)")
            return False
        
        return True
    
    def load_with_streaming(self, file_path: str, chunk_size=8192):
        """流式加载大文件"""
        content = ""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content += chunk
                    
                    # 定期让出控制权
                    if len(content) % (chunk_size * 10) == 0:
                        time.sleep(0.001)  # 1ms延迟
                        
        except Exception as e:
            print(f"❌ 加载文件失败: {file_path} - {str(e)}")
            return None
        
        return content

# 并行文档处理
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def process_document_parallel(file_paths: List[str], max_workers=None):
    """并行处理多个文档"""
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(file_paths))
    
    results = []
    
    # 使用进程池处理CPU密集型任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(load_and_process_file, file_path): file_path 
            for file_path in file_paths
        }
        
        for future in future_to_file:
            file_path = future_to_file[future]
            try:
                result = future.result(timeout=60)  # 60秒超时
                results.append(result)
            except Exception as e:
                print(f"❌ 处理文件失败: {file_path} - {str(e)}")
    
    return results

def load_and_process_file(file_path: str):
    """单个文件处理函数（在子进程中运行）"""
    loader = OptimizedDocumentLoader()
    
    if not loader.should_process_file(file_path):
        return None
    
    content = loader.load_with_streaming(file_path)
    
    if content:
        # 基础预处理
        content = content.strip()
        content = ' '.join(content.split())  # 标准化空白字符
        
        return {
            "file_path": file_path,
            "content": content,
            "size": len(content)
        }
    
    return None
```

### 2. 向量化优化

```python
class OptimizedVectorStore:
    """优化的向量存储"""
    
    def __init__(self):
        self.batch_size = 100
        self.embedding_cache = {}
    
    def embed_texts_efficiently(self, texts: List[str], embedding_model):
        """高效的文本嵌入"""
        embeddings = []
        
        # 检查缓存
        cached_embeddings = {}
        texts_to_embed = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                cached_embeddings[i] = self.embedding_cache[text_hash]
            else:
                texts_to_embed.append((i, text, text_hash))
        
        # 批量处理未缓存的文本
        if texts_to_embed:
            batch_texts = [item[1] for item in texts_to_embed]
            
            # 分批处理
            for i in range(0, len(batch_texts), self.batch_size):
                batch = batch_texts[i:i + self.batch_size]
                
                try:
                    batch_embeddings = embedding_model.embed_documents(batch)
                    
                    # 更新缓存和结果
                    for j, embedding in enumerate(batch_embeddings):
                        original_index = texts_to_embed[i + j][0]
                        text_hash = texts_to_embed[i + j][2]
                        
                        embeddings.append((original_index, embedding))
                        self.embedding_cache[text_hash] = embedding
                        
                except Exception as e:
                    print(f"❌ 嵌入批次失败: {str(e)}")
        
        # 合并缓存和新计算的嵌入
        final_embeddings = [None] * len(texts)
        
        for index, embedding in cached_embeddings.items():
            final_embeddings[index] = embedding
        
        for index, embedding in embeddings:
            final_embeddings[index] = embedding
        
        return final_embeddings
    
    def optimize_index(self, vectorstore, force_rebuild=False):
        """优化向量索引"""
        
        # 检查索引健康状况
        stats = vectorstore.get_statistics()
        
        if force_rebuild or self._should_rebuild_index(stats):
            print("🔄 重建向量索引以优化性能...")
            
            # 备份现有数据
            backup_data = vectorstore.export_data()
            
            # 重建索引
            vectorstore.rebuild_index(optimize=True)
            
            print("✅ 索引重建完成")
        
        return vectorstore
    
    def _should_rebuild_index(self, stats: dict) -> bool:
        """判断是否需要重建索引"""
        
        # 删除比例过高
        if stats.get("deleted_ratio", 0) > 0.3:
            return True
        
        # 索引碎片化严重
        if stats.get("fragmentation", 0) > 0.5:
            return True
        
        # 查询性能下降
        if stats.get("avg_query_time", 0) > 1.0:  # 超过1秒
            return True
        
        return False
```

### 3. 内存管理优化

```python
import gc
import psutil
import threading
from typing import Optional

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb=2048):
        self.max_memory_mb = max_memory_mb
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def start_monitoring(self, check_interval=10):
        """开始内存监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_memory(self, check_interval: int):
        """内存监控循环"""
        while self.monitoring:
            usage = self.get_memory_usage()
            
            if usage > self.max_memory_mb * 0.8:  # 80%警告
                print(f"⚠️ 内存使用率高: {usage:.1f}MB / {self.max_memory_mb}MB")
                self._cleanup_memory()
            
            if usage > self.max_memory_mb:  # 超出限制
                print(f"🚨 内存超出限制: {usage:.1f}MB / {self.max_memory_mb}MB")
                self._force_cleanup()
            
            time.sleep(check_interval)
    
    def _cleanup_memory(self):
        """清理内存"""
        print("🧹 执行内存清理...")
        
        # 强制垃圾回收
        collected = gc.collect()
        print(f"🗑️ 回收了 {collected} 个对象")
        
        # 清理全局缓存
        if hasattr(self, 'global_cache'):
            self.global_cache.clear()
    
    def _force_cleanup(self):
        """强制内存清理"""
        self._cleanup_memory()
        
        # 更激进的清理策略
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux only

# 使用上下文管理器自动管理内存
class MemoryAwareTemplate:
    """内存感知的模板基类"""
    
    def __init__(self, max_memory_mb=1024):
        self.memory_manager = MemoryManager(max_memory_mb)
    
    def __enter__(self):
        self.memory_manager.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_manager.stop_monitoring()
        self.memory_manager._cleanup_memory()

# 使用示例
with MemoryAwareTemplate(max_memory_mb=512) as template:
    # 在内存监控下执行操作
    result = template.process_large_dataset()
```

## 🔄 系统级优化

### 1. 连接池管理

```python
import aiohttp
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests

class OptimizedHTTPClient:
    """优化的HTTP客户端"""
    
    def __init__(self, max_connections=100, max_connections_per_host=20):
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        # 配置连接适配器
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_connections_per_host,
            pool_maxsize=max_connections
        )
        
        # 创建会话
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置默认超时
        self.session.timeout = (5, 30)  # 连接超时5秒，读取超时30秒
    
    async def create_async_session(self):
        """创建异步会话"""
        connector = aiohttp.TCPConnector(
            limit=100,          # 总连接数限制
            limit_per_host=20,  # 每个主机连接数限制
            ttl_dns_cache=300,  # DNS缓存5分钟
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,           # 总超时
            connect=5           # 连接超时
        )
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )

# LLM模板的HTTP优化
class OptimizedOpenAITemplate(OpenAITemplate):
    """优化的OpenAI模板"""
    
    def __init__(self):
        super().__init__()
        self.http_client = OptimizedHTTPClient()
    
    def setup(self, **parameters):
        super().setup(**parameters)
        
        # 使用优化的HTTP客户端
        if hasattr(self.llm, '_client'):
            self.llm._client.session = self.http_client.session
```

### 2. 异步优化

```python
import asyncio
from typing import List, Coroutine, Any
import aiofiles

class AsyncOptimizer:
    """异步操作优化器"""
    
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(self, coro: Coroutine) -> Any:
        """使用信号量限制并发"""
        async with self.semaphore:
            return await coro
    
    async def batch_execute(self, coroutines: List[Coroutine]) -> List[Any]:
        """批量执行协程"""
        
        # 创建受限制的任务
        limited_coroutines = [
            self.run_with_semaphore(coro) 
            for coro in coroutines
        ]
        
        # 使用gather执行，但处理异常
        results = await asyncio.gather(*limited_coroutines, return_exceptions=True)
        
        # 分离成功和失败的结果
        successes = []
        failures = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append((i, result))
            else:
                successes.append((i, result))
        
        if failures:
            print(f"⚠️ {len(failures)} 个任务失败")
            for i, error in failures:
                print(f"   任务 {i}: {str(error)}")
        
        return results
    
    async def progressive_execution(self, coroutines: List[Coroutine], 
                                  batch_size=5, delay=1.0) -> List[Any]:
        """渐进式执行 - 避免突发负载"""
        
        all_results = []
        
        for i in range(0, len(coroutines), batch_size):
            batch = coroutines[i:i + batch_size]
            
            print(f"🚀 执行批次 {i//batch_size + 1}/{(len(coroutines)-1)//batch_size + 1}")
            
            batch_results = await self.batch_execute(batch)
            all_results.extend(batch_results)
            
            # 批次间延迟
            if i + batch_size < len(coroutines):
                await asyncio.sleep(delay)
        
        return all_results

# 异步文件I/O优化
class AsyncFileHandler:
    """异步文件处理器"""
    
    @staticmethod
    async def read_files_async(file_paths: List[str]) -> List[str]:
        """异步读取多个文件"""
        
        async def read_single_file(file_path: str) -> str:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
            except Exception as e:
                print(f"❌ 读取文件失败: {file_path} - {str(e)}")
                return ""
        
        tasks = [read_single_file(path) for path in file_paths]
        contents = await asyncio.gather(*tasks)
        
        return contents
    
    @staticmethod
    async def write_files_async(file_data: List[tuple]) -> None:
        """异步写入多个文件"""
        
        async def write_single_file(file_path: str, content: str) -> None:
            try:
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            except Exception as e:
                print(f"❌ 写入文件失败: {file_path} - {str(e)}")
        
        tasks = [write_single_file(path, content) for path, content in file_data]
        await asyncio.gather(*tasks)
```

## 📈 监控和分析

### 1. 性能指标收集

```python
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: float
    operation: str
    duration: float
    success: bool
    memory_usage: float
    error: Optional[str] = None

class PerformanceCollector:
    """性能指标收集器"""
    
    def __init__(self, max_history=1000):
        self.metrics: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_metric(self, operation: str, duration: float, 
                     success: bool, memory_usage: float, error: str = None):
        """记录性能指标"""
        
        metric = PerformanceMetric(
            timestamp=time.time(),
            operation=operation,
            duration=duration,
            success=success,
            memory_usage=memory_usage,
            error=error
        )
        
        with self.lock:
            self.metrics.append(metric)
            if success:
                self.operation_stats[operation].append(duration)
    
    def get_statistics(self, operation: str = None) -> Dict[str, Any]:
        """获取统计信息"""
        
        with self.lock:
            if operation:
                durations = self.operation_stats[operation]
            else:
                durations = [m.duration for m in self.metrics if m.success]
        
        if not durations:
            return {"count": 0}
        
        durations.sort()
        n = len(durations)
        
        return {
            "count": n,
            "avg_duration": sum(durations) / n,
            "min_duration": durations[0],
            "max_duration": durations[-1],
            "p50_duration": durations[n // 2],
            "p95_duration": durations[int(n * 0.95)],
            "p99_duration": durations[int(n * 0.99)]
        }
    
    def get_error_analysis(self) -> Dict[str, int]:
        """获取错误分析"""
        
        error_counts = defaultdict(int)
        
        with self.lock:
            for metric in self.metrics:
                if not metric.success and metric.error:
                    error_counts[metric.error] += 1
        
        return dict(error_counts)

# 性能监控装饰器
def monitor_performance(collector: PerformanceCollector, operation_name: str):
    """性能监控装饰器"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                duration = end_time - start_time
                memory_usage = end_memory - start_memory
                
                collector.record_metric(
                    operation=operation_name,
                    duration=duration,
                    success=success,
                    memory_usage=memory_usage,
                    error=error
                )
            
            return result
        return wrapper
    return decorator

# 使用示例
performance_collector = PerformanceCollector()

@monitor_performance(performance_collector, "llm_call")
def monitored_llm_call(template, prompt):
    """被监控的LLM调用"""
    return template.run(prompt)
```

### 2. 实时监控面板

```python
import json
import datetime
from typing import Dict, Any

class MonitoringDashboard:
    """监控面板"""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
    
    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        
        # 总体统计
        overall_stats = self.collector.get_statistics()
        
        # 操作级别统计
        operation_stats = {}
        for operation in self.collector.operation_stats.keys():
            operation_stats[operation] = self.collector.get_statistics(operation)
        
        # 错误分析
        error_analysis = self.collector.get_error_analysis()
        
        # 最近趋势（最近100个指标）
        recent_metrics = list(self.collector.metrics)[-100:]
        recent_durations = [m.duration for m in recent_metrics if m.success]
        
        # 成功率
        total_count = len(recent_metrics)
        success_count = sum(1 for m in recent_metrics if m.success)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_statistics": overall_stats,
            "operation_statistics": operation_stats,
            "error_analysis": error_analysis,
            "recent_trend": {
                "success_rate": success_rate,
                "avg_duration_recent": sum(recent_durations) / len(recent_durations) if recent_durations else 0,
                "total_requests": total_count
            }
        }
    
    def export_metrics(self, file_path: str):
        """导出指标到文件"""
        
        report = self.generate_report()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 监控报告已导出到: {file_path}")
    
    def print_summary(self):
        """打印监控摘要"""
        
        report = self.generate_report()
        
        print("\n" + "="*50)
        print("📊 性能监控摘要")
        print("="*50)
        
        overall = report["overall_statistics"]
        if overall["count"] > 0:
            print(f"总请求数: {overall['count']}")
            print(f"平均耗时: {overall['avg_duration']:.3f}秒")
            print(f"P95延迟: {overall['p95_duration']:.3f}秒")
            print(f"P99延迟: {overall['p99_duration']:.3f}秒")
        
        recent = report["recent_trend"]
        print(f"成功率: {recent['success_rate']:.2%}")
        print(f"最近平均耗时: {recent['avg_duration_recent']:.3f}秒")
        
        if report["error_analysis"]:
            print("\n❌ 错误统计:")
            for error, count in report["error_analysis"].items():
                print(f"   {error}: {count}次")
        
        print("\n📈 各操作统计:")
        for operation, stats in report["operation_statistics"].items():
            if stats["count"] > 0:
                print(f"   {operation}: {stats['count']}次, "
                      f"平均{stats['avg_duration']:.3f}秒")
```

## 🔧 配置优化

### 1. 环境特定优化

```python
# config_optimizer.py
import os
from typing import Dict, Any

class ConfigOptimizer:
    """配置优化器"""
    
    @staticmethod
    def get_optimized_config(environment: str = "development") -> Dict[str, Any]:
        """获取优化的配置"""
        
        base_config = {
            "llm": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 30.0
            },
            "data": {
                "chunk_size": 1000,
                "batch_size": 10
            },
            "cache": {
                "enabled": True,
                "ttl": 3600
            }
        }
        
        if environment == "development":
            return ConfigOptimizer._optimize_for_development(base_config)
        elif environment == "production":
            return ConfigOptimizer._optimize_for_production(base_config)
        elif environment == "testing":
            return ConfigOptimizer._optimize_for_testing(base_config)
        else:
            return base_config
    
    @staticmethod
    def _optimize_for_development(config: Dict[str, Any]) -> Dict[str, Any]:
        """开发环境优化"""
        
        # 优先调试和快速迭代
        config["llm"]["model_name"] = "gpt-3.5-turbo"  # 便宜的模型
        config["llm"]["max_tokens"] = 500              # 减少token使用
        config["data"]["chunk_size"] = 500             # 小块便于调试
        config["data"]["batch_size"] = 5               # 小批次减少等待
        config["cache"]["enabled"] = False             # 禁用缓存确保数据新鲜
        
        # 添加调试选项
        config["debug"] = {
            "verbose": True,
            "log_level": "DEBUG",
            "save_intermediate_results": True
        }
        
        return config
    
    @staticmethod
    def _optimize_for_production(config: Dict[str, Any]) -> Dict[str, Any]:
        """生产环境优化"""
        
        # 优先性能和稳定性
        config["llm"]["model_name"] = "gpt-4"          # 高质量模型
        config["llm"]["max_tokens"] = 2000             # 更多输出
        config["llm"]["timeout"] = 60.0                # 更长超时
        config["data"]["chunk_size"] = 1500            # 大块提高效率
        config["data"]["batch_size"] = 20              # 大批次提高吞吐
        config["cache"]["enabled"] = True              # 启用缓存
        config["cache"]["ttl"] = 7200                  # 更长缓存时间
        
        # 添加生产选项
        config["production"] = {
            "monitoring_enabled": True,
            "error_reporting": True,
            "performance_logging": True,
            "rate_limiting": True
        }
        
        return config
    
    @staticmethod
    def _optimize_for_testing(config: Dict[str, Any]) -> Dict[str, Any]:
        """测试环境优化"""
        
        # 优先速度和可预测性
        config["llm"]["model_name"] = "gpt-3.5-turbo"
        config["llm"]["temperature"] = 0.1             # 低随机性
        config["llm"]["max_tokens"] = 200              # 快速响应
        config["data"]["chunk_size"] = 300             # 小块快速处理
        config["data"]["batch_size"] = 3               # 小批次
        config["cache"]["enabled"] = False             # 禁用缓存确保一致性
        
        # 添加测试选项
        config["testing"] = {
            "mock_api_calls": True,
            "deterministic_results": True,
            "fast_mode": True
        }
        
        return config

# 自动配置调优
class AutoTuner:
    """自动配置调优器"""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
    
    def suggest_optimizations(self) -> Dict[str, str]:
        """基于性能数据建议优化"""
        
        suggestions = []
        stats = self.collector.get_statistics()
        
        if not stats or stats["count"] < 10:
            return {"message": "数据不足，需要更多性能数据"}
        
        # 分析响应时间
        if stats["avg_duration"] > 5.0:
            suggestions.append("🐌 平均响应时间过长，建议：")
            suggestions.append("   - 使用更快的模型(gpt-3.5-turbo)")
            suggestions.append("   - 减少max_tokens参数")
            suggestions.append("   - 启用缓存")
        
        if stats["p95_duration"] > 10.0:
            suggestions.append("⏰ P95延迟过高，建议：")
            suggestions.append("   - 优化网络连接")
            suggestions.append("   - 增加超时重试")
            suggestions.append("   - 考虑使用CDN")
        
        # 分析错误率
        error_analysis = self.collector.get_error_analysis()
        total_errors = sum(error_analysis.values())
        error_rate = total_errors / stats["count"]
        
        if error_rate > 0.05:  # 5%错误率
            suggestions.append("❌ 错误率过高，建议：")
            suggestions.append("   - 增加重试机制")
            suggestions.append("   - 改善错误处理")
            suggestions.append("   - 检查API配额")
        
        return {
            "suggestions": suggestions,
            "current_stats": stats,
            "error_analysis": error_analysis
        }
```

## 📋 优化检查清单

### 🔍 性能审查清单

使用以下清单定期审查你的系统性能：

```markdown
## LLM优化检查

- [ ] 根据任务选择合适的模型
- [ ] 优化temperature和max_tokens参数
- [ ] 实现智能缓存策略
- [ ] 使用批处理减少API调用
- [ ] 启用异步处理（如适用）
- [ ] 监控API使用量和成本
- [ ] 实现重试和错误处理

## 数据处理优化检查

- [ ] 设置合理的文件大小限制
- [ ] 优化文本分割参数
- [ ] 使用并行处理大批量数据
- [ ] 定期优化向量索引
- [ ] 监控内存使用情况
- [ ] 实现流式处理（如需要）

## 系统级优化检查

- [ ] 配置连接池和重试策略
- [ ] 优化数据库查询
- [ ] 使用CDN加速（如适用）
- [ ] 实现负载均衡（如需要）
- [ ] 监控系统资源使用
- [ ] 设置性能告警

## 监控和分析检查

- [ ] 收集关键性能指标
- [ ] 设置监控面板
- [ ] 定期分析性能趋势
- [ ] 实现错误追踪
- [ ] 建立性能基线
- [ ] 定期性能回顾
```

通过遵循这些最佳实践和使用提供的优化工具，你可以显著提升LangChain Learning模板系统的性能和效率。记住，优化是一个持续的过程，需要根据实际使用情况不断调整和改进。