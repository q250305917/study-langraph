# 故障排除指南

本指南提供了LangChain Learning模板系统常见问题的诊断和解决方案。

## 🚨 快速诊断工具

### 系统健康检查脚本

```python
#!/usr/bin/env python3
# health_check.py - 系统健康检查工具

import os
import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

class HealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.results = []
        self.errors = []
    
    def check_all(self) -> Dict[str, bool]:
        """执行所有检查"""
        
        checks = [
            ("Python版本", self.check_python_version),
            ("环境变量", self.check_environment_variables),
            ("必需依赖", self.check_required_packages),
            ("可选依赖", self.check_optional_packages),
            ("API连接", self.check_api_connectivity),
            ("文件权限", self.check_file_permissions),
            ("磁盘空间", self.check_disk_space),
            ("内存可用", self.check_memory)
        ]
        
        results = {}
        
        for check_name, check_func in checks:
            try:
                success, message = check_func()
                results[check_name] = success
                
                status = "✅" if success else "❌"
                print(f"{status} {check_name}: {message}")
                
                if not success:
                    self.errors.append(f"{check_name}: {message}")
                    
            except Exception as e:
                results[check_name] = False
                print(f"❌ {check_name}: 检查失败 - {str(e)}")
                self.errors.append(f"{check_name}: 检查失败 - {str(e)}")
        
        return results
    
    def check_python_version(self) -> Tuple[bool, str]:
        """检查Python版本"""
        version = sys.version_info
        
        if version.major >= 3 and version.minor >= 8:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python版本过低: {version.major}.{version.minor}.{version.micro} (需要3.8+)"
    
    def check_environment_variables(self) -> Tuple[bool, str]:
        """检查环境变量"""
        required_vars = ["OPENAI_API_KEY"]
        optional_vars = ["ANTHROPIC_API_KEY", "HUGGINGFACE_API_TOKEN"]
        
        missing_required = [var for var in required_vars if not os.getenv(var)]
        missing_optional = [var for var in optional_vars if not os.getenv(var)]
        
        if missing_required:
            return False, f"缺少必需环境变量: {', '.join(missing_required)}"
        
        message = "所有必需环境变量已设置"
        if missing_optional:
            message += f" (可选: {', '.join(missing_optional)} 未设置)"
        
        return True, message
    
    def check_required_packages(self) -> Tuple[bool, str]:
        """检查必需的Python包"""
        required_packages = [
            ("langchain", "0.1.0"),
            ("langchain_openai", "0.1.0"),
            ("openai", "1.0.0")
        ]
        
        missing_packages = []
        
        for package_name, min_version in required_packages:
            try:
                module = importlib.import_module(package_name)
                # 简单版本检查
                if hasattr(module, '__version__'):
                    version = module.__version__
                else:
                    version = "未知"
                
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            return False, f"缺少必需包: {', '.join(missing_packages)}"
        
        return True, "所有必需包已安装"
    
    def check_optional_packages(self) -> Tuple[bool, str]:
        """检查可选的Python包"""
        optional_packages = [
            "chromadb", "faiss-cpu", "sentence-transformers",
            "pypdf", "python-docx", "tiktoken"
        ]
        
        installed = []
        missing = []
        
        for package_name in optional_packages:
            try:
                importlib.import_module(package_name.replace('-', '_'))
                installed.append(package_name)
            except ImportError:
                missing.append(package_name)
        
        message = f"已安装: {len(installed)}/{len(optional_packages)}"
        if missing:
            message += f" (缺少: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})"
        
        return True, message  # 可选包不影响基本功能
    
    def check_api_connectivity(self) -> Tuple[bool, str]:
        """检查API连接"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return False, "OPENAI_API_KEY未设置"
        
        try:
            import requests
            
            # 简单的API连通性测试
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "API连接正常"
            elif response.status_code == 401:
                return False, "API密钥无效"
            else:
                return False, f"API响应异常: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"网络连接失败: {str(e)}"
        except Exception as e:
            return False, f"API测试失败: {str(e)}"
    
    def check_file_permissions(self) -> Tuple[bool, str]:
        """检查文件权限"""
        test_dirs = ["./data", "./logs", "./cache"]
        
        permission_issues = []
        
        for dir_path in test_dirs:
            try:
                # 尝试创建目录
                os.makedirs(dir_path, exist_ok=True)
                
                # 测试写入权限
                test_file = os.path.join(dir_path, "test_permission.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                
                # 清理测试文件
                os.remove(test_file)
                
            except PermissionError:
                permission_issues.append(dir_path)
            except Exception as e:
                permission_issues.append(f"{dir_path} ({str(e)})")
        
        if permission_issues:
            return False, f"权限问题: {', '.join(permission_issues)}"
        
        return True, "文件权限正常"
    
    def check_disk_space(self) -> Tuple[bool, str]:
        """检查磁盘空间"""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:  # 小于1GB
                return False, f"磁盘空间不足: {free_gb:.1f}GB"
            elif free_gb < 5.0:  # 小于5GB警告
                return True, f"磁盘空间较低: {free_gb:.1f}GB"
            else:
                return True, f"磁盘空间充足: {free_gb:.1f}GB"
                
        except Exception as e:
            return False, f"无法检查磁盘空间: {str(e)}"
    
    def check_memory(self) -> Tuple[bool, str]:
        """检查内存"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 0.5:  # 小于512MB
                return False, f"可用内存不足: {available_gb:.1f}GB"
            elif available_gb < 2.0:  # 小于2GB警告
                return True, f"可用内存较低: {available_gb:.1f}GB"
            else:
                return True, f"可用内存充足: {available_gb:.1f}GB"
                
        except ImportError:
            return True, "无法检查内存 (psutil未安装)"
        except Exception as e:
            return False, f"无法检查内存: {str(e)}"
    
    def generate_report(self) -> str:
        """生成诊断报告"""
        
        report = "🏥 系统健康检查报告\n"
        report += "=" * 40 + "\n\n"
        
        if not self.errors:
            report += "✅ 系统状态良好，所有检查通过！\n"
        else:
            report += f"❌ 发现 {len(self.errors)} 个问题：\n\n"
            
            for i, error in enumerate(self.errors, 1):
                report += f"{i}. {error}\n"
            
            report += "\n📝 解决建议：\n"
            report += self._generate_suggestions()
        
        return report
    
    def _generate_suggestions(self) -> str:
        """生成解决建议"""
        
        suggestions = []
        
        for error in self.errors:
            if "Python版本过低" in error:
                suggestions.append("- 升级Python到3.8或更高版本")
            
            elif "环境变量" in error:
                suggestions.append("- 设置缺少的环境变量:")
                suggestions.append("  export OPENAI_API_KEY='your-api-key'")
            
            elif "缺少必需包" in error:
                suggestions.append("- 安装缺少的包:")
                suggestions.append("  pip install langchain langchain-openai")
            
            elif "API密钥无效" in error:
                suggestions.append("- 检查API密钥是否正确和有效")
                suggestions.append("- 确认API账户余额充足")
            
            elif "网络连接失败" in error:
                suggestions.append("- 检查网络连接")
                suggestions.append("- 如在企业网络，检查代理设置")
            
            elif "权限问题" in error:
                suggestions.append("- 修复文件权限问题:")
                suggestions.append("  chmod 755 ./data ./logs ./cache")
            
            elif "磁盘空间" in error:
                suggestions.append("- 清理磁盘空间")
                suggestions.append("- 删除临时文件和缓存")
            
            elif "内存不足" in error:
                suggestions.append("- 关闭不必要的程序释放内存")
                suggestions.append("- 考虑增加系统内存")
        
        return "\n".join(suggestions) if suggestions else "- 请参考详细文档获取解决方案"

# 运行健康检查
if __name__ == "__main__":
    checker = HealthChecker()
    results = checker.check_all()
    print("\n" + checker.generate_report())
```

## 🐛 常见问题分类解决

### 1. 认证和API问题

#### 问题：API密钥认证失败

**症状**:
```
AuthenticationError: Invalid API key provided
```

**诊断步骤**:
```python
# api_auth_debug.py
import os
import requests

def debug_api_auth():
    """调试API认证问题"""
    
    print("🔍 API认证诊断")
    print("-" * 30)
    
    # 检查环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY环境变量未设置")
        print("解决方案:")
        print("1. 设置环境变量: export OPENAI_API_KEY='your-key'")
        print("2. 或创建.env文件")
        return
    
    # 检查密钥格式
    if not api_key.startswith("sk-"):
        print("❌ API密钥格式错误")
        print(f"当前密钥: {api_key[:10]}...")
        print("OpenAI API密钥应以'sk-'开头")
        return
    
    # 测试API连接
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ API认证成功")
            models = response.json()
            print(f"可用模型数量: {len(models.get('data', []))}")
        
        elif response.status_code == 401:
            print("❌ API密钥无效")
            print("解决方案:")
            print("1. 检查密钥是否正确复制")
            print("2. 确认密钥未过期")
            print("3. 检查API账户状态")
        
        elif response.status_code == 429:
            print("⚠️ API调用频率超限")
            print("解决方案:")
            print("1. 减少请求频率")
            print("2. 实现重试机制")
            print("3. 升级API计划")
        
        else:
            print(f"❓ 未知API错误: {response.status_code}")
            print(f"响应: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ 网络连接失败")
        print("解决方案:")
        print("1. 检查网络连接")
        print("2. 检查防火墙设置")
        print("3. 如在企业网络，配置代理")
    
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        print("解决方案:")
        print("1. 增加超时时间")
        print("2. 检查网络稳定性")

if __name__ == "__main__":
    debug_api_auth()
```

#### 问题：API配额超限

**症状**:
```
RateLimitError: You exceeded your current quota
```

**解决方案**:
```python
# quota_monitor.py
import time
import requests
from datetime import datetime, timedelta

class QuotaMonitor:
    """API配额监控器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.usage_log = []
    
    def check_billing_info(self):
        """检查账单信息"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            # 获取账单使用情况
            response = requests.get(
                "https://api.openai.com/v1/usage",
                headers=headers,
                params={
                    "date": datetime.now().strftime("%Y-%m-%d")
                }
            )
            
            if response.status_code == 200:
                usage_data = response.json()
                return usage_data
            else:
                print(f"无法获取使用情况: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"检查配额失败: {str(e)}")
            return None
    
    def implement_rate_limiting(self, calls_per_minute=60):
        """实现速率限制"""
        
        def rate_limit_decorator(func):
            call_times = []
            
            def wrapper(*args, **kwargs):
                now = time.time()
                
                # 清理一分钟前的记录
                call_times[:] = [t for t in call_times if now - t < 60]
                
                # 检查是否超过限制
                if len(call_times) >= calls_per_minute:
                    sleep_time = 60 - (now - call_times[0])
                    print(f"⏳ 速率限制：等待 {sleep_time:.1f} 秒")
                    time.sleep(sleep_time)
                
                # 记录调用时间
                call_times.append(now)
                
                return func(*args, **kwargs)
            
            return wrapper
        
        return rate_limit_decorator

# 使用示例
@QuotaMonitor(os.getenv("OPENAI_API_KEY")).implement_rate_limiting(calls_per_minute=30)
def safe_llm_call(template, prompt):
    """带速率限制的LLM调用"""
    return template.run(prompt)
```

### 2. 内存和性能问题

#### 问题：内存溢出

**症状**:
```
MemoryError: Unable to allocate memory
```

**诊断和解决**:
```python
# memory_diagnostics.py
import psutil
import gc
import tracemalloc
from typing import Dict, Any

class MemoryDiagnostics:
    """内存诊断工具"""
    
    def __init__(self):
        self.baseline_memory = None
        tracemalloc.start()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取详细内存使用情况"""
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,      # 物理内存
            "vms_mb": memory_info.vms / 1024 / 1024,      # 虚拟内存
            "percent": process.memory_percent(),           # 内存使用百分比
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def track_memory_growth(self, operation_name: str):
        """跟踪内存增长"""
        
        if self.baseline_memory is None:
            self.baseline_memory = self.get_memory_usage()
            print(f"📊 设置内存基线: {self.baseline_memory['rss_mb']:.1f}MB")
        
        current_memory = self.get_memory_usage()
        growth = current_memory['rss_mb'] - self.baseline_memory['rss_mb']
        
        print(f"📈 {operation_name}: +{growth:.1f}MB "
              f"(总计: {current_memory['rss_mb']:.1f}MB)")
        
        if growth > 100:  # 增长超过100MB
            print("⚠️ 内存增长异常，建议检查:")
            print("   - 是否有内存泄漏")
            print("   - 缓存是否过大")
            print("   - 数据是否及时释放")
        
        return growth
    
    def analyze_memory_hotspots(self) -> Dict[str, Any]:
        """分析内存热点"""
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\n🔥 内存使用热点 (Top 10):")
        for i, stat in enumerate(top_stats[:10], 1):
            print(f"{i:2d}. {stat}")
        
        return {
            "total_memory_mb": sum(stat.size for stat in top_stats) / 1024 / 1024,
            "total_blocks": sum(stat.count for stat in top_stats),
            "top_files": [stat.traceback.format() for stat in top_stats[:5]]
        }
    
    def force_garbage_collection(self) -> int:
        """强制垃圾回收"""
        
        print("🧹 执行垃圾回收...")
        
        before_memory = self.get_memory_usage()
        collected = gc.collect()
        after_memory = self.get_memory_usage()
        
        freed_mb = before_memory['rss_mb'] - after_memory['rss_mb']
        
        print(f"🗑️ 回收了 {collected} 个对象")
        print(f"💾 释放了 {freed_mb:.1f}MB 内存")
        
        return collected

# 内存优化建议
class MemoryOptimizer:
    """内存优化器"""
    
    @staticmethod
    def optimize_document_processing():
        """优化文档处理内存使用"""
        
        tips = [
            "🔹 使用流式处理大文件",
            "🔹 及时释放不需要的文档对象",
            "🔹 限制同时处理的文档数量",
            "🔹 使用生成器而不是列表",
            "🔹 定期执行垃圾回收",
            "🔹 避免在循环中累积大量数据"
        ]
        
        return tips
    
    @staticmethod
    def optimize_vector_storage():
        """优化向量存储内存使用"""
        
        tips = [
            "🔹 使用批处理而不是逐个处理",
            "🔹 选择内存友好的向量数据库",
            "🔹 定期清理临时向量",
            "🔹 使用适当的向量维度",
            "🔹 实现向量数据压缩",
            "🔹 避免重复存储相同向量"
        ]
        
        return tips

# 内存监控装饰器
def monitor_memory(operation_name: str):
    """内存监控装饰器"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            diagnostics = MemoryDiagnostics()
            
            # 执行前记录
            before = diagnostics.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                # 执行后记录
                after = diagnostics.get_memory_usage()
                growth = after['rss_mb'] - before['rss_mb']
                
                print(f"💾 {operation_name}: {growth:+.1f}MB")
                
                # 内存增长过多时警告
                if growth > 50:
                    print(f"⚠️ {operation_name} 内存增长异常: {growth:.1f}MB")
                    diagnostics.analyze_memory_hotspots()
                
                return result
                
            except MemoryError:
                print(f"❌ {operation_name} 内存不足")
                diagnostics.force_garbage_collection()
                raise
        
        return wrapper
    return decorator

# 使用示例
@monitor_memory("文档加载")
def load_documents_with_monitoring(file_paths):
    # 你的文档加载逻辑
    pass
```

#### 问题：处理速度慢

**症状**:
- 响应时间超过预期
- CPU使用率高
- 系统卡顿

**性能分析工具**:
```python
# performance_profiler.py
import time
import cProfile
import pstats
from functools import wraps
from typing import Dict, List, Callable

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.execution_times = {}
        self.call_counts = {}
    
    def profile_function(self, func_name: str = None):
        """函数性能分析装饰器"""
        
        def decorator(func: Callable):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = None
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # 记录执行时间
                    if name not in self.execution_times:
                        self.execution_times[name] = []
                        self.call_counts[name] = 0
                    
                    self.execution_times[name].append(duration)
                    self.call_counts[name] += 1
                    
                    if duration > 5.0:  # 超过5秒的调用
                        print(f"🐌 慢查询警告: {name} 耗时 {duration:.2f}秒")
                
                return result
            
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Dict]:
        """获取性能报告"""
        
        report = {}
        
        for func_name, times in self.execution_times.items():
            if times:
                report[func_name] = {
                    "call_count": self.call_counts[func_name],
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "last_time": times[-1]
                }
        
        return report
    
    def print_performance_summary(self):
        """打印性能摘要"""
        
        report = self.get_performance_report()
        
        print("\n📊 性能分析报告")
        print("=" * 50)
        
        # 按总时间排序
        sorted_funcs = sorted(
            report.items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        
        for func_name, stats in sorted_funcs[:10]:  # 显示前10个
            print(f"\n🔍 {func_name}")
            print(f"   调用次数: {stats['call_count']}")
            print(f"   总耗时: {stats['total_time']:.2f}秒")
            print(f"   平均耗时: {stats['avg_time']:.3f}秒")
            print(f"   最长耗时: {stats['max_time']:.3f}秒")

# 代码热点分析
def profile_code_hotspots(func):
    """代码热点分析装饰器"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # 分析结果
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            print(f"\n🔥 {func.__name__} 热点分析:")
            stats.print_stats(10)  # 显示前10个热点
        
        return result
    
    return wrapper

# 批处理优化建议
class BatchOptimizer:
    """批处理优化器"""
    
    @staticmethod
    def suggest_batch_size(item_count: int, processing_time_per_item: float) -> int:
        """建议批处理大小"""
        
        if processing_time_per_item < 0.1:  # 快速操作
            return min(100, item_count)
        elif processing_time_per_item < 1.0:  # 中等操作
            return min(50, item_count)
        else:  # 慢操作
            return min(10, item_count)
    
    @staticmethod
    def optimize_parallel_processing(item_count: int) -> Dict[str, int]:
        """优化并行处理参数"""
        
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        
        if item_count < cpu_count:
            workers = item_count
        elif item_count < cpu_count * 4:
            workers = cpu_count
        else:
            workers = min(cpu_count * 2, 16)  # 最多16个worker
        
        batch_size = max(1, item_count // workers)
        
        return {
            "workers": workers,
            "batch_size": batch_size,
            "estimated_batches": (item_count + batch_size - 1) // batch_size
        }

# 使用示例
profiler = PerformanceProfiler()

@profiler.profile_function("LLM调用")
@profile_code_hotspots
def optimized_llm_call(template, prompt):
    """优化的LLM调用"""
    return template.run(prompt)
```

### 3. 数据处理问题

#### 问题：文档加载失败

**症状**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
FileNotFoundError: No such file or directory
```

**解决方案**:
```python
# document_loader_fixer.py
import os
import chardet
from typing import Optional, Tuple, List

class DocumentLoaderFixer:
    """文档加载问题修复器"""
    
    def __init__(self):
        self.supported_encodings = ['utf-8', 'gbk', 'gb2312', 'ascii', 'latin-1']
    
    def detect_file_encoding(self, file_path: str) -> Optional[str]:
        """检测文件编码"""
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # 读取前10KB
                result = chardet.detect(raw_data)
                
                encoding = result.get('encoding')
                confidence = result.get('confidence', 0)
                
                print(f"📝 {file_path}:")
                print(f"   检测编码: {encoding}")
                print(f"   置信度: {confidence:.2f}")
                
                return encoding if confidence > 0.7 else None
                
        except Exception as e:
            print(f"❌ 编码检测失败: {str(e)}")
            return None
    
    def safe_file_read(self, file_path: str) -> Tuple[Optional[str], str]:
        """安全地读取文件"""
        
        if not os.path.exists(file_path):
            return None, f"文件不存在: {file_path}"
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB
            return None, f"文件过大: {file_size / 1024 / 1024:.1f}MB"
        
        # 尝试检测编码
        detected_encoding = self.detect_file_encoding(file_path)
        
        # 尝试不同编码
        encodings_to_try = [detected_encoding] + self.supported_encodings
        encodings_to_try = [enc for enc in encodings_to_try if enc]  # 去除None
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    print(f"✅ 成功读取文件 (编码: {encoding})")
                    return content, ""
                    
            except UnicodeDecodeError:
                print(f"❌ 编码 {encoding} 失败")
                continue
            except Exception as e:
                return None, f"读取错误: {str(e)}"
        
        return None, "所有编码尝试均失败"
    
    def batch_fix_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量修复文件问题"""
        
        results = {
            "success": [],
            "failed": [],
            "stats": {}
        }
        
        for file_path in file_paths:
            content, error = self.safe_file_read(file_path)
            
            if content:
                results["success"].append({
                    "file": file_path,
                    "size": len(content),
                    "lines": content.count('\n') + 1
                })
            else:
                results["failed"].append({
                    "file": file_path,
                    "error": error
                })
        
        results["stats"] = {
            "total": len(file_paths),
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"]),
            "success_rate": len(results["success"]) / len(file_paths) if file_paths else 0
        }
        
        return results
    
    def suggest_fixes(self, error_message: str) -> List[str]:
        """根据错误信息建议修复方案"""
        
        suggestions = []
        
        if "UnicodeDecodeError" in error_message:
            suggestions.extend([
                "🔧 编码问题修复:",
                "   - 使用chardet检测文件编码",
                "   - 尝试常见编码: utf-8, gbk, latin-1",
                "   - 考虑转换文件编码为UTF-8"
            ])
        
        elif "FileNotFoundError" in error_message:
            suggestions.extend([
                "🔧 文件路径问题修复:",
                "   - 检查文件路径是否正确",
                "   - 使用绝对路径避免相对路径问题",
                "   - 确认文件确实存在"
            ])
        
        elif "PermissionError" in error_message:
            suggestions.extend([
                "🔧 权限问题修复:",
                "   - 检查文件读取权限",
                "   - 使用 chmod 修改权限",
                "   - 确认用户有足够权限"
            ])
        
        elif "MemoryError" in error_message:
            suggestions.extend([
                "🔧 内存问题修复:",
                "   - 使用流式读取大文件",
                "   - 分块处理文件内容",
                "   - 增加系统内存"
            ])
        
        return suggestions

# 文件处理工具集
class FileUtilities:
    """文件处理工具集"""
    
    @staticmethod
    def convert_encoding(file_path: str, target_encoding: str = 'utf-8') -> bool:
        """转换文件编码"""
        
        fixer = DocumentLoaderFixer()
        content, error = fixer.safe_file_read(file_path)
        
        if not content:
            print(f"❌ 无法读取文件: {error}")
            return False
        
        try:
            # 备份原文件
            backup_path = f"{file_path}.backup"
            os.rename(file_path, backup_path)
            
            # 写入新编码
            with open(file_path, 'w', encoding=target_encoding) as f:
                f.write(content)
            
            print(f"✅ 文件编码已转换为 {target_encoding}")
            print(f"📄 原文件备份为: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ 编码转换失败: {str(e)}")
            return False
    
    @staticmethod
    def validate_file_structure(file_path: str) -> Dict[str, Any]:
        """验证文件结构"""
        
        info = {
            "exists": os.path.exists(file_path),
            "readable": False,
            "size_mb": 0,
            "lines": 0,
            "encoding": None,
            "issues": []
        }
        
        if not info["exists"]:
            info["issues"].append("文件不存在")
            return info
        
        try:
            # 检查可读性
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1)
            info["readable"] = True
        except:
            info["issues"].append("文件不可读或编码问题")
        
        # 文件大小
        info["size_mb"] = os.path.getsize(file_path) / 1024 / 1024
        
        if info["size_mb"] > 50:
            info["issues"].append(f"文件过大: {info['size_mb']:.1f}MB")
        
        return info

# 使用示例
def fix_document_loading_issues(file_paths: List[str]):
    """修复文档加载问题"""
    
    fixer = DocumentLoaderFixer()
    results = fixer.batch_fix_files(file_paths)
    
    print("📊 批量修复结果:")
    print(f"   成功: {results['stats']['success_count']}")
    print(f"   失败: {results['stats']['failed_count']}")
    print(f"   成功率: {results['stats']['success_rate']:.1%}")
    
    # 显示失败详情
    if results["failed"]:
        print("\n❌ 失败文件:")
        for item in results["failed"]:
            print(f"   {item['file']}: {item['error']}")
            
            # 提供修复建议
            suggestions = fixer.suggest_fixes(item['error'])
            for suggestion in suggestions:
                print(f"     {suggestion}")
```

### 4. 向量存储问题

#### 问题：向量数据库连接失败

**诊断和修复工具**:
```python
# vectorstore_diagnostics.py
import os
import tempfile
from typing import Dict, Any, Optional

class VectorStoreDiagnostics:
    """向量存储诊断工具"""
    
    def __init__(self):
        self.test_results = {}
    
    def test_chroma_connection(self) -> Dict[str, Any]:
        """测试Chroma连接"""
        
        result = {
            "success": False,
            "error": None,
            "details": {}
        }
        
        try:
            import chromadb
            
            # 测试内存数据库
            client = chromadb.Client()
            collection = client.create_collection("test_collection")
            
            # 测试基本操作
            collection.add(
                documents=["测试文档"],
                ids=["test_1"]
            )
            
            # 测试查询
            results = collection.query(
                query_texts=["测试"],
                n_results=1
            )
            
            result["success"] = True
            result["details"] = {
                "collection_count": len(client.list_collections()),
                "query_results": len(results['documents'][0])
            }
            
            # 清理
            client.delete_collection("test_collection")
            
        except ImportError:
            result["error"] = "ChromaDB未安装"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def test_faiss_installation(self) -> Dict[str, Any]:
        """测试FAISS安装"""
        
        result = {
            "success": False,
            "error": None,
            "details": {}
        }
        
        try:
            import faiss
            import numpy as np
            
            # 创建测试向量
            d = 64  # 向量维度
            vectors = np.random.random((100, d)).astype('float32')
            
            # 创建索引
            index = faiss.IndexFlatL2(d)
            index.add(vectors)
            
            # 测试搜索
            query = np.random.random((1, d)).astype('float32')
            distances, indices = index.search(query, 5)
            
            result["success"] = True
            result["details"] = {
                "index_size": index.ntotal,
                "search_results": len(indices[0])
            }
            
        except ImportError:
            result["error"] = "FAISS未安装"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def test_embedding_models(self) -> Dict[str, Any]:
        """测试嵌入模型"""
        
        results = {}
        
        # 测试OpenAI Embeddings
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings
                
                embeddings = OpenAIEmbeddings()
                test_vectors = embeddings.embed_documents(["测试文本"])
                
                results["openai"] = {
                    "success": True,
                    "dimension": len(test_vectors[0]),
                    "error": None
                }
                
            except Exception as e:
                results["openai"] = {
                    "success": False,
                    "error": str(e)
                }
        else:
            results["openai"] = {
                "success": False,
                "error": "OPENAI_API_KEY未设置"
            }
        
        # 测试SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            test_vectors = model.encode(["测试文本"])
            
            results["sentence_transformers"] = {
                "success": True,
                "dimension": len(test_vectors[0]),
                "error": None
            }
            
        except ImportError:
            results["sentence_transformers"] = {
                "success": False,
                "error": "sentence-transformers未安装"
            }
        except Exception as e:
            results["sentence_transformers"] = {
                "success": False,
                "error": str(e)
            }
        
        return results
    
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """运行完整诊断"""
        
        print("🔍 向量存储诊断开始...")
        
        # 测试各个组件
        chroma_result = self.test_chroma_connection()
        faiss_result = self.test_faiss_installation()
        embedding_results = self.test_embedding_models()
        
        # 汇总结果
        summary = {
            "chroma": chroma_result,
            "faiss": faiss_result,
            "embeddings": embedding_results,
            "overall_status": "healthy"
        }
        
        # 检查整体状态
        all_tests = [chroma_result, faiss_result] + list(embedding_results.values())
        failed_tests = [test for test in all_tests if not test["success"]]
        
        if len(failed_tests) > len(all_tests) / 2:
            summary["overall_status"] = "critical"
        elif failed_tests:
            summary["overall_status"] = "warning"
        
        self.print_diagnostic_report(summary)
        return summary
    
    def print_diagnostic_report(self, summary: Dict[str, Any]):
        """打印诊断报告"""
        
        print("\n📊 向量存储诊断报告")
        print("=" * 40)
        
        # ChromaDB状态
        chroma = summary["chroma"]
        status = "✅" if chroma["success"] else "❌"
        print(f"{status} ChromaDB: {'正常' if chroma['success'] else chroma['error']}")
        
        # FAISS状态
        faiss = summary["faiss"]
        status = "✅" if faiss["success"] else "❌"
        print(f"{status} FAISS: {'正常' if faiss['success'] else faiss['error']}")
        
        # 嵌入模型状态
        print("\n📝 嵌入模型:")
        for model_name, result in summary["embeddings"].items():
            status = "✅" if result["success"] else "❌"
            if result["success"]:
                dim = result.get("dimension", "未知")
                print(f"{status} {model_name}: 正常 (维度: {dim})")
            else:
                print(f"{status} {model_name}: {result['error']}")
        
        # 整体状态
        overall = summary["overall_status"]
        if overall == "healthy":
            print("\n🎉 系统状态健康")
        elif overall == "warning":
            print("\n⚠️ 系统存在问题，但可正常使用")
        else:
            print("\n🚨 系统存在严重问题")
        
        return summary

# 向量存储修复工具
class VectorStoreFixKit:
    """向量存储修复工具包"""
    
    @staticmethod
    def install_missing_packages():
        """安装缺失的包"""
        
        packages = {
            "chromadb": "pip install chromadb",
            "faiss-cpu": "pip install faiss-cpu",
            "sentence-transformers": "pip install sentence-transformers"
        }
        
        print("📦 检查并安装缺失的包...")
        
        for package, install_cmd in packages.items():
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} 已安装")
            except ImportError:
                print(f"❌ {package} 未安装")
                print(f"   安装命令: {install_cmd}")
    
    @staticmethod
    def repair_chroma_database(db_path: str):
        """修复Chroma数据库"""
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            print(f"🔧 修复Chroma数据库: {db_path}")
            
            # 尝试重建数据库
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(allow_reset=True)
            )
            
            # 获取所有集合
            collections = client.list_collections()
            print(f"发现 {len(collections)} 个集合")
            
            # 检查每个集合
            for collection in collections:
                try:
                    count = collection.count()
                    print(f"✅ 集合 {collection.name}: {count} 个文档")
                except Exception as e:
                    print(f"❌ 集合 {collection.name} 损坏: {str(e)}")
                    # 可以选择删除损坏的集合
                    # client.delete_collection(collection.name)
            
            print("🎉 数据库修复完成")
            
        except Exception as e:
            print(f"❌ 数据库修复失败: {str(e)}")
    
    @staticmethod
    def optimize_vector_storage():
        """优化向量存储建议"""
        
        tips = [
            "💡 向量存储优化建议:",
            "",
            "🔹 选择合适的向量数据库:",
            "   - 小规模数据: ChromaDB (本地)",
            "   - 大规模数据: FAISS + 持久化",
            "   - 生产环境: Pinecone, Weaviate",
            "",
            "🔹 优化向量维度:",
            "   - OpenAI ada-002: 1536维",
            "   - SentenceTransformers: 384-768维",
            "   - 选择合适的模型平衡精度和速度",
            "",
            "🔹 索引优化:",
            "   - 定期重建索引提高查询速度",
            "   - 使用适当的相似度计算方法",
            "   - 实现增量更新机制",
            "",
            "🔹 数据管理:",
            "   - 定期清理过期数据",
            "   - 实现数据备份机制",
            "   - 监控存储空间使用"
        ]
        
        return "\n".join(tips)

# 使用示例
def diagnose_and_fix_vectorstore():
    """诊断并修复向量存储问题"""
    
    diagnostics = VectorStoreDiagnostics()
    fix_kit = VectorStoreFixKit()
    
    # 运行诊断
    results = diagnostics.run_full_diagnostics()
    
    # 根据结果提供修复建议
    if results["overall_status"] != "healthy":
        print("\n🔧 修复建议:")
        
        if not results["chroma"]["success"]:
            print("- 安装ChromaDB: pip install chromadb")
        
        if not results["faiss"]["success"]:
            print("- 安装FAISS: pip install faiss-cpu")
        
        embedding_issues = [
            name for name, result in results["embeddings"].items()
            if not result["success"]
        ]
        
        if embedding_issues:
            print(f"- 修复嵌入模型问题: {', '.join(embedding_issues)}")
    
    # 显示优化建议
    print("\n" + fix_kit.optimize_vector_storage())

if __name__ == "__main__":
    diagnose_and_fix_vectorstore()
```

## 🛠️ 自动修复工具

```python
# auto_fixer.py
import os
import sys
import subprocess
import json
from typing import List, Dict, Any, Callable

class AutoFixer:
    """自动问题修复器"""
    
    def __init__(self):
        self.fix_registry = {}
        self.register_fixes()
    
    def register_fix(self, error_pattern: str, fix_function: Callable):
        """注册修复函数"""
        self.fix_registry[error_pattern] = fix_function
    
    def register_fixes(self):
        """注册所有修复方法"""
        
        self.register_fix("ModuleNotFoundError", self.fix_missing_module)
        self.register_fix("AuthenticationError", self.fix_auth_error)
        self.register_fix("UnicodeDecodeError", self.fix_encoding_error)
        self.register_fix("MemoryError", self.fix_memory_error)
        self.register_fix("ConnectionError", self.fix_connection_error)
    
    def fix_missing_module(self, error_message: str) -> bool:
        """修复缺失模块问题"""
        
        # 提取模块名
        import re
        match = re.search(r"No module named '(\w+)'", error_message)
        
        if not match:
            return False
        
        module_name = match.group(1)
        
        # 常见模块的安装命令映射
        install_commands = {
            "langchain": "pip install langchain",
            "openai": "pip install openai",
            "chromadb": "pip install chromadb",
            "faiss": "pip install faiss-cpu",
            "sentence_transformers": "pip install sentence-transformers",
            "tiktoken": "pip install tiktoken",
            "pypdf": "pip install pypdf",
            "docx": "pip install python-docx"
        }
        
        if module_name in install_commands:
            command = install_commands[module_name]
            print(f"🔧 自动安装缺失模块: {module_name}")
            
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode == 0:
                    print(f"✅ 模块 {module_name} 安装成功")
                    return True
                else:
                    print(f"❌ 模块安装失败: {result.stderr}")
                    return False
                    
            except subprocess.TimeoutExpired:
                print(f"❌ 模块安装超时: {module_name}")
                return False
            except Exception as e:
                print(f"❌ 模块安装异常: {str(e)}")
                return False
        
        else:
            print(f"❓ 未知模块: {module_name}，请手动安装")
            return False
    
    def fix_auth_error(self, error_message: str) -> bool:
        """修复认证错误"""
        
        print("🔧 修复API认证问题...")
        
        # 检查环境变量
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ OPENAI_API_KEY环境变量未设置")
            print("请设置API密钥:")
            print("export OPENAI_API_KEY='your-api-key'")
            return False
        
        if not api_key.startswith("sk-"):
            print("❌ API密钥格式错误")
            print("OpenAI API密钥应以'sk-'开头")
            return False
        
        # 可以添加更多自动修复逻辑
        print("✅ API密钥格式正确，请检查密钥有效性")
        return True
    
    def fix_encoding_error(self, error_message: str) -> bool:
        """修复编码错误"""
        
        print("🔧 修复文件编码问题...")
        
        # 这里可以实现自动编码转换
        # 由于需要具体文件路径，这里只给出建议
        
        suggestions = [
            "检测文件编码: chardet.detect()",
            "转换为UTF-8编码",
            "使用合适的编码参数打开文件"
        ]
        
        for suggestion in suggestions:
            print(f"💡 {suggestion}")
        
        return True
    
    def fix_memory_error(self, error_message: str) -> bool:
        """修复内存错误"""
        
        print("🔧 修复内存问题...")
        
        # 执行垃圾回收
        import gc
        collected = gc.collect()
        print(f"🗑️ 清理了 {collected} 个对象")
        
        # 提供内存优化建议
        suggestions = [
            "减少batch_size参数",
            "使用流式处理",
            "分批处理数据",
            "增加系统内存"
        ]
        
        for suggestion in suggestions:
            print(f"💡 {suggestion}")
        
        return True
    
    def fix_connection_error(self, error_message: str) -> bool:
        """修复连接错误"""
        
        print("🔧 修复网络连接问题...")
        
        # 检查网络连通性
        try:
            import requests
            response = requests.get("https://www.google.com", timeout=5)
            print("✅ 网络连接正常")
            
            # 检查API端点
            response = requests.get("https://api.openai.com", timeout=5)
            print("✅ OpenAI API端点可达")
            
            return True
            
        except requests.exceptions.RequestException:
            print("❌ 网络连接问题")
            
            suggestions = [
                "检查网络连接",
                "检查防火墙设置",
                "检查代理配置",
                "稍后重试"
            ]
            
            for suggestion in suggestions:
                print(f"💡 {suggestion}")
            
            return False
    
    def auto_fix(self, error_message: str) -> bool:
        """自动修复错误"""
        
        print(f"🔍 分析错误: {error_message}")
        
        for pattern, fix_function in self.fix_registry.items():
            if pattern in error_message:
                print(f"🎯 匹配到修复模式: {pattern}")
                return fix_function(error_message)
        
        print("❓ 未找到自动修复方案")
        return False

# 错误监控和自动修复装饰器
def auto_fix_on_error(auto_fixer: AutoFixer):
    """错误自动修复装饰器"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_message = str(e)
                print(f"❌ 捕获错误: {error_message}")
                
                # 尝试自动修复
                if auto_fixer.auto_fix(error_message):
                    print("🔄 重试执行...")
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_e:
                        print(f"❌ 重试后仍然失败: {str(retry_e)}")
                        raise
                else:
                    print("🚨 自动修复失败，需要手动处理")
                    raise
        
        return wrapper
    return decorator

# 使用示例
auto_fixer = AutoFixer()

@auto_fix_on_error(auto_fixer)
def protected_function():
    """受保护的函数，会自动尝试修复错误"""
    # 你的代码逻辑
    pass
```

## 📝 问题报告模板

当遇到无法自动解决的问题时，请使用以下模板报告问题：

```markdown
# 问题报告

## 基本信息
- **系统**: [Windows/macOS/Linux]
- **Python版本**: 
- **LangChain版本**: 
- **发生时间**: 

## 错误描述
<!-- 详细描述遇到的问题 -->

## 错误信息
```
<!-- 粘贴完整的错误堆栈 -->
```

## 重现步骤
1. 
2. 
3. 

## 已尝试的解决方案
- [ ] 运行健康检查脚本
- [ ] 检查环境变量设置
- [ ] 重新安装依赖包
- [ ] 清理缓存和临时文件
- [ ] 其他: 

## 环境信息
<!-- 运行以下命令并粘贴输出 -->
```bash
python health_check.py
pip list | grep langchain
echo $OPENAI_API_KEY | head -c 10
```

## 附加信息
<!-- 任何可能相关的额外信息 -->
```

通过使用这个全面的故障排除指南，你应该能够诊断和解决大部分常见问题。如果问题仍然存在，请使用问题报告模板提交详细的问题报告。