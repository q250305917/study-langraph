"""
测试模块

本模块包含 LangChain 学习项目的所有测试代码。
测试采用 pytest 框架，遵循测试最佳实践，确保代码质量和可靠性。

测试结构：
- unit/: 单元测试，测试单个函数或类的功能
- integration/: 集成测试，测试模块间的交互
- e2e/: 端到端测试，测试完整的用户场景
- fixtures/: 测试夹具和共享测试数据
- conftest.py: pytest 配置和共享夹具

测试类型：
1. 单元测试：测试单个函数、类或方法
2. 集成测试：测试模块、组件间的集成
3. 功能测试：测试业务功能的正确性
4. 性能测试：测试代码的性能和效率
5. 安全测试：测试安全相关的功能
6. API测试：测试外部API的集成（可跳过）

测试约定：
- 测试文件以 test_ 开头
- 测试类以 Test 开头
- 测试方法以 test_ 开头
- 使用描述性的测试方法名
- 每个测试只测试一个功能点
- 测试应该独立且可重复执行

标记(Markers)：
- @pytest.mark.unit: 单元测试
- @pytest.mark.integration: 集成测试
- @pytest.mark.slow: 慢速测试
- @pytest.mark.api: 需要API密钥的测试
- @pytest.mark.optional: 可选测试（依赖外部服务）
"""

# 测试版本信息
__test_version__ = "0.1.0"

# 测试配置
TEST_CONFIG = {
    "timeout": 30,  # 默认测试超时时间（秒）
    "retry_count": 3,  # 失败重试次数
    "parallel_workers": 4,  # 并行测试工作进程数
    "coverage_threshold": 80,  # 覆盖率阈值（百分比）
}

# 测试数据目录
TEST_DATA_DIRS = {
    "fixtures": "tests/fixtures",
    "sample_data": "tests/data",
    "expected_outputs": "tests/expected",
    "temp": "tests/temp",
}

# 测试分类
TEST_CATEGORIES = {
    "core": "核心功能测试",
    "chains": "链功能测试", 
    "agents": "代理功能测试",
    "tools": "工具功能测试",
    "utils": "工具函数测试",
    "integration": "集成测试",
    "performance": "性能测试",
    "security": "安全测试",
}

def get_test_config():
    """
    获取测试配置
    
    Returns:
        dict: 测试配置字典
    """
    import os
    
    config = TEST_CONFIG.copy()
    
    # 从环境变量覆盖配置
    config["timeout"] = int(os.getenv("TEST_TIMEOUT", config["timeout"]))
    config["retry_count"] = int(os.getenv("TEST_RETRY_COUNT", config["retry_count"]))
    config["parallel_workers"] = int(os.getenv("TEST_WORKERS", config["parallel_workers"]))
    config["coverage_threshold"] = int(os.getenv("COVERAGE_THRESHOLD", config["coverage_threshold"]))
    
    return config

def setup_test_environment():
    """
    设置测试环境
    
    创建必要的测试目录，设置测试数据，配置日志等
    """
    import os
    import tempfile
    from pathlib import Path
    
    # 创建测试数据目录
    for dir_name, dir_path in TEST_DATA_DIRS.items():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 设置临时目录
    temp_dir = tempfile.mkdtemp(prefix="langchain_test_")
    os.environ["TEST_TEMP_DIR"] = temp_dir
    
    # 设置测试环境变量
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    # 跳过需要API密钥的测试（如果没有设置）
    api_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "GOOGLE_API_KEY",
        "COHERE_API_KEY",
    ]
    
    missing_keys = [key for key in api_keys if not os.getenv(key)]
    if missing_keys:
        os.environ["SKIP_API_TESTS"] = "true"
        print(f"⚠️  跳过API测试，缺少密钥: {', '.join(missing_keys)}")

def cleanup_test_environment():
    """
    清理测试环境
    
    删除临时文件，重置环境变量等
    """
    import os
    import shutil
    
    # 清理临时目录
    temp_dir = os.getenv("TEST_TEMP_DIR")
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 清理环境变量
    test_env_vars = [
        "TEST_TEMP_DIR",
        "ENVIRONMENT",
        "SKIP_API_TESTS",
    ]
    
    for var in test_env_vars:
        os.environ.pop(var, None)

def get_test_data_path(filename):
    """
    获取测试数据文件的完整路径
    
    Args:
        filename (str): 测试数据文件名
        
    Returns:
        str: 测试数据文件的完整路径
    """
    from pathlib import Path
    
    data_dir = Path(TEST_DATA_DIRS["sample_data"])
    return str(data_dir / filename)

def load_test_fixture(fixture_name):
    """
    加载测试夹具数据
    
    Args:
        fixture_name (str): 夹具名称
        
    Returns:
        dict: 夹具数据
    """
    import json
    from pathlib import Path
    
    fixture_path = Path(TEST_DATA_DIRS["fixtures"]) / f"{fixture_name}.json"
    
    if not fixture_path.exists():
        raise FileNotFoundError(f"测试夹具不存在: {fixture_path}")
    
    with open(fixture_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_mock_llm_response(text, usage=None):
    """
    创建模拟的LLM响应
    
    Args:
        text (str): 响应文本
        usage (dict, optional): 使用统计信息
        
    Returns:
        dict: 模拟的LLM响应数据
    """
    response = {
        "text": text,
        "model": "mock-model",
        "created": 1234567890,
        "usage": usage or {
            "prompt_tokens": len(text.split()) * 2,
            "completion_tokens": len(text.split()),
            "total_tokens": len(text.split()) * 3,
        }
    }
    
    return response

def assert_response_format(response, expected_keys=None):
    """
    断言响应格式正确
    
    Args:
        response: 要验证的响应对象
        expected_keys (list, optional): 期望的键列表
    """
    assert response is not None, "响应不能为空"
    
    if expected_keys:
        if isinstance(response, dict):
            for key in expected_keys:
                assert key in response, f"响应中缺少键: {key}"
        else:
            for key in expected_keys:
                assert hasattr(response, key), f"响应对象缺少属性: {key}"

def run_test_suite(test_type="all", verbose=False):
    """
    运行测试套件
    
    Args:
        test_type (str): 测试类型 ("unit", "integration", "all")
        verbose (bool): 是否显示详细输出
    """
    import subprocess
    import sys
    
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "api":
        cmd.extend(["-m", "api"])
    
    # 添加覆盖率报告
    cmd.extend([
        "--cov=src/langchain_learning",
        "--cov-report=term-missing",
        "--cov-report=html",
    ])
    
    print(f"🧪 运行测试: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=".", capture_output=False)
    
    return result.returncode == 0

# 定义公共接口
__all__ = [
    # 版本和配置
    "__test_version__",
    "TEST_CONFIG",
    "TEST_DATA_DIRS",
    "TEST_CATEGORIES",
    
    # 配置函数
    "get_test_config",
    
    # 环境管理
    "setup_test_environment",
    "cleanup_test_environment",
    
    # 测试数据
    "get_test_data_path",
    "load_test_fixture",
    
    # 模拟工具
    "create_mock_llm_response",
    
    # 断言工具
    "assert_response_format",
    
    # 测试运行
    "run_test_suite",
]

# 如果直接运行此模块，设置测试环境并运行测试
if __name__ == "__main__":
    import sys
    
    print("🧪 LangChain 学习项目测试套件")
    print("=" * 40)
    print(f"测试版本: {__test_version__}")
    print(f"测试分类: {list(TEST_CATEGORIES.keys())}")
    
    # 设置测试环境
    setup_test_environment()
    
    try:
        # 运行测试
        test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
        success = run_test_suite(test_type=test_type, verbose=True)
        
        if success:
            print("\n✅ 所有测试通过!")
        else:
            print("\n❌ 部分测试失败!")
            sys.exit(1)
            
    finally:
        # 清理测试环境
        cleanup_test_environment()