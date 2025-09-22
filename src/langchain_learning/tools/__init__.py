"""
工具模块 (tools)

本模块包含 LangChain 中各种工具（Tools）的实现、自定义工具开发和工具集成。
工具是代理可以调用的外部函数或服务，用于扩展代理的能力。

主要包含的工具类型：
- 搜索工具：网页搜索、学术搜索、新闻搜索
- 计算工具：数学计算、统计分析、数据处理
- 文件工具：文件读写、格式转换、文档处理
- API 工具：第三方 API 集成、数据库连接
- 实用工具：时间日期、文本处理、格式化
- 自定义工具：展示如何创建自定义工具

工具特性：
- 函数签名：清晰的输入输出定义
- 错误处理：优雅的异常处理和恢复
- 参数验证：输入参数的验证和清理
- 文档说明：详细的工具描述和使用说明
- 测试覆盖：完整的单元测试和集成测试
- 性能优化：缓存、并发和资源管理
"""

# 版本信息
__version__ = "0.1.0"

# 计划实现的工具类型
PLANNED_TOOLS = [
    "search_tools",      # 搜索工具集
    "math_tools",        # 数学计算工具
    "file_tools",        # 文件操作工具  
    "web_tools",         # 网页工具
    "database_tools",    # 数据库工具
    "api_tools",         # API 集成工具
    "text_tools",        # 文本处理工具
    "image_tools",       # 图像处理工具
    "audio_tools",       # 音频处理工具
    "datetime_tools",    # 日期时间工具
    "validation_tools",  # 数据验证工具
    "custom_tools",      # 自定义工具示例
]

# 工具类型描述
TOOL_DESCRIPTIONS = {
    "search_tools": "搜索工具：网页搜索、学术搜索、新闻搜索等",
    "math_tools": "数学工具：计算器、统计分析、数学函数等",
    "file_tools": "文件工具：文件读写、格式转换、压缩解压等",
    "web_tools": "网页工具：网页抓取、HTML解析、URL处理等",
    "database_tools": "数据库工具：SQL查询、数据库连接、数据导入导出等",
    "api_tools": "API工具：第三方API集成、HTTP请求、认证等",
    "text_tools": "文本工具：文本处理、格式化、编码转换等",
    "image_tools": "图像工具：图像处理、格式转换、特征提取等",
    "audio_tools": "音频工具：音频处理、格式转换、语音识别等",
    "datetime_tools": "时间工具：日期时间处理、时区转换、格式化等",
    "validation_tools": "验证工具：数据验证、格式检查、内容过滤等",
    "custom_tools": "自定义工具：展示如何创建和集成自定义工具",
}

# 工具分类
TOOL_CATEGORIES = {
    "data_processing": ["file_tools", "database_tools", "text_tools", "validation_tools"],
    "external_apis": ["search_tools", "api_tools", "web_tools"],
    "computation": ["math_tools"],
    "media": ["image_tools", "audio_tools"],
    "utilities": ["datetime_tools", "validation_tools"],
    "development": ["custom_tools"],
}

# 尝试导入已实现的工具模块
_available_tools = []
_import_errors = []

# 尝试导入各种工具的实现
for tool_type in PLANNED_TOOLS:
    try:
        # 动态导入模块
        module = __import__(f".{tool_type}", package=__name__, level=1)
        _available_tools.append(tool_type)
    except ImportError as e:
        _import_errors.append((tool_type, str(e)))

def get_available_tools():
    """
    获取当前可用的工具类型列表
    
    Returns:
        list: 已成功导入的工具类型名称列表
    """
    return _available_tools.copy()

def get_import_errors():
    """
    获取导入失败的工具和错误信息
    
    Returns:
        list: 包含 (工具类型, 错误信息) 元组的列表
    """
    return _import_errors.copy()

def get_tool_status():
    """
    获取所有工具的状态信息
    
    Returns:
        dict: 包含工具状态的字典
    """
    status = {}
    
    for tool in PLANNED_TOOLS:
        if tool in _available_tools:
            status[tool] = {
                "available": True,
                "description": TOOL_DESCRIPTIONS.get(tool, "无描述"),
                "error": None
            }
        else:
            error_info = next(
                (error for name, error in _import_errors if name == tool),
                "工具未实现"
            )
            status[tool] = {
                "available": False,
                "description": TOOL_DESCRIPTIONS.get(tool, "无描述"),
                "error": error_info
            }
    
    return status

def get_tools_by_category(category):
    """
    根据分类获取工具列表
    
    Args:
        category (str): 工具分类 (data_processing, external_apis, etc.)
        
    Returns:
        list: 属于指定分类的工具类型列表
    """
    return TOOL_CATEGORIES.get(category, [])

def print_tool_status():
    """
    打印所有工具的状态信息
    """
    print(f"\n🛠️  工具模块状态 (tools v{__version__}):")
    print("=" * 60)
    
    status = get_tool_status()
    
    for tool, info in status.items():
        status_icon = "✅" if info["available"] else "❌"
        print(f"{status_icon} {tool:16} - {info['description']}")
        
        if not info["available"] and info["error"] != "工具未实现":
            print(f"   错误: {info['error']}")
    
    print(f"\n📊 统计: {len(_available_tools)}/{len(PLANNED_TOOLS)} 个工具类型可用")
    
    if _import_errors:
        print(f"\n⚠️  导入错误: {len(_import_errors)} 个")

def print_tool_categories():
    """
    打印工具分类信息
    """
    print(f"\n📂 工具分类:")
    print("=" * 40)
    
    for category, tools in TOOL_CATEGORIES.items():
        print(f"\n🔸 {category.upper().replace('_', ' ')}:")
        for tool in tools:
            status = "✅" if tool in _available_tools else "❌"
            description = TOOL_DESCRIPTIONS.get(tool, "无描述")
            print(f"  {status} {tool}: {description}")

def get_tool_examples():
    """
    获取各种工具的使用示例
    
    Returns:
        dict: 包含工具类型和示例代码的字典
    """
    examples = {
        "search_tools": """
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper

# DuckDuckGo 搜索工具
search = DuckDuckGoSearchRun()
result = search.run("LangChain 是什么")

# Wikipedia 搜索工具
wikipedia = WikipediaAPIWrapper()
wiki_result = wikipedia.run("Artificial Intelligence")
        """,
        
        "math_tools": """
from langchain.tools import ShellTool
from langchain.agents import load_tools

# 加载数学工具
math_tools = load_tools(["llm-math"], llm=llm)

# 或者自定义计算器工具
def calculator(expression: str) -> str:
    \"\"\"执行数学计算\"\"\"
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建工具
from langchain.tools import Tool
calc_tool = Tool(
    name="Calculator",
    description="执行数学计算",
    func=calculator
)
        """,
        
        "custom_tools": """
from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel

class WeatherInput(BaseModel):
    location: str
    unit: str = "celsius"

class WeatherTool(BaseTool):
    name = "weather"
    description = "获取指定地点的天气信息"
    args_schema = WeatherInput
    
    def _run(self, location: str, unit: str = "celsius") -> str:
        # 这里应该调用实际的天气API
        return f"{location}的天气: 晴天，25°{unit}"
    
    async def _arun(self, location: str, unit: str = "celsius") -> str:
        # 异步版本
        return self._run(location, unit)

# 使用自定义工具
weather_tool = WeatherTool()
result = weather_tool.run({"location": "北京", "unit": "celsius"})
        """
    }
    
    return examples

def print_tool_examples():
    """
    打印工具的使用示例
    """
    examples = get_tool_examples()
    
    print(f"\n📝 工具使用示例:")
    print("=" * 60)
    
    for tool_type, example in examples.items():
        print(f"\n🛠️  {tool_type} 示例:")
        print("-" * 40)
        print(example)

def create_tool_registry():
    """
    创建工具注册表，用于管理和发现工具
    
    Returns:
        dict: 工具注册表
    """
    registry = {
        "available_tools": _available_tools,
        "tool_descriptions": TOOL_DESCRIPTIONS,
        "tool_categories": TOOL_CATEGORIES,
        "import_errors": _import_errors,
    }
    
    return registry

# 定义公共接口
__all__ = [
    # 版本信息
    "__version__",
    
    # 状态查询函数
    "get_available_tools",
    "get_import_errors",
    "get_tool_status",
    "print_tool_status",
    
    # 分类相关函数
    "get_tools_by_category", 
    "print_tool_categories",
    
    # 示例函数
    "get_tool_examples",
    "print_tool_examples",
    
    # 工具注册表
    "create_tool_registry",
    
    # 工具描述和分类
    "PLANNED_TOOLS",
    "TOOL_DESCRIPTIONS",
    "TOOL_CATEGORIES",
]

# 如果直接运行此模块，显示状态信息和示例
if __name__ == "__main__":
    print_tool_status()
    print_tool_categories()
    print_tool_examples()