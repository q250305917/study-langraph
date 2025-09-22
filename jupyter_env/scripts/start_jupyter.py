#!/usr/bin/env python3
"""
Jupyter学习环境启动脚本

用于启动LangChain学习的Jupyter环境，包括环境检查、配置加载和服务启动。
"""

import os
import sys
import subprocess
import webbrowser
import time
import signal
from pathlib import Path
from datetime import datetime

def check_environment():
    """检查环境配置和依赖"""
    print("🔍 检查学习环境...")
    
    issues = []
    
    # 检查Python版本
    if sys.version_info < (3, 9):
        issues.append(f"Python版本过低: {sys.version}")
    else:
        print(f"✅ Python版本: {sys.version.split()[0]}")
    
    # 检查核心依赖
    required_packages = {
        'jupyter': 'Jupyter',
        'langchain': 'LangChain',
        'langchain_openai': 'LangChain OpenAI',
        'python_dotenv': '环境变量管理',
        'matplotlib': '数据可视化',
        'pandas': '数据处理'
    }
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {description}: 已安装")
        except ImportError:
            issues.append(f"缺少依赖: {description} ({package})")
    
    # 检查API密钥配置
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        
        api_keys = {
            'OPENAI_API_KEY': 'OpenAI',
            'ANTHROPIC_API_KEY': 'Anthropic'
        }
        
        found_keys = []
        for key, provider in api_keys.items():
            if os.getenv(key):
                found_keys.append(provider)
        
        if found_keys:
            print(f"✅ API密钥: {', '.join(found_keys)}")
        else:
            issues.append("未配置API密钥（可选，部分功能需要）")
    else:
        issues.append("未找到.env文件（可选，用于API密钥配置）")
    
    # 显示问题和解决方案
    if issues:
        print("\n⚠️ 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 解决方案:")
        print("   • 运行环境配置脚本: python scripts/setup_env.py")
        print("   • 配置API密钥: 在项目根目录创建.env文件")
        print("   • 安装缺失依赖: pip install [package_name]")
        
        choice = input("\n是否仍要继续启动？(y/n): ")
        if choice.lower() != 'y':
            return False
    
    print("✅ 环境检查完成")
    return True

def setup_environment():
    """设置环境变量和路径"""
    base_dir = Path(__file__).parent.parent
    notebooks_dir = base_dir / "notebooks"
    
    # 确保notebook目录存在
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    env_vars = {
        'JUPYTER_CONFIG_DIR': str(base_dir / "configs"),
        'JUPYTER_DATA_DIR': str(base_dir / "data"),
        'JUPYTER_RUNTIME_DIR': str(base_dir / "runtime"),
        'PYTHONPATH': str(notebooks_dir / "utils")
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # 创建必要的目录
    for dir_path in env_vars.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return notebooks_dir

def create_welcome_page():
    """创建欢迎页面"""
    welcome_content = """
# 🎓 欢迎来到LangChain学习环境！

## 📚 课程导航

### 🌟 基础概念
- **01_langchain_introduction.ipynb** - LangChain基础介绍
- **02_llm_basics.ipynb** - 大语言模型基础
- **03_prompts_templates.ipynb** - 提示词和模板

### 🔧 核心组件  
- **01_chains_introduction.ipynb** - 链的介绍
- **02_agents_basics.ipynb** - 代理基础
- **03_memory_systems.ipynb** - 记忆系统

### 🚀 高级应用
- **01_rag_systems.ipynb** - RAG系统
- **02_multi_agent.ipynb** - 多代理系统
- **03_evaluation.ipynb** - 评估系统

### 💼 实战项目
- **01_chatbot_project.ipynb** - 聊天机器人项目
- **02_qa_system.ipynb** - 问答系统项目
- **03_document_analysis.ipynb** - 文档分析项目

## 🛠️ 工具

### 📈 学习工具
- **progress_tracker.py** - 进度追踪器
- **code_snippets.ipynb** - 代码片段库
- **troubleshooting.ipynb** - 故障排除指南

## 🎯 学习建议

1. **按顺序学习**: 从基础概念开始，逐步进阶
2. **动手实践**: 运行每个代码示例，完成练习题
3. **记录笔记**: 使用Markdown记录重要概念
4. **定期复习**: 查看进度追踪器，巩固学习成果

## 🆘 需要帮助？

- 📖 查看 `utils/troubleshooting.ipynb`
- 🔍 搜索LangChain官方文档
- 💬 参与社区讨论

---

**祝你学习愉快！🎉**
"""
    
    welcome_file = Path(__file__).parent.parent / "notebooks" / "README.md"
    with open(welcome_file, 'w', encoding='utf-8') as f:
        f.write(welcome_content)
    
    return welcome_file

def start_jupyter_lab(notebooks_dir, port=8888):
    """启动JupyterLab服务"""
    
    # 创建欢迎页面
    welcome_file = create_welcome_page()
    
    # 准备启动命令
    cmd = [
        sys.executable, "-m", "jupyter", "lab",
        "--notebook-dir", str(notebooks_dir),
        "--ip", "127.0.0.1",
        "--port", str(port),
        "--no-browser" if os.getenv("NO_BROWSER") else "",
        "--allow-root",
        "--NotebookApp.token=''",
        "--NotebookApp.password=''",
        "--NotebookApp.disable_check_xsrf=True"
    ]
    
    # 移除空字符串
    cmd = [arg for arg in cmd if arg]
    
    print(f"📂 工作目录: {notebooks_dir}")
    print(f"🌐 服务地址: http://127.0.0.1:{port}")
    print("🚀 启动JupyterLab...")
    print("-" * 50)
    print("💡 使用提示：")
    print("  • Ctrl+C 停止服务器")
    print("  • 浏览器会自动打开学习环境")
    print("  • 建议从 '01_基础概念' 开始学习")
    print("-" * 50)
    
    try:
        # 启动JupyterLab
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 等待服务启动
        server_started = False
        start_time = time.time()
        
        while time.time() - start_time < 30:  # 30秒超时
            if process.poll() is not None:
                break
                
            try:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                    
                    # 检测服务是否启动
                    if "http://127.0.0.1:" in line and not server_started:
                        server_started = True
                        
                        # 自动打开浏览器
                        if not os.getenv("NO_BROWSER"):
                            time.sleep(2)  # 等待服务完全启动
                            try:
                                webbrowser.open(f"http://127.0.0.1:{port}")
                                print(f"✅ 已在浏览器中打开学习环境")
                            except:
                                print(f"⚠️ 无法自动打开浏览器，请手动访问: http://127.0.0.1:{port}")
                        
                        break
            except:
                break
        
        if not server_started:
            print("❌ JupyterLab启动失败")
            return False
        
        # 保持运行并处理输出
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.strip())
        except KeyboardInterrupt:
            print("\n📝 正在关闭JupyterLab...")
            process.terminate()
            
            # 等待进程结束
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            print("👋 JupyterLab已停止")
            return True
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

def show_startup_banner():
    """显示启动横幅"""
    banner = f"""
{'='*60}
🎓 LangChain Jupyter学习环境
{'='*60}

📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🐍 Python版本: {sys.version.split()[0]}
📁 项目路径: {Path(__file__).parent.parent}

正在启动学习环境...

"""
    print(banner)

def handle_signal(signum, frame):
    """处理信号"""
    print(f"\n收到信号 {signum}，正在安全退出...")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        # 显示启动横幅
        show_startup_banner()
        
        # 检查环境
        if not check_environment():
            print("❌ 环境检查失败，无法启动")
            return 1
        
        # 设置环境
        notebooks_dir = setup_environment()
        
        # 查找可用端口
        port = 8888
        for test_port in range(8888, 8898):
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', test_port))
                sock.close()
                if result != 0:
                    port = test_port
                    break
            except:
                continue
        
        # 启动JupyterLab
        success = start_jupyter_lab(notebooks_dir, port)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n👋 用户取消启动")
        return 0
    except Exception as e:
        print(f"❌ 启动过程中出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)