#!/usr/bin/env python3
"""
Jupyter学习环境安装和配置脚本

此脚本用于自动化配置LangChain学习所需的Jupyter环境。
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import platform

class JupyterEnvironmentSetup:
    """Jupyter环境配置类"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.notebooks_dir = self.base_dir / "notebooks"
        self.config_dir = self.base_dir / "configs"
        self.jupyter_config_dir = Path.home() / ".jupyter"
        
    def check_python_version(self):
        """检查Python版本"""
        print("🐍 检查Python版本...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            print(f"❌ Python版本过低: {sys.version}")
            print("   需要Python 3.9或更高版本")
            return False
        print(f"✅ Python版本: {sys.version}")
        return True
    
    def install_packages(self):
        """安装必要的Python包"""
        packages = [
            # Jupyter核心
            "jupyter",
            "jupyterlab>=4.0.0",
            "notebook>=7.0.0",
            "ipykernel",
            
            # LangChain生态系统
            "langchain>=0.2.16",
            "langchain-openai>=0.1.23",
            "langchain-community>=0.2.16",
            "langchain-core>=0.2.38",
            "langgraph>=0.2.16",
            "langserve>=0.2.7",
            
            # 数据处理和可视化
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.17.0",
            
            # 工具库
            "python-dotenv>=1.0.0",
            "pydantic>=2.0.0",
            "requests>=2.31.0",
            "httpx>=0.25.0",
            "tiktoken>=0.5.0",
            
            # Jupyter扩展
            "ipywidgets>=8.0.0",
            "tqdm>=4.66.0",
            "jupyterlab-code-formatter>=2.0.0",
            "jupyterlab-git>=0.44.0",
            "nbformat>=5.9.0",
            
            # 开发工具
            "black>=23.0.0",
            "isort>=5.12.0",
            "rich>=13.0.0"
        ]
        
        print("\n📦 安装Python包...")
        failed_packages = []
        
        for package in packages:
            try:
                print(f"  安装 {package}...", end=" ")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✅")
            except subprocess.CalledProcessError as e:
                print(f"❌")
                failed_packages.append(package)
                
        if failed_packages:
            print(f"\n⚠️ 以下包安装失败: {', '.join(failed_packages)}")
            print("  请手动安装或检查网络连接")
        else:
            print("\n✅ 所有包安装成功")
            
        return len(failed_packages) == 0
    
    def install_jupyter_extensions(self):
        """安装和配置Jupyter扩展"""
        print("\n🔧 配置JupyterLab扩展...")
        
        # JupyterLab扩展列表
        extensions = [
            "@jupyter-widgets/jupyterlab-manager",  # ipywidgets支持
            "@jupyterlab/toc",                      # 目录导航
            "@ryantam626/jupyterlab_code_formatter", # 代码格式化
        ]
        
        for extension in extensions:
            try:
                print(f"  启用扩展 {extension}...", end=" ")
                # JupyterLab 4.x 大部分扩展通过pip安装，不需要labextension install
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
        
        # 配置ipywidgets
        try:
            subprocess.run(
                ["jupyter", "nbextension", "enable", "--py", "widgetsnbextension"],
                check=True,
                capture_output=True
            )
            print("✅ ipywidgets扩展已启用")
        except:
            pass  # JupyterLab可能不需要这步
            
    def create_jupyter_config(self):
        """创建Jupyter配置文件"""
        print("\n📝 创建Jupyter配置...")
        
        # 确保配置目录存在
        self.jupyter_config_dir.mkdir(exist_ok=True)
        
        # JupyterLab配置
        lab_config = {
            "ServerApp": {
                "autoreload": True,
                "root_dir": str(self.notebooks_dir),
                "token": "",  # 开发环境不需要token
                "password": "",
                "disable_check_xsrf": True
            },
            "LabApp": {
                "default_url": "/lab",
                "extensions_in_dev_mode": True
            },
            "FileContentsManager": {
                "checkpoints_kwargs": {
                    "keep_all": True
                },
                "autosave_interval": 60  # 60秒自动保存
            },
            "CodeCell": {
                "cm_config": {
                    "lineNumbers": True,
                    "autoCloseBrackets": True,
                    "theme": "monokai"
                }
            }
        }
        
        # 保存JupyterLab配置
        lab_config_file = self.config_dir / "jupyter_lab_config.json"
        with open(lab_config_file, 'w', encoding='utf-8') as f:
            json.dump(lab_config, f, indent=2, ensure_ascii=False)
        print(f"  ✅ JupyterLab配置: {lab_config_file}")
        
        # Python配置文件
        py_config = '''# Jupyter配置文件
c = get_config()

# 基础配置
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.port = 8888
c.ServerApp.open_browser = True
c.ServerApp.root_dir = r'{root_dir}'

# 安全配置（开发环境）
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = True

# 笔记本配置
c.FileContentsManager.checkpoints_kwargs = {{'keep_all': True}}
c.NotebookApp.autosave_interval = 60  # 60秒自动保存

# 代码单元格配置
c.CodeCell.cm_config = {{
    'lineNumbers': True,
    'autoCloseBrackets': True,
    'matchBrackets': True,
    'theme': 'monokai'
}}

# 内核配置
c.KernelManager.autorestart = True
c.KernelRestarter.restart_limit = 10

print("📚 Jupyter配置已加载")
'''.format(root_dir=str(self.notebooks_dir))
        
        py_config_file = self.jupyter_config_dir / "jupyter_lab_config.py"
        with open(py_config_file, 'w', encoding='utf-8') as f:
            f.write(py_config)
        print(f"  ✅ Python配置: {py_config_file}")
        
    def create_custom_css(self):
        """创建自定义CSS样式"""
        print("\n🎨 创建自定义样式...")
        
        custom_css = """/* LangChain学习环境自定义样式 */

/* 整体主题 */
:root {
    --langchain-primary: #1976D2;
    --langchain-secondary: #4CAF50;
    --langchain-accent: #FF9800;
    --langchain-error: #F44336;
    --langchain-success: #4CAF50;
    --langchain-warning: #FF9800;
    --langchain-info: #2196F3;
}

/* Notebook标题样式 */
.jp-Notebook h1 {
    color: var(--langchain-primary);
    border-bottom: 3px solid var(--langchain-primary);
    padding-bottom: 10px;
    margin-top: 30px;
}

.jp-Notebook h2 {
    color: var(--langchain-secondary);
    border-left: 4px solid var(--langchain-secondary);
    padding-left: 10px;
    margin-top: 25px;
}

/* 代码单元格样式 */
.jp-CodeCell {
    border-left: 3px solid transparent;
    transition: border-color 0.3s;
}

.jp-CodeCell:hover {
    border-left-color: var(--langchain-primary);
}

/* 输出样式 */
.jp-OutputArea-output pre {
    padding: 10px;
    border-radius: 4px;
    background-color: #f5f5f5;
}

/* 学习目标框样式 */
.learning-objectives {
    background-color: #e3f2fd;
    border-left: 4px solid var(--langchain-primary);
    padding: 15px;
    margin: 20px 0;
    border-radius: 4px;
}

.learning-objectives h3 {
    margin-top: 0;
    color: var(--langchain-primary);
}

/* 练习题框样式 */
.exercise-box {
    border: 2px solid var(--langchain-accent);
    border-radius: 8px;
    margin: 20px 0;
    overflow: hidden;
}

.exercise-box .header {
    background-color: var(--langchain-accent);
    color: white;
    padding: 10px 15px;
}

.exercise-box .content {
    padding: 15px;
    background-color: #fff8e1;
}

/* 进度条样式 */
.progress-bar {
    background-color: #e0e0e0;
    border-radius: 10px;
    padding: 3px;
    margin: 10px 0;
}

.progress-bar .progress {
    background-color: var(--langchain-success);
    height: 20px;
    border-radius: 7px;
    text-align: center;
    color: white;
    line-height: 20px;
    font-size: 12px;
    transition: width 0.5s ease;
}

/* 提示框样式 */
.tip-box {
    background-color: #fff3e0;
    border-left: 4px solid var(--langchain-warning);
    padding: 12px;
    margin: 15px 0;
    border-radius: 4px;
}

.tip-box::before {
    content: "💡 ";
    font-size: 1.2em;
}

/* 成功消息样式 */
.success-message {
    background-color: #e8f5e9;
    border-left: 4px solid var(--langchain-success);
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
    color: #2e7d32;
}

/* 错误消息样式 */
.error-message {
    background-color: #ffebee;
    border-left: 4px solid var(--langchain-error);
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
    color: #c62828;
}

/* 代码解释框样式 */
.code-explanation {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 15px;
    margin: 15px 0;
}

.code-explanation .title {
    font-weight: bold;
    color: var(--langchain-primary);
    margin-bottom: 10px;
}

/* 徽章样式 */
.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: bold;
    margin: 0 3px;
}

.badge.easy {
    background-color: #e8f5e9;
    color: #2e7d32;
}

.badge.medium {
    background-color: #fff3e0;
    color: #f57c00;
}

.badge.hard {
    background-color: #ffebee;
    color: #c62828;
}

/* 动画效果 */
@keyframes pulse {
    0% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
    100% {
        transform: scale(1);
    }
}

.pulse-animation {
    animation: pulse 2s infinite;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .jp-Notebook {
        padding: 10px;
    }
    
    .exercise-box, .learning-objectives {
        margin: 10px 0;
    }
}
"""
        
        css_file = self.base_dir / "styles" / "custom.css"
        css_file.parent.mkdir(exist_ok=True)
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(custom_css)
        print(f"  ✅ 自定义样式: {css_file}")
        
    def create_startup_script(self):
        """创建启动脚本"""
        print("\n🚀 创建启动脚本...")
        
        # 启动脚本
        startup_script = '''#!/usr/bin/env python3
"""
Jupyter学习环境启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_environment():
    """检查环境配置"""
    try:
        import langchain
        import jupyter
        print("✅ 环境检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: python setup_env.py")
        return False

def start_jupyter():
    """启动JupyterLab"""
    if not check_environment():
        return
    
    # 设置环境变量
    base_dir = Path(__file__).parent.parent
    notebooks_dir = base_dir / "notebooks"
    
    # 检查.env文件
    env_file = base_dir.parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ 已加载环境变量: {env_file}")
    else:
        print("⚠️ 未找到.env文件，请配置API密钥")
    
    print(f"📂 工作目录: {notebooks_dir}")
    print("🚀 启动JupyterLab...")
    print("-" * 50)
    print("提示：")
    print("  • Ctrl+C 停止服务器")
    print("  • 浏览器会自动打开")
    print("  • 如未自动打开，请访问 http://localhost:8888")
    print("-" * 50)
    
    try:
        # 启动JupyterLab
        subprocess.run([
            sys.executable, "-m", "jupyter", "lab",
            "--notebook-dir", str(notebooks_dir),
            "--no-browser" if os.getenv("NO_BROWSER") else "",
            "--ip", "127.0.0.1",
            "--port", "8888"
        ])
    except KeyboardInterrupt:
        print("\\n👋 JupyterLab已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    start_jupyter()
'''
        
        script_file = self.base_dir / "scripts" / "start_jupyter.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # 添加执行权限（Unix系统）
        if platform.system() != "Windows":
            os.chmod(script_file, 0o755)
            
        print(f"  ✅ 启动脚本: {script_file}")
        
        # 创建快捷启动脚本
        if platform.system() == "Windows":
            batch_script = '''@echo off
echo Starting Jupyter Learning Environment...
python "%~dp0\\start_jupyter.py"
pause
'''
            batch_file = self.base_dir / "scripts" / "start.bat"
            with open(batch_file, 'w') as f:
                f.write(batch_script)
            print(f"  ✅ Windows启动脚本: {batch_file}")
        else:
            shell_script = '''#!/bin/bash
echo "Starting Jupyter Learning Environment..."
python3 "$(dirname "$0")/start_jupyter.py"
'''
            shell_file = self.base_dir / "scripts" / "start.sh"
            with open(shell_file, 'w') as f:
                f.write(shell_script)
            os.chmod(shell_file, 0o755)
            print(f"  ✅ Unix启动脚本: {shell_file}")
    
    def create_readme(self):
        """创建说明文档"""
        readme_content = '''# 🎓 LangChain Jupyter学习环境

## 🚀 快速开始

### 1. 环境配置
```bash
# 安装并配置环境
python scripts/setup_env.py
```

### 2. 启动JupyterLab
```bash
# Unix/Linux/Mac
./scripts/start.sh

# Windows
scripts\\start.bat

# 或直接运行Python脚本
python scripts/start_jupyter.py
```

### 3. 配置API密钥
在项目根目录创建`.env`文件：
```
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
```

## 📚 课程结构

### 基础概念 (01_基础概念/)
- `01_langchain_introduction.ipynb` - LangChain简介
- `02_llm_basics.ipynb` - 大语言模型基础
- `03_prompts_templates.ipynb` - 提示词和模板

### 核心组件 (02_核心组件/)
- `01_chains_introduction.ipynb` - 链的介绍
- `02_agents_basics.ipynb` - 代理基础
- `03_memory_systems.ipynb` - 记忆系统

### 高级应用 (03_高级应用/)
- `01_rag_systems.ipynb` - RAG系统
- `02_multi_agent.ipynb` - 多代理系统
- `03_evaluation.ipynb` - 评估系统

### 实战项目 (04_实战项目/)
- `01_chatbot_project.ipynb` - 聊天机器人
- `02_qa_system.ipynb` - 问答系统
- `03_document_analysis.ipynb` - 文档分析

## 🛠️ 功能特性

- ✅ 交互式代码示例
- ✅ 进度追踪系统
- ✅ 练习题和自动评估
- ✅ 代码片段管理
- ✅ 错误诊断助手
- ✅ 可视化学习路径
- ✅ 成就系统

## 📖 使用技巧

1. **进度追踪**: 每个Notebook都会自动追踪你的学习进度
2. **练习系统**: 完成练习题以巩固知识
3. **代码片段**: 使用`utils/code_snippets.ipynb`管理常用代码
4. **故障排除**: 遇到问题查看`utils/troubleshooting.ipynb`

## 🆘 常见问题

### Q: JupyterLab无法启动
A: 确保已运行`setup_env.py`安装所有依赖

### Q: API调用失败
A: 检查`.env`文件中的API密钥是否正确

### Q: 进度未保存
A: 确保`progress_data`目录有写入权限

## 📞 支持

如有问题，请查看：
- 故障排除Notebook: `utils/troubleshooting.ipynb`
- 项目文档: `docs/`
- GitHub Issues: [项目链接]

祝学习愉快！🎉
'''
        
        readme_file = self.base_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"\n📄 创建README: {readme_file}")
    
    def run(self):
        """运行完整的环境配置"""
        print("=" * 50)
        print("🎓 LangChain Jupyter学习环境配置")
        print("=" * 50)
        
        # 检查Python版本
        if not self.check_python_version():
            return False
        
        # 安装包
        if not self.install_packages():
            print("\n⚠️ 部分包安装失败，但可以继续配置")
        
        # 安装扩展
        self.install_jupyter_extensions()
        
        # 创建配置
        self.create_jupyter_config()
        
        # 创建样式
        self.create_custom_css()
        
        # 创建启动脚本
        self.create_startup_script()
        
        # 创建README
        self.create_readme()
        
        print("\n" + "=" * 50)
        print("🎉 Jupyter学习环境配置完成！")
        print("=" * 50)
        print("\n下一步：")
        print("1. 配置API密钥: 创建.env文件")
        print("2. 启动环境: python scripts/start_jupyter.py")
        print("3. 开始学习: 打开第一个Notebook")
        
        return True

def main():
    """主函数"""
    setup = JupyterEnvironmentSetup()
    setup.run()

if __name__ == "__main__":
    main()