#!/usr/bin/env python3
"""
文档生成系统快速开始脚本
功能：一键生成项目的完整文档
作者：自动文档生成系统
"""

import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    try:
        import jinja2
    except ImportError:
        missing_deps.append("jinja2")
    
    if missing_deps:
        logger.error("缺少必要依赖，请安装:")
        for dep in missing_deps:
            logger.error(f"  pip install {dep}")
        return False
    
    return True

def generate_documentation():
    """生成完整文档"""
    try:
        # 导入文档生成模块
        from generator import generate_all_docs, check_dependencies as check_gen_deps
        
        logger.info("开始生成文档...")
        
        # 检查生成器依赖
        deps = check_gen_deps()
        logger.info(f"依赖检查: {deps}")
        
        # 配置文件路径
        config_path = Path("docs.yaml")
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            logger.info("使用默认配置生成文档")
            config_path = None
        
        # 生成所有文档
        success = generate_all_docs(
            config_path=config_path,
            output_path="docs/generated/"
        )
        
        if success:
            logger.info("✅ 文档生成成功!")
            logger.info("生成的文档位于: docs/generated/")
            logger.info("可以使用以下命令启动本地服务器查看文档:")
            logger.info("  python -m http.server 8000 --directory docs/generated/")
            return True
        else:
            logger.error("❌ 文档生成失败")
            return False
            
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.error("请确保文档生成模块可用")
        return False
    except Exception as e:
        logger.error(f"生成过程中出错: {e}")
        return False

def main():
    """主函数"""
    logger.info("LangChain学习项目 - 文档生成器")
    logger.info("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 生成文档
    if generate_documentation():
        logger.info("文档生成完成! 🎉")
        
        # 提示后续步骤
        logger.info("\n后续步骤:")
        logger.info("1. 查看生成的文档: docs/generated/")
        logger.info("2. 启动本地服务器预览: ")
        logger.info("   cd docs/generated && python -m http.server 8000")
        logger.info("3. 在浏览器中访问: http://localhost:8000")
        logger.info("4. 自定义配置: 编辑 docs.yaml 文件")
        logger.info("5. 重新生成: 再次运行此脚本")
        
        sys.exit(0)
    else:
        logger.error("文档生成失败")
        sys.exit(1)

if __name__ == "__main__":
    main()