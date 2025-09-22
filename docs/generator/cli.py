#!/usr/bin/env python3
"""
文档生成命令行工具
功能：提供doc-gen命令行接口，支持API文档、教程生成和多格式输出
作者：自动文档生成系统
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional
import logging

# 导入本地模块
from .config import ConfigManager, DocumentationConfig, create_config_template
from .api_generator import APIDocumentationGenerator, APIDocumentationConfig
from .tutorial_generator import TutorialGenerator, TutorialConfig
from .formatters.base_formatter import FormatterConfig, create_multi_formatter
from .formatters.markdown_formatter import MarkdownFormatter
from .formatters.html_formatter import HTMLFormatter
from .formatters.pdf_formatter import PDFFormatter, check_pdf_dependencies

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentationCLI:
    """文档生成命令行工具主类"""
    
    def __init__(self):
        """初始化CLI工具"""
        self.config_manager: Optional[ConfigManager] = None
        self.start_time = time.time()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """
        创建命令行参数解析器
        
        Returns:
            参数解析器
        """
        parser = argparse.ArgumentParser(
            prog='doc-gen',
            description='自动化文档生成工具',
            epilog='更多信息请访问: https://github.com/example/doc-generator'
        )
        
        # 全局选项
        parser.add_argument(
            '--config', '-c',
            type=Path,
            help='配置文件路径 (默认: docs.yaml)'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='启用详细输出'
        )
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='静默模式（仅输出错误）'
        )
        parser.add_argument(
            '--version',
            action='version',
            version='doc-gen 1.0.0'
        )
        
        # 创建子命令
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # generate 命令
        self._add_generate_parser(subparsers)
        
        # api 命令
        self._add_api_parser(subparsers)
        
        # tutorial 命令
        self._add_tutorial_parser(subparsers)
        
        # config 命令
        self._add_config_parser(subparsers)
        
        # init 命令
        self._add_init_parser(subparsers)
        
        # watch 命令
        self._add_watch_parser(subparsers)
        
        # serve 命令
        self._add_serve_parser(subparsers)
        
        return parser
    
    def _add_generate_parser(self, subparsers):
        """添加generate子命令"""
        generate_parser = subparsers.add_parser(
            'generate',
            help='生成所有文档',
            description='生成API文档和教程文档'
        )
        generate_parser.add_argument(
            '--all', '-a',
            action='store_true',
            help='生成所有类型的文档'
        )
        generate_parser.add_argument(
            '--api',
            action='store_true',
            help='生成API文档'
        )
        generate_parser.add_argument(
            '--tutorials',
            action='store_true',
            help='生成教程文档'
        )
        generate_parser.add_argument(
            '--format', '-f',
            choices=['markdown', 'html', 'pdf'],
            action='append',
            help='输出格式 (可多选)'
        )
        generate_parser.add_argument(
            '--output', '-o',
            type=Path,
            help='输出目录'
        )
        generate_parser.add_argument(
            '--clean',
            action='store_true',
            help='生成前清理输出目录'
        )
    
    def _add_api_parser(self, subparsers):
        """添加api子命令"""
        api_parser = subparsers.add_parser(
            'api',
            help='生成API文档',
            description='从Python代码生成API参考文档'
        )
        api_parser.add_argument(
            'source',
            type=Path,
            nargs='?',
            help='源代码路径 (文件或目录)'
        )
        api_parser.add_argument(
            '--modules',
            type=str,
            action='append',
            help='要处理的模块名称'
        )
        api_parser.add_argument(
            '--include-private',
            action='store_true',
            help='包含私有成员'
        )
        api_parser.add_argument(
            '--exclude-source',
            action='store_true',
            help='排除源代码链接'
        )
        api_parser.add_argument(
            '--no-inheritance',
            action='store_true',
            help='不包含继承关系'
        )
        api_parser.add_argument(
            '--output', '-o',
            type=Path,
            help='输出目录'
        )
    
    def _add_tutorial_parser(self, subparsers):
        """添加tutorial子命令"""
        tutorial_parser = subparsers.add_parser(
            'tutorial',
            help='生成教程文档',
            description='从示例代码生成教程文档'
        )
        tutorial_parser.add_argument(
            'examples',
            type=Path,
            nargs='?',
            help='示例代码目录'
        )
        tutorial_parser.add_argument(
            '--include-output',
            action='store_true',
            help='包含代码执行结果'
        )
        tutorial_parser.add_argument(
            '--execute',
            action='store_true',
            help='执行代码获取输出'
        )
        tutorial_parser.add_argument(
            '--language',
            choices=['zh', 'en'],
            default='zh',
            help='教程语言'
        )
        tutorial_parser.add_argument(
            '--output', '-o',
            type=Path,
            help='输出目录'
        )
    
    def _add_config_parser(self, subparsers):
        """添加config子命令"""
        config_parser = subparsers.add_parser(
            'config',
            help='配置管理',
            description='管理文档生成配置'
        )
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        
        # init 子命令
        config_subparsers.add_parser('init', help='创建默认配置文件')
        
        # show 子命令
        config_subparsers.add_parser('show', help='显示当前配置')
        
        # validate 子命令
        config_subparsers.add_parser('validate', help='验证配置文件')
        
        # set 子命令
        set_parser = config_subparsers.add_parser('set', help='设置配置项')
        set_parser.add_argument('key', help='配置键')
        set_parser.add_argument('value', help='配置值')
    
    def _add_init_parser(self, subparsers):
        """添加init子命令"""
        init_parser = subparsers.add_parser(
            'init',
            help='初始化项目',
            description='在当前目录初始化文档项目'
        )
        init_parser.add_argument(
            '--name',
            help='项目名称'
        )
        init_parser.add_argument(
            '--template',
            choices=['basic', 'advanced', 'api-only', 'tutorial-only'],
            default='basic',
            help='项目模板'
        )
        init_parser.add_argument(
            '--force',
            action='store_true',
            help='强制覆盖现有文件'
        )
    
    def _add_watch_parser(self, subparsers):
        """添加watch子命令"""
        watch_parser = subparsers.add_parser(
            'watch',
            help='监视文件变化',
            description='监视源文件变化并自动重新生成文档'
        )
        watch_parser.add_argument(
            '--source',
            type=Path,
            help='要监视的源目录'
        )
        watch_parser.add_argument(
            '--interval',
            type=int,
            default=2,
            help='检查间隔（秒）'
        )
    
    def _add_serve_parser(self, subparsers):
        """添加serve子命令"""
        serve_parser = subparsers.add_parser(
            'serve',
            help='启动文档服务器',
            description='启动本地HTTP服务器查看生成的文档'
        )
        serve_parser.add_argument(
            '--port', '-p',
            type=int,
            default=8000,
            help='服务器端口'
        )
        serve_parser.add_argument(
            '--host',
            default='localhost',
            help='服务器主机'
        )
        serve_parser.add_argument(
            '--docs-dir',
            type=Path,
            help='文档目录'
        )
    
    def setup_logging(self, args):
        """
        设置日志配置
        
        Args:
            args: 命令行参数
        """
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
    
    def load_config(self, config_path: Optional[Path] = None) -> DocumentationConfig:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            文档配置对象
        """
        self.config_manager = ConfigManager(config_path)
        return self.config_manager.get_config()
    
    def cmd_generate(self, args) -> int:
        """执行generate命令"""
        try:
            config = self.load_config(args.config)
            
            # 确定生成类型
            generate_api = args.all or args.api or not (args.tutorials)
            generate_tutorials = args.all or args.tutorials or not (args.api)
            
            # 确定输出格式
            if args.format:
                formats = args.format
            else:
                formats = config.output.formats
            
            # 确定输出目录
            output_dir = args.output or Path(config.output.directory)
            
            # 清理输出目录
            if args.clean and output_dir.exists():
                import shutil
                shutil.rmtree(output_dir)
                logger.info(f"Cleaned output directory: {output_dir}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            success = True
            
            # 生成API文档
            if generate_api:
                logger.info("生成API文档...")
                api_success = self._generate_api_docs(config, output_dir, formats)
                success = success and api_success
            
            # 生成教程文档
            if generate_tutorials:
                logger.info("生成教程文档...")
                tutorial_success = self._generate_tutorial_docs(config, output_dir, formats)
                success = success and tutorial_success
            
            if success:
                logger.info("文档生成完成！")
                return 0
            else:
                logger.error("文档生成过程中出现错误")
                return 1
                
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return 1
    
    def cmd_api(self, args) -> int:
        """执行api命令"""
        try:
            config = self.load_config(args.config)
            
            # 创建API配置
            api_config = APIDocumentationConfig(
                include_private=args.include_private,
                include_source=not args.exclude_source,
                include_inheritance=not args.no_inheritance
            )
            
            # 确定源路径
            if args.source:
                source_path = args.source
            else:
                source_paths = self.config_manager.get_source_paths()
                if not source_paths:
                    logger.error("未指定源路径且配置中无有效源目录")
                    return 1
                source_path = source_paths[0]  # 使用第一个源目录
            
            # 确定输出目录
            output_dir = args.output or Path(config.output.directory) / "api"
            
            # 生成API文档
            generator = APIDocumentationGenerator(api_config)
            success = generator.generate_documentation(source_path, output_dir)
            
            if success:
                logger.info(f"API文档生成完成: {output_dir}")
                return 0
            else:
                logger.error("API文档生成失败")
                return 1
                
        except Exception as e:
            logger.error(f"API文档生成失败: {e}")
            return 1
    
    def cmd_tutorial(self, args) -> int:
        """执行tutorial命令"""
        try:
            config = self.load_config(args.config)
            
            # 创建教程配置
            tutorial_config = TutorialConfig(
                include_output=args.include_output,
                code_execution=args.execute,
                language=args.language
            )
            
            # 确定示例目录
            if args.examples:
                examples_dir = args.examples
            else:
                examples_dir = Path("examples")
                if not examples_dir.exists():
                    logger.error("未指定示例目录且默认examples目录不存在")
                    return 1
            
            # 确定输出目录
            output_dir = args.output or Path(config.output.directory) / "tutorials"
            
            # 生成教程文档
            generator = TutorialGenerator(tutorial_config)
            success = generator.generate_tutorial_documentation(examples_dir, output_dir)
            
            if success:
                logger.info(f"教程文档生成完成: {output_dir}")
                return 0
            else:
                logger.error("教程文档生成失败")
                return 1
                
        except Exception as e:
            logger.error(f"教程文档生成失败: {e}")
            return 1
    
    def cmd_config(self, args) -> int:
        """执行config命令"""
        try:
            if args.config_action == 'init':
                config_path = Path("docs.yaml")
                create_config_template(config_path)
                logger.info(f"配置模板已创建: {config_path}")
                
            elif args.config_action == 'show':
                config = self.load_config(args.config)
                self._print_config_summary(config)
                
            elif args.config_action == 'validate':
                self.config_manager = ConfigManager(args.config)
                errors = self.config_manager.validate_config()
                if errors:
                    logger.error("配置验证失败:")
                    for error in errors:
                        logger.error(f"  - {error}")
                    return 1
                else:
                    logger.info("配置验证通过")
                    
            elif args.config_action == 'set':
                self.config_manager = ConfigManager(args.config)
                # 简单的配置设置（可以扩展）
                logger.info(f"设置 {args.key} = {args.value}")
                
            return 0
            
        except Exception as e:
            logger.error(f"配置命令失败: {e}")
            return 1
    
    def cmd_init(self, args) -> int:
        """执行init命令"""
        try:
            # 创建基本目录结构
            directories = [
                "docs",
                "docs/templates",
                "docs/assets",
                "src",
                "examples"
            ]
            
            for dir_name in directories:
                dir_path = Path(dir_name)
                if not dir_path.exists() or args.force:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"创建目录: {dir_path}")
            
            # 创建配置文件
            config_path = Path("docs.yaml")
            if not config_path.exists() or args.force:
                create_config_template(config_path)
                logger.info(f"创建配置文件: {config_path}")
            
            # 创建示例文件
            self._create_example_files(args.force)
            
            logger.info("项目初始化完成!")
            return 0
            
        except Exception as e:
            logger.error(f"项目初始化失败: {e}")
            return 1
    
    def cmd_watch(self, args) -> int:
        """执行watch命令"""
        try:
            import time
            import os
            
            config = self.load_config(args.config)
            source_dir = args.source or Path("src")
            
            if not source_dir.exists():
                logger.error(f"源目录不存在: {source_dir}")
                return 1
            
            logger.info(f"开始监视目录: {source_dir}")
            logger.info("按 Ctrl+C 停止监视")
            
            last_modified = {}
            
            while True:
                try:
                    # 检查文件变化
                    current_modified = {}
                    for file_path in source_dir.rglob("*.py"):
                        stat = file_path.stat()
                        current_modified[str(file_path)] = stat.st_mtime
                    
                    # 检测变化
                    changed_files = []
                    for file_path, mtime in current_modified.items():
                        if file_path not in last_modified or last_modified[file_path] != mtime:
                            changed_files.append(file_path)
                    
                    if changed_files:
                        logger.info(f"检测到文件变化: {len(changed_files)} 个文件")
                        
                        # 重新生成文档
                        fake_args = argparse.Namespace(
                            config=args.config,
                            all=True,
                            api=False,
                            tutorials=False,
                            format=None,
                            output=None,
                            clean=False
                        )
                        self.cmd_generate(fake_args)
                    
                    last_modified = current_modified
                    time.sleep(args.interval)
                    
                except KeyboardInterrupt:
                    logger.info("停止监视")
                    break
            
            return 0
            
        except Exception as e:
            logger.error(f"监视失败: {e}")
            return 1
    
    def cmd_serve(self, args) -> int:
        """执行serve命令"""
        try:
            import http.server
            import socketserver
            import webbrowser
            
            # 确定文档目录
            docs_dir = args.docs_dir or Path("docs/generated")
            
            if not docs_dir.exists():
                logger.error(f"文档目录不存在: {docs_dir}")
                logger.info("请先生成文档: doc-gen generate")
                return 1
            
            # 切换到文档目录
            os.chdir(docs_dir)
            
            # 启动服务器
            handler = http.server.SimpleHTTPRequestHandler
            
            with socketserver.TCPServer((args.host, args.port), handler) as httpd:
                url = f"http://{args.host}:{args.port}"
                logger.info(f"文档服务器启动: {url}")
                logger.info("按 Ctrl+C 停止服务器")
                
                # 自动打开浏览器
                try:
                    webbrowser.open(url)
                except:
                    pass
                
                httpd.serve_forever()
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("服务器已停止")
            return 0
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            return 1
    
    def _generate_api_docs(self, config: DocumentationConfig, 
                          output_dir: Path, formats: List[str]) -> bool:
        """生成API文档"""
        try:
            api_config = APIDocumentationConfig(
                include_private=config.api_docs.include_private,
                include_source=config.api_docs.include_source_links,
                include_inheritance=config.api_docs.include_inheritance,
                include_examples=config.api_docs.include_examples,
                sort_members=config.api_docs.sort_members
            )
            
            generator = APIDocumentationGenerator(api_config)
            
            # 处理每个源目录
            source_paths = self.config_manager.get_source_paths()
            success = True
            
            for source_path in source_paths:
                api_output_dir = output_dir / "api"
                if not generator.generate_documentation(source_path, api_output_dir):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"API文档生成失败: {e}")
            return False
    
    def _generate_tutorial_docs(self, config: DocumentationConfig, 
                               output_dir: Path, formats: List[str]) -> bool:
        """生成教程文档"""
        try:
            tutorial_config = TutorialConfig(
                auto_generate=config.tutorials.auto_generate,
                include_output=config.tutorials.include_output,
                code_execution=config.tutorials.execute_code,
                language=config.tutorials.language,
                difficulty_levels=config.tutorials.difficulty_levels,
                format_style=config.tutorials.format_style
            )
            
            generator = TutorialGenerator(tutorial_config)
            
            # 查找示例目录
            examples_dir = Path("examples")
            if not examples_dir.exists():
                logger.warning("示例目录不存在，跳过教程生成")
                return True
            
            tutorial_output_dir = output_dir / "tutorials"
            return generator.generate_tutorial_documentation(examples_dir, tutorial_output_dir)
            
        except Exception as e:
            logger.error(f"教程文档生成失败: {e}")
            return False
    
    def _print_config_summary(self, config: DocumentationConfig):
        """打印配置摘要"""
        print(f"项目名称: {config.project_name}")
        print(f"项目版本: {config.project_version}")
        print(f"源目录: {config.source.directories}")
        print(f"输出目录: {config.output.directory}")
        print(f"输出格式: {config.output.formats}")
        print(f"包含私有成员: {config.api_docs.include_private}")
        print(f"生成教程: {config.tutorials.auto_generate}")
        print(f"教程语言: {config.tutorials.language}")
    
    def _create_example_files(self, force: bool = False):
        """创建示例文件"""
        # 创建示例Python文件
        example_py = Path("src/example_module.py")
        if not example_py.exists() or force:
            example_py.parent.mkdir(parents=True, exist_ok=True)
            with open(example_py, 'w', encoding='utf-8') as f:
                f.write('''"""
示例模块
功能：演示文档生成系统的使用
"""

class ExampleClass:
    """示例类，用于演示API文档生成"""
    
    def __init__(self, name: str):
        """
        初始化示例类
        
        Args:
            name: 实例名称
        """
        self.name = name
    
    def greet(self, message: str = "Hello") -> str:
        """
        返回问候消息
        
        Args:
            message: 问候消息
            
        Returns:
            格式化的问候字符串
            
        Examples:
            >>> obj = ExampleClass("World")
            >>> obj.greet()
            'Hello, World!'
        """
        return f"{message}, {self.name}!"


def example_function(x: int, y: int) -> int:
    """
    示例函数，计算两个数的和
    
    Args:
        x: 第一个数
        y: 第二个数
        
    Returns:
        两数之和
    """
    return x + y
''')
            logger.info(f"创建示例文件: {example_py}")
        
        # 创建示例教程文件
        tutorial_py = Path("examples/basic_example.py")
        if not tutorial_py.exists() or force:
            tutorial_py.parent.mkdir(parents=True, exist_ok=True)
            with open(tutorial_py, 'w', encoding='utf-8') as f:
                f.write('''"""
基础教程示例
标题：快速开始指南
作者：文档生成系统
难度：初级
时间：10分钟
类型：quick_start
标签：基础,入门,快速开始
"""

# 第1步：导入模块
# 首先我们需要导入必要的模块

from src.example_module import ExampleClass, example_function

# 第2步：创建实例
# 创建一个ExampleClass的实例

example = ExampleClass("LangChain")

# 第3步：调用方法
# 使用实例的方法

greeting = example.greet("欢迎使用")
print(greeting)

# 第4步：使用函数
# 调用示例函数

result = example_function(10, 20)
print(f"计算结果: {result}")

# 结论
# 通过这个简单的示例，你已经学会了基础用法
''')
            logger.info(f"创建教程文件: {tutorial_py}")
    
    def run(self, args: List[str] = None) -> int:
        """
        运行CLI工具
        
        Args:
            args: 命令行参数列表，如果为None则使用sys.argv
            
        Returns:
            退出码
        """
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # 设置日志
        self.setup_logging(parsed_args)
        
        # 检查命令
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        # 执行命令
        command_map = {
            'generate': self.cmd_generate,
            'api': self.cmd_api,
            'tutorial': self.cmd_tutorial,
            'config': self.cmd_config,
            'init': self.cmd_init,
            'watch': self.cmd_watch,
            'serve': self.cmd_serve
        }
        
        command_func = command_map.get(parsed_args.command)
        if command_func:
            try:
                result = command_func(parsed_args)
                
                # 显示执行时间
                elapsed = time.time() - self.start_time
                logger.info(f"执行时间: {elapsed:.2f}秒")
                
                return result
            except KeyboardInterrupt:
                logger.info("操作被用户中断")
                return 130
        else:
            logger.error(f"未知命令: {parsed_args.command}")
            return 1


def main():
    """主入口函数"""
    cli = DocumentationCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()