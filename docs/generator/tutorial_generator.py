"""
教程文档生成器
功能：基于示例代码和注释自动生成教程文档，支持步骤化教学和代码演示
作者：自动文档生成系统
"""

import re
import ast
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# 导入本地模块
from .code_parser import CodeParser, CodeElement, NodeType
from .template_engine import TemplateEngine, TemplateManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleType(Enum):
    """示例类型枚举"""
    BASIC = "basic"           # 基础示例
    ADVANCED = "advanced"     # 高级示例
    TUTORIAL = "tutorial"     # 教程示例
    QUICK_START = "quick_start"  # 快速开始
    USE_CASE = "use_case"     # 用例演示


@dataclass
class TutorialConfig:
    """教程生成配置"""
    include_output: bool = True      # 是否包含代码执行结果
    auto_generate: bool = True       # 是否自动生成教程结构
    language: str = "zh"            # 教程语言
    difficulty_levels: List[str] = None  # 难度级别
    include_prerequisites: bool = True   # 是否包含前置要求
    code_execution: bool = False     # 是否执行代码获取输出
    format_style: str = "step_by_step"  # 格式风格
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ["初级", "中级", "高级"]


@dataclass
class CodeBlock:
    """代码块数据结构"""
    content: str
    language: str = "python"
    line_start: int = 1
    line_end: int = 1
    description: Optional[str] = None
    expected_output: Optional[str] = None
    is_executable: bool = True
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class TutorialSection:
    """教程章节数据结构"""
    title: str
    description: str
    code_blocks: List[CodeBlock]
    subsections: List['TutorialSection'] = None
    order: int = 0
    difficulty: str = "初级"
    prerequisites: List[str] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class ExampleMetadata:
    """示例元数据"""
    title: str
    description: str
    author: str = ""
    created_date: str = ""
    updated_date: str = ""
    version: str = "1.0"
    tags: List[str] = None
    difficulty: str = "初级"
    example_type: ExampleType = ExampleType.BASIC
    prerequisites: List[str] = None
    estimated_time: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class TutorialDocument:
    """教程文档数据结构"""
    metadata: ExampleMetadata
    sections: List[TutorialSection]
    introduction: str = ""
    conclusion: str = ""
    references: List[str] = None
    
    def __post_init__(self):
        if self.references is None:
            self.references = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class ExampleParser:
    """示例代码解析器"""
    
    def __init__(self):
        """初始化示例解析器"""
        self.code_parser = CodeParser()
    
    def parse_example_file(self, file_path: Path) -> Optional[TutorialDocument]:
        """
        解析示例文件
        
        Args:
            file_path: 示例文件路径
            
        Returns:
            教程文档对象或None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取元数据
            metadata = self._extract_metadata(content, file_path)
            
            # 解析代码和注释
            sections = self._parse_content(content, file_path)
            
            # 提取介绍和结论
            introduction, conclusion = self._extract_intro_conclusion(content)
            
            return TutorialDocument(
                metadata=metadata,
                sections=sections,
                introduction=introduction,
                conclusion=conclusion
            )
            
        except Exception as e:
            logger.error(f"Error parsing example file {file_path}: {e}")
            return None
    
    def _extract_metadata(self, content: str, file_path: Path) -> ExampleMetadata:
        """
        从文件内容中提取元数据
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            示例元数据
        """
        metadata = ExampleMetadata(
            title=file_path.stem.replace('_', ' ').title(),
            description=""
        )
        
        # 尝试从文件头部注释提取元数据
        lines = content.split('\n')
        in_header = False
        
        for line in lines:
            stripped = line.strip()
            
            # 检测文件头部注释
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_header = not in_header
                continue
            
            if in_header or stripped.startswith('#'):
                # 提取各种元数据字段
                if self._extract_field(stripped, ['title:', '标题:', 'Title:']):
                    metadata.title = self._extract_field_value(stripped)
                elif self._extract_field(stripped, ['description:', '描述:', 'Description:']):
                    metadata.description = self._extract_field_value(stripped)
                elif self._extract_field(stripped, ['author:', '作者:', 'Author:']):
                    metadata.author = self._extract_field_value(stripped)
                elif self._extract_field(stripped, ['difficulty:', '难度:', 'Difficulty:']):
                    metadata.difficulty = self._extract_field_value(stripped)
                elif self._extract_field(stripped, ['type:', '类型:', 'Type:']):
                    type_value = self._extract_field_value(stripped).lower()
                    try:
                        metadata.example_type = ExampleType(type_value)
                    except ValueError:
                        metadata.example_type = ExampleType.BASIC
                elif self._extract_field(stripped, ['time:', '时间:', 'Time:']):
                    metadata.estimated_time = self._extract_field_value(stripped)
                elif self._extract_field(stripped, ['tags:', '标签:', 'Tags:']):
                    tags_str = self._extract_field_value(stripped)
                    metadata.tags = [tag.strip() for tag in tags_str.split(',')]
        
        # 如果没有找到描述，使用文件的第一个文档字符串
        if not metadata.description:
            try:
                tree = ast.parse(content)
                if (tree.body and isinstance(tree.body[0], ast.Expr) and
                    isinstance(tree.body[0].value, ast.Constant)):
                    metadata.description = tree.body[0].value.value.strip()
            except:
                pass
        
        return metadata
    
    def _extract_field(self, line: str, patterns: List[str]) -> bool:
        """检查行是否包含指定的字段模式"""
        line_lower = line.lower()
        return any(pattern.lower() in line_lower for pattern in patterns)
    
    def _extract_field_value(self, line: str) -> str:
        """从字段行中提取值"""
        # 移除注释符号和字段名
        cleaned = re.sub(r'^[#\'"]+\s*', '', line)
        if ':' in cleaned:
            return cleaned.split(':', 1)[1].strip()
        return cleaned.strip()
    
    def _parse_content(self, content: str, file_path: Path) -> List[TutorialSection]:
        """
        解析文件内容，提取教程章节
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            教程章节列表
        """
        sections = []
        lines = content.split('\n')
        current_section = None
        current_code_block = []
        current_description = []
        in_code = False
        in_comment_block = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 检测章节标题（注释中的#号或特殊格式）
            if self._is_section_header(stripped):
                # 保存当前章节
                if current_section:
                    if current_code_block:
                        current_section.code_blocks.append(self._create_code_block(
                            '\n'.join(current_code_block), '\n'.join(current_description)
                        ))
                    sections.append(current_section)
                
                # 创建新章节
                title = self._extract_section_title(stripped)
                current_section = TutorialSection(
                    title=title,
                    description="",
                    code_blocks=[],
                    order=len(sections)
                )
                current_code_block = []
                current_description = []
                in_code = False
            
            # 检测代码块分隔
            elif stripped.startswith('# ') and not self._is_section_header(stripped):
                # 代码注释，作为描述
                if current_code_block:
                    # 保存当前代码块
                    if current_section:
                        current_section.code_blocks.append(self._create_code_block(
                            '\n'.join(current_code_block), '\n'.join(current_description)
                        ))
                    current_code_block = []
                    current_description = []
                
                current_description.append(stripped[2:])  # 移除'# '
                in_code = False
            
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                # 文档字符串块
                in_comment_block = not in_comment_block
                if not in_comment_block and current_description:
                    # 文档字符串结束，收集描述
                    pass
            
            elif in_comment_block:
                # 在文档字符串内
                current_description.append(stripped)
            
            elif stripped and not stripped.startswith('#'):
                # 代码行
                current_code_block.append(line)
                in_code = True
            
            elif not stripped and in_code:
                # 空行，但在代码中
                current_code_block.append(line)
            
            i += 1
        
        # 保存最后一个章节
        if current_section:
            if current_code_block:
                current_section.code_blocks.append(self._create_code_block(
                    '\n'.join(current_code_block), '\n'.join(current_description)
                ))
            sections.append(current_section)
        
        # 如果没有找到明确的章节，创建默认章节
        if not sections:
            sections = self._create_default_sections(content)
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """判断是否为章节标题"""
        patterns = [
            r'^#+\s+[第\d]+[章节步]\s*[:：]',  # # 第1章：
            r'^#+\s+[步骤第]*\d+[.、：:]',     # # 步骤1：
            r'^#+\s+[STEP步骤]\s*\d+',        # # STEP 1
            r'^#+\s+[一二三四五六七八九十]+[、.：:]',  # # 一、
        ]
        
        return any(re.match(pattern, line, re.IGNORECASE) for pattern in patterns)
    
    def _extract_section_title(self, line: str) -> str:
        """从标题行提取章节标题"""
        # 移除#号和多余空格
        title = re.sub(r'^#+\s*', '', line).strip()
        return title
    
    def _create_code_block(self, code: str, description: str) -> CodeBlock:
        """创建代码块对象"""
        return CodeBlock(
            content=code.strip(),
            description=description.strip() if description else None,
            is_executable=self._is_executable_code(code)
        )
    
    def _is_executable_code(self, code: str) -> bool:
        """判断代码是否可执行"""
        try:
            # 尝试解析代码
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _create_default_sections(self, content: str) -> List[TutorialSection]:
        """创建默认章节结构"""
        # 如果没有明确章节，创建单个主要章节
        section = TutorialSection(
            title="主要示例",
            description="代码示例演示",
            code_blocks=[CodeBlock(content=content, description="完整示例代码")],
            order=0
        )
        return [section]
    
    def _extract_intro_conclusion(self, content: str) -> Tuple[str, str]:
        """提取介绍和结论部分"""
        introduction = ""
        conclusion = ""
        
        lines = content.split('\n')
        intro_patterns = ['介绍', 'introduction', '概述', 'overview']
        concl_patterns = ['结论', 'conclusion', '总结', 'summary']
        
        current_section = None
        current_content = []
        
        for line in lines:
            stripped = line.strip().lower()
            
            if any(pattern in stripped for pattern in intro_patterns):
                if current_content and current_section == 'intro':
                    introduction = '\n'.join(current_content)
                current_section = 'intro'
                current_content = []
            elif any(pattern in stripped for pattern in concl_patterns):
                if current_content and current_section == 'intro':
                    introduction = '\n'.join(current_content)
                current_section = 'conclusion'
                current_content = []
            elif current_section and line.startswith('#'):
                current_content.append(line)
        
        if current_content:
            if current_section == 'intro':
                introduction = '\n'.join(current_content)
            elif current_section == 'conclusion':
                conclusion = '\n'.join(current_content)
        
        return introduction, conclusion


class TutorialGenerator:
    """教程文档生成器主类"""
    
    def __init__(self, config: TutorialConfig):
        """
        初始化教程生成器
        
        Args:
            config: 教程生成配置
        """
        self.config = config
        self.parser = ExampleParser()
        self.template_engine = TemplateEngine()
        self.template_manager = TemplateManager(self.template_engine)
    
    def generate_tutorial_documentation(self, examples_dir: Path, 
                                      output_dir: Path) -> bool:
        """
        生成教程文档
        
        Args:
            examples_dir: 示例代码目录
            output_dir: 输出目录
            
        Returns:
            生成是否成功
        """
        try:
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 查找所有示例文件
            example_files = list(examples_dir.glob("**/*.py"))
            if not example_files:
                logger.warning(f"No Python files found in {examples_dir}")
                return False
            
            # 解析示例文件
            tutorials = []
            for file_path in example_files:
                tutorial = self.parser.parse_example_file(file_path)
                if tutorial:
                    tutorials.append(tutorial)
            
            if not tutorials:
                logger.error("No valid tutorials found")
                return False
            
            # 按类型和难度分组
            grouped_tutorials = self._group_tutorials(tutorials)
            
            # 生成各种教程文档
            success_count = 0
            for tutorial in tutorials:
                if self._generate_single_tutorial(tutorial, output_dir):
                    success_count += 1
            
            # 生成索引文档
            self._generate_tutorial_index(grouped_tutorials, output_dir)
            
            logger.info(f"Generated {success_count}/{len(tutorials)} tutorials successfully")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error generating tutorial documentation: {e}")
            return False
    
    def _group_tutorials(self, tutorials: List[TutorialDocument]) -> Dict[str, List[TutorialDocument]]:
        """
        按类型和难度分组教程
        
        Args:
            tutorials: 教程列表
            
        Returns:
            分组后的教程字典
        """
        grouped = {
            'by_type': {},
            'by_difficulty': {},
            'all': tutorials
        }
        
        for tutorial in tutorials:
            # 按类型分组
            type_key = tutorial.metadata.example_type.value
            if type_key not in grouped['by_type']:
                grouped['by_type'][type_key] = []
            grouped['by_type'][type_key].append(tutorial)
            
            # 按难度分组
            difficulty = tutorial.metadata.difficulty
            if difficulty not in grouped['by_difficulty']:
                grouped['by_difficulty'][difficulty] = []
            grouped['by_difficulty'][difficulty].append(tutorial)
        
        return grouped
    
    def _generate_single_tutorial(self, tutorial: TutorialDocument, 
                                output_dir: Path) -> bool:
        """
        生成单个教程文档
        
        Args:
            tutorial: 教程文档对象
            output_dir: 输出目录
            
        Returns:
            生成是否成功
        """
        try:
            # 构建模板上下文
            context = {
                'tutorial': tutorial.to_dict(),
                'config': asdict(self.config),
                'generator_info': {
                    'name': 'Tutorial Documentation Generator',
                    'version': '1.0.0'
                }
            }
            
            # 渲染教程文档
            content = self.template_manager.render_by_type('tutorial', 'main', context)
            
            # 生成文件名
            safe_title = re.sub(r'[^\w\s-]', '', tutorial.metadata.title)
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"{safe_title.lower()}.md"
            
            # 写入文件
            output_file = output_dir / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated tutorial: {tutorial.metadata.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating tutorial {tutorial.metadata.title}: {e}")
            return False
    
    def _generate_tutorial_index(self, grouped_tutorials: Dict[str, Any], 
                               output_dir: Path):
        """
        生成教程索引文档
        
        Args:
            grouped_tutorials: 分组的教程数据
            output_dir: 输出目录
        """
        try:
            # 构建索引数据
            index_data = {
                'total_tutorials': len(grouped_tutorials['all']),
                'by_type': {},
                'by_difficulty': {},
                'featured': []
            }
            
            # 按类型统计
            for type_name, tutorials in grouped_tutorials['by_type'].items():
                index_data['by_type'][type_name] = {
                    'count': len(tutorials),
                    'tutorials': [
                        {
                            'title': t.metadata.title,
                            'description': t.metadata.description,
                            'difficulty': t.metadata.difficulty,
                            'estimated_time': t.metadata.estimated_time,
                            'tags': t.metadata.tags
                        } for t in tutorials
                    ]
                }
            
            # 按难度统计
            for difficulty, tutorials in grouped_tutorials['by_difficulty'].items():
                index_data['by_difficulty'][difficulty] = len(tutorials)
            
            # 推荐教程（选择一些优质教程）
            featured_tutorials = [t for t in grouped_tutorials['all'] 
                                if t.metadata.example_type == ExampleType.TUTORIAL][:5]
            index_data['featured'] = [
                {
                    'title': t.metadata.title,
                    'description': t.metadata.description,
                    'difficulty': t.metadata.difficulty
                } for t in featured_tutorials
            ]
            
            # 渲染索引文档
            context = {
                'index': index_data,
                'config': asdict(self.config),
                'generator_info': {
                    'name': 'Tutorial Documentation Generator',
                    'version': '1.0.0'
                }
            }
            
            content = self.template_manager.render_by_type('documentation', 'index', context)
            
            # 写入索引文件
            index_file = output_dir / "tutorials-index.md"
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("Generated tutorial index")
            
        except Exception as e:
            logger.error(f"Error generating tutorial index: {e}")


# 便捷函数
def generate_tutorials(examples_dir: Path, output_dir: Path, 
                      config: Optional[TutorialConfig] = None) -> bool:
    """
    生成教程文档的便捷函数
    
    Args:
        examples_dir: 示例代码目录
        output_dir: 输出目录
        config: 教程生成配置
        
    Returns:
        生成是否成功
    """
    if config is None:
        config = TutorialConfig()
    
    generator = TutorialGenerator(config)
    return generator.generate_tutorial_documentation(examples_dir, output_dir)


def create_tutorial_config(**kwargs) -> TutorialConfig:
    """
    创建教程配置的便捷函数
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        配置对象
    """
    return TutorialConfig(**kwargs)


if __name__ == "__main__":
    # 测试教程生成器
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate tutorial documentation')
    parser.add_argument('examples', type=Path, help='Examples directory')
    parser.add_argument('output', type=Path, help='Output directory')
    parser.add_argument('--include-output', action='store_true', help='Include code output')
    parser.add_argument('--no-auto-generate', action='store_true', help='Disable auto generation')
    
    args = parser.parse_args()
    
    config = TutorialConfig(
        include_output=args.include_output,
        auto_generate=not args.no_auto_generate
    )
    
    success = generate_tutorials(args.examples, args.output, config)
    sys.exit(0 if success else 1)