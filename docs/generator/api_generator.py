"""
API文档生成器
功能：自动从Python代码生成API参考文档，支持类、函数、方法的详细文档化
作者：自动文档生成系统
"""

import os
import sys
import importlib
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

# 导入本地模块
from .code_parser import CodeParser, CodeElement, NodeType, FunctionElement, ClassElement
from .template_engine import TemplateEngine, TemplateManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIDocumentationConfig:
    """API文档生成配置"""
    include_private: bool = False  # 是否包含私有成员
    include_source: bool = True   # 是否包含源码链接
    include_inheritance: bool = True  # 是否包含继承关系
    include_examples: bool = True  # 是否包含使用示例
    group_by_module: bool = True  # 是否按模块分组
    sort_members: bool = True     # 是否排序成员
    output_format: str = "markdown"  # 输出格式
    language: str = "zh"         # 文档语言


@dataclass 
class ModuleDocumentation:
    """模块文档数据结构"""
    name: str
    path: str
    docstring: Optional[str]
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    imports: List[str]
    file_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class BaseDocumentationGenerator(ABC):
    """文档生成器抽象基类"""
    
    def __init__(self, config: APIDocumentationConfig):
        """
        初始化文档生成器
        
        Args:
            config: 文档生成配置
        """
        self.config = config
        self.parser = CodeParser()
        self.template_engine = TemplateEngine()
        self.template_manager = TemplateManager(self.template_engine)
    
    @abstractmethod
    def generate_documentation(self, source_path: Path, 
                             output_path: Path) -> bool:
        """
        生成文档的抽象方法
        
        Args:
            source_path: 源代码路径
            output_path: 输出路径
            
        Returns:
            生成是否成功
        """
        pass
    
    def _should_include_member(self, name: str) -> bool:
        """
        判断是否应该包含某个成员
        
        Args:
            name: 成员名称
            
        Returns:
            是否包含
        """
        if name.startswith('__') and name.endswith('__'):
            # 魔术方法，只包含常用的
            common_magic = {'__init__', '__str__', '__repr__', '__call__', 
                          '__enter__', '__exit__', '__len__', '__iter__'}
            return name in common_magic
        
        if name.startswith('_') and not self.config.include_private:
            return False
        
        return True
    
    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """
        从文档字符串中提取示例代码
        
        Args:
            docstring: 文档字符串
            
        Returns:
            示例代码列表
        """
        if not docstring:
            return []
        
        examples = []
        lines = docstring.split('\n')
        in_example = False
        current_example = []
        
        for line in lines:
            stripped = line.strip()
            
            # 检测示例开始
            if any(keyword in stripped.lower() for keyword in 
                  ['example:', 'examples:', '示例:', '例子:']):
                in_example = True
                if current_example:
                    examples.append('\n'.join(current_example))
                    current_example = []
                continue
            
            # 检测代码块
            if stripped.startswith('>>>') or stripped.startswith('```'):
                in_example = True
                current_example.append(line)
                continue
            
            # 如果在示例中，收集代码行
            if in_example:
                if stripped == '' or line.startswith('    '):
                    current_example.append(line)
                else:
                    # 示例结束
                    if current_example:
                        examples.append('\n'.join(current_example))
                        current_example = []
                    in_example = False
        
        # 添加最后一个示例
        if current_example:
            examples.append('\n'.join(current_example))
        
        return [ex.strip() for ex in examples if ex.strip()]


class APIDocumentationGenerator(BaseDocumentationGenerator):
    """API文档生成器主类"""
    
    def __init__(self, config: APIDocumentationConfig):
        """
        初始化API文档生成器
        
        Args:
            config: 文档生成配置
        """
        super().__init__(config)
        self.processed_modules: Set[str] = set()
    
    def generate_documentation(self, source_path: Path, 
                             output_path: Path) -> bool:
        """
        生成API文档
        
        Args:
            source_path: 源代码路径（文件或目录）
            output_path: 输出目录
            
        Returns:
            生成是否成功
        """
        try:
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            if source_path.is_file():
                # 处理单个文件
                return self._process_file(source_path, output_path)
            elif source_path.is_dir():
                # 处理整个目录
                return self._process_directory(source_path, output_path)
            else:
                logger.error(f"Source path does not exist: {source_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return False
    
    def _process_directory(self, source_dir: Path, output_dir: Path) -> bool:
        """
        处理源代码目录
        
        Args:
            source_dir: 源代码目录
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        success_count = 0
        total_count = 0
        
        # 解析目录中的所有Python文件
        modules = self.parser.parse_directory(source_dir, recursive=True)
        
        for module in modules:
            total_count += 1
            if self._generate_module_documentation(module, output_dir):
                success_count += 1
        
        # 生成总索引文件
        self._generate_index_documentation(modules, output_dir)
        
        logger.info(f"Processed {success_count}/{total_count} modules successfully")
        return success_count > 0
    
    def _process_file(self, source_file: Path, output_dir: Path) -> bool:
        """
        处理单个源文件
        
        Args:
            source_file: 源文件路径
            output_dir: 输出目录
            
        Returns:
            处理是否成功
        """
        module = self.parser.parse_file(source_file)
        if module:
            return self._generate_module_documentation(module, output_dir)
        return False
    
    def _generate_module_documentation(self, module: CodeElement, 
                                     output_dir: Path) -> bool:
        """
        为单个模块生成文档
        
        Args:
            module: 模块代码元素
            output_dir: 输出目录
            
        Returns:
            生成是否成功
        """
        try:
            # 避免重复处理
            if module.name in self.processed_modules:
                return True
            
            self.processed_modules.add(module.name)
            
            # 构建模块文档数据
            module_doc = self._build_module_documentation(module)
            
            # 渲染模块文档
            context = {
                'module': module_doc.to_dict(),
                'config': asdict(self.config),
                'generator_info': {
                    'name': 'API Documentation Generator',
                    'version': '1.0.0'
                }
            }
            
            # 生成主模块文档
            content = self.template_manager.render_by_type('api', 'module', context)
            
            # 写入文档文件
            output_file = output_dir / f"{module.name}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated documentation for module: {module.name}")
            
            # 如果配置要求，为每个类生成单独文档
            if self.config.group_by_module:
                self._generate_class_documentations(module_doc, output_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating module documentation: {e}")
            return False
    
    def _build_module_documentation(self, module: CodeElement) -> ModuleDocumentation:
        """
        构建模块文档数据
        
        Args:
            module: 模块代码元素
            
        Returns:
            模块文档数据
        """
        classes = []
        functions = []
        variables = []
        
        # 处理模块中的各种元素
        for child in module.children:
            if not self._should_include_member(child.name):
                continue
            
            if child.node_type == NodeType.CLASS:
                classes.append(self._build_class_info(child))
            elif child.node_type == NodeType.FUNCTION:
                functions.append(self._build_function_info(child))
            elif child.node_type == NodeType.VARIABLE:
                variables.append(self._build_variable_info(child))
        
        # 排序成员（如果配置要求）
        if self.config.sort_members:
            classes.sort(key=lambda x: x['name'])
            functions.sort(key=lambda x: x['name'])
            variables.sort(key=lambda x: x['name'])
        
        return ModuleDocumentation(
            name=module.name,
            path=str(module.source_file),
            docstring=module.docstring.content if module.docstring else None,
            classes=classes,
            functions=functions,
            variables=variables,
            imports=[],  # TODO: 从AST提取导入信息
            file_info={
                'line_count': sum(1 for _ in open(module.source_file)) if module.source_file.exists() else 0,
                'size_bytes': module.source_file.stat().st_size if module.source_file.exists() else 0
            }
        )
    
    def _build_class_info(self, class_element: ClassElement) -> Dict[str, Any]:
        """
        构建类信息
        
        Args:
            class_element: 类代码元素
            
        Returns:
            类信息字典
        """
        methods = []
        properties = []
        
        # 处理方法和属性
        for method in class_element.methods:
            if self._should_include_member(method.name):
                methods.append(self._build_function_info(method))
        
        for prop in class_element.properties:
            if self._should_include_member(prop.name):
                properties.append(self._build_function_info(prop))
        
        # 排序成员
        if self.config.sort_members:
            methods.sort(key=lambda x: (x['name'] != '__init__', x['name']))
            properties.sort(key=lambda x: x['name'])
        
        class_info = {
            'name': class_element.name,
            'full_name': class_element.get_full_name(),
            'docstring': class_element.docstring.content if class_element.docstring else None,
            'line_number': class_element.line_number,
            'base_classes': class_element.base_classes,
            'methods': methods,
            'properties': properties,
            'attributes': [self._build_variable_info(attr) for attr in class_element.attributes
                          if self._should_include_member(attr.name)]
        }
        
        # 添加示例（如果配置要求）
        if self.config.include_examples and class_element.docstring:
            class_info['examples'] = self._extract_examples_from_docstring(
                class_element.docstring.content
            )
        
        return class_info
    
    def _build_function_info(self, func_element: FunctionElement) -> Dict[str, Any]:
        """
        构建函数/方法信息
        
        Args:
            func_element: 函数代码元素
            
        Returns:
            函数信息字典
        """
        func_info = {
            'name': func_element.name,
            'full_name': func_element.get_full_name(),
            'docstring': func_element.docstring.content if func_element.docstring else None,
            'line_number': func_element.line_number,
            'parameters': [param.to_dict() for param in func_element.parameters],
            'return_type': func_element.return_annotation,
            'is_async': func_element.is_async,
            'is_property': func_element.is_property,
            'is_staticmethod': func_element.is_staticmethod,
            'is_classmethod': func_element.is_classmethod,
            'node_type': func_element.node_type.value
        }
        
        # 添加示例（如果配置要求）
        if self.config.include_examples and func_element.docstring:
            func_info['examples'] = self._extract_examples_from_docstring(
                func_element.docstring.content
            )
        
        return func_info
    
    def _build_variable_info(self, var_element) -> Dict[str, Any]:
        """
        构建变量信息
        
        Args:
            var_element: 变量代码元素
            
        Returns:
            变量信息字典
        """
        return {
            'name': var_element.name,
            'full_name': var_element.get_full_name(),
            'line_number': var_element.line_number,
            'type_annotation': var_element.type_annotation,
            'default_value': var_element.default_value
        }
    
    def _generate_class_documentations(self, module_doc: ModuleDocumentation, 
                                     output_dir: Path):
        """
        为模块中的每个类生成单独的文档文件
        
        Args:
            module_doc: 模块文档数据
            output_dir: 输出目录
        """
        for class_info in module_doc.classes:
            try:
                context = {
                    'class': class_info,
                    'module': module_doc.to_dict(),
                    'config': asdict(self.config)
                }
                
                content = self.template_manager.render_by_type('api', 'class', context)
                
                # 写入类文档文件
                class_file = output_dir / f"{module_doc.name}.{class_info['name']}.md"
                with open(class_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.debug(f"Generated class documentation: {class_info['name']}")
                
            except Exception as e:
                logger.error(f"Error generating class documentation for {class_info['name']}: {e}")
    
    def _generate_index_documentation(self, modules: List[CodeElement], 
                                    output_dir: Path):
        """
        生成API文档索引
        
        Args:
            modules: 模块列表
            output_dir: 输出目录
        """
        try:
            # 构建索引数据
            index_data = {
                'modules': [],
                'total_classes': 0,
                'total_functions': 0,
                'total_modules': len(modules)
            }
            
            for module in modules:
                module_info = {
                    'name': module.name,
                    'path': str(module.source_file),
                    'docstring': module.docstring.content if module.docstring else None,
                    'class_count': len([c for c in module.children if c.node_type == NodeType.CLASS]),
                    'function_count': len([c for c in module.children if c.node_type == NodeType.FUNCTION])
                }
                
                index_data['modules'].append(module_info)
                index_data['total_classes'] += module_info['class_count']
                index_data['total_functions'] += module_info['function_count']
            
            # 排序模块
            if self.config.sort_members:
                index_data['modules'].sort(key=lambda x: x['name'])
            
            # 渲染索引文档
            context = {
                'index': index_data,
                'config': asdict(self.config),
                'generator_info': {
                    'name': 'API Documentation Generator',
                    'version': '1.0.0'
                }
            }
            
            content = self.template_manager.render_by_type('documentation', 'index', context)
            
            # 写入索引文件
            index_file = output_dir / "index.md"
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("Generated API documentation index")
            
        except Exception as e:
            logger.error(f"Error generating index documentation: {e}")


# 便捷函数
def generate_api_docs(source_path: Path, output_path: Path, 
                     config: Optional[APIDocumentationConfig] = None) -> bool:
    """
    生成API文档的便捷函数
    
    Args:
        source_path: 源代码路径
        output_path: 输出路径
        config: 文档生成配置，如果为None则使用默认配置
        
    Returns:
        生成是否成功
    """
    if config is None:
        config = APIDocumentationConfig()
    
    generator = APIDocumentationGenerator(config)
    return generator.generate_documentation(source_path, output_path)


def create_api_config(**kwargs) -> APIDocumentationConfig:
    """
    创建API文档配置的便捷函数
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        配置对象
    """
    return APIDocumentationConfig(**kwargs)


if __name__ == "__main__":
    # 测试API文档生成器
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate API documentation')
    parser.add_argument('source', type=Path, help='Source code path')
    parser.add_argument('output', type=Path, help='Output directory')
    parser.add_argument('--include-private', action='store_true', help='Include private members')
    parser.add_argument('--exclude-source', action='store_true', help='Exclude source links')
    
    args = parser.parse_args()
    
    config = APIDocumentationConfig(
        include_private=args.include_private,
        include_source=not args.exclude_source
    )
    
    success = generate_api_docs(args.source, args.output, config)
    sys.exit(0 if success else 1)