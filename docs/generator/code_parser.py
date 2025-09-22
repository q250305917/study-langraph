"""
代码解析器模块
功能：使用AST解析Python代码，提取文档字符串、类型注解和代码结构
作者：自动文档生成系统
"""

import ast
import inspect
import importlib
import types
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """代码节点类型枚举"""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    PROPERTY = "property"


@dataclass
class DocString:
    """文档字符串数据结构"""
    content: str
    line_number: int
    source_file: Path
    
    def __post_init__(self):
        """格式化文档字符串内容"""
        if self.content:
            # 清理docstring格式，移除多余的空白行
            lines = self.content.strip().split('\n')
            # 移除首尾空行
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            self.content = '\n'.join(lines)


@dataclass
class Parameter:
    """函数参数信息"""
    name: str
    type_annotation: Optional[str]
    default_value: Optional[str]
    is_keyword_only: bool = False
    is_positional_only: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'type': self.type_annotation,
            'default': self.default_value,
            'keyword_only': self.is_keyword_only,
            'positional_only': self.is_positional_only
        }


@dataclass
class CodeElement:
    """代码元素基类"""
    name: str
    node_type: NodeType
    docstring: Optional[DocString]
    line_number: int
    source_file: Path
    parent: Optional['CodeElement'] = None
    children: List['CodeElement'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child: 'CodeElement'):
        """添加子元素"""
        child.parent = self
        self.children.append(child)
    
    def get_full_name(self) -> str:
        """获取完整名称（包含父级路径）"""
        if self.parent and self.parent.node_type != NodeType.MODULE:
            return f"{self.parent.get_full_name()}.{self.name}"
        return self.name


@dataclass
class FunctionElement(CodeElement):
    """函数/方法元素"""
    parameters: List[Parameter]
    return_annotation: Optional[str]
    is_async: bool = False
    is_property: bool = False
    is_staticmethod: bool = False
    is_classmethod: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.parameters is None:
            self.parameters = []


@dataclass
class ClassElement(CodeElement):
    """类元素"""
    base_classes: List[str]
    methods: List[FunctionElement]
    properties: List[FunctionElement]
    attributes: List['VariableElement']
    
    def __post_init__(self):
        super().__post_init__()
        if self.base_classes is None:
            self.base_classes = []
        if self.methods is None:
            self.methods = []
        if self.properties is None:
            self.properties = []
        if self.attributes is None:
            self.attributes = []


@dataclass
class VariableElement(CodeElement):
    """变量/属性元素"""
    type_annotation: Optional[str]
    default_value: Optional[str]


class CodeParser:
    """代码解析器 - 使用AST解析Python代码"""
    
    def __init__(self):
        """初始化解析器"""
        self.current_file: Optional[Path] = None
        self.current_module: Optional[CodeElement] = None
        
    def parse_file(self, file_path: Path) -> Optional[CodeElement]:
        """
        解析单个Python文件
        
        Args:
            file_path: Python文件路径
            
        Returns:
            解析后的模块元素，如果解析失败返回None
        """
        try:
            self.current_file = file_path
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 解析AST
            tree = ast.parse(source_code, filename=str(file_path))
            
            # 创建模块元素
            module_element = CodeElement(
                name=file_path.stem,
                node_type=NodeType.MODULE,
                docstring=self._extract_docstring(tree),
                line_number=1,
                source_file=file_path
            )
            
            self.current_module = module_element
            
            # 遍历AST节点
            for node in tree.body:
                element = self._parse_node(node)
                if element:
                    module_element.add_child(element)
            
            logger.info(f"Successfully parsed file: {file_path}")
            return module_element
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
    
    def parse_directory(self, directory_path: Path, 
                       recursive: bool = True) -> List[CodeElement]:
        """
        解析目录中的所有Python文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归解析子目录
            
        Returns:
            解析后的模块元素列表
        """
        modules = []
        
        # 定义Python文件模式
        pattern = "**/*.py" if recursive else "*.py"
        
        for file_path in directory_path.glob(pattern):
            # 跳过__pycache__和虚拟环境目录
            if "__pycache__" in str(file_path) or "venv" in str(file_path):
                continue
                
            module = self.parse_file(file_path)
            if module:
                modules.append(module)
        
        logger.info(f"Parsed {len(modules)} modules from {directory_path}")
        return modules
    
    def _parse_node(self, node: ast.AST) -> Optional[CodeElement]:
        """
        解析AST节点
        
        Args:
            node: AST节点
            
        Returns:
            解析后的代码元素
        """
        if isinstance(node, ast.FunctionDef):
            return self._parse_function(node)
        elif isinstance(node, ast.AsyncFunctionDef):
            return self._parse_function(node, is_async=True)
        elif isinstance(node, ast.ClassDef):
            return self._parse_class(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            return self._parse_variable(node)
        
        return None
    
    def _parse_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], 
                       is_async: bool = False) -> FunctionElement:
        """
        解析函数定义
        
        Args:
            node: 函数AST节点
            is_async: 是否异步函数
            
        Returns:
            函数元素
        """
        # 提取参数信息
        parameters = self._extract_parameters(node.args)
        
        # 提取返回类型注解
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns)
        
        # 检查装饰器
        is_property = any(self._get_decorator_name(d) == 'property' 
                         for d in node.decorator_list)
        is_staticmethod = any(self._get_decorator_name(d) == 'staticmethod' 
                             for d in node.decorator_list)
        is_classmethod = any(self._get_decorator_name(d) == 'classmethod' 
                            for d in node.decorator_list)
        
        # 确定节点类型
        node_type = NodeType.PROPERTY if is_property else NodeType.FUNCTION
        
        return FunctionElement(
            name=node.name,
            node_type=node_type,
            docstring=self._extract_docstring(node),
            line_number=node.lineno,
            source_file=self.current_file,
            parameters=parameters,
            return_annotation=return_annotation,
            is_async=is_async,
            is_property=is_property,
            is_staticmethod=is_staticmethod,
            is_classmethod=is_classmethod
        )
    
    def _parse_class(self, node: ast.ClassDef) -> ClassElement:
        """
        解析类定义
        
        Args:
            node: 类AST节点
            
        Returns:
            类元素
        """
        # 提取基类
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(ast.unparse(base))
        
        class_element = ClassElement(
            name=node.name,
            node_type=NodeType.CLASS,
            docstring=self._extract_docstring(node),
            line_number=node.lineno,
            source_file=self.current_file,
            base_classes=base_classes,
            methods=[],
            properties=[],
            attributes=[]
        )
        
        # 解析类成员
        for child_node in node.body:
            if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_element = self._parse_function(child_node, 
                                                  isinstance(child_node, ast.AsyncFunctionDef))
                if func_element.is_property:
                    class_element.properties.append(func_element)
                else:
                    func_element.node_type = NodeType.METHOD
                    class_element.methods.append(func_element)
                class_element.add_child(func_element)
            elif isinstance(child_node, (ast.Assign, ast.AnnAssign)):
                var_element = self._parse_variable(child_node)
                if var_element:
                    class_element.attributes.append(var_element)
                    class_element.add_child(var_element)
        
        return class_element
    
    def _parse_variable(self, node: Union[ast.Assign, ast.AnnAssign]) -> Optional[VariableElement]:
        """
        解析变量/属性定义
        
        Args:
            node: 变量AST节点
            
        Returns:
            变量元素或None
        """
        name = None
        type_annotation = None
        default_value = None
        
        if isinstance(node, ast.AnnAssign):
            # 类型注解的赋值 (x: int = 5)
            if isinstance(node.target, ast.Name):
                name = node.target.id
                type_annotation = ast.unparse(node.annotation)
                if node.value:
                    default_value = ast.unparse(node.value)
        elif isinstance(node, ast.Assign):
            # 普通赋值 (x = 5)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                default_value = ast.unparse(node.value)
        
        if name and not name.startswith('_'):  # 跳过私有变量
            return VariableElement(
                name=name,
                node_type=NodeType.VARIABLE,
                docstring=None,  # 变量通常没有docstring
                line_number=node.lineno,
                source_file=self.current_file,
                type_annotation=type_annotation,
                default_value=default_value
            )
        
        return None
    
    def _extract_parameters(self, args: ast.arguments) -> List[Parameter]:
        """
        提取函数参数信息
        
        Args:
            args: 函数参数AST节点
            
        Returns:
            参数列表
        """
        parameters = []
        
        # 普通参数
        for i, arg in enumerate(args.args):
            param = Parameter(
                name=arg.arg,
                type_annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                default_value=None
            )
            
            # 检查是否有默认值
            defaults_offset = len(args.args) - len(args.defaults)
            if i >= defaults_offset:
                default_index = i - defaults_offset
                param.default_value = ast.unparse(args.defaults[default_index])
            
            parameters.append(param)
        
        # 关键字专用参数
        for i, arg in enumerate(args.kwonlyargs):
            default_value = None
            if i < len(args.kw_defaults) and args.kw_defaults[i]:
                default_value = ast.unparse(args.kw_defaults[i])
            
            param = Parameter(
                name=arg.arg,
                type_annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                default_value=default_value,
                is_keyword_only=True
            )
            parameters.append(param)
        
        # *args参数
        if args.vararg:
            param = Parameter(
                name=f"*{args.vararg.arg}",
                type_annotation=ast.unparse(args.vararg.annotation) if args.vararg.annotation else None,
                default_value=None
            )
            parameters.append(param)
        
        # **kwargs参数
        if args.kwarg:
            param = Parameter(
                name=f"**{args.kwarg.arg}",
                type_annotation=ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else None,
                default_value=None
            )
            parameters.append(param)
        
        return parameters
    
    def _extract_docstring(self, node: ast.AST) -> Optional[DocString]:
        """
        提取节点的文档字符串
        
        Args:
            node: AST节点
            
        Returns:
            文档字符串对象或None
        """
        if (hasattr(node, 'body') and node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            
            docstring_content = node.body[0].value.value
            return DocString(
                content=docstring_content,
                line_number=node.body[0].lineno,
                source_file=self.current_file
            )
        
        return None
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """
        获取装饰器名称
        
        Args:
            decorator: 装饰器AST节点
            
        Returns:
            装饰器名称
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        else:
            return ast.unparse(decorator)


# 工具函数
def get_module_info(module_path: Path) -> Dict[str, Any]:
    """
    获取模块的基本信息
    
    Args:
        module_path: 模块文件路径
        
    Returns:
        模块信息字典
    """
    parser = CodeParser()
    module_element = parser.parse_file(module_path)
    
    if not module_element:
        return {}
    
    info = {
        'name': module_element.name,
        'file': str(module_element.source_file),
        'docstring': module_element.docstring.content if module_element.docstring else None,
        'classes': len([c for c in module_element.children if c.node_type == NodeType.CLASS]),
        'functions': len([c for c in module_element.children if c.node_type == NodeType.FUNCTION]),
        'variables': len([c for c in module_element.children if c.node_type == NodeType.VARIABLE])
    }
    
    return info


if __name__ == "__main__":
    # 测试代码解析器
    import sys
    
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if file_path.exists():
            parser = CodeParser()
            module = parser.parse_file(file_path)
            if module:
                print(f"Parsed module: {module.name}")
                print(f"Classes: {len([c for c in module.children if c.node_type == NodeType.CLASS])}")
                print(f"Functions: {len([c for c in module.children if c.node_type == NodeType.FUNCTION])}")
            else:
                print("Failed to parse file")
        else:
            print("File not found")
    else:
        print("Usage: python code_parser.py <file_path>")