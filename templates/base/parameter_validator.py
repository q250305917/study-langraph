"""
参数验证器模块

本模块提供了强大而灵活的参数验证功能，用于确保模板参数的有效性和一致性。
支持多种验证规则、自定义验证器和复杂的数据结构验证。

核心特性：
1. 类型验证：支持基础类型和复杂类型的验证
2. 约束验证：数值范围、字符串长度、集合大小等约束条件
3. 模式验证：正则表达式、预定义模式等
4. 嵌套验证：支持嵌套字典和列表的递归验证
5. 自定义验证：支持自定义验证函数和复杂业务逻辑
6. 错误聚合：收集所有验证错误，提供详细的错误报告
7. 类型转换：自动类型转换和值标准化

设计原理：
- 组合模式：通过组合多个验证规则实现复杂验证
- 责任链模式：按优先级执行验证规则
- 策略模式：支持不同的验证策略
- 工厂模式：动态创建验证器实例
"""

import re
import json
import inspect
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Type, TypeVar, Callable, 
    Set, Tuple, Pattern, get_origin, get_args
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
from pathlib import Path

from ...src.langchain_learning.core.logger import get_logger
from ...src.langchain_learning.core.exceptions import ValidationError, ErrorCodes

logger = get_logger(__name__)

T = TypeVar('T')


class ValidationLevel(Enum):
    """验证级别枚举"""
    STRICT = "strict"      # 严格验证，任何错误都抛出异常
    WARNING = "warning"    # 警告模式，记录警告但不抛出异常
    LENIENT = "lenient"    # 宽松模式，尽可能转换和修复数据


class ValidationType(Enum):
    """验证类型枚举"""
    TYPE = "type"                    # 类型验证
    RANGE = "range"                  # 数值范围验证
    LENGTH = "length"                # 长度验证
    PATTERN = "pattern"              # 模式验证
    ALLOWED_VALUES = "allowed_values"  # 允许值验证
    CUSTOM = "custom"                # 自定义验证
    REQUIRED = "required"            # 必需字段验证
    FORMAT = "format"                # 格式验证


@dataclass
class ValidationResult:
    """
    验证结果数据类
    
    包含验证的详细结果信息，包括成功状态、错误信息、警告信息等。
    """
    success: bool                                        # 验证是否成功
    errors: List[str] = field(default_factory=list)     # 错误信息列表
    warnings: List[str] = field(default_factory=list)   # 警告信息列表
    fixed_data: Optional[Dict[str, Any]] = None         # 修复后的数据
    validation_time: float = 0.0                        # 验证耗时
    
    def add_error(self, message: str) -> None:
        """添加错误信息"""
        self.errors.append(message)
        self.success = False
    
    def add_warning(self, message: str) -> None:
        """添加警告信息"""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """合并另一个验证结果"""
        if not other.success:
            self.success = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.validation_time += other.validation_time
    
    def __str__(self) -> str:
        if self.success:
            status = "SUCCESS"
            if self.warnings:
                status += f" (with {len(self.warnings)} warnings)"
        else:
            status = f"FAILED ({len(self.errors)} errors"
            if self.warnings:
                status += f", {len(self.warnings)} warnings"
            status += ")"
        
        return f"ValidationResult({status})"


@dataclass  
class ValidationRule:
    """
    验证规则数据类
    
    定义单个验证规则的配置和行为。
    """
    rule_type: ValidationType                      # 验证类型
    target_field: Optional[str] = None            # 目标字段名，None表示应用到整个数据
    parameters: Dict[str, Any] = field(default_factory=dict)  # 规则参数
    error_message: Optional[str] = None           # 自定义错误信息
    warning_message: Optional[str] = None         # 自定义警告信息
    enabled: bool = True                          # 是否启用此规则
    priority: int = 0                             # 优先级，数值越大优先级越高
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.error_message:
            self.error_message = self._generate_default_error_message()
    
    def _generate_default_error_message(self) -> str:
        """生成默认错误信息"""
        field_desc = f"Field '{self.target_field}'" if self.target_field else "Value"
        
        if self.rule_type == ValidationType.TYPE:
            expected_type = self.parameters.get('expected_type', 'unknown')
            return f"{field_desc} must be of type {expected_type}"
        elif self.rule_type == ValidationType.RANGE:
            min_val = self.parameters.get('min_value')
            max_val = self.parameters.get('max_value')
            if min_val is not None and max_val is not None:
                return f"{field_desc} must be between {min_val} and {max_val}"
            elif min_val is not None:
                return f"{field_desc} must be >= {min_val}"
            elif max_val is not None:
                return f"{field_desc} must be <= {max_val}"
        elif self.rule_type == ValidationType.LENGTH:
            min_len = self.parameters.get('min_length')
            max_len = self.parameters.get('max_length')
            if min_len is not None and max_len is not None:
                return f"{field_desc} length must be between {min_len} and {max_len}"
            elif min_len is not None:
                return f"{field_desc} length must be >= {min_len}"
            elif max_len is not None:
                return f"{field_desc} length must be <= {max_len}"
        elif self.rule_type == ValidationType.PATTERN:
            pattern = self.parameters.get('pattern')
            return f"{field_desc} must match pattern: {pattern}"
        elif self.rule_type == ValidationType.ALLOWED_VALUES:
            allowed = self.parameters.get('allowed_values', [])
            return f"{field_desc} must be one of: {allowed}"
        elif self.rule_type == ValidationType.REQUIRED:
            return f"{field_desc} is required"
        
        return f"{field_desc} validation failed"


class BaseValidator(ABC):
    """
    验证器抽象基类
    
    定义验证器的通用接口，所有具体的验证器都应该继承此类。
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        """
        初始化验证器
        
        Args:
            validation_level: 验证级别
        """
        self.validation_level = validation_level
        self.rules: List[ValidationRule] = []
        
    @abstractmethod
    def validate(self, data: Any, field_name: Optional[str] = None) -> ValidationResult:
        """
        执行验证
        
        Args:
            data: 要验证的数据
            field_name: 字段名称
            
        Returns:
            验证结果
        """
        pass
    
    def add_rule(self, rule: ValidationRule) -> None:
        """添加验证规则"""
        self.rules.append(rule)
        # 按优先级排序
        self.rules.sort(key=lambda r: r.priority, reverse=True)


class TypeValidator(BaseValidator):
    """
    类型验证器
    
    验证数据是否符合指定的类型要求，支持基础类型和复杂类型。
    """
    
    def __init__(self, expected_type: Type, validation_level: ValidationLevel = ValidationLevel.STRICT):
        """
        初始化类型验证器
        
        Args:
            expected_type: 期望的数据类型
            validation_level: 验证级别
        """
        super().__init__(validation_level)
        self.expected_type = expected_type
        
    def validate(self, data: Any, field_name: Optional[str] = None) -> ValidationResult:
        """验证数据类型"""
        start_time = datetime.now()
        result = ValidationResult()
        
        try:
            if self._is_type_match(data, self.expected_type):
                result.success = True
            else:
                # 尝试类型转换
                if self.validation_level in [ValidationLevel.WARNING, ValidationLevel.LENIENT]:
                    converted_data = self._try_convert(data, self.expected_type)
                    if converted_data is not None:
                        result.success = True
                        result.fixed_data = {field_name: converted_data} if field_name else converted_data
                        if self.validation_level == ValidationLevel.WARNING:
                            result.add_warning(
                                f"Type mismatch for {field_name or 'value'}: "
                                f"expected {self.expected_type.__name__}, got {type(data).__name__}. "
                                f"Auto-converted to {self.expected_type.__name__}."
                            )
                    else:
                        result.add_error(
                            f"Type mismatch for {field_name or 'value'}: "
                            f"expected {self.expected_type.__name__}, got {type(data).__name__}"
                        )
                else:
                    result.add_error(
                        f"Type mismatch for {field_name or 'value'}: "
                        f"expected {self.expected_type.__name__}, got {type(data).__name__}"
                    )
        
        except Exception as e:
            result.add_error(f"Type validation failed for {field_name or 'value'}: {str(e)}")
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _is_type_match(self, data: Any, expected_type: Type) -> bool:
        """检查类型是否匹配"""
        # 处理泛型类型
        origin = get_origin(expected_type)
        if origin is not None:
            # 泛型类型（如List[str], Dict[str, int]等）
            if origin == list:
                return isinstance(data, list)
            elif origin == dict:
                return isinstance(data, dict)
            elif origin == tuple:
                return isinstance(data, tuple)
            elif origin == set:
                return isinstance(data, set)
            elif origin == Union:
                # Union类型，检查是否匹配任一类型
                type_args = get_args(expected_type)
                return any(self._is_type_match(data, arg) for arg in type_args)
        
        # 基础类型检查
        return isinstance(data, expected_type)
    
    def _try_convert(self, data: Any, expected_type: Type) -> Any:
        """尝试类型转换"""
        try:
            # 处理None值
            if data is None:
                return None
            
            # 字符串转换
            if expected_type == str:
                return str(data)
            
            # 数值转换
            elif expected_type == int:
                if isinstance(data, str):
                    # 处理字符串数字
                    return int(float(data))  # 先转float再转int，处理"1.0"这样的情况
                return int(data)
            
            elif expected_type == float:
                return float(data)
            
            # 布尔转换
            elif expected_type == bool:
                if isinstance(data, str):
                    return data.lower() in ('true', '1', 'yes', 'on')
                return bool(data)
            
            # 列表转换
            elif expected_type == list:
                if isinstance(data, str):
                    # 尝试JSON解析
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, list):
                            return parsed
                    except:
                        pass
                    # 逗号分隔
                    return [item.strip() for item in data.split(',') if item.strip()]
                elif hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
                    return list(data)
            
            # 字典转换
            elif expected_type == dict:
                if isinstance(data, str):
                    try:
                        return json.loads(data)
                    except:
                        pass
                
            # 路径转换
            elif expected_type == Path:
                return Path(str(data))
            
            # 日期时间转换
            elif expected_type == datetime:
                if isinstance(data, str):
                    # 尝试常见的日期时间格式
                    formats = [
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d',
                        '%Y/%m/%d %H:%M:%S',
                        '%Y/%m/%d',
                        '%d/%m/%Y %H:%M:%S',
                        '%d/%m/%Y'
                    ]
                    for fmt in formats:
                        try:
                            return datetime.strptime(data, fmt)
                        except:
                            continue
            
            # 使用类型的构造函数尝试转换
            return expected_type(data)
            
        except Exception:
            return None


class RangeValidator(BaseValidator):
    """
    范围验证器
    
    验证数值是否在指定范围内。
    """
    
    def __init__(
        self, 
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        初始化范围验证器
        
        Args:
            min_value: 最小值
            max_value: 最大值
            validation_level: 验证级别
        """
        super().__init__(validation_level)
        self.min_value = min_value
        self.max_value = max_value
        
    def validate(self, data: Any, field_name: Optional[str] = None) -> ValidationResult:
        """验证数值范围"""
        start_time = datetime.now()
        result = ValidationResult()
        
        try:
            # 确保数据是数值类型
            if not isinstance(data, (int, float)):
                try:
                    data = float(data)
                except (ValueError, TypeError):
                    result.add_error(f"{field_name or 'Value'} must be numeric for range validation")
                    return result
            
            # 检查最小值
            if self.min_value is not None and data < self.min_value:
                if self.validation_level == ValidationLevel.LENIENT:
                    result.fixed_data = {field_name: self.min_value} if field_name else self.min_value
                    result.add_warning(
                        f"{field_name or 'Value'} ({data}) is below minimum ({self.min_value}), "
                        f"adjusted to {self.min_value}"
                    )
                else:
                    result.add_error(
                        f"{field_name or 'Value'} ({data}) must be >= {self.min_value}"
                    )
            
            # 检查最大值
            if self.max_value is not None and data > self.max_value:
                if self.validation_level == ValidationLevel.LENIENT:
                    result.fixed_data = {field_name: self.max_value} if field_name else self.max_value
                    result.add_warning(
                        f"{field_name or 'Value'} ({data}) is above maximum ({self.max_value}), "
                        f"adjusted to {self.max_value}"
                    )
                else:
                    result.add_error(
                        f"{field_name or 'Value'} ({data}) must be <= {self.max_value}"
                    )
            
            if not result.errors:
                result.success = True
                
        except Exception as e:
            result.add_error(f"Range validation failed for {field_name or 'value'}: {str(e)}")
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result


class LengthValidator(BaseValidator):
    """
    长度验证器
    
    验证字符串、列表等对象的长度是否在指定范围内。
    """
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        初始化长度验证器
        
        Args:
            min_length: 最小长度
            max_length: 最大长度
            validation_level: 验证级别
        """
        super().__init__(validation_level)
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, data: Any, field_name: Optional[str] = None) -> ValidationResult:
        """验证长度"""
        start_time = datetime.now()
        result = ValidationResult()
        
        try:
            # 获取长度
            if hasattr(data, '__len__'):
                length = len(data)
            else:
                result.add_error(f"{field_name or 'Value'} does not have length attribute")
                return result
            
            # 检查最小长度
            if self.min_length is not None and length < self.min_length:
                result.add_error(
                    f"{field_name or 'Value'} length ({length}) must be >= {self.min_length}"
                )
            
            # 检查最大长度
            if self.max_length is not None and length > self.max_length:
                if self.validation_level == ValidationLevel.LENIENT and isinstance(data, str):
                    # 截断字符串
                    truncated = data[:self.max_length]
                    result.fixed_data = {field_name: truncated} if field_name else truncated
                    result.add_warning(
                        f"{field_name or 'Value'} length ({length}) exceeds maximum ({self.max_length}), "
                        f"truncated to {self.max_length} characters"
                    )
                else:
                    result.add_error(
                        f"{field_name or 'Value'} length ({length}) must be <= {self.max_length}"
                    )
            
            if not result.errors:
                result.success = True
                
        except Exception as e:
            result.add_error(f"Length validation failed for {field_name or 'value'}: {str(e)}")
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result


class PatternValidator(BaseValidator):
    """
    模式验证器
    
    使用正则表达式验证字符串是否符合指定模式。
    """
    
    def __init__(
        self,
        pattern: Union[str, Pattern],
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        初始化模式验证器
        
        Args:
            pattern: 正则表达式模式
            validation_level: 验证级别
        """
        super().__init__(validation_level)
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern
    
    def validate(self, data: Any, field_name: Optional[str] = None) -> ValidationResult:
        """验证模式匹配"""
        start_time = datetime.now()
        result = ValidationResult()
        
        try:
            # 确保数据是字符串
            if not isinstance(data, str):
                data_str = str(data)
                if self.validation_level in [ValidationLevel.WARNING, ValidationLevel.LENIENT]:
                    result.add_warning(
                        f"{field_name or 'Value'} is not a string, converted to string for pattern matching"
                    )
                else:
                    result.add_error(f"{field_name or 'Value'} must be a string for pattern validation")
                    return result
            else:
                data_str = data
            
            # 执行模式匹配
            if self.pattern.match(data_str):
                result.success = True
            else:
                result.add_error(
                    f"{field_name or 'Value'} ('{data_str}') does not match pattern: {self.pattern.pattern}"
                )
                
        except Exception as e:
            result.add_error(f"Pattern validation failed for {field_name or 'value'}: {str(e)}")
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result


class AllowedValuesValidator(BaseValidator):
    """
    允许值验证器
    
    验证数据是否在允许的值列表中。
    """
    
    def __init__(
        self,
        allowed_values: List[Any],
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        初始化允许值验证器
        
        Args:
            allowed_values: 允许的值列表
            validation_level: 验证级别
        """
        super().__init__(validation_level)
        self.allowed_values = set(allowed_values)  # 使用set提高查找效率
    
    def validate(self, data: Any, field_name: Optional[str] = None) -> ValidationResult:
        """验证值是否被允许"""
        start_time = datetime.now()
        result = ValidationResult()
        
        try:
            if data in self.allowed_values:
                result.success = True
            else:
                # 尝试类型转换后再检查
                converted_found = False
                if self.validation_level in [ValidationLevel.WARNING, ValidationLevel.LENIENT]:
                    for allowed_value in self.allowed_values:
                        try:
                            if type(allowed_value)(data) == allowed_value:
                                result.success = True
                                result.fixed_data = {field_name: allowed_value} if field_name else allowed_value
                                result.add_warning(
                                    f"{field_name or 'Value'} converted from {data} to {allowed_value}"
                                )
                                converted_found = True
                                break
                        except:
                            continue
                
                if not converted_found:
                    result.add_error(
                        f"{field_name or 'Value'} ('{data}') must be one of: {sorted(list(self.allowed_values))}"
                    )
                
        except Exception as e:
            result.add_error(f"Allowed values validation failed for {field_name or 'value'}: {str(e)}")
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result


class CustomValidator(BaseValidator):
    """
    自定义验证器
    
    支持自定义验证函数的验证器。
    """
    
    def __init__(
        self,
        validation_function: Callable[[Any], Union[bool, Tuple[bool, str]]],
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ):
        """
        初始化自定义验证器
        
        Args:
            validation_function: 验证函数，返回bool或(bool, error_message)元组
            validation_level: 验证级别
        """
        super().__init__(validation_level)
        self.validation_function = validation_function
    
    def validate(self, data: Any, field_name: Optional[str] = None) -> ValidationResult:
        """执行自定义验证"""
        start_time = datetime.now()
        result = ValidationResult()
        
        try:
            validation_result = self.validation_function(data)
            
            if isinstance(validation_result, tuple):
                is_valid, error_message = validation_result
                if not is_valid:
                    result.add_error(error_message)
                else:
                    result.success = True
            elif isinstance(validation_result, bool):
                if validation_result:
                    result.success = True
                else:
                    result.add_error(f"{field_name or 'Value'} failed custom validation")
            else:
                result.add_error("Custom validation function must return bool or (bool, str) tuple")
                
        except Exception as e:
            result.add_error(f"Custom validation failed for {field_name or 'value'}: {str(e)}")
        
        finally:
            result.validation_time = (datetime.now() - start_time).total_seconds()
        
        return result


class ParameterValidator:
    """
    参数验证器主类
    
    统一管理和执行多种验证规则，提供完整的参数验证功能。
    支持复杂的验证场景和灵活的配置选项。
    
    核心功能：
    1. 多重验证：支持对同一参数应用多个验证规则
    2. 嵌套验证：支持复杂数据结构的递归验证
    3. 批量验证：一次性验证多个参数
    4. 错误聚合：收集所有验证错误和警告
    5. 自动修复：在宽松模式下自动修复数据
    6. 性能监控：记录验证性能指标
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        """
        初始化参数验证器
        
        Args:
            validation_level: 默认验证级别
        """
        self.validation_level = validation_level
        self.validators: Dict[str, List[BaseValidator]] = {}
        self.global_validators: List[BaseValidator] = []
        
        logger.debug(f"Initialized ParameterValidator with level: {validation_level.value}")
    
    def add_validator(self, field_name: Optional[str], validator: BaseValidator) -> None:
        """
        添加验证器
        
        Args:
            field_name: 字段名称，None表示全局验证器
            validator: 验证器实例
        """
        if field_name is None:
            self.global_validators.append(validator)
        else:
            if field_name not in self.validators:
                self.validators[field_name] = []
            self.validators[field_name].append(validator)
        
        logger.debug(f"Added validator {validator.__class__.__name__} for field: {field_name or 'global'}")
    
    def add_type_validator(self, field_name: str, expected_type: Type) -> None:
        """添加类型验证器的便捷方法"""
        validator = TypeValidator(expected_type, self.validation_level)
        self.add_validator(field_name, validator)
    
    def add_range_validator(
        self, 
        field_name: str, 
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> None:
        """添加范围验证器的便捷方法"""
        validator = RangeValidator(min_value, max_value, self.validation_level)
        self.add_validator(field_name, validator)
    
    def add_length_validator(
        self,
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> None:
        """添加长度验证器的便捷方法"""
        validator = LengthValidator(min_length, max_length, self.validation_level)
        self.add_validator(field_name, validator)
    
    def add_pattern_validator(self, field_name: str, pattern: Union[str, Pattern]) -> None:
        """添加模式验证器的便捷方法"""
        validator = PatternValidator(pattern, self.validation_level)
        self.add_validator(field_name, validator)
    
    def add_allowed_values_validator(self, field_name: str, allowed_values: List[Any]) -> None:
        """添加允许值验证器的便捷方法"""
        validator = AllowedValuesValidator(allowed_values, self.validation_level)
        self.add_validator(field_name, validator)
    
    def add_custom_validator(
        self, 
        field_name: str, 
        validation_function: Callable[[Any], Union[bool, Tuple[bool, str]]]
    ) -> None:
        """添加自定义验证器的便捷方法"""
        validator = CustomValidator(validation_function, self.validation_level)
        self.add_validator(field_name, validator)
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        验证参数字典
        
        Args:
            data: 要验证的参数字典
            
        Returns:
            验证结果
        """
        start_time = datetime.now()
        overall_result = ValidationResult(success=True)
        fixed_data = data.copy()
        
        try:
            # 执行全局验证器
            for validator in self.global_validators:
                result = validator.validate(data)
                overall_result.merge(result)
                if result.fixed_data is not None:
                    if isinstance(result.fixed_data, dict):
                        fixed_data.update(result.fixed_data)
                    else:
                        # 全局验证器返回的是整个数据的修复版本
                        fixed_data = result.fixed_data
            
            # 执行字段级验证器
            for field_name, validators in self.validators.items():
                if field_name in data:
                    field_value = data[field_name]
                    
                    for validator in validators:
                        result = validator.validate(field_value, field_name)
                        overall_result.merge(result)
                        
                        # 应用修复的数据
                        if result.fixed_data is not None:
                            if isinstance(result.fixed_data, dict) and field_name in result.fixed_data:
                                fixed_data[field_name] = result.fixed_data[field_name]
                            else:
                                fixed_data[field_name] = result.fixed_data
                        
                        # 如果验证失败且是严格模式，可以选择提前退出
                        if not result.success and self.validation_level == ValidationLevel.STRICT:
                            # 继续执行其他验证，收集所有错误
                            pass
                else:
                    # 字段不存在，检查是否有必需验证器
                    for validator in validators:
                        if isinstance(validator, TypeValidator):
                            # 假设类型验证器意味着字段是必需的
                            overall_result.add_error(f"Required field '{field_name}' is missing")
            
            # 设置修复后的数据
            if fixed_data != data:
                overall_result.fixed_data = fixed_data
            
        except Exception as e:
            overall_result.add_error(f"Validation process failed: {str(e)}")
            logger.error(f"Validation process failed: {str(e)}")
        
        finally:
            overall_result.validation_time = (datetime.now() - start_time).total_seconds()
        
        logger.debug(
            f"Validation completed: {overall_result.success}, "
            f"{len(overall_result.errors)} errors, "
            f"{len(overall_result.warnings)} warnings, "
            f"time: {overall_result.validation_time:.3f}s"
        )
        
        return overall_result
    
    def validate_single(self, field_name: str, value: Any) -> ValidationResult:
        """
        验证单个字段
        
        Args:
            field_name: 字段名称
            value: 字段值
            
        Returns:
            验证结果
        """
        if field_name not in self.validators:
            return ValidationResult(success=True)
        
        overall_result = ValidationResult(success=True)
        
        for validator in self.validators[field_name]:
            result = validator.validate(value, field_name)
            overall_result.merge(result)
        
        return overall_result
    
    def clear_validators(self, field_name: Optional[str] = None) -> None:
        """
        清除验证器
        
        Args:
            field_name: 字段名称，None表示清除所有验证器
        """
        if field_name is None:
            self.validators.clear()
            self.global_validators.clear()
            logger.debug("Cleared all validators")
        elif field_name in self.validators:
            del self.validators[field_name]
            logger.debug(f"Cleared validators for field: {field_name}")
    
    def get_validator_info(self) -> Dict[str, Any]:
        """获取验证器信息"""
        info = {
            "validation_level": self.validation_level.value,
            "global_validators": len(self.global_validators),
            "field_validators": {},
            "total_validators": len(self.global_validators)
        }
        
        for field_name, validators in self.validators.items():
            info["field_validators"][field_name] = len(validators)
            info["total_validators"] += len(validators)
        
        return info


# 预定义的常用验证器工厂函数
def create_email_validator() -> PatternValidator:
    """创建邮箱地址验证器"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return PatternValidator(email_pattern)


def create_url_validator() -> PatternValidator:
    """创建URL验证器"""
    url_pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return PatternValidator(url_pattern)


def create_phone_validator() -> PatternValidator:
    """创建电话号码验证器"""
    phone_pattern = r'^\+?1?-?\.?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$'
    return PatternValidator(phone_pattern)


def create_positive_number_validator() -> CustomValidator:
    """创建正数验证器"""
    def validate_positive(value):
        try:
            num = float(value)
            return (num > 0, f"Value must be positive, got {num}")
        except (ValueError, TypeError):
            return (False, f"Value must be a number, got {type(value).__name__}")
    
    return CustomValidator(validate_positive)


def create_file_path_validator() -> CustomValidator:
    """创建文件路径验证器"""
    def validate_file_path(value):
        try:
            path = Path(value)
            if path.exists() and path.is_file():
                return True
            else:
                return (False, f"File does not exist: {value}")
        except Exception as e:
            return (False, f"Invalid file path: {e}")
    
    return CustomValidator(validate_file_path)