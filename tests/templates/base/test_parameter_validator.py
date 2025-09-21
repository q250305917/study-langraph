"""
ParameterValidator的单元测试

测试参数验证器的各种验证功能，包括类型验证、范围验证、长度验证、
模式验证、自定义验证等。
"""

import pytest
import re
from pathlib import Path
from datetime import datetime

from templates.base.parameter_validator import (
    ParameterValidator, ValidationResult, ValidationRule, ValidationLevel,
    ValidationType, TypeValidator, RangeValidator, LengthValidator,
    PatternValidator, AllowedValuesValidator, CustomValidator,
    create_email_validator, create_url_validator, create_phone_validator,
    create_positive_number_validator, create_file_path_validator
)


class TestValidationResult:
    """ValidationResult的单元测试"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        result = ValidationResult(success=True)
        assert result.success is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.fixed_data is None
    
    def test_error_management(self):
        """测试错误管理"""
        result = ValidationResult(success=True)
        
        result.add_error("First error")
        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0] == "First error"
        
        result.add_error("Second error")
        assert len(result.errors) == 2
    
    def test_warning_management(self):
        """测试警告管理"""
        result = ValidationResult(success=True)
        
        result.add_warning("First warning")
        assert result.success is True  # 警告不影响成功状态
        assert len(result.warnings) == 1
        
        result.add_warning("Second warning")
        assert len(result.warnings) == 2
    
    def test_result_merging(self):
        """测试结果合并"""
        result1 = ValidationResult(success=True)
        result1.add_warning("Warning 1")
        result1.validation_time = 0.1
        
        result2 = ValidationResult(success=False)
        result2.add_error("Error 1")
        result2.add_warning("Warning 2")
        result2.validation_time = 0.2
        
        result1.merge(result2)
        
        assert result1.success is False  # 合并后失败
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 2
        assert result1.validation_time == 0.3
    
    def test_string_representation(self):
        """测试字符串表示"""
        # 成功的结果
        success_result = ValidationResult(success=True)
        assert "SUCCESS" in str(success_result)
        
        # 带警告的成功结果
        success_with_warnings = ValidationResult(success=True)
        success_with_warnings.add_warning("Test warning")
        assert "SUCCESS" in str(success_with_warnings)
        assert "warnings" in str(success_with_warnings)
        
        # 失败的结果
        failure_result = ValidationResult(success=False)
        failure_result.add_error("Test error")
        assert "FAILED" in str(failure_result)


class TestTypeValidator:
    """TypeValidator的单元测试"""
    
    def test_basic_type_validation(self):
        """测试基本类型验证"""
        validator = TypeValidator(str)
        
        # 正确类型
        result = validator.validate("test string")
        assert result.success is True
        
        # 错误类型
        result = validator.validate(123)
        assert result.success is False
        assert len(result.errors) == 1
    
    def test_type_conversion_lenient(self):
        """测试宽松模式下的类型转换"""
        validator = TypeValidator(str, ValidationLevel.LENIENT)
        
        result = validator.validate(123)
        assert result.success is True
        assert result.fixed_data == "123"
    
    def test_type_conversion_warning(self):
        """测试警告模式下的类型转换"""
        validator = TypeValidator(int, ValidationLevel.WARNING)
        
        result = validator.validate("42")
        assert result.success is True
        assert result.fixed_data == 42
        assert len(result.warnings) == 1
    
    def test_numeric_conversions(self):
        """测试数值类型转换"""
        # 字符串到整数
        int_validator = TypeValidator(int, ValidationLevel.LENIENT)
        result = int_validator.validate("123")
        assert result.success is True
        assert result.fixed_data == 123
        
        # 字符串到浮点数
        float_validator = TypeValidator(float, ValidationLevel.LENIENT)
        result = float_validator.validate("123.45")
        assert result.success is True
        assert result.fixed_data == 123.45
        
        # 处理"1.0"格式的整数
        result = int_validator.validate("1.0")
        assert result.success is True
        assert result.fixed_data == 1
    
    def test_boolean_conversions(self):
        """测试布尔类型转换"""
        validator = TypeValidator(bool, ValidationLevel.LENIENT)
        
        # 真值转换
        true_values = ["true", "1", "yes", "on"]
        for value in true_values:
            result = validator.validate(value)
            assert result.success is True
            assert result.fixed_data is True
        
        # 假值转换
        false_values = ["false", "0", "no", "off"]
        for value in false_values:
            result = validator.validate(value)
            assert result.success is True
            assert result.fixed_data is False
    
    def test_list_conversions(self):
        """测试列表类型转换"""
        validator = TypeValidator(list, ValidationLevel.LENIENT)
        
        # JSON字符串转换
        result = validator.validate('["a", "b", "c"]')
        assert result.success is True
        assert result.fixed_data == ["a", "b", "c"]
        
        # 逗号分隔字符串转换
        result = validator.validate("a, b, c")
        assert result.success is True
        assert result.fixed_data == ["a", "b", "c"]
        
        # 元组转换
        result = validator.validate(("a", "b", "c"))
        assert result.success is True
        assert result.fixed_data == ["a", "b", "c"]
    
    def test_dict_conversions(self):
        """测试字典类型转换"""
        validator = TypeValidator(dict, ValidationLevel.LENIENT)
        
        # JSON字符串转换
        result = validator.validate('{"key": "value"}')
        assert result.success is True
        assert result.fixed_data == {"key": "value"}
    
    def test_path_conversions(self):
        """测试Path类型转换"""
        validator = TypeValidator(Path, ValidationLevel.LENIENT)
        
        result = validator.validate("/tmp/test.txt")
        assert result.success is True
        assert isinstance(result.fixed_data, Path)
        assert str(result.fixed_data) == "/tmp/test.txt"
    
    def test_conversion_failure(self):
        """测试转换失败的情况"""
        validator = TypeValidator(int, ValidationLevel.LENIENT)
        
        # 无法转换的字符串
        result = validator.validate("not_a_number")
        assert result.success is False
        assert result.fixed_data is None


class TestRangeValidator:
    """RangeValidator的单元测试"""
    
    def test_valid_range(self):
        """测试有效范围"""
        validator = RangeValidator(min_value=0, max_value=100)
        
        # 在范围内
        result = validator.validate(50)
        assert result.success is True
        
        # 边界值
        result = validator.validate(0)
        assert result.success is True
        
        result = validator.validate(100)
        assert result.success is True
    
    def test_out_of_range(self):
        """测试超出范围"""
        validator = RangeValidator(min_value=0, max_value=100)
        
        # 低于最小值
        result = validator.validate(-10)
        assert result.success is False
        assert "must be >= 0" in result.errors[0]
        
        # 高于最大值
        result = validator.validate(150)
        assert result.success is False
        assert "must be <= 100" in result.errors[0]
    
    def test_range_adjustment_lenient(self):
        """测试宽松模式下的范围调整"""
        validator = RangeValidator(min_value=0, max_value=100, validation_level=ValidationLevel.LENIENT)
        
        # 低于最小值，调整到最小值
        result = validator.validate(-10)
        assert result.success is True
        assert result.fixed_data == 0
        assert len(result.warnings) == 1
        
        # 高于最大值，调整到最大值
        result = validator.validate(150)
        assert result.success is True
        assert result.fixed_data == 100
        assert len(result.warnings) == 1
    
    def test_only_min_value(self):
        """测试只有最小值的情况"""
        validator = RangeValidator(min_value=10)
        
        result = validator.validate(5)
        assert result.success is False
        
        result = validator.validate(15)
        assert result.success is True
    
    def test_only_max_value(self):
        """测试只有最大值的情况"""
        validator = RangeValidator(max_value=10)
        
        result = validator.validate(15)
        assert result.success is False
        
        result = validator.validate(5)
        assert result.success is True
    
    def test_string_number_conversion(self):
        """测试字符串数字的转换"""
        validator = RangeValidator(min_value=0, max_value=100)
        
        result = validator.validate("50")
        assert result.success is True
        
        result = validator.validate("not_a_number")
        assert result.success is False


class TestLengthValidator:
    """LengthValidator的单元测试"""
    
    def test_string_length_validation(self):
        """测试字符串长度验证"""
        validator = LengthValidator(min_length=3, max_length=10)
        
        # 有效长度
        result = validator.validate("hello")
        assert result.success is True
        
        # 太短
        result = validator.validate("hi")
        assert result.success is False
        assert "length (2) must be >= 3" in result.errors[0]
        
        # 太长
        result = validator.validate("this_is_too_long")
        assert result.success is False
        assert "length (16) must be <= 10" in result.errors[0]
        
        # 边界值
        result = validator.validate("abc")  # 长度3
        assert result.success is True
        
        result = validator.validate("1234567890")  # 长度10
        assert result.success is True
    
    def test_list_length_validation(self):
        """测试列表长度验证"""
        validator = LengthValidator(min_length=2, max_length=5)
        
        # 有效长度
        result = validator.validate([1, 2, 3])
        assert result.success is True
        
        # 太短
        result = validator.validate([1])
        assert result.success is False
        
        # 太长
        result = validator.validate([1, 2, 3, 4, 5, 6])
        assert result.success is False
    
    def test_string_truncation_lenient(self):
        """测试宽松模式下的字符串截断"""
        validator = LengthValidator(max_length=5, validation_level=ValidationLevel.LENIENT)
        
        result = validator.validate("this_is_too_long")
        assert result.success is True
        assert result.fixed_data == "this_"
        assert len(result.warnings) == 1
    
    def test_only_min_length(self):
        """测试只有最小长度的情况"""
        validator = LengthValidator(min_length=5)
        
        result = validator.validate("short")
        assert result.success is True
        
        result = validator.validate("hi")
        assert result.success is False
    
    def test_only_max_length(self):
        """测试只有最大长度的情况"""
        validator = LengthValidator(max_length=5)
        
        result = validator.validate("hello")
        assert result.success is True
        
        result = validator.validate("toolong")
        assert result.success is False
    
    def test_no_length_attribute(self):
        """测试没有长度属性的对象"""
        validator = LengthValidator(min_length=1)
        
        result = validator.validate(42)  # int没有__len__方法
        assert result.success is False
        assert "does not have length attribute" in result.errors[0]


class TestPatternValidator:
    """PatternValidator的单元测试"""
    
    def test_string_pattern_matching(self):
        """测试字符串模式匹配"""
        # 数字模式
        validator = PatternValidator(r'^\d+$')
        
        result = validator.validate("12345")
        assert result.success is True
        
        result = validator.validate("abc123")
        assert result.success is False
    
    def test_compiled_pattern(self):
        """测试编译后的正则表达式"""
        pattern = re.compile(r'^[a-zA-Z]+$')
        validator = PatternValidator(pattern)
        
        result = validator.validate("hello")
        assert result.success is True
        
        result = validator.validate("hello123")
        assert result.success is False
    
    def test_non_string_input(self):
        """测试非字符串输入"""
        validator = PatternValidator(r'^\d+$')
        
        # 严格模式下应该失败
        validator_strict = PatternValidator(r'^\d+$', ValidationLevel.STRICT)
        result = validator_strict.validate(123)
        assert result.success is False
        assert "must be a string" in result.errors[0]
        
        # 警告模式下应该转换
        validator_warning = PatternValidator(r'^\d+$', ValidationLevel.WARNING)
        result = validator_warning.validate(123)
        assert result.success is True
        assert len(result.warnings) == 1
    
    def test_email_pattern(self):
        """测试邮箱模式"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        validator = PatternValidator(email_pattern)
        
        # 有效邮箱
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ]
        
        for email in valid_emails:
            result = validator.validate(email)
            assert result.success is True, f"Should accept email: {email}"
        
        # 无效邮箱
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test.example.com"
        ]
        
        for email in invalid_emails:
            result = validator.validate(email)
            assert result.success is False, f"Should reject email: {email}"


class TestAllowedValuesValidator:
    """AllowedValuesValidator的单元测试"""
    
    def test_valid_values(self):
        """测试有效值"""
        allowed_values = ["red", "green", "blue"]
        validator = AllowedValuesValidator(allowed_values)
        
        # 有效值
        for value in allowed_values:
            result = validator.validate(value)
            assert result.success is True
        
        # 无效值
        result = validator.validate("yellow")
        assert result.success is False
        assert "must be one of" in result.errors[0]
    
    def test_mixed_type_values(self):
        """测试混合类型的允许值"""
        allowed_values = [1, "two", 3.0, True]
        validator = AllowedValuesValidator(allowed_values)
        
        for value in allowed_values:
            result = validator.validate(value)
            assert result.success is True
        
        result = validator.validate(2)
        assert result.success is False
    
    def test_type_conversion_lenient(self):
        """测试宽松模式下的类型转换"""
        allowed_values = [1, 2, 3]
        validator = AllowedValuesValidator(allowed_values, ValidationLevel.LENIENT)
        
        # 字符串"1"应该转换为整数1
        result = validator.validate("1")
        assert result.success is True
        assert result.fixed_data == 1
        assert len(result.warnings) == 1
    
    def test_empty_allowed_values(self):
        """测试空的允许值列表"""
        validator = AllowedValuesValidator([])
        
        result = validator.validate("any_value")
        assert result.success is False


class TestCustomValidator:
    """CustomValidator的单元测试"""
    
    def test_simple_custom_validation(self):
        """测试简单的自定义验证"""
        def is_even(value):
            try:
                num = int(value)
                return num % 2 == 0
            except:
                return False
        
        validator = CustomValidator(is_even)
        
        result = validator.validate(4)
        assert result.success is True
        
        result = validator.validate(3)
        assert result.success is False
    
    def test_custom_validation_with_message(self):
        """测试带自定义错误消息的验证"""
        def validate_positive_with_message(value):
            try:
                num = float(value)
                if num > 0:
                    return True
                else:
                    return (False, f"Value must be positive, got {num}")
            except:
                return (False, f"Value must be a number, got {type(value).__name__}")
        
        validator = CustomValidator(validate_positive_with_message)
        
        result = validator.validate(5)
        assert result.success is True
        
        result = validator.validate(-5)
        assert result.success is False
        assert "must be positive" in result.errors[0]
        
        result = validator.validate("not_a_number")
        assert result.success is False
        assert "must be a number" in result.errors[0]
    
    def test_custom_validation_exception(self):
        """测试自定义验证函数抛出异常"""
        def failing_validator(value):
            raise ValueError("Validation function failed")
        
        validator = CustomValidator(failing_validator)
        
        result = validator.validate("any_value")
        assert result.success is False
        assert "Custom validation failed" in result.errors[0]
    
    def test_invalid_return_type(self):
        """测试无效的返回类型"""
        def invalid_validator(value):
            return "invalid_return_type"
        
        validator = CustomValidator(invalid_validator)
        
        result = validator.validate("any_value")
        assert result.success is False
        assert "must return bool or (bool, str) tuple" in result.errors[0]


class TestParameterValidator:
    """ParameterValidator主类的单元测试"""
    
    def test_basic_initialization(self):
        """测试基本初始化"""
        validator = ParameterValidator()
        
        assert validator.validation_level == ValidationLevel.STRICT
        assert len(validator.validators) == 0
        assert len(validator.global_validators) == 0
    
    def test_add_validators(self):
        """测试添加验证器"""
        validator = ParameterValidator()
        
        # 添加类型验证器
        validator.add_type_validator("name", str)
        validator.add_type_validator("age", int)
        
        assert "name" in validator.validators
        assert "age" in validator.validators
        assert len(validator.validators["name"]) == 1
        assert len(validator.validators["age"]) == 1
    
    def test_convenience_methods(self):
        """测试便捷方法"""
        validator = ParameterValidator()
        
        # 添加各种类型的验证器
        validator.add_range_validator("score", min_value=0, max_value=100)
        validator.add_length_validator("description", min_length=10, max_length=500)
        validator.add_pattern_validator("email", r'^[^@]+@[^@]+\.[^@]+$')
        validator.add_allowed_values_validator("status", ["active", "inactive", "pending"])
        validator.add_custom_validator("id", lambda x: isinstance(x, int) and x > 0)
        
        assert "score" in validator.validators
        assert "description" in validator.validators
        assert "email" in validator.validators
        assert "status" in validator.validators
        assert "id" in validator.validators
    
    def test_single_field_validation(self):
        """测试单字段验证"""
        validator = ParameterValidator()
        validator.add_type_validator("name", str)
        validator.add_range_validator("age", min_value=0, max_value=150)
        
        # 有效值
        result = validator.validate_single("name", "John")
        assert result.success is True
        
        result = validator.validate_single("age", 25)
        assert result.success is True
        
        # 无效值
        result = validator.validate_single("name", 123)
        assert result.success is False
        
        result = validator.validate_single("age", -5)
        assert result.success is False
        
        # 未知字段
        result = validator.validate_single("unknown_field", "value")
        assert result.success is True  # 没有验证器的字段默认通过
    
    def test_batch_validation(self):
        """测试批量验证"""
        validator = ParameterValidator()
        validator.add_type_validator("name", str)
        validator.add_type_validator("age", int)
        validator.add_range_validator("age", min_value=0, max_value=150)
        validator.add_length_validator("name", min_length=2, max_length=50)
        
        # 有效数据
        valid_data = {
            "name": "John Doe",
            "age": 30
        }
        
        result = validator.validate(valid_data)
        assert result.success is True
        assert len(result.errors) == 0
        
        # 无效数据
        invalid_data = {
            "name": "J",  # 太短
            "age": -5     # 负数
        }
        
        result = validator.validate(invalid_data)
        assert result.success is False
        assert len(result.errors) >= 2  # 至少两个错误
    
    def test_missing_required_fields(self):
        """测试缺少必需字段"""
        validator = ParameterValidator()
        validator.add_type_validator("required_field", str)
        
        # 缺少必需字段
        data = {"other_field": "value"}
        result = validator.validate(data)
        
        assert result.success is False
        assert any("Required field 'required_field' is missing" in error for error in result.errors)
    
    def test_validation_level_strict(self):
        """测试严格验证级别"""
        validator = ParameterValidator(ValidationLevel.STRICT)
        validator.add_type_validator("number", int)
        
        data = {"number": "123"}  # 字符串而不是整数
        result = validator.validate(data)
        
        assert result.success is False
    
    def test_validation_level_lenient(self):
        """测试宽松验证级别"""
        validator = ParameterValidator(ValidationLevel.LENIENT)
        validator.add_type_validator("number", int)
        
        data = {"number": "123"}  # 字符串可以转换为整数
        result = validator.validate(data)
        
        assert result.success is True
        assert result.fixed_data is not None
        assert result.fixed_data["number"] == 123
    
    def test_multiple_validators_per_field(self):
        """测试单个字段的多个验证器"""
        validator = ParameterValidator()
        validator.add_type_validator("password", str)
        validator.add_length_validator("password", min_length=8, max_length=100)
        validator.add_pattern_validator("password", r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)')
        
        # 密码太短
        result = validator.validate({"password": "Ab1"})
        assert result.success is False
        
        # 没有大写字母
        result = validator.validate({"password": "abcd1234"})
        assert result.success is False
        
        # 有效密码
        result = validator.validate({"password": "ValidPassword123"})
        assert result.success is True
    
    def test_global_validators(self):
        """测试全局验证器"""
        def validate_no_empty_strings(data):
            for key, value in data.items():
                if isinstance(value, str) and not value.strip():
                    return (False, f"Field '{key}' cannot be empty")
            return True
        
        validator = ParameterValidator()
        global_validator = CustomValidator(validate_no_empty_strings)
        validator.add_validator(None, global_validator)  # None表示全局验证器
        
        # 包含空字符串的数据
        data = {"name": "John", "description": "   "}
        result = validator.validate(data)
        
        assert result.success is False
        assert "cannot be empty" in result.errors[0]
    
    def test_clear_validators(self):
        """测试清除验证器"""
        validator = ParameterValidator()
        validator.add_type_validator("field1", str)
        validator.add_type_validator("field2", int)
        
        assert len(validator.validators) == 2
        
        # 清除特定字段的验证器
        validator.clear_validators("field1")
        assert "field1" not in validator.validators
        assert "field2" in validator.validators
        
        # 清除所有验证器
        validator.clear_validators()
        assert len(validator.validators) == 0
    
    def test_validator_info(self):
        """测试验证器信息"""
        validator = ParameterValidator(ValidationLevel.WARNING)
        validator.add_type_validator("field1", str)
        validator.add_range_validator("field2", min_value=0, max_value=100)
        
        global_validator = CustomValidator(lambda x: True)
        validator.add_validator(None, global_validator)
        
        info = validator.get_validator_info()
        
        assert info["validation_level"] == "warning"
        assert info["global_validators"] == 1
        assert info["field_validators"]["field1"] == 1
        assert info["field_validators"]["field2"] == 1
        assert info["total_validators"] == 3


class TestPrebuiltValidators:
    """预构建验证器的单元测试"""
    
    def test_email_validator(self):
        """测试邮箱验证器"""
        validator = create_email_validator()
        
        # 有效邮箱
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ]
        
        for email in valid_emails:
            result = validator.validate(email)
            assert result.success is True, f"Should accept email: {email}"
        
        # 无效邮箱
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test.example.com"
        ]
        
        for email in invalid_emails:
            result = validator.validate(email)
            assert result.success is False, f"Should reject email: {email}"
    
    def test_url_validator(self):
        """测试URL验证器"""
        validator = create_url_validator()
        
        # 有效URL
        valid_urls = [
            "http://example.com",
            "https://www.example.com",
            "https://example.com/path/to/page?param=value#section"
        ]
        
        for url in valid_urls:
            result = validator.validate(url)
            assert result.success is True, f"Should accept URL: {url}"
        
        # 无效URL
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # 不支持ftp
            "example.com",        # 缺少协议
        ]
        
        for url in invalid_urls:
            result = validator.validate(url)
            assert result.success is False, f"Should reject URL: {url}"
    
    def test_positive_number_validator(self):
        """测试正数验证器"""
        validator = create_positive_number_validator()
        
        # 正数
        result = validator.validate(5)
        assert result.success is True
        
        result = validator.validate("3.14")
        assert result.success is True
        
        # 非正数
        result = validator.validate(0)
        assert result.success is False
        
        result = validator.validate(-5)
        assert result.success is False
        
        # 非数字
        result = validator.validate("not_a_number")
        assert result.success is False
    
    def test_file_path_validator(self):
        """测试文件路径验证器"""
        validator = create_file_path_validator()
        
        # 创建一个临时文件用于测试
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            temp_file_path = tf.name
        
        try:
            # 存在的文件
            result = validator.validate(temp_file_path)
            assert result.success is True
            
            # 不存在的文件
            result = validator.validate("/nonexistent/file.txt")
            assert result.success is False
            assert "does not exist" in result.errors[0]
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


if __name__ == "__main__":
    pytest.main([__file__])