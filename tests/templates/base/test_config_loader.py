"""
ConfigLoader的单元测试

测试配置加载器的各种功能，包括文件加载、环境变量加载、
配置合并、验证、缓存等。
"""

import pytest
import os
import json
import yaml
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from templates.base.config_loader import (
    ConfigLoader, ConfigSource, ConfigFormat, ConfigSourceType,
    ConfigCache, FileConfigLoader, EnvironmentConfigLoader, 
    DictConfigLoader, ConfigWatcher
)
from templates.base.template_base import TemplateConfig, TemplateType, ParameterSchema


class TestConfigCache:
    """ConfigCache的单元测试"""
    
    def test_basic_cache_operations(self):
        """测试基本缓存操作"""
        cache = ConfigCache()
        
        # 设置缓存
        data = {"key": "value"}
        cache.set("test_key", data, ttl=60)
        
        # 获取缓存
        cached_data = cache.get("test_key")
        assert cached_data == data
        assert cached_data is not data  # 应该是副本
        
        # 不存在的键
        assert cache.get("nonexistent") is None
    
    def test_cache_expiration(self):
        """测试缓存过期"""
        cache = ConfigCache()
        
        # 设置短TTL的缓存
        data = {"key": "value"}
        cache.set("test_key", data, ttl=0.1)  # 0.1秒
        
        # 立即获取应该成功
        assert cache.get("test_key") == data
        
        # 等待过期
        time.sleep(0.2)
        
        # 过期后应该返回None
        assert cache.get("test_key") is None
    
    def test_cache_removal(self):
        """测试缓存删除"""
        cache = ConfigCache()
        
        cache.set("test_key", {"key": "value"})
        assert cache.get("test_key") is not None
        
        cache.remove("test_key")
        assert cache.get("test_key") is None
    
    def test_cache_clear(self):
        """测试缓存清空"""
        cache = ConfigCache()
        
        cache.set("key1", {"data": "1"})
        cache.set("key2", {"data": "2"})
        
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_stats(self):
        """测试缓存统计"""
        cache = ConfigCache()
        
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        
        cache.set("key1", {"data": "1"})
        cache.set("key2", {"data": "2"})
        
        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert "key1" in stats["cache_keys"]
        assert "key2" in stats["cache_keys"]


class TestConfigSource:
    """ConfigSource的单元测试"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        source = ConfigSource(
            source_type=ConfigSourceType.FILE,
            source_path="/path/to/config.yaml",
            priority=100
        )
        
        assert source.source_type == ConfigSourceType.FILE
        assert source.source_path == "/path/to/config.yaml"
        assert source.priority == 100
        assert source.format == ConfigFormat.YAML  # 自动检测
    
    def test_format_detection(self):
        """测试格式自动检测"""
        test_cases = [
            ("/path/config.yaml", ConfigFormat.YAML),
            ("/path/config.yml", ConfigFormat.YAML),
            ("/path/config.json", ConfigFormat.JSON),
            ("/path/.env", ConfigFormat.ENV),
            ("/path/config", ConfigFormat.YAML)  # 默认
        ]
        
        for path, expected_format in test_cases:
            source = ConfigSource(
                source_type=ConfigSourceType.FILE,
                source_path=path
            )
            assert source.format == expected_format
    
    def test_environment_source(self):
        """测试环境变量源"""
        source = ConfigSource(
            source_type=ConfigSourceType.ENVIRONMENT,
            source_path="environment"
        )
        
        assert source.format == ConfigFormat.ENV


class TestFileConfigLoader:
    """FileConfigLoader的单元测试"""
    
    def test_yaml_file_loading(self):
        """测试YAML文件加载"""
        loader = FileConfigLoader()
        
        # 创建临时YAML文件
        yaml_data = {
            "name": "TestTemplate",
            "version": "1.0.0",
            "parameters": {
                "input_text": {
                    "type": "str",
                    "required": True
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_data, f)
            temp_path = f.name
        
        try:
            source = ConfigSource(
                source_type=ConfigSourceType.FILE,
                source_path=temp_path,
                format=ConfigFormat.YAML
            )
            
            loaded_data = loader.load(source)
            
            assert loaded_data["name"] == "TestTemplate"
            assert loaded_data["version"] == "1.0.0"
            assert "parameters" in loaded_data
            
        finally:
            os.unlink(temp_path)
    
    def test_json_file_loading(self):
        """测试JSON文件加载"""
        loader = FileConfigLoader()
        
        json_data = {
            "name": "JSONTemplate",
            "description": "A JSON-based template config"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name
        
        try:
            source = ConfigSource(
                source_type=ConfigSourceType.FILE,
                source_path=temp_path,
                format=ConfigFormat.JSON
            )
            
            loaded_data = loader.load(source)
            
            assert loaded_data["name"] == "JSONTemplate"
            assert loaded_data["description"] == "A JSON-based template config"
            
        finally:
            os.unlink(temp_path)
    
    def test_env_file_loading(self):
        """测试.env文件加载"""
        loader = FileConfigLoader()
        
        env_content = """
# This is a comment
NAME=EnvTemplate
VERSION=2.0.0
DEBUG=true
PORT=8080
TAGS=tag1,tag2,tag3
EMPTY_VALUE=
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            temp_path = f.name
        
        try:
            source = ConfigSource(
                source_type=ConfigSourceType.FILE,
                source_path=temp_path,
                format=ConfigFormat.ENV
            )
            
            loaded_data = loader.load(source)
            
            assert loaded_data["NAME"] == "EnvTemplate"
            assert loaded_data["VERSION"] == "2.0.0"
            assert loaded_data["DEBUG"] is True
            assert loaded_data["PORT"] == 8080
            assert loaded_data["TAGS"] == ["tag1", "tag2", "tag3"]
            assert loaded_data["EMPTY_VALUE"] == ""
            
        finally:
            os.unlink(temp_path)
    
    def test_environment_variable_substitution(self):
        """测试环境变量替换"""
        loader = FileConfigLoader()
        
        # 设置测试环境变量
        os.environ["TEST_VAR"] = "test_value"
        os.environ["TEST_PORT"] = "9000"
        
        try:
            yaml_data = {
                "host": "${TEST_VAR}",
                "port": "${TEST_PORT}",
                "fallback": "${NONEXISTENT:default_value}",
                "nested": {
                    "value": "${TEST_VAR}_suffix"
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(yaml_data, f)
                temp_path = f.name
            
            try:
                source = ConfigSource(
                    source_type=ConfigSourceType.FILE,
                    source_path=temp_path
                )
                
                loaded_data = loader.load(source)
                
                assert loaded_data["host"] == "test_value"
                assert loaded_data["port"] == "9000"
                assert loaded_data["fallback"] == "default_value"
                assert loaded_data["nested"]["value"] == "test_value_suffix"
                
            finally:
                os.unlink(temp_path)
                
        finally:
            # 清理环境变量
            del os.environ["TEST_VAR"]
            del os.environ["TEST_PORT"]
    
    def test_config_inheritance(self):
        """测试配置继承"""
        loader = FileConfigLoader()
        
        # 创建父配置文件
        parent_config = {
            "name": "ParentTemplate",
            "version": "1.0.0",
            "parameters": {
                "base_param": {
                    "type": "str",
                    "required": True
                }
            }
        }
        
        # 创建子配置文件
        child_config = {
            "extends": ["parent.yaml"],
            "name": "ChildTemplate",  # 覆盖父配置
            "parameters": {
                "child_param": {
                    "type": "int",
                    "required": False
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 写入父配置
            parent_path = Path(temp_dir) / "parent.yaml"
            with open(parent_path, 'w') as f:
                yaml.dump(parent_config, f)
            
            # 写入子配置
            child_path = Path(temp_dir) / "child.yaml"
            with open(child_path, 'w') as f:
                yaml.dump(child_config, f)
            
            source = ConfigSource(
                source_type=ConfigSourceType.FILE,
                source_path=str(child_path)
            )
            
            loaded_data = loader.load(source)
            
            # 检查继承结果
            assert loaded_data["name"] == "ChildTemplate"  # 子配置覆盖
            assert loaded_data["version"] == "1.0.0"       # 从父配置继承
            assert "base_param" in loaded_data["parameters"]  # 从父配置继承
            assert "child_param" in loaded_data["parameters"]  # 子配置添加
    
    def test_file_not_found(self):
        """测试文件不存在的情况"""
        loader = FileConfigLoader()
        
        # 必需文件不存在
        source = ConfigSource(
            source_type=ConfigSourceType.FILE,
            source_path="/nonexistent/file.yaml",
            required=True
        )
        
        with pytest.raises(Exception):  # 应该抛出ConfigurationError
            loader.load(source)
        
        # 可选文件不存在
        source = ConfigSource(
            source_type=ConfigSourceType.FILE,
            source_path="/nonexistent/file.yaml",
            required=False
        )
        
        loaded_data = loader.load(source)
        assert loaded_data == {}
    
    def test_invalid_yaml_format(self):
        """测试无效的YAML格式"""
        loader = FileConfigLoader()
        
        invalid_yaml = """
name: TestTemplate
version: 1.0.0
parameters:
  - invalid_yaml_structure
    missing_key:
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            source = ConfigSource(
                source_type=ConfigSourceType.FILE,
                source_path=temp_path
            )
            
            with pytest.raises(Exception):  # 应该抛出ConfigurationError
                loader.load(source)
                
        finally:
            os.unlink(temp_path)


class TestEnvironmentConfigLoader:
    """EnvironmentConfigLoader的单元测试"""
    
    def test_basic_environment_loading(self):
        """测试基本环境变量加载"""
        loader = EnvironmentConfigLoader()
        
        # 设置测试环境变量
        test_env_vars = {
            "TEMPLATE_NAME": "EnvTemplate",
            "TEMPLATE_VERSION": "1.0.0",
            "TEMPLATE_DEBUG": "true",
            "TEMPLATE_PORT": "8080"
        }
        
        # 备份原有环境变量
        original_vars = {}
        for key in test_env_vars:
            original_vars[key] = os.environ.get(key)
            os.environ[key] = test_env_vars[key]
        
        try:
            source = ConfigSource(
                source_type=ConfigSourceType.ENVIRONMENT,
                source_path="environment",
                options={"prefix": "TEMPLATE_"}
            )
            
            loaded_data = loader.load(source)
            
            assert loaded_data["name"] == "EnvTemplate"
            assert loaded_data["version"] == "1.0.0"
            assert loaded_data["debug"] is True
            assert loaded_data["port"] == 8080
            
        finally:
            # 恢复原有环境变量
            for key, value in original_vars.items():
                if value is None:
                    if key in os.environ:
                        del os.environ[key]
                else:
                    os.environ[key] = value
    
    def test_nested_environment_variables(self):
        """测试嵌套环境变量"""
        loader = EnvironmentConfigLoader()
        
        # 设置嵌套环境变量（使用双下划线分隔）
        test_env_vars = {
            "APP_DATABASE__HOST": "localhost",
            "APP_DATABASE__PORT": "5432",
            "APP_DATABASE__NAME": "mydb"
        }
        
        original_vars = {}
        for key in test_env_vars:
            original_vars[key] = os.environ.get(key)
            os.environ[key] = test_env_vars[key]
        
        try:
            source = ConfigSource(
                source_type=ConfigSourceType.ENVIRONMENT,
                source_path="environment",
                options={"prefix": "APP_"}
            )
            
            loaded_data = loader.load(source)
            
            assert "database" in loaded_data
            assert loaded_data["database"]["host"] == "localhost"
            assert loaded_data["database"]["port"] == 5432
            assert loaded_data["database"]["name"] == "mydb"
            
        finally:
            for key, value in original_vars.items():
                if value is None:
                    if key in os.environ:
                        del os.environ[key]
                else:
                    os.environ[key] = value
    
    def test_type_conversion(self):
        """测试类型转换"""
        loader = EnvironmentConfigLoader()
        
        test_cases = [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("42", 42),
            ("3.14", 3.14),
            ('{"key": "value"}', {"key": "value"}),
            ('["a", "b", "c"]', ["a", "b", "c"]),
            ("a,b,c", ["a", "b", "c"]),
            ("simple_string", "simple_string")
        ]
        
        for env_value, expected in test_cases:
            converted = loader._convert_env_value(env_value)
            assert converted == expected, f"Failed to convert {env_value} to {expected}, got {converted}"


class TestDictConfigLoader:
    """DictConfigLoader的单元测试"""
    
    def test_dict_loading(self):
        """测试字典加载"""
        loader = DictConfigLoader()
        
        config_dict = {
            "name": "DictTemplate",
            "version": "1.0.0",
            "parameters": {
                "param1": {
                    "type": "str",
                    "required": True
                }
            }
        }
        
        source = ConfigSource(
            source_type=ConfigSourceType.DICT,
            source_path=config_dict  # 直接传递字典
        )
        
        loaded_data = loader.load(source)
        
        assert loaded_data["name"] == "DictTemplate"
        assert loaded_data["version"] == "1.0.0"
        assert "parameters" in loaded_data
    
    def test_environment_variable_substitution_in_dict(self):
        """测试字典中的环境变量替换"""
        loader = DictConfigLoader()
        
        os.environ["TEST_DICT_VAR"] = "dict_test_value"
        
        try:
            config_dict = {
                "name": "${TEST_DICT_VAR}",
                "nested": {
                    "value": "${TEST_DICT_VAR}_nested"
                }
            }
            
            source = ConfigSource(
                source_type=ConfigSourceType.DICT,
                source_path=config_dict
            )
            
            loaded_data = loader.load(source)
            
            assert loaded_data["name"] == "dict_test_value"
            assert loaded_data["nested"]["value"] == "dict_test_value_nested"
            
        finally:
            del os.environ["TEST_DICT_VAR"]


class TestConfigLoader:
    """ConfigLoader主类的单元测试"""
    
    def test_basic_initialization(self):
        """测试基本初始化"""
        loader = ConfigLoader()
        
        assert loader.cache is not None
        assert len(loader.sources) == 0
        assert len(loader.loaders) == 3  # FILE, ENVIRONMENT, DICT
    
    def test_add_sources(self):
        """测试添加配置源"""
        loader = ConfigLoader()
        
        # 添加文件源
        loader.add_file_source("config.yaml", priority=50)
        
        # 添加环境变量源
        loader.add_env_source(prefix="APP_", priority=100)
        
        # 添加字典源
        loader.add_dict_source({"key": "value"}, priority=25)
        
        assert len(loader.sources) == 3
        
        # 检查优先级排序
        priorities = [source.priority for source in loader.sources]
        assert priorities == sorted(priorities)  # 应该按优先级排序
    
    def test_config_merging(self):
        """测试配置合并"""
        loader = ConfigLoader()
        
        # 添加多个配置源
        base_config = {
            "name": "BaseTemplate",
            "version": "1.0.0",
            "parameters": {
                "base_param": {"type": "str"}
            }
        }
        
        override_config = {
            "name": "OverrideTemplate",  # 覆盖基础配置
            "parameters": {
                "override_param": {"type": "int"}  # 添加新参数
            }
        }
        
        loader.add_dict_source(base_config, priority=10)
        loader.add_dict_source(override_config, priority=20)
        
        merged_config = loader._merge_configs([
            (10, base_config),
            (20, override_config)
        ])
        
        assert merged_config["name"] == "OverrideTemplate"  # 高优先级覆盖
        assert merged_config["version"] == "1.0.0"         # 低优先级保留
        assert len(merged_config["parameters"]) == 2       # 参数合并
    
    def test_template_config_conversion(self):
        """测试转换为TemplateConfig对象"""
        loader = ConfigLoader()
        
        config_data = {
            "name": "TestTemplate",
            "version": "2.0.0",
            "template_type": "llm",
            "description": "A test template",
            "parameters": {
                "input_text": {
                    "type": "str",
                    "required": True,
                    "description": "Input text parameter"
                },
                "max_tokens": {
                    "type": "int",
                    "required": False,
                    "default": 100
                }
            },
            "dependencies": ["langchain", "openai"],
            "tags": ["test", "example"]
        }
        
        template_config = loader._dict_to_template_config(config_data)
        
        assert isinstance(template_config, TemplateConfig)
        assert template_config.name == "TestTemplate"
        assert template_config.version == "2.0.0"
        assert template_config.template_type == TemplateType.LLM
        assert template_config.description == "A test template"
        assert len(template_config.parameters) == 2
        assert "langchain" in template_config.dependencies
        assert "test" in template_config.tags
        
        # 检查参数转换
        input_param = template_config.get_parameter_schema("input_text")
        assert input_param.name == "input_text"
        assert input_param.type == str
        assert input_param.required is True
        
        tokens_param = template_config.get_parameter_schema("max_tokens")
        assert tokens_param.type == int
        assert tokens_param.required is False
        assert tokens_param.default == 100
    
    def test_load_config_with_sources(self):
        """测试从多个源加载配置"""
        loader = ConfigLoader(cache_enabled=False)  # 禁用缓存以便测试
        
        # 添加基础配置
        base_config = {
            "name": "MultiSourceTemplate",
            "version": "1.0.0"
        }
        loader.add_dict_source(base_config, priority=10)
        
        # 模拟环境变量覆盖
        os.environ["TEMPLATE_VERSION"] = "2.0.0"
        os.environ["TEMPLATE_DEBUG"] = "true"
        
        try:
            loader.add_env_source("TEMPLATE_", priority=20)
            
            config = loader.load_config()
            
            assert isinstance(config, TemplateConfig)
            assert config.name == "MultiSourceTemplate"
            assert config.version == "2.0.0"  # 环境变量覆盖
            
        finally:
            # 清理环境变量
            if "TEMPLATE_VERSION" in os.environ:
                del os.environ["TEMPLATE_VERSION"]
            if "TEMPLATE_DEBUG" in os.environ:
                del os.environ["TEMPLATE_DEBUG"]
    
    def test_cache_operations(self):
        """测试缓存操作"""
        loader = ConfigLoader(cache_enabled=True)
        
        stats = loader.get_cache_stats()
        assert stats["total_entries"] == 0
        
        # 添加一个配置源并加载（会使用缓存）
        config_dict = {"name": "CachedTemplate"}
        loader.add_dict_source(config_dict)
        
        # 第一次加载
        config1 = loader.load_config()
        
        # 第二次加载应该使用缓存
        config2 = loader.load_config()
        
        assert config1.name == config2.name
        
        # 清空缓存
        loader.clear_cache()
        stats = loader.get_cache_stats()
        assert stats["total_entries"] == 0
    
    def test_default_config_loading(self):
        """测试默认配置加载"""
        loader = ConfigLoader()
        
        # 加载不存在的模板配置应该返回默认配置
        config = loader.load_default("NonExistentTemplate")
        
        assert isinstance(config, TemplateConfig)
        assert config.name == "NonExistentTemplate"
        assert "Default configuration" in config.description
    
    @patch('templates.base.config_loader.Observer')
    def test_file_watching(self, mock_observer):
        """测试文件监控功能"""
        loader = ConfigLoader()
        
        # 添加一个文件源并启用监控
        loader.add_file_source("test.yaml", watch=True)
        
        # 启动文件监控
        loader.start_file_watching()
        
        # 验证Observer被调用
        mock_observer.assert_called_once()
        
        # 停止文件监控
        loader.stop_file_watching()
    
    def test_reload_functionality(self):
        """测试重载功能"""
        loader = ConfigLoader(cache_enabled=True)
        
        # 添加配置源
        config_dict = {"name": "ReloadTest", "version": "1.0.0"}
        loader.add_dict_source(config_dict)
        
        # 加载配置
        config = loader.load_config()
        assert config.version == "1.0.0"
        
        # 修改配置数据
        config_dict["version"] = "2.0.0"
        
        # 重载配置
        loader.reload()
        
        # 重新加载应该反映更改
        new_config = loader.load_config()
        assert new_config.version == "2.0.0"


if __name__ == "__main__":
    pytest.main([__file__])