"""
配置管理模块的单元测试

测试ConfigLoader、各种ConfigSource和配置验证功能。
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_learning.core.config import (
    ConfigLoader,
    EnvironmentConfigSource,
    FileConfigSource,
    ArgumentConfigSource,
    AppConfig,
    ConfigFormat,
    setup_config,
    get_config,
    get_config_value
)
from langchain_learning.core.exceptions import ConfigurationError, ValidationError


class TestEnvironmentConfigSource:
    """测试环境变量配置源"""
    
    def test_load_basic(self):
        """测试基本的环境变量加载"""
        with patch.dict('os.environ', {
            'LANGCHAIN_TEST_KEY': 'test_value',
            'LANGCHAIN_DEBUG': 'true',
            'LANGCHAIN_PORT': '8080',
            'OTHER_KEY': 'should_be_ignored'
        }):
            source = EnvironmentConfigSource(prefix="LANGCHAIN_")
            config = source.load()
            
            assert config['test_key'] == 'test_value'
            assert config['debug'] is True
            assert config['port'] == 8080
            assert 'other_key' not in config
    
    def test_value_conversion(self):
        """测试值类型转换"""
        with patch.dict('os.environ', {
            'LANGCHAIN_BOOL_TRUE': 'true',
            'LANGCHAIN_BOOL_FALSE': 'false',
            'LANGCHAIN_INT': '123',
            'LANGCHAIN_FLOAT': '12.34',
            'LANGCHAIN_JSON': '{"key": "value"}',
            'LANGCHAIN_LIST': 'item1,item2,item3',
            'LANGCHAIN_STRING': 'simple_string'
        }):
            source = EnvironmentConfigSource(prefix="LANGCHAIN_")
            config = source.load()
            
            assert config['bool_true'] is True
            assert config['bool_false'] is False
            assert config['int'] == 123
            assert config['float'] == 12.34
            assert config['json'] == {"key": "value"}
            assert config['list'] == ['item1', 'item2', 'item3']
            assert config['string'] == 'simple_string'
    
    def test_is_available(self):
        """测试可用性检查"""
        source = EnvironmentConfigSource()
        assert source.is_available() is True


class TestFileConfigSource:
    """测试文件配置源"""
    
    def test_json_file(self):
        """测试JSON文件加载"""
        config_data = {
            "app_name": "test_app",
            "debug": True,
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            source = FileConfigSource(temp_path, format=ConfigFormat.JSON)
            assert source.is_available() is True
            
            loaded_config = source.load()
            assert loaded_config == config_data
        finally:
            Path(temp_path).unlink()
    
    def test_yaml_file(self):
        """测试YAML文件加载"""
        config_data = {
            "app_name": "test_app",
            "debug": True,
            "features": ["feature1", "feature2"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            source = FileConfigSource(temp_path, format=ConfigFormat.YAML)
            loaded_config = source.load()
            assert loaded_config == config_data
        finally:
            Path(temp_path).unlink()
    
    def test_env_file(self):
        """测试.env文件加载"""
        env_content = """
# 这是注释
APP_NAME=test_app
DEBUG=true
DATABASE_URL="postgresql://localhost:5432/test"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            temp_path = f.name
        
        try:
            source = FileConfigSource(temp_path, format=ConfigFormat.ENV)
            loaded_config = source.load()
            
            assert loaded_config['APP_NAME'] == 'test_app'
            assert loaded_config['DEBUG'] == 'true'
            assert loaded_config['DATABASE_URL'] == 'postgresql://localhost:5432/test'
        finally:
            Path(temp_path).unlink()
    
    def test_file_not_found(self):
        """测试文件不存在的情况"""
        source = FileConfigSource("/nonexistent/file.json", required=False)
        assert source.is_available() is False
        assert source.load() == {}
        
        # 测试必需文件不存在时抛出异常
        source_required = FileConfigSource("/nonexistent/file.json", required=True)
        with pytest.raises(ConfigurationError):
            source_required.load()
    
    def test_format_detection(self):
        """测试格式自动检测"""
        # 测试不同扩展名的格式检测
        json_source = FileConfigSource("test.json")
        assert json_source.format == ConfigFormat.JSON
        
        yaml_source = FileConfigSource("test.yaml")
        assert yaml_source.format == ConfigFormat.YAML
        
        yml_source = FileConfigSource("test.yml")
        assert yml_source.format == ConfigFormat.YAML
        
        env_source = FileConfigSource("test.env")
        assert env_source.format == ConfigFormat.ENV


class TestArgumentConfigSource:
    """测试命令行参数配置源"""
    
    def test_basic_parsing(self):
        """测试基本参数解析"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--port', type=int, default=8000)
        
        args = ['--config', 'test.yaml', '--debug', '--port', '9000']
        source = ArgumentConfigSource(parser, args)
        
        config = source.load()
        assert config['config'] == 'test.yaml'
        assert config['debug'] is True
        assert config['port'] == 9000
    
    def test_default_parser(self):
        """测试默认解析器"""
        args = ['--log-level', 'DEBUG', '--env', 'development']
        source = ArgumentConfigSource(args=args)
        
        config = source.load()
        assert config['log_level'] == 'DEBUG'
        assert config['env'] == 'development'


class TestConfigLoader:
    """测试配置加载器"""
    
    def test_single_source(self):
        """测试单个配置源"""
        loader = ConfigLoader(AppConfig)
        
        # 添加环境变量源
        with patch.dict('os.environ', {
            'LANGCHAIN_APP_NAME': 'test_app',
            'LANGCHAIN_DEBUG': 'true'
        }):
            loader.add_env_source()
            config = loader.load()
            
            assert config.app_name == 'test_app'
            assert config.debug is True
    
    def test_multiple_sources_priority(self):
        """测试多个配置源的优先级"""
        loader = ConfigLoader(AppConfig)
        
        # 创建临时配置文件
        config_data = {
            "app_name": "file_app",
            "debug": False,
            "api_port": 8080
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # 添加文件源（优先级50）
            loader.add_file_source(temp_path, priority=50)
            
            # 添加环境变量源（优先级100，更高）
            with patch.dict('os.environ', {
                'LANGCHAIN_APP_NAME': 'env_app',  # 应该覆盖文件中的值
                'LANGCHAIN_API_HOST': 'env_host'  # 文件中没有，应该被添加
            }):
                loader.add_env_source(priority=100)
                config = loader.load()
                
                # 环境变量应该覆盖文件配置
                assert config.app_name == 'env_app'
                assert config.api_host == 'env_host'
                # 文件中的值应该保留（环境变量中没有）
                assert config.debug is False
                assert config.api_port == 8080
        finally:
            Path(temp_path).unlink()
    
    def test_config_validation(self):
        """测试配置验证"""
        loader = ConfigLoader(AppConfig)
        
        # 测试无效的环境值
        with patch.dict('os.environ', {
            'LANGCHAIN_ENVIRONMENT': 'invalid_env'  # 不在允许的值中
        }):
            loader.add_env_source()
            
            with pytest.raises(ValidationError):
                loader.load()
    
    def test_get_method(self):
        """测试get方法"""
        loader = ConfigLoader(AppConfig)
        
        with patch.dict('os.environ', {
            'LANGCHAIN_APP_NAME': 'test_app',
            'LANGCHAIN_API_PORT': '9000'
        }):
            loader.add_env_source()
            config = loader.load()
            
            # 测试获取存在的值
            assert loader.get('app_name') == 'test_app'
            assert loader.get('api_port') == 9000
            
            # 测试获取不存在的值
            assert loader.get('nonexistent_key') is None
            assert loader.get('nonexistent_key', 'default') == 'default'
    
    def test_nested_config_merge(self):
        """测试嵌套配置合并"""
        loader = ConfigLoader()
        
        # 模拟两个配置源
        class MockSource1:
            priority = 50
            def is_available(self):
                return True
            def load(self):
                return {
                    "database": {
                        "host": "localhost",
                        "port": 5432
                    },
                    "redis": {
                        "host": "redis1"
                    }
                }
        
        class MockSource2:
            priority = 100
            def is_available(self):
                return True
            def load(self):
                return {
                    "database": {
                        "port": 5433,  # 覆盖端口
                        "name": "testdb"  # 添加新字段
                    },
                    "redis": {
                        "port": 6379  # 添加新字段
                    }
                }
        
        loader.add_source(MockSource1())
        loader.add_source(MockSource2())
        
        config = loader.load()
        
        # 验证嵌套合并
        assert config.database["host"] == "localhost"  # 来自source1
        assert config.database["port"] == 5433  # 被source2覆盖
        assert config.database["name"] == "testdb"  # 来自source2
        assert config.redis["host"] == "redis1"  # 来自source1
        assert config.redis["port"] == 6379  # 来自source2


class TestGlobalConfigFunctions:
    """测试全局配置函数"""
    
    def test_setup_and_get_config(self):
        """测试全局配置设置和获取"""
        with patch.dict('os.environ', {
            'LANGCHAIN_APP_NAME': 'global_test_app'
        }):
            # 设置全局配置
            loader = setup_config(
                schema_class=AppConfig,
                include_args=False
            )
            
            # 获取配置
            config = get_config()
            assert config.app_name == 'global_test_app'
            
            # 测试配置值获取
            assert get_config_value('app_name') == 'global_test_app'
            assert get_config_value('nonexistent', 'default') == 'default'
    
    def test_config_file_loading(self):
        """测试从配置文件加载"""
        config_data = {
            "app_name": "file_test_app",
            "version": "2.0.0"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = setup_config(
                config_files=[temp_path],
                include_args=False
            )
            
            config = get_config()
            assert config.app_name == 'file_test_app'
            assert config.version == '2.0.0'
        finally:
            Path(temp_path).unlink()


class TestAppConfig:
    """测试应用配置模型"""
    
    def test_default_values(self):
        """测试默认值"""
        config = AppConfig()
        
        assert config.app_name == "langchain-learning"
        assert config.version == "0.1.0"
        assert config.environment == "development"
        assert config.debug is False
        assert config.log_level == "INFO"
    
    def test_validation(self):
        """测试配置验证"""
        # 测试有效的环境值
        config = AppConfig(environment="production")
        assert config.environment == "production"
        
        # 测试无效的环境值
        with pytest.raises(ValidationError):
            AppConfig(environment="invalid")
        
        # 测试端口范围验证
        config = AppConfig(api_port=8080)
        assert config.api_port == 8080
        
        with pytest.raises(ValidationError):
            AppConfig(api_port=0)
        
        with pytest.raises(ValidationError):
            AppConfig(api_port=99999)
    
    def test_field_updates(self):
        """测试字段更新"""
        config = AppConfig(
            app_name="custom_app",
            debug=True,
            api_port=9000,
            llm_temperature=0.8
        )
        
        assert config.app_name == "custom_app"
        assert config.debug is True
        assert config.api_port == 9000
        assert config.llm_temperature == 0.8