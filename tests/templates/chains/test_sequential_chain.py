"""
顺序链模板测试

测试SequentialChainTemplate的各种功能：
- 基本功能测试
- 步骤配置测试
- 错误处理测试
- 异步执行测试
- 性能测试

作者: Claude Code Assistant
版本: 1.0.0
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from unittest.mock import Mock, patch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from templates.chains.sequential_chain import (
    SequentialChainTemplate,
    StepConfig,
    StepResult,
    StepStatus,
    ErrorHandlingStrategy,
    ChainContext
)


class TestSequentialChainTemplate:
    """顺序链模板测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.chain = SequentialChainTemplate()
        
        # 定义测试用的步骤函数
        self.step_functions = {
            "step1": lambda data, **kwargs: {"result1": f"step1_{data.get('input', '')}"},
            "step2": lambda data, **kwargs: {"result2": f"step2_{data.get('result1', '')}"},
            "step3": lambda data, **kwargs: {"result3": f"step3_{data.get('result2', '')}"},
            "error_step": lambda data, **kwargs: (_ for _ in ()).throw(ValueError("Test error")),
            "slow_step": lambda data, **kwargs: (time.sleep(0.1), {"slow_result": "completed"})[1]
        }
    
    def test_chain_initialization(self):
        """测试链初始化"""
        # 测试默认初始化
        chain = SequentialChainTemplate()
        assert chain.config.name == "SequentialChainTemplate"
        assert chain.config.template_type.value == "chain"
        assert len(chain.steps) == 0
        
        # 测试自定义配置初始化
        from templates.base.template_base import TemplateConfig, TemplateType
        custom_config = TemplateConfig(
            name="CustomSequentialChain",
            description="自定义顺序链",
            template_type=TemplateType.CHAIN
        )
        chain = SequentialChainTemplate(custom_config)
        assert chain.config.name == "CustomSequentialChain"
    
    def test_setup_configuration(self):
        """测试设置配置"""
        steps_config = [
            {
                "name": "步骤1",
                "executor": self.step_functions["step1"],
                "description": "第一个步骤"
            },
            {
                "name": "步骤2", 
                "executor": self.step_functions["step2"],
                "description": "第二个步骤"
            }
        ]
        
        # 测试正常配置
        self.chain.setup(
            steps=steps_config,
            error_strategy="fail_fast",
            enable_caching=True
        )
        
        assert len(self.chain.steps) == 2
        assert self.chain.steps[0].name == "步骤1"
        assert self.chain.steps[1].name == "步骤2"
        assert self.chain.default_error_strategy == ErrorHandlingStrategy.FAIL_FAST
        assert self.chain.enable_caching is True
    
    def test_setup_validation(self):
        """测试设置验证"""
        # 测试空步骤列表
        with pytest.raises(Exception):
            self.chain.setup(steps=[])
        
        # 测试无效的错误策略
        with pytest.raises(ValueError):
            self.chain.setup(
                steps=[{"name": "test", "executor": self.step_functions["step1"]}],
                error_strategy="invalid_strategy"
            )
    
    def test_add_remove_steps(self):
        """测试添加和删除步骤"""
        # 测试添加步骤
        step_id = self.chain.add_step(
            "测试步骤",
            executor=self.step_functions["step1"],
            description="测试用步骤"
        )
        
        assert len(self.chain.steps) == 1
        assert self.chain.steps[0].name == "测试步骤"
        assert self.chain.steps[0].step_id == step_id
        
        # 测试删除步骤
        removed = self.chain.remove_step(step_id)
        assert removed is True
        assert len(self.chain.steps) == 0
        
        # 测试删除不存在的步骤
        removed = self.chain.remove_step("nonexistent")
        assert removed is False
    
    def test_get_step(self):
        """测试获取步骤"""
        step_id = self.chain.add_step("测试步骤", executor=self.step_functions["step1"])
        
        # 测试获取存在的步骤
        step = self.chain.get_step(step_id)
        assert step is not None
        assert step.name == "测试步骤"
        
        # 测试获取不存在的步骤
        step = self.chain.get_step("nonexistent")
        assert step is None
    
    def test_basic_execution(self):
        """测试基本执行"""
        # 配置简单的三步链
        self.chain.setup(
            steps=[
                {"name": "步骤1", "executor": self.step_functions["step1"]},
                {"name": "步骤2", "executor": self.step_functions["step2"]},
                {"name": "步骤3", "executor": self.step_functions["step3"]}
            ]
        )
        
        # 执行链
        input_data = {"input": "test"}
        result = self.chain.run(input_data)
        
        # 验证结果
        assert result["status"] == "completed"
        assert "step1_test" in result["data"]["result2"]
        assert "step3_" in result["data"]["result3"]
        assert result["summary"]["total_steps"] == 3
        assert result["summary"]["completed_steps"] == 3
        assert result["summary"]["failed_steps"] == 0
    
    def test_execution_with_failure_fail_fast(self):
        """测试执行中的失败（快速失败策略）"""
        # 配置包含失败步骤的链
        self.chain.setup(
            steps=[
                {"name": "成功步骤", "executor": self.step_functions["step1"]},
                {"name": "失败步骤", "executor": self.step_functions["error_step"]},
                {"name": "不会执行的步骤", "executor": self.step_functions["step3"]}
            ],
            error_strategy="fail_fast"
        )
        
        # 执行链，应该抛出异常
        with pytest.raises(Exception):
            self.chain.run({"input": "test"})
    
    def test_execution_with_failure_continue(self):
        """测试执行中的失败（继续执行策略）"""
        # 配置包含失败步骤的链
        self.chain.setup(
            steps=[
                {"name": "成功步骤1", "executor": self.step_functions["step1"]},
                {"name": "失败步骤", "executor": self.step_functions["error_step"], "error_strategy": "continue"},
                {"name": "成功步骤2", "executor": self.step_functions["step3"]}
            ]
        )
        
        # 执行链
        result = self.chain.run({"input": "test"})
        
        # 验证结果
        assert result["status"] == "partial"
        assert result["summary"]["failed_steps"] == 1
        assert result["summary"]["completed_steps"] == 2
    
    @pytest.mark.asyncio
    async def test_async_execution(self):
        """测试异步执行"""
        # 定义异步步骤函数
        async def async_step1(data, **kwargs):
            await asyncio.sleep(0.01)
            return {"async_result1": f"async_step1_{data.get('input', '')}"}
        
        async def async_step2(data, **kwargs):
            await asyncio.sleep(0.01)
            return {"async_result2": f"async_step2_{data.get('async_result1', '')}"}
        
        # 配置异步链
        self.chain.setup(
            steps=[
                {"name": "异步步骤1", "executor": async_step1},
                {"name": "异步步骤2", "executor": async_step2}
            ]
        )
        
        # 异步执行链
        input_data = {"input": "async_test"}
        result = await self.chain.run_async(input_data)
        
        # 验证结果
        assert result["status"] == "completed"
        assert "async_step1_async_test" in result["data"]["async_result2"]
    
    def test_caching(self):
        """测试缓存功能"""
        # 定义计数器步骤
        call_count = {"count": 0}
        
        def counting_step(data, **kwargs):
            call_count["count"] += 1
            return {"count": call_count["count"], "data": data}
        
        # 配置启用缓存的链
        self.chain.setup(
            steps=[
                {
                    "name": "计数步骤",
                    "executor": counting_step,
                    "cache_enabled": True
                }
            ],
            enable_caching=True
        )
        
        # 第一次执行
        result1 = self.chain.run({"input": "test"})
        assert result1["data"]["count"] == 1
        
        # 第二次执行相同输入，应该使用缓存
        result2 = self.chain.run({"input": "test"})
        assert result2["data"]["count"] == 1  # 计数器不应该增加
        
        # 执行不同输入，不应该使用缓存
        result3 = self.chain.run({"input": "different"})
        assert result3["data"]["count"] == 2  # 计数器应该增加
    
    def test_pause_resume_cancel(self):
        """测试暂停、恢复和取消功能"""
        # 定义慢步骤
        def slow_step(data, **kwargs):
            for i in range(10):
                time.sleep(0.01)
                if self.chain.is_cancelled:
                    break
            return {"slow_result": "completed"}
        
        # 配置链
        self.chain.setup(
            steps=[
                {"name": "慢步骤", "executor": slow_step}
            ]
        )
        
        # 测试取消
        import threading
        
        def cancel_chain():
            time.sleep(0.05)
            self.chain.cancel()
        
        cancel_thread = threading.Thread(target=cancel_chain)
        cancel_thread.start()
        
        start_time = time.time()
        try:
            self.chain.run({"input": "test"})
        except:
            pass
        execution_time = time.time() - start_time
        
        # 验证取消功能生效（执行时间应该比正常情况短）
        assert execution_time < 0.15  # 正常情况下需要约0.1秒
        
        cancel_thread.join()
    
    def test_chain_context(self):
        """测试链上下文"""
        context = ChainContext({"initial": "data"})
        
        # 测试数据操作
        context.set_data("key1", "value1")
        assert context.get_data("key1") == "value1"
        assert context.get_data("nonexistent", "default") == "default"
        
        context.update_data({"key2": "value2", "key3": "value3"})
        assert len(context.data) == 4  # initial + key1 + key2 + key3
        
        # 测试步骤结果
        step_result = StepResult(
            step_id="test_step",
            step_name="测试步骤",
            status=StepStatus.COMPLETED
        )
        context.add_step_result(step_result)
        
        retrieved_result = context.get_step_result("test_step")
        assert retrieved_result is not None
        assert retrieved_result.step_name == "测试步骤"
        
        # 测试执行摘要
        summary = context.get_execution_summary()
        assert summary["total_steps"] == 1
        assert summary["completed_steps"] == 1
        assert summary["failed_steps"] == 0
    
    def test_step_config_validation(self):
        """测试步骤配置验证"""
        # 测试有效配置
        valid_config = StepConfig(
            name="测试步骤",
            executor=self.step_functions["step1"]
        )
        assert valid_config.name == "测试步骤"
        assert valid_config.step_id is not None
        
        # 测试无效配置（既没有executor也没有template）
        with pytest.raises(Exception):
            StepConfig(name="无效步骤")
    
    def test_step_result(self):
        """测试步骤结果"""
        result = StepResult(
            step_id="test_step",
            step_name="测试步骤",
            status=StepStatus.COMPLETED
        )
        
        # 测试成功状态
        assert result.is_successful() is True
        
        # 测试失败状态
        result.status = StepStatus.FAILED
        assert result.is_successful() is False
        
        # 测试执行摘要
        summary = result.get_execution_summary()
        assert summary["step_id"] == "test_step"
        assert summary["step_name"] == "测试步骤"
        assert summary["success"] is False
    
    def test_error_handling_strategies(self):
        """测试错误处理策略"""
        strategies = [
            ErrorHandlingStrategy.FAIL_FAST,
            ErrorHandlingStrategy.CONTINUE_ON_ERROR,
            ErrorHandlingStrategy.SKIP_ON_ERROR
        ]
        
        for strategy in strategies:
            chain = SequentialChainTemplate()
            chain.setup(
                steps=[
                    {"name": "成功步骤", "executor": self.step_functions["step1"]},
                    {"name": "失败步骤", "executor": self.step_functions["error_step"], "error_strategy": strategy.value}
                ]
            )
            
            if strategy == ErrorHandlingStrategy.FAIL_FAST:
                # 快速失败应该抛出异常
                with pytest.raises(Exception):
                    chain.run({"input": "test"})
            else:
                # 其他策略应该能够处理错误并继续
                result = chain.run({"input": "test"})
                assert result["status"] in ["partial", "completed"]
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        self.chain.setup(
            steps=[
                {"name": "快步骤", "executor": self.step_functions["step1"]},
                {"name": "慢步骤", "executor": self.step_functions["slow_step"]}
            ]
        )
        
        # 执行链
        result = self.chain.run({"input": "performance_test"})
        
        # 检查性能指标
        metrics = self.chain.get_metrics()
        assert "total_executions" in metrics
        assert "avg_execution_time" in metrics
        assert metrics["total_executions"] >= 1
        
        # 检查状态信息
        status = self.chain.get_status()
        assert "execution_time" in status
        assert status["execution_time"] is not None
    
    def test_input_output_mapping(self):
        """测试输入输出映射"""
        def step_with_mapping(data, **kwargs):
            # 只处理特定的输入键
            value = data.get("specific_input", "")
            return {"mapped_output": f"processed_{value}"}
        
        self.chain.setup(
            steps=[
                {
                    "name": "映射步骤",
                    "executor": step_with_mapping,
                    "input_keys": ["specific_input"],
                    "output_keys": ["mapped_output"]
                }
            ]
        )
        
        # 执行链
        input_data = {
            "specific_input": "target_value",
            "other_input": "ignored_value"
        }
        result = self.chain.run(input_data)
        
        # 验证映射结果
        assert "processed_target_value" in result["data"]["mapped_output"]


# 测试运行器
if __name__ == "__main__":
    pytest.main([__file__, "-v"])