"""
多模型模板测试用例

本模块提供了多模型模板的核心测试覆盖，包括：
- 模型池管理测试
- 路由策略测试
- 模型对比测试
- 故障转移测试
- 统计信息测试
- Mock测试（避免实际API调用）

测试重点：
1. 验证多模型管理功能
2. 测试智能路由逻辑
3. 确保故障转移机制
4. 验证性能统计准确性
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# 导入被测试的模块
from templates.llm.multi_model_template import (
    MultiModelTemplate, 
    ModelResponse, 
    ComparisonResult,
    RoutingStrategy
)
from templates.llm.openai_template import OpenAITemplate
from templates.llm.anthropic_template import AnthropicTemplate
from src.langchain_learning.core.exceptions import ValidationError, ConfigurationError, APIError


class TestMultiModelTemplate:
    """多模型模板测试类"""
    
    @pytest.fixture
    def template(self):
        """创建测试用的多模型模板实例"""
        return MultiModelTemplate()
    
    @pytest.fixture
    def mock_openai_template(self):
        """创建模拟的OpenAI模板"""
        mock_template = Mock(spec=OpenAITemplate)
        mock_template.setup = Mock()
        mock_template.execute = Mock()
        mock_template.execute_async = Mock()
        return mock_template
    
    @pytest.fixture
    def mock_anthropic_template(self):
        """创建模拟的Anthropic模板"""
        mock_template = Mock(spec=AnthropicTemplate)
        mock_template.setup = Mock()
        mock_template.execute = Mock()
        mock_template.execute_async = Mock()
        return mock_template
    
    @pytest.fixture
    def sample_model_response(self):
        """创建示例模型响应"""
        return ModelResponse(
            model_name="test-model",
            content="这是测试响应",
            response_time=1.5,
            tokens_used=100,
            estimated_cost=0.001,
            success=True
        )
    
    def test_template_initialization(self, template):
        """测试模板初始化"""
        assert template is not None
        assert template.config.name == "MultiModelTemplate"
        assert template.config.template_type.value == "llm"
        assert len(template.models) == 0
        assert template.routing_strategy == RoutingStrategy.SMART
    
    def test_setup_with_valid_parameters(self, template):
        """测试使用有效参数进行设置"""
        template.setup(
            routing_strategy="cost_optimized",
            fallback_models=["model1", "model2"],
            max_parallel_requests=5
        )
        
        assert template.routing_strategy == RoutingStrategy.COST_OPTIMIZED
        assert template.fallback_models == ["model1", "model2"]
        assert template.max_parallel_requests == 5
    
    def test_setup_with_invalid_strategy(self, template):
        """测试使用无效路由策略"""
        with pytest.raises(ValidationError):
            template.setup(routing_strategy="invalid_strategy")
    
    def test_add_model(self, template, mock_openai_template):
        """测试添加模型到模型池"""
        setup_params = {
            "api_key": "test-key",
            "model_name": "gpt-3.5-turbo"
        }
        
        template.add_model(
            name="gpt-3.5",
            template=mock_openai_template,
            setup_params=setup_params,
            priority=2,
            cost_per_1k_tokens=0.002
        )
        
        # 验证模型已添加
        assert "gpt-3.5" in template.models
        assert template.models["gpt-3.5"].priority == 2
        assert template.models["gpt-3.5"].cost_per_1k_tokens == 0.002
        
        # 验证模板被设置
        mock_openai_template.setup.assert_called_once_with(**setup_params)
        
        # 验证统计信息初始化
        assert "gpt-3.5" in template.model_stats
        assert template.model_stats["gpt-3.5"]["requests"] == 0
    
    def test_remove_model(self, template, mock_openai_template):
        """测试从模型池中移除模型"""
        # 先添加模型
        template.add_model(
            name="test-model",
            template=mock_openai_template,
            setup_params={"api_key": "test"}
        )
        
        assert "test-model" in template.models
        
        # 移除模型
        template.remove_model("test-model")
        
        assert "test-model" not in template.models
        assert "test-model" not in template.model_stats
    
    def test_execute_without_models(self, template):
        """测试没有模型时执行调用"""
        with pytest.raises(RuntimeError, match="No models configured"):
            template.execute("测试输入")
    
    def test_execute_with_single_model(self, template, mock_openai_template, sample_model_response):
        """测试单模型执行"""
        # 设置模拟响应
        from templates.llm.openai_template import OpenAIResponse
        mock_openai_response = OpenAIResponse(
            content="测试响应",
            model="gpt-3.5-turbo",
            usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
            finish_reason="stop",
            response_time=1.0,
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50,
            estimated_cost=0.001
        )
        mock_openai_template.execute.return_value = mock_openai_response
        
        # 添加模型
        template.add_model(
            name="gpt-3.5",
            template=mock_openai_template,
            setup_params={"api_key": "test"}
        )
        
        # 执行调用
        result = template.execute("测试输入")
        
        # 验证结果
        assert isinstance(result, ModelResponse)
        assert result.model_name == "gpt-3.5"
        assert result.content == "测试响应"
        assert result.success is True
        
        # 验证统计信息更新
        assert template.model_stats["gpt-3.5"]["requests"] == 1
        assert template.model_stats["gpt-3.5"]["successes"] == 1
    
    def test_routing_strategies(self, template, mock_openai_template, mock_anthropic_template):
        """测试不同的路由策略"""
        # 添加两个模型
        template.add_model("fast-model", mock_openai_template, {"api_key": "test"}, priority=1, cost_per_1k_tokens=0.001)
        template.add_model("quality-model", mock_anthropic_template, {"api_key": "test"}, priority=3, cost_per_1k_tokens=0.01)
        
        # 测试成本优化路由
        template.routing_strategy = RoutingStrategy.COST_OPTIMIZED
        selected = template._route_request("测试", {})
        assert selected == "fast-model"  # 应该选择更便宜的模型
        
        # 测试质量优化路由
        template.routing_strategy = RoutingStrategy.QUALITY
        selected = template._route_request("测试", {})
        assert selected == "quality-model"  # 应该选择高优先级模型
        
        # 测试轮询路由
        template.routing_strategy = RoutingStrategy.ROUND_ROBIN
        selected1 = template._route_request("测试", {})
        selected2 = template._route_request("测试", {})
        assert selected1 != selected2  # 应该轮换选择
    
    def test_preferred_model_routing(self, template, mock_openai_template, mock_anthropic_template):
        """测试指定偏好模型的路由"""
        template.add_model("model1", mock_openai_template, {"api_key": "test"})
        template.add_model("model2", mock_anthropic_template, {"api_key": "test"})
        
        # 指定偏好模型
        selected = template._route_request("测试", {"preferred_model": "model2"})
        assert selected == "model2"
    
    def test_compare_models(self, template, mock_openai_template, mock_anthropic_template):
        """测试模型对比功能"""
        # 设置模拟响应
        from templates.llm.openai_template import OpenAIResponse
        from templates.llm.anthropic_template import AnthropicResponse
        
        openai_response = OpenAIResponse(
            content="OpenAI响应", model="gpt-3.5-turbo", usage={}, finish_reason="stop",
            response_time=1.0, total_tokens=100, prompt_tokens=50, completion_tokens=50, estimated_cost=0.002
        )
        
        anthropic_response = AnthropicResponse(
            content="Anthropic响应", model="claude-3-sonnet", usage={}, stop_reason="stop",
            response_time=2.0, input_tokens=50, output_tokens=50, estimated_cost=0.001
        )
        
        mock_openai_template.execute.return_value = openai_response
        mock_anthropic_template.execute.return_value = anthropic_response
        
        # 添加模型
        template.add_model("openai", mock_openai_template, {"api_key": "test"})
        template.add_model("anthropic", mock_anthropic_template, {"api_key": "test"})
        
        # 执行对比
        comparison = template.compare_models("测试问题")
        
        # 验证对比结果
        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.responses) == 2
        assert comparison.fastest_model == "openai"  # 响应时间更短
        assert comparison.cheapest_model == "anthropic"  # 成本更低
    
    def test_fallback_mechanism(self, template, mock_openai_template, mock_anthropic_template):
        """测试故障转移机制"""
        # 设置一个模型失败，另一个成功
        mock_openai_template.execute.side_effect = APIError("API错误")
        
        from templates.llm.anthropic_template import AnthropicResponse
        success_response = AnthropicResponse(
            content="备用响应", model="claude-3-sonnet", usage={}, stop_reason="stop",
            response_time=1.0, input_tokens=50, output_tokens=50, estimated_cost=0.001
        )
        mock_anthropic_template.execute.return_value = success_response
        
        # 添加模型
        template.add_model("primary", mock_openai_template, {"api_key": "test"})
        template.add_model("fallback", mock_anthropic_template, {"api_key": "test"})
        
        # 设置备用模型
        template.fallback_models = ["fallback"]
        
        # 模拟主模型路由到失败的模型
        with patch.object(template, '_route_request', return_value="primary"):
            result = template.execute("测试")
        
        # 应该使用备用模型的响应
        assert result.model_name == "fallback"
        assert result.content == "备用响应"
    
    def test_model_health_tracking(self, template, mock_openai_template):
        """测试模型健康状态跟踪"""
        template.add_model("test-model", mock_openai_template, {"api_key": "test"})
        
        # 初始状态应该是健康的
        assert template._is_model_healthy("test-model") is True
        
        # 模拟一些失败的请求
        for _ in range(10):
            error_response = ModelResponse.from_error("test-model", Exception("错误"))
            template._update_model_stats("test-model", error_response)
        
        # 健康分数应该下降
        assert template.model_stats["test-model"]["health_score"] < 0.5
        assert template._is_model_healthy("test-model") is False
    
    def test_statistics_tracking(self, template, mock_openai_template):
        """测试统计信息跟踪"""
        template.add_model("test-model", mock_openai_template, {"api_key": "test"})
        
        # 模拟一些成功的请求
        for i in range(5):
            response = ModelResponse(
                model_name="test-model",
                content=f"响应{i}",
                response_time=1.0 + i * 0.1,
                tokens_used=100,
                estimated_cost=0.001,
                success=True
            )
            template._update_model_stats("test-model", response)
        
        # 检查统计信息
        stats = template.model_stats["test-model"]
        assert stats["requests"] == 5
        assert stats["successes"] == 5
        assert stats["failures"] == 0
        assert stats["average_response_time"] == 1.2  # (1.0+1.1+1.2+1.3+1.4)/5
        assert stats["health_score"] > 0.9
    
    def test_global_statistics(self, template, mock_openai_template):
        """测试全局统计信息"""
        template.add_model("test-model", mock_openai_template, {"api_key": "test"})
        
        # 模拟一些请求
        template.total_requests = 10
        template.successful_requests = 8
        template.total_cost = 0.05
        template.total_response_time = 15.0
        
        stats = template.get_global_statistics()
        
        assert stats["total_requests"] == 10
        assert stats["successful_requests"] == 8
        assert stats["success_rate"] == 0.8
        assert stats["total_cost"] == 0.05
        assert stats["average_response_time"] == 1.875  # 15.0/8
    
    def test_get_example(self, template):
        """测试获取使用示例"""
        example = template.get_example()
        
        assert "description" in example
        assert "setup_parameters" in example
        assert "model_configuration" in example
        assert "usage_code" in example
        
        # 验证示例包含必要信息
        assert len(example["model_configuration"]) >= 2  # 至少两个模型示例
    
    def test_model_response_creation(self):
        """测试ModelResponse创建方法"""
        # 测试从OpenAI响应创建
        from templates.llm.openai_template import OpenAIResponse
        openai_resp = OpenAIResponse(
            content="测试", model="gpt-3.5-turbo", usage={}, finish_reason="stop",
            response_time=1.0, total_tokens=100, prompt_tokens=50, completion_tokens=50, estimated_cost=0.001
        )
        
        model_resp = ModelResponse.from_openai_response("test-model", openai_resp)
        assert model_resp.model_name == "test-model"
        assert model_resp.content == "测试"
        assert model_resp.tokens_used == 100
        
        # 测试从错误创建
        error_resp = ModelResponse.from_error("test-model", Exception("测试错误"))
        assert error_resp.model_name == "test-model"
        assert error_resp.success is False
        assert error_resp.error_message == "测试错误"
    
    def test_quality_evaluation(self, template):
        """测试响应质量评估"""
        responses = [
            ModelResponse("model1", "短响应", 1.0, 50, 0.001),
            ModelResponse("model2", "这是一个中等长度的响应，包含更多信息和细节", 2.0, 200, 0.002),
            ModelResponse("model3", "非常详细的响应" * 20, 3.0, 500, 0.005)  # 很长的响应
        ]
        
        # 添加模型以便获取优先级
        for i, resp in enumerate(responses):
            mock_template = Mock()
            template.add_model(resp.model_name, mock_template, {"api_key": "test"}, priority=i+1)
        
        best_model = template._evaluate_quality(responses)
        
        # 应该选择中等长度且有合理优先级的模型
        assert best_model in ["model1", "model2", "model3"]


# 异步测试
class TestMultiModelTemplateAsync:
    """多模型模板异步功能测试"""
    
    @pytest.mark.asyncio
    async def test_execute_async(self):
        """测试异步执行"""
        template = MultiModelTemplate()
        
        # 创建异步模拟模板
        mock_template = Mock()
        
        from templates.llm.openai_template import OpenAIResponse
        mock_response = OpenAIResponse(
            content="异步响应", model="gpt-3.5-turbo", usage={}, finish_reason="stop",
            response_time=1.0, total_tokens=100, prompt_tokens=50, completion_tokens=50, estimated_cost=0.001
        )
        
        # 设置异步方法
        async def mock_execute_async(*args, **kwargs):
            return mock_response
        
        mock_template.execute_async = mock_execute_async
        mock_template.setup = Mock()
        
        template.add_model("async-model", mock_template, {"api_key": "test"})
        
        result = await template.execute_async("测试输入")
        
        assert isinstance(result, ModelResponse)
        assert result.content == "异步响应"
    
    @pytest.mark.asyncio
    async def test_compare_models_async(self):
        """测试异步模型对比"""
        template = MultiModelTemplate()
        
        # 创建两个异步模拟模板
        mock_template1 = Mock()
        mock_template2 = Mock()
        
        from templates.llm.openai_template import OpenAIResponse
        resp1 = OpenAIResponse(
            content="响应1", model="model1", usage={}, finish_reason="stop",
            response_time=1.0, total_tokens=100, prompt_tokens=50, completion_tokens=50, estimated_cost=0.001
        )
        resp2 = OpenAIResponse(
            content="响应2", model="model2", usage={}, finish_reason="stop",
            response_time=2.0, total_tokens=150, prompt_tokens=75, completion_tokens=75, estimated_cost=0.002
        )
        
        async def mock_execute1(*args, **kwargs):
            return resp1
        async def mock_execute2(*args, **kwargs):
            return resp2
        
        mock_template1.execute_async = mock_execute1
        mock_template1.setup = Mock()
        mock_template2.execute_async = mock_execute2
        mock_template2.setup = Mock()
        
        template.add_model("model1", mock_template1, {"api_key": "test"})
        template.add_model("model2", mock_template2, {"api_key": "test"})
        
        comparison = await template.compare_models_async("测试问题")
        
        assert isinstance(comparison, ComparisonResult)
        assert len(comparison.responses) == 2
        assert comparison.fastest_model == "model1"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])