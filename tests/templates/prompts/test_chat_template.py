"""
ChatTemplate 单元测试

测试对话模板的各项功能：
- 基本设置和配置
- 对话历史管理
- 多轮对话功能
- 参数替换和条件分支
- 错误处理
"""

import unittest
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from templates.prompts.chat_template import (
    ChatTemplate, Message, MessageRole, ConversationContext,
    ConversationHistory, ConversationState, ChatResponse
)


class MockLLMTemplate:
    """模拟LLM模板用于测试"""
    
    def execute(self, prompt: str, **kwargs):
        """返回模拟响应"""
        class MockResponse:
            def __init__(self):
                self.content = "这是一个测试响应"
                self.total_tokens = 50
        
        return MockResponse()


class TestChatTemplate(unittest.TestCase):
    """ChatTemplate 测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_llm = MockLLMTemplate()
        self.chat_template = ChatTemplate()
    
    def test_basic_setup(self):
        """测试基本设置"""
        self.chat_template.setup(
            role_name="测试助手",
            role_description="用于测试的AI助手",
            personality="友好、专业",
            llm_template=self.mock_llm
        )
        
        self.assertEqual(self.chat_template.role_name, "测试助手")
        self.assertIsNotNone(self.chat_template.llm_template)
        self.assertIn("测试助手", self.chat_template.role_template)
    
    def test_conversation_creation(self):
        """测试对话创建"""
        self.chat_template.setup(
            role_name="助手",
            llm_template=self.mock_llm
        )
        
        # 创建对话
        history, context = self.chat_template._get_or_create_conversation("test_conv")
        
        self.assertIsInstance(history, ConversationHistory)
        self.assertIsInstance(context, ConversationContext)
        self.assertEqual(context.conversation_id, "test_conv")
    
    def test_message_handling(self):
        """测试消息处理"""
        history = ConversationHistory()
        
        # 添加消息
        message = Message(
            role=MessageRole.USER,
            content="你好"
        )
        history.add_message(message)
        
        self.assertEqual(len(history.messages), 1)
        self.assertEqual(history.messages[0].content, "你好")
    
    def test_conversation_history_limit(self):
        """测试对话历史长度限制"""
        history = ConversationHistory(max_messages=3)
        
        # 添加超过限制的消息
        for i in range(5):
            message = Message(
                role=MessageRole.USER,
                content=f"消息 {i+1}"
            )
            history.add_message(message)
        
        # 应该只保留最近的消息
        self.assertLessEqual(len(history.messages), 3)
    
    def test_context_parameters(self):
        """测试上下文参数"""
        context = ConversationContext(
            conversation_id="test",
            state=ConversationState.ACTIVE
        )
        
        # 设置参数
        context.set_parameter("user_level", "初学者")
        context.set_parameter("topic", "Python")
        
        self.assertEqual(context.get_parameter("user_level"), "初学者")
        self.assertEqual(context.get_parameter("topic"), "Python")
        self.assertIsNone(context.get_parameter("nonexistent"))


class TestMessage(unittest.TestCase):
    """Message 测试类"""
    
    def test_message_creation(self):
        """测试消息创建"""
        message = Message(
            role=MessageRole.USER,
            content="测试消息",
            metadata={"type": "test"}
        )
        
        self.assertEqual(message.role, MessageRole.USER)
        self.assertEqual(message.content, "测试消息")
        self.assertEqual(message.metadata["type"], "test")
        self.assertIsInstance(message.timestamp, datetime)
    
    def test_message_serialization(self):
        """测试消息序列化"""
        message = Message(
            role=MessageRole.ASSISTANT,
            content="助手回复"
        )
        
        # 转换为字典
        message_dict = message.to_dict()
        
        self.assertEqual(message_dict["role"], "assistant")
        self.assertEqual(message_dict["content"], "助手回复")
        self.assertIn("timestamp", message_dict)
        
        # 从字典恢复
        restored_message = Message.from_dict(message_dict)
        self.assertEqual(restored_message.role, MessageRole.ASSISTANT)
        self.assertEqual(restored_message.content, "助手回复")


class TestConversationHistory(unittest.TestCase):
    """ConversationHistory 测试类"""
    
    def test_message_filtering(self):
        """测试消息过滤"""
        history = ConversationHistory()
        
        # 添加不同类型的消息
        user_msg = Message(MessageRole.USER, "用户消息")
        assistant_msg = Message(MessageRole.ASSISTANT, "助手消息")
        system_msg = Message(MessageRole.SYSTEM, "系统消息")
        
        history.add_message(user_msg)
        history.add_message(assistant_msg)
        history.add_message(system_msg)
        
        # 测试角色过滤
        user_messages = history.get_messages(role_filter=MessageRole.USER)
        self.assertEqual(len(user_messages), 1)
        self.assertEqual(user_messages[0].content, "用户消息")
        
        # 测试数量限制
        limited_messages = history.get_messages(limit=2)
        self.assertEqual(len(limited_messages), 2)
    
    def test_context_messages(self):
        """测试上下文消息获取"""
        history = ConversationHistory()
        
        # 添加多条消息
        for i in range(15):
            message = Message(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"消息 {i+1}"
            )
            history.add_message(message)
        
        # 获取上下文消息
        context_messages = history.get_context_messages(context_length=5)
        
        # 应该返回最近的5条消息
        self.assertLessEqual(len(context_messages), 5)


if __name__ == "__main__":
    unittest.main()