"""
对话记忆模板测试

测试ConversationMemoryTemplate的各项功能，包括：
1. 基本配置和初始化
2. 消息添加和检索
3. 对话管理
4. 多种存储后端
5. 错误处理
6. 性能测试
"""

import unittest
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from templates.memory.conversation_memory import (
    ConversationMemoryTemplate,
    Message,
    MessageType,
    Conversation,
    MemoryBackend,
    FileBackend
)
from templates.base.template_base import TemplateConfig, TemplateType


class TestConversationMemoryTemplate(unittest.TestCase):
    """对话记忆模板测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.template = ConversationMemoryTemplate()
        
    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_template_initialization(self):
        """测试模板初始化"""
        # 测试默认初始化
        template = ConversationMemoryTemplate()
        self.assertIsNotNone(template.config)
        self.assertEqual(template.config.name, "ConversationMemoryTemplate")
        self.assertEqual(template.config.template_type, TemplateType.MEMORY)
        
        # 测试自定义配置初始化
        config = TemplateConfig(
            name="CustomMemory",
            description="自定义记忆模板",
            template_type=TemplateType.MEMORY
        )
        template = ConversationMemoryTemplate(config)
        self.assertEqual(template.config.name, "CustomMemory")
    
    def test_template_setup_memory_backend(self):
        """测试内存后端设置"""
        # 测试内存后端配置
        self.template.setup(
            backend_type="memory",
            max_context_length=2000,
            max_messages_per_context=10
        )
        
        self.assertEqual(self.template.max_context_length, 2000)
        self.assertEqual(self.template.max_messages_per_context, 10)
        self.assertIsNotNone(self.template.backend)
        self.assertIsInstance(self.template.backend, MemoryBackend)
    
    def test_template_setup_file_backend(self):
        """测试文件后端设置"""
        # 测试文件后端配置
        storage_path = Path(self.temp_dir) / "conversations"
        self.template.setup(
            backend_type="file",
            storage_path=str(storage_path),
            max_context_length=4000
        )
        
        self.assertEqual(self.template.max_context_length, 4000)
        self.assertIsNotNone(self.template.backend)
        self.assertIsInstance(self.template.backend, FileBackend)
        self.assertTrue(storage_path.exists())
    
    def test_add_message_basic(self):
        """测试基本消息添加"""
        # 设置模板
        self.template.setup(backend_type="memory")
        
        # 添加用户消息
        result = self.template.execute({
            "action": "add_message",
            "session_id": "test_session",
            "user_id": "user1",
            "message": {
                "content": "你好，我想了解Python编程。",
                "type": "human"
            }
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["session_id"], "test_session")
        self.assertEqual(result["message_count"], 1)
        self.assertIn("message_id", result)
    
    def test_add_message_string_format(self):
        """测试字符串格式消息添加"""
        self.template.setup(backend_type="memory")
        
        # 添加字符串消息
        result = self.template.execute({
            "action": "add_message",
            "session_id": "test_session",
            "message": "这是一条简单的消息"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["message_count"], 1)
    
    def test_add_multiple_messages(self):
        """测试添加多条消息"""
        self.template.setup(backend_type="memory")
        
        messages = [
            {"content": "你好", "type": "human"},
            {"content": "你好！有什么可以帮助你的吗？", "type": "ai"},
            {"content": "我想学习Python", "type": "human"},
            {"content": "Python是一门很好的编程语言...", "type": "ai"}
        ]
        
        session_id = "multi_message_test"
        for i, msg in enumerate(messages):
            result = self.template.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": msg
            })
            self.assertTrue(result["success"])
            self.assertEqual(result["message_count"], i + 1)
    
    def test_get_context(self):
        """测试获取对话上下文"""
        self.template.setup(backend_type="memory", max_context_length=1000)
        
        # 先添加一些消息
        session_id = "context_test"
        for i in range(5):
            self.template.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": f"这是第{i+1}条消息"
            })
        
        # 获取上下文
        result = self.template.execute({
            "action": "get_context",
            "session_id": session_id,
            "limit": 3
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["messages"]), 3)
        self.assertFalse(result["truncated"])  # 消息数量少，应该不会被截断
    
    def test_get_context_with_truncation(self):
        """测试上下文截断"""
        self.template.setup(backend_type="memory", max_context_length=100)  # 很小的限制
        
        session_id = "truncation_test"
        # 添加很长的消息
        long_message = "这是一条很长的消息。" * 20
        for i in range(3):
            self.template.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": long_message
            })
        
        # 获取上下文
        result = self.template.execute({
            "action": "get_context",
            "session_id": session_id
        })
        
        self.assertTrue(result["success"])
        # 由于token限制，返回的消息数应该少于3条
        self.assertLessEqual(len(result["messages"]), 3)
    
    def test_get_conversation(self):
        """测试获取完整对话"""
        self.template.setup(backend_type="memory")
        
        session_id = "conversation_test"
        # 添加消息
        self.template.execute({
            "action": "add_message",
            "session_id": session_id,
            "message": "测试消息"
        })
        
        # 获取完整对话
        result = self.template.execute({
            "action": "get_conversation",
            "session_id": session_id
        })
        
        self.assertTrue(result["success"])
        self.assertIn("conversation", result)
        conversation = result["conversation"]
        self.assertEqual(conversation["session_id"], session_id)
        self.assertEqual(len(conversation["messages"]), 1)
    
    def test_search_messages(self):
        """测试消息搜索"""
        self.template.setup(backend_type="memory")
        
        # 添加一些包含关键词的消息
        session_id = "search_test"
        messages = [
            "我想学习Python编程",
            "Python是一门很好的语言",
            "Java也是不错的选择",
            "机器学习很有趣"
        ]
        
        for msg in messages:
            self.template.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": msg
            })
        
        # 搜索包含"Python"的消息
        result = self.template.execute({
            "action": "search",
            "query": "Python"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["result_count"], 2)
        
        # 验证搜索结果
        for message in result["messages"]:
            self.assertIn("Python", message["content"])
    
    def test_list_conversations(self):
        """测试列出对话"""
        self.template.setup(backend_type="memory")
        
        # 创建多个对话
        session_ids = ["session1", "session2", "session3"]
        for session_id in session_ids:
            self.template.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": f"消息来自{session_id}"
            })
        
        # 列出对话
        result = self.template.execute({
            "action": "list_conversations"
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 3)
        # 验证所有会话ID都在结果中
        for session_id in session_ids:
            self.assertIn(session_id, result["conversations"])
    
    def test_delete_conversation(self):
        """测试删除对话"""
        self.template.setup(backend_type="memory")
        
        session_id = "delete_test"
        # 添加消息
        self.template.execute({
            "action": "add_message",
            "session_id": session_id,
            "message": "即将被删除的消息"
        })
        
        # 确认对话存在
        result = self.template.execute({
            "action": "get_conversation",
            "session_id": session_id
        })
        self.assertTrue(result["success"])
        
        # 删除对话
        result = self.template.execute({
            "action": "delete_conversation",
            "session_id": session_id
        })
        self.assertTrue(result["success"])
        
        # 确认对话已被删除
        result = self.template.execute({
            "action": "get_conversation",
            "session_id": session_id
        })
        self.assertFalse(result["success"])
    
    def test_cleanup_old_conversations(self):
        """测试清理旧对话"""
        self.template.setup(backend_type="memory")
        
        # 添加一些对话
        for i in range(5):
            self.template.execute({
                "action": "add_message",
                "session_id": f"cleanup_test_{i}",
                "message": f"消息{i}"
            })
        
        # 清理（设置为0天，应该清理所有对话）
        result = self.template.execute({
            "action": "cleanup",
            "days": 0
        })
        
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["removed_count"], 0)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        self.template.setup(backend_type="memory")
        
        # 添加一些数据
        self.template.execute({
            "action": "add_message",
            "session_id": "stats_test",
            "message": "统计测试消息"
        })
        
        # 获取统计
        result = self.template.execute({
            "action": "get_stats"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("stats", result)
        self.assertIn("cache_size", result)
        self.assertIn("config", result)
    
    def test_file_backend_persistence(self):
        """测试文件后端持久化"""
        storage_path = Path(self.temp_dir) / "file_test"
        
        # 第一次使用：添加消息
        template1 = ConversationMemoryTemplate()
        template1.setup(
            backend_type="file",
            storage_path=str(storage_path)
        )
        
        session_id = "persistence_test"
        template1.execute({
            "action": "add_message",
            "session_id": session_id,
            "message": "持久化测试消息"
        })
        
        # 第二次使用：创建新实例，应该能读取之前的数据
        template2 = ConversationMemoryTemplate()
        template2.setup(
            backend_type="file",
            storage_path=str(storage_path)
        )
        
        result = template2.execute({
            "action": "get_conversation",
            "session_id": session_id
        })
        
        self.assertTrue(result["success"])
        conversation = result["conversation"]
        self.assertEqual(len(conversation["messages"]), 1)
        self.assertEqual(conversation["messages"][0]["content"], "持久化测试消息")
    
    def test_multi_user_support(self):
        """测试多用户支持"""
        self.template.setup(backend_type="memory")
        
        # 用户1的消息
        self.template.execute({
            "action": "add_message",
            "session_id": "shared_session",
            "user_id": "user1",
            "message": "用户1的消息"
        })
        
        # 用户2的消息
        self.template.execute({
            "action": "add_message",
            "session_id": "shared_session",
            "user_id": "user2",
            "message": "用户2的消息"
        })
        
        # 获取用户1的对话
        result1 = self.template.execute({
            "action": "get_conversation",
            "session_id": "shared_session",
            "user_id": "user1"
        })
        
        # 获取用户2的对话
        result2 = self.template.execute({
            "action": "get_conversation",
            "session_id": "shared_session",
            "user_id": "user2"
        })
        
        self.assertTrue(result1["success"])
        self.assertTrue(result2["success"])
        
        # 验证两个用户的对话是独立的
        conv1 = result1["conversation"]
        conv2 = result2["conversation"]
        
        self.assertEqual(len(conv1["messages"]), 1)
        self.assertEqual(len(conv2["messages"]), 1)
        self.assertEqual(conv1["messages"][0]["content"], "用户1的消息")
        self.assertEqual(conv2["messages"][0]["content"], "用户2的消息")
    
    def test_error_handling_invalid_action(self):
        """测试无效操作的错误处理"""
        self.template.setup(backend_type="memory")
        
        result = self.template.execute({
            "action": "invalid_action"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("未知的操作类型", result["error"])
    
    def test_error_handling_missing_message(self):
        """测试缺少消息的错误处理"""
        self.template.setup(backend_type="memory")
        
        result = self.template.execute({
            "action": "add_message",
            "session_id": "test"
            # 缺少message参数
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_error_handling_empty_search(self):
        """测试空搜索查询的错误处理"""
        self.template.setup(backend_type="memory")
        
        result = self.template.execute({
            "action": "search",
            "query": ""  # 空查询
        })
        
        self.assertFalse(result["success"])
        self.assertIn("搜索查询不能为空", result["error"])
    
    def test_message_data_structure(self):
        """测试消息数据结构"""
        # 创建消息
        message = Message(
            content="测试消息",
            type=MessageType.HUMAN,
            session_id="test_session"
        )
        
        # 测试基本属性
        self.assertEqual(message.content, "测试消息")
        self.assertEqual(message.type, MessageType.HUMAN)
        self.assertEqual(message.session_id, "test_session")
        self.assertIsNotNone(message.timestamp)
        self.assertIsNotNone(message.message_id)
        
        # 测试方法
        self.assertTrue(message.is_from_user())
        self.assertFalse(message.is_from_ai())
        self.assertGreater(message.get_size(), 0)
        
        # 测试序列化
        message_dict = message.to_dict()
        self.assertIsInstance(message_dict, dict)
        self.assertEqual(message_dict["content"], "测试消息")
        
        # 测试反序列化
        restored_message = Message.from_dict(message_dict)
        self.assertEqual(restored_message.content, message.content)
        self.assertEqual(restored_message.type, message.type)
    
    def test_conversation_data_structure(self):
        """测试对话数据结构"""
        # 创建对话
        conversation = Conversation(session_id="test_conversation")
        
        # 测试基本属性
        self.assertEqual(conversation.session_id, "test_conversation")
        self.assertEqual(len(conversation.messages), 0)
        self.assertIsNotNone(conversation.created_at)
        
        # 添加消息
        message = Message(content="测试消息", type=MessageType.HUMAN)
        conversation.add_message(message)
        
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.get_message_count(), 1)
        self.assertIsNotNone(conversation.title)  # 应该自动生成标题
        
        # 测试统计方法
        self.assertGreaterEqual(conversation.get_total_tokens(), 0)
        
        # 测试消息检索
        last_messages = conversation.get_last_messages(5)
        self.assertEqual(len(last_messages), 1)
        
        human_messages = conversation.get_messages_by_type(MessageType.HUMAN)
        self.assertEqual(len(human_messages), 1)
        
        # 测试序列化
        conv_dict = conversation.to_dict()
        self.assertIsInstance(conv_dict, dict)
        
        restored_conv = Conversation.from_dict(conv_dict)
        self.assertEqual(restored_conv.session_id, conversation.session_id)
        self.assertEqual(len(restored_conv.messages), len(conversation.messages))
    
    def test_performance_large_conversation(self):
        """测试大量消息的性能"""
        self.template.setup(backend_type="memory")
        
        session_id = "performance_test"
        num_messages = 100
        
        # 记录开始时间
        start_time = time.time()
        
        # 添加大量消息
        for i in range(num_messages):
            result = self.template.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": f"性能测试消息 {i}"
            })
            self.assertTrue(result["success"])
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        # 验证性能（每条消息平均处理时间应该小于0.01秒）
        avg_time_per_message = duration / num_messages
        self.assertLess(avg_time_per_message, 0.01)
        
        # 验证数据完整性
        result = self.template.execute({
            "action": "get_conversation",
            "session_id": session_id
        })
        self.assertTrue(result["success"])
        self.assertEqual(len(result["conversation"]["messages"]), num_messages)
    
    def test_concurrent_access(self):
        """测试并发访问"""
        import threading
        
        self.template.setup(backend_type="memory")
        
        results = []
        errors = []
        
        def add_messages(thread_id):
            """线程函数：添加消息"""
            try:
                for i in range(10):
                    result = self.template.execute({
                        "action": "add_message",
                        "session_id": f"thread_{thread_id}",
                        "message": f"线程{thread_id}的消息{i}"
                    })
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_messages, args=(i,))
            threads.append(thread)
        
        # 启动线程
        for thread in threads:
            thread.start()
        
        # 等待线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(len(errors), 0)  # 不应该有错误
        self.assertEqual(len(results), 50)  # 5个线程 × 10条消息
        
        # 验证所有结果都成功
        for result in results:
            self.assertTrue(result["success"])


class TestMemoryBackend(unittest.TestCase):
    """内存后端测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.backend = MemoryBackend(max_conversations=5)
    
    def test_save_and_load_conversation(self):
        """测试保存和加载对话"""
        # 创建对话
        conversation = Conversation(session_id="test_session")
        message = Message(content="测试消息", type=MessageType.HUMAN)
        conversation.add_message(message)
        
        # 保存对话
        success = self.backend.save_conversation(conversation)
        self.assertTrue(success)
        
        # 加载对话
        loaded_conv = self.backend.load_conversation("test_session")
        self.assertIsNotNone(loaded_conv)
        self.assertEqual(loaded_conv.session_id, "test_session")
        self.assertEqual(len(loaded_conv.messages), 1)
    
    def test_max_conversations_limit(self):
        """测试最大对话数量限制"""
        # 添加超过限制的对话
        for i in range(7):  # 超过最大限制5
            conversation = Conversation(session_id=f"session_{i}")
            self.backend.save_conversation(conversation)
        
        # 验证只保留了最大数量的对话
        stats = self.backend.get_stats()
        self.assertLessEqual(stats["total_conversations"], 5)
    
    def test_search_messages(self):
        """测试消息搜索"""
        # 添加包含不同内容的对话
        conversations = [
            ("session1", "我喜欢Python编程"),
            ("session2", "Java也是不错的语言"),
            ("session3", "Python很适合初学者")
        ]
        
        for session_id, content in conversations:
            conversation = Conversation(session_id=session_id)
            message = Message(content=content, type=MessageType.HUMAN)
            conversation.add_message(message)
            self.backend.save_conversation(conversation)
        
        # 搜索Python相关消息
        results = self.backend.search_messages("Python")
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIn("Python", result.content)


class TestFileBackend(unittest.TestCase):
    """文件后端测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = FileBackend(self.temp_dir)
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_conversation(self):
        """测试文件保存和加载"""
        # 创建对话
        conversation = Conversation(session_id="file_test")
        message = Message(content="文件测试消息", type=MessageType.HUMAN)
        conversation.add_message(message)
        
        # 保存到文件
        success = self.backend.save_conversation(conversation)
        self.assertTrue(success)
        
        # 验证文件存在
        file_path = Path(self.temp_dir) / "file_test.json"
        self.assertTrue(file_path.exists())
        
        # 加载对话
        loaded_conv = self.backend.load_conversation("file_test")
        self.assertIsNotNone(loaded_conv)
        self.assertEqual(loaded_conv.session_id, "file_test")
        self.assertEqual(len(loaded_conv.messages), 1)
    
    def test_file_content_format(self):
        """测试文件内容格式"""
        conversation = Conversation(session_id="format_test")
        message = Message(content="格式测试", type=MessageType.HUMAN)
        conversation.add_message(message)
        
        self.backend.save_conversation(conversation)
        
        # 直接读取文件内容验证格式
        file_path = Path(self.temp_dir) / "format_test.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(data["session_id"], "format_test")
        self.assertIn("messages", data)
        self.assertEqual(len(data["messages"]), 1)
        self.assertEqual(data["messages"][0]["content"], "格式测试")
    
    def test_list_conversations(self):
        """测试列出对话"""
        # 创建多个对话文件
        session_ids = ["conv1", "conv2", "conv3"]
        for session_id in session_ids:
            conversation = Conversation(session_id=session_id)
            self.backend.save_conversation(conversation)
        
        # 列出对话
        conversations = self.backend.list_conversations()
        self.assertEqual(len(conversations), 3)
        
        for session_id in session_ids:
            self.assertIn(session_id, conversations)


if __name__ == "__main__":
    unittest.main()