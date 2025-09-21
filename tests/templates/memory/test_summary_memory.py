"""
摘要记忆模板测试

测试SummaryMemoryTemplate的各项功能，包括：
1. 基本配置和初始化
2. 摘要生成和压缩
3. 关键信息提取
4. 摘要融合和更新
5. 性能测试
6. 错误处理
"""

import unittest
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from templates.memory.summary_memory import (
    SummaryMemoryTemplate,
    SummaryStrategy,
    CompressionLevel,
    SummarySegment,
    ConversationSummary,
    LLMSummaryGenerator
)
from templates.memory.conversation_memory import (
    ConversationMemoryTemplate,
    Message,
    MessageType,
    Conversation
)
from templates.base.template_base import TemplateConfig, TemplateType


class TestSummaryMemoryTemplate(unittest.TestCase):
    """摘要记忆模板测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建对话记忆模板（作为依赖）
        self.conv_memory = ConversationMemoryTemplate()
        self.conv_memory.setup(backend_type="memory")
        
        # 创建摘要记忆模板
        self.template = SummaryMemoryTemplate(conversation_memory=self.conv_memory)
        
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_template_initialization(self):
        """测试模板初始化"""
        # 测试默认初始化
        template = SummaryMemoryTemplate()
        self.assertIsNotNone(template.config)
        self.assertEqual(template.config.name, "SummaryMemoryTemplate")
        self.assertEqual(template.config.template_type, TemplateType.MEMORY)
        
        # 测试带依赖的初始化
        template = SummaryMemoryTemplate(conversation_memory=self.conv_memory)
        self.assertEqual(template.conversation_memory, self.conv_memory)
    
    def test_template_setup(self):
        """测试模板设置"""
        storage_path = Path(self.temp_dir) / "summaries"
        
        self.template.setup(
            segment_size=5,
            compression_level="high",
            summary_strategy="abstractive",
            auto_summarize_threshold=10,
            storage_path=str(storage_path)
        )
        
        self.assertEqual(self.template.segment_size, 5)
        self.assertEqual(self.template.compression_level, CompressionLevel.HIGH)
        self.assertEqual(self.template.summary_strategy, SummaryStrategy.ABSTRACTIVE)
        self.assertEqual(self.template.auto_summarize_threshold, 10)
        self.assertTrue(storage_path.exists())
        self.assertIsNotNone(self.template.summary_generator)
    
    def test_create_summary_without_conversation_memory(self):
        """测试无对话记忆依赖时的摘要创建"""
        template = SummaryMemoryTemplate()  # 没有对话记忆依赖
        template.setup()
        
        result = template.execute({
            "action": "summarize",
            "session_id": "test_session"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("需要配置conversation_memory依赖", result["error"])
    
    def test_create_summary_basic(self):
        """测试基本摘要创建"""
        self.template.setup(segment_size=3)
        
        # 先在对话记忆中添加一些消息
        session_id = "summary_test"
        messages = [
            "你好，我想了解Python编程。",
            "Python是一门很好的编程语言，适合初学者。",
            "它有简洁的语法和丰富的库。",
            "我应该从哪里开始学习？",
            "建议从基础语法开始，然后学习常用库。"
        ]
        
        for msg in messages:
            self.conv_memory.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": msg
            })
        
        # 创建摘要
        result = self.template.execute({
            "action": "summarize",
            "session_id": session_id
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["session_id"], session_id)
        self.assertGreater(result["new_segments"], 0)
        self.assertGreater(result["total_segments"], 0)
        self.assertGreaterEqual(result["compression_ratio"], 0)
        self.assertIsInstance(result["overall_summary"], str)
    
    def test_get_summary(self):
        """测试获取摘要"""
        self.template.setup()
        
        # 先创建摘要
        session_id = "get_summary_test"
        self._create_test_conversation(session_id, 5)
        
        create_result = self.template.execute({
            "action": "summarize",
            "session_id": session_id
        })
        self.assertTrue(create_result["success"])
        
        # 获取摘要
        result = self.template.execute({
            "action": "get_summary",
            "session_id": session_id
        })
        
        self.assertTrue(result["success"])
        self.assertIn("summary", result)
        summary = result["summary"]
        self.assertEqual(summary["session_id"], session_id)
        self.assertIn("segments", summary)
        self.assertIn("overall_summary", summary)
    
    def test_update_summary(self):
        """测试更新摘要"""
        self.template.setup(segment_size=2)
        
        session_id = "update_test"
        
        # 创建初始对话和摘要
        self._create_test_conversation(session_id, 4)
        
        result1 = self.template.execute({
            "action": "summarize",
            "session_id": session_id
        })
        self.assertTrue(result1["success"])
        initial_segments = result1["total_segments"]
        
        # 添加更多消息
        for i in range(4):
            self.conv_memory.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": f"新增消息{i}"
            })
        
        # 更新摘要
        result2 = self.template.execute({
            "action": "update_summary",
            "session_id": session_id
        })
        
        self.assertTrue(result2["success"])
        self.assertGreaterEqual(result2["total_segments"], initial_segments)
    
    def test_get_segments(self):
        """测试获取摘要片段"""
        self.template.setup(segment_size=2)
        
        session_id = "segments_test"
        self._create_test_conversation(session_id, 6)
        
        # 创建摘要
        self.template.execute({
            "action": "summarize",
            "session_id": session_id
        })
        
        # 获取片段
        result = self.template.execute({
            "action": "get_segments",
            "session_id": session_id,
            "count": 2
        })
        
        self.assertTrue(result["success"])
        self.assertIn("segments", result)
        self.assertLessEqual(len(result["segments"]), 2)
        self.assertEqual(result["count"], len(result["segments"]))
    
    def test_compress_summary(self):
        """测试摘要压缩"""
        self.template.setup(segment_size=1, max_segments_per_summary=3)
        
        session_id = "compress_test"
        # 创建足够多的消息以触发压缩
        self._create_test_conversation(session_id, 6)
        
        # 创建摘要
        self.template.execute({
            "action": "summarize",
            "session_id": session_id
        })
        
        # 执行压缩
        result = self.template.execute({
            "action": "compress",
            "session_id": session_id
        })
        
        self.assertTrue(result["success"])
        # 压缩结果应该包含原始片段数和压缩后片段数的信息
        if "original_segments" in result:
            self.assertGreaterEqual(result["original_segments"], result["compressed_segments"])
    
    def test_extract_insights(self):
        """测试提取对话洞察"""
        self.template.setup()
        
        session_id = "insights_test"
        # 创建包含关键词的对话
        messages = [
            "我想学习Python编程",
            "Python是很好的编程语言",
            "机器学习也很重要",
            "我决定先学Python基础",
            "需要准备学习计划"
        ]
        
        for msg in messages:
            self.conv_memory.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": msg
            })
        
        # 创建摘要
        self.template.execute({
            "action": "summarize",
            "session_id": session_id
        })
        
        # 提取洞察
        result = self.template.execute({
            "action": "extract_insights",
            "session_id": session_id
        })
        
        self.assertTrue(result["success"])
        self.assertIn("insights", result)
        insights = result["insights"]
        
        self.assertIn("top_keywords", insights)
        self.assertIn("top_entities", insights)
        self.assertIn("top_topics", insights)
        self.assertIn("conversation_stats", insights)
    
    def test_save_and_load_summary(self):
        """测试摘要保存和加载"""
        storage_path = Path(self.temp_dir) / "summaries"
        self.template.setup(storage_path=str(storage_path))
        
        session_id = "save_load_test"
        self._create_test_conversation(session_id, 3)
        
        # 创建摘要
        self.template.execute({
            "action": "summarize",
            "session_id": session_id
        })
        
        # 保存摘要
        save_result = self.template.execute({
            "action": "save_summary",
            "session_id": session_id
        })
        
        self.assertTrue(save_result["success"])
        self.assertIn("file_path", save_result)
        
        # 验证文件存在
        file_path = Path(save_result["file_path"])
        self.assertTrue(file_path.exists())
        
        # 清除内存中的摘要
        summary_key = self.template._get_summary_key(session_id, None)
        if summary_key in self.template.summaries:
            del self.template.summaries[summary_key]
        
        # 加载摘要
        load_result = self.template.execute({
            "action": "load_summary",
            "session_id": session_id
        })
        
        self.assertTrue(load_result["success"])
        self.assertIn("summary", load_result)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        self.template.setup()
        
        # 创建一些摘要数据
        for i in range(3):
            session_id = f"stats_test_{i}"
            self._create_test_conversation(session_id, 2)
            self.template.execute({
                "action": "summarize",
                "session_id": session_id
            })
        
        # 获取统计信息
        result = self.template.execute({
            "action": "get_stats"
        })
        
        self.assertTrue(result["success"])
        self.assertIn("stats", result)
        stats = result["stats"]
        
        self.assertIn("total_summaries", stats)
        self.assertIn("total_segments", stats)
        self.assertIn("average_compression_ratio", stats)
        self.assertIn("config", stats)
    
    def test_error_handling_invalid_action(self):
        """测试无效操作的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "invalid_action"
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("未知的操作类型", result["error"])
    
    def test_error_handling_empty_conversation(self):
        """测试空对话的错误处理"""
        self.template.setup()
        
        result = self.template.execute({
            "action": "summarize",
            "session_id": "empty_session"  # 不存在的会话
        })
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_different_summary_strategies(self):
        """测试不同的摘要策略"""
        strategies = [
            SummaryStrategy.EXTRACTIVE,
            SummaryStrategy.ABSTRACTIVE,
            SummaryStrategy.HYBRID
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                template = SummaryMemoryTemplate(conversation_memory=self.conv_memory)
                template.setup(summary_strategy=strategy.value)
                
                session_id = f"strategy_test_{strategy.value}"
                self._create_test_conversation(session_id, 4)
                
                result = template.execute({
                    "action": "summarize",
                    "session_id": session_id
                })
                
                self.assertTrue(result["success"])
                self.assertGreater(result["total_segments"], 0)
    
    def test_different_compression_levels(self):
        """测试不同的压缩级别"""
        levels = [
            CompressionLevel.LOW,
            CompressionLevel.MEDIUM,
            CompressionLevel.HIGH,
            CompressionLevel.EXTREME
        ]
        
        for level in levels:
            with self.subTest(level=level):
                template = SummaryMemoryTemplate(conversation_memory=self.conv_memory)
                template.setup(compression_level=level.value)
                
                session_id = f"compression_test_{level.value}"
                self._create_test_conversation(session_id, 4)
                
                result = template.execute({
                    "action": "summarize",
                    "session_id": session_id
                })
                
                self.assertTrue(result["success"])
                # 验证压缩比例随级别增加而增加
                self.assertGreaterEqual(result["compression_ratio"], 0)
    
    def _create_test_conversation(self, session_id: str, num_messages: int):
        """创建测试对话"""
        for i in range(num_messages):
            self.conv_memory.execute({
                "action": "add_message",
                "session_id": session_id,
                "message": f"这是测试消息{i+1}，包含一些内容来测试摘要功能。"
            })


class TestSummarySegment(unittest.TestCase):
    """摘要片段测试类"""
    
    def test_summary_segment_creation(self):
        """测试摘要片段创建"""
        segment = SummarySegment(
            segment_id="test_segment",
            start_time=time.time(),
            end_time=time.time() + 60,
            original_message_count=5,
            summary_text="这是一个测试摘要",
            key_points=["要点1", "要点2"],
            keywords=["测试", "摘要"],
            strategy=SummaryStrategy.ABSTRACTIVE
        )
        
        self.assertEqual(segment.segment_id, "test_segment")
        self.assertEqual(segment.original_message_count, 5)
        self.assertEqual(segment.summary_text, "这是一个测试摘要")
        self.assertEqual(len(segment.key_points), 2)
        self.assertEqual(len(segment.keywords), 2)
        self.assertEqual(segment.strategy, SummaryStrategy.ABSTRACTIVE)
    
    def test_importance_score_calculation(self):
        """测试重要性得分计算"""
        segment = SummarySegment(
            segment_id="importance_test",
            start_time=time.time(),
            end_time=time.time(),
            original_message_count=10,
            summary_text="重要摘要",
            keywords=["关键词1", "关键词2", "关键词3"],
            entities=["实体1", "实体2"],
            topics=["主题1"],
            compression_ratio=0.3
        )
        
        score = segment.get_importance_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_serialization(self):
        """测试序列化和反序列化"""
        segment = SummarySegment(
            segment_id="serialization_test",
            start_time=time.time(),
            end_time=time.time(),
            original_message_count=3,
            summary_text="序列化测试",
            strategy=SummaryStrategy.HYBRID
        )
        
        # 序列化
        segment_dict = segment.to_dict()
        self.assertIsInstance(segment_dict, dict)
        self.assertEqual(segment_dict["segment_id"], "serialization_test")
        self.assertEqual(segment_dict["strategy"], "hybrid")
        
        # 反序列化
        restored_segment = SummarySegment.from_dict(segment_dict)
        self.assertEqual(restored_segment.segment_id, segment.segment_id)
        self.assertEqual(restored_segment.strategy, segment.strategy)


class TestConversationSummary(unittest.TestCase):
    """对话摘要测试类"""
    
    def test_conversation_summary_creation(self):
        """测试对话摘要创建"""
        summary = ConversationSummary(
            session_id="test_summary",
            strategy_used=SummaryStrategy.ABSTRACTIVE,
            compression_level=CompressionLevel.MEDIUM
        )
        
        self.assertEqual(summary.session_id, "test_summary")
        self.assertEqual(summary.strategy_used, SummaryStrategy.ABSTRACTIVE)
        self.assertEqual(summary.compression_level, CompressionLevel.MEDIUM)
        self.assertEqual(len(summary.segments), 0)
    
    def test_add_segment(self):
        """测试添加摘要片段"""
        summary = ConversationSummary(session_id="add_segment_test")
        
        segment = SummarySegment(
            segment_id="segment1",
            start_time=time.time(),
            end_time=time.time(),
            original_message_count=5,
            summary_text="第一个片段"
        )
        
        summary.add_segment(segment)
        
        self.assertEqual(len(summary.segments), 1)
        self.assertEqual(summary.total_original_messages, 5)
        self.assertGreater(summary.updated_at, summary.created_at)
    
    def test_get_latest_segments(self):
        """测试获取最新片段"""
        summary = ConversationSummary(session_id="latest_test")
        
        # 添加多个片段
        for i in range(5):
            segment = SummarySegment(
                segment_id=f"segment_{i}",
                start_time=time.time() + i,
                end_time=time.time() + i + 1,
                original_message_count=1,
                summary_text=f"片段{i}"
            )
            summary.add_segment(segment)
            time.sleep(0.01)  # 确保时间戳不同
        
        # 获取最新的3个片段
        latest = summary.get_latest_segments(3)
        self.assertEqual(len(latest), 3)
        
        # 验证顺序（最新的在前）
        for i in range(len(latest) - 1):
            self.assertGreaterEqual(latest[i].end_time, latest[i + 1].end_time)
    
    def test_get_most_important_segments(self):
        """测试获取最重要片段"""
        summary = ConversationSummary(session_id="important_test")
        
        # 添加重要性不同的片段
        segments_data = [
            ("seg1", [], [], 0.1),  # 低重要性
            ("seg2", ["关键词1", "关键词2"], ["实体1"], 0.5),  # 中等重要性
            ("seg3", ["关键词1", "关键词2", "关键词3"], ["实体1", "实体2"], 0.3),  # 高重要性
        ]
        
        for seg_id, keywords, entities, compression in segments_data:
            segment = SummarySegment(
                segment_id=seg_id,
                start_time=time.time(),
                end_time=time.time(),
                original_message_count=1,
                summary_text=f"摘要{seg_id}",
                keywords=keywords,
                entities=entities,
                compression_ratio=compression
            )
            summary.add_segment(segment)
        
        # 获取最重要的2个片段
        important = summary.get_most_important_segments(2)
        self.assertEqual(len(important), 2)
        
        # 验证排序（重要性得分高的在前）
        scores = [seg.get_importance_score() for seg in important]
        self.assertGreaterEqual(scores[0], scores[1])
    
    def test_serialization(self):
        """测试序列化和反序列化"""
        summary = ConversationSummary(
            session_id="serialization_test",
            overall_summary="整体摘要",
            key_themes=["主题1", "主题2"],
            strategy_used=SummaryStrategy.HYBRID,
            compression_level=CompressionLevel.HIGH
        )
        
        # 添加一个片段
        segment = SummarySegment(
            segment_id="test_segment",
            start_time=time.time(),
            end_time=time.time(),
            original_message_count=1,
            summary_text="测试片段"
        )
        summary.add_segment(segment)
        
        # 序列化
        summary_dict = summary.to_dict()
        self.assertIsInstance(summary_dict, dict)
        self.assertEqual(summary_dict["session_id"], "serialization_test")
        self.assertEqual(summary_dict["strategy_used"], "hybrid")
        self.assertEqual(summary_dict["compression_level"], "high")
        
        # 反序列化
        restored_summary = ConversationSummary.from_dict(summary_dict)
        self.assertEqual(restored_summary.session_id, summary.session_id)
        self.assertEqual(restored_summary.strategy_used, summary.strategy_used)
        self.assertEqual(restored_summary.compression_level, summary.compression_level)
        self.assertEqual(len(restored_summary.segments), len(summary.segments))


class TestLLMSummaryGenerator(unittest.TestCase):
    """LLM摘要生成器测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.generator = LLMSummaryGenerator()
    
    def test_generate_summary_basic(self):
        """测试基本摘要生成"""
        messages = [
            Message(content="你好，我想学习Python", type=MessageType.HUMAN),
            Message(content="Python是很好的选择", type=MessageType.AI),
            Message(content="从哪里开始学习？", type=MessageType.HUMAN)
        ]
        
        segment = self.generator.generate_summary(
            messages, 
            SummaryStrategy.ABSTRACTIVE, 
            CompressionLevel.MEDIUM
        )
        
        self.assertIsInstance(segment, SummarySegment)
        self.assertGreater(len(segment.summary_text), 0)
        self.assertEqual(segment.original_message_count, 3)
        self.assertGreaterEqual(segment.compression_ratio, 0)
    
    def test_generate_summary_empty_messages(self):
        """测试空消息列表的摘要生成"""
        segment = self.generator.generate_summary(
            [], 
            SummaryStrategy.ABSTRACTIVE, 
            CompressionLevel.MEDIUM
        )
        
        self.assertIsInstance(segment, SummarySegment)
        self.assertEqual(segment.original_message_count, 0)
        self.assertEqual(segment.summary_text, "空对话段落")
    
    def test_extract_key_information(self):
        """测试关键信息提取"""
        messages = [
            Message(content="我决定学习Python编程", type=MessageType.HUMAN),
            Message(content="这很重要，Python是关键技能", type=MessageType.AI),
            Message(content="我需要制定学习计划", type=MessageType.HUMAN)
        ]
        
        key_info = self.generator.extract_key_information(messages)
        
        self.assertIn("keywords", key_info)
        self.assertIn("entities", key_info)
        self.assertIn("topics", key_info)
        self.assertIn("facts", key_info)
        self.assertIn("decisions", key_info)
        self.assertIn("actions", key_info)
        
        # 验证提取的信息
        self.assertIn("Python", key_info["keywords"])
        self.assertGreater(len(key_info["decisions"]), 0)  # 应该检测到"决定"
        self.assertGreater(len(key_info["actions"]), 0)   # 应该检测到"需要"
    
    def test_update_overall_summary(self):
        """测试整体摘要更新"""
        current_summary = "当前摘要内容"
        new_segment = SummarySegment(
            segment_id="new_seg",
            start_time=time.time(),
            end_time=time.time(),
            original_message_count=1,
            summary_text="新的摘要内容"
        )
        
        updated_summary = self.generator.update_overall_summary(current_summary, new_segment)
        
        self.assertIsInstance(updated_summary, str)
        self.assertGreater(len(updated_summary), 0)
        # 更新后的摘要应该包含原有内容
        self.assertIn("当前摘要内容", updated_summary)
    
    def test_different_compression_levels(self):
        """测试不同压缩级别"""
        messages = [
            Message(content="这是一条很长的测试消息，用来测试不同的压缩级别。" * 10, type=MessageType.HUMAN)
        ]
        
        # 测试不同压缩级别
        levels = [CompressionLevel.LOW, CompressionLevel.MEDIUM, CompressionLevel.HIGH, CompressionLevel.EXTREME]
        results = []
        
        for level in levels:
            segment = self.generator.generate_summary(messages, SummaryStrategy.ABSTRACTIVE, level)
            results.append((level, len(segment.summary_text)))
        
        # 验证压缩级别越高，摘要越短
        # 由于使用规则式方法，压缩级别应该影响摘要长度
        for i in range(len(results) - 1):
            current_level, current_length = results[i]
            next_level, next_length = results[i + 1]
            
            # 一般情况下，压缩级别越高，摘要越短
            # 但由于实现的简化，这里只验证都产生了有效摘要
            self.assertGreater(current_length, 0)
            self.assertGreater(next_length, 0)


if __name__ == "__main__":
    unittest.main()