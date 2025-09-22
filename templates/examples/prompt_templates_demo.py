#!/usr/bin/env python3
"""
Prompt模板演示脚本

本脚本演示了四种Prompt模板的基本使用方法：
1. ChatTemplate - 多轮对话模板
2. CompletionTemplate - 文本补全模板  
3. FewShotTemplate - 少样本学习模板
4. RolePlayingTemplate - 角色扮演模板

运行前请确保：
1. 已安装所需依赖
2. 设置了相应的API密钥（如果使用在线LLM）
3. 激活了Python虚拟环境

使用方法：
    python prompt_templates_demo.py
"""

import os
import sys
import time
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from templates.prompts import (
    ChatTemplate, CompletionTemplate, FewShotTemplate, RolePlayingTemplate,
    Example, ExampleType, RoleProfile, RoleType, InteractionMode
)

# 模拟LLM模板（用于演示，实际使用时需要真实的LLM）
class MockLLMTemplate:
    """模拟LLM模板，用于演示目的"""
    
    def __init__(self):
        self.model_name = "mock-llm"
    
    def setup(self, **kwargs):
        """模拟设置"""
        pass
    
    def execute(self, prompt: str, **kwargs):
        """模拟执行，返回简单的响应"""
        class MockResponse:
            def __init__(self, content: str):
                self.content = content
                self.total_tokens = len(content.split())
        
        # 简单的响应生成逻辑
        if "问题" in prompt or "什么" in prompt:
            content = "这是一个很好的问题。基于我的理解，我认为..."
        elif "代码" in prompt or "编程" in prompt:
            content = """```python
def example_function():
    '''这是一个示例函数'''
    return "Hello, World!"
```"""
        elif "继续" in prompt or "续写" in prompt:
            content = "基于上下文，我将继续这个话题。让我们深入探讨..."
        else:
            content = "感谢您的输入。我理解您的需求，让我为您提供相应的帮助..."
        
        return MockResponse(content)


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_subsection(title: str):
    """打印子章节标题"""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def demo_chat_template():
    """演示对话模板"""
    print_section("ChatTemplate 演示")
    
    # 创建模拟LLM
    llm = MockLLMTemplate()
    
    # 创建对话模板
    chat_template = ChatTemplate()
    
    print("1. 设置对话模板...")
    chat_template.setup(
        role_name="Python编程助手",
        role_description="专业的Python编程指导老师，擅长解答编程问题",
        personality="耐心、专业、善于解释复杂概念",
        expertise=["Python编程", "算法设计", "代码优化"],
        conversation_style="循序渐进的教学方式",
        llm_template=llm
    )
    
    print("✓ 对话模板设置完成")
    
    print("\n2. 进行多轮对话...")
    
    # 第一轮对话
    print("\n用户: 我想学习Python的列表推导式")
    response1 = chat_template.run(
        "我想学习Python的列表推导式",
        conversation_id="python_learning"
    )
    print(f"助手: {response1.message.content}")
    print(f"建议操作: {', '.join(response1.suggested_actions)}")
    
    # 第二轮对话
    print("\n用户: 能给我一个具体的例子吗？")
    response2 = chat_template.run(
        "能给我一个具体的例子吗？",
        conversation_id="python_learning"  # 同一对话
    )
    print(f"助手: {response2.message.content}")
    
    # 获取对话信息
    conv_info = chat_template.get_conversation_info("python_learning")
    print(f"\n对话统计: 共 {conv_info['message_count']} 条消息")
    
    return chat_template


def demo_completion_template():
    """演示补全模板"""
    print_section("CompletionTemplate 演示")
    
    llm = MockLLMTemplate()
    
    # 创建补全模板
    completion_template = CompletionTemplate()
    
    print("1. 设置补全模板...")
    completion_template.setup(
        completion_type="article",
        strategy="continue",
        target_length=500,
        style="professional",
        llm_template=llm
    )
    
    print("✓ 补全模板设置完成")
    
    print("\n2. 文本续写演示...")
    
    input_text = "人工智能技术的发展正在深刻改变我们的生活方式。从智能手机的语音助手到自动驾驶汽车"
    
    print(f"原始文本: {input_text}")
    
    result = completion_template.run(
        input_text,
        completion_type="article",
        target_length=300
    )
    
    print(f"补全结果: {result.completed_text}")
    print(f"质量分数: {result.quality_score:.2f}")
    print(f"新增长度: {result.added_length} 字符")
    print(f"改进建议: {', '.join(result.suggestions)}")
    
    print("\n3. 代码生成演示...")
    
    code_input = """
def fibonacci(n):
    '''计算斐波那契数列的第n项'''
    if n <= 1:
        return n
    # TODO: 实现递归逻辑
"""
    
    code_result = completion_template.run(
        code_input,
        completion_type="code",
        target_length=200
    )
    
    print(f"代码补全结果:\n{code_result.completed_text}")
    
    return completion_template


def demo_few_shot_template():
    """演示少样本学习模板"""
    print_section("FewShotTemplate 演示")
    
    llm = MockLLMTemplate()
    
    # 创建少样本学习模板
    few_shot_template = FewShotTemplate()
    
    print("1. 设置少样本学习模板...")
    few_shot_template.setup(
        example_type="classification",
        selection_strategy="adaptive",
        max_examples=3,
        llm_template=llm
    )
    
    print("✓ 少样本学习模板设置完成")
    
    print("\n2. 添加示例...")
    
    # 添加情感分类示例
    examples = [
        Example(
            input_text="这部电影真的很棒，演员表演出色！",
            output_text="positive",
            example_type=ExampleType.CLASSIFICATION
        ),
        Example(
            input_text="服务态度太差了，完全不推荐。",
            output_text="negative",
            example_type=ExampleType.CLASSIFICATION
        ),
        Example(
            input_text="价格还算合理，没什么特别的。",
            output_text="neutral",
            example_type=ExampleType.CLASSIFICATION
        ),
        Example(
            input_text="质量超出预期，非常满意！",
            output_text="positive",
            example_type=ExampleType.CLASSIFICATION
        )
    ]
    
    few_shot_template.add_examples(examples)
    print(f"✓ 已添加 {len(examples)} 个示例")
    
    print("\n3. 进行情感分类...")
    
    test_texts = [
        "这家餐厅的菜品味道不错，值得推荐。",
        "等了半个小时才上菜，效率太低了。",
        "价格中等，服务一般般。"
    ]
    
    for text in test_texts:
        result = few_shot_template.run(text)
        print(f"输入: {text}")
        print(f"分类结果: {result.prediction}")
        print(f"使用示例数: {len(result.selected_examples)}")
        print()
    
    # 获取统计信息
    stats = few_shot_template.get_statistics()
    print(f"模板统计: 总示例数 {stats['total_examples']}, 平均质量 {stats['average_quality']:.2f}")
    
    return few_shot_template


def demo_role_playing_template():
    """演示角色扮演模板"""
    print_section("RolePlayingTemplate 演示")
    
    llm = MockLLMTemplate()
    
    # 创建角色扮演模板
    role_template = RolePlayingTemplate()
    
    print("1. 设置角色扮演模板...")
    role_template.setup(
        role_name="医生",
        interaction_mode="consultation",
        llm_template=llm
    )
    
    print("✓ 角色扮演模板设置完成")
    
    print("\n2. 医生咨询演示...")
    
    consultation_queries = [
        "医生，我最近总是感觉很累，这可能是什么原因？",
        "我应该如何改善我的睡眠质量？",
        "谢谢医生的建议，我会注意的。"
    ]
    
    for query in consultation_queries:
        print(f"\n患者: {query}")
        response = role_template.run(
            query,
            session_id="medical_consultation",
            scenario="患者健康咨询"
        )
        print(f"医生: {response.response_text}")
        
        if response.professional_advice:
            print(f"专业建议: {response.professional_advice}")
        
        if response.disclaimers:
            print(f"免责声明: {'; '.join(response.disclaimers)}")
    
    print("\n3. 切换到教师角色...")
    
    role_template.set_active_role("教师")
    
    teaching_response = role_template.run(
        "老师，我不理解二次函数的概念",
        interaction_mode="teaching",
        session_id="math_class"
    )
    
    print(f"\n学生: 老师，我不理解二次函数的概念")
    print(f"老师: {teaching_response.response_text}")
    print(f"后续话题: {', '.join(teaching_response.next_topics)}")
    
    # 列出可用角色
    available_roles = role_template.list_available_roles()
    print(f"\n可用角色: {[role['name'] for role in available_roles]}")
    
    return role_template


def demo_integration():
    """演示模板集成使用"""
    print_section("模板集成演示")
    
    llm = MockLLMTemplate()
    
    print("演示多个模板的协同使用...")
    
    # 使用角色扮演模板生成问题
    role_template = RolePlayingTemplate()
    role_template.setup(role_name="教师", llm_template=llm)
    
    # 使用补全模板扩展内容
    completion_template = CompletionTemplate()
    completion_template.setup(llm_template=llm)
    
    # 使用对话模板进行互动
    chat_template = ChatTemplate()
    chat_template.setup(
        role_name="学习助手",
        llm_template=llm
    )
    
    print("✓ 多个模板已准备就绪")
    
    # 模拟一个完整的学习场景
    print("\n学习场景演示：")
    
    # 1. 老师提出问题
    teacher_question = role_template.run(
        "请为学生出一道关于Python基础的练习题",
        interaction_mode="teaching"
    )
    print(f"老师: {teacher_question.response_text}")
    
    # 2. 学习助手提供帮助
    assistant_help = chat_template.run(
        "学生需要帮助理解这个题目",
        conversation_id="learning_session"
    )
    print(f"助手: {assistant_help.message.content}")
    
    print("\n✓ 集成演示完成")


def demo_advanced_features():
    """演示高级功能"""
    print_section("高级功能演示")
    
    llm = MockLLMTemplate()
    
    print_subsection("1. 自定义角色创建")
    
    # 创建自定义角色
    role_template = RolePlayingTemplate()
    role_template.setup(llm_template=llm)
    
    custom_role = role_template.create_custom_role(
        name="AI产品经理",
        role_type=RoleType.BUSINESS,
        title="资深产品经理",
        background="计算机科学硕士，10年产品经验",
        specialties=["AI产品设计", "用户体验", "技术管理"],
        personality="创新、务实、用户导向",
        communication_style="结构化思维，数据驱动"
    )
    
    print(f"✓ 创建自定义角色: {custom_role.name}")
    
    # 使用自定义角色
    role_template.set_active_role("AI产品经理")
    product_response = role_template.run(
        "我们如何设计一个更好的AI聊天产品？",
        interaction_mode="consultation"
    )
    print(f"产品经理: {product_response.response_text}")
    
    print_subsection("2. 模板配置导出导入")
    
    # 导出角色配置
    role_data = role_template.export_role("AI产品经理")
    print("✓ 角色配置已导出")
    
    # 创建新模板并导入配置
    new_role_template = RolePlayingTemplate()
    new_role_template.setup(llm_template=llm)
    success = new_role_template.import_role(role_data)
    print(f"✓ 角色配置导入: {'成功' if success else '失败'}")
    
    print_subsection("3. 动态示例管理")
    
    few_shot = FewShotTemplate()
    few_shot.setup(llm_template=llm)
    
    # 批量添加示例
    examples_data = [
        {"input": "优秀的产品", "output": "positive", "type": "classification"},
        {"input": "糟糕的体验", "output": "negative", "type": "classification"},
        {"input": "普通的服务", "output": "neutral", "type": "classification"}
    ]
    
    added_count = few_shot.bulk_add_examples(examples_data)
    print(f"✓ 批量添加示例: {added_count} 个")
    
    # 清理低质量示例
    cleaned = few_shot.cleanup_low_quality_examples(min_quality=0.5)
    print(f"✓ 清理低质量示例: {cleaned} 个")


def main():
    """主函数"""
    print("🚀 Prompt模板系统演示")
    print("=" * 60)
    print("本演示将展示四种核心模板的使用方法:")
    print("1. ChatTemplate - 多轮对话")
    print("2. CompletionTemplate - 文本补全")  
    print("3. FewShotTemplate - 少样本学习")
    print("4. RolePlayingTemplate - 角色扮演")
    print("5. 模板集成使用")
    print("6. 高级功能演示")
    
    try:
        # 逐个演示各个模板
        demo_chat_template()
        demo_completion_template()
        demo_few_shot_template()
        demo_role_playing_template()
        demo_integration()
        demo_advanced_features()
        
        print_section("演示完成")
        print("✅ 所有模板演示已完成！")
        print("\n📝 要点总结:")
        print("- 所有模板都支持LLM集成")
        print("- 提供统一的setup()和run()接口")
        print("- 支持丰富的配置选项和参数化")
        print("- 包含完整的错误处理和状态管理")
        print("- 可以灵活组合使用以构建复杂应用")
        
        print("\n🔗 相关文档:")
        print("- 模板详细文档: templates/examples/tutorials/")
        print("- API参考: templates/README.md")
        print("- 更多示例: templates/examples/")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        print("请检查依赖是否正确安装，或查看错误详情进行调试。")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())