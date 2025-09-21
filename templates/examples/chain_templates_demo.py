#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chain模板演示程序

本文件展示了如何使用各种Chain模板，包括：
1. SequentialChainTemplate - 顺序链模板
2. ParallelChainTemplate - 并行链模板  
3. ConditionalChainTemplate - 条件链模板
4. PipelineChainTemplate - 管道链模板

每个演示都包含：
- 模板创建和配置
- 数据准备和执行
- 结果展示和分析
- 错误处理和最佳实践

作者: Claude Code Assistant
版本: 1.0.0
创建时间: 2024-09-21
"""

import asyncio
import time
import json
import sys
import os
from typing import Dict, Any, List
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入链模板
from templates.chains import (
    create_chain,
    get_available_chain_types,
    SequentialChainTemplate,
    ParallelChainTemplate,
    ConditionalChainTemplate,
    PipelineChainTemplate,
    ConditionConfig,
    ConditionType,
    ComparisonOperator
)

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChainDemoRunner:
    """Chain模板演示运行器"""
    
    def __init__(self):
        """初始化演示运行器"""
        self.results = {}
        self.demo_data = self._prepare_demo_data()
        
        logger.info("初始化Chain模板演示运行器")
    
    def _prepare_demo_data(self) -> Dict[str, Any]:
        """准备演示数据"""
        return {
            # === 文本处理数据 ===
            "text_data": {
                "content": "这是一个测试文档，用于演示各种链模板的功能。文档包含了文本处理、分析和转换的示例。",
                "language": "zh",
                "type": "document",
                "metadata": {
                    "author": "demo",
                    "created_at": "2024-09-21",
                    "category": "test"
                }
            },
            
            # === 用户数据 ===
            "user_data": {
                "id": "12345",
                "name": "张三",
                "level": "VIP",
                "score": 85,
                "registration_days": 120,
                "preferences": ["tech", "ai", "python"],
                "activity": {
                    "login_count": 50,
                    "last_login": "2024-09-20"
                }
            },
            
            # === 数据处理数据 ===
            "raw_data": {
                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "labels": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                "config": {
                    "normalize": True,
                    "filter_outliers": False,
                    "aggregation": "mean"
                }
            }
        }
    
    def run_all_demos(self) -> None:
        """运行所有演示"""
        logger.info("=" * 60)
        logger.info("开始Chain模板演示")
        logger.info("=" * 60)
        
        # 显示可用的链类型
        self._show_available_chain_types()
        
        # 运行各个演示
        demos = [
            ("顺序链演示", self.demo_sequential_chain),
            ("并行链演示", self.demo_parallel_chain),
            ("条件链演示", self.demo_conditional_chain),
            ("管道链演示", self.demo_pipeline_chain),
            ("链嵌套演示", self.demo_nested_chains),
            ("异步执行演示", self.demo_async_execution),
            ("错误处理演示", self.demo_error_handling)
        ]
        
        for demo_name, demo_func in demos:
            try:
                logger.info(f"\n{'='*20} {demo_name} {'='*20}")
                demo_func()
                logger.info(f"✓ {demo_name}完成")
            except Exception as e:
                logger.error(f"✗ {demo_name}失败: {str(e)}")
        
        # 显示总结
        self._show_demo_summary()
    
    def _show_available_chain_types(self) -> None:
        """显示可用的链类型"""
        logger.info("\n可用的链类型:")
        for chain_type in get_available_chain_types():
            logger.info(f"  - {chain_type}")
    
    def demo_sequential_chain(self) -> None:
        """演示顺序链模板"""
        logger.info("创建顺序链模板...")
        
        # 定义步骤执行函数
        def step1_clean_text(data, **kwargs):
            """步骤1：文本清洗"""
            content = data.get("content", "")
            cleaned = content.strip().replace("  ", " ")
            logger.info(f"步骤1 - 文本清洗: '{content[:30]}...' -> '{cleaned[:30]}...'")
            return {"cleaned_content": cleaned, "original_length": len(content)}
        
        def step2_analyze_text(data, **kwargs):
            """步骤2：文本分析"""
            content = data.get("cleaned_content", "")
            word_count = len(content.split())
            char_count = len(content)
            logger.info(f"步骤2 - 文本分析: 字数={word_count}, 字符数={char_count}")
            return {
                "cleaned_content": content,
                "word_count": word_count,
                "char_count": char_count,
                "analysis_time": time.time()
            }
        
        def step3_generate_summary(data, **kwargs):
            """步骤3：生成摘要"""
            content = data.get("cleaned_content", "")
            word_count = data.get("word_count", 0)
            summary = f"文档摘要：包含{word_count}个词的文本内容"
            logger.info(f"步骤3 - 生成摘要: {summary}")
            return {
                "summary": summary,
                "processed_at": time.time(),
                "statistics": {
                    "word_count": word_count,
                    "char_count": data.get("char_count", 0)
                }
            }
        
        # 创建和配置顺序链
        chain = create_chain('sequential')
        chain.setup(
            steps=[
                {
                    "name": "文本清洗",
                    "executor": step1_clean_text,
                    "description": "清洗和预处理文本内容"
                },
                {
                    "name": "文本分析",
                    "executor": step2_analyze_text,
                    "description": "分析文本的基本统计信息"
                },
                {
                    "name": "生成摘要",
                    "executor": step3_generate_summary,
                    "description": "基于分析结果生成文档摘要"
                }
            ],
            error_strategy="fail_fast",
            enable_caching=True
        )
        
        # 执行链
        start_time = time.time()
        result = chain.run(self.demo_data["text_data"])
        execution_time = time.time() - start_time
        
        # 保存结果
        self.results["sequential"] = {
            "result": result,
            "execution_time": execution_time
        }
        
        # 显示结果
        logger.info(f"顺序链执行完成，耗时: {execution_time:.3f}秒")
        logger.info(f"最终结果: {json.dumps(result['data'], ensure_ascii=False, indent=2)}")
        logger.info(f"执行摘要: 共{result['summary']['total_steps']}步，成功{result['summary']['completed_steps']}步")
    
    def demo_parallel_chain(self) -> None:
        """演示并行链模板"""
        logger.info("创建并行链模板...")
        
        # 定义并行分支函数
        def model_a_predict(data, **kwargs):
            """模型A预测"""
            user_score = data.get("score", 0)
            prediction = f"ModelA预测：用户价值等级 = {'高' if user_score > 80 else '中' if user_score > 50 else '低'}"
            time.sleep(0.5)  # 模拟处理时间
            logger.info(f"模型A完成预测: {prediction}")
            return {"model_a_result": prediction, "confidence": 0.85}
        
        def model_b_predict(data, **kwargs):
            """模型B预测"""
            level = data.get("level", "")
            prediction = f"ModelB预测：用户类型 = {level}用户，推荐优先级高"
            time.sleep(0.3)  # 模拟处理时间
            logger.info(f"模型B完成预测: {prediction}")
            return {"model_b_result": prediction, "confidence": 0.92}
        
        def rule_engine(data, **kwargs):
            """规则引擎"""
            days = data.get("registration_days", 0)
            login_count = data.get("activity", {}).get("login_count", 0)
            rule_result = f"规则引擎：注册{days}天，登录{login_count}次，评级为活跃用户"
            time.sleep(0.2)  # 模拟处理时间
            logger.info(f"规则引擎完成分析: {rule_result}")
            return {"rule_result": rule_result, "activity_score": min(100, days + login_count)}
        
        # 创建和配置并行链
        chain = create_chain('parallel')
        chain.setup(
            branches=[
                {
                    "name": "模型A预测",
                    "executor": model_a_predict,
                    "priority": 2,
                    "timeout": 5.0,
                    "description": "使用模型A进行用户价值预测"
                },
                {
                    "name": "模型B预测",
                    "executor": model_b_predict,
                    "priority": 2,
                    "timeout": 5.0,
                    "description": "使用模型B进行用户类型预测"
                },
                {
                    "name": "规则引擎",
                    "executor": rule_engine,
                    "priority": 1,
                    "timeout": 3.0,
                    "description": "基于规则的用户分析"
                }
            ],
            max_workers=3,
            execution_mode="thread",
            aggregation_strategy="all"
        )
        
        # 执行链
        start_time = time.time()
        result = chain.run(self.demo_data["user_data"])
        execution_time = time.time() - start_time
        
        # 保存结果
        self.results["parallel"] = {
            "result": result,
            "execution_time": execution_time
        }
        
        # 显示结果
        logger.info(f"并行链执行完成，耗时: {execution_time:.3f}秒")
        logger.info(f"聚合结果: {json.dumps(result['aggregated_data'], ensure_ascii=False, indent=2)}")
        logger.info(f"执行摘要: 共{result['summary']['total_branches']}个分支，成功{result['summary']['completed_branches']}个")
    
    def demo_conditional_chain(self) -> None:
        """演示条件链模板"""
        logger.info("创建条件链模板...")
        
        # 定义条件分支函数
        def handle_vip_user(data, **kwargs):
            """处理VIP用户"""
            user_name = data.get("name", "用户")
            service = f"VIP用户{user_name}：提供专属客服、优先处理、专享优惠"
            logger.info(f"VIP用户处理: {service}")
            return {"service_type": "VIP", "service_description": service, "priority": "highest"}
        
        def handle_high_value_user(data, **kwargs):
            """处理高价值用户"""
            user_name = data.get("name", "用户")
            service = f"高价值用户{user_name}：提供增值服务、快速响应、个性化推荐"
            logger.info(f"高价值用户处理: {service}")
            return {"service_type": "high_value", "service_description": service, "priority": "high"}
        
        def handle_new_user(data, **kwargs):
            """处理新用户"""
            user_name = data.get("name", "用户")
            service = f"新用户{user_name}：提供入门指导、新手福利、学习资源"
            logger.info(f"新用户处理: {service}")
            return {"service_type": "new_user", "service_description": service, "priority": "medium"}
        
        def handle_normal_user(data, **kwargs):
            """处理普通用户"""
            user_name = data.get("name", "用户")
            service = f"普通用户{user_name}：提供标准服务、常规支持、基础功能"
            logger.info(f"普通用户处理: {service}")
            return {"service_type": "normal", "service_description": service, "priority": "normal"}
        
        # 创建和配置条件链
        chain = create_chain('conditional')
        chain.setup(
            branches=[
                {
                    "name": "VIP用户分支",
                    "condition": {
                        "name": "vip_user",
                        "type": "value",
                        "field_path": "level",
                        "operator": "in",
                        "value": ["VIP", "SVIP", "PREMIUM"]
                    },
                    "executor": handle_vip_user,
                    "priority": 3,
                    "description": "处理VIP级别用户"
                },
                {
                    "name": "高价值用户分支",
                    "condition": {
                        "name": "high_value_user",
                        "type": "value",
                        "field_path": "score",
                        "operator": "gt",
                        "value": 80
                    },
                    "executor": handle_high_value_user,
                    "priority": 2,
                    "description": "处理高价值用户"
                },
                {
                    "name": "新用户分支",
                    "condition": {
                        "name": "new_user",
                        "type": "value",
                        "field_path": "registration_days",
                        "operator": "lt",
                        "value": 30
                    },
                    "executor": handle_new_user,
                    "priority": 1,
                    "description": "处理新注册用户"
                }
            ],
            default_branch={
                "name": "普通用户分支",
                "executor": handle_normal_user,
                "description": "处理普通用户"
            },
            evaluation_strategy="first_match",
            cache_enabled=True
        )
        
        # 执行链
        start_time = time.time()
        result = chain.run(self.demo_data["user_data"])
        execution_time = time.time() - start_time
        
        # 保存结果
        self.results["conditional"] = {
            "result": result,
            "execution_time": execution_time
        }
        
        # 显示结果
        logger.info(f"条件链执行完成，耗时: {execution_time:.3f}秒")
        logger.info(f"选中分支: {result['selected_branch']['name']}")
        logger.info(f"服务结果: {json.dumps(result['output_data'], ensure_ascii=False, indent=2)}")
        
        # 显示条件评估结果
        logger.info("条件评估结果:")
        for evaluation in result['condition_evaluations']:
            logger.info(f"  - {evaluation['branch_name']}: {evaluation['result']}")
    
    def demo_pipeline_chain(self) -> None:
        """演示管道链模板"""
        logger.info("创建管道链模板...")
        
        # 定义各阶段函数
        def preprocess_data(data, context, **kwargs):
            """数据预处理"""
            values = data.get("values", [])
            processed_values = [x * 2 for x in values]  # 简单的数据变换
            logger.info(f"数据预处理: {len(values)}个值 -> {len(processed_values)}个处理后的值")
            return {"processed_values": processed_values, "processing_steps": ["multiply_by_2"]}
        
        def feature_extraction_a(data, **kwargs):
            """特征提取A"""
            values = data.get("processed_values", [])
            features_a = {"mean": sum(values) / len(values), "max": max(values), "min": min(values)}
            logger.info(f"特征提取A: {features_a}")
            return {"features_a": features_a}
        
        def feature_extraction_b(data, **kwargs):
            """特征提取B"""
            values = data.get("processed_values", [])
            features_b = {"sum": sum(values), "count": len(values), "variance": sum((x - sum(values)/len(values))**2 for x in values) / len(values)}
            logger.info(f"特征提取B: {features_b}")
            return {"features_b": features_b}
        
        def aggregate_results(data, **kwargs):
            """聚合结果"""
            features_a = data.get("features_a", {})
            features_b = data.get("features_b", {})
            
            final_result = {
                "basic_stats": features_a,
                "advanced_stats": features_b,
                "summary": f"数据包含{features_b.get('count', 0)}个值，平均值为{features_a.get('mean', 0):.2f}"
            }
            logger.info(f"结果聚合完成: {final_result['summary']}")
            return final_result
        
        # 创建和配置管道链
        chain = create_chain('pipeline')
        chain.setup(
            stages=[
                {
                    "name": "数据预处理阶段",
                    "stage_type": "sequential",
                    "template_config": {
                        "steps": [
                            {
                                "name": "数据预处理",
                                "executor": preprocess_data,
                                "description": "对原始数据进行预处理"
                            }
                        ]
                    },
                    "description": "第一阶段：数据预处理"
                },
                {
                    "name": "特征提取阶段",
                    "stage_type": "parallel",
                    "template_config": {
                        "branches": [
                            {
                                "name": "基础特征提取",
                                "executor": feature_extraction_a,
                                "description": "提取基础统计特征"
                            },
                            {
                                "name": "高级特征提取",
                                "executor": feature_extraction_b,
                                "description": "提取高级统计特征"
                            }
                        ]
                    },
                    "dependencies": ["stage_0"],
                    "description": "第二阶段：并行特征提取"
                },
                {
                    "name": "结果聚合阶段",
                    "stage_type": "sequential",
                    "template_config": {
                        "steps": [
                            {
                                "name": "聚合结果",
                                "executor": aggregate_results,
                                "description": "聚合所有特征提取结果"
                            }
                        ]
                    },
                    "dependencies": ["stage_1"],
                    "description": "第三阶段：结果聚合"
                }
            ],
            data_flow_mode="pipeline",
            max_parallel_stages=2
        )
        
        # 执行链
        start_time = time.time()
        result = chain.run(self.demo_data["raw_data"])
        execution_time = time.time() - start_time
        
        # 保存结果
        self.results["pipeline"] = {
            "result": result,
            "execution_time": execution_time
        }
        
        # 显示结果
        logger.info(f"管道链执行完成，耗时: {execution_time:.3f}秒")
        logger.info(f"最终输出: {json.dumps(result['output_data'], ensure_ascii=False, indent=2)}")
        logger.info(f"阶段输出: {list(result['stage_outputs'].keys())}")
        logger.info(f"执行摘要: 共{result['summary']['total_stages']}个阶段，成功{result['summary']['completed_stages']}个")
    
    def demo_nested_chains(self) -> None:
        """演示链嵌套"""
        logger.info("创建嵌套链演示...")
        
        # 创建子链：文本处理链
        def tokenize_text(data, **kwargs):
            """文本分词"""
            content = data.get("content", "")
            tokens = content.split()
            logger.info(f"文本分词: {len(tokens)}个词")
            return {"tokens": tokens, "token_count": len(tokens)}
        
        def analyze_tokens(data, **kwargs):
            """词汇分析"""
            tokens = data.get("tokens", [])
            unique_tokens = list(set(tokens))
            logger.info(f"词汇分析: 唯一词汇{len(unique_tokens)}个")
            return {"unique_tokens": unique_tokens, "vocabulary_size": len(unique_tokens)}
        
        # 创建文本处理子链
        text_chain = create_chain('sequential')
        text_chain.setup(
            steps=[
                {"name": "文本分词", "executor": tokenize_text},
                {"name": "词汇分析", "executor": analyze_tokens}
            ]
        )
        
        # 在管道链中使用子链
        def use_text_chain(data, context, **kwargs):
            """使用文本处理子链"""
            logger.info("调用文本处理子链...")
            text_result = text_chain.run(data)
            return text_result["data"]
        
        def analyze_user_content(data, **kwargs):
            """分析用户内容偏好"""
            vocabulary_size = data.get("vocabulary_size", 0)
            preferences = []
            if vocabulary_size > 10:
                preferences.append("详细表达")
            if vocabulary_size > 5:
                preferences.append("丰富词汇")
            
            result = {"content_preferences": preferences, "complexity_score": vocabulary_size / 10}
            logger.info(f"用户内容分析: {result}")
            return result
        
        # 创建主管道链
        main_chain = create_chain('pipeline')
        main_chain.setup(
            stages=[
                {
                    "name": "文本处理阶段",
                    "stage_type": "custom",
                    "executor": use_text_chain,
                    "description": "使用子链进行文本处理"
                },
                {
                    "name": "用户分析阶段",
                    "stage_type": "sequential",
                    "template_config": {
                        "steps": [
                            {
                                "name": "内容偏好分析",
                                "executor": analyze_user_content
                            }
                        ]
                    },
                    "dependencies": ["stage_0"],
                    "description": "分析用户的内容偏好"
                }
            ]
        )
        
        # 执行嵌套链
        start_time = time.time()
        result = main_chain.run(self.demo_data["text_data"])
        execution_time = time.time() - start_time
        
        # 保存结果
        self.results["nested"] = {
            "result": result,
            "execution_time": execution_time
        }
        
        logger.info(f"嵌套链执行完成，耗时: {execution_time:.3f}秒")
        logger.info(f"最终结果: {json.dumps(result['output_data'], ensure_ascii=False, indent=2)}")
    
    def demo_async_execution(self) -> None:
        """演示异步执行"""
        logger.info("创建异步执行演示...")
        
        async def async_demo():
            # 定义异步函数
            async def async_process_a(data, **kwargs):
                """异步处理A"""
                await asyncio.sleep(0.5)  # 模拟异步操作
                result = {"process_a": f"异步处理A完成: {data.get('name', 'unknown')}"}
                logger.info(f"异步处理A完成")
                return result
            
            async def async_process_b(data, **kwargs):
                """异步处理B"""
                await asyncio.sleep(0.3)  # 模拟异步操作
                result = {"process_b": f"异步处理B完成: {data.get('level', 'unknown')}"}
                logger.info(f"异步处理B完成")
                return result
            
            def sync_process_c(data, **kwargs):
                """同步处理C"""
                time.sleep(0.2)  # 模拟同步操作
                result = {"process_c": f"同步处理C完成: {data.get('score', 0)}"}
                logger.info(f"同步处理C完成")
                return result
            
            # 创建并行链进行异步执行
            chain = create_chain('parallel')
            chain.setup(
                branches=[
                    {"name": "异步处理A", "executor": async_process_a},
                    {"name": "异步处理B", "executor": async_process_b}, 
                    {"name": "同步处理C", "executor": sync_process_c}
                ],
                max_workers=3
            )
            
            # 异步执行链
            start_time = time.time()
            result = await chain.run_async(self.demo_data["user_data"])
            execution_time = time.time() - start_time
            
            # 保存结果
            self.results["async"] = {
                "result": result,
                "execution_time": execution_time
            }
            
            logger.info(f"异步执行完成，耗时: {execution_time:.3f}秒")
            logger.info(f"异步结果: {json.dumps(result['aggregated_data'], ensure_ascii=False, indent=2)}")
        
        # 运行异步演示
        asyncio.run(async_demo())
    
    def demo_error_handling(self) -> None:
        """演示错误处理"""
        logger.info("创建错误处理演示...")
        
        # 定义会出错的函数
        def step_success(data, **kwargs):
            """成功的步骤"""
            logger.info("步骤执行成功")
            return {"success_result": "成功处理数据"}
        
        def step_failure(data, **kwargs):
            """失败的步骤"""
            logger.error("模拟步骤失败")
            raise ValueError("这是一个故意的错误，用于演示错误处理")
        
        def step_recovery(data, **kwargs):
            """恢复步骤"""
            logger.info("执行恢复逻辑")
            return {"recovery_result": "从错误中恢复"}
        
        # 演示不同的错误处理策略
        error_strategies = ["fail_fast", "continue", "skip"]
        
        for strategy in error_strategies:
            logger.info(f"\n测试错误处理策略: {strategy}")
            
            try:
                # 创建顺序链
                chain = create_chain('sequential')
                chain.setup(
                    steps=[
                        {"name": "成功步骤1", "executor": step_success},
                        {"name": "失败步骤", "executor": step_failure, "error_strategy": strategy},
                        {"name": "恢复步骤", "executor": step_recovery}
                    ],
                    error_strategy=strategy
                )
                
                # 执行链
                result = chain.run({"test_data": "error_handling"})
                logger.info(f"策略 {strategy} - 执行结果: {result['status']}")
                
            except Exception as e:
                logger.info(f"策略 {strategy} - 捕获异常: {str(e)}")
        
        # 保存错误处理演示结果
        self.results["error_handling"] = {
            "strategies_tested": error_strategies,
            "status": "completed"
        }
    
    def _show_demo_summary(self) -> None:
        """显示演示总结"""
        logger.info("\n" + "=" * 60)
        logger.info("Chain模板演示总结")
        logger.info("=" * 60)
        
        for demo_name, demo_result in self.results.items():
            if "execution_time" in demo_result:
                logger.info(f"{demo_name:15}: 执行时间 {demo_result['execution_time']:.3f}秒")
            else:
                logger.info(f"{demo_name:15}: {demo_result.get('status', '已完成')}")
        
        logger.info("\n演示要点总结:")
        logger.info("1. 顺序链适用于步骤依次执行的场景")
        logger.info("2. 并行链适用于多个独立任务同时执行的场景")
        logger.info("3. 条件链适用于根据条件选择执行路径的场景")
        logger.info("4. 管道链适用于复杂工作流编排的场景")
        logger.info("5. 链嵌套允许构建更复杂的处理逻辑")
        logger.info("6. 异步执行提高了并发处理能力")
        logger.info("7. 错误处理机制保证了系统的稳定性")


def main():
    """主函数"""
    try:
        # 创建演示运行器
        demo_runner = ChainDemoRunner()
        
        # 运行所有演示
        demo_runner.run_all_demos()
        
        logger.info("\n所有Chain模板演示完成！")
        
    except KeyboardInterrupt:
        logger.info("\n演示被用户中断")
    except Exception as e:
        logger.error(f"演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()