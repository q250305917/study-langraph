"""
学习进度追踪系统

用于追踪和可视化LangChain学习进度的核心模块。
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum

# 可视化库（在Jupyter中使用）
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display, HTML, Markdown
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ProgressStatus(Enum):
    """学习状态枚举"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    MASTERED = "mastered"


class DifficultyLevel(Enum):
    """难度级别枚举"""
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class LearningSession:
    """学习会话数据类"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    sections_completed: List[str] = field(default_factory=list)
    exercises_completed: List[str] = field(default_factory=list)
    notes: str = ""
    
    def end_session(self):
        """结束会话"""
        self.end_time = datetime.now()
        if self.start_time:
            delta = self.end_time - self.start_time
            self.duration_minutes = delta.total_seconds() / 60


@dataclass
class LessonProgress:
    """课程进度数据类"""
    lesson_id: str
    lesson_name: str
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    progress_percentage: float = 0.0
    total_sections: int = 0
    completed_sections: int = 0
    total_exercises: int = 0
    completed_exercises: int = 0
    total_time_minutes: float = 0.0
    last_accessed: Optional[datetime] = None
    sessions: List[LearningSession] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    badges: List[str] = field(default_factory=list)


@dataclass
class Achievement:
    """成就数据类"""
    achievement_id: str
    name: str
    description: str
    icon: str
    unlocked: bool = False
    unlocked_date: Optional[datetime] = None
    progress: float = 0.0
    requirement: float = 100.0


class ProgressTracker:
    """
    学习进度追踪器
    
    主要功能：
    1. 追踪学习进度和时间
    2. 管理学习会话
    3. 记录练习完成情况
    4. 计算学习统计
    5. 生成进度报告
    6. 管理成就系统
    """
    
    def __init__(self, user_id: str = "default", data_dir: str = None):
        """
        初始化进度追踪器
        
        Args:
            user_id: 用户标识
            data_dir: 数据存储目录
        """
        self.user_id = user_id
        self.data_dir = Path(data_dir or "progress_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_file = self.data_dir / f"{user_id}_progress.json"
        self.current_session: Optional[LearningSession] = None
        self.current_lesson: Optional[str] = None
        
        # 加载进度数据
        self.progress_data = self._load_progress()
        
        # 初始化成就系统
        self.achievements = self._init_achievements()
    
    def _load_progress(self) -> Dict[str, Any]:
        """加载进度数据"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 转换日期字符串为datetime对象
                    return self._deserialize_dates(data)
            except Exception as e:
                print(f"⚠️ 加载进度数据失败: {e}")
                return self._init_empty_progress()
        return self._init_empty_progress()
    
    def _init_empty_progress(self) -> Dict[str, Any]:
        """初始化空进度数据"""
        return {
            "user_id": self.user_id,
            "created_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_time_minutes": 0.0,
            "total_sessions": 0,
            "total_lessons_completed": 0,
            "total_exercises_completed": 0,
            "current_streak_days": 0,
            "longest_streak_days": 0,
            "last_study_date": None,
            "lessons": {},
            "achievements": {},
            "statistics": {
                "average_session_minutes": 0.0,
                "preferred_study_time": None,
                "most_difficult_topic": None,
                "strongest_area": None
            }
        }
    
    def _save_progress(self):
        """保存进度数据"""
        try:
            # 序列化数据
            data = self._serialize_dates(self.progress_data)
            
            # 保存到文件
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                
            # 更新最后修改时间
            self.progress_data["last_updated"] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"❌ 保存进度数据失败: {e}")
    
    def _serialize_dates(self, obj: Any) -> Any:
        """序列化日期对象"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_dates(item) for item in obj]
        elif isinstance(obj, (LessonProgress, LearningSession, Achievement)):
            return self._serialize_dates(asdict(obj))
        elif isinstance(obj, (ProgressStatus, DifficultyLevel)):
            return obj.value
        return obj
    
    def _deserialize_dates(self, obj: Any) -> Any:
        """反序列化日期字符串"""
        if isinstance(obj, str):
            try:
                # 尝试解析ISO格式日期
                if 'T' in obj and len(obj) > 10:
                    return datetime.fromisoformat(obj)
            except:
                pass
            return obj
        elif isinstance(obj, dict):
            return {k: self._deserialize_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_dates(item) for item in obj]
        return obj
    
    def _init_achievements(self) -> Dict[str, Achievement]:
        """初始化成就系统"""
        achievements = {
            "first_lesson": Achievement(
                "first_lesson", 
                "初学者", 
                "完成第一节课程",
                "🎯",
                requirement=1
            ),
            "week_streak": Achievement(
                "week_streak",
                "坚持一周",
                "连续学习7天",
                "🔥",
                requirement=7
            ),
            "speed_learner": Achievement(
                "speed_learner",
                "速学达人",
                "单日学习超过2小时",
                "⚡",
                requirement=120
            ),
            "exercise_master": Achievement(
                "exercise_master",
                "练习大师",
                "完成50个练习题",
                "💪",
                requirement=50
            ),
            "knowledge_seeker": Achievement(
                "knowledge_seeker",
                "知识探索者",
                "完成10节不同课程",
                "🎓",
                requirement=10
            ),
            "perfectionist": Achievement(
                "perfectionist",
                "完美主义者",
                "获得5个满分练习",
                "⭐",
                requirement=5
            ),
            "night_owl": Achievement(
                "night_owl",
                "夜猫子",
                "在晚上10点后学习",
                "🦉",
                requirement=1
            ),
            "early_bird": Achievement(
                "early_bird",
                "早起鸟",
                "在早上6点前学习",
                "🐦",
                requirement=1
            )
        }
        
        # 从保存的数据中恢复成就状态
        if "achievements" in self.progress_data:
            for aid, adata in self.progress_data["achievements"].items():
                if aid in achievements:
                    achievements[aid].unlocked = adata.get("unlocked", False)
                    achievements[aid].progress = adata.get("progress", 0.0)
                    if adata.get("unlocked_date"):
                        achievements[aid].unlocked_date = self._deserialize_dates(
                            adata["unlocked_date"]
                        )
        
        return achievements
    
    def start_lesson(self, lesson_id: str, lesson_name: str = None):
        """
        开始学习课程
        
        Args:
            lesson_id: 课程ID
            lesson_name: 课程名称
        """
        self.current_lesson = lesson_id
        
        # 创建新会话
        session_id = hashlib.md5(
            f"{lesson_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        self.current_session = LearningSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        # 初始化或更新课程进度
        if lesson_id not in self.progress_data["lessons"]:
            self.progress_data["lessons"][lesson_id] = {
                "lesson_id": lesson_id,
                "lesson_name": lesson_name or lesson_id,
                "status": ProgressStatus.IN_PROGRESS.value,
                "progress_percentage": 0.0,
                "total_sections": 0,
                "completed_sections": 0,
                "total_exercises": 0,
                "completed_exercises": 0,
                "total_time_minutes": 0.0,
                "last_accessed": datetime.now().isoformat(),
                "sessions": [],
                "scores": {},
                "badges": []
            }
        else:
            self.progress_data["lessons"][lesson_id]["last_accessed"] = datetime.now().isoformat()
            self.progress_data["lessons"][lesson_id]["status"] = ProgressStatus.IN_PROGRESS.value
        
        # 更新学习连续天数
        self._update_streak()
        
        self._save_progress()
        
        print(f"🚀 开始学习: {lesson_name or lesson_id}")
        print(f"   会话ID: {session_id}")
    
    def end_lesson(self):
        """结束当前课程学习"""
        if not self.current_session:
            return
        
        # 结束会话
        self.current_session.end_session()
        
        # 更新课程数据
        if self.current_lesson and self.current_lesson in self.progress_data["lessons"]:
            lesson = self.progress_data["lessons"][self.current_lesson]
            
            # 添加会话记录
            lesson["sessions"].append(self._serialize_dates(self.current_session))
            
            # 更新总时间
            lesson["total_time_minutes"] += self.current_session.duration_minutes
            self.progress_data["total_time_minutes"] += self.current_session.duration_minutes
            
            # 更新统计
            self.progress_data["total_sessions"] += 1
            
            # 计算进度百分比
            if lesson["total_sections"] > 0:
                lesson["progress_percentage"] = (
                    lesson["completed_sections"] / lesson["total_sections"] * 100
                )
            
            # 检查是否完成
            if lesson["progress_percentage"] >= 100:
                lesson["status"] = ProgressStatus.COMPLETED.value
                self.progress_data["total_lessons_completed"] += 1
                self._check_achievement("first_lesson")
                self._check_achievement("knowledge_seeker")
        
        # 检查学习时长成就
        if self.current_session.duration_minutes >= 120:
            self._unlock_achievement("speed_learner")
        
        # 检查时间相关成就
        hour = self.current_session.start_time.hour
        if hour >= 22 or hour < 4:
            self._unlock_achievement("night_owl")
        elif hour < 6:
            self._unlock_achievement("early_bird")
        
        self._save_progress()
        
        print(f"📚 结束学习，本次时长: {self.current_session.duration_minutes:.1f}分钟")
        
        self.current_session = None
        self.current_lesson = None
    
    def complete_section(self, section_name: str):
        """
        完成章节
        
        Args:
            section_name: 章节名称
        """
        if not self.current_lesson:
            return
        
        lesson = self.progress_data["lessons"].get(self.current_lesson)
        if not lesson:
            return
        
        # 记录到当前会话
        if self.current_session and section_name not in self.current_session.sections_completed:
            self.current_session.sections_completed.append(section_name)
        
        # 更新完成数量
        lesson["completed_sections"] += 1
        
        # 更新进度
        if lesson["total_sections"] > 0:
            lesson["progress_percentage"] = (
                lesson["completed_sections"] / lesson["total_sections"] * 100
            )
        
        self._save_progress()
        
        print(f"✅ 完成章节: {section_name}")
    
    def complete_exercise(self, exercise_name: str, score: float = None):
        """
        完成练习
        
        Args:
            exercise_name: 练习名称
            score: 得分（0-100）
        """
        if not self.current_lesson:
            return
        
        lesson = self.progress_data["lessons"].get(self.current_lesson)
        if not lesson:
            return
        
        # 记录到当前会话
        if self.current_session and exercise_name not in self.current_session.exercises_completed:
            self.current_session.exercises_completed.append(exercise_name)
        
        # 更新完成数量
        lesson["completed_exercises"] += 1
        self.progress_data["total_exercises_completed"] += 1
        
        # 记录得分
        if score is not None:
            lesson["scores"][exercise_name] = score
            
            # 检查满分成就
            if score >= 100:
                perfect_count = sum(1 for s in lesson["scores"].values() if s >= 100)
                self.achievements["perfectionist"].progress = perfect_count
                if perfect_count >= 5:
                    self._unlock_achievement("perfectionist")
        
        # 检查练习大师成就
        self.achievements["exercise_master"].progress = self.progress_data["total_exercises_completed"]
        if self.progress_data["total_exercises_completed"] >= 50:
            self._unlock_achievement("exercise_master")
        
        self._save_progress()
        
        result_msg = f"💪 完成练习: {exercise_name}"
        if score is not None:
            result_msg += f" (得分: {score:.1f}/100)"
        print(result_msg)
    
    def set_lesson_info(self, lesson_id: str, total_sections: int, total_exercises: int):
        """
        设置课程信息
        
        Args:
            lesson_id: 课程ID
            total_sections: 总章节数
            total_exercises: 总练习数
        """
        if lesson_id in self.progress_data["lessons"]:
            self.progress_data["lessons"][lesson_id]["total_sections"] = total_sections
            self.progress_data["lessons"][lesson_id]["total_exercises"] = total_exercises
            self._save_progress()
    
    def get_lesson_progress(self, lesson_id: str = None) -> Optional[Dict[str, Any]]:
        """
        获取课程进度
        
        Args:
            lesson_id: 课程ID，默认为当前课程
            
        Returns:
            课程进度数据
        """
        lesson_id = lesson_id or self.current_lesson
        if not lesson_id:
            return None
        
        return self.progress_data["lessons"].get(lesson_id)
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """
        获取整体学习进度
        
        Returns:
            整体进度统计
        """
        lessons = self.progress_data["lessons"]
        
        # 计算统计数据
        total_lessons = len(lessons)
        completed_lessons = sum(
            1 for l in lessons.values() 
            if l.get("status") == ProgressStatus.COMPLETED.value
        )
        
        total_progress = 0.0
        if total_lessons > 0:
            total_progress = sum(
                l.get("progress_percentage", 0) for l in lessons.values()
            ) / total_lessons
        
        # 计算平均会话时长
        avg_session = 0.0
        if self.progress_data["total_sessions"] > 0:
            avg_session = self.progress_data["total_time_minutes"] / self.progress_data["total_sessions"]
        
        return {
            "user_id": self.user_id,
            "total_time_hours": self.progress_data["total_time_minutes"] / 60,
            "total_sessions": self.progress_data["total_sessions"],
            "total_lessons": total_lessons,
            "completed_lessons": completed_lessons,
            "total_exercises_completed": self.progress_data["total_exercises_completed"],
            "overall_progress": total_progress,
            "current_streak_days": self.progress_data["current_streak_days"],
            "longest_streak_days": self.progress_data["longest_streak_days"],
            "average_session_minutes": avg_session,
            "achievements_unlocked": sum(1 for a in self.achievements.values() if a.unlocked),
            "achievements_total": len(self.achievements)
        }
    
    def _update_streak(self):
        """更新学习连续天数"""
        today = datetime.now().date()
        
        if self.progress_data["last_study_date"]:
            last_date = datetime.fromisoformat(
                self.progress_data["last_study_date"]
            ).date()
            
            days_diff = (today - last_date).days
            
            if days_diff == 0:
                # 今天已经学习过
                pass
            elif days_diff == 1:
                # 连续学习
                self.progress_data["current_streak_days"] += 1
            else:
                # 中断了
                self.progress_data["current_streak_days"] = 1
        else:
            # 第一次学习
            self.progress_data["current_streak_days"] = 1
        
        # 更新最长连续天数
        if self.progress_data["current_streak_days"] > self.progress_data["longest_streak_days"]:
            self.progress_data["longest_streak_days"] = self.progress_data["current_streak_days"]
        
        # 更新最后学习日期
        self.progress_data["last_study_date"] = today.isoformat()
        
        # 检查连续学习成就
        if self.progress_data["current_streak_days"] >= 7:
            self._unlock_achievement("week_streak")
    
    def _check_achievement(self, achievement_id: str):
        """
        检查成就进度
        
        Args:
            achievement_id: 成就ID
        """
        if achievement_id not in self.achievements:
            return
        
        achievement = self.achievements[achievement_id]
        
        if achievement_id == "first_lesson":
            achievement.progress = self.progress_data["total_lessons_completed"]
        elif achievement_id == "week_streak":
            achievement.progress = self.progress_data["current_streak_days"]
        elif achievement_id == "exercise_master":
            achievement.progress = self.progress_data["total_exercises_completed"]
        elif achievement_id == "knowledge_seeker":
            achievement.progress = len([
                l for l in self.progress_data["lessons"].values()
                if l.get("status") == ProgressStatus.COMPLETED.value
            ])
        
        # 检查是否达成
        if achievement.progress >= achievement.requirement and not achievement.unlocked:
            self._unlock_achievement(achievement_id)
    
    def _unlock_achievement(self, achievement_id: str):
        """
        解锁成就
        
        Args:
            achievement_id: 成就ID
        """
        if achievement_id not in self.achievements:
            return
        
        achievement = self.achievements[achievement_id]
        if achievement.unlocked:
            return
        
        achievement.unlocked = True
        achievement.unlocked_date = datetime.now()
        achievement.progress = achievement.requirement
        
        # 保存到进度数据
        self.progress_data["achievements"][achievement_id] = {
            "unlocked": True,
            "unlocked_date": achievement.unlocked_date.isoformat(),
            "progress": achievement.progress
        }
        
        self._save_progress()
        
        print(f"🏆 解锁成就: {achievement.icon} {achievement.name}")
        print(f"   {achievement.description}")
    
    def get_achievements(self) -> List[Achievement]:
        """
        获取所有成就
        
        Returns:
            成就列表
        """
        return list(self.achievements.values())
    
    def visualize_progress(self):
        """可视化学习进度（在Jupyter中使用）"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ 可视化功能需要matplotlib和IPython")
            return
        
        overall = self.get_overall_progress()
        
        # 创建进度仪表板
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 整体进度环形图
        ax1 = plt.subplot(2, 3, 1)
        progress = overall["overall_progress"]
        remaining = 100 - progress
        ax1.pie([progress, remaining], 
                labels=['已完成', '待学习'],
                colors=['#4CAF50', '#e0e0e0'],
                autopct='%1.1f%%',
                startangle=90,
                counterclock=False)
        ax1.set_title('整体学习进度', fontsize=14, fontweight='bold')
        
        # 2. 学习时长统计
        ax2 = plt.subplot(2, 3, 2)
        categories = ['总时长\n(小时)', '平均会话\n(分钟)', '总会话数']
        values = [
            overall["total_time_hours"],
            overall["average_session_minutes"],
            overall["total_sessions"]
        ]
        bars = ax2.bar(categories, values, color=['#2196F3', '#FF9800', '#9C27B0'])
        ax2.set_title('学习时长统计', fontsize=14, fontweight='bold')
        ax2.set_ylabel('数值')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}' if isinstance(val, float) else str(val),
                    ha='center', va='bottom')
        
        # 3. 课程完成情况
        ax3 = plt.subplot(2, 3, 3)
        completed = overall["completed_lessons"]
        total = overall["total_lessons"]
        in_progress = total - completed
        
        ax3.barh(['课程'], [completed], label='已完成', color='#4CAF50')
        ax3.barh(['课程'], [in_progress], left=[completed], label='进行中', color='#FFC107')
        ax3.set_xlim(0, max(total, 1))
        ax3.set_title('课程完成情况', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.set_xlabel('课程数量')
        
        # 4. 学习连续性
        ax4 = plt.subplot(2, 3, 4)
        streak_data = {
            '当前连续': overall["current_streak_days"],
            '最长连续': overall["longest_streak_days"]
        }
        ax4.bar(streak_data.keys(), streak_data.values(), color=['#FF5722', '#E91E63'])
        ax4.set_title('学习连续性（天）', fontsize=14, fontweight='bold')
        ax4.set_ylabel('天数')
        
        # 5. 练习完成统计
        ax5 = plt.subplot(2, 3, 5)
        exercises = overall["total_exercises_completed"]
        ax5.text(0.5, 0.5, str(exercises), 
                fontsize=48, fontweight='bold',
                ha='center', va='center', color='#673AB7')
        ax5.text(0.5, 0.2, '练习已完成', 
                fontsize=14, ha='center', va='center')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('练习完成数', fontsize=14, fontweight='bold')
        
        # 6. 成就进度
        ax6 = plt.subplot(2, 3, 6)
        unlocked = overall["achievements_unlocked"]
        total_achievements = overall["achievements_total"]
        locked = total_achievements - unlocked
        
        ax6.pie([unlocked, locked],
                labels=['已解锁', '未解锁'],
                colors=['#FFD700', '#C0C0C0'],
                autopct='%1.0f',
                startangle=90)
        ax6.set_title(f'成就进度 ({unlocked}/{total_achievements})', 
                     fontsize=14, fontweight='bold')
        
        plt.suptitle(f'📊 {self.user_id} 的学习进度仪表板', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # 显示详细成就列表
        self._display_achievements()
    
    def _display_achievements(self):
        """显示成就列表（在Jupyter中使用）"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        html = """
        <div style="margin-top: 20px;">
            <h3>🏆 成就系统</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        """
        
        for achievement in self.achievements.values():
            status_color = "#4CAF50" if achievement.unlocked else "#9E9E9E"
            opacity = "1" if achievement.unlocked else "0.5"
            
            html += f"""
            <div style="border: 2px solid {status_color}; border-radius: 8px; 
                        padding: 10px; width: 200px; opacity: {opacity};">
                <div style="font-size: 24px; text-align: center;">{achievement.icon}</div>
                <div style="font-weight: bold; text-align: center; margin: 5px 0;">
                    {achievement.name}
                </div>
                <div style="font-size: 12px; text-align: center; color: #666;">
                    {achievement.description}
                </div>
                <div style="margin-top: 8px;">
                    <div style="background-color: #e0e0e0; border-radius: 4px; height: 8px;">
                        <div style="background-color: {status_color}; height: 8px; 
                                    border-radius: 4px; width: {achievement.progress/achievement.requirement*100:.0f}%;">
                        </div>
                    </div>
                    <div style="font-size: 10px; text-align: center; margin-top: 2px;">
                        {achievement.progress:.0f}/{achievement.requirement:.0f}
                    </div>
                </div>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        display(HTML(html))
    
    def export_report(self, filename: str = None) -> str:
        """
        导出学习报告
        
        Args:
            filename: 导出文件名
            
        Returns:
            报告内容
        """
        overall = self.get_overall_progress()
        
        report = f"""
# 📊 学习进度报告

**用户**: {self.user_id}  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 整体统计

- **总学习时长**: {overall['total_time_hours']:.1f} 小时
- **总会话数**: {overall['total_sessions']} 次
- **平均会话时长**: {overall['average_session_minutes']:.1f} 分钟
- **课程完成**: {overall['completed_lessons']}/{overall['total_lessons']}
- **练习完成**: {overall['total_exercises_completed']} 个
- **整体进度**: {overall['overall_progress']:.1f}%
- **当前连续学习**: {overall['current_streak_days']} 天
- **最长连续学习**: {overall['longest_streak_days']} 天
- **成就解锁**: {overall['achievements_unlocked']}/{overall['achievements_total']}

## 📚 课程详情

"""
        
        # 添加课程详情
        for lesson_id, lesson in self.progress_data["lessons"].items():
            status_icon = "✅" if lesson["status"] == ProgressStatus.COMPLETED.value else "🔄"
            report += f"""
### {status_icon} {lesson['lesson_name']}
- 进度: {lesson['progress_percentage']:.1f}%
- 章节: {lesson['completed_sections']}/{lesson['total_sections']}
- 练习: {lesson['completed_exercises']}/{lesson['total_exercises']}
- 学习时长: {lesson['total_time_minutes']:.1f} 分钟
- 会话次数: {len(lesson['sessions'])} 次
"""
            
            # 添加得分信息
            if lesson['scores']:
                avg_score = sum(lesson['scores'].values()) / len(lesson['scores'])
                report += f"- 平均得分: {avg_score:.1f}/100\n"
        
        # 添加成就列表
        report += "\n## 🏆 成就系统\n\n"
        
        unlocked_achievements = [a for a in self.achievements.values() if a.unlocked]
        locked_achievements = [a for a in self.achievements.values() if not a.unlocked]
        
        if unlocked_achievements:
            report += "### 已解锁\n"
            for achievement in unlocked_achievements:
                unlock_date = achievement.unlocked_date.strftime('%Y-%m-%d') if achievement.unlocked_date else ""
                report += f"- {achievement.icon} **{achievement.name}** - {achievement.description} ({unlock_date})\n"
        
        if locked_achievements:
            report += "\n### 未解锁\n"
            for achievement in locked_achievements:
                progress_pct = achievement.progress / achievement.requirement * 100
                report += f"- {achievement.icon} {achievement.name} - {achievement.description} (进度: {progress_pct:.0f}%)\n"
        
        # 保存到文件
        if filename:
            output_file = self.data_dir / filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 报告已导出: {output_file}")
        
        return report


# 便捷函数
_default_tracker = None

def get_tracker(user_id: str = "default") -> ProgressTracker:
    """获取默认进度追踪器实例"""
    global _default_tracker
    if _default_tracker is None or _default_tracker.user_id != user_id:
        _default_tracker = ProgressTracker(user_id)
    return _default_tracker

def start_lesson(lesson_id: str, lesson_name: str = None):
    """便捷函数：开始课程"""
    tracker = get_tracker()
    tracker.start_lesson(lesson_id, lesson_name)

def end_lesson():
    """便捷函数：结束课程"""
    tracker = get_tracker()
    tracker.end_lesson()

def complete_section(section_name: str):
    """便捷函数：完成章节"""
    tracker = get_tracker()
    tracker.complete_section(section_name)

def complete_exercise(exercise_name: str, score: float = None):
    """便捷函数：完成练习"""
    tracker = get_tracker()
    tracker.complete_exercise(exercise_name, score)

def show_progress():
    """便捷函数：显示进度"""
    tracker = get_tracker()
    tracker.visualize_progress()

def export_report(filename: str = None):
    """便捷函数：导出报告"""
    tracker = get_tracker()
    return tracker.export_report(filename)