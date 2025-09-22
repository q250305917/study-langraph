"""
LangChain学习项目的文本处理工具模块

本模块提供常用的文本处理功能，包括：
- 字符串操作和格式化
- 文本清洗和预处理
- 文本分割和合并
- 编码转换和检测
- 正则表达式工具
- 多语言文本处理
"""

import re
import html
import unicodedata
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import hashlib
import base64
import urllib.parse

from ..core.logger import get_logger

logger = get_logger(__name__)


def clean_text(
    text: str,
    remove_extra_spaces: bool = True,
    remove_newlines: bool = False,
    normalize_unicode: bool = True,
    unescape_html: bool = True
) -> str:
    """
    清洗文本内容
    
    Args:
        text: 待清洗的文本
        remove_extra_spaces: 是否移除多余空格
        remove_newlines: 是否移除换行符
        normalize_unicode: 是否标准化Unicode字符
        unescape_html: 是否反转义HTML实体
        
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 反转义HTML实体
    if unescape_html:
        text = html.unescape(text)
    
    # Unicode标准化
    if normalize_unicode:
        text = unicodedata.normalize('NFKC', text)
    
    # 移除换行符
    if remove_newlines:
        text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 移除多余空格
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    preserve_words: bool = True
) -> str:
    """
    截断文本到指定长度
    
    Args:
        text: 待截断的文本
        max_length: 最大长度
        suffix: 截断后缀
        preserve_words: 是否保持单词完整性
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    if not preserve_words:
        return text[:max_length - len(suffix)] + suffix
    
    # 在单词边界截断
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    智能分割文本
    
    Args:
        text: 待分割的文本
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        separators: 分隔符列表，按优先级排序
        
    Returns:
        文本块列表
    """
    if not text:
        return []
    
    if separators is None:
        separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']
    
    chunks = []
    current_chunk = ""
    
    # 递归分割函数
    def _split_with_separators(text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= chunk_size:
            return [text] if text else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator not in text:
            return _split_with_separators(text, remaining_separators)
        
        parts = text.split(separator)
        result = []
        current = ""
        
        for part in parts:
            if len(current + separator + part) <= chunk_size:
                current = current + separator + part if current else part
            else:
                if current:
                    result.append(current)
                
                if len(part) > chunk_size:
                    # 递归分割过长的部分
                    sub_chunks = _split_with_separators(part, remaining_separators)
                    result.extend(sub_chunks)
                    current = ""
                else:
                    current = part
        
        if current:
            result.append(current)
        
        return result
    
    # 执行分割
    initial_chunks = _split_with_separators(text, separators)
    
    # 处理重叠
    if chunk_overlap > 0:
        overlapped_chunks = []
        for i, chunk in enumerate(initial_chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # 添加与前一个块的重叠部分
                prev_chunk = initial_chunks[i - 1]
                overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
                overlapped_chunks.append(overlap_text + " " + chunk)
        
        return overlapped_chunks
    
    return initial_chunks


def extract_keywords(
    text: str,
    min_length: int = 3,
    max_length: int = 20,
    exclude_stopwords: bool = True,
    top_k: Optional[int] = None
) -> List[str]:
    """
    提取文本关键词
    
    Args:
        text: 输入文本
        min_length: 最小词长
        max_length: 最大词长
        exclude_stopwords: 是否排除停用词
        top_k: 返回前k个关键词
        
    Returns:
        关键词列表
    """
    # 简单的关键词提取实现
    # 实际项目中可以使用jieba、NLTK等专业库
    
    # 中英文停用词
    stopwords = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with',
        'to', 'for', 'of', 'as', 'by', 'from', 'up', 'into', 'over', 'after',
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好'
    }
    
    # 清洗文本
    cleaned_text = clean_text(text.lower())
    
    # 提取候选词
    # 这里使用简单的正则表达式，实际应用中需要更复杂的分词
    words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]+\b', cleaned_text)
    
    # 过滤词长
    words = [word for word in words if min_length <= len(word) <= max_length]
    
    # 排除停用词
    if exclude_stopwords:
        words = [word for word in words if word not in stopwords]
    
    # 统计词频
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # 按频率排序
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    # 返回关键词
    keywords = [word for word, count in sorted_words]
    
    if top_k:
        keywords = keywords[:top_k]
    
    return keywords


def detect_language(text: str) -> str:
    """
    简单的语言检测
    
    Args:
        text: 输入文本
        
    Returns:
        语言代码 (zh, en, ja, ko等)
    """
    if not text:
        return "unknown"
    
    # 统计不同字符集的比例
    total_chars = len(text)
    if total_chars == 0:
        return "unknown"
    
    # 中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    chinese_ratio = chinese_chars / total_chars
    
    # 日文字符
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    japanese_ratio = japanese_chars / total_chars
    
    # 韩文字符
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    korean_ratio = korean_chars / total_chars
    
    # 英文字符
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    english_ratio = english_chars / total_chars
    
    # 判断语言
    if chinese_ratio > 0.3:
        return "zh"
    elif japanese_ratio > 0.1:
        return "ja"
    elif korean_ratio > 0.1:
        return "ko"
    elif english_ratio > 0.5:
        return "en"
    else:
        return "unknown"


def calculate_text_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """
    计算文本相似度
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
        method: 相似度计算方法 (jaccard, cosine)
        
    Returns:
        相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # 预处理文本
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if method == "jaccard":
        # Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    elif method == "cosine":
        # 简化的余弦相似度
        intersection = len(words1.intersection(words2))
        magnitude1 = len(words1) ** 0.5
        magnitude2 = len(words2) ** 0.5
        return intersection / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0.0
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def format_text_for_display(
    text: str,
    width: int = 80,
    indent: int = 0,
    align: str = "left"
) -> str:
    """
    格式化文本用于显示
    
    Args:
        text: 输入文本
        width: 显示宽度
        indent: 缩进字符数
        align: 对齐方式 (left, center, right)
        
    Returns:
        格式化后的文本
    """
    if not text:
        return ""
    
    # 分割成行
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= width - indent:
            current_line = current_line + " " + word if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # 应用缩进和对齐
    formatted_lines = []
    for line in lines:
        if align == "center":
            line = line.center(width - indent)
        elif align == "right":
            line = line.rjust(width - indent)
        
        formatted_lines.append(" " * indent + line)
    
    return "\n".join(formatted_lines)


def escape_special_chars(text: str, escape_type: str = "regex") -> str:
    """
    转义特殊字符
    
    Args:
        text: 输入文本
        escape_type: 转义类型 (regex, html, url, json)
        
    Returns:
        转义后的文本
    """
    if not text:
        return ""
    
    if escape_type == "regex":
        # 转义正则表达式特殊字符
        return re.escape(text)
    
    elif escape_type == "html":
        # 转义HTML特殊字符
        return html.escape(text)
    
    elif escape_type == "url":
        # URL编码
        return urllib.parse.quote(text)
    
    elif escape_type == "json":
        # 转义JSON特殊字符
        import json
        return json.dumps(text)[1:-1]  # 移除外层引号
    
    else:
        raise ValueError(f"Unsupported escape type: {escape_type}")


def generate_text_hash(text: str, algorithm: str = "md5") -> str:
    """
    生成文本哈希值
    
    Args:
        text: 输入文本
        algorithm: 哈希算法 (md5, sha1, sha256)
        
    Returns:
        哈希值字符串
    """
    if not text:
        return ""
    
    text_bytes = text.encode('utf-8')
    
    if algorithm == "md5":
        return hashlib.md5(text_bytes).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text_bytes).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text_bytes).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def encode_decode_text(
    text: str,
    encoding: str = "base64",
    operation: str = "encode"
) -> str:
    """
    编码或解码文本
    
    Args:
        text: 输入文本
        encoding: 编码方式 (base64, hex)
        operation: 操作类型 (encode, decode)
        
    Returns:
        编码或解码后的文本
    """
    if not text:
        return ""
    
    if encoding == "base64":
        if operation == "encode":
            return base64.b64encode(text.encode('utf-8')).decode('ascii')
        elif operation == "decode":
            return base64.b64decode(text.encode('ascii')).decode('utf-8')
    
    elif encoding == "hex":
        if operation == "encode":
            return text.encode('utf-8').hex()
        elif operation == "decode":
            return bytes.fromhex(text).decode('utf-8')
    
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


def extract_urls(text: str) -> List[str]:
    """
    从文本中提取URL
    
    Args:
        text: 输入文本
        
    Returns:
        URL列表
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    return urls


def extract_emails(text: str) -> List[str]:
    """
    从文本中提取邮箱地址
    
    Args:
        text: 输入文本
        
    Returns:
        邮箱地址列表
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails


def extract_phone_numbers(text: str, country: str = "cn") -> List[str]:
    """
    从文本中提取电话号码
    
    Args:
        text: 输入文本
        country: 国家代码 (cn, us)
        
    Returns:
        电话号码列表
    """
    if country == "cn":
        # 中国手机号码模式
        phone_pattern = r'1[3-9]\d{9}'
    elif country == "us":
        # 美国电话号码模式
        phone_pattern = r'\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})'
    else:
        # 通用模式
        phone_pattern = r'[\+]?[1-9]?[0-9]{7,15}'
    
    phones = re.findall(phone_pattern, text)
    
    # 处理美国电话号码的元组结果
    if country == "us" and phones:
        phones = [f"({area}){prefix}-{number}" for area, prefix, number in phones]
    
    return phones


def word_count(text: str, count_type: str = "words") -> int:
    """
    统计文本词数
    
    Args:
        text: 输入文本
        count_type: 统计类型 (words, characters, characters_no_spaces, lines)
        
    Returns:
        统计数量
    """
    if not text:
        return 0
    
    if count_type == "words":
        return len(re.findall(r'\b\w+\b', text))
    elif count_type == "characters":
        return len(text)
    elif count_type == "characters_no_spaces":
        return len(re.sub(r'\s', '', text))
    elif count_type == "lines":
        return len(text.split('\n'))
    else:
        raise ValueError(f"Unsupported count type: {count_type}")


def generate_summary(text: str, max_sentences: int = 3) -> str:
    """
    生成文本摘要
    
    简单的基于句子重要性的摘要算法。
    
    Args:
        text: 输入文本
        max_sentences: 最大句子数
        
    Returns:
        摘要文本
    """
    if not text:
        return ""
    
    # 分割句子
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    # 简单的句子重要性评分（基于长度和位置）
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # 位置权重（开头和结尾的句子更重要）
        position_weight = 1.0
        if i == 0 or i == len(sentences) - 1:
            position_weight = 1.5
        elif i < len(sentences) * 0.3:
            position_weight = 1.2
        
        # 长度权重（适中长度的句子更重要）
        length_weight = min(len(sentence) / 100, 1.0)
        
        score = length_weight * position_weight
        scored_sentences.append((sentence, score, i))
    
    # 选择得分最高的句子
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    selected_sentences = scored_sentences[:max_sentences]
    
    # 按原顺序排列
    selected_sentences.sort(key=lambda x: x[2])
    
    return '. '.join([s[0] for s in selected_sentences]) + '.'