"""
文本处理工具模块的单元测试

测试text_utils中的各种文本处理函数。
"""

import pytest
from langchain_learning.utils.text_utils import (
    clean_text,
    truncate_text,
    split_text,
    extract_keywords,
    detect_language,
    calculate_text_similarity,
    format_text_for_display,
    escape_special_chars,
    generate_text_hash,
    encode_decode_text,
    extract_urls,
    extract_emails,
    extract_phone_numbers,
    word_count,
    generate_summary
)


class TestCleanText:
    """测试文本清洗功能"""
    
    def test_basic_cleaning(self):
        """测试基本清洗"""
        text = "  Hello   World  \n\n  "
        result = clean_text(text)
        assert result == "Hello World"
    
    def test_html_unescape(self):
        """测试HTML实体反转义"""
        text = "&lt;div&gt;Hello &amp; World&lt;/div&gt;"
        result = clean_text(text, unescape_html=True)
        assert "<div>" in result
        assert "&" in result
    
    def test_newline_removal(self):
        """测试换行符移除"""
        text = "Line 1\nLine 2\r\nLine 3"
        result = clean_text(text, remove_newlines=True)
        assert "\n" not in result
        assert "\r" not in result
        assert "Line 1 Line 2 Line 3" == result
    
    def test_unicode_normalization(self):
        """测试Unicode标准化"""
        text = "café"  # 可能包含组合字符
        result = clean_text(text, normalize_unicode=True)
        # 应该标准化为预组合形式
        assert result == "café"
    
    def test_empty_text(self):
        """测试空文本"""
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestTruncateText:
    """测试文本截断功能"""
    
    def test_basic_truncation(self):
        """测试基本截断"""
        text = "This is a long sentence that needs to be truncated"
        result = truncate_text(text, max_length=20)
        assert len(result) <= 20
        assert result.endswith("...")
    
    def test_preserve_words(self):
        """测试保持单词完整性"""
        text = "This is a test sentence"
        result = truncate_text(text, max_length=15, preserve_words=True)
        # 应该在单词边界截断
        assert not result.split()[-1].startswith("...")  # 最后一个词应该是完整的（除了省略号）
    
    def test_no_truncation_needed(self):
        """测试不需要截断的情况"""
        text = "Short text"
        result = truncate_text(text, max_length=20)
        assert result == text
    
    def test_custom_suffix(self):
        """测试自定义后缀"""
        text = "This is a long text"
        result = truncate_text(text, max_length=15, suffix="[...]")
        assert result.endswith("[...]")


class TestSplitText:
    """测试文本分割功能"""
    
    def test_basic_split(self):
        """测试基本分割"""
        text = "This is a test. " * 100  # 创建长文本
        chunks = split_text(text, chunk_size=50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50 or len(chunk.split()) == 1  # 除非是单个长单词
    
    def test_split_by_paragraphs(self):
        """测试按段落分割"""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = split_text(text, chunk_size=15, separators=['\n\n', '. '])
        
        # 应该优先在段落边界分割
        assert len(chunks) >= 3
    
    def test_chunk_overlap(self):
        """测试块重叠"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = split_text(text, chunk_size=30, chunk_overlap=10)
        
        # 检查是否有重叠内容
        if len(chunks) > 1:
            # 第二个块应该包含第一个块的部分内容
            assert any(word in chunks[1] for word in chunks[0].split()[-3:])
    
    def test_empty_text(self):
        """测试空文本"""
        assert split_text("") == []
        assert split_text(None) == []


class TestExtractKeywords:
    """测试关键词提取功能"""
    
    def test_basic_extraction(self):
        """测试基本关键词提取"""
        text = "Python is a powerful programming language. Python is easy to learn."
        keywords = extract_keywords(text, top_k=5)
        
        assert "python" in keywords  # 应该转换为小写
        assert len(keywords) <= 5
    
    def test_exclude_stopwords(self):
        """测试排除停用词"""
        text = "The quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text, exclude_stopwords=True)
        
        # 停用词应该被排除
        assert "the" not in keywords
        assert "over" not in keywords
        # 实际单词应该被保留
        assert "quick" in keywords or "brown" in keywords
    
    def test_length_filtering(self):
        """测试长度过滤"""
        text = "A very long supercalifragilisticexpialidocious word test"
        keywords = extract_keywords(text, min_length=4, max_length=10)
        
        # 太短或太长的词应该被过滤
        for keyword in keywords:
            assert 4 <= len(keyword) <= 10
    
    def test_chinese_text(self):
        """测试中文文本"""
        text = "人工智能是计算机科学的一个分支，人工智能技术发展很快"
        keywords = extract_keywords(text, top_k=3)
        
        assert len(keywords) > 0
        # 应该包含一些中文关键词
        assert any(len(keyword) > 1 for keyword in keywords)


class TestDetectLanguage:
    """测试语言检测功能"""
    
    def test_chinese_detection(self):
        """测试中文检测"""
        text = "这是一段中文文本，用于测试语言检测功能。"
        result = detect_language(text)
        assert result == "zh"
    
    def test_english_detection(self):
        """测试英文检测"""
        text = "This is an English text for language detection testing."
        result = detect_language(text)
        assert result == "en"
    
    def test_japanese_detection(self):
        """测试日文检测"""
        text = "これは日本語のテストです。ひらがなとカタカナが含まれています。"
        result = detect_language(text)
        assert result == "ja"
    
    def test_mixed_language(self):
        """测试混合语言"""
        text = "Hello 你好 world 世界"
        result = detect_language(text)
        # 应该检测出占比较高的语言
        assert result in ["zh", "en"]
    
    def test_empty_text(self):
        """测试空文本"""
        assert detect_language("") == "unknown"
        assert detect_language(None) == "unknown"


class TestCalculateTextSimilarity:
    """测试文本相似度计算"""
    
    def test_identical_texts(self):
        """测试相同文本"""
        text = "This is a test sentence"
        similarity = calculate_text_similarity(text, text)
        assert similarity == 1.0
    
    def test_completely_different_texts(self):
        """测试完全不同的文本"""
        text1 = "apple banana cherry"
        text2 = "dog elephant fox"
        similarity = calculate_text_similarity(text1, text2)
        assert similarity == 0.0
    
    def test_partially_similar_texts(self):
        """测试部分相似的文本"""
        text1 = "the quick brown fox"
        text2 = "the slow brown dog"
        similarity = calculate_text_similarity(text1, text2)
        assert 0 < similarity < 1
    
    def test_jaccard_vs_cosine(self):
        """测试不同相似度算法"""
        text1 = "hello world test"
        text2 = "hello test example"
        
        jaccard = calculate_text_similarity(text1, text2, method="jaccard")
        cosine = calculate_text_similarity(text1, text2, method="cosine")
        
        assert 0 <= jaccard <= 1
        assert 0 <= cosine <= 1
    
    def test_empty_texts(self):
        """测试空文本"""
        assert calculate_text_similarity("", "test") == 0.0
        assert calculate_text_similarity("test", "") == 0.0
        assert calculate_text_similarity("", "") == 0.0


class TestFormatTextForDisplay:
    """测试文本显示格式化"""
    
    def test_basic_formatting(self):
        """测试基本格式化"""
        text = "This is a long sentence that should be wrapped to multiple lines for better display"
        result = format_text_for_display(text, width=20)
        
        lines = result.split('\n')
        for line in lines:
            assert len(line) <= 20
    
    def test_indentation(self):
        """测试缩进"""
        text = "Hello World"
        result = format_text_for_display(text, width=20, indent=4)
        
        lines = result.split('\n')
        for line in lines:
            if line.strip():  # 非空行
                assert line.startswith("    ")
    
    def test_alignment(self):
        """测试对齐"""
        text = "Center"
        result = format_text_for_display(text, width=20, align="center")
        
        # 应该居中对齐
        assert result.strip() == text
        assert len(result) <= 20


class TestEscapeSpecialChars:
    """测试特殊字符转义"""
    
    def test_regex_escape(self):
        """测试正则表达式转义"""
        text = "Hello. (World) [Test] {Regex} ^$*+?|"
        result = escape_special_chars(text, escape_type="regex")
        
        # 特殊字符应该被转义
        assert "\\." in result
        assert "\\(" in result
        assert "\\[" in result
    
    def test_html_escape(self):
        """测试HTML转义"""
        text = "<div>Hello & World</div>"
        result = escape_special_chars(text, escape_type="html")
        
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
    
    def test_url_escape(self):
        """测试URL编码"""
        text = "hello world@example.com"
        result = escape_special_chars(text, escape_type="url")
        
        assert "%20" in result  # 空格应该被编码
        assert "%40" in result  # @应该被编码
    
    def test_json_escape(self):
        """测试JSON转义"""
        text = 'Hello "World" \n Test'
        result = escape_special_chars(text, escape_type="json")
        
        assert '\\"' in result  # 引号应该被转义
        assert '\\n' in result  # 换行符应该被转义


class TestGenerateTextHash:
    """测试文本哈希生成"""
    
    def test_md5_hash(self):
        """测试MD5哈希"""
        text = "Hello World"
        hash_value = generate_text_hash(text, algorithm="md5")
        
        assert len(hash_value) == 32  # MD5哈希长度
        assert hash_value.isalnum()  # 应该是十六进制
    
    def test_sha256_hash(self):
        """测试SHA256哈希"""
        text = "Hello World"
        hash_value = generate_text_hash(text, algorithm="sha256")
        
        assert len(hash_value) == 64  # SHA256哈希长度
    
    def test_consistent_hashing(self):
        """测试哈希一致性"""
        text = "Test consistency"
        hash1 = generate_text_hash(text)
        hash2 = generate_text_hash(text)
        
        assert hash1 == hash2
    
    def test_different_texts_different_hashes(self):
        """测试不同文本产生不同哈希"""
        hash1 = generate_text_hash("Text 1")
        hash2 = generate_text_hash("Text 2")
        
        assert hash1 != hash2


class TestEncodeDecodeText:
    """测试文本编码解码"""
    
    def test_base64_encoding(self):
        """测试Base64编码"""
        text = "Hello World 你好世界"
        encoded = encode_decode_text(text, encoding="base64", operation="encode")
        decoded = encode_decode_text(encoded, encoding="base64", operation="decode")
        
        assert decoded == text
    
    def test_hex_encoding(self):
        """测试十六进制编码"""
        text = "Hello"
        encoded = encode_decode_text(text, encoding="hex", operation="encode")
        decoded = encode_decode_text(encoded, encoding="hex", operation="decode")
        
        assert decoded == text
        assert all(c in "0123456789abcdef" for c in encoded)
    
    def test_unicode_support(self):
        """测试Unicode支持"""
        text = "测试Unicode编码 🌟"
        encoded = encode_decode_text(text, encoding="base64", operation="encode")
        decoded = encode_decode_text(encoded, encoding="base64", operation="decode")
        
        assert decoded == text


class TestExtractUrls:
    """测试URL提取"""
    
    def test_basic_url_extraction(self):
        """测试基本URL提取"""
        text = "Visit https://example.com and http://test.org for more info"
        urls = extract_urls(text)
        
        assert "https://example.com" in urls
        assert "http://test.org" in urls
        assert len(urls) == 2
    
    def test_complex_urls(self):
        """测试复杂URL"""
        text = "Check https://example.com/path?param=value&other=123#section"
        urls = extract_urls(text)
        
        assert len(urls) == 1
        assert "param=value" in urls[0]
        assert "#section" in urls[0]
    
    def test_no_urls(self):
        """测试无URL文本"""
        text = "This text has no URLs in it"
        urls = extract_urls(text)
        
        assert len(urls) == 0


class TestExtractEmails:
    """测试邮箱提取"""
    
    def test_basic_email_extraction(self):
        """测试基本邮箱提取"""
        text = "Contact us at test@example.com or support@company.org"
        emails = extract_emails(text)
        
        assert "test@example.com" in emails
        assert "support@company.org" in emails
        assert len(emails) == 2
    
    def test_complex_emails(self):
        """测试复杂邮箱"""
        text = "Send to user.name+tag@sub.domain.com"
        emails = extract_emails(text)
        
        assert len(emails) == 1
        assert "user.name+tag@sub.domain.com" in emails
    
    def test_no_emails(self):
        """测试无邮箱文本"""
        text = "This text has no email addresses"
        emails = extract_emails(text)
        
        assert len(emails) == 0


class TestExtractPhoneNumbers:
    """测试电话号码提取"""
    
    def test_chinese_phone_numbers(self):
        """测试中国手机号"""
        text = "我的手机号是13812345678，请联系我"
        phones = extract_phone_numbers(text, country="cn")
        
        assert "13812345678" in phones
    
    def test_us_phone_numbers(self):
        """测试美国电话号码"""
        text = "Call me at (555) 123-4567 or 555.987.6543"
        phones = extract_phone_numbers(text, country="us")
        
        assert len(phones) >= 1  # 至少提取到一个
    
    def test_no_phones(self):
        """测试无电话号码文本"""
        text = "This text has no phone numbers"
        phones = extract_phone_numbers(text)
        
        assert len(phones) == 0


class TestWordCount:
    """测试词数统计"""
    
    def test_word_count(self):
        """测试单词计数"""
        text = "This is a test sentence with seven words"
        count = word_count(text, count_type="words")
        assert count == 8
    
    def test_character_count(self):
        """测试字符计数"""
        text = "Hello"
        count = word_count(text, count_type="characters")
        assert count == 5
    
    def test_character_count_no_spaces(self):
        """测试不含空格的字符计数"""
        text = "Hello World"
        count = word_count(text, count_type="characters_no_spaces")
        assert count == 10  # 不包含空格
    
    def test_line_count(self):
        """测试行数统计"""
        text = "Line 1\nLine 2\nLine 3"
        count = word_count(text, count_type="lines")
        assert count == 3


class TestGenerateSummary:
    """测试摘要生成"""
    
    def test_basic_summary(self):
        """测试基本摘要生成"""
        text = (
            "This is the first sentence. "
            "This is the second sentence. "
            "This is the third sentence. "
            "This is the fourth sentence. "
            "This is the fifth sentence."
        )
        summary = generate_summary(text, max_sentences=2)
        
        # 摘要应该包含原文的部分内容
        assert len(summary) < len(text)
        assert "sentence" in summary
    
    def test_short_text_summary(self):
        """测试短文本摘要"""
        text = "This is a short text."
        summary = generate_summary(text, max_sentences=3)
        
        # 短文本应该返回原文
        assert summary == text
    
    def test_empty_text_summary(self):
        """测试空文本摘要"""
        assert generate_summary("") == ""
        assert generate_summary(None) == ""