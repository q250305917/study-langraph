"""
LangChain学习项目的工具函数库

本模块提供项目中常用的工具函数，包括：
- 文本处理工具 (text_utils)
- 文件操作工具 (file_utils)
- 数据验证工具 (validation_utils)
- 时间处理工具 (time_utils)
- 网络请求工具 (http_utils)
"""

# 导入主要的工具函数，方便使用
from .text_utils import (
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

from .file_utils import (
    ensure_directory,
    safe_read_file,
    safe_write_file,
    read_json_file,
    write_json_file,
    read_yaml_file,
    write_yaml_file,
    read_csv_file,
    write_csv_file,
    get_file_info,
    format_file_size,
    calculate_file_hash,
    find_files,
    copy_file,
    move_file,
    delete_file,
    cleanup_directory,
    create_archive,
    extract_archive,
    TemporaryFile,
    TemporaryDirectory
)

# 版本信息
__version__ = "1.0.0"

# 导出的公共接口
__all__ = [
    # 文本处理
    "clean_text",
    "truncate_text", 
    "split_text",
    "extract_keywords",
    "detect_language",
    "calculate_text_similarity",
    "format_text_for_display",
    "escape_special_chars",
    "generate_text_hash",
    "encode_decode_text",
    "extract_urls",
    "extract_emails",
    "extract_phone_numbers",
    "word_count",
    "generate_summary",
    
    # 文件操作
    "ensure_directory",
    "safe_read_file",
    "safe_write_file",
    "read_json_file",
    "write_json_file",
    "read_yaml_file",
    "write_yaml_file",
    "read_csv_file",
    "write_csv_file",
    "get_file_info",
    "format_file_size",
    "calculate_file_hash",
    "find_files",
    "copy_file",
    "move_file",
    "delete_file",
    "cleanup_directory",
    "create_archive",
    "extract_archive",
    "TemporaryFile",
    "TemporaryDirectory",
]