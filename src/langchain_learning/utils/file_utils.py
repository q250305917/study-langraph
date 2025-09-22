"""
LangChain学习项目的文件操作工具模块

本模块提供常用的文件和目录操作功能，包括：
- 文件读写和编码处理
- 目录操作和路径处理
- 文件格式检测和转换
- 批量文件操作
- 临时文件管理
- 文件监控和同步
"""

import os
import shutil
import tempfile
import json
import yaml
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Callable
import mimetypes
import hashlib
from datetime import datetime
import zipfile
import tarfile

from ..core.logger import get_logger
from ..core.exceptions import ResourceError, ValidationError, ErrorCodes

logger = get_logger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def safe_read_file(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    fallback_encodings: Optional[List[str]] = None
) -> str:
    """
    安全读取文件，自动处理编码问题
    
    Args:
        file_path: 文件路径
        encoding: 主要编码
        fallback_encodings: 备用编码列表
        
    Returns:
        文件内容
        
    Raises:
        ResourceError: 文件读取失败
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ResourceError(
            f"File not found: {file_path}",
            error_code=ErrorCodes.FILE_NOT_FOUND,
            context={"file_path": str(file_path)}
        )
    
    if fallback_encodings is None:
        fallback_encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    
    # 尝试不同编码
    encodings_to_try = [encoding] + [enc for enc in fallback_encodings if enc != encoding]
    
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            logger.debug(f"Successfully read file with encoding {enc}: {file_path}")
            return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ResourceError(
                f"Failed to read file: {e}",
                error_code=ErrorCodes.FILE_NOT_FOUND,
                cause=e
            )
    
    raise ResourceError(
        f"Could not decode file with any of the attempted encodings: {encodings_to_try}",
        error_code=ErrorCodes.VALIDATION_FORMAT_ERROR,
        context={"file_path": str(file_path), "encodings": encodings_to_try}
    )


def safe_write_file(
    file_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    backup: bool = True
) -> None:
    """
    安全写入文件，支持备份
    
    Args:
        file_path: 文件路径
        content: 文件内容
        encoding: 编码
        backup: 是否创建备份
        
    Raises:
        ResourceError: 文件写入失败
    """
    file_path = Path(file_path)
    
    # 确保父目录存在
    ensure_directory(file_path.parent)
    
    # 创建备份
    if backup and file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        logger.debug(f"Successfully wrote file: {file_path}")
    except Exception as e:
        raise ResourceError(
            f"Failed to write file: {e}",
            error_code=ErrorCodes.PERMISSION_DENIED,
            cause=e
        )


def read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        JSON数据
        
    Raises:
        ValidationError: JSON格式错误
    """
    try:
        content = safe_read_file(file_path)
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON format: {e}",
            error_code=ErrorCodes.VALIDATION_FORMAT_ERROR,
            context={"file_path": str(file_path)},
            cause=e
        )


def write_json_file(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    写入JSON文件
    
    Args:
        file_path: 文件路径
        data: JSON数据
        indent: 缩进
        ensure_ascii: 是否确保ASCII编码
    """
    content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
    safe_write_file(file_path, content)


def read_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取YAML文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        YAML数据
        
    Raises:
        ValidationError: YAML格式错误
    """
    try:
        content = safe_read_file(file_path)
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValidationError(
            f"Invalid YAML format: {e}",
            error_code=ErrorCodes.VALIDATION_FORMAT_ERROR,
            context={"file_path": str(file_path)},
            cause=e
        )


def write_yaml_file(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    default_flow_style: bool = False
) -> None:
    """
    写入YAML文件
    
    Args:
        file_path: 文件路径
        data: YAML数据
        default_flow_style: 是否使用流式风格
    """
    content = yaml.dump(data, allow_unicode=True, default_flow_style=default_flow_style)
    safe_write_file(file_path, content)


def read_csv_file(
    file_path: Union[str, Path],
    delimiter: str = ",",
    encoding: str = "utf-8"
) -> List[Dict[str, Any]]:
    """
    读取CSV文件
    
    Args:
        file_path: 文件路径
        delimiter: 分隔符
        encoding: 编码
        
    Returns:
        CSV数据列表
    """
    file_path = Path(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)
    except Exception as e:
        raise ResourceError(
            f"Failed to read CSV file: {e}",
            error_code=ErrorCodes.FILE_NOT_FOUND,
            cause=e
        )


def write_csv_file(
    file_path: Union[str, Path],
    data: List[Dict[str, Any]],
    delimiter: str = ",",
    encoding: str = "utf-8"
) -> None:
    """
    写入CSV文件
    
    Args:
        file_path: 文件路径
        data: CSV数据
        delimiter: 分隔符
        encoding: 编码
    """
    if not data:
        return
    
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    try:
        with open(file_path, 'w', encoding=encoding, newline='') as f:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(data)
        logger.debug(f"Successfully wrote CSV file: {file_path}")
    except Exception as e:
        raise ResourceError(
            f"Failed to write CSV file: {e}",
            error_code=ErrorCodes.PERMISSION_DENIED,
            cause=e
        )


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件信息字典
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ResourceError(
            f"File not found: {file_path}",
            error_code=ErrorCodes.FILE_NOT_FOUND
        )
    
    stat = file_path.stat()
    
    return {
        "name": file_path.name,
        "path": str(file_path.absolute()),
        "size": stat.st_size,
        "size_human": format_file_size(stat.st_size),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "accessed": datetime.fromtimestamp(stat.st_atime),
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "suffix": file_path.suffix,
        "mime_type": mimetypes.guess_type(str(file_path))[0],
        "permissions": oct(stat.st_mode)[-3:]
    }


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        人类可读的文件大小
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法
        
    Returns:
        哈希值
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ResourceError(
            f"File not found: {file_path}",
            error_code=ErrorCodes.FILE_NOT_FOUND
        )
    
    hash_algo = getattr(hashlib, algorithm.lower(), None)
    if not hash_algo:
        raise ValidationError(
            f"Unsupported hash algorithm: {algorithm}",
            error_code=ErrorCodes.VALIDATION_FORMAT_ERROR
        )
    
    hasher = hash_algo()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
    include_dirs: bool = False
) -> List[Path]:
    """
    查找文件
    
    Args:
        directory: 搜索目录
        pattern: 文件模式
        recursive: 是否递归搜索
        include_dirs: 是否包含目录
        
    Returns:
        文件路径列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise ResourceError(
            f"Directory not found: {directory}",
            error_code=ErrorCodes.FILE_NOT_FOUND
        )
    
    if recursive:
        glob_pattern = f"**/{pattern}"
    else:
        glob_pattern = pattern
    
    files = []
    for path in directory.glob(glob_pattern):
        if path.is_file() or (include_dirs and path.is_dir()):
            files.append(path)
    
    return sorted(files)


def copy_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> Path:
    """
    复制文件
    
    Args:
        source: 源文件路径
        destination: 目标路径
        overwrite: 是否覆盖存在的文件
        
    Returns:
        目标文件路径
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise ResourceError(
            f"Source file not found: {source}",
            error_code=ErrorCodes.FILE_NOT_FOUND
        )
    
    if destination.exists() and not overwrite:
        raise ResourceError(
            f"Destination file already exists: {destination}",
            error_code=ErrorCodes.VALIDATION_REQUIRED_ERROR
        )
    
    # 确保目标目录存在
    ensure_directory(destination.parent)
    
    try:
        shutil.copy2(source, destination)
        logger.debug(f"Copied file: {source} -> {destination}")
        return destination
    except Exception as e:
        raise ResourceError(
            f"Failed to copy file: {e}",
            error_code=ErrorCodes.PERMISSION_DENIED,
            cause=e
        )


def move_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> Path:
    """
    移动文件
    
    Args:
        source: 源文件路径
        destination: 目标路径
        overwrite: 是否覆盖存在的文件
        
    Returns:
        目标文件路径
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise ResourceError(
            f"Source file not found: {source}",
            error_code=ErrorCodes.FILE_NOT_FOUND
        )
    
    if destination.exists() and not overwrite:
        raise ResourceError(
            f"Destination file already exists: {destination}",
            error_code=ErrorCodes.VALIDATION_REQUIRED_ERROR
        )
    
    # 确保目标目录存在
    ensure_directory(destination.parent)
    
    try:
        shutil.move(str(source), str(destination))
        logger.debug(f"Moved file: {source} -> {destination}")
        return destination
    except Exception as e:
        raise ResourceError(
            f"Failed to move file: {e}",
            error_code=ErrorCodes.PERMISSION_DENIED,
            cause=e
        )


def delete_file(file_path: Union[str, Path], safe: bool = True) -> bool:
    """
    删除文件
    
    Args:
        file_path: 文件路径
        safe: 是否安全删除（移到回收站或创建备份）
        
    Returns:
        是否删除成功
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    try:
        if safe:
            # 创建备份
            backup_path = file_path.with_suffix(file_path.suffix + f'.deleted_{int(datetime.now().timestamp())}')
            shutil.move(str(file_path), str(backup_path))
            logger.debug(f"Safely deleted file (moved to backup): {file_path} -> {backup_path}")
        else:
            file_path.unlink()
            logger.debug(f"Deleted file: {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def cleanup_directory(
    directory: Union[str, Path],
    keep_recent: int = 10,
    file_pattern: str = "*",
    dry_run: bool = False
) -> List[Path]:
    """
    清理目录，保留最近的文件
    
    Args:
        directory: 目录路径
        keep_recent: 保留最近的文件数量
        file_pattern: 文件模式
        dry_run: 是否只是预览，不实际删除
        
    Returns:
        删除的文件列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    # 获取所有文件并按修改时间排序
    files = find_files(directory, file_pattern, recursive=False)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    # 确定要删除的文件
    files_to_delete = files[keep_recent:] if len(files) > keep_recent else []
    
    if dry_run:
        logger.info(f"Dry run: would delete {len(files_to_delete)} files")
        return files_to_delete
    
    # 删除文件
    deleted_files = []
    for file_path in files_to_delete:
        if delete_file(file_path, safe=True):
            deleted_files.append(file_path)
    
    logger.info(f"Cleaned up directory {directory}: deleted {len(deleted_files)} files")
    return deleted_files


def create_archive(
    source_path: Union[str, Path],
    archive_path: Union[str, Path],
    format: str = "zip",
    compression: bool = True
) -> Path:
    """
    创建压缩包
    
    Args:
        source_path: 源路径
        archive_path: 压缩包路径
        format: 压缩格式 (zip, tar, tar.gz)
        compression: 是否压缩
        
    Returns:
        压缩包路径
    """
    source_path = Path(source_path)
    archive_path = Path(archive_path)
    
    if not source_path.exists():
        raise ResourceError(
            f"Source path not found: {source_path}",
            error_code=ErrorCodes.FILE_NOT_FOUND
        )
    
    ensure_directory(archive_path.parent)
    
    try:
        if format == "zip":
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED if compression else zipfile.ZIP_STORED) as zf:
                if source_path.is_file():
                    zf.write(source_path, source_path.name)
                else:
                    for file_path in source_path.rglob('*'):
                        if file_path.is_file():
                            zf.write(file_path, file_path.relative_to(source_path))
        
        elif format in ["tar", "tar.gz"]:
            mode = "w:gz" if format == "tar.gz" else "w"
            with tarfile.open(archive_path, mode) as tf:
                tf.add(source_path, arcname=source_path.name)
        
        else:
            raise ValidationError(
                f"Unsupported archive format: {format}",
                error_code=ErrorCodes.VALIDATION_FORMAT_ERROR
            )
        
        logger.debug(f"Created archive: {archive_path}")
        return archive_path
        
    except Exception as e:
        raise ResourceError(
            f"Failed to create archive: {e}",
            error_code=ErrorCodes.PERMISSION_DENIED,
            cause=e
        )


def extract_archive(
    archive_path: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> Path:
    """
    解压缩包
    
    Args:
        archive_path: 压缩包路径
        destination: 解压目标路径
        overwrite: 是否覆盖存在的文件
        
    Returns:
        解压目标路径
    """
    archive_path = Path(archive_path)
    destination = Path(destination)
    
    if not archive_path.exists():
        raise ResourceError(
            f"Archive not found: {archive_path}",
            error_code=ErrorCodes.FILE_NOT_FOUND
        )
    
    if destination.exists() and not overwrite:
        raise ResourceError(
            f"Destination already exists: {destination}",
            error_code=ErrorCodes.VALIDATION_REQUIRED_ERROR
        )
    
    ensure_directory(destination)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(destination)
        
        elif archive_path.suffix in ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(destination)
        
        else:
            raise ValidationError(
                f"Unsupported archive format: {archive_path.suffix}",
                error_code=ErrorCodes.VALIDATION_FORMAT_ERROR
            )
        
        logger.debug(f"Extracted archive: {archive_path} -> {destination}")
        return destination
        
    except Exception as e:
        raise ResourceError(
            f"Failed to extract archive: {e}",
            error_code=ErrorCodes.PERMISSION_DENIED,
            cause=e
        )


class TemporaryFile:
    """临时文件管理器"""
    
    def __init__(self, suffix: str = "", prefix: str = "tmp", directory: Optional[str] = None):
        """
        初始化临时文件
        
        Args:
            suffix: 文件后缀
            prefix: 文件前缀
            directory: 临时目录
        """
        self.suffix = suffix
        self.prefix = prefix
        self.directory = directory
        self._file = None
        self._path = None
    
    def __enter__(self) -> Path:
        """进入上下文管理器"""
        self._file = tempfile.NamedTemporaryFile(
            suffix=self.suffix,
            prefix=self.prefix,
            dir=self.directory,
            delete=False
        )
        self._path = Path(self._file.name)
        self._file.close()
        return self._path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self._path and self._path.exists():
            self._path.unlink()


class TemporaryDirectory:
    """临时目录管理器"""
    
    def __init__(self, prefix: str = "tmp", directory: Optional[str] = None):
        """
        初始化临时目录
        
        Args:
            prefix: 目录前缀
            directory: 父目录
        """
        self.prefix = prefix
        self.directory = directory
        self._temp_dir = None
        self._path = None
    
    def __enter__(self) -> Path:
        """进入上下文管理器"""
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix=self.prefix,
            dir=self.directory
        )
        self._path = Path(self._temp_dir.name)
        return self._path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self._temp_dir:
            self._temp_dir.cleanup()