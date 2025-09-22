"""
消息序列化模块

提供消息和响应的序列化/反序列化功能，包括：
- JSON序列化
- 二进制序列化
- 压缩序列化
- 加密序列化
"""

from typing import Any, Dict, Optional, Union
import json
import pickle
import gzip
import base64
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from .protocols import Message, Response, MessageType, Priority

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """序列化异常"""
    pass


class BaseSerializer(ABC):
    """序列化器基类"""
    
    @abstractmethod
    def serialize_message(self, message: Message) -> str:
        """序列化消息"""
        pass
    
    @abstractmethod
    def deserialize_message(self, data: str) -> Message:
        """反序列化消息"""
        pass
    
    @abstractmethod
    def serialize_response(self, response: Response) -> str:
        """序列化响应"""
        pass
    
    @abstractmethod
    def deserialize_response(self, data: str) -> Response:
        """反序列化响应"""
        pass


class JSONSerializer(BaseSerializer):
    """JSON序列化器"""
    
    def __init__(self, ensure_ascii: bool = False, indent: Optional[int] = None):
        self.ensure_ascii = ensure_ascii
        self.indent = indent
    
    def serialize_message(self, message: Message) -> str:
        """序列化消息为JSON"""
        try:
            # 转换为字典
            data = message.to_dict()
            
            # 处理特殊类型
            data = self._prepare_for_json(data)
            
            # 序列化为JSON
            json_str = json.dumps(
                data,
                ensure_ascii=self.ensure_ascii,
                indent=self.indent,
                default=self._json_default
            )
            
            # 添加类型前缀
            return f"MSG:{json_str}"
            
        except Exception as e:
            raise SerializationError(f"消息序列化失败: {e}")
    
    def deserialize_message(self, data: str) -> Message:
        """从JSON反序列化消息"""
        try:
            # 移除类型前缀
            if data.startswith("MSG:"):
                json_str = data[4:]
            else:
                json_str = data
            
            # 解析JSON
            data_dict = json.loads(json_str)
            
            # 后处理
            data_dict = self._process_from_json(data_dict)
            
            # 创建消息对象
            return Message.from_dict(data_dict)
            
        except Exception as e:
            raise SerializationError(f"消息反序列化失败: {e}")
    
    def serialize_response(self, response: Response) -> str:
        """序列化响应为JSON"""
        try:
            # 转换为字典
            data = response.to_dict()
            
            # 处理特殊类型
            data = self._prepare_for_json(data)
            
            # 序列化为JSON
            json_str = json.dumps(
                data,
                ensure_ascii=self.ensure_ascii,
                indent=self.indent,
                default=self._json_default
            )
            
            # 添加类型前缀
            return f"RSP:{json_str}"
            
        except Exception as e:
            raise SerializationError(f"响应序列化失败: {e}")
    
    def deserialize_response(self, data: str) -> Response:
        """从JSON反序列化响应"""
        try:
            # 移除类型前缀
            if data.startswith("RSP:"):
                json_str = data[4:]
            else:
                json_str = data
            
            # 解析JSON
            data_dict = json.loads(json_str)
            
            # 后处理
            data_dict = self._process_from_json(data_dict)
            
            # 创建响应对象
            return Response.from_dict(data_dict)
            
        except Exception as e:
            raise SerializationError(f"响应反序列化失败: {e}")
    
    def _prepare_for_json(self, data: Any) -> Any:
        """为JSON序列化准备数据"""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, (MessageType, Priority)):
            return data.value
        else:
            return data
    
    def _process_from_json(self, data: Any) -> Any:
        """从JSON后处理数据"""
        if isinstance(data, dict):
            return {k: self._process_from_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_from_json(item) for item in data]
        else:
            return data
    
    def _json_default(self, obj: Any) -> Any:
        """JSON序列化默认处理器"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (MessageType, Priority)):
            return obj.value
        else:
            raise TypeError(f"对象 {type(obj)} 不支持JSON序列化")


class PickleSerializer(BaseSerializer):
    """Pickle序列化器（二进制）"""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol
    
    def serialize_message(self, message: Message) -> str:
        """序列化消息为Pickle"""
        try:
            # 使用pickle序列化
            binary_data = pickle.dumps(message, protocol=self.protocol)
            
            # 转换为base64字符串
            b64_data = base64.b64encode(binary_data).decode('ascii')
            
            return f"MSG:{b64_data}"
            
        except Exception as e:
            raise SerializationError(f"消息Pickle序列化失败: {e}")
    
    def deserialize_message(self, data: str) -> Message:
        """从Pickle反序列化消息"""
        try:
            # 移除类型前缀
            if data.startswith("MSG:"):
                b64_data = data[4:]
            else:
                b64_data = data
            
            # 解码base64
            binary_data = base64.b64decode(b64_data.encode('ascii'))
            
            # Pickle反序列化
            message = pickle.loads(binary_data)
            
            if not isinstance(message, Message):
                raise SerializationError("反序列化的对象不是Message类型")
            
            return message
            
        except Exception as e:
            raise SerializationError(f"消息Pickle反序列化失败: {e}")
    
    def serialize_response(self, response: Response) -> str:
        """序列化响应为Pickle"""
        try:
            # 使用pickle序列化
            binary_data = pickle.dumps(response, protocol=self.protocol)
            
            # 转换为base64字符串
            b64_data = base64.b64encode(binary_data).decode('ascii')
            
            return f"RSP:{b64_data}"
            
        except Exception as e:
            raise SerializationError(f"响应Pickle序列化失败: {e}")
    
    def deserialize_response(self, data: str) -> Response:
        """从Pickle反序列化响应"""
        try:
            # 移除类型前缀
            if data.startswith("RSP:"):
                b64_data = data[4:]
            else:
                b64_data = data
            
            # 解码base64
            binary_data = base64.b64decode(b64_data.encode('ascii'))
            
            # Pickle反序列化
            response = pickle.loads(binary_data)
            
            if not isinstance(response, Response):
                raise SerializationError("反序列化的对象不是Response类型")
            
            return response
            
        except Exception as e:
            raise SerializationError(f"响应Pickle反序列化失败: {e}")


class CompressedJSONSerializer(JSONSerializer):
    """压缩JSON序列化器"""
    
    def __init__(self, compression_level: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.compression_level = compression_level
    
    def serialize_message(self, message: Message) -> str:
        """序列化并压缩消息"""
        try:
            # 先用JSON序列化
            json_str = super().serialize_message(message)
            
            # 压缩
            compressed_data = gzip.compress(
                json_str.encode('utf-8'),
                compresslevel=self.compression_level
            )
            
            # 转换为base64
            b64_data = base64.b64encode(compressed_data).decode('ascii')
            
            return f"CMSG:{b64_data}"
            
        except Exception as e:
            raise SerializationError(f"消息压缩序列化失败: {e}")
    
    def deserialize_message(self, data: str) -> Message:
        """解压缩并反序列化消息"""
        try:
            # 移除类型前缀
            if data.startswith("CMSG:"):
                b64_data = data[5:]
            else:
                # 尝试普通JSON反序列化
                return super().deserialize_message(data)
            
            # 解码base64
            compressed_data = base64.b64decode(b64_data.encode('ascii'))
            
            # 解压缩
            json_str = gzip.decompress(compressed_data).decode('utf-8')
            
            # JSON反序列化
            return super().deserialize_message(json_str)
            
        except Exception as e:
            raise SerializationError(f"消息解压缩反序列化失败: {e}")
    
    def serialize_response(self, response: Response) -> str:
        """序列化并压缩响应"""
        try:
            # 先用JSON序列化
            json_str = super().serialize_response(response)
            
            # 压缩
            compressed_data = gzip.compress(
                json_str.encode('utf-8'),
                compresslevel=self.compression_level
            )
            
            # 转换为base64
            b64_data = base64.b64encode(compressed_data).decode('ascii')
            
            return f"CRSP:{b64_data}"
            
        except Exception as e:
            raise SerializationError(f"响应压缩序列化失败: {e}")
    
    def deserialize_response(self, data: str) -> Response:
        """解压缩并反序列化响应"""
        try:
            # 移除类型前缀
            if data.startswith("CRSP:"):
                b64_data = data[5:]
            else:
                # 尝试普通JSON反序列化
                return super().deserialize_response(data)
            
            # 解码base64
            compressed_data = base64.b64decode(b64_data.encode('ascii'))
            
            # 解压缩
            json_str = gzip.decompress(compressed_data).decode('utf-8')
            
            # JSON反序列化
            return super().deserialize_response(json_str)
            
        except Exception as e:
            raise SerializationError(f"响应解压缩反序列化失败: {e}")


class EncryptedSerializer(BaseSerializer):
    """加密序列化器"""
    
    def __init__(self, encryption_key: bytes, base_serializer: BaseSerializer = None):
        self.encryption_key = encryption_key
        self.base_serializer = base_serializer or JSONSerializer()
        
        # 验证密钥长度
        if len(encryption_key) not in [16, 24, 32]:
            raise ValueError("加密密钥长度必须是16、24或32字节")
    
    def _encrypt(self, data: str) -> str:
        """加密数据"""
        try:
            from cryptography.fernet import Fernet
            import hashlib
            
            # 使用提供的密钥派生Fernet密钥
            fernet_key = base64.urlsafe_b64encode(
                hashlib.sha256(self.encryption_key).digest()
            )
            fernet = Fernet(fernet_key)
            
            # 加密数据
            encrypted_data = fernet.encrypt(data.encode('utf-8'))
            
            # 转换为base64字符串
            return base64.b64encode(encrypted_data).decode('ascii')
            
        except ImportError:
            raise SerializationError("加密功能需要安装cryptography库")
        except Exception as e:
            raise SerializationError(f"加密失败: {e}")
    
    def _decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            from cryptography.fernet import Fernet
            import hashlib
            
            # 使用提供的密钥派生Fernet密钥
            fernet_key = base64.urlsafe_b64encode(
                hashlib.sha256(self.encryption_key).digest()
            )
            fernet = Fernet(fernet_key)
            
            # 解码base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))
            
            # 解密数据
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return decrypted_data.decode('utf-8')
            
        except ImportError:
            raise SerializationError("解密功能需要安装cryptography库")
        except Exception as e:
            raise SerializationError(f"解密失败: {e}")
    
    def serialize_message(self, message: Message) -> str:
        """序列化并加密消息"""
        try:
            # 先用基础序列化器序列化
            serialized_data = self.base_serializer.serialize_message(message)
            
            # 加密
            encrypted_data = self._encrypt(serialized_data)
            
            return f"EMSG:{encrypted_data}"
            
        except Exception as e:
            raise SerializationError(f"消息加密序列化失败: {e}")
    
    def deserialize_message(self, data: str) -> Message:
        """解密并反序列化消息"""
        try:
            # 移除类型前缀
            if data.startswith("EMSG:"):
                encrypted_data = data[5:]
            else:
                raise SerializationError("不是加密消息格式")
            
            # 解密
            decrypted_data = self._decrypt(encrypted_data)
            
            # 基础反序列化
            return self.base_serializer.deserialize_message(decrypted_data)
            
        except Exception as e:
            raise SerializationError(f"消息解密反序列化失败: {e}")
    
    def serialize_response(self, response: Response) -> str:
        """序列化并加密响应"""
        try:
            # 先用基础序列化器序列化
            serialized_data = self.base_serializer.serialize_response(response)
            
            # 加密
            encrypted_data = self._encrypt(serialized_data)
            
            return f"ERSP:{encrypted_data}"
            
        except Exception as e:
            raise SerializationError(f"响应加密序列化失败: {e}")
    
    def deserialize_response(self, data: str) -> Response:
        """解密并反序列化响应"""
        try:
            # 移除类型前缀
            if data.startswith("ERSP:"):
                encrypted_data = data[5:]
            else:
                raise SerializationError("不是加密响应格式")
            
            # 解密
            decrypted_data = self._decrypt(encrypted_data)
            
            # 基础反序列化
            return self.base_serializer.deserialize_response(decrypted_data)
            
        except Exception as e:
            raise SerializationError(f"响应解密反序列化失败: {e}")


class MessageSerializer:
    """消息序列化器管理器"""
    
    def __init__(self, default_type: str = "json"):
        self.serializers: Dict[str, BaseSerializer] = {
            "json": JSONSerializer(),
            "pickle": PickleSerializer(),
            "compressed": CompressedJSONSerializer(),
        }
        self.default_type = default_type
    
    def add_serializer(self, name: str, serializer: BaseSerializer) -> None:
        """添加序列化器"""
        self.serializers[name] = serializer
        logger.info(f"添加序列化器: {name}")
    
    def remove_serializer(self, name: str) -> bool:
        """移除序列化器"""
        if name in self.serializers:
            del self.serializers[name]
            logger.info(f"移除序列化器: {name}")
            return True
        return False
    
    def get_serializer(self, name: str = None) -> BaseSerializer:
        """获取序列化器"""
        serializer_name = name or self.default_type
        if serializer_name not in self.serializers:
            raise ValueError(f"序列化器 {serializer_name} 不存在")
        return self.serializers[serializer_name]
    
    def serialize_message(self, message: Message, serializer_type: str = None) -> str:
        """序列化消息"""
        serializer = self.get_serializer(serializer_type)
        return serializer.serialize_message(message)
    
    def deserialize_message(self, data: str) -> Message:
        """反序列化消息（自动检测类型）"""
        # 根据前缀自动检测序列化类型
        if data.startswith("MSG:"):
            return self.serializers["json"].deserialize_message(data)
        elif data.startswith("CMSG:"):
            return self.serializers["compressed"].deserialize_message(data)
        elif data.startswith("EMSG:"):
            # 需要预先配置加密序列化器
            if "encrypted" in self.serializers:
                return self.serializers["encrypted"].deserialize_message(data)
            else:
                raise SerializationError("未配置加密序列化器")
        else:
            # 尝试默认序列化器
            return self.get_serializer().deserialize_message(data)
    
    def serialize_response(self, response: Response, serializer_type: str = None) -> str:
        """序列化响应"""
        serializer = self.get_serializer(serializer_type)
        return serializer.serialize_response(response)
    
    def deserialize_response(self, data: str) -> Response:
        """反序列化响应（自动检测类型）"""
        # 根据前缀自动检测序列化类型
        if data.startswith("RSP:"):
            return self.serializers["json"].deserialize_response(data)
        elif data.startswith("CRSP:"):
            return self.serializers["compressed"].deserialize_response(data)
        elif data.startswith("ERSP:"):
            # 需要预先配置加密序列化器
            if "encrypted" in self.serializers:
                return self.serializers["encrypted"].deserialize_response(data)
            else:
                raise SerializationError("未配置加密序列化器")
        else:
            # 尝试默认序列化器
            return self.get_serializer().deserialize_response(data)
    
    def list_serializers(self) -> List[str]:
        """列出所有序列化器"""
        return list(self.serializers.keys())
    
    def get_serializer_stats(self) -> Dict[str, Any]:
        """获取序列化器统计"""
        return {
            'total_serializers': len(self.serializers),
            'available_types': list(self.serializers.keys()),
            'default_type': self.default_type
        }