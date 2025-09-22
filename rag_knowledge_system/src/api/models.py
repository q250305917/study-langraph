"""
API数据模型

定义API接口的请求和响应数据模型，使用Pydantic进行数据验证。
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    content: str = Field(..., description="文档内容")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="文档元数据")
    filename: Optional[str] = Field(None, description="文件名")
    content_type: Optional[str] = Field(None, description="内容类型")


class DocumentBatchUploadRequest(BaseModel):
    """批量文档上传请求"""
    documents: List[DocumentUploadRequest] = Field(..., description="文档列表")
    collection_name: Optional[str] = Field(None, description="集合名称")


class DocumentResponse(BaseModel):
    """文档响应"""
    id: str = Field(..., description="文档ID")
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    created_time: datetime = Field(..., description="创建时间")
    updated_time: datetime = Field(..., description="更新时间")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., min_length=1, description="搜索查询")
    k: int = Field(default=5, ge=1, le=50, description="返回结果数量")
    search_type: str = Field(default="similarity", description="搜索类型")
    include_scores: bool = Field(default=False, description="是否包含分数")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    
    @validator('search_type')
    def validate_search_type(cls, v):
        allowed_types = ['similarity', 'mmr', 'hybrid']
        if v not in allowed_types:
            raise ValueError(f"搜索类型必须是: {', '.join(allowed_types)}")
        return v


class SearchResponse(BaseModel):
    """搜索响应"""
    query: str = Field(..., description="搜索查询")
    results: List[DocumentResponse] = Field(..., description="搜索结果")
    scores: Optional[List[float]] = Field(None, description="相关性分数")
    total_results: int = Field(..., description="结果总数")
    search_time: float = Field(..., description="搜索耗时（秒）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="搜索元数据")


class RAGRequest(BaseModel):
    """RAG查询请求"""
    question: str = Field(..., min_length=1, description="用户问题")
    k: int = Field(default=5, ge=1, le=20, description="检索文档数量")
    include_sources: bool = Field(default=True, description="是否包含来源")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    history: Optional[List[Dict[str, str]]] = Field(None, description="对话历史")
    
    @validator('history')
    def validate_history(cls, v):
        if v:
            for item in v:
                if not isinstance(item, dict) or 'role' not in item or 'content' not in item:
                    raise ValueError("历史记录格式错误，需要包含role和content字段")
                if item['role'] not in ['user', 'assistant']:
                    raise ValueError("role必须是user或assistant")
        return v


class RAGResponse(BaseModel):
    """RAG响应"""
    question: str = Field(..., description="用户问题")
    answer: str = Field(..., description="生成的回答")
    sources: List[DocumentResponse] = Field(default_factory=list, description="参考来源")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    response_time: float = Field(..., description="响应耗时（秒）")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="响应元数据")


class DocumentDeleteRequest(BaseModel):
    """文档删除请求"""
    document_ids: Optional[List[str]] = Field(None, description="文档ID列表")
    filters: Optional[Dict[str, Any]] = Field(None, description="删除过滤条件")
    
    @validator('document_ids', 'filters')
    def validate_delete_params(cls, v, values):
        document_ids = values.get('document_ids')
        filters = values.get('filters')
        
        if not document_ids and not filters:
            raise ValueError("必须提供document_ids或filters")
        return v


class CollectionInfo(BaseModel):
    """集合信息"""
    name: str = Field(..., description="集合名称")
    document_count: int = Field(..., description="文档数量")
    total_size: int = Field(..., description="总大小（字节）")
    created_time: datetime = Field(..., description="创建时间")
    last_updated: datetime = Field(..., description="最后更新时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="集合元数据")


class SystemStatus(BaseModel):
    """系统状态"""
    status: str = Field(..., description="系统状态")
    version: str = Field(..., description="系统版本")
    uptime: float = Field(..., description="运行时间（秒）")
    collections: List[CollectionInfo] = Field(default_factory=list, description="集合列表")
    
    # 性能统计
    total_documents: int = Field(default=0, description="文档总数")
    total_searches: int = Field(default=0, description="搜索总数")
    avg_search_time: float = Field(default=0.0, description="平均搜索时间")
    avg_response_time: float = Field(default=0.0, description="平均响应时间")
    
    # 资源使用
    memory_usage: Dict[str, Any] = Field(default_factory=dict, description="内存使用情况")
    disk_usage: Dict[str, Any] = Field(default_factory=dict, description="磁盘使用情况")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间")


class BulkOperationRequest(BaseModel):
    """批量操作请求"""
    operation: str = Field(..., description="操作类型")
    targets: List[str] = Field(..., description="目标列表")
    parameters: Optional[Dict[str, Any]] = Field(None, description="操作参数")
    
    @validator('operation')
    def validate_operation(cls, v):
        allowed_ops = ['delete', 'update', 'reindex', 'export']
        if v not in allowed_ops:
            raise ValueError(f"操作类型必须是: {', '.join(allowed_ops)}")
        return v


class BulkOperationResponse(BaseModel):
    """批量操作响应"""
    operation: str = Field(..., description="操作类型")
    total_items: int = Field(..., description="总项目数")
    successful_items: int = Field(..., description="成功项目数")
    failed_items: int = Field(..., description="失败项目数")
    processing_time: float = Field(..., description="处理时间")
    errors: List[str] = Field(default_factory=list, description="错误列表")
    results: Optional[Dict[str, Any]] = Field(None, description="操作结果")


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    section: str = Field(..., description="配置节")
    updates: Dict[str, Any] = Field(..., description="更新内容")
    
    @validator('section')
    def validate_section(cls, v):
        allowed_sections = [
            'embedding', 'vectorstore', 'retrieval', 
            'generation', 'api', 'monitoring'
        ]
        if v not in allowed_sections:
            raise ValueError(f"配置节必须是: {', '.join(allowed_sections)}")
        return v


class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="健康状态")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="各项检查结果")
    
    # 系统信息
    system_info: Dict[str, Any] = Field(default_factory=dict, description="系统信息")
    
    # 依赖服务状态
    dependencies: Dict[str, str] = Field(default_factory=dict, description="依赖服务状态")


class MetricsResponse(BaseModel):
    """指标响应"""
    timestamp: datetime = Field(default_factory=datetime.now, description="指标时间")
    
    # 业务指标
    business_metrics: Dict[str, Any] = Field(default_factory=dict, description="业务指标")
    
    # 性能指标
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="性能指标")
    
    # 系统指标
    system_metrics: Dict[str, Any] = Field(default_factory=dict, description="系统指标")


class ExportRequest(BaseModel):
    """导出请求"""
    export_type: str = Field(..., description="导出类型")
    format: str = Field(default="json", description="导出格式")
    filters: Optional[Dict[str, Any]] = Field(None, description="导出过滤条件")
    include_metadata: bool = Field(default=True, description="是否包含元数据")
    
    @validator('export_type')
    def validate_export_type(cls, v):
        allowed_types = ['documents', 'collections', 'statistics', 'logs']
        if v not in allowed_types:
            raise ValueError(f"导出类型必须是: {', '.join(allowed_types)}")
        return v
    
    @validator('format')
    def validate_format(cls, v):
        allowed_formats = ['json', 'csv', 'xlsx', 'xml']
        if v not in allowed_formats:
            raise ValueError(f"导出格式必须是: {', '.join(allowed_formats)}")
        return v


class ImportRequest(BaseModel):
    """导入请求"""
    data_type: str = Field(..., description="数据类型")
    data: Union[List[Dict[str, Any]], str] = Field(..., description="导入数据")
    merge_strategy: str = Field(default="append", description="合并策略")
    validation_level: str = Field(default="strict", description="验证级别")
    
    @validator('data_type')
    def validate_data_type(cls, v):
        allowed_types = ['documents', 'collections', 'configurations']
        if v not in allowed_types:
            raise ValueError(f"数据类型必须是: {', '.join(allowed_types)}")
        return v
    
    @validator('merge_strategy')
    def validate_merge_strategy(cls, v):
        allowed_strategies = ['append', 'replace', 'update', 'skip']
        if v not in allowed_strategies:
            raise ValueError(f"合并策略必须是: {', '.join(allowed_strategies)}")
        return v


# 便捷函数
def create_error_response(error_type: str, 
                         message: str, 
                         details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """创建错误响应"""
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details
    )


def create_success_response(data: Any, 
                          message: str = "操作成功") -> Dict[str, Any]:
    """创建成功响应"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now()
    }