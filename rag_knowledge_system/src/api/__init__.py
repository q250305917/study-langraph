"""
API模块

提供RAG知识库系统的RESTful API接口，包括文档管理、检索查询、
系统监控等功能。
"""

from .app import create_app
from .routes import (
    documents_router,
    search_router,
    management_router,
    health_router
)
from .models import (
    DocumentRequest,
    SearchRequest,
    SearchResponse,
    SystemStatus
)

__all__ = [
    # 应用
    "create_app",
    
    # 路由
    "documents_router",
    "search_router", 
    "management_router",
    "health_router",
    
    # 模型
    "DocumentRequest",
    "SearchRequest",
    "SearchResponse",
    "SystemStatus"
]