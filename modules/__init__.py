"""
Các module tích hợp Supabase cho hệ thống RAG
"""

from .supabase_client import get_supabase_client, check_connection
from .user_management import (
    register_user,
    login_user,
    logout_user,
    get_user_profile,
    update_user_profile
)
from .document_storage import (
    upload_document,
    get_document,
    get_user_documents,
    delete_document,
    update_document_metadata
)
from .vector_store import (
    create_embeddings_table,
    store_embeddings,
    search_similar_vectors,
    delete_document_embeddings,
    get_document_embeddings
)
from .query_history import (
    create_query_history_table,
    save_query,
    update_query_response,
    save_feedback,
    get_user_query_history,
    get_query_details,
    delete_query
)

__all__ = [
    # Supabase client
    'get_supabase_client',
    'check_connection',
    
    # User management
    'register_user',
    'login_user',
    'logout_user',
    'get_user_profile',
    'update_user_profile',
    
    # Document storage
    'upload_document',
    'get_document',
    'get_user_documents',
    'delete_document',
    'update_document_metadata',
    
    # Vector store
    'create_embeddings_table',
    'store_embeddings',
    'search_similar_vectors',
    'delete_document_embeddings',
    'get_document_embeddings',
    
    # Query history
    'create_query_history_table',
    'save_query',
    'update_query_response',
    'save_feedback',
    'get_user_query_history',
    'get_query_details',
    'delete_query'
] 