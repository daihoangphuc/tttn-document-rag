"""
Cấu hình kết nối Supabase
"""
import os
import logging
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Thiết lập logging
logger = logging.getLogger(__name__)

# Đảm bảo biến môi trường được nạp
load_dotenv()

# Lấy thông tin cấu hình từ biến môi trường
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Khởi tạo client
_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """
    Lấy instance của Supabase client
    
    Returns:
        Client: Supabase client
    """
    global _supabase_client
    
    # Khởi tạo client nếu chưa có
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Thiết lập session token nếu có
    try:
        from flask import session, has_request_context
        if has_request_context() and 'access_token' in session:
            access_token = session.get('access_token')
            refresh_token = session.get('refresh_token')
            if access_token:
                _supabase_client.auth.set_session(access_token, refresh_token)
    except Exception as e:
        logger.warning(f"Không thể thiết lập token cho Supabase client: {str(e)}")
    
    return _supabase_client

def get_supabase() -> Client:
    """
    Lấy instance của Supabase client (alias cho get_supabase_client)
    
    Returns:
        Client: Supabase client
    """
    return get_supabase_client()

def init_app(app):
    """
    Khởi tạo cấu hình Supabase cho ứng dụng Flask
    
    Args:
        app: Ứng dụng Flask
    """
    app.config["SUPABASE_URL"] = SUPABASE_URL
    app.config["SUPABASE_KEY"] = SUPABASE_KEY
    
    # Thử khởi tạo client nếu biến môi trường đã được cấu hình
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            get_supabase_client()
            logger.info("Đã khởi tạo cấu hình Supabase cho ứng dụng")
        except Exception as e:
            logger.error(f"Không thể khởi tạo Supabase: {str(e)}") 