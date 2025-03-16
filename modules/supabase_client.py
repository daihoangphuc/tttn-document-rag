import os
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

# Thiết lập logging
logger = logging.getLogger('supabase_client')

# Tải biến môi trường
load_dotenv()

# Lấy thông tin kết nối Supabase từ biến môi trường
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Biến toàn cục để lưu trữ client
_supabase_client = None

def get_supabase_client() -> Client:
    """
    Lấy hoặc khởi tạo Supabase client
    
    Returns:
        Client: Supabase client đã khởi tạo
    """
    global _supabase_client
    
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL và SUPABASE_KEY phải được cấu hình trong file .env")
        
        try:
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Đã khởi tạo kết nối với Supabase thành công")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo kết nối với Supabase: {str(e)}")
            raise
    
    return _supabase_client

def check_connection():
    """
    Kiểm tra kết nối với Supabase
    
    Returns:
        bool: True nếu kết nối thành công, False nếu không
    """
    try:
        client = get_supabase_client()
        # Chỉ kiểm tra xem client có được khởi tạo thành công không
        if client:
            logger.info("Kết nối Supabase thành công")
            return True
        return False
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra kết nối với Supabase: {str(e)}")
        return False 