"""
Module này cung cấp các hàm để kết nối với Supabase
"""
import os
import logging
import time
from dotenv import load_dotenv

# Thiết lập logging
logger = logging.getLogger(__name__)

# Đảm bảo rằng các biến môi trường được nạp từ file .env
load_dotenv()

# Lấy thông tin kết nối từ biến môi trường
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Biến global để lưu trữ kết nối Supabase
supabase = None
last_connection_attempt = 0
connection_retry_interval = 60  # Số giây giữa các lần thử kết nối lại

def get_supabase_client():
    """
    Trả về client Supabase đã được khởi tạo
    
    Returns:
        Client: Đối tượng Supabase client hoặc None nếu không thể kết nối
    """
    global supabase, last_connection_attempt
    
    # Kiểm tra xem chúng ta đã có client chưa
    if supabase is not None:
        return supabase
    
    # Kiểm tra xem đã đủ thời gian để thử kết nối lại
    current_time = time.time()
    if (current_time - last_connection_attempt) < connection_retry_interval:
        if not supabase:
            logger.warning("Đang sử dụng chế độ offline do không thể kết nối Supabase")
            return None
        return supabase
        
    # Cập nhật thời gian thử kết nối
    last_connection_attempt = current_time
    
    # Kiểm tra các biến môi trường
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL hoặc SUPABASE_KEY không được cấu hình")
        raise ValueError("SUPABASE_URL hoặc SUPABASE_KEY không được cấu hình trong biến môi trường. Vui lòng kiểm tra file .env")
    
    try:
        # Import động để tránh circular import
        from supabase import create_client
        
        # Thử kết nối với timeout ngắn
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Kiểm tra kết nối bằng cách thực hiện một truy vấn đơn giản
        try:
            supabase.auth.get_session()
            logger.info("Đã khởi tạo kết nối Supabase thành công")
        except Exception as check_error:
            logger.warning(f"Cảnh báo: Client Supabase đã được tạo nhưng kiểm tra kết nối không thành công: {check_error}")
            # Vẫn trả về client vì có thể lỗi này là do chưa có session
        
        return supabase
    except ImportError as ie:
        logger.error(f"Không thể import thư viện Supabase: {ie}")
        raise ImportError(f"Không thể import thư viện Supabase. Vui lòng cài đặt với lệnh 'pip install supabase'")
    except Exception as e:
        logger.error(f"Lỗi khi kết nối đến Supabase: {e}")
        supabase = None
        
        # Phân tích lỗi để đưa ra thông báo cụ thể hơn
        error_message = str(e).lower()
        if "timeout" in error_message or "timed out" in error_message:
            raise ConnectionError(f"Kết nối đến Supabase bị timeout. Vui lòng kiểm tra kết nối mạng và URL Supabase.")
        elif "invalid key" in error_message or "invalid token" in error_message:
            raise ValueError(f"SUPABASE_KEY không hợp lệ. Vui lòng kiểm tra lại khóa API.")
        elif "not found" in error_message or "404" in error_message:
            raise ConnectionError(f"URL Supabase không tìm thấy. Vui lòng kiểm tra lại SUPABASE_URL.")
        else:
            raise ConnectionError(f"Không thể kết nối đến Supabase: {e}")

def test_connection():
    """
    Kiểm tra kết nối đến Supabase
    
    Returns:
        tuple: (bool, dict) - True/False nếu kết nối thành công và thông tin chi tiết về kết nối
    """
    try:
        client = get_supabase_client()
        if not client:
            return False, {"error": "Không thể khởi tạo client Supabase"}
            
        # Thực hiện một truy vấn đơn giản để kiểm tra kết nối
        try:
            result = client.auth.get_user()
            logger.info("Đã kết nối thành công đến Supabase")
            return True, {"status": "connected"}
        except Exception as auth_error:
            # Thử một loại truy vấn khác nếu auth không hoạt động
            try:
                # Thử truy vấn đến một bảng công khai nếu có
                test_result = client.table('test').select('*').limit(1).execute()
                logger.info("Đã kết nối thành công đến Supabase (kiểm tra bảng)")
                return True, {"status": "connected_db_only"}
            except Exception as db_error:
                logger.error(f"Lỗi kết nối database: {db_error}")
                return False, {"error": f"Lỗi xác thực và database: {auth_error} | {db_error}"}
    except Exception as e:
        logger.error(f"Lỗi kết nối: {e}")
        return False, {"error": str(e)} 