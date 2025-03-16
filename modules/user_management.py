import logging
import json
from datetime import datetime
from .supabase_client import get_supabase_client

# Thiết lập logging
logger = logging.getLogger('user_management')

def register_user(email, password, metadata=None):
    """
    Đăng ký người dùng mới
    
    Args:
        email (str): Email của người dùng
        password (str): Mật khẩu của người dùng
        metadata (dict, optional): Thông tin bổ sung về người dùng
    
    Returns:
        dict: Thông tin về người dùng đã đăng ký
    """
    try:
        client = get_supabase_client()
        
        # Đăng ký người dùng mới
        response = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": metadata or {}
            }
        })
        
        # Lưu thông tin người dùng vào bảng users
        if response.user:
            user_data = {
                "id": response.user.id,
                "email": email,
                "created_at": datetime.now().isoformat(),
                "metadata": json.dumps(metadata or {})
            }
            
            client.table("users").insert(user_data).execute()
        
        return {
            "status": "success",
            "user": response.user.id if response.user else None,
            "message": "Đăng ký thành công"
        }
    except Exception as e:
        logger.error(f"Lỗi khi đăng ký người dùng: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def login_user(email, password):
    """
    Đăng nhập người dùng
    
    Args:
        email (str): Email của người dùng
        password (str): Mật khẩu của người dùng
    
    Returns:
        dict: Thông tin phiên đăng nhập
    """
    try:
        client = get_supabase_client()
        
        # Đăng nhập
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        # Cập nhật thời gian đăng nhập gần nhất
        if response.user:
            client.table("users").update({
                "last_login": datetime.now().isoformat()
            }).eq("id", response.user.id).execute()
        
        return {
            "status": "success",
            "session": {
                "access_token": response.session.access_token if response.session else None,
                "expires_at": response.session.expires_at if response.session else None
            },
            "user": response.user.id if response.user else None,
            "message": "Đăng nhập thành công"
        }
    except Exception as e:
        logger.error(f"Lỗi khi đăng nhập: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def logout_user(jwt):
    """
    Đăng xuất người dùng
    
    Args:
        jwt (str): JWT token của phiên đăng nhập
    
    Returns:
        dict: Kết quả đăng xuất
    """
    try:
        client = get_supabase_client()
        client.auth.sign_out()
        
        return {
            "status": "success",
            "message": "Đăng xuất thành công"
        }
    except Exception as e:
        logger.error(f"Lỗi khi đăng xuất: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_user_profile(user_id):
    """
    Lấy thông tin hồ sơ người dùng
    
    Args:
        user_id (str): ID của người dùng
    
    Returns:
        dict: Thông tin hồ sơ người dùng
    """
    try:
        client = get_supabase_client()
        
        response = client.table("users").select("*").eq("id", user_id).execute()
        
        if response.data and len(response.data) > 0:
            return {
                "status": "success",
                "profile": response.data[0]
            }
        else:
            return {
                "status": "error",
                "message": "Không tìm thấy người dùng"
            }
    except Exception as e:
        logger.error(f"Lỗi khi lấy hồ sơ người dùng: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def update_user_profile(user_id, profile_data):
    """
    Cập nhật hồ sơ người dùng
    
    Args:
        user_id (str): ID của người dùng
        profile_data (dict): Dữ liệu hồ sơ cần cập nhật
    
    Returns:
        dict: Kết quả cập nhật
    """
    try:
        client = get_supabase_client()
        
        # Cập nhật thông tin người dùng
        response = client.table("users").update(profile_data).eq("id", user_id).execute()
        
        return {
            "status": "success",
            "message": "Cập nhật hồ sơ thành công",
            "profile": response.data[0] if response.data and len(response.data) > 0 else None
        }
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật hồ sơ người dùng: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 