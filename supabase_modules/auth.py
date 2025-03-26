"""
Xác thực người dùng với Supabase Authentication
"""
import os
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable
from flask import request, redirect, url_for, session, jsonify, flash
from werkzeug.exceptions import Unauthorized
from supabase import create_client, Client
from werkzeug.security import generate_password_hash, check_password_hash

# Thiết lập logging
logger = logging.getLogger(__name__)

# Khởi tạo Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# In thông tin cho debug
logger.info(f"SUPABASE_URL: {supabase_url or 'Not set'}")
logger.info(f"SUPABASE_KEY: {'Set (length ' + str(len(supabase_key)) + ')' if supabase_key else 'Not set'}")

supabase: Client = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        # Kiểm tra kết nối
        test_query = supabase.from_("chats").select("*").limit(1).execute()
        logger.info(f"Đã khởi tạo kết nối Supabase thành công, test có {len(test_query.data or [])} kết quả")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Supabase client: {str(e)}")
        supabase = None
else:
    logger.warning("Thiếu thông tin cấu hình SUPABASE_URL hoặc SUPABASE_KEY")

def register_user(email, password):
    """
    Đăng ký người dùng mới với Supabase
    
    Args:
        email (str): Email người dùng
        password (str): Mật khẩu người dùng
    
    Returns:
        tuple: (bool, str) - (trạng thái thành công, thông báo)
    """
    if not supabase:
        return False, "Chưa cấu hình kết nối Supabase"

    try:
        # Tạo user với Supabase Auth
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if response.user and response.user.id:
            logger.info(f"Đăng ký thành công cho người dùng: {email}")
            return True, "Đăng ký thành công! Vui lòng xác nhận email của bạn."
        else:
            logger.warning(f"Không thể đăng ký người dùng: {email}")
            return False, "Có lỗi xảy ra trong quá trình đăng ký. Vui lòng thử lại sau."
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Lỗi khi đăng ký người dùng {email}: {error_msg}")
        
        if "User already registered" in error_msg:
            return False, "Email này đã được đăng ký. Vui lòng sử dụng email khác hoặc đăng nhập."
        
        return False, f"Lỗi đăng ký: {error_msg}"

def login_user(email, password):
    """
    Đăng nhập người dùng với Supabase
    
    Args:
        email (str): Email người dùng
        password (str): Mật khẩu người dùng
    
    Returns:
        tuple: (bool, str, dict) - (trạng thái thành công, thông báo, user data nếu thành công)
    """
    if not supabase:
        return False, "Chưa cấu hình kết nối Supabase", None

    try:
        # Đăng nhập với Supabase Auth
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user and response.user.id:
            user_data = {
                "id": response.user.id,
                "email": response.user.email,
                "created_at": response.user.created_at
            }
            
            # Lưu session token
            session['access_token'] = response.session.access_token
            session['refresh_token'] = response.session.refresh_token
            session['user_id'] = response.user.id
            session['email'] = response.user.email
            
            logger.info(f"Đăng nhập thành công: {email}")
            return True, "Đăng nhập thành công!", user_data
        else:
            logger.warning(f"Đăng nhập thất bại cho user: {email}")
            return False, "Đăng nhập thất bại. Vui lòng kiểm tra lại thông tin đăng nhập.", None
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Lỗi khi đăng nhập với user {email}: {error_msg}")
        
        if "Invalid login credentials" in error_msg:
            return False, "Email hoặc mật khẩu không đúng.", None
        
        return False, f"Lỗi đăng nhập: {error_msg}", None

def logout_user():
    """
    Đăng xuất người dùng
    
    Returns:
        bool: Trạng thái thành công
    """
    if not supabase:
        return False

    try:
        # Xóa session trên Supabase
        if 'access_token' in session:
            supabase.auth.sign_out()
        
        # Xóa session trên server
        session.pop('access_token', None)
        session.pop('refresh_token', None)
        session.pop('user_id', None)
        session.pop('email', None)
        
        logger.info("Đăng xuất thành công")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi đăng xuất: {str(e)}")
        return False

def verify_session():
    """
    Kiểm tra phiên đăng nhập hiện tại
    
    Returns:
        bool: True nếu phiên hợp lệ, False nếu không
    """
    if not supabase:
        return False

    try:
        if 'access_token' not in session:
            return False
        
        # Sử dụng token để lấy thông tin user hiện tại
        user = supabase.auth.get_user(session['access_token'])
        
        if user and user.user and user.user.id:
            # Cập nhật session nếu cần
            if user.user.id != session.get('user_id'):
                session['user_id'] = user.user.id
                session['email'] = user.user.email
            
            return True
        
        return False
    except Exception as e:
        logger.error(f"Lỗi khi xác thực phiên: {str(e)}")
        return False

def get_current_user():
    """
    Lấy thông tin người dùng hiện tại
    
    Returns:
        dict: Thông tin người dùng hoặc None nếu không đăng nhập
    """
    if not verify_session():
        return None
    
    try:
        user = supabase.auth.get_user(session['access_token'])
        
        if user and user.user:
            return {
                "id": user.user.id,
                "email": user.user.email,
                "created_at": user.user.created_at
            }
        
        return None
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin người dùng hiện tại: {str(e)}")
        return None

def change_password(user_id, current_password, new_password):
    """
    Thay đổi mật khẩu người dùng
    
    Args:
        user_id (str): ID người dùng
        current_password (str): Mật khẩu hiện tại
        new_password (str): Mật khẩu mới
    
    Returns:
        tuple: (bool, str) - (trạng thái thành công, thông báo)
    """
    if not supabase:
        return False, "Chưa cấu hình kết nối Supabase"

    try:
        # Trước tiên xác thực mật khẩu hiện tại bằng cách đăng nhập
        user = get_current_user()
        if not user:
            return False, "Bạn chưa đăng nhập"
        
        try:
            # Kiểm tra mật khẩu hiện tại bằng cách thử đăng nhập
            response = supabase.auth.sign_in_with_password({
                "email": user["email"],
                "password": current_password
            })
            
            if not response.user:
                return False, "Mật khẩu hiện tại không đúng"
        except Exception:
            return False, "Mật khẩu hiện tại không đúng"
        
        # Cập nhật mật khẩu
        supabase.auth.update_user({
            "password": new_password
        })
        
        logger.info(f"Đã đổi mật khẩu thành công cho user ID: {user_id}")
        return True, "Đổi mật khẩu thành công!"
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Lỗi khi đổi mật khẩu cho user ID {user_id}: {error_msg}")
        
        if "Password should be at least" in error_msg:
            return False, "Mật khẩu mới phải có ít nhất 6 ký tự"
        
        return False, f"Lỗi đổi mật khẩu: {error_msg}"

def reset_password_request(email):
    """
    Gửi email đặt lại mật khẩu
    
    Args:
        email (str): Email người dùng
    
    Returns:
        tuple: (bool, str) - (trạng thái thành công, thông báo)
    """
    if not supabase:
        return False, "Chưa cấu hình kết nối Supabase"

    try:
        supabase.auth.reset_password_email(email)
        logger.info(f"Đã gửi email đặt lại mật khẩu cho: {email}")
        return True, "Đã gửi email đặt lại mật khẩu. Vui lòng kiểm tra hộp thư của bạn."
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Lỗi khi gửi email đặt lại mật khẩu cho {email}: {error_msg}")
        return False, f"Lỗi khi gửi email đặt lại mật khẩu: {error_msg}"

def update_user_profile(user_id, data):
    """
    Cập nhật thông tin hồ sơ người dùng
    
    Args:
        user_id (str): ID người dùng
        data (dict): Dữ liệu cần cập nhật, có thể bao gồm các trường như 'email', 'full_name', 'avatar_url', 'phone', v.v.
    
    Returns:
        tuple: (bool, str, dict) - (trạng thái thành công, thông báo, dữ liệu người dùng đã cập nhật)
    """
    if not supabase:
        return False, "Chưa cấu hình kết nối Supabase", None
    
    try:
        # Kiểm tra người dùng hiện tại
        current_user = get_current_user()
        if not current_user or current_user['id'] != user_id:
            return False, "Không có quyền cập nhật thông tin người dùng này", None
        
        update_data = {}
        
        # Kiểm tra và thêm các trường cần cập nhật
        if 'full_name' in data or 'fullname' in data:
            update_data['user_metadata'] = {
                'full_name': data.get('full_name') or data.get('fullname')
            }
        
        if 'phone' in data:
            update_data['phone'] = data['phone']
            
        if 'avatar_url' in data:
            update_data['user_metadata'] = update_data.get('user_metadata', {})
            update_data['user_metadata']['avatar_url'] = data['avatar_url']
        
        # Nếu không có dữ liệu cập nhật, trả về thông báo
        if not update_data:
            return False, "Không có thông tin nào được cập nhật", None
        
        # Thực hiện cập nhật
        try:
            response = supabase.auth.update_user(update_data)
            
            if response and response.user:
                logger.info(f"Đã cập nhật thông tin cho user ID: {user_id}")
                return True, "Cập nhật thông tin thành công", response.user
            
            return False, "Không thể cập nhật thông tin người dùng", None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Lỗi khi cập nhật thông tin người dùng {user_id}: {error_msg}")
            return False, f"Lỗi khi cập nhật thông tin: {error_msg}", None
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Lỗi không xác định khi cập nhật thông tin người dùng {user_id}: {error_msg}")
        return False, f"Lỗi không xác định: {error_msg}", None
        

def require_auth(f: Callable) -> Callable:
    """
    Decorator để yêu cầu xác thực trước khi truy cập route
    
    Args:
        f (Callable): Hàm route cần bảo vệ
        
    Returns:
        Callable: Hàm wrapper đã xử lý xác thực
    """
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        user = get_current_user()
        if not user:
            # Nếu là request AJAX, trả về lỗi 401
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": "Không có quyền truy cập"}), 401
            # Nếu không, chuyển hướng đến trang đăng nhập
            flash("Vui lòng đăng nhập để tiếp tục", "warning")
            return redirect(url_for('login_page'))
        # Gắn người dùng vào request để sử dụng trong hàm xử lý route
        request.user = user
        return f(*args, **kwargs)
    return decorated_function

def setup_auth_routes(app):
    """
    Thiết lập các route liên quan đến xác thực
    
    Args:
        app: Flask app
    """
    
    @app.route('/api/auth/check', methods=['GET'])
    def check_auth_status():
        """
        Kiểm tra trạng thái đăng nhập của người dùng
        """
        user = get_current_user()
        
        if user:
            return jsonify({
                "logged_in": True,
                "user_id": user['id'],
                "email": user['email']
            })
        
        return jsonify({
            "logged_in": False
        }) 