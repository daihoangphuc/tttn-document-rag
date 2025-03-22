"""
Module này cung cấp các hàm để xác thực người dùng thông qua Supabase
"""
import logging
import os
import functools
from flask import redirect, url_for, session, flash, request
from dotenv import load_dotenv

# Thiết lập logging
logger = logging.getLogger(__name__)

# Khởi tạo biến
supabase = None

def get_auth_client():
    """
    Lấy client Supabase cho auth từ client module
    """
    global supabase
    if supabase is None:
        from .client import get_supabase_client
        try:
            supabase = get_supabase_client()
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo Auth client: {str(e)}")
            return None
    return supabase

# Decorator kiểm tra đăng nhập
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập để sử dụng tính năng này', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def sign_up(email, password, metadata=None):
    """
    Đăng ký người dùng mới
    
    Args:
        email (str): Email người dùng
        password (str): Mật khẩu người dùng
        metadata (dict, optional): Thông tin bổ sung về người dùng
    
    Returns:
        dict: Thông tin đăng ký người dùng với các key: success, user, error
    """
    try:
        auth_client = get_auth_client()
        if not auth_client:
            return {
                'success': False,
                'user': None,
                'error': "Không thể kết nối đến dịch vụ xác thực"
            }
            
        # Chuẩn bị dữ liệu đúng định dạng API Supabase
        signup_data = {
            "email": email,
            "password": password
        }
        
        # Thêm metadata nếu có
        if metadata:
            signup_data["data"] = metadata
        
        # Gọi API đăng ký
        user_data = auth_client.auth.sign_up(signup_data)
        
        # Kiểm tra kết quả trả về
        if not user_data or not user_data.user:
            logger.error(f"Không thể tạo người dùng với email: {email}")
            return {
                'success': False,
                'user': None,
                'error': "Không thể tạo người dùng mới. Vui lòng thử lại sau."
            }
            
        # Chuyển đổi đối tượng User thành dict để dễ truy xuất
        user_dict = {
            'id': user_data.user.id,
            'email': user_data.user.email,
            'created_at': user_data.user.created_at,
            'user_metadata': user_data.user.user_metadata or {}
        }
        
        # Tạo bản ghi người dùng trong database (nếu cần)
        try:
            # Tạo bản ghi người dùng trong bảng profiles nếu cần
            if metadata and 'username' in metadata:
                username = metadata['username']
                # Thêm bản ghi vào bảng profiles
                auth_client.table('profiles').insert({
                    'id': user_dict['id'],
                    'username': username,
                    'email': email,
                    'avatar_url': None,
                    'updated_at': 'now()'
                }).execute()
        except Exception as profile_error:
            logger.warning(f"Lỗi khi tạo profile: {str(profile_error)}, nhưng tài khoản vẫn được tạo")
            # Không trả về lỗi ở đây vì tài khoản đã được tạo
            
        logger.info(f"Đăng ký thành công cho email: {email}")
        return {
            'success': True,
            'user': user_dict,
            'error': None
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Lỗi khi đăng ký người dùng: {error_msg}")
        
        # Phân tích lỗi để trả về thông báo thân thiện hơn
        if "duplicate key" in error_msg.lower() or "already exists" in error_msg.lower():
            error_msg = "Email này đã được sử dụng. Vui lòng thử email khác."
        elif "database" in error_msg.lower():
            error_msg = "Lỗi cơ sở dữ liệu. Vui lòng thử lại sau."
        elif "network" in error_msg.lower() or "connect" in error_msg.lower():
            error_msg = "Không thể kết nối đến dịch vụ xác thực. Vui lòng kiểm tra kết nối mạng."
        elif "password" in error_msg.lower():
            error_msg = "Mật khẩu không đáp ứng yêu cầu an toàn. Vui lòng sử dụng mật khẩu mạnh hơn."
            
        return {
            'success': False,
            'user': None,
            'error': error_msg
        }

def sign_in(email, password):
    """
    Đăng nhập người dùng
    
    Args:
        email (str): Email người dùng
        password (str): Mật khẩu người dùng
    
    Returns:
        dict: Thông tin đăng nhập người dùng với các key: success, user, error
    """
    try:
        auth_client = get_auth_client()
        if not auth_client:
            return {
                'success': False,
                'user': None,
                'error': "Không thể kết nối đến dịch vụ xác thực"
            }
            
        # Gọi phương thức đăng nhập bằng email và password
        signin_data = {
            "email": email,
            "password": password
        }
        
        user_data = auth_client.auth.sign_in_with_password(signin_data)
        
        # Chuyển đổi đối tượng User thành dict để dễ dàng truy xuất trong app.py
        user_dict = {
            'id': user_data.user.id,
            'email': user_data.user.email,
            'created_at': user_data.user.created_at,
            'user_metadata': user_data.user.user_metadata or {}
        }
        
        # Lưu access_token và refresh_token nếu cần
        if hasattr(user_data, 'session') and user_data.session:
            session['access_token'] = user_data.session.access_token
            session['refresh_token'] = user_data.session.refresh_token
            
        logger.info(f"Đăng nhập thành công cho email: {email}")
        return {
            'success': True,
            'user': user_dict,
            'error': None
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Lỗi khi đăng nhập: {error_msg}")
        
        # Phân tích lỗi để trả về thông báo thân thiện hơn
        if "invalid login credentials" in error_msg.lower():
            error_msg = "Email hoặc mật khẩu không chính xác"
        elif "database" in error_msg.lower():
            error_msg = "Lỗi cơ sở dữ liệu. Vui lòng thử lại sau."
        elif "network" in error_msg.lower() or "connect" in error_msg.lower():
            error_msg = "Không thể kết nối đến dịch vụ xác thực. Vui lòng kiểm tra kết nối mạng."
            
        return {
            'success': False,
            'user': None,
            'error': error_msg
        }

def sign_out(session_id=None):
    """
    Đăng xuất người dùng
    
    Args:
        session_id (str, optional): ID phiên làm việc cần đăng xuất
    
    Returns:
        bool: True nếu đăng xuất thành công, False nếu có lỗi
    """
    try:
        auth_client = get_auth_client()
        if auth_client:
            auth_client.auth.sign_out()
        session.clear()
        logger.info("Đăng xuất thành công")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi đăng xuất: {str(e)}")
        return False

def get_user(jwt=None):
    """
    Lấy thông tin người dùng hiện tại
    
    Args:
        jwt (str, optional): JWT token của người dùng
    
    Returns:
        dict: Thông tin người dùng hoặc None nếu không tìm thấy
    """
    try:
        auth_client = get_auth_client()
        if not auth_client:
            return None
        
        user_response = auth_client.auth.get_user()
        
        if user_response and user_response.user:
            # Chuyển đổi user thành dict
            user_dict = {
                'id': user_response.user.id,
                'email': user_response.user.email,
                'created_at': user_response.user.created_at,
                'user_metadata': user_response.user.user_metadata or {}
            }
            return user_dict
        return None
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin người dùng: {str(e)}")
        return None

def update_user(user_data):
    """
    Cập nhật thông tin người dùng
    
    Args:
        user_data (dict): Dữ liệu người dùng cần cập nhật
    
    Returns:
        dict: Thông tin người dùng đã cập nhật hoặc None nếu có lỗi
    """
    try:
        auth_client = get_auth_client()
        if not auth_client:
            return None
        
        updated_user_response = auth_client.auth.update_user(user_data)
        
        if updated_user_response and updated_user_response.user:
            # Chuyển đổi user thành dict
            user_dict = {
                'id': updated_user_response.user.id,
                'email': updated_user_response.user.email,
                'created_at': updated_user_response.user.created_at,
                'updated_at': updated_user_response.user.updated_at,
                'user_metadata': updated_user_response.user.user_metadata or {}
            }
            return user_dict
        
        return None
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật thông tin người dùng: {str(e)}")
        return None

def get_current_user():
    """
    Lấy thông tin người dùng hiện tại từ session
    
    Returns:
        dict: Thông tin người dùng hiện tại hoặc None nếu chưa đăng nhập
    """
    if 'user_id' not in session:
        return None
        
    user_id = session.get('user_id')
    user_email = session.get('user_email')
    
    if not user_id:
        return None
        
    return {
        'id': user_id,
        'email': user_email
    }

def is_authenticated():
    """
    Kiểm tra xem người dùng đã đăng nhập hay chưa
    
    Returns:
        bool: True nếu người dùng đã đăng nhập, False nếu chưa
    """
    try:
        from flask import session, has_request_context
        if has_request_context():
            return 'user_id' in session
        return False
    except (ImportError, RuntimeError):
        return False 