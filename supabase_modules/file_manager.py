"""
Quản lý tập tin người dùng
"""
import os
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from .config import get_supabase_client

# Thiết lập logging
logger = logging.getLogger('supabase_file_manager')

# Thư mục gốc chứa tệp tin tải lên
BASE_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')

def ensure_user_upload_dir(user_id: str) -> str:
    """
    Đảm bảo thư mục upload cho người dùng tồn tại
    
    Args:
        user_id (str): ID của người dùng
        
    Returns:
        str: Đường dẫn đến thư mục upload của người dùng
    """
    user_dir = os.path.join(BASE_UPLOAD_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"Đã tạo thư mục upload cho người dùng {user_id}: {user_dir}")
    return user_dir

def get_user_upload_dir(user_id: str) -> str:
    """
    Lấy đường dẫn thư mục upload của người dùng mà không tạo mới
    
    Args:
        user_id (str): ID của người dùng
        
    Returns:
        str: Đường dẫn đến thư mục upload của người dùng
    """
    return os.path.join(BASE_UPLOAD_DIR, user_id)

def save_user_file(user_id: str, file: FileStorage) -> Tuple[bool, str]:
    """
    Lưu tệp tin của người dùng vào thư mục riêng
    
    Args:
        user_id (str): ID của người dùng
        file (FileStorage): Đối tượng tệp tin tải lên
        
    Returns:
        Tuple[bool, str]: Tuple (thành công, đường dẫn tệp tin hoặc thông báo lỗi)
    """
    try:
        # Đảm bảo thư mục tồn tại
        user_dir = ensure_user_upload_dir(user_id)
        
        # Lấy tên tệp tin an toàn
        filename = secure_filename(file.filename)
        
        # Đường dẫn đầy đủ
        file_path = os.path.join(user_dir, filename)
        
        # Lưu tệp tin
        file.save(file_path)
        
        # Cập nhật bảng files trong Supabase
        supabase = get_supabase_client()
        
        # Thêm bản ghi vào bảng files
        response = supabase.table('files').insert({
            'user_id': user_id,
            'filename': filename,
            'file_path': os.path.join(user_id, filename),
            'status': 'active'
        }).execute()
        
        logger.info(f"Đã lưu tệp tin {filename} cho người dùng {user_id}")
        return (True, file_path)
    except Exception as e:
        logger.error(f"Lỗi khi lưu tệp tin cho người dùng {user_id}: {str(e)}")
        return (False, str(e))

def get_user_files(user_id: str) -> List[Dict[str, Any]]:
    """
    Lấy danh sách tệp tin của người dùng
    
    Args:
        user_id (str): ID của người dùng
        
    Returns:
        List[Dict[str, Any]]: Danh sách tệp tin
    """
    try:
        supabase = get_supabase_client()
        response = supabase.table('files').select('*').eq('user_id', user_id).eq('status', 'active').execute()
        
        if not response.data:
            return []
            
        return response.data
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách tệp tin của người dùng {user_id}: {str(e)}")
        return []

def delete_user_file(user_id: str, filename: str) -> bool:
    """
    Xóa tệp tin của người dùng
    
    Args:
        user_id (str): ID của người dùng
        filename (str): Tên tệp tin cần xóa
        
    Returns:
        bool: True nếu xóa thành công, False nếu có lỗi
    """
    try:
        # Đường dẫn tệp tin
        file_path = os.path.join(ensure_user_upload_dir(user_id), secure_filename(filename))
        
        # Kiểm tra xem tệp tin có tồn tại không
        if not os.path.exists(file_path):
            logger.warning(f"Tệp tin {filename} không tồn tại cho người dùng {user_id}")
            # Vẫn tiếp tục xóa trong cơ sở dữ liệu nếu tệp không tồn tại trong filesystem
        else:
            # Xóa tệp tin từ hệ thống tệp
            os.remove(file_path)
            logger.info(f"Đã xóa tệp tin {filename} của người dùng {user_id} khỏi hệ thống tệp")
        
        # Xóa tệp tin khỏi Supabase thay vì chỉ đánh dấu đã xóa
        supabase = get_supabase_client()
        
        # Thử xóa bằng nhiều cách khác nhau để đảm bảo xóa thành công
        try:
            # Cách 1: Xóa bằng DELETE với điều kiện filename và user_id
            response = supabase.table('files').delete().eq('user_id', user_id).eq('filename', filename).execute()
            
            # Kiểm tra kết quả
            if response.data and len(response.data) > 0:
                logger.info(f"Đã xóa tệp tin {filename} của người dùng {user_id} khỏi cơ sở dữ liệu (Cách 1)")
            else:
                # Cách 2: Xóa bằng cách sử dụng file_path thay vì filename
                file_path_in_db = os.path.join(user_id, secure_filename(filename))
                response = supabase.table('files').delete().eq('user_id', user_id).eq('file_path', file_path_in_db).execute()
                
                if response.data and len(response.data) > 0:
                    logger.info(f"Đã xóa tệp tin {filename} của người dùng {user_id} khỏi cơ sở dữ liệu (Cách 2)")
                else:
                    # Cách 3: Sử dụng SQL trực tiếp
                    response = supabase.table('files').delete().filter('user_id', 'eq', user_id).filter('filename', 'eq', filename).execute()
                    logger.info(f"Đã thử xóa tệp tin bằng truy vấn trực tiếp: {response.data}")
                    
                    # Thông báo
                    if not response.data or len(response.data) == 0:
                        logger.warning(f"Không tìm thấy bản ghi tệp tin {filename} của người dùng {user_id} trong cơ sở dữ liệu sau khi thử 3 cách")
        except Exception as e:
            logger.error(f"Lỗi khi xóa tệp tin từ Supabase: {str(e)}")
            # Không return False ở đây, vẫn tiếp tục để xóa tệp ra khỏi hệ thống RAG
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi xóa tệp tin {filename} của người dùng {user_id}: {str(e)}")
        return False

def migrate_files_to_user_directory(user_id: str) -> Dict[str, Any]:
    """
    Di chuyển các tệp tin từ thư mục uploads/ chung sang thư mục người dùng
    
    Args:
        user_id (str): ID của người dùng
        
    Returns:
        Dict[str, Any]: Kết quả di chuyển
    """
    try:
        # Đảm bảo thư mục người dùng tồn tại
        user_dir = ensure_user_upload_dir(user_id)
        
        # Kiểm tra xem thư mục gốc có tồn tại không
        if not os.path.exists(BASE_UPLOAD_DIR):
            logger.warning(f"Thư mục uploads gốc không tồn tại: {BASE_UPLOAD_DIR}")
            return {
                'success': False,
                'message': 'Thư mục uploads gốc không tồn tại',
                'migrated_files': 0,
                'total_files': 0
            }
            
        # Lấy danh sách tệp tin trong thư mục gốc, bao gồm cả tệp hệ thống
        files_in_root = [f for f in os.listdir(BASE_UPLOAD_DIR) 
                         if os.path.isfile(os.path.join(BASE_UPLOAD_DIR, f))]
        
        # Các tệp hệ thống cần di chuyển
        system_files = ['faiss_index.bin', 'rag_state.json', 'tfidf_matrix.pkl', 
                       'tfidf_vectorizer.pkl', 'vectors.pkl']
        
        # Số tệp tin đã di chuyển
        moved_count = 0
        
        # Di chuyển từng tệp tin
        for filename in files_in_root:
            source_path = os.path.join(BASE_UPLOAD_DIR, filename)
            dest_path = os.path.join(user_dir, filename)
            
            # Bỏ qua nếu tệp tin đã tồn tại trong thư mục người dùng
            if os.path.exists(dest_path):
                continue
            
            # Xác định loại tệp tin
            is_system_file = filename in system_files
            
            # Di chuyển tệp tin
            shutil.copy2(source_path, dest_path)
            
            # Kiểm tra xem tệp tin đã được di chuyển thành công
            if os.path.exists(dest_path):
                moved_count += 1
                
                # Chỉ cập nhật Supabase nếu không phải là tệp hệ thống
                if not is_system_file:
                    # Cập nhật bảng files trong Supabase
                    supabase = get_supabase_client()
                        
                    response = supabase.table('files').insert({
                        'user_id': user_id,
                        'filename': filename,
                        'file_path': os.path.join(user_id, filename),
                        'status': 'active'
                    }).execute()
        
        logger.info(f"Đã di chuyển {moved_count}/{len(files_in_root)} tệp tin vào thư mục của người dùng {user_id}")
        return {
            'success': True,
            'migrated_files': moved_count,
            'total_files': len(files_in_root)
        }
    except Exception as e:
        logger.error(f"Lỗi khi di chuyển tệp tin cho người dùng {user_id}: {str(e)}")
        return {
            'success': False,
            'message': str(e),
            'migrated_files': 0,
            'total_files': 0
        }

def get_file_path(user_id: str, filename: str) -> Optional[str]:
    """
    Lấy đường dẫn đầy đủ đến tệp tin của người dùng
    
    Args:
        user_id (str): ID của người dùng
        filename (str): Tên tệp tin
        
    Returns:
        Optional[str]: Đường dẫn đầy đủ hoặc None nếu tệp tin không tồn tại
    """
    # Đường dẫn trong thư mục của người dùng
    user_file_path = os.path.join(ensure_user_upload_dir(user_id), secure_filename(filename))
    
    # Kiểm tra xem tệp tin có tồn tại trong thư mục người dùng không
    if os.path.exists(user_file_path):
        return user_file_path
        
    # Đường dẫn trong thư mục gốc (khả năng tương thích ngược)
    root_file_path = os.path.join(BASE_UPLOAD_DIR, secure_filename(filename))
    
    # Kiểm tra xem tệp tin có tồn tại trong thư mục gốc không
    if os.path.exists(root_file_path):
        return root_file_path
        
    # Tệp tin không tồn tại
    return None 