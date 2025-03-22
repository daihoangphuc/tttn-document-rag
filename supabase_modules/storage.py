"""
Module này cung cấp các hàm để làm việc với Storage của Supabase
"""
import os
import logging
from .client import get_supabase_client

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Quản lý lưu trữ file trên Supabase Storage
    """
    
    def __init__(self, bucket_name="documents"):
        """
        Khởi tạo quản lý lưu trữ
        
        Args:
            bucket_name (str): Tên bucket mặc định
        """
        self.client = get_supabase_client()
        self.bucket_name = bucket_name
        
    def ensure_bucket_exists(self, bucket_name=None):
        """
        Đảm bảo bucket tồn tại, nếu không thì tạo mới
        
        Args:
            bucket_name (str, optional): Tên bucket cần kiểm tra
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        bucket = bucket_name or self.bucket_name
        try:
            buckets = self.client.storage.list_buckets()
            bucket_exists = any(b.name == bucket for b in buckets)
            
            if not bucket_exists:
                self.client.storage.create_bucket(bucket)
                logger.info(f"Đã tạo bucket mới: {bucket}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra/tạo bucket: {e}")
            return False
    
    def _ensure_bucket_exists(self):
        """Đảm bảo bucket mặc định tồn tại, nếu không thì tạo mới"""
        self.ensure_bucket_exists(self.bucket_name)
    
    def upload_file(self, file_path, destination_path=None, bucket_name=None):
        """
        Tải lên file lên Supabase Storage
        
        Args:
            file_path (str): Đường dẫn đến file cần tải lên
            destination_path (str, optional): Đường dẫn đích trên Storage
            bucket_name (str, optional): Tên bucket để lưu trữ
        
        Returns:
            dict: Thông tin về file đã tải lên hoặc None nếu có lỗi
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        if destination_path is None:
            destination_path = os.path.basename(file_path)
            
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                
            result = self.client.storage.from_(bucket_name).upload(
                path=destination_path,
                file=file_content,
                file_options={"content-type": "application/octet-stream"}
            )
            
            logger.info(f"Đã tải lên file thành công: {destination_path}")
            return result
        except Exception as e:
            logger.error(f"Lỗi khi tải lên file: {str(e)}")
            return None
    
    def save_user_file(self, file_obj, user_id, bucket_name=None):
        """
        Lưu file từ request object (Flask) vào Supabase Storage
        
        Args:
            file_obj (FileStorage): Đối tượng file từ request.files
            user_id (str): ID của người dùng để tạo đường dẫn
            bucket_name (str, optional): Tên bucket để lưu trữ
            
        Returns:
            str: Đường dẫn của file trên Storage hoặc None nếu có lỗi
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        # Đảm bảo bucket tồn tại
        self.ensure_bucket_exists(bucket_name)
            
        try:
            # Tạo đường dẫn an toàn cho file theo user_id/filename
            import re
            from urllib.parse import quote
            
            # Lấy tên file và loại bỏ ký tự không hợp lệ
            filename = file_obj.filename
            # Mã hóa tên file để tránh lỗi với các ký tự đặc biệt
            safe_filename = re.sub(r'[^\w\-\.]', '_', filename)
            
            # Tạo đường dẫn với ID người dùng và tên file an toàn
            file_path = f"{user_id}/{safe_filename}"
            
            # Đọc nội dung file
            file_content = file_obj.read()
            # Reset con trỏ file để có thể đọc lại sau này nếu cần
            file_obj.seek(0)
            
            # Tải lên file vào Supabase Storage
            result = self.client.storage.from_(bucket_name).upload(
                path=file_path,
                file=file_content,
                file_options={"content-type": file_obj.content_type or "application/octet-stream"}
            )
            
            logger.info(f"Đã lưu file người dùng thành công: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Lỗi khi lưu file người dùng: {str(e)}")
            raise e
    
    def download_file(self, file_path, destination_path=None, bucket_name=None):
        """
        Tải xuống file từ Supabase Storage
        
        Args:
            file_path (str): Đường dẫn file trên Storage
            destination_path (str, optional): Đường dẫn lưu file cục bộ
            bucket_name (str, optional): Tên bucket chứa file
            
        Returns:
            bool: True nếu tải xuống thành công, False nếu có lỗi
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        if destination_path is None:
            destination_path = os.path.basename(file_path)
            
        try:
            file_data = self.client.storage.from_(bucket_name).download(file_path)
            
            with open(destination_path, 'wb') as f:
                f.write(file_data)
                
            logger.info(f"Đã tải xuống file thành công: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải xuống file: {str(e)}")
            return False
    
    def list_files(self, path="", bucket_name=None):
        """
        Liệt kê danh sách file trong bucket
        
        Args:
            path (str, optional): Đường dẫn thư mục trên Storage
            bucket_name (str, optional): Tên bucket cần liệt kê
            
        Returns:
            list: Danh sách các file hoặc None nếu có lỗi
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        try:
            files = self.client.storage.from_(bucket_name).list(path)
            return files
        except Exception as e:
            logger.error(f"Lỗi khi liệt kê file: {str(e)}")
            return None
    
    def delete_file(self, file_path, bucket_name=None):
        """
        Xóa file từ Supabase Storage
        
        Args:
            file_path (str): Đường dẫn file trên Storage
            bucket_name (str, optional): Tên bucket chứa file
            
        Returns:
            bool: True nếu xóa thành công, False nếu có lỗi
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        try:
            self.client.storage.from_(bucket_name).remove([file_path])
            logger.info(f"Đã xóa file thành công: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa file: {str(e)}")
            return False
    
    def get_public_url(self, file_path, bucket_name=None):
        """
        Lấy URL công khai của file
        
        Args:
            file_path (str): Đường dẫn file trên Storage
            bucket_name (str, optional): Tên bucket chứa file
            
        Returns:
            str: URL công khai hoặc None nếu có lỗi
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        try:
            url = self.client.storage.from_(bucket_name).get_public_url(file_path)
            return url
        except Exception as e:
            logger.error(f"Lỗi khi lấy URL công khai: {str(e)}")
            return None 