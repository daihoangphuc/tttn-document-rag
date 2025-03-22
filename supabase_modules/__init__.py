# Thư mục supabase_modules chứa các tiện ích để tích hợp với Supabase

# File này đánh dấu thư mục này là một Python package

# Re-export các hàm và lớp cơ bản
try:
    # Tránh circular import bằng cách import động
    def create_client(url, key, **options):
        """
        Tạo và trả về một Supabase client
        
        Args:
            url (str): URL của Supabase project
            key (str): API key của Supabase project
            **options: Các tùy chọn khác
            
        Returns:
            Client: Supabase client
        """
        from supabase import create_client as _create_client
        return _create_client(url, key, **options)
    
    # Định nghĩa Client cho backward compatibility
    Client = None
    def get_client_class():
        """Lấy class Client từ thư viện supabase"""
        try:
            from supabase.client import Client as SupabaseClient
            return SupabaseClient
        except ImportError:
            try:
                # Cố gắng import Client từ thư viện supabase
                from supabase import Client as SupabaseClient
                return SupabaseClient
            except ImportError:
                raise ImportError("Không thể import Client từ thư viện supabase")
    
    # Export lại các thành phần này
    __all__ = ['create_client', 'get_client_class']
    
except ImportError as e:
    import logging
    logging.getLogger(__name__).error(f"Không thể import thư viện supabase: {str(e)}")
    
    # Tạo các placeholder
    def create_client(*args, **kwargs):
        raise ImportError("Không thể import supabase. Vui lòng cài đặt: pip install supabase")
    
    def get_client_class():
        raise ImportError("Không thể import supabase. Vui lòng cài đặt: pip install supabase")
    
    __all__ = ['create_client', 'get_client_class']

# Xuất ra các module con để dễ sử dụng
from . import auth
from . import storage
from . import chat
from . import client

# Xuất ra các hàm tiện ích
from .client import get_supabase_client, test_connection

# Thêm vào __all__
__all__.extend(['auth', 'storage', 'chat', 'client', 'get_supabase_client', 'test_connection'])

# Import các module con
try:
    from . import auth
    from . import storage
    from . import chat
    
    # Thêm vào __all__
    __all__.extend(['auth', 'storage', 'chat'])
except ImportError:
    pass

# File này được tạo trống để đánh dấu thư mục supabase là một Python package 