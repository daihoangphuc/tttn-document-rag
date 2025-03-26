"""
Các hàm tiện ích cho tích hợp Supabase vào hệ thống RAG
"""
import os
import json
import logging
from flask import session
from .config import get_supabase_client
from .file_manager import get_user_files, ensure_user_upload_dir, migrate_files_to_user_directory
from .chat_history import migrate_chat_history

# Thiết lập logging
logger = logging.getLogger('RAG_System')

def get_user_id_from_session():
    """
    Lấy user_id từ session hiện tại
    
    Returns:
        str: User ID hoặc None nếu không có session
    """
    return session.get('user_id')

def get_user_files_with_metadata(user_id):
    """
    Lấy danh sách file của người dùng kèm theo metadata
    
    Args:
        user_id: ID của người dùng
        
    Returns:
        list: Danh sách file với metadata
    """
    try:
        supabase = get_supabase_client()
        
        # Lấy thông tin file từ Supabase
        response = supabase.table('files') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('status', 'active') \
            .execute()
        
        if response.data:
            return response.data
        
        return []
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách file: {str(e)}")
        return []

def migrate_localStorage_to_supabase(user_id, chat_history_json):
    """
    Di chuyển dữ liệu từ localStorage sang Supabase
    
    Args:
        user_id: ID người dùng
        chat_history_json: Chuỗi JSON lịch sử trò chuyện từ localStorage
        
    Returns:
        dict: Kết quả di chuyển
    """
    try:
        # Parse JSON
        chat_history = json.loads(chat_history_json) if chat_history_json else {}
        
        # Di chuyển lịch sử trò chuyện
        migrated_chats = migrate_chat_history(user_id, chat_history)
        
        return {
            'success': True,
            'migrated_chats': migrated_chats,
        }
    except Exception as e:
        logger.error(f"Lỗi khi di chuyển dữ liệu localStorage: {str(e)}")
        return {
            'success': False,
            'message': str(e),
            'migrated_chats': 0,
        }

def initialize_user_data(user_id, chat_history_json=None):
    """
    Khởi tạo dữ liệu cho người dùng mới
    
    Args:
        user_id: ID người dùng
        chat_history_json: Chuỗi JSON lịch sử trò chuyện từ localStorage (nếu có)
        
    Returns:
        dict: Kết quả khởi tạo
    """
    try:
        result = {
            'success': True,
            'migrated_chats': 0,
            'migrated_files': 0,
            'message': ''
        }
        
        # Đảm bảo thư mục upload tồn tại
        ensure_user_upload_dir(user_id)
        
        # Di chuyển lịch sử trò chuyện từ localStorage nếu có
        if chat_history_json:
            chat_result = migrate_localStorage_to_supabase(user_id, chat_history_json)
            result['migrated_chats'] = chat_result.get('migrated_chats', 0)
            
            if not chat_result.get('success', False):
                result['message'] += f"Lỗi khi di chuyển lịch sử trò chuyện: {chat_result.get('message', '')}. "
                result['success'] = False
        
        # Di chuyển các file từ thư mục chung vào thư mục người dùng
        file_result = migrate_files_to_user_directory(user_id)
        result['migrated_files'] = file_result.get('migrated_files', 0)
        
        if not file_result.get('success', False):
            result['message'] += f"Lỗi khi di chuyển files: {file_result.get('message', '')}. "
            result['success'] = False
            
        return result
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo dữ liệu người dùng: {str(e)}")
        return {
            'success': False,
            'message': str(e),
            'migrated_chats': 0,
            'migrated_files': 0
        }

def format_chat_history_for_frontend(chat_history):
    """
    Định dạng lịch sử trò chuyện cho frontend
    
    Args:
        chat_history: Dữ liệu lịch sử trò chuyện từ Supabase
        
    Returns:
        dict: Lịch sử trò chuyện đã định dạng
    """
    formatted_history = {}
    
    for chat in chat_history:
        chat_id = chat.get('id')
        chat_title = chat.get('title', 'Cuộc trò chuyện không tiêu đề')
        
        # Thêm chat vào kết quả với đầy đủ thông tin cần thiết cho frontend
        formatted_history[chat_id] = {
            'id': chat_id,
            'title': chat_title,
            'timestamp': chat.get('updated_at') or chat.get('created_at', ''),
            'last_message': chat.get('last_message', ''),
            'messages': []
        }
        
        # Thêm tin nhắn
        if 'messages' in chat and chat['messages']:
            for msg in chat['messages']:
                formatted_history[chat_id]['messages'].append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', ''),
                    'timestamp': msg.get('created_at', '')
                })
    
    return formatted_history 