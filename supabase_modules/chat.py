"""
Module này cung cấp các hàm để làm việc với chức năng Chat trong Supabase
"""
import logging
from datetime import datetime
from .client import get_supabase_client

logger = logging.getLogger(__name__)

class ChatManager:
    """
    Quản lý cuộc trò chuyện sử dụng Supabase
    """
    
    def __init__(self, table_name="chats"):
        """
        Khởi tạo Chat Manager
        
        Args:
            table_name (str): Tên bảng lưu trữ chat
        """
        self.client = get_supabase_client()
        self.table_name = table_name
    
    def create_conversation(self, user_id, title="Cuộc trò chuyện mới", metadata=None):
        """
        Tạo cuộc trò chuyện mới
        
        Args:
            user_id (str): ID của người dùng
            title (str, optional): Tiêu đề của cuộc trò chuyện
            metadata (dict, optional): Thông tin bổ sung về cuộc trò chuyện
            
        Returns:
            dict: Thông tin về cuộc trò chuyện đã tạo hoặc None nếu có lỗi
        """
        try:
            current_time = datetime.now().isoformat()
            
            conversation_data = {
                "user_id": user_id,
                "title": title,
                "created_at": current_time,
                "updated_at": current_time,
                "metadata": metadata or {},
                "is_active": True
            }
            
            result = self.client.table("conversations").insert(conversation_data).execute()
            
            if result.data:
                logger.info(f"Đã tạo cuộc trò chuyện mới: {result.data[0]['id']}")
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Lỗi khi tạo cuộc trò chuyện: {str(e)}")
            return None
    
    def send_message(self, conversation_id, content, user_id, is_bot=False, metadata=None):
        """
        Gửi tin nhắn mới trong cuộc trò chuyện
        
        Args:
            conversation_id (str): ID cuộc trò chuyện
            content (str): Nội dung tin nhắn
            user_id (str): ID người dùng
            is_bot (bool, optional): Có phải tin nhắn từ bot không
            metadata (dict, optional): Thông tin bổ sung về tin nhắn
            
        Returns:
            dict: Thông tin về tin nhắn đã gửi hoặc None nếu có lỗi
        """
        try:
            current_time = datetime.now().isoformat()
            
            message_data = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "content": content,
                "is_bot": is_bot,
                "created_at": current_time,
                "metadata": metadata or {}
            }
            
            result = self.client.table("messages").insert(message_data).execute()
            
            # Cập nhật thời gian cuộc trò chuyện
            self.client.table("conversations").update(
                {"updated_at": current_time}
            ).eq("id", conversation_id).execute()
            
            if result.data:
                logger.info(f"Đã gửi tin nhắn mới trong cuộc trò chuyện: {conversation_id}")
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Lỗi khi gửi tin nhắn: {str(e)}")
            return None
    
    def get_conversation(self, conversation_id):
        """
        Lấy thông tin về cuộc trò chuyện
        
        Args:
            conversation_id (str): ID cuộc trò chuyện
            
        Returns:
            dict: Thông tin về cuộc trò chuyện hoặc None nếu không tìm thấy
        """
        try:
            result = self.client.table("conversations").select("*").eq("id", conversation_id).execute()
            
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin cuộc trò chuyện: {str(e)}")
            return None
    
    def get_messages(self, conversation_id, limit=50, offset=0):
        """
        Lấy danh sách tin nhắn trong cuộc trò chuyện
        
        Args:
            conversation_id (str): ID cuộc trò chuyện
            limit (int, optional): Số lượng tin nhắn cần lấy
            offset (int, optional): Vị trí bắt đầu
            
        Returns:
            list: Danh sách tin nhắn hoặc None nếu có lỗi
        """
        try:
            result = self.client.table("messages").select("*") \
                .eq("conversation_id", conversation_id) \
                .order("created_at", desc=False) \
                .limit(limit).offset(offset).execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Lỗi khi lấy tin nhắn: {str(e)}")
            return None
    
    def get_user_conversations(self, user_id, limit=10, offset=0):
        """
        Lấy danh sách cuộc trò chuyện của người dùng
        
        Args:
            user_id (str): ID người dùng
            limit (int, optional): Số lượng cuộc trò chuyện cần lấy
            offset (int, optional): Vị trí bắt đầu
            
        Returns:
            list: Danh sách cuộc trò chuyện hoặc None nếu có lỗi
        """
        try:
            result = self.client.table("conversations").select("*") \
                .eq("user_id", user_id) \
                .eq("is_active", True) \
                .order("updated_at", desc=True) \
                .limit(limit).offset(offset).execute()
            
            return result.data
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách cuộc trò chuyện: {str(e)}")
            return None
    
    def delete_conversation(self, conversation_id):
        """
        Xóa cuộc trò chuyện (đánh dấu là không hoạt động)
        
        Args:
            conversation_id (str): ID cuộc trò chuyện
            
        Returns:
            bool: True nếu xóa thành công, False nếu có lỗi
        """
        try:
            self.client.table("conversations").update(
                {"is_active": False}
            ).eq("id", conversation_id).execute()
            
            logger.info(f"Đã xóa cuộc trò chuyện: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa cuộc trò chuyện: {str(e)}")
            return False 