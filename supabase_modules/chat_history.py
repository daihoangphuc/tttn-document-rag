"""
Quản lý lịch sử chat với Supabase
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from supabase import create_client, Client
from flask import session
from supabase_modules.config import get_supabase_client, get_supabase

# Thiết lập logging
logger = logging.getLogger(__name__)

# Khởi tạo Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Đã khởi tạo kết nối Supabase thành công")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Supabase client: {str(e)}")
else:
    logger.warning("Thiếu thông tin cấu hình SUPABASE_URL hoặc SUPABASE_KEY")


def validate_and_format_uuid(uuid_string: str, param_name: str = "uuid") -> Optional[str]:
    """
    Kiểm tra và định dạng chuỗi UUID.
    
    Args:
        uuid_string (str): Chuỗi cần kiểm tra là UUID
        param_name (str): Tên tham số (để ghi log)
        
    Returns:
        Optional[str]: Chuỗi UUID đã được định dạng hoặc None nếu không hợp lệ
    """
    import uuid
    
    if not uuid_string:
        logger.error(f"{param_name} không được để trống")
        return None
        
    # Xóa khoảng trắng và chuyển sang chữ thường
    cleaned_uuid = uuid_string.strip().lower()
    
    # Xóa dấu gạch ngang nếu có và thêm lại để đảm bảo định dạng đúng
    try:
        # Xử lý trường hợp UUID không có dấu gạch
        if len(cleaned_uuid) == 32 and "-" not in cleaned_uuid:
            formatted_uuid = f"{cleaned_uuid[0:8]}-{cleaned_uuid[8:12]}-{cleaned_uuid[12:16]}-{cleaned_uuid[16:20]}-{cleaned_uuid[20:]}"
            uuid_obj = uuid.UUID(formatted_uuid)
            return str(uuid_obj)
        else:
            # Xử lý UUID thông thường
            uuid_obj = uuid.UUID(cleaned_uuid)
            return str(uuid_obj)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"{param_name} không phải là UUID hợp lệ: {uuid_string}, lỗi: {str(e)}")
        return None


def create_chat(
    user_id: str, title: str = "Cuộc trò chuyện mới"
) -> Optional[Dict[str, Any]]:
    """
    Tạo một cuộc trò chuyện mới

    Args:
        user_id (str): ID người dùng
        title (str): Tiêu đề của cuộc trò chuyện

    Returns:
        Optional[Dict[str, Any]]: Thông tin cuộc trò chuyện đã tạo hoặc None nếu có lỗi
    """
    supabase = get_supabase()
    if not supabase:
        logger.error("Không thể tạo cuộc trò chuyện: Chưa cấu hình kết nối Supabase")
        return None

    logger.info(f"Đang tạo chat mới cho user_id: {user_id} với tiêu đề: {title}")

    try:
        # Kiểm tra user_id có hợp lệ không
        validated_user_id = validate_and_format_uuid(user_id, "user_id")
        if not validated_user_id:
            return None
        
        user_id = validated_user_id

        # Kiểm tra kết nối Supabase
        connection_ok = False
        max_retries = 3
        retry_count = 0
        
        while not connection_ok and retry_count < max_retries:
            try:
                # Thử fetch một record bất kỳ để kiểm tra kết nối
                supabase = get_supabase()  # Lấy client mới mỗi lần thử lại
                test_query = supabase.table("chats").select("*").limit(1).execute()
                logger.info(
                    f"Kết nối Supabase OK, bản ghi test: {len(test_query.data or [])}"
                )
                connection_ok = True
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                logger.error(f"Lỗi kết nối Supabase (lần thử {retry_count}/{max_retries}): {error_message}")
                
                if "Server disconnected" in error_message:
                    logger.info("Đang thử kết nối lại với Supabase...")
                    import time
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại
                else:
                    # Nếu không phải lỗi đứt kết nối, không cần thử lại
                    return None
        
        if not connection_ok:
            logger.error(f"Không thể kết nối với Supabase sau {max_retries} lần thử")
            return None

        # Tạo chat mới
        current_time = datetime.utcnow().isoformat()
        chat_data = {
            "user_id": user_id,
            "title": title,
            "last_message": "",
            "created_at": current_time,
            "updated_at": current_time,
        }

        logger.info(f"Gửi yêu cầu tạo chat mới với dữ liệu: {chat_data}")

        # Thực hiện insert và bắt mọi ngoại lệ có thể xảy ra
        max_insert_retries = 3
        insert_retry_count = 0
        insert_success = False
        
        while not insert_success and insert_retry_count < max_insert_retries:
            try:
                # Lấy client mới mỗi lần thử lại
                supabase = get_supabase()
                response = supabase.table("chats").insert(chat_data).execute()
                insert_success = True
            except Exception as e:
                insert_retry_count += 1
                error_message = str(e)
                logger.error(f"Ngoại lệ khi gọi API tạo chat mới (lần thử {insert_retry_count}/{max_insert_retries}): {error_message}")
                
                if "Server disconnected" in error_message:
                    logger.info("Đang thử kết nối lại với Supabase...")
                    import time
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại
                else:
                    # Nếu không phải lỗi đứt kết nối, không cần thử lại
                    return None
        
        if not insert_success:
            logger.error(f"Không thể tạo chat mới sau {max_insert_retries} lần thử")
            return None

        # Kiểm tra response
        if not response:
            logger.error("Không nhận được phản hồi từ Supabase khi tạo chat mới")
            return None

        # Kiểm tra dữ liệu trả về thay vì kiểm tra error
        if not response.data:
            logger.error(f"Lỗi khi tạo chat mới trên Supabase: Không có dữ liệu trả về")
            return None

        if len(response.data) > 0:
            new_chat = response.data[0]
            chat_id = new_chat.get("id")
            if not chat_id:
                logger.error(f"Tạo chat mới thành công nhưng không có ID: {new_chat}")
                return None

            logger.info(
                f"Đã tạo cuộc trò chuyện mới thành công cho user ID: {user_id}, chat ID: {chat_id}"
            )

            # Kiểm tra xem chat có thực sự được tạo không
            verification_success = False
            verification_retries = 2
            verification_count = 0
            
            while not verification_success and verification_count < verification_retries:
                try:
                    # Lấy client mới mỗi lần thử lại
                    supabase = get_supabase()
                    verification_query = (
                        supabase.table("chats").select("*").eq("id", chat_id).execute()
                    )
                    if verification_query.data and len(verification_query.data) > 0:
                        logger.info(
                            f"Đã xác minh chat mới tạo tồn tại trong database với ID: {chat_id}"
                        )
                        verification_success = True
                    else:
                        logger.warning(
                            f"Không thể xác minh chat mới đã tạo (ID: {chat_id})"
                        )
                        verification_count += 1
                        import time
                        time.sleep(1)  # Chờ 1 giây trước khi thử lại
                except Exception as e:
                    verification_count += 1
                    logger.warning(f"Lỗi khi xác minh chat mới tạo (lần thử {verification_count}/{verification_retries}): {str(e)}")
                    import time
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại
            
            # Vẫn trả về chat vì có thể là vấn đề với truy vấn xác minh
            return new_chat

        logger.warning(
            f"Không thể tạo cuộc trò chuyện mới cho user ID: {user_id}, response: {response}"
        )
        return None

    except Exception as e:
        logger.error(f"Lỗi khi tạo cuộc trò chuyện cho user ID {user_id}: {str(e)}")
        import traceback

        logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return None


def get_chats(user_id: str) -> List[Dict[str, Any]]:
    """
    Lấy danh sách các cuộc trò chuyện của người dùng

    Args:
        user_id (str): ID người dùng

    Returns:
        List[Dict[str, Any]]: Danh sách các cuộc trò chuyện
    """
    supabase = get_supabase()
    if not supabase:
        logger.error(
            "Không thể lấy danh sách trò chuyện: Chưa cấu hình kết nối Supabase"
        )
        return []

    try:
        response = (
            supabase.table("chats")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )

        if response.data:
            return response.data

        return []

    except Exception as e:
        logger.error(
            f"Lỗi khi lấy danh sách trò chuyện cho user ID {user_id}: {str(e)}"
        )
        return []


def get_chat(chat_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin một cuộc trò chuyện cụ thể

    Args:
        chat_id (str): ID của cuộc trò chuyện
        user_id (str): ID người dùng (để kiểm tra quyền)

    Returns:
        Optional[Dict[str, Any]]: Thông tin cuộc trò chuyện hoặc None nếu có lỗi
    """
    supabase = get_supabase()
    if not supabase:
        logger.error(
            "Không thể lấy thông tin trò chuyện: Chưa cấu hình kết nối Supabase"
        )
        return None

    try:
        # Kiểm tra xem chat_id có phải là UUID hợp lệ không
        validated_chat_id = validate_and_format_uuid(chat_id, "chat_id")
        if not validated_chat_id:
            return None
        
        chat_id = validated_chat_id

        # Tiếp tục với chat_id đã được xác thực
        max_retries = 3
        retry_count = 0
        response = None
        
        while response is None and retry_count < max_retries:
            try:
                # Lấy client mới mỗi lần thử lại
                supabase = get_supabase()
                response = (
                    supabase.table("chats")
                    .select("*")
                    .eq("id", chat_id)
                    .eq("user_id", user_id)
                    .execute()
                )
                
                if response.data and len(response.data) > 0:
                    return response.data[0]
                    
                # Nếu không tìm thấy, tăng retry_count và kiểm tra xem có cần thử lại không
                retry_count += 1
                if retry_count >= max_retries:
                    logger.warning(
                        f"Không tìm thấy cuộc trò chuyện ID {chat_id} cho user ID {user_id}"
                    )
                    return None
                else:
                    # Đã tìm kiếm thành công nhưng không có kết quả, không cần thử lại
                    break
                    
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                logger.error(f"Lỗi khi lấy thông tin trò chuyện ID {chat_id} (lần thử {retry_count}/{max_retries}): {error_message}")
                
                if "Server disconnected" in error_message:
                    logger.info("Đang thử kết nối lại với Supabase...")
                    import time
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại
                elif retry_count >= max_retries:
                    return None

        return None

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin trò chuyện ID {chat_id}: {str(e)}")
        return None


def update_chat_title(chat_id: str, user_id: str, title: str) -> bool:
    """
    Cập nhật tiêu đề cuộc trò chuyện

    Args:
        chat_id (str): ID của cuộc trò chuyện
        user_id (str): ID người dùng (để kiểm tra quyền)
        title (str): Tiêu đề mới

    Returns:
        bool: True nếu cập nhật thành công, False nếu có lỗi
    """
    supabase = get_supabase()
    if not supabase:
        logger.error(
            "Không thể cập nhật tiêu đề trò chuyện: Chưa cấu hình kết nối Supabase"
        )
        return False

    try:
        # Kiểm tra xem cuộc trò chuyện có thuộc về người dùng không
        chat = get_chat(chat_id, user_id)
        if not chat:
            logger.warning(
                f"Không thể cập nhật tiêu đề: Không tìm thấy trò chuyện ID {chat_id} cho user ID {user_id}"
            )
            return False

        response = (
            supabase.table("chats")
            .update({"title": title, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", chat_id)
            .eq("user_id", user_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            logger.info(f"Đã cập nhật tiêu đề cho trò chuyện ID {chat_id}")
            return True

        logger.warning(f"Không thể cập nhật tiêu đề cho trò chuyện ID {chat_id}")
        return False

    except Exception as e:
        logger.error(f"Lỗi khi cập nhật tiêu đề trò chuyện ID {chat_id}: {str(e)}")
        return False


def delete_chat(chat_id: str, user_id: str) -> bool:
    """
    Xóa một cuộc trò chuyện

    Args:
        chat_id (str): ID của cuộc trò chuyện
        user_id (str): ID người dùng (để kiểm tra quyền)

    Returns:
        bool: True nếu xóa thành công, False nếu có lỗi
    """
    supabase = get_supabase()
    if not supabase:
        logger.error("Không thể xóa trò chuyện: Chưa cấu hình kết nối Supabase")
        return False

    try:
        # Kiểm tra xem chat_id có hợp lệ không
        validated_chat_id = validate_and_format_uuid(chat_id, "chat_id")
        if not validated_chat_id:
            return False
        
        chat_id = validated_chat_id

        # Kiểm tra xem user_id có hợp lệ không
        validated_user_id = validate_and_format_uuid(user_id, "user_id")
        if not validated_user_id:
            return False
        
        user_id = validated_user_id

        logger.info(f"Đang xóa chat ID {chat_id} của user ID {user_id}")

        # Kiểm tra xem cuộc trò chuyện có thuộc về người dùng không
        chat = get_chat(chat_id, user_id)
        if not chat:
            logger.warning(
                f"Không thể xóa: Không tìm thấy trò chuyện ID {chat_id} cho user ID {user_id}"
            )
            return False

        # Xóa tin nhắn trong cuộc trò chuyện trước
        try:
            delete_messages_response = (
                supabase.table("messages").delete().eq("chat_id", chat_id).execute()
            )

            if delete_messages_response and delete_messages_response.data:
                message_count = len(delete_messages_response.data)
                logger.info(f"Đã xóa {message_count} tin nhắn từ chat ID {chat_id}")
            else:
                logger.info(f"Không tìm thấy tin nhắn nào để xóa cho chat ID {chat_id}")

        except Exception as e:
            logger.warning(f"Lỗi khi xóa tin nhắn cho chat ID {chat_id}: {str(e)}")
            # Vẫn tiếp tục xóa cuộc trò chuyện

        # Xóa cuộc trò chuyện
        try:
            delete_chat_response = (
                supabase.table("chats")
                .delete()
                .eq("id", chat_id)
                .eq("user_id", user_id)
                .execute()
            )

            if not delete_chat_response:
                logger.error(f"Không nhận được phản hồi khi xóa chat ID {chat_id}")
                return False

            if not delete_chat_response.data:
                logger.warning(
                    f"Không tìm thấy dữ liệu trả về khi xóa chat ID {chat_id}"
                )
                # Vẫn coi là thành công vì có thể chat đã không tồn tại
                return True

            if len(delete_chat_response.data) > 0:
                logger.info(f"Đã xóa thành công chat ID {chat_id}")
                return True

            logger.warning(f"Không thể xác nhận xóa thành công chat ID {chat_id}")
            return False

        except Exception as e:
            logger.error(f"Lỗi khi gọi API xóa chat ID {chat_id}: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Lỗi khi xóa trò chuyện ID {chat_id}: {str(e)}")
        import traceback

        logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return False


def add_message(
    chat_id: str, user_id: str, role: str, content: str
) -> Optional[Dict[str, Any]]:
    """
    Thêm một tin nhắn mới vào cuộc trò chuyện

    Args:
        chat_id (str): ID của cuộc trò chuyện
        user_id (str): ID người dùng (để kiểm tra quyền)
        role (str): Vai trò của người gửi ('user' hoặc 'assistant')
        content (str): Nội dung tin nhắn

    Returns:
        Optional[Dict[str, Any]]: Thông tin tin nhắn đã thêm hoặc None nếu có lỗi
    """
    supabase = get_supabase()
    if not supabase:
        logger.error("Không thể thêm tin nhắn: Chưa cấu hình kết nối Supabase")
        return None

    try:
        # Kiểm tra xem chat_id có phải là UUID hợp lệ không
        validated_chat_id = validate_and_format_uuid(chat_id, "chat_id")
        if not validated_chat_id:
            return None
        
        chat_id = validated_chat_id

        # Kiểm tra xem cuộc trò chuyện có thuộc về người dùng không
        max_retries = 3
        retry_count = 0
        chat = None
        
        while chat is None and retry_count < max_retries:
            try:
                supabase = get_supabase()  # Lấy client mới mỗi lần thử lại
                chat = get_chat(chat_id, user_id)
                
                if chat is None:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.warning(
                            f"Không thể thêm tin nhắn: Không tìm thấy trò chuyện ID {chat_id} cho user ID {user_id}"
                        )
                        return None
                    else:
                        logger.info(f"Đang thử lấy thông tin chat lần {retry_count+1}/{max_retries}...")
                        import time
                        time.sleep(1)  # Chờ 1 giây trước khi thử lại
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                logger.error(f"Lỗi khi kiểm tra chat (lần thử {retry_count}/{max_retries}): {error_message}")
                
                if "Server disconnected" in error_message:
                    logger.info("Đang thử kết nối lại với Supabase...")
                    import time
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại
                elif retry_count >= max_retries:
                    return None

        # Lấy thời gian hiện tại
        current_time = datetime.utcnow().isoformat()

        # Tạo dữ liệu tin nhắn
        message_data = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "created_at": current_time,
        }

        # Thêm tin nhắn mới
        logger.info(
            f"Đang thêm tin nhắn mới vào chat ID {chat_id}, vai trò: {role}, nội dung: {content[:50] if len(content) > 50 else content}..."
        )

        # Thử thêm tin nhắn với cơ chế retry
        max_message_retries = 3
        message_retry_count = 0
        message_success = False
        message_response = None
        
        while not message_success and message_retry_count < max_message_retries:
            try:
                supabase = get_supabase()  # Lấy client mới mỗi lần thử lại
                message_response = supabase.table("messages").insert(message_data).execute()
                message_success = True
            except Exception as e:
                message_retry_count += 1
                error_message = str(e)
                logger.error(f"Lỗi khi gọi API thêm tin nhắn (lần thử {message_retry_count}/{max_message_retries}): {error_message}")
                
                if "Server disconnected" in error_message:
                    logger.info("Đang thử kết nối lại với Supabase...")
                    import time
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại
                elif message_retry_count >= max_message_retries:
                    return None

        if not message_success or not message_response or not message_response.data:
            logger.error(f"Không nhận được phản hồi hợp lệ khi thêm tin nhắn sau {max_message_retries} lần thử")
            return None

        # Cập nhật thông tin cuộc trò chuyện (last_message và updated_at)
        # Trích xuất tin nhắn ngắn để hiển thị
        short_message = content
        if len(short_message) > 100:
            short_message = short_message[:97] + "..."

        chat_update_data = {"last_message": short_message, "updated_at": current_time}

        # Thử cập nhật chat với cơ chế retry
        update_retries = 2  # Ít lần thử hơn vì đây là thao tác thứ cấp
        update_retry_count = 0
        update_success = False
        
        while not update_success and update_retry_count < update_retries:
            try:
                supabase = get_supabase()  # Lấy client mới mỗi lần thử lại
                update_response = (
                    supabase.table("chats")
                    .update(chat_update_data)
                    .eq("id", chat_id)
                    .execute()
                )

                if update_response and update_response.data:
                    logger.info(
                        f"Đã cập nhật thông tin chat ID {chat_id} với last_message và updated_at mới"
                    )
                    update_success = True
                else:
                    logger.warning(
                        f"Không thể cập nhật thông tin chat ID {chat_id}, có thể cần kiểm tra truy vấn"
                    )
                    update_retry_count += 1
                    if update_retry_count < update_retries:
                        import time
                        time.sleep(1)  # Chờ 1 giây trước khi thử lại
            except Exception as e:
                update_retry_count += 1
                logger.warning(f"Lỗi khi cập nhật thông tin chat ID {chat_id} (lần thử {update_retry_count}/{update_retries}): {str(e)}")
                if update_retry_count < update_retries:
                    import time
                    time.sleep(1)  # Chờ 1 giây trước khi thử lại

        if len(message_response.data) > 0:
            message_id = message_response.data[0].get("id")
            logger.info(
                f"Đã thêm tin nhắn mới thành công vào trò chuyện ID {chat_id}, message ID: {message_id}"
            )
            return message_response.data[0]

        logger.warning(
            f"Không thể thêm tin nhắn vào trò chuyện ID {chat_id}, phản hồi từ API không chứa dữ liệu"
        )
        return None

    except Exception as e:
        logger.error(f"Lỗi khi thêm tin nhắn vào trò chuyện ID {chat_id}: {str(e)}")
        import traceback

        logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return None


def get_messages(chat_id: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Lấy danh sách tin nhắn của một cuộc trò chuyện

    Args:
        chat_id (str): ID của cuộc trò chuyện
        user_id (str): ID người dùng (để kiểm tra quyền)

    Returns:
        List[Dict[str, Any]]: Danh sách các tin nhắn
    """
    supabase = get_supabase()
    if not supabase:
        logger.error("Không thể lấy danh sách tin nhắn: Chưa cấu hình kết nối Supabase")
        return []

    try:
        # Kiểm tra xem chat_id có phải là UUID hợp lệ không
        validated_chat_id = validate_and_format_uuid(chat_id, "chat_id")
        if not validated_chat_id:
            return []
        
        chat_id = validated_chat_id

        # Kiểm tra xem cuộc trò chuyện có thuộc về người dùng không
        chat = get_chat(chat_id, user_id)
        if not chat:
            logger.warning(
                f"Không thể lấy tin nhắn: Không tìm thấy trò chuyện ID {chat_id} cho user ID {user_id}"
            )
            return []

        logger.info(f"Đang lấy tin nhắn cho chat ID {chat_id} của user ID {user_id}")

        try:
            response = (
                supabase.table("messages")
                .select("*")
                .eq("chat_id", chat_id)
                .order("created_at", desc=False)
                .execute()
            )
        except Exception as e:
            logger.error(f"Lỗi khi gọi API lấy tin nhắn từ chat ID {chat_id}: {str(e)}")
            return []

        if not response:
            logger.warning("Không nhận được phản hồi từ Supabase khi lấy tin nhắn")
            return []

        if not response.data:
            logger.info(f"Không tìm thấy tin nhắn nào cho chat ID {chat_id}")
            return []

        message_count = len(response.data)
        logger.info(f"Đã lấy {message_count} tin nhắn từ trò chuyện ID {chat_id}")
        return response.data

    except Exception as e:
        logger.error(f"Lỗi khi lấy tin nhắn từ trò chuyện ID {chat_id}: {str(e)}")
        import traceback

        logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return []


def migrate_chat_history(user_id: str, chat_history_json: str) -> Tuple[int, List[str]]:
    """
    Di chuyển lịch sử trò chuyện từ localStorage sang Supabase

    Args:
        user_id (str): ID người dùng
        chat_history_json (str): Chuỗi JSON chứa lịch sử trò chuyện từ localStorage

    Returns:
        Tuple[int, List[str]]: (Số lượng cuộc trò chuyện đã di chuyển, danh sách lỗi nếu có)
    """
    supabase = get_supabase()
    if not supabase:
        logger.error(
            "Không thể di chuyển lịch sử trò chuyện: Chưa cấu hình kết nối Supabase"
        )
        return 0, ["Chưa cấu hình kết nối Supabase"]

    errors = []
    migrated_count = 0

    try:
        chat_history = json.loads(chat_history_json)

        for conversation_id, conversation in chat_history.items():
            try:
                # Tạo cuộc trò chuyện mới
                title = conversation.get("title", "Cuộc trò chuyện đã nhập")

                new_chat = create_chat(user_id, title)
                if not new_chat:
                    errors.append(f"Không thể tạo cuộc trò chuyện: {title}")
                    continue

                # Thêm các tin nhắn vào cuộc trò chuyện
                messages = conversation.get("messages", [])
                for message in messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")

                    if not add_message(new_chat["id"], user_id, role, content):
                        errors.append(
                            f"Không thể thêm tin nhắn vào cuộc trò chuyện: {title}"
                        )

                migrated_count += 1
                logger.info(
                    f"Đã di chuyển cuộc trò chuyện {title} với {len(messages)} tin nhắn"
                )

            except Exception as e:
                errors.append(f"Lỗi khi di chuyển cuộc trò chuyện: {str(e)}")
                logger.error(
                    f"Lỗi khi di chuyển cuộc trò chuyện ID {conversation_id}: {str(e)}"
                )

        return migrated_count, errors

    except json.JSONDecodeError as e:
        logger.error(f"Lỗi khi phân tích JSON lịch sử trò chuyện: {str(e)}")
        return 0, [f"Lỗi khi phân tích JSON lịch sử trò chuyện: {str(e)}"]

    except Exception as e:
        logger.error(f"Lỗi không xác định khi di chuyển lịch sử trò chuyện: {str(e)}")
        return 0, [f"Lỗi không xác định: {str(e)}"]


def get_full_chat_history(user_id: str) -> Dict[str, Any]:
    """
    Lấy toàn bộ lịch sử chat của người dùng, bao gồm cả tin nhắn

    Args:
        user_id (str): ID người dùng

    Returns:
        Dict[str, Any]: Lịch sử chat đã được định dạng cho frontend
    """
    supabase = get_supabase()
    if not supabase:
        logger.error("Không thể lấy lịch sử trò chuyện: Chưa cấu hình kết nối Supabase")
        return {}

    try:
        # Kiểm tra user_id có hợp lệ không
        validated_user_id = validate_and_format_uuid(user_id, "user_id")
        if not validated_user_id:
            return {}
        
        user_id = validated_user_id

        logger.info(f"Đang lấy toàn bộ lịch sử chat cho user ID {user_id}")

        # Lấy danh sách các cuộc trò chuyện
        try:
            chats = get_chats(user_id)
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách cuộc trò chuyện: {str(e)}")
            return {}

        if not chats:
            logger.info(f"Không tìm thấy cuộc trò chuyện nào cho user ID {user_id}")
            return {}

        logger.info(
            f"Tìm thấy {len(chats)} cuộc trò chuyện cho user ID {user_id}, đang lấy chi tiết tin nhắn"
        )

        result = {}

        # Với mỗi cuộc trò chuyện, lấy tin nhắn
        for chat in chats:
            try:
                chat_id = chat["id"]
                messages = get_messages(chat_id, user_id)

                # Định dạng kết quả
                result[chat_id] = {
                    "id": chat_id,
                    "title": chat.get("title", "Cuộc trò chuyện không tiêu đề"),
                    "last_message": chat.get("last_message", ""),
                    "timestamp": chat.get("updated_at", chat.get("created_at")),
                    "messages": [
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                            "timestamp": msg.get("created_at"),
                        }
                        for msg in messages
                    ],
                }
                logger.info(f"Đã lấy {len(messages)} tin nhắn cho chat ID {chat_id}")
            except Exception as e:
                logger.error(
                    f"Lỗi khi xử lý chat ID {chat.get('id', 'unknown')}: {str(e)}"
                )
                # Tiếp tục với chat tiếp theo
                continue

        logger.info(
            f"Đã hoàn thành việc lấy {len(result)} cuộc trò chuyện cho user ID {user_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử trò chuyện cho user ID {user_id}: {str(e)}")
        import traceback

        logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return {}


def sync_chat_with_local_storage(user_id: str, chat_data: Dict[str, Any]) -> bool:
    """
    Đồng bộ hóa một cuộc trò chuyện từ localStorage lên Supabase

    Args:
        user_id (str): ID người dùng
        chat_data (Dict[str, Any]): Dữ liệu cuộc trò chuyện từ localStorage

    Returns:
        bool: True nếu đồng bộ thành công, False nếu có lỗi
    """
    supabase = get_supabase()
    if not supabase:
        logger.error("Không thể đồng bộ trò chuyện: Chưa cấu hình kết nối Supabase")
        return False

    try:
        # Kiểm tra dữ liệu chat hợp lệ
        if not chat_data or not isinstance(chat_data, dict):
            logger.error(f"Dữ liệu chat không hợp lệ: {type(chat_data)}")
            return False

        # Kiểm tra user_id hợp lệ
        validated_user_id = validate_and_format_uuid(user_id, "user_id")
        if not validated_user_id:
            return False
        
        user_id = validated_user_id

        chat_id = chat_data.get("id")
        title = chat_data.get("title", "Cuộc trò chuyện không tiêu đề")
        messages = chat_data.get("messages", [])

        if not chat_id:
            logger.error("Không thể đồng bộ chat không có ID")
            return False

        logger.info(
            f"Đang đồng bộ chat ID {chat_id} với tiêu đề '{title}' cho user ID {user_id}"
        )

        # Kiểm tra xem cuộc trò chuyện đã tồn tại chưa (dựa vào ID từ localStorage)
        try:
            existing_chats = (
                supabase.table("chats").select("id").eq("user_id", user_id).execute()
            )

            if not existing_chats or not existing_chats.data:
                logger.info("Không tìm thấy chat nào cho user này, đang tạo mới")
            else:
                existing_chat_ids = [chat["id"] for chat in existing_chats.data]

                # Nếu chat đã tồn tại, bỏ qua (tránh trùng lặp)
                if chat_id in existing_chat_ids:
                    logger.info(f"Bỏ qua cuộc trò chuyện ID {chat_id} đã tồn tại")
                    return True
        except Exception as e:
            logger.warning(f"Lỗi khi kiểm tra chat hiện có: {str(e)}")
            # Vẫn tiếp tục tạo chat mới

        # Tạo cuộc trò chuyện mới
        logger.info(f"Tạo chat mới từ chat local với tiêu đề: {title}")
        new_chat = create_chat(user_id, title)
        if not new_chat:
            logger.error(f"Không thể tạo cuộc trò chuyện mới cho user ID {user_id}")
            return False

        new_chat_id = new_chat.get("id")
        if not new_chat_id:
            logger.error("Chat mới được tạo nhưng không có ID")
            return False

        # Thêm các tin nhắn
        logger.info(f"Đang thêm {len(messages)} tin nhắn vào chat mới ID {new_chat_id}")
        message_success_count = 0
        message_fail_count = 0

        for index, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", "")

            if not content:
                logger.warning(f"Bỏ qua tin nhắn trống ở vị trí {index}")
                continue

            try:
                message_result = add_message(new_chat_id, user_id, role, content)
                if message_result:
                    message_success_count += 1
                else:
                    message_fail_count += 1
                    logger.warning(
                        f"Không thể thêm tin nhắn thứ {index} vào chat ID {new_chat_id}"
                    )
            except Exception as e:
                message_fail_count += 1
                logger.error(f"Lỗi khi thêm tin nhắn thứ {index}: {str(e)}")

        logger.info(
            f"Đã đồng bộ chat '{title}' với {message_success_count} tin nhắn thành công, {message_fail_count} thất bại"
        )
        return (
            message_fail_count == 0 or message_success_count > 0
        )  # Thành công nếu không có lỗi hoặc ít nhất 1 tin nhắn được thêm

    except Exception as e:
        logger.error(f"Lỗi khi đồng bộ cuộc trò chuyện cho user ID {user_id}: {str(e)}")
        import traceback

        logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return False


def sync_all_chats_with_local_storage(
    user_id: str, chat_history_json: str
) -> Tuple[int, List[str]]:
    """
    Đồng bộ hóa tất cả cuộc trò chuyện từ localStorage lên Supabase

    Args:
        user_id (str): ID người dùng
        chat_history_json (str): Chuỗi JSON chứa lịch sử trò chuyện từ localStorage

    Returns:
        Tuple[int, List[str]]: (Số lượng cuộc trò chuyện đã đồng bộ, danh sách lỗi nếu có)
    """
    supabase = get_supabase()
    if not supabase:
        logger.error(
            "Không thể đồng bộ lịch sử trò chuyện: Chưa cấu hình kết nối Supabase"
        )
        return 0, ["Chưa cấu hình kết nối Supabase"]

    errors = []
    synced_count = 0

    try:
        # Kiểm tra user_id hợp lệ
        validated_user_id = validate_and_format_uuid(user_id, "user_id")
        if not validated_user_id:
            error_msg = f"sync_all_chats_with_local_storage: user_id không hợp lệ: {user_id}, lỗi: {str(e)}"
            logger.error(error_msg)
            return 0, [error_msg]
        
        user_id = validated_user_id

        # Kiểm tra chat_history_json
        if not chat_history_json or not isinstance(chat_history_json, str):
            error_msg = f"Dữ liệu JSON không hợp lệ: {type(chat_history_json)}"
            logger.error(error_msg)
            return 0, [error_msg]

        logger.info(
            f"Đang chuẩn bị đồng bộ tất cả chat từ localStorage cho user ID {user_id}"
        )

        try:
            chat_history = json.loads(chat_history_json)
        except json.JSONDecodeError as e:
            error_msg = f"Lỗi khi phân tích JSON lịch sử trò chuyện: {str(e)}"
            logger.error(error_msg)
            return 0, [error_msg]

        if not chat_history or not isinstance(chat_history, dict):
            error_msg = (
                f"Dữ liệu JSON không chứa lịch sử chat hợp lệ: {type(chat_history)}"
            )
            logger.error(error_msg)
            return 0, [error_msg]

        chat_count = len(chat_history)
        logger.info(f"Tìm thấy {chat_count} cuộc trò chuyện cần đồng bộ")

        if chat_count == 0:
            logger.info("Không có cuộc trò chuyện nào để đồng bộ")
            return 0, []

        # Đồng bộ từng cuộc trò chuyện
        for chat_id, chat_data in chat_history.items():
            try:
                logger.info(f"Đang đồng bộ chat ID {chat_id}...")

                if not chat_data or not isinstance(chat_data, dict):
                    errors.append(f"Chat ID {chat_id}: Dữ liệu không hợp lệ")
                    logger.warning(f"Bỏ qua chat ID {chat_id}: Dữ liệu không hợp lệ")
                    continue

                if sync_chat_with_local_storage(user_id, chat_data):
                    synced_count += 1
                    logger.info(f"Đã đồng bộ thành công chat ID {chat_id}")
                else:
                    errors.append(f"Không thể đồng bộ cuộc trò chuyện ID {chat_id}")
                    logger.warning(f"Không thể đồng bộ chat ID {chat_id}")
            except Exception as e:
                error_msg = f"Lỗi khi đồng bộ cuộc trò chuyện ID {chat_id}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        logger.info(
            f"Đã hoàn thành đồng bộ {synced_count}/{chat_count} chat, với {len(errors)} lỗi"
        )
        return synced_count, errors

    except Exception as e:
        error_msg = f"Lỗi không xác định khi đồng bộ lịch sử trò chuyện: {str(e)}"
        logger.error(error_msg)
        import traceback

        logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        return 0, [error_msg]


# Kết thúc module
