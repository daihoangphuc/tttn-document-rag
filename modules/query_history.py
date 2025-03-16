import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from .supabase_client import get_supabase_client

# Thiết lập logging
logger = logging.getLogger('query_history')

def create_query_history_table():
    """
    Tạo bảng lịch sử truy vấn trong Supabase
    
    Returns:
        dict: Kết quả tạo bảng
    """
    try:
        client = get_supabase_client()
        
        # Kiểm tra xem bảng đã tồn tại chưa
        try:
            client.table("query_history").select("id").limit(1).execute()
            logger.info("Bảng query_history đã tồn tại")
            return {
                "status": "success",
                "message": "Bảng query_history đã tồn tại"
            }
        except Exception:
            # Bảng chưa tồn tại, tiếp tục tạo
            pass
        
        # Lưu ý: Trong phiên bản mới của Supabase, bạn cần tạo bảng thông qua giao diện web
        # hoặc sử dụng các công cụ migration. Không thể thực hiện trực tiếp từ API.
        logger.info("Bạn cần tạo bảng query_history thông qua giao diện Supabase")
        return {
            "status": "warning",
            "message": "Bạn cần tạo bảng query_history thông qua giao diện Supabase"
        }
    except Exception as e:
        logger.error(f"Lỗi khi tạo bảng query_history: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def save_query(query: str, response: str = None, user_id: str = None, documents: List[Dict[str, Any]] = None, metadata: Dict[str, Any] = None):
    """
    Lưu truy vấn vào lịch sử
    
    Args:
        query (str): Câu truy vấn
        response (str, optional): Câu trả lời
        user_id (str, optional): ID của người dùng
        documents (List[Dict], optional): Danh sách tài liệu liên quan
        metadata (Dict, optional): Metadata bổ sung
    
    Returns:
        dict: Kết quả lưu truy vấn
    """
    try:
        client = get_supabase_client()
        
        # Chuẩn bị dữ liệu
        query_data = {
            "query": query,
            "user_id": user_id,
            "created_at": datetime.now().isoformat()
        }
        
        if response:
            query_data["response"] = response
            
        if documents:
            query_data["documents"] = json.dumps(documents)
            
        if metadata:
            query_data["metadata"] = json.dumps(metadata)
        
        # Lưu vào bảng query_history
        result = client.table("query_history").insert(query_data).execute()
        
        if result.data and len(result.data) > 0:
            return {
                "status": "success",
                "query_id": result.data[0]["id"],
                "message": "Lưu truy vấn thành công"
            }
        else:
            return {
                "status": "error",
                "message": "Lỗi khi lưu truy vấn"
            }
    except Exception as e:
        logger.error(f"Lỗi khi lưu truy vấn: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def update_query_response(query_id: int, response: str, documents: List[Dict[str, Any]] = None):
    """
    Cập nhật câu trả lời cho truy vấn
    
    Args:
        query_id (int): ID của truy vấn
        response (str): Câu trả lời
        documents (List[Dict], optional): Danh sách tài liệu liên quan
    
    Returns:
        dict: Kết quả cập nhật
    """
    try:
        client = get_supabase_client()
        
        # Chuẩn bị dữ liệu
        update_data = {
            "response": response
        }
        
        if documents:
            update_data["documents"] = json.dumps(documents)
        
        # Cập nhật vào bảng query_history
        result = client.table("query_history").update(update_data).eq("id", query_id).execute()
        
        if result.data and len(result.data) > 0:
            return {
                "status": "success",
                "message": "Cập nhật câu trả lời thành công"
            }
        else:
            return {
                "status": "error",
                "message": "Lỗi khi cập nhật câu trả lời"
            }
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật câu trả lời: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def save_feedback(query_id: int, feedback: Dict[str, Any]):
    """
    Lưu phản hồi của người dùng về câu trả lời
    
    Args:
        query_id (int): ID của truy vấn
        feedback (Dict): Phản hồi của người dùng
    
    Returns:
        dict: Kết quả lưu phản hồi
    """
    try:
        client = get_supabase_client()
        
        # Cập nhật vào bảng query_history
        result = client.table("query_history").update({
            "feedback": json.dumps(feedback)
        }).eq("id", query_id).execute()
        
        if result.data and len(result.data) > 0:
            return {
                "status": "success",
                "message": "Lưu phản hồi thành công"
            }
        else:
            return {
                "status": "error",
                "message": "Lỗi khi lưu phản hồi"
            }
    except Exception as e:
        logger.error(f"Lỗi khi lưu phản hồi: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_user_query_history(user_id: str, limit: int = 20, offset: int = 0):
    """
    Lấy lịch sử truy vấn của người dùng
    
    Args:
        user_id (str): ID của người dùng
        limit (int, optional): Số lượng kết quả tối đa
        offset (int, optional): Vị trí bắt đầu
    
    Returns:
        dict: Lịch sử truy vấn
    """
    try:
        client = get_supabase_client()
        
        # Lấy tổng số truy vấn
        count_response = client.table("query_history").select(
            "count", count="exact"
        ).eq("user_id", user_id).execute()
        
        total_count = count_response.count if hasattr(count_response, 'count') else 0
        
        # Lấy danh sách truy vấn
        response = client.table("query_history").select(
            "id", "query", "response", "created_at", "feedback"
        ).eq("user_id", user_id).order("created_at", desc=True).range(offset, offset + limit - 1).execute()
        
        return {
            "status": "success",
            "queries": response.data,
            "count": len(response.data),
            "total_count": total_count
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử truy vấn: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_query_details(query_id: int):
    """
    Lấy chi tiết của một truy vấn
    
    Args:
        query_id (int): ID của truy vấn
    
    Returns:
        dict: Chi tiết truy vấn
    """
    try:
        client = get_supabase_client()
        
        response = client.table("query_history").select("*").eq("id", query_id).execute()
        
        if response.data and len(response.data) > 0:
            return {
                "status": "success",
                "query": response.data[0]
            }
        else:
            return {
                "status": "error",
                "message": "Không tìm thấy truy vấn"
            }
    except Exception as e:
        logger.error(f"Lỗi khi lấy chi tiết truy vấn: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def delete_query(query_id: int, user_id: str = None):
    """
    Xóa một truy vấn
    
    Args:
        query_id (int): ID của truy vấn
        user_id (str, optional): ID của người dùng (để kiểm tra quyền)
    
    Returns:
        dict: Kết quả xóa
    """
    try:
        client = get_supabase_client()
        
        # Kiểm tra quyền (nếu cần)
        if user_id:
            query_response = client.table("query_history").select("user_id").eq("id", query_id).execute()
            
            if not query_response.data or len(query_response.data) == 0:
                return {
                    "status": "error",
                    "message": "Không tìm thấy truy vấn"
                }
            
            if query_response.data[0]["user_id"] != user_id:
                return {
                    "status": "error",
                    "message": "Không có quyền xóa truy vấn này"
                }
        
        # Xóa truy vấn
        client.table("query_history").delete().eq("id", query_id).execute()
        
        return {
            "status": "success",
            "message": "Xóa truy vấn thành công"
        }
    except Exception as e:
        logger.error(f"Lỗi khi xóa truy vấn: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 