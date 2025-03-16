import os
import logging
import json
from datetime import datetime
from .supabase_client import get_supabase_client

# Thiết lập logging
logger = logging.getLogger('document_storage')

def upload_document(file_data, filename, user_id=None, metadata=None):
    """
    Upload tài liệu lên Supabase Storage
    
    Args:
        file_data (bytes): Dữ liệu file
        filename (str): Tên file
        user_id (str, optional): ID của người dùng
        metadata (dict, optional): Metadata của tài liệu
    
    Returns:
        dict: Thông tin về tài liệu đã upload
    """
    try:
        client = get_supabase_client()
        
        # Tạo đường dẫn lưu trữ
        storage_path = f"documents/{user_id or 'public'}/{filename}"
        
        # Upload file lên Supabase Storage
        response = client.storage.from_("documents").upload(
            storage_path,
            file_data,
            {"content-type": "application/octet-stream"}
        )
        
        # Lấy URL công khai của file
        file_url = client.storage.from_("documents").get_public_url(storage_path)
        
        # Lưu thông tin tài liệu vào bảng documents
        document_data = {
            "filename": filename,
            "storage_path": storage_path,
            "file_url": file_url,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        result = client.table("documents").insert(document_data).execute()
        
        if result.data and len(result.data) > 0:
            return {
                "status": "success",
                "document_id": result.data[0]["id"],
                "file_url": file_url,
                "message": "Upload tài liệu thành công"
            }
        else:
            return {
                "status": "error",
                "message": "Lỗi khi lưu thông tin tài liệu"
            }
    except Exception as e:
        logger.error(f"Lỗi khi upload tài liệu: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_document(document_id):
    """
    Lấy thông tin tài liệu
    
    Args:
        document_id (int): ID của tài liệu
    
    Returns:
        dict: Thông tin tài liệu
    """
    try:
        client = get_supabase_client()
        
        response = client.table("documents").select("*").eq("id", document_id).execute()
        
        if response.data and len(response.data) > 0:
            return {
                "status": "success",
                "document": response.data[0]
            }
        else:
            return {
                "status": "error",
                "message": "Không tìm thấy tài liệu"
            }
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin tài liệu: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_user_documents(user_id):
    """
    Lấy danh sách tài liệu của người dùng
    
    Args:
        user_id (str): ID của người dùng
    
    Returns:
        dict: Danh sách tài liệu
    """
    try:
        client = get_supabase_client()
        
        response = client.table("documents").select("*").eq("user_id", user_id).execute()
        
        return {
            "status": "success",
            "documents": response.data
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách tài liệu: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def delete_document(document_id, user_id=None):
    """
    Xóa tài liệu
    
    Args:
        document_id (int): ID của tài liệu
        user_id (str, optional): ID của người dùng (để kiểm tra quyền)
    
    Returns:
        dict: Kết quả xóa tài liệu
    """
    try:
        client = get_supabase_client()
        
        # Lấy thông tin tài liệu
        document_response = client.table("documents").select("*").eq("id", document_id).execute()
        
        if not document_response.data or len(document_response.data) == 0:
            return {
                "status": "error",
                "message": "Không tìm thấy tài liệu"
            }
        
        document = document_response.data[0]
        
        # Kiểm tra quyền (nếu cần)
        if user_id and document["user_id"] != user_id:
            return {
                "status": "error",
                "message": "Không có quyền xóa tài liệu này"
            }
        
        # Xóa file từ Storage
        storage_path = document["storage_path"]
        client.storage.from_("documents").remove([storage_path])
        
        # Xóa thông tin tài liệu từ bảng documents
        client.table("documents").delete().eq("id", document_id).execute()
        
        # Xóa các chunks và embeddings liên quan
        client.table("document_chunks").delete().eq("document_id", document_id).execute()
        
        return {
            "status": "success",
            "message": "Xóa tài liệu thành công"
        }
    except Exception as e:
        logger.error(f"Lỗi khi xóa tài liệu: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def update_document_metadata(document_id, metadata, user_id=None):
    """
    Cập nhật metadata của tài liệu
    
    Args:
        document_id (int): ID của tài liệu
        metadata (dict): Metadata mới
        user_id (str, optional): ID của người dùng (để kiểm tra quyền)
    
    Returns:
        dict: Kết quả cập nhật
    """
    try:
        client = get_supabase_client()
        
        # Kiểm tra quyền (nếu cần)
        if user_id:
            document_response = client.table("documents").select("user_id").eq("id", document_id).execute()
            
            if not document_response.data or len(document_response.data) == 0:
                return {
                    "status": "error",
                    "message": "Không tìm thấy tài liệu"
                }
            
            if document_response.data[0]["user_id"] != user_id:
                return {
                    "status": "error",
                    "message": "Không có quyền cập nhật tài liệu này"
                }
        
        # Cập nhật metadata
        response = client.table("documents").update({
            "metadata": json.dumps(metadata),
            "updated_at": datetime.now().isoformat()
        }).eq("id", document_id).execute()
        
        if response.data and len(response.data) > 0:
            return {
                "status": "success",
                "document": response.data[0],
                "message": "Cập nhật metadata thành công"
            }
        else:
            return {
                "status": "error",
                "message": "Lỗi khi cập nhật metadata"
            }
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật metadata: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 