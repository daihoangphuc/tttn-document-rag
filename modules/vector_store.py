import os
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .supabase_client import get_supabase_client

# Thiết lập logging
logger = logging.getLogger('vector_store')

def create_embeddings_table():
    """
    Tạo bảng và các chức năng cần thiết cho vector embeddings trong Supabase
    
    Returns:
        dict: Kết quả tạo bảng
    """
    try:
        client = get_supabase_client()
        
        # Kiểm tra xem bảng đã tồn tại chưa
        # Lưu ý: Đây là một cách đơn giản để kiểm tra, trong thực tế bạn có thể cần sử dụng
        # các API khác của Supabase để kiểm tra cấu trúc bảng
        try:
            client.table("document_embeddings").select("id").limit(1).execute()
            logger.info("Bảng document_embeddings đã tồn tại")
            return {
                "status": "success",
                "message": "Bảng document_embeddings đã tồn tại"
            }
        except Exception:
            # Bảng chưa tồn tại, tiếp tục tạo
            pass
        
        # Lưu ý: Trong phiên bản mới của Supabase, bạn cần tạo bảng thông qua giao diện web
        # hoặc sử dụng các công cụ migration. Không thể thực hiện trực tiếp từ API.
        logger.info("Bạn cần tạo bảng document_embeddings thông qua giao diện Supabase")
        return {
            "status": "warning",
            "message": "Bạn cần tạo bảng document_embeddings thông qua giao diện Supabase"
        }
    except Exception as e:
        logger.error(f"Lỗi khi tạo bảng document_embeddings: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def store_embeddings(document_id: int, chunk_id: int, embedding: List[float], content: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Lưu vector embedding vào Supabase
    
    Args:
        document_id (int): ID của tài liệu
        chunk_id (int): ID của chunk
        embedding (List[float]): Vector embedding
        content (str): Nội dung text của chunk
        metadata (Dict, optional): Metadata của chunk
    
    Returns:
        dict: Kết quả lưu embedding
    """
    try:
        client = get_supabase_client()
        
        # Chuyển đổi embedding thành định dạng phù hợp
        embedding_str = str(embedding).replace('[', '{').replace(']', '}')
        
        # Chuẩn bị dữ liệu
        embedding_data = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "embedding": embedding_str,
            "content": content,
            "metadata": json.dumps(metadata or {})
        }
        
        # Lưu vào bảng document_embeddings
        result = client.table("document_embeddings").insert(embedding_data).execute()
        
        if result.data and len(result.data) > 0:
            return {
                "status": "success",
                "embedding_id": result.data[0]["id"],
                "message": "Lưu embedding thành công"
            }
        else:
            return {
                "status": "error",
                "message": "Lỗi khi lưu embedding"
            }
    except Exception as e:
        logger.error(f"Lỗi khi lưu embedding: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def search_similar_vectors(query_embedding: List[float], limit: int = 5, threshold: float = 0.7, filter_metadata: Optional[Dict[str, Any]] = None):
    """
    Tìm kiếm các vector tương tự
    
    Args:
        query_embedding (List[float]): Vector embedding của câu query
        limit (int, optional): Số lượng kết quả tối đa
        threshold (float, optional): Ngưỡng độ tương đồng (0-1)
        filter_metadata (Dict, optional): Lọc theo metadata
    
    Returns:
        dict: Kết quả tìm kiếm
    """
    try:
        client = get_supabase_client()
        
        # Chuyển đổi embedding thành định dạng phù hợp
        embedding_str = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Xây dựng câu truy vấn SQL
        sql = f"""
        select 
            id,
            document_id,
            chunk_id,
            content,
            metadata,
            1 - (embedding <=> '{embedding_str}') as similarity
        from document_embeddings
        where 1 - (embedding <=> '{embedding_str}') > {threshold}
        """
        
        # Thêm điều kiện lọc metadata nếu có
        if filter_metadata:
            for key, value in filter_metadata.items():
                if isinstance(value, str):
                    sql += f" and metadata->'{key}' = '\"{value}\"'"
                else:
                    sql += f" and metadata->'{key}' = '{json.dumps(value)}'"
        
        # Hoàn thiện câu truy vấn
        sql += f"""
        order by similarity desc
        limit {limit};
        """
        
        # Thực thi truy vấn
        result = client.query(sql).execute()
        
        return {
            "status": "success",
            "results": result.data,
            "count": len(result.data)
        }
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm vector: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def delete_document_embeddings(document_id: int):
    """
    Xóa tất cả embeddings của một tài liệu
    
    Args:
        document_id (int): ID của tài liệu
    
    Returns:
        dict: Kết quả xóa
    """
    try:
        client = get_supabase_client()
        
        # Xóa tất cả embeddings của tài liệu
        client.table("document_embeddings").delete().eq("document_id", document_id).execute()
        
        return {
            "status": "success",
            "message": f"Đã xóa tất cả embeddings của tài liệu {document_id}"
        }
    except Exception as e:
        logger.error(f"Lỗi khi xóa embeddings: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_document_embeddings(document_id: int):
    """
    Lấy tất cả embeddings của một tài liệu
    
    Args:
        document_id (int): ID của tài liệu
    
    Returns:
        dict: Danh sách embeddings
    """
    try:
        client = get_supabase_client()
        
        response = client.table("document_embeddings").select(
            "id", "chunk_id", "content", "metadata"
        ).eq("document_id", document_id).execute()
        
        return {
            "status": "success",
            "embeddings": response.data,
            "count": len(response.data)
        }
    except Exception as e:
        logger.error(f"Lỗi khi lấy embeddings: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 