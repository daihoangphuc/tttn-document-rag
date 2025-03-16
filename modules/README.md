# Supabase Integration Modules

Các module này cung cấp tích hợp Supabase cho hệ thống RAG (Retrieval-Augmented Generation), bao gồm các chức năng:

- Quản lý người dùng (đăng ký, đăng nhập, cập nhật thông tin)
- Lưu trữ tài liệu
- Vector database cho embeddings
- Lưu trữ lịch sử truy vấn

## Cài đặt

Đảm bảo bạn đã cài đặt các thư viện cần thiết:

```bash
pip install supabase python-dotenv
```

## Cấu hình

Tạo file `.env` trong thư mục gốc của dự án với các thông tin sau:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-api-key
```

## Cấu trúc module

```
modules/
├── __init__.py             # Export các hàm chính
├── supabase_client.py      # Kết nối đến Supabase
├── user_management.py      # Quản lý người dùng
├── document_storage.py     # Lưu trữ tài liệu
├── vector_store.py         # Vector database
└── query_history.py        # Lịch sử truy vấn
```

## Sử dụng cơ bản

### Kết nối đến Supabase

```python
from modules import get_supabase_client, check_connection

# Lấy client Supabase
client = get_supabase_client()

# Kiểm tra kết nối
if check_connection():
    print("Kết nối thành công!")
else:
    print("Kết nối thất bại!")
```

### Quản lý người dùng

```python
from modules import register_user, login_user, get_user_profile

# Đăng ký người dùng mới
register_result = register_user("user@example.com", "password123", {"name": "Example User"})

# Đăng nhập
login_result = login_user("user@example.com", "password123")

# Lấy thông tin người dùng
if login_result["status"] == "success":
    user_id = login_result["user"]["id"]
    profile = get_user_profile(user_id)
```

### Lưu trữ tài liệu

```python
from modules import upload_document, get_user_documents

# Upload tài liệu
with open("document.pdf", "rb") as f:
    file_data = f.read()
    
upload_result = upload_document(
    file_data=file_data,
    filename="document.pdf",
    user_id="user-id",
    metadata={"description": "Tài liệu mẫu"}
)

# Lấy danh sách tài liệu của người dùng
documents = get_user_documents("user-id")
```

### Vector Database

```python
from modules import create_embeddings_table, store_embeddings, search_similar_vectors

# Tạo bảng embeddings (chỉ cần chạy một lần)
create_embeddings_table()

# Lưu embedding
embedding = [0.1, 0.2, ..., 0.5]  # Vector 1536 chiều
store_result = store_embeddings(
    document_id=1,
    chunk_id=1,
    embedding=embedding,
    content="Nội dung chunk văn bản",
    metadata={"position": 0}
)

# Tìm kiếm vector tương tự
search_result = search_similar_vectors(
    query_embedding=embedding,
    limit=5,
    threshold=0.7
)
```

### Lịch sử truy vấn

```python
from modules import save_query, get_user_query_history

# Lưu truy vấn
save_result = save_query(
    query="Câu hỏi của người dùng",
    response="Câu trả lời",
    user_id="user-id",
    documents=[{"id": 1, "title": "Tài liệu liên quan"}]
)

# Lấy lịch sử truy vấn
history = get_user_query_history("user-id", limit=10)
```

## Ví dụ đầy đủ

Xem file `examples/supabase_example.py` để biết cách sử dụng đầy đủ các module.

## Thiết lập Supabase

1. Tạo tài khoản và dự án trên [Supabase](https://supabase.com)
2. Kích hoạt extension `pgvector` trong SQL Editor:
   ```sql
   create extension if not exists vector;
   ```
3. Tạo các bảng cần thiết bằng cách chạy hàm `create_embeddings_table()` và `create_query_history_table()`

## Lưu ý

- Đảm bảo bạn đã thiết lập đúng các quyền (RLS - Row Level Security) trong Supabase để bảo vệ dữ liệu
- Sử dụng các hàm có sẵn trong module thay vì truy cập trực tiếp vào Supabase để đảm bảo tính nhất quán
- Xử lý lỗi phù hợp khi sử dụng các hàm, vì tất cả các hàm đều trả về kết quả dạng dict với trường `status` là "success" hoặc "error" 