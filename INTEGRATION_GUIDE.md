# Hướng dẫn tích hợp Supabase vào hệ thống RAG

Tài liệu này hướng dẫn cách tích hợp xác thực người dùng và quản lý dữ liệu với Supabase vào hệ thống RAG hiện tại.

## 1. Cài đặt các gói phụ thuộc

Đầu tiên, cài đặt các gói phụ thuộc cần thiết:

```bash
pip install supabase python-dotenv
```

## 2. Thiết lập biến môi trường

Tạo file `.env` trong thư mục gốc của dự án và thêm các thông tin cấu hình Supabase:

```
SUPABASE_URL=https://your-project-url.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_KEY=your-supabase-service-key
FLASK_SECRET_KEY=your-flask-secret-key
```

## 3. Thiết lập cơ sở dữ liệu Supabase

Chạy tập tin SQL trong SQL Editor của Supabase:

1. Đăng nhập vào trang quản trị Supabase
2. Chọn "SQL Editor" từ menu bên trái
3. Tạo một truy vấn mới và dán nội dung từ file `supabase_modules/setup_database.sql`
4. Chạy truy vấn để thiết lập các bảng và chính sách

## 4. Tích hợp vào hệ thống hiện có

### Bước 1: Cập nhật app.py

Ở đầu file `app.py`, thêm các import cần thiết:

```python
from flask import Flask, render_template, request, jsonify, send_from_directory, render_template_string, session, redirect, url_for, flash
from supabase_integration import setup_auth_routes, enhanced_upload_file, enhanced_remove_file, get_enhanced_index_html
import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()
```

### Bước 2: Khởi tạo ứng dụng Flask với secret key

Cập nhật khởi tạo ứng dụng Flask:

```python
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')
```

### Bước 3: Thiết lập routes xác thực

Sau khi khởi tạo ứng dụng Flask, thêm dòng:

```python
# Thiết lập routes xác thực
setup_auth_routes(app, index_html)
```

### Bước 4: Cập nhật route upload và remove

Thay thế route upload hiện tại:

```python
@app.route('/upload', methods=['POST'])
def upload_file():
    return enhanced_upload_file(UPLOAD_FOLDER, index_html, global_all_files, load_settings, extract_text_pdf, extract_text_docx, extract_text_txt, add_document, save_state)
```

Thay thế route remove hiện tại:

```python
@app.route('/remove', methods=['POST'])
def remove_file():
    return enhanced_remove_file(index_html, global_all_files, load_settings, remove_document)
```

## 5. Cập nhật giao diện người dùng

Thay thế biến `index_html` hiện tại bằng phiên bản nâng cao:

```python
index_html = get_enhanced_index_html()
```

Hoặc tích hợp các phần tử UI từ phiên bản nâng cao vào template hiện tại.

## 6. Chạy ứng dụng

Đảm bảo tất cả các thư mục template (`templates/`) đã được tạo và chứa các file HTML cần thiết:
- login.html
- register.html
- profile.html
- forgot_password.html

Sau đó, khởi động ứng dụng:

```bash
python app.py
```

## 7. Kiểm tra chức năng

Sau khi triển khai, hãy kiểm tra các chức năng:

1. Đăng ký tài khoản mới
2. Đăng nhập
3. Tải lên và quản lý file
4. Xem hồ sơ người dùng
5. Đăng xuất

## 8. Cấu trúc module Supabase

### supabase_modules/config.py
- Khởi tạo kết nối Supabase

### supabase_modules/auth.py
- Xử lý đăng ký, đăng nhập, đăng xuất
- Quản lý phiên người dùng

### supabase_modules/chat_history.py
- Quản lý lịch sử trò chuyện

### supabase_modules/file_manager.py
- Lưu trữ và quản lý file theo người dùng

### supabase_modules/helpers.py
- Các hàm tiện ích

### supabase_integration.py
- Tích hợp các module trên vào hệ thống RAG

## 9. Giải quyết vấn đề

Nếu gặp lỗi khi tích hợp, hãy kiểm tra:

1. Biến môi trường đã được thiết lập đúng
2. Cơ sở dữ liệu Supabase đã được thiết lập chính xác
3. Tất cả các routes đã được định nghĩa đúng
4. Thư mục uploads có quyền ghi

## 10. Bảo mật

- Đảm bảo sử dụng HTTPS trong môi trường sản xuất
- Không bao giờ để lộ các khóa API Supabase
- Thường xuyên thay đổi Flask secret key 