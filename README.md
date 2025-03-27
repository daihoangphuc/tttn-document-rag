# Hệ thống RAG Chatbot

## Giới thiệu
Hệ thống RAG (Retrieval-Augmented Generation) Chatbot là một ứng dụng cho phép người dùng tải lên tài liệu và đặt câu hỏi dựa trên nội dung của các tài liệu đó. Hệ thống sử dụng mô hình ngôn ngữ Gemini của Google để tạo ra câu trả lời chính xác và phù hợp với ngữ cảnh. Hệ thống đã được tích hợp với Supabase để quản lý người dùng, lưu trữ lịch sử trò chuyện và tài liệu.

## Quy trình hoạt động

### 1. Tổng quan hệ thống

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Người dùng     │────▶│  Web Interface  │────▶│  Flask Server   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────┬───────┘
                                                          │
                                                          ▼
 ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
 │                 │     │                 │     │                 │
 │  Gemini API     │◀───▶│  RAG Engine     │◀───▶│  Vector Store   │
 │                 │     │                 │     │                 │
 └─────────────────┘     └─────────────────┘     └─────────┬───────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │                 │
                                                 │  Supabase       │
                                                 │                 │
                                                 └─────────────────┘
```

### 2. Quy trình xử lý tài liệu

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│ Upload File │────▶│ Extract Text│────▶│ Chunking    │────▶│ Embedding   │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
                                                                   ▼
                                                            ┌─────────────┐
                                                            │             │
                                                            │ Vector Store│
                                                            │             │
                                                            └─────────────┘
```

### 3. Quy trình trả lời câu hỏi

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│ Câu hỏi     │────▶│ Query       │────▶│ Retrieval   │────▶│ Reranking   │
│             │     │ Transform   │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
                                                                   ▼
 ┌─────────────┐                                            ┌─────────────┐
 │             │                                            │             │
 │ Trả lời     │◀───────────────────────────────────────────│ Gemini LLM  │
 │             │                                            │             │
 └─────────────┘                                            └─────────────┘
```

### 4. Chi tiết các thành phần chính

#### 4.1. Xử lý tài liệu

1. **Upload File**:
   - Hỗ trợ các định dạng: PDF, DOCX, TXT
   - API Endpoint: `/upload`
   - Hỗ trợ lưu trữ tài liệu theo người dùng (sau khi đăng nhập)

2. **Extract Text**:
   - Trích xuất văn bản từ file
   - Các hàm: `extract_text_pdf()`, `extract_text_docx()`, `extract_text_txt()`

3. **Chunking**:
   - Chia văn bản thành các đoạn nhỏ
   - Các phương pháp: Sentence Windows, Paragraph, Semantic, Token, Adaptive, Hierarchical, Contextual, Multi-granularity, Hybrid (Mặc định)

4. **Embedding**:
   - Chuyển đổi các đoạn văn bản thành vector
   - Sử dụng mô hình: SentenceTransformer

5. **Vector Store**:
   - Lưu trữ và đánh chỉ mục các vector
   - Sử dụng FAISS để tìm kiếm hiệu quả
   - Tích hợp lưu trữ dữ liệu trong Supabase theo người dùng

#### 4.2. Trả lời câu hỏi

1. **Query Transform**:
   - Tối ưu hóa câu hỏi cho tiếng Việt
   - Hàm: `transform_query_for_vietnamese()`

2. **Retrieval**:
   - Tìm kiếm các đoạn văn bản liên quan nhất
   - Sử dụng tìm kiếm vector similarity

3. **Reranking**:
   - Sắp xếp lại kết quả để tăng độ chính xác
   - Hàm: `rerank_results_for_vietnamese()`

4. **Context Building**:
   - Xây dựng ngữ cảnh tối ưu từ các đoạn văn bản
   - Hàm: `build_optimized_context()`

5. **Gemini LLM**:
   - Tạo câu trả lời dựa trên ngữ cảnh và câu hỏi
   - Hàm: `generate_with_retry()`
   - Hỗ trợ chuyển đổi API key khi cần

#### 4.3. Quản lý người dùng và dữ liệu

1. **Xác thực người dùng**:
   - Đăng ký, đăng nhập, đăng xuất, quên mật khẩu
   - Quản lý phiên người dùng
   - API Endpoints: `/login`, `/register`, `/logout`, `/forgot-password`

2. **Quản lý hồ sơ người dùng**:
   - Xem và cập nhật thông tin cá nhân
   - Đổi mật khẩu
   - API Endpoints: `/profile`, `/api/user/profile`, `/api/user/change-password`

3. **Lịch sử trò chuyện**:
   - Lưu trữ lịch sử trò chuyện theo người dùng
   - Tạo, xem, cập nhật, xóa cuộc trò chuyện
   - API Endpoints: `/api/chat/history`, `/api/chat`, `/api/chat/<chat_id>/message`

4. **Quản lý tài liệu**:
   - Lưu trữ và quản lý tài liệu theo người dùng
   - Xóa tài liệu
   - API Endpoints: `/upload`, `/remove`, `/delete-file`

#### 5. Các tính năng đặc biệt

1. **Tối ưu hóa cho tiếng Việt**:
   - Xử lý đặc thù cho ngôn ngữ tiếng Việt
   - Sử dụng thư viện underthesea, pyvi

2. **Quản lý API Key**:
   - Hỗ trợ nhiều API key
   - Tự động chuyển đổi khi gặp lỗi quota

3. **Đánh giá hiệu suất**:
   - Theo dõi thời gian truy xuất và trả lời
   - API Endpoint: `/api/performance`

4. **Tùy chỉnh chunking**:
   - Cho phép người dùng chọn phương pháp chunking
   - Tùy chỉnh tham số qua API: `/settings`

5. **Tích hợp Supabase**:
   - Lưu trữ và quản lý dữ liệu
   - Xác thực và phân quyền người dùng
   - Đồng bộ hóa dữ liệu từ localStorage

6. **Xử lý nhiều câu hỏi**:
   - Phát hiện và xử lý câu hỏi phức hợp
   - Hàm: `split_multiple_questions()`

7. **Gợi ý câu hỏi**:
   - Tạo các câu hỏi liên quan đến tài liệu
   - Hàm: `generate_similar_questions()`

## Cấu trúc dự án
```
BE/
├── app.py                        # File chính của ứng dụng
├── supabase_integration.py       # Module tích hợp Supabase
├── .env                          # File chứa các biến môi trường và API keys
├── .env.example                  # Mẫu cho file .env
├── requirements.txt              # Danh sách các thư viện cần thiết
├── README.md                     # Tài liệu hướng dẫn
├── performance_metrics.csv       # Dữ liệu đánh giá hiệu suất
├── supabase_modules/             # Thư mục chứa các module Supabase
│   ├── __init__.py               # File khởi tạo
│   ├── auth.py                   # Module xác thực người dùng
│   ├── config.py                 # Cấu hình Supabase
│   ├── chat_history.py           # Quản lý lịch sử trò chuyện
│   ├── file_manager.py           # Quản lý file người dùng
│   ├── helpers.py                # Các hàm tiện ích
│   └── setup_database.sql        # Script SQL tạo bảng trong Supabase
├── templates/                    # Thư mục chứa các template HTML
│   ├── index.html                # Giao diện chính của chatbot
│   ├── login.html                # Trang đăng nhập
│   ├── register.html             # Trang đăng ký
│   ├── profile.html              # Trang hồ sơ người dùng
│   ├── forgot_password.html      # Trang quên mật khẩu
│   └── integration_help.html     # Trang hướng dẫn tích hợp
├── static/                       # Thư mục chứa các file tĩnh
│   └── js/                       # Thư mục JavaScript
├── uploads/                      # Thư mục lưu trữ các file được tải lên
│   ├── .gitkeep                  # File để giữ thư mục uploads trên git
│   ├── rag_state.json            # Trạng thái của hệ thống RAG
│   ├── faiss_index.bin           # Index vector FAISS
│   ├── vectors.pkl               # Vector embedding được lưu trữ
│   ├── tfidf_vectorizer.pkl      # Mô hình TF-IDF vectorizer
│   └── tfidf_matrix.pkl          # Ma trận TF-IDF
```

## Cài đặt

### Yêu cầu
- Python 3.8 trở lên
- Các thư viện Python được liệt kê trong `requirements.txt`
- Tài khoản Supabase (cho chức năng xác thực và lưu trữ)

### Các bước cài đặt
1. Clone repository:
```bash
git clone <repository-url>
cd BE/
```

2. Tạo và kích hoạt môi trường ảo:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

4. Cấu hình file `.env`:
   - Tạo file `.env` trong thư mục BE (nếu chưa có)
   - Tham khảo file `.env.example` để cấu hình
   - Thêm các API key và cấu hình cần thiết:
```
# Gemini API Keys (danh sách các key, phân tách bằng dấu phẩy)
GEMINI_API_KEYS=key1,key2,key3,key4,key5

# Ngrok Auth Token
NGROK_AUTH_TOKEN=your_ngrok_token

# Cấu hình ứng dụng
DEBUG=True
PORT=5000
HOST=0.0.0.0
FLASK_SECRET_KEY=your_secret_key_here

# Cấu hình Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_JWT_SECRET=your_jwt_secret

# Cấu hình Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
REDIRECT_URI=https://your_supabase_url.supabase.co/auth/v1/callback
```

5. Thiết lập cơ sở dữ liệu Supabase:
   - Đăng nhập vào Supabase và tạo dự án mới
   - Chạy script SQL trong file `supabase_modules/setup_database.sql` trên SQL Editor của Supabase
   - Lấy URL và API Key của dự án để cấu hình trong file `.env`

## Chạy ứng dụng

### Môi trường phát triển
```bash
python app.py
```
Ứng dụng sẽ chạy ở địa chỉ http://localhost:5000 và tạo một URL công khai thông qua ngrok.

### Môi trường sản xuất
Để triển khai ứng dụng trong môi trường sản xuất, bạn có thể sử dụng Gunicorn:

1. Cài đặt Gunicorn:
```bash
pip install gunicorn
```

2. Chạy ứng dụng với Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Hoặc sử dụng Docker:

1. Tạo Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

2. Build và chạy Docker container:
```bash
docker build -t rag-chatbot .
docker run -p 5000:5000 --env-file .env rag-chatbot
```

## Sử dụng

### Giao diện web
1. Truy cập vào URL được hiển thị khi khởi động ứng dụng
2. Đăng ký tài khoản mới hoặc đăng nhập (nếu đã có tài khoản)
3. Tải lên tài liệu (hỗ trợ định dạng .txt, .pdf, .docx)
4. Chọn phương pháp chunking phù hợp
5. Đặt câu hỏi và nhận câu trả lời

### API Endpoints

#### 1. Xác thực người dùng
- **Đăng ký**: `/register` (GET, POST)
- **Đăng nhập**: `/login` (GET, POST)
- **Đăng xuất**: `/logout` (GET)
- **Quên mật khẩu**: `/forgot-password` (GET, POST)
- **Xem hồ sơ**: `/profile` (GET)
- **Cập nhật hồ sơ**: `/api/user/profile` (POST)
- **Đổi mật khẩu**: `/api/user/change-password` (POST)

#### 2. Quản lý trò chuyện
- **Lấy lịch sử chat**: `/api/chat/history` (GET)
- **Tạo chat mới**: `/api/chat` (POST)
- **Thêm tin nhắn**: `/api/chat/<chat_id>/message` (POST)
- **Lấy tin nhắn**: `/api/chat/<chat_id>/messages` (GET)
- **Xóa chat**: `/api/chat/<chat_id>` (DELETE)
- **Xóa tất cả chat**: `/api/chats/delete-all` (DELETE)
- **Đồng bộ chat từ localStorage**: `/api/chat/sync` (POST)

#### 3. Quản lý tài liệu
- **Upload file**: `/upload` (POST)
- **Xóa file**: `/remove` (POST), `/delete-file` (POST)
- **Khởi tạo dữ liệu người dùng**: `/init-user-data` (POST)

#### 4. Trả lời câu hỏi
- **Trả lời câu hỏi**: `/api/answer` (POST)
- **Body** (JSON):
```json
{
    "question": "Câu hỏi của bạn",
    "top_k": 10,
    "threshold": 5.0,
    "model": "gemini-2.0-flash",
    "chunking_method": "sentence_windows"
}
```

#### 5. Hiệu suất và cấu hình
- **Lấy hiệu suất**: `/api/performance` (GET)
- **Danh sách phương pháp chunking**: `/api/chunking_methods` (GET)
- **Lưu cài đặt**: `/settings` (POST)
- **Xem embeddings**: `/api/embeddings` (GET)
- **Đánh giá hệ thống**: `/api/evaluate` (POST)
- **Kiểm tra kết nối Supabase**: `/api/supabase-check` (GET)

## Phương pháp Chunking
Hệ thống hỗ trợ nhiều phương pháp chunking khác nhau:

1. **Sentence Windows**: Chia văn bản thành các cửa sổ câu chồng lấp
2. **Paragraph**: Chia văn bản theo đoạn văn
3. **Semantic**: Chia văn bản dựa trên ngữ nghĩa
4. **Token**: Chia văn bản dựa trên số lượng token
5. **Adaptive**: Điều chỉnh kích thước chunk dựa trên nội dung
6. **Hierarchical**: Chia văn bản theo cấu trúc phân cấp
7. **Contextual**: Chia văn bản dựa trên ngữ cảnh
8. **Multi-granularity**: Kết hợp nhiều mức độ chi tiết
9. **Hybrid**: Kết hợp nhiều phương pháp chunking (Mặc định)

## Tối ưu hóa hiệu suất
- Điều chỉnh các tham số chunking trong phần cài đặt
- Sử dụng phương pháp chunking phù hợp với loại tài liệu
- Điều chỉnh giá trị top_k và threshold khi truy vấn
- Sử dụng các phương pháp tối ưu hóa cho tiếng Việt
- Đăng nhập để lưu trữ và sử dụng dữ liệu cá nhân

## Xử lý lỗi
- Kiểm tra định dạng file upload
- Đảm bảo API keys (Gemini và Supabase) hợp lệ trong file .env
- Kiểm tra logs để phát hiện và khắc phục lỗi
- Sử dụng cơ chế chuyển đổi API key khi gặp lỗi quota
- Đảm bảo kết nối với Supabase thông qua `/api/supabase-check`

## Bảo mật
- Sử dụng HTTPS trong môi trường sản xuất
- Không bao giờ để lộ các khóa API (Gemini, Supabase)
- Thường xuyên thay đổi mật khẩu người dùng và khóa JWT
- Sao lưu dữ liệu định kỳ

## Đóng góp
Vui lòng gửi pull request hoặc mở issue để đóng góp vào dự án.
