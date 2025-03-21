# Hệ thống RAG Chatbot

## Giới thiệu
Hệ thống RAG (Retrieval-Augmented Generation) Chatbot là một ứng dụng cho phép người dùng tải lên tài liệu và đặt câu hỏi dựa trên nội dung của các tài liệu đó. Hệ thống sử dụng mô hình ngôn ngữ Gemini của Google để tạo ra câu trả lời chính xác và phù hợp với ngữ cảnh.

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
 └─────────────────┘     └─────────────────┘     └─────────────────┘
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

2. **Extract Text**:
   - Trích xuất văn bản từ file
   - Các hàm: `extract_text_pdf()`, `extract_text_docx()`, `extract_text_txt()`

3. **Chunking**:
   - Chia văn bản thành các đoạn nhỏ
   - Các phương pháp: Sentence Windows, Paragraph, Semantic, Token, Adaptive, Hierarchical, Contextual, Multi-granularity, Hybrid

4. **Embedding**:
   - Chuyển đổi các đoạn văn bản thành vector
   - Sử dụng mô hình: SentenceTransformer

5. **Vector Store**:
   - Lưu trữ và đánh chỉ mục các vector
   - Sử dụng FAISS để tìm kiếm hiệu quả

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

## Cấu trúc dự án
```
BE/
├── app.py                # File chính của ứng dụng
├── .env                  # File chứa các biến môi trường và API keys
├── README.md             # Tài liệu hướng dẫn
├── templates/            # Thư mục chứa các template HTML
└── uploads/              # Thư mục lưu trữ các file được tải lên
```

## Cài đặt

### Yêu cầu
- Python 3.8 trở lên
- Các thư viện Python được liệt kê trong `requirements.txt`

### Các bước cài đặt
1. Clone repository:
```bash
git clone <repository-url>
cd ChatBot_RAG/BE
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
```

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
2. Tải lên tài liệu (hỗ trợ định dạng .txt, .pdf, .docx)
3. Chọn phương pháp chunking phù hợp
4. Đặt câu hỏi và nhận câu trả lời

### API Endpoints

#### 1. Trả lời câu hỏi
- **URL**: `/api/answer`
- **Method**: POST
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

#### 2. Upload File
- **URL**: `/upload`
- **Method**: POST
- **Body** (form-data):
  - `file`: File cần upload (Hỗ trợ: .txt, .pdf, .docx)
  - `chunking_method`: Phương pháp chia chunk

#### 3. Xóa File
- **URL**: `/remove`
- **Method**: POST
- **Body** (form-data):
  - `filename`: Tên file cần xóa

#### 4. Lấy Thông tin Hiệu suất
- **URL**: `/api/performance`
- **Method**: GET
- **Query Parameters**:
  - `chunking_method`: (tùy chọn) Phương pháp chia chunk cần phân tích

#### 5. Đánh giá Hệ thống
- **URL**: `/api/evaluate`
- **Method**: POST
- **Body** (JSON):
```json
{
    "queries": ["câu hỏi 1", "câu hỏi 2"],
    "top_k": 5,
    "chunking_method": "sentence_windows"
}
```

#### 6. Lấy Danh sách Phương pháp Chunking
- **URL**: `/api/chunking_methods`
- **Method**: GET

#### 7. Lưu Cài đặt
- **URL**: `/settings`
- **Method**: POST
- **Body** (form-data): Các tham số cấu hình chunking

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

## Xử lý lỗi
- Kiểm tra định dạng file upload
- Đảm bảo API keys hợp lệ trong file .env
- Kiểm tra logs để phát hiện và khắc phục lỗi

## Đóng góp
Vui lòng gửi pull request hoặc mở issue để đóng góp vào dự án.
