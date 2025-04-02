# Cài đặt các gói cần thiết
# Chạy lệnh sau trong cell riêng:
#!pip install flask pyngrok google-generativeai transformers sentence-transformers PyPDF2 python-docx nltk faiss-cpu scikit-learn pandas matplotlib tqdm underthesea pyvi rank_bm25 supabase python-dotenv flask-session

# Thông báo: Dự án này đã được cải tiến với các tính năng tối ưu hóa quá trình truy xuất (retrieval) cho tiếng Việt:
# 1. Query transformation: Sử dụng các kỹ thuật biến đổi truy vấn để tăng tính liên quan giữa truy vấn và tài liệu
# 2. Reranking: Sau khi truy xuất các đoạn tài liệu ban đầu, áp dụng mô hình rerank để sắp xếp lại các kết quả theo mức độ liên quan thực sự
# Các thư viện NLP tiếng Việt (underthesea, pyvi, rank_bm25) được sử dụng để tối ưu hóa cho tiếng Việt

# Import Flask và các thư viện cần thiết
from flask import Flask, request, render_template_string, redirect, url_for, jsonify, flash, session, render_template
import google.generativeai as genai  # Sửa lại cách import
from pyngrok import ngrok
import time  # Thêm import time để xử lý delay giữa các lần thử
import re  # Thêm import re cho regular expressions
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv  # Thêm import này để sử dụng load_dotenv
from werkzeug.utils import secure_filename
from functools import wraps
# Thư viện NLP cho tiếng Việt
try:
    from underthesea import word_tokenize, text_normalize
    from pyvi import ViTokenizer
    from rank_bm25 import BM25Okapi
    VIETNAMESE_NLP_AVAILABLE = True
except ImportError:
    VIETNAMESE_NLP_AVAILABLE = False
    print("Thư viện NLP tiếng Việt không khả dụng. Một số tính năng có thể bị hạn chế.")

# Import các module Supabase
from supabase_modules.config import init_app as init_supabase, get_supabase_client
from supabase_modules.auth import register_user, login_user, logout_user, get_current_user, verify_session, require_auth, change_password, reset_password_request, setup_auth_routes
from supabase_modules.chat_history import create_chat, get_chats, get_chat, update_chat_title, delete_chat, add_message, get_messages, migrate_chat_history
from supabase_modules.file_manager import ensure_user_upload_dir, save_user_file, get_user_files, delete_user_file, migrate_files_to_user_directory, get_file_path
from supabase_modules.helpers import get_user_id_from_session, migrate_localStorage_to_supabase, initialize_user_data, format_chat_history_for_frontend, get_user_files_with_metadata
# Import module supabase_integration
import supabase_integration
from supabase_integration import setup_auth_routes, enhanced_upload_file, enhanced_remove_file, get_enhanced_index_html

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RAG_System')

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key_here")

# Tải biến môi trường từ file .env
load_dotenv()

# Đường dẫn thư mục upload
upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(upload_folder, exist_ok=True)

# Khởi tạo Supabase
init_supabase(app)

# Lấy danh sách API keys từ biến môi trường và tách thành list
API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
# Loại bỏ các key trống nếu có
API_KEYS = [key.strip() for key in API_KEYS if key.strip()]
current_key_index = 0
MAX_RETRIES = len(API_KEYS)  # Số lần thử tối đa bằng số lượng API key

def initialize_gemini():
    """
    Khởi tạo model Gemini với API key hiện tại
    
    Hàm này cấu hình API key cho Gemini và khởi tạo model.
    Nếu có lỗi, hàm sẽ ghi log và trả về None.
    
    Returns:
        GenerativeModel hoặc None: Model Gemini đã khởi tạo hoặc None nếu có lỗi
    """
    global current_key_index
    try:
        genai.configure(api_key=API_KEYS[current_key_index])
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo Gemini: {str(e)}")
        return None

def switch_api_key():
    """
    Chuyển sang API key tiếp theo trong danh sách
    
    Khi một API key gặp lỗi hoặc hết quota, hàm này sẽ chuyển sang key tiếp theo
    và khởi tạo lại model Gemini.
    
    Returns:
        GenerativeModel hoặc None: Model Gemini mới hoặc None nếu có lỗi
    """
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return initialize_gemini()

# Khởi tạo Gemini model
gemini_model = initialize_gemini()

# Cấu hình mặc định cho Gemini
gemini_config = {
    "temperature": 0.2,  # Độ sáng tạo (thấp hơn = ít sáng tạo hơn, nhất quán hơn)
    "top_p": 0.95,       # Lọc xác suất tích lũy (nucleus sampling)
    "top_k": 40,         # Số lượng token có xác suất cao nhất để xem xét
    "max_output_tokens": 4096  # Độ dài tối đa của câu trả lời
}

def generate_with_retry(prompt, config, max_retries=MAX_RETRIES):
    """
    Gọi Gemini API với cơ chế retry và chuyển đổi API key
    
    Hàm này gọi Gemini API và tự động thử lại với API key khác nếu gặp lỗi.
    Điều này giúp xử lý các trường hợp hết quota hoặc lỗi tạm thời.
    
    Args:
        prompt (str): Prompt gửi đến Gemini
        config (dict): Cấu hình cho request
        max_retries (int): Số lần thử tối đa
        
    Returns:
        str: Phản hồi từ Gemini hoặc thông báo lỗi
    """
    global gemini_model
    
    # Đảm bảo không có trường response_mime_type trong config
    if "response_mime_type" in config:
        config_copy = config.copy()
        config_copy.pop("response_mime_type", None)
    else:
        config_copy = config
    
    retries = 0
    while retries < max_retries:
        try:
            if gemini_model is None:
                gemini_model = initialize_gemini()
                if gemini_model is None:
                    raise Exception("Không thể khởi tạo model Gemini")
            
            # Gọi Gemini API
            response = gemini_model.generate_content(prompt, generation_config=config_copy)
            
            # Trả về text từ response
            return response.text
        
        except Exception as e:
            error_message = str(e)
            logger.error(f"Lỗi khi gọi Gemini API (lần thử {retries+1}/{max_retries}): {error_message}")
            
            # Nếu lỗi liên quan đến quota hoặc API key
            if "quota" in error_message.lower() or "api key" in error_message.lower() or "rate limit" in error_message.lower():
                logger.info(f"Chuyển sang API key tiếp theo...")
                gemini_model = switch_api_key()
            
            # Tăng số lần thử và đợi một chút trước khi thử lại
            retries += 1
            time.sleep(1)  # Đợi 1 giây trước khi thử lại
    
    # Nếu đã thử hết số lần mà vẫn lỗi
    return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn lúc này. Vui lòng thử lại sau."

def fix_vietnamese_spacing(text):
    """
    Sửa lỗi tách từ tiếng Việt bằng cách gọi Gemini
    
    Args:
        text (str): Văn bản cần sửa lỗi tách từ
        
    Returns:
        str: Văn bản đã được sửa lỗi tách từ
    """
    try:
        # Tạo prompt để sửa lỗi tách từ
        prompt = f"""Hãy sửa lỗi tách từ trong câu sau để đảm bảo câu giữ nguyên nội dung và ý nghĩa, nhưng các từ được tách đúng cách:

    {text}
    
    LƯU Ý QUAN TRỌNG:
    1. Không thêm bất kỳ giải thích, phân tích hay nhận xét nào
    2. Không đề cập đến việc phân tích lĩnh vực hay chuyên môn
    3. Không có lời mở đầu hay kết luận
    4. QUAN TRỌNG NHẤT: Giữ nguyên cấu trúc gốc của nó
    5. Đảm bảo các từ tiếng Việt được viết liền mạch, không bị tách ra thành các âm tiết riêng biệt với dấu cách ở giữa
    6. Giữ nguyên định dạng liệt kê như a), b), c) hoặc (a), (b), (c) trong văn bản gốc
    7. Giữ nguyên các dấu xuống dòng và khoảng cách giữa các đoạn văn"""
            
        # Sử dụng hàm generate_with_retry đã có với temperature thấp để đảm bảo độ chính xác
        fixed_text = generate_with_retry(prompt, {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            })
            
        return fixed_text
    except Exception as e:
        logger.error(f"Lỗi khi sửa lỗi tách từ tiếng Việt: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return text

# HTML template cho trang web
index_html = open('templates/index.html', 'r', encoding='utf-8').read()

# Tải dữ liệu cho NLTK
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

# Thiết lập thư mục lưu trữ file upload
import os
import sys

# Kiểm tra môi trường
is_colab = 'google.colab' in sys.modules

if is_colab:
    # Nếu đang chạy trong Colab, mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    upload_folder = '/content/drive/MyDrive/notebooklm_uploads'
else:
    # Nếu đang chạy trong môi trường local
    upload_folder = './uploads'

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# --- Các hàm trích xuất văn bản từ file ---
import PyPDF2
import docx
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_System")

def extract_text_pdf(file_path):
    """
    Trích xuất văn bản từ file PDF
    
    Hàm này đọc nội dung từ file PDF và chuyển đổi thành văn bản thuần túy.
    
    Args:
        file_path (str): Đường dẫn đến file PDF
        
    Returns:
        str: Nội dung văn bản đã trích xuất
    """
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            
            # Lưu trữ nội dung của tất cả các trang
            all_pages_text = []
            
            # Trích xuất văn bản từ từng trang
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Thêm metadata về số trang nhưng giữ nguyên định dạng gốc
                    all_pages_text.append(f"[Trang {page_num + 1}] {page_text}")
            
            # Kết hợp nội dung của tất cả các trang
            text = "\n".join(all_pages_text)
            
            # Xử lý các trường hợp nội dung trải dài qua nhiều trang
            # Tìm các dòng bị cắt giữa chừng ở cuối trang
            text = re.sub(r'(\w+)-\s*\[Trang (\d+)\]', r'\1[Trang \2]', text)
            
            # Đảm bảo giữ nguyên định dạng liệt kê
            # Bảo vệ các định dạng liệt kê như a), b), c), (a), (b), (c), v.v.
            text = re.sub(r'(\s[a-z]\)|\s\([a-z]\))', lambda m: m.group(0), text)
        
        # Đảm bảo trả về một chuỗi văn bản
        if isinstance(text, list):
            text = "\n".join(text)
        return text
    except Exception as e:
        logger.error(f"Lỗi khi đọc file PDF {file_path}: {str(e)}")
        return ""

def extract_text_docx(file_path):
    """
    Trích xuất văn bản từ file DOCX (Microsoft Word)
    
    Hàm này đọc nội dung từ file DOCX và chuyển đổi thành văn bản thuần túy.
    Đã cải tiến để đánh dấu thông tin về trang chính xác hơn.
    
    Args:
        file_path (str): Đường dẫn đến file DOCX
        
    Returns:
        str: Văn bản trích xuất từ file DOCX
    """
    try:
        import docx
        doc = docx.Document(file_path)
        
        full_text = []
        page_count = 1
        paragraph_count = 0
        char_count = 0
        
        # Phân tích số lượng ký tự trung bình mỗi trang cho DOCX
        # Thông thường một trang A4 với font 12pt có khoảng 2000-3000 ký tự
        estimated_page_size = 2500  # Số ký tự trung bình trên một trang DOCX chuẩn
        
        # Thêm đánh dấu trang đầu tiên
        full_text.append(f"[Trang {page_count}]")
        
        # Phân tích tất cả các đoạn văn
        paragraphs_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs_text.append(text)
                char_count += len(text)
        
        # Ước tính tổng số trang dựa trên tổng số ký tự
        total_pages = max(1, int(char_count / estimated_page_size) + 1)
        
        # Phân bổ đánh dấu trang dựa trên số lượng đoạn và tổng số trang
        if len(paragraphs_text) > 0:
            # Nếu có ít nhất 1 đoạn văn bản
            paragraphs_per_page = max(1, len(paragraphs_text) // total_pages)
            
            # Thêm các đoạn văn vào văn bản với đánh dấu trang
            current_page = 1
            for i, para_text in enumerate(paragraphs_text):
                # Thêm dấu trang mới nếu cần
                if i > 0 and i % paragraphs_per_page == 0 and current_page < total_pages:
                    current_page += 1
                    full_text.append(f"[Trang {current_page}]")
                
                # Thêm đoạn văn
                full_text.append(para_text)
        
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file DOCX {file_path}: {str(e)}")
        return ""

def extract_text_txt(file_path):
    """
    Đọc nội dung từ file văn bản thuần túy (TXT)
    
    Args:
        file_path (str): Đường dẫn đến file TXT
        
    Returns:
        str: Nội dung văn bản đã đọc với thông tin trang được thêm vào
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # Thêm thông tin trang cho file TXT
        lines = raw_text.split('\n')
        
        # Lọc bỏ dòng trống
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Phân tích số lượng ký tự
        char_count = len(raw_text)
        
        # Ước tính số trang dựa trên số ký tự
        # Thông thường một trang A4 với font 12pt có khoảng 2000-3000 ký tự
        estimated_page_size = 2500
        total_pages = max(1, int(char_count / estimated_page_size) + 1)
        
        # Nếu ít hơn 1 trang, chỉ trả về văn bản ban đầu
        if total_pages <= 1:
            return f"[Trang 1]\n{raw_text}"
        
        # Khởi tạo kết quả
        processed_text = []
        
        # Phân bổ đánh dấu trang dựa trên số lượng dòng và tổng số trang
        if len(non_empty_lines) > 0:
            # Số dòng trên mỗi trang
            lines_per_page = max(1, len(non_empty_lines) // total_pages)
            
            # Thêm các dòng vào văn bản với đánh dấu trang
            current_page = 1
            processed_text.append(f"[Trang {current_page}]")
            
            line_count = 0
            for line in lines:
                processed_text.append(line)
                if line.strip():  # Chỉ đếm các dòng không trống
                    line_count += 1
                    
                    # Thêm đánh dấu trang mới nếu cần
                    if line_count % lines_per_page == 0 and current_page < total_pages:
                        current_page += 1
                        processed_text.append(f"[Trang {current_page}]")
        
        return "\n".join(processed_text)
    except UnicodeDecodeError:
        try:
            # Thử lại với encoding khác
            with open(file_path, "r", encoding="latin-1") as f:
                raw_text = f.read()
            
            # Xử lý tương tự như trên
            lines = raw_text.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            char_count = len(raw_text)
            estimated_page_size = 2500
            total_pages = max(1, int(char_count / estimated_page_size) + 1)
            
            if total_pages <= 1:
                return f"[Trang 1]\n{raw_text}"
            
            processed_text = []
            if len(non_empty_lines) > 0:
                lines_per_page = max(1, len(non_empty_lines) // total_pages)
                current_page = 1
                processed_text.append(f"[Trang {current_page}]")
                
                line_count = 0
                for line in lines:
                    processed_text.append(line)
                    if line.strip():
                        line_count += 1
                        if line_count % lines_per_page == 0 and current_page < total_pages:
                            current_page += 1
                            processed_text.append(f"[Trang {current_page}]")
            
            return "\n".join(processed_text)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file TXT {file_path} với encoding latin-1: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Lỗi khi đọc file TXT {file_path}: {str(e)}")
        return ""

# --- Các phương pháp chunking khác nhau ---
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import re

def create_sentence_windows(text, window_size=3, step=1):
    """
    Chia văn bản thành các cửa sổ câu chồng lấp
    
    Phương pháp này chia văn bản thành các đoạn nhỏ, mỗi đoạn chứa một số câu
    liên tiếp. Các cửa sổ có thể chồng lấp lên nhau để đảm bảo ngữ cảnh được
    giữ nguyên.
    
    Args:
        text (str): Văn bản cần chia
        window_size (int): Số câu trong mỗi cửa sổ
        step (int): Số câu di chuyển mỗi lần tạo cửa sổ mới
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        sentences = sent_tokenize(text)
        windows = []
        
        for i in range(0, len(sentences) - window_size + 1, step):
            window_text = " ".join(sentences[i:i+window_size])
            windows.append(window_text)
        
        # Nếu không có windows nào được tạo (ví dụ: văn bản quá ngắn)
        if not windows and sentences:
            windows = [" ".join(sentences)]
            
        return windows
    except Exception as e:
        logger.error(f"Lỗi khi tạo sentence windows: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_paragraph_chunks(text, max_chars=1000, overlap=200):
    """
    Chia văn bản theo đoạn văn
    
    Phương pháp này chia văn bản thành các đoạn dựa trên dấu xuống dòng,
    đảm bảo mỗi đoạn không vượt quá số ký tự tối đa và có phần chồng lấp
    để giữ ngữ cảnh.
    
    Args:
        text (str): Văn bản cần chia
        max_chars (int): Số ký tự tối đa trong mỗi đoạn
        overlap (int): Số ký tự chồng lấp giữa các đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Nếu đoạn văn hiện tại + đoạn mới vượt quá giới hạn
            if len(current_chunk) + len(para) > max_chars and current_chunk:
                chunks.append(current_chunk)
                # Giữ lại phần overlap từ chunk trước
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Thêm chunk cuối cùng nếu còn
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo paragraph chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_semantic_chunks(text, min_words=50, max_words=200):
    """
    Chia văn bản dựa trên ngữ nghĩa
    
    Phương pháp này cố gắng chia văn bản thành các đoạn có ý nghĩa ngữ nghĩa
    hoàn chỉnh, dựa trên cấu trúc câu và đoạn văn.
    
    Args:
        text (str): Văn bản cần chia
        min_words (int): Số từ tối thiểu trong mỗi đoạn
        max_words (int): Số từ tối đa trong mỗi đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Phát hiện các mẫu điều khoản luật
        law_article_pattern = re.compile(r'(Điều \d+\..*?)(?=Điều \d+\.|$)', re.DOTALL)
        law_articles = law_article_pattern.findall(text)
        
        # Nếu phát hiện các điều khoản luật, ưu tiên chunking theo điều khoản
        if law_articles:
            chunks = []
            
            for article in law_articles:
                # Nếu điều khoản quá dài, chia nhỏ
                if len(article.split()) > max_words * 2:
                    sentences = sent_tokenize(article)
                    current_chunk = []
                    current_word_count = 0
                    
                    for sentence in sentences:
                        sentence_word_count = len(sentence.split())
                        
                        if current_word_count + sentence_word_count > max_words and current_word_count >= min_words:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                            current_word_count = sentence_word_count
                        else:
                            current_chunk.append(sentence)
                            current_word_count += sentence_word_count
                    
                    # Thêm chunk cuối cùng nếu còn
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                else:
                    chunks.append(article)
            
            # Xử lý phần còn lại của văn bản
            remaining_text = text
            for article in law_articles:
                remaining_text = remaining_text.replace(article, "")
            
            if remaining_text.strip():
                # Chia phần còn lại thành các đoạn văn
                paragraphs = remaining_text.split('\n\n')
                current_chunk = []
                current_word_count = 0
                
                for para in paragraphs:
                    if not para.strip():
                        continue
                        
                    para_word_count = len(para.split())
                    
                    if current_word_count + para_word_count > max_words and current_word_count >= min_words:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [para]
                        current_word_count = para_word_count
                    else:
                        current_chunk.append(para)
                        current_word_count += para_word_count
                
                # Thêm chunk cuối cùng nếu còn
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
            
            return chunks
        
        # Nếu không phát hiện điều khoản luật, chia theo đoạn văn và ngữ nghĩa
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            if current_word_count + sentence_word_count > max_words and current_word_count >= min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Thêm chunk cuối cùng nếu còn
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo semantic chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_token_chunks(text, max_tokens=1000, overlap_tokens=100):
    """
    Chia văn bản dựa trên số lượng token
    
    Phương pháp này chia văn bản thành các đoạn với số lượng token cố định,
    phù hợp với giới hạn đầu vào của các mô hình ngôn ngữ.
    
    Args:
        text (str): Văn bản cần chia
        max_tokens (int): Số token tối đa trong mỗi đoạn
        overlap_tokens (int): Số token chồng lấp giữa các đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        from transformers import AutoTokenizer
        
        # Sử dụng tokenizer phù hợp với mô hình LLM
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        
        # Tokenize toàn bộ văn bản
        tokens = tokenizer.encode(text)
        
        # Tạo chunks dựa trên số lượng token
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap_tokens):
            # Lấy phần tokens cho chunk hiện tại
            chunk_tokens = tokens[i:i + max_tokens]
            
            # Chuyển tokens thành văn bản
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Thêm vào danh sách chunks
            chunks.append(chunk_text)
            
            # Nếu đã xử lý hết tokens, dừng vòng lặp
            if i + max_tokens >= len(tokens):
                break
        
        return chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo token chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_adaptive_chunks(text, min_words=50, max_words=300, overlap_ratio=0.2):
    """
    Điều chỉnh kích thước chunk dựa trên nội dung
    
    Phương pháp này tự động điều chỉnh kích thước của các đoạn dựa trên
    nội dung và cấu trúc của văn bản, giúp tạo ra các đoạn có ý nghĩa hơn.
    
    Args:
        text (str): Văn bản cần chia
        min_words (int): Số từ tối thiểu trong mỗi đoạn
        max_words (int): Số từ tối đa trong mỗi đoạn
        overlap_ratio (float): Tỷ lệ chồng lấp giữa các đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Chuẩn hóa văn bản
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Phát hiện các ranh giới tự nhiên
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Phát hiện các mục và tiêu đề
        section_patterns = [
            r'^(Chương|CHƯƠNG)\s+\d+.*$',
            r'^(Điều|ĐIỀU)\s+\d+.*$',
            r'^(Mục|MỤC)\s+\d+.*$',
            r'^(Phần|PHẦN)\s+\d+.*$',
            r'^([A-Z0-9]\.|\d+\.|[IVX]+\.).*$',  # Các đánh số như A., 1., I., v.v.
            r'^([a-z]\.|\d+\.).*$',  # Các đánh số như a., 1., v.v.
        ]
        
        # Xác định các ranh giới quan trọng
        important_boundaries = []
        
        for i, para in enumerate(paragraphs):
            # Kiểm tra xem đoạn văn có phải là tiêu đề hoặc mục không
            is_section_header = any(re.match(pattern, para.strip(), re.MULTILINE) for pattern in section_patterns)
            
            if is_section_header:
                important_boundaries.append(i)
        
        # Tạo chunks dựa trên ranh giới quan trọng và kích thước
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
                
            para_word_count = len(para.split())
            
            # Nếu đoạn văn là ranh giới quan trọng và chunk hiện tại đã đủ lớn
            if i in important_boundaries and current_word_count >= min_words:
                # Kết thúc chunk hiện tại
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Bắt đầu chunk mới với đoạn văn hiện tại
                current_chunk = [para]
                current_word_count = para_word_count
            
            # Nếu thêm đoạn văn này sẽ vượt quá kích thước tối đa
            elif current_word_count + para_word_count > max_words and current_word_count >= min_words:
                # Kết thúc chunk hiện tại
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Tính toán số từ chồng lấp
                overlap_words = int(current_word_count * overlap_ratio)
                
                # Lấy phần chồng lấp từ chunk trước
                overlap_text = ' '.join(current_chunk[-overlap_words:]) if overlap_words > 0 and len(current_chunk) > overlap_words else ''
                
                # Bắt đầu chunk mới với phần chồng lấp và đoạn văn hiện tại
                current_chunk = []
                if overlap_text:
                    current_chunk.append(overlap_text)
                current_chunk.append(para)
                current_word_count = len(' '.join(current_chunk).split())
            
            # Nếu không, thêm đoạn văn vào chunk hiện tại
            else:
                current_chunk.append(para)
                current_word_count += para_word_count
        
        # Thêm chunk cuối cùng nếu còn
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Xử lý các chunk quá nhỏ
        final_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_word_count = len(chunk.split())
            
            # Nếu chunk quá nhỏ và không phải chunk cuối cùng
            if chunk_word_count < min_words and i < len(chunks) - 1:
                # Gộp với chunk tiếp theo
                chunks[i+1] = chunk + ' ' + chunks[i+1]
            else:
                final_chunks.append(chunk)
        
        # Đảm bảo không có chunk nào quá lớn
        result_chunks = []
        for chunk in final_chunks:
            chunk_word_count = len(chunk.split())
            
            if chunk_word_count > max_words * 1.5:  # Nếu chunk lớn hơn 150% kích thước tối đa
                # Chia nhỏ chunk này thành các phần nhỏ hơn
                words = chunk.split()
                for i in range(0, len(words), max_words):
                    sub_chunk = ' '.join(words[i:i+max_words])
                    if len(sub_chunk.split()) >= min_words:
                        result_chunks.append(sub_chunk)
            else:
                result_chunks.append(chunk)
        
        return result_chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo adaptive chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_hierarchical_chunks(text, max_chunk_size=1000, min_chunk_size=100):
    """
    Chia văn bản theo cấu trúc phân cấp
    
    Phương pháp này phân tích cấu trúc phân cấp của văn bản (tiêu đề, mục, đoạn)
    và chia thành các đoạn dựa trên cấu trúc đó, giúp giữ nguyên ngữ cảnh và
    mối quan hệ giữa các phần.
    
    Args:
        text (str): Văn bản cần chia
        max_chunk_size (int): Kích thước tối đa của mỗi đoạn
        min_chunk_size (int): Kích thước tối thiểu của mỗi đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Chuẩn hóa văn bản
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Các mẫu regex để phát hiện cấu trúc phân cấp
        patterns = {
            'chapter': r'(Chương|CHƯƠNG)\s+(\d+|[IVX]+)[:\.\s]*(.*?)(?=(?:Chương|CHƯƠNG)\s+\d+|$)',
            'section': r'(Mục|MỤC)\s+(\d+|[IVX]+)[:\.\s]*(.*?)(?=(?:Mục|MỤC)\s+\d+|Chương|CHƯƠNG|$)',
            'article': r'(Điều|ĐIỀU)\s+(\d+)[:\.\s]*(.*?)(?=(?:Điều|ĐIỀU)\s+\d+|Mục|MỤC|Chương|CHƯƠNG|$)',
            'paragraph': r'(\d+)[\.:\s]+(.*?)(?=\d+[\.:\s]+|Điều|ĐIỀU|Mục|MỤC|Chương|CHƯƠNG|$)',
            'point': r'([a-z])[\.:\s]+(.*?)(?=[a-z][\.:\s]+|\d+[\.:\s]+|Điều|ĐIỀU|Mục|MỤC|Chương|CHƯƠNG|$)'
        }
        
        # Phân tích cấu trúc phân cấp
        def parse_hierarchical_structure(text, level=0, path=None):
            if path is None:
                path = []
            
            if level >= len(patterns.keys()):
                return [{'text': text, 'path': path.copy()}]
            
            level_name = list(patterns.keys())[level]
            pattern = patterns[level_name]
            
            # Tìm các phần ở cấp độ hiện tại
            matches = list(re.finditer(pattern, text, re.DOTALL))
            
            if not matches:
                # Nếu không tìm thấy cấu trúc ở cấp độ này, chuyển sang cấp độ tiếp theo
                return parse_hierarchical_structure(text, level + 1, path)
            
            result = []
            
            for i, match in enumerate(matches):
                # Lấy tiêu đề và nội dung
                prefix = match.group(1)
                number = match.group(2)
                content = match.group(3) if len(match.groups()) > 2 else ""
                
                # Cập nhật đường dẫn
                current_path = path.copy()
                current_path.append(f"{prefix} {number}")
                
                # Phân tích cấp độ tiếp theo
                sub_chunks = parse_hierarchical_structure(content, level + 1, current_path)
                result.extend(sub_chunks)
            
            return result
        
        # Phân tích cấu trúc phân cấp của văn bản
        try:
            hierarchical_chunks = parse_hierarchical_structure(text)
        except Exception as e:
            logger.error(f"Lỗi khi phân tích cấu trúc phân cấp: {str(e)}")
            # Fallback: chia văn bản thành các đoạn
            paragraphs = re.split(r'\n\s*\n', text)
            hierarchical_chunks = [{'text': para, 'path': []} for para in paragraphs if para.strip()]
        
        # Tạo chunks với thông tin ngữ cảnh
        final_chunks = []
        
        for item in hierarchical_chunks:
            chunk_text = item['text']
            path = item['path']
            
            # Tạo breadcrumbs
            breadcrumbs = " > ".join(path) if path else "Văn bản"
            
            # Nếu chunk quá lớn, chia nhỏ hơn
            if len(chunk_text) > max_chunk_size:
                # Chia thành các đoạn văn
                paragraphs = re.split(r'\n\s*\n', chunk_text)
                
                current_chunk = []
                current_size = 0
                
                for para in paragraphs:
                    if not para.strip():
                        continue
                        
                    para_size = len(para)
                    
                    if current_size + para_size > max_chunk_size and current_size >= min_chunk_size:
                        # Tạo chunk với breadcrumbs
                        chunk_with_context = f"[Ngữ cảnh: {breadcrumbs}]\n\n" + "\n\n".join(current_chunk)
                        final_chunks.append(chunk_with_context)
                        
                        # Bắt đầu chunk mới
                        current_chunk = [para]
                        current_size = para_size
                    else:
                        current_chunk.append(para)
                        current_size += para_size
                    
                    # Thêm chunk cuối cùng nếu còn
                    if current_chunk:
                        chunk_with_context = f"[Ngữ cảnh: {breadcrumbs}]\n\n" + "\n\n".join(current_chunk)
                        final_chunks.append(chunk_with_context)
                else:
                    # Nếu chunk đủ nhỏ, thêm trực tiếp
                    chunk_with_context = f"[Ngữ cảnh: {breadcrumbs}]\n\n{chunk_text}"
                    final_chunks.append(chunk_with_context)
            
            # Xử lý các chunk quá nhỏ
            result_chunks = []
            i = 0
            
            while i < len(final_chunks):
                current_chunk = final_chunks[i]
                
                # Nếu chunk quá nhỏ và không phải chunk cuối cùng
                if len(current_chunk) < min_chunk_size and i < len(final_chunks) - 1:
                    # Gộp với chunk tiếp theo
                    next_chunk = final_chunks[i + 1]
                    merged_chunk = current_chunk + "\n\n" + next_chunk
                    result_chunks.append(merged_chunk)
                    i += 2
                else:
                    result_chunks.append(current_chunk)
                    i += 1
            
            return result_chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo hierarchical chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_contextual_chunks(text, max_chunk_size=1000, min_chunk_size=100, overlap_ratio=0.2):
    """
    Chia văn bản dựa trên ngữ cảnh
    
    Phương pháp này cố gắng giữ nguyên ngữ cảnh khi chia văn bản, đảm bảo
    các thông tin liên quan đến nhau nằm trong cùng một đoạn.
    
    Args:
        text (str): Văn bản cần chia
        max_chunk_size (int): Kích thước tối đa của mỗi đoạn
        min_chunk_size (int): Kích thước tối thiểu của mỗi đoạn
        overlap_ratio (float): Tỷ lệ chồng lấp giữa các đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Phiên bản đơn giản hóa: chia theo đoạn văn với thông tin ngữ cảnh
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            para_size = len(para)
            
            if current_size + para_size > max_chunk_size and current_size >= min_chunk_size:
                # Kết thúc chunk hiện tại
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    context_info = f"[Ngữ cảnh: Phần văn bản]"
                    chunks.append(f"{context_info}\n\n{chunk_text}")
                
                # Bắt đầu chunk mới
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Thêm chunk cuối cùng nếu còn
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            context_info = f"[Ngữ cảnh: Phần văn bản]"
            chunks.append(f"{context_info}\n\n{chunk_text}")
        
        return chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo contextual chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_multi_granularity_chunks(text, config=None):
    """
    Kết hợp nhiều mức độ chi tiết
    
    Phương pháp này tạo ra các đoạn ở nhiều mức độ chi tiết khác nhau
    (câu, đoạn văn, mục), giúp hệ thống có thể truy xuất thông tin ở
    nhiều cấp độ khác nhau.
    
    Args:
        text (str): Văn bản cần chia
        config (dict): Cấu hình cho việc chia đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Cấu hình mặc định nếu không được cung cấp
        if not config:
            config = {
                "max_section_length": 3000,
                "max_paragraph_length": 1000,
                "max_sentence_window": 3
            }
        
        all_chunks = []
        
        # 1. Tạo chunk cấp độ tài liệu (nếu tài liệu không quá dài)
        if len(text) <= 10000:  # Giới hạn 10k ký tự cho toàn bộ tài liệu
            all_chunks.append({
                "text": text,
                "level": "document",
                "metadata": {"granularity": "document"}
            })
        
        # 2. Tạo chunks cấp độ phần (sections)
        # Phát hiện các phần dựa trên tiêu đề
        section_patterns = [
            r'(^|\n)#+\s+(.+?)(?=\n#+\s+|\Z)',  # Markdown headings
            r'(^|\n)([A-Z][A-Z\s]+)(?=\n)',     # ALL CAPS HEADINGS
            r'(^|\n)([IVXLCDM]+\.\s+.+?)(?=\n[IVXLCDM]+\.\s+|\Z)',  # Roman numeral sections
            r'(^|\n)(\d+\.\s+[A-Z].+?)(?=\n\d+\.\s+[A-Z]|\Z)',      # Numbered sections
            r'(^|\n)(Chương \d+.+?)(?=\nChương \d+|\Z)',           # Chương sections
            r'(^|\n)(Điều \d+.+?)(?=\nĐiều \d+|\Z)',               # Điều sections
            r'(^|\n)(Mục \d+.+?)(?=\nMục \d+|\Z)',                 # Mục sections
            r'(^|\n)(Phần \d+.+?)(?=\nPhần \d+|\Z)',               # Phần sections
        ]
        
        sections = []
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        section_text = match[1]
                    else:
                        section_text = match
                    
                    if len(section_text) <= config["max_section_length"]:
                        sections.append(section_text)
        
        # Nếu không tìm thấy sections, chia tài liệu thành các phần có kích thước bằng nhau
        if not sections and len(text) > config["max_section_length"]:
            words = text.split()
            section_size = min(len(words), config["max_section_length"] // 5)  # Ước tính 5 ký tự/từ
            
            for i in range(0, len(words), section_size):
                section_text = " ".join(words[i:i+section_size])
                sections.append(section_text)
        
        # Thêm các sections vào danh sách chunks
        for i, section in enumerate(sections):
            all_chunks.append({
                "text": section,
                "level": "section",
                "metadata": {
                    "granularity": "section",
                    "section_index": i,
                    "section_count": len(sections)
                }
            })
        
        # 3. Tạo chunks cấp độ đoạn văn (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        for i, para in enumerate(paragraphs):
            if para.strip() and len(para) <= config["max_paragraph_length"]:
                all_chunks.append({
                    "text": para,
                    "level": "paragraph",
                    "metadata": {
                        "granularity": "paragraph",
                        "paragraph_index": i,
                        "paragraph_count": len(paragraphs)
                    }
                })
        
        # 4. Tạo chunks cấp độ câu (sentence windows)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        window_size = config["max_sentence_window"]
        
        for i in range(len(sentences) - window_size + 1):
            window = " ".join(sentences[i:i+window_size])
            all_chunks.append({
                "text": window,
                "level": "sentence",
                "metadata": {
                    "granularity": "sentence",
                    "window_index": i,
                    "window_size": window_size,
                    "sentence_count": len(sentences)
                }
            })
        
        # Chuyển đổi định dạng để tương thích với hệ thống hiện tại
        final_chunks = []
        for chunk_obj in all_chunks:
            chunk_text = chunk_obj["text"]
            # Thêm metadata vào text để có thể truy xuất sau này
            chunk_text = f"[Granularity: {chunk_obj['level']}] {chunk_text}"
            final_chunks.append(chunk_text)
        
        return final_chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo multi-granularity chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

def create_hybrid_chunks(text, config=None):
    """
    Chunking kết hợp nhiều phương pháp để tối ưu kết quả
    
    Đây là phương pháp tiên tiến nhất, kết hợp nhiều kỹ thuật chunking khác nhau
    để tạo ra bộ chunks tối ưu. Phương pháp này phân tích cấu trúc văn bản,
    xử lý riêng các phần đặc biệt (như điều khoản luật), và áp dụng các
    phương pháp chunking khác nhau cho từng loại nội dung.
    
    Args:
        text (str): Văn bản cần chia
        config (dict): Cấu hình cho việc chia đoạn
        
    Returns:
        list: Danh sách các đoạn văn bản đã chia
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Cấu hình mặc định nếu không được cung cấp
        if not config:
            config = {
                "sentence_windows": {"window_size": 3, "step": 1},
                "paragraph": {"max_chars": 1500, "overlap": 200},
                "semantic": {"min_words": 50, "max_words": 200},
                "token": {"max_tokens": 1500, "overlap_tokens": 100},
                "contextual": {"max_chunk_size": 1500, "min_chunk_size": 100, "overlap_ratio": 0.2},
                "multi_granularity": {
                    "max_section_length": 3000,
                    "max_paragraph_length": 1000,
                    "max_sentence_window": 3
                }
            }
        
        all_chunks = []
        
        # 1. Phát hiện và xử lý các điều khoản luật trước
        law_article_pattern = re.compile(r'(Điều \d+\..*?)(?=Điều \d+\.|$)', re.DOTALL)
        law_articles = law_article_pattern.findall(text)
        
        # Xử lý các điều khoản luật
        if law_articles:
            for article in law_articles:
                # Sử dụng chunking theo đoạn văn cho điều khoản luật
                article_chunks = create_paragraph_chunks(article)
                all_chunks.extend(article_chunks)
                
                # Loại bỏ điều khoản đã xử lý khỏi văn bản gốc
                text = text.replace(article, "")
        
        # 2. Tạo chunks theo nhiều cấp độ chi tiết
        multi_granularity_chunks = create_multi_granularity_chunks(text, config.get("multi_granularity"))
        all_chunks.extend(multi_granularity_chunks)
        
        # 3. Tạo chunks dựa trên ngữ cảnh
        contextual_chunks = create_contextual_chunks(
            text,
            max_chunk_size=config["contextual"]["max_chunk_size"],
            min_chunk_size=config["contextual"]["min_chunk_size"],
            overlap_ratio=config["contextual"]["overlap_ratio"]
        )
        all_chunks.extend(contextual_chunks)
        
        # 4. Xử lý phần còn lại của văn bản bằng chunking theo đoạn văn
        if text.strip():
            remaining_chunks = create_paragraph_chunks(text)
            all_chunks.extend(remaining_chunks)
        
        # 5. Nếu không có chunks nào được tạo, thử phương pháp khác
        if not all_chunks:
            all_chunks = create_sentence_windows(text)
        
        # 6. Loại bỏ các chunks trùng lặp
        unique_chunks = []
        seen_chunks = set()
        
        for chunk in all_chunks:
            # Chuẩn hóa chunk để so sánh
            normalized_chunk = re.sub(r'\s+', ' ', chunk).strip()
            if normalized_chunk not in seen_chunks:
                seen_chunks.add(normalized_chunk)
                unique_chunks.append(chunk)
        
        # 7. Sắp xếp chunks theo độ dài để ưu tiên các chunks có kích thước vừa phải
        sorted_chunks = sorted(unique_chunks, key=lambda x: abs(len(x) - 1000))  # 1000 là độ dài lý tưởng
        
        return sorted_chunks
    except Exception as e:
        logger.error(f"Lỗi khi tạo hybrid chunks: {str(e)}")
        # Trả về văn bản gốc nếu có lỗi
        return [text] if text else []

# --- Cấu hình và mô hình ---
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai  # Sửa lại cách import
import json
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
from collections import Counter

# Cấu hình embedding model - nâng cấp lên mô hình mạnh hơn
embedder = SentenceTransformer('all-mpnet-base-v2') 
d = embedder.get_sentence_embedding_dimension()

# Khởi tạo biến toàn cục
global_vector_list = []   # Lưu các vector embeddings
global_metadata = []      # Lưu metadata của chunks
global_all_files = {}     # Theo dõi các tài liệu đã thêm vào
faiss_index = None        # FAISS index
tfidf_vectorizer = None   # Vectorizer cho tìm kiếm từ khóa
tfidf_matrix = None       # Ma trận TF-IDF
global_performance = []   # Lưu thông tin hiệu suất
performance_metrics = {
    "queries": [],
    "retrieval_times": [],
    "answer_times": [],
    "chunk_counts": [],
    "chunking_methods": []
}

# Giá trị mặc định cho các phương pháp chunking
default_configs = {
    "sentence_windows": {"window_size": 3, "step": 1},
    "paragraph": {"max_chars": 1000, "overlap": 200},
    "semantic": {"min_words": 100, "max_words": 400},
    "token": {"max_tokens": 1000, "overlap_tokens": 100}
}

# Lưu và nạp trạng thái
def save_state():
    """Lưu trạng thái hiện tại của hệ thống RAG"""
    # Xác định thư mục lưu trữ dựa trên người dùng đăng nhập
    from supabase_modules.helpers import get_user_id_from_session
    from supabase_modules.file_manager import ensure_user_upload_dir
    from flask import has_request_context
    
    # Kiểm tra xem có đang trong ngữ cảnh request không
    if has_request_context():
        user_id = get_user_id_from_session()
        if user_id:
            # Nếu có người dùng đăng nhập, lưu vào thư mục của họ
            save_dir = ensure_user_upload_dir(user_id)
        else:
            # Nếu không có người dùng đăng nhập, lưu vào thư mục uploads chung
            save_dir = upload_folder
    else:
        # Nếu không trong ngữ cảnh request (ví dụ: khi khởi động), sử dụng thư mục mặc định
        save_dir = upload_folder
        logger.info("Khởi động ứng dụng: Lưu dữ liệu vào thư mục mặc định")
    
    state = {
        'metadata': global_metadata,
        'all_files': global_all_files,
        'date': datetime.now().isoformat()
    }
    # Lưu metadata và danh sách file
    with open(os.path.join(save_dir, 'rag_state.json'), 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    
    # Lưu vectors
    if global_vector_list:
        vectors = np.vstack(global_vector_list).astype('float32')
        with open(os.path.join(save_dir, 'vectors.pkl'), 'wb') as f:
            pickle.dump(vectors, f)
    
    # Lưu index FAISS nếu đã có
    if faiss_index is not None:
        faiss.write_index(faiss_index, os.path.join(save_dir, 'faiss_index.bin'))
    
    # Lưu TF-IDF vectorizer và matrix
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(os.path.join(save_dir, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(tfidf_matrix, f)

def load_state():
    """Nạp trạng thái hệ thống RAG từ file"""
    global global_metadata, global_all_files, global_vector_list, global_content_dict, global_reranker
    global faiss_index, tfidf_vectorizer, tfidf_matrix
    
    # Reset lại tất cả các biến global trước khi nạp dữ liệu mới
    global_metadata = []
    global_all_files = {}
    global_vector_list = []
    faiss_index = None
    tfidf_vectorizer = None
    tfidf_matrix = None
    
    # Xác định thư mục nạp dựa trên người dùng đăng nhập
    from supabase_modules.helpers import get_user_id_from_session
    from supabase_modules.file_manager import ensure_user_upload_dir, get_user_upload_dir
    from flask import has_request_context
    
    # Kiểm tra xem có đang trong ngữ cảnh request không
    if has_request_context():
        user_id = get_user_id_from_session()
        if user_id:
            # Nếu có người dùng đăng nhập, nạp từ thư mục của họ
            load_dir = get_user_upload_dir(user_id)
            if not os.path.exists(load_dir):
                load_dir = ensure_user_upload_dir(user_id)
                logger.info(f"Tạo thư mục người dùng mới: {load_dir}")
                return False
        else:
            # Nếu không có người dùng đăng nhập, nạp từ thư mục uploads chung
            load_dir = upload_folder
    else:
        # Nếu không trong ngữ cảnh request (ví dụ: khi khởi động), sử dụng thư mục mặc định
        load_dir = upload_folder
        logger.info("Khởi động ứng dụng: Nạp dữ liệu từ thư mục mặc định")
    
    # Nạp metadata và danh sách file
    state_file = os.path.join(load_dir, 'rag_state.json')
    if not os.path.exists(state_file):
        logger.info(f"Không tìm thấy file trạng thái tại {state_file}")
        return False
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Đảm bảo kiểu dữ liệu tương thích
        global_metadata = state.get('metadata', [])
        if not isinstance(global_metadata, list):
            global_metadata = []
            
        global_all_files = state.get('all_files', {})
        if not isinstance(global_all_files, dict):
            global_all_files = {}
        
        # Nạp vectors nếu có
        vectors_file = os.path.join(load_dir, 'vectors.pkl')
        if os.path.exists(vectors_file):
            with open(vectors_file, 'rb') as f:
                vectors = pickle.load(f)
            global_vector_list = [vectors]
            
            # Nạp index FAISS nếu có
            faiss_index_file = os.path.join(load_dir, 'faiss_index.bin')
            if os.path.exists(faiss_index_file):
                faiss_index = faiss.read_index(faiss_index_file)
            else:
                # Tạo index mới nếu không tìm thấy file
                initialize_faiss_index()
                
            # Nạp TF-IDF vectorizer và matrix nếu có
            tfidf_vectorizer_file = os.path.join(load_dir, 'tfidf_vectorizer.pkl')
            tfidf_matrix_file = os.path.join(load_dir, 'tfidf_matrix.pkl')
            
            if os.path.exists(tfidf_vectorizer_file) and os.path.exists(tfidf_matrix_file):
                with open(tfidf_vectorizer_file, 'rb') as f:
                    tfidf_vectorizer = pickle.load(f)
                with open(tfidf_matrix_file, 'rb') as f:
                    tfidf_matrix = pickle.load(f)
            else:
                # Tạo TF-IDF mới nếu không tìm thấy file
                initialize_tfidf()
        
        logger.info(f"Đã nạp dữ liệu từ {load_dir}: {len(global_metadata)} chunks, {len(global_all_files)} file")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi nạp dữ liệu từ {load_dir}: {str(e)}")
        # Reset lại tất cả các biến global nếu có lỗi
        global_metadata = []
        global_all_files = {}
        global_vector_list = []
        faiss_index = None
        tfidf_vectorizer = None
        tfidf_matrix = None
        return False

# --- Hàm xây dựng lại các indices ---
def rebuild_indices():
    """Xây dựng lại cả FAISS index và TF-IDF index"""
    global faiss_index, global_vector_list, global_metadata, tfidf_vectorizer, tfidf_matrix
    
    if not global_vector_list or not global_metadata:
        logger.warning("Không có dữ liệu để xây dựng indices")
        return False
    
    try:
        # 1. Xây dựng FAISS index
        vectors = np.array(global_vector_list).astype('float32')
        dimension = vectors.shape[1]  # Lấy số chiều của vectors
        
        # Tạo index mới và thêm tất cả vectors
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(vectors)
        faiss_index = new_index
        
        logger.info(f"Đã xây dựng lại FAISS index với {faiss_index.ntotal} vectors")
        
        # 2. Xây dựng TF-IDF index
        # Thu thập tất cả các đoạn văn bản
        all_texts = [meta["text"] for meta in global_metadata]
        
        # Kiểm tra số lượng tài liệu
        if len(all_texts) < 2:
            logger.warning("Số lượng tài liệu quá ít để xây dựng TF-IDF index. Cần ít nhất 2 tài liệu.")
            # Tạo một TF-IDF vectorizer đơn giản hơn cho trường hợp ít tài liệu
            new_vectorizer = TfidfVectorizer(
                lowercase=True,
                min_df=1,  # Giảm min_df xuống 1 để hoạt động với ít tài liệu
                max_df=1.0,  # Tăng max_df lên 1.0
                ngram_range=(1, 1),  # Chỉ sử dụng unigrams
                max_features=5000
            )
        else:
            # Tạo và huấn luyện TF-IDF vectorizer mới cho trường hợp thông thường
            new_vectorizer = TfidfVectorizer(
                lowercase=True,
                min_df=1,  # Giảm min_df xuống 1
                max_df=0.95,  # Giảm max_df xuống 0.95
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000
            )
        
        # Tạo ma trận TF-IDF mới
        try:
            new_matrix = new_vectorizer.fit_transform(all_texts)
            tfidf_vectorizer = new_vectorizer
            tfidf_matrix = new_matrix
            
            logger.info(f"Đã xây dựng lại TF-IDF index với {tfidf_matrix.shape[0]} documents và {tfidf_matrix.shape[1]} features")
        except Exception as e:
            logger.error(f"Lỗi khi xây dựng TF-IDF matrix: {str(e)}")
            # Tạo một TF-IDF vectorizer đơn giản nhất có thể
            simple_vectorizer = TfidfVectorizer(
                lowercase=True,
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1)
            )
            new_matrix = simple_vectorizer.fit_transform(all_texts)
            tfidf_vectorizer = simple_vectorizer
            tfidf_matrix = new_matrix
            logger.info(f"Đã xây dựng TF-IDF index đơn giản với {tfidf_matrix.shape[0]} documents")
        
        # Lưu trạng thái sau khi xây dựng lại indices
        save_state()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi xây dựng lại indices: {str(e)}")
        return False

# --- Hàm thêm tài liệu với các phương pháp chunking khác nhau ---
def add_document(text, filename, chunking_method="sentence_windows", config=None):
    """Thêm tài liệu vào hệ thống RAG"""
    global global_metadata, faiss_index, embedder, tfidf_vectorizer, tfidf_matrix, global_all_files, global_vector_list
    
    if not text or len(text.strip()) == 0:
        return 0
    
    # Đảm bảo text là chuỗi
    if isinstance(text, list):
        text = "\n".join(text)
    
    # Xóa tài liệu cũ nếu đã tồn tại
    remove_document(filename)
    
    # Tạo chunks dựa trên phương pháp chunking
    chunks = []
    
    if chunking_method == "sentence_windows":
        chunks = create_sentence_windows(text)
    elif chunking_method == "paragraph":
        chunks = create_paragraph_chunks(text)
    elif chunking_method == "semantic":
        chunks = create_semantic_chunks(text)
    elif chunking_method == "token":
        chunks = create_token_chunks(text)
    elif chunking_method == "hybrid":
        chunks = create_hybrid_chunks(text, config)
    elif chunking_method == "adaptive":
        chunks = create_adaptive_chunks(text)
    elif chunking_method == "hierarchical":
        chunks = create_hierarchical_chunks(text)
    elif chunking_method == "contextual":
        chunks = create_contextual_chunks(text)
    elif chunking_method == "multi_granularity":
        chunks = create_multi_granularity_chunks(text, config)
    else:
        # Mặc định sử dụng sentence_windows
        chunks = create_sentence_windows(text)
        chunking_method = "sentence_windows"
    
    # Đảm bảo chunks là một danh sách chuỗi
    if not isinstance(chunks, list):
        chunks = [chunks]
    
    if not chunks:
        return 0
    
    # Phát hiện các điều khoản luật và đoạn văn quan trọng
    law_article_chunks = []
    important_section_chunks = []
    
    # Mẫu regex để phát hiện điều khoản luật
    law_article_pattern = r'(Điều|ĐIỀU)\s+(\d+)[:\.\s]'
    
    # Mẫu regex để phát hiện các đoạn văn quan trọng
    important_section_patterns = [
        r'(quan trọng|chú ý|lưu ý|chú ý rằng|lưu ý rằng)',
        r'(cần phải|bắt buộc|phải|không được|cấm)',
        r'(quy định|quy chế|nguyên tắc|chính sách)',
        r'(định nghĩa|khái niệm|thuật ngữ)',
        r'(mục đích|phạm vi|đối tượng áp dụng)',
        r'(trách nhiệm|nghĩa vụ|quyền hạn|quyền lợi)',
        r'(hình thức|mức độ|chế tài|xử phạt|xử lý)',
        r'(thời hạn|thời gian|kỳ hạn|thời hiệu)',
        r'(điều kiện|yêu cầu|tiêu chuẩn|tiêu chí)',
        r'(kết luận|tóm tắt|tổng kết|tổng hợp)'
    ]
    
    # Phát hiện các điều khoản luật và đoạn văn quan trọng
    for i, chunk in enumerate(chunks):
        # Phát hiện điều khoản luật
        article_match = re.search(law_article_pattern, chunk)
        if article_match:
            law_article_chunks.append(i)
        
        # Phát hiện đoạn văn quan trọng
        for pattern in important_section_patterns:
            if re.search(pattern, chunk.lower()):
                important_section_chunks.append(i)
                break
    
    # Mã hóa chunks thành vectors
    try:
        vectors = embedder.encode(chunks, convert_to_tensor=False).astype('float32')
    except Exception as e:
        logger.error(f"Lỗi khi mã hóa chunks: {str(e)}")
        return 0
    
    # Thêm vectors vào FAISS index
    if faiss_index is None:
        # Tạo FAISS index mới nếu chưa có
        dimension = vectors.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
    
    # Thêm vectors vào index
    faiss_index.add(vectors)
    
    # Cập nhật metadata
    for i, chunk in enumerate(chunks):
        # Xác định số điều khoản nếu là điều khoản luật
        article_number = None
        if i in law_article_chunks:
            article_match = re.search(law_article_pattern, chunk)
            if article_match:
                article_number = article_match.group(2)
        
        # Thêm metadata cho chunk
        metadata = {
            "text": chunk,
            "filename": filename,
            "chunking_method": chunking_method,
            "is_law_article": i in law_article_chunks,
            "article_number": article_number,
            "is_important_section": i in important_section_chunks,
            "original_text": chunk,  # Lưu trữ văn bản gốc
            "timestamp": datetime.now().isoformat()
        }
        
        # Trích xuất và lưu thông tin trang cụ thể cho chunk
        page_matches = re.findall(r'\[Trang (\d+)\]', chunk)
        if page_matches:
            # Lưu tất cả các trang được đề cập trong chunk
            metadata["pages"] = [int(page) for page in page_matches]
            # Lưu trang đầu tiên làm trang chính
            metadata["primary_page"] = int(page_matches[0])
        else:
            metadata["pages"] = []
            metadata["primary_page"] = None
        
        global_metadata.append(metadata)
        global_vector_list.append(vectors[i])
    
    # Cập nhật TF-IDF matrix
    if tfidf_vectorizer is None:
        # Tạo TF-IDF vectorizer mới nếu chưa có
        tfidf_vectorizer = TfidfVectorizer(
            min_df=1, max_df=0.85, stop_words=None, 
            use_idf=True, norm='l2', ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
    else:
        # Cập nhật TF-IDF matrix với chunks mới
        try:
            # Lấy tất cả các chunks hiện có
            all_chunks = [meta["text"] for meta in global_metadata]
            
            # Tạo lại TF-IDF matrix
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_chunks)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật TF-IDF matrix: {str(e)}")
            # Tạo mới TF-IDF vectorizer nếu có lỗi
            tfidf_vectorizer = TfidfVectorizer(
                min_df=1, max_df=0.85, stop_words=None, 
                use_idf=True, norm='l2', ngram_range=(1, 2)
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform([meta["text"] for meta in global_metadata])
    
    # Cập nhật global_all_files với thông tin về tài liệu mới
    global_all_files[filename] = {
        "chunk_count": len(chunks),
        "chunking_method": chunking_method,
        "timestamp": datetime.now().isoformat()
    }
    
    # Lưu trạng thái
    save_state()
    
    return len(chunks)

# --- Xóa tài liệu ---
def remove_document(filename):
    """Xóa tài liệu từ hệ thống RAG"""
    global global_vector_list, global_metadata, global_all_files, faiss_index, tfidf_vectorizer, tfidf_matrix
    
    if filename not in global_all_files:
        return False, f"Không tìm thấy tài liệu '{filename}'"
    
    try:
        logger.info(f"Bắt đầu xóa tài liệu '{filename}'")
        
        # Kiểm tra tính đồng bộ của dữ liệu trước khi xử lý
        if len(global_vector_list) != len(global_metadata):
            logger.warning(f"Phát hiện không đồng bộ: global_vector_list ({len(global_vector_list)}) vs global_metadata ({len(global_metadata)})")
        
        # Kiểm tra cấu trúc của global_vector_list
        is_nested_array = False
        if len(global_vector_list) == 1 and isinstance(global_vector_list[0], np.ndarray):
            # Trường hợp load_state đã nạp một mảng numpy duy nhất
            is_nested_array = True
            # Chuyển đổi thành danh sách các vector riêng lẻ nếu global_vector_list là một mảng numpy duy nhất
            if len(global_vector_list[0].shape) > 1:
                # Đây là mảng 2D, chuyển về danh sách các vector 1D
                vectors_array = global_vector_list[0]
                global_vector_list = [vectors_array[i] for i in range(vectors_array.shape[0])]
                logger.info(f"Đã chuyển đổi global_vector_list từ mảng numpy sang danh sách {len(global_vector_list)} vector")
        
        # Các chỉ số của vectors cần giữ lại và chỉ số của vectors liên quan đến file cần xóa
        indices_to_keep = []
        indices_to_remove = []
        
        # Tìm các chunks thuộc tài liệu để xóa và các chunks khác để giữ lại
        new_vectors = []
        new_metadata = []
        
        # Lặp qua danh sách metadata và vector một cách an toàn
        max_index = min(len(global_vector_list), len(global_metadata))
        for i in range(max_index):
            try:
                if global_metadata[i]["filename"] != filename:
                    # Giữ lại các metadata và vector không thuộc file cần xóa
                    new_metadata.append(global_metadata[i])
                    new_vectors.append(global_vector_list[i])
                    indices_to_keep.append(i)
                else:
                    # Đánh dấu các indices cần xóa
                    indices_to_remove.append(i)
            except (IndexError, KeyError) as e:
                logger.error(f"Lỗi khi xử lý phần tử thứ {i}: {str(e)}")
                continue
        
        # Ghi log số lượng chunks đã xóa và giữ lại
        logger.info(f"Đã xác định {len(indices_to_remove)} chunks để xóa và {len(indices_to_keep)} chunks để giữ lại")
        
        # Cập nhật danh sách vector và metadata
        global_vector_list = new_vectors
        global_metadata = new_metadata
        
        # Xóa khỏi danh sách tài liệu
        if filename in global_all_files:
            del global_all_files[filename]
        
        # Tạo lại FAISS index 
        if global_vector_list and len(global_vector_list) > 0:
            try:
                # Xác định phương pháp chuyển đổi phù hợp dựa trên loại dữ liệu
                if isinstance(global_vector_list[0], list):
                    # Nếu là list của list
                    vectors = np.array(global_vector_list, dtype=np.float32)
                elif isinstance(global_vector_list[0], np.ndarray):
                    # Nếu đã là mảng numpy
                    if len(global_vector_list) == 1:
                        vectors = global_vector_list[0].astype(np.float32)
                    else:
                        vectors = np.stack(global_vector_list).astype(np.float32)
                else:
                    # Trường hợp khác, thử chuyển đổi bình thường
                    vectors = np.array(global_vector_list, dtype=np.float32)
                
                # Đảm bảo vectors có đúng kích thước
                if len(vectors.shape) == 1:
                    # Nếu chỉ còn 1 vector
                    vector_dim = len(vectors)
                    vectors = vectors.reshape(1, vector_dim)
                    logger.info(f"Reshape vectors thành {vectors.shape}")
                
                # Đảm bảo vectors có 2 chiều
                if len(vectors.shape) != 2:
                    raise ValueError(f"Vectors phải có 2 chiều, hiện tại là {vectors.shape}")
                
                dimension = vectors.shape[1]
                
                # Tạo index mới
                new_index = faiss.IndexFlatL2(dimension)
                new_index.add(vectors)
                faiss_index = new_index
                
                logger.info(f"Đã tạo lại FAISS index với {faiss_index.ntotal} vectors sau khi xóa file")
                
                # Cập nhật TF-IDF chỉ nếu cần thiết
                if len(new_metadata) > 0:
                    # Lấy tất cả các chunks còn lại
                    all_texts = []
                    for meta in new_metadata:
                        if "text" in meta and meta["text"]:
                            all_texts.append(meta["text"])
                    
                    if all_texts:
                        # Tạo lại TF-IDF từ các văn bản còn lại
                        global tfidf_vectorizer, tfidf_matrix
                        if tfidf_vectorizer is not None:
                            try:
                                tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
                                logger.info(f"Đã cập nhật TF-IDF matrix với {len(all_texts)} chunks còn lại")
                            except Exception as e:
                                logger.warning(f"Không thể cập nhật TF-IDF matrix với vectorizer hiện tại: {str(e)}")
                                # Tạo mới TF-IDF vectorizer
                                tfidf_vectorizer = TfidfVectorizer(
                                    min_df=1, max_df=0.85, stop_words=None,
                                    use_idf=True, norm='l2', ngram_range=(1, 2)
                                )
                                tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
                                logger.info(f"Đã tạo TF-IDF index mới với {len(all_texts)} chunks")
                
            except Exception as e:
                logger.error(f"Lỗi khi tạo lại FAISS index: {str(e)}")
                faiss_index = None
                # Thử khởi tạo lại index với kích thước mặc định
                try:
                    initialize_faiss_index()
                    logger.info("Đã khởi tạo lại FAISS index với kích thước mặc định")
                except:
                    logger.error("Không thể khởi tạo lại FAISS index")
                return False, f"Lỗi khi tạo lại index sau khi xóa: {str(e)}"
        else:
            # Nếu không còn vectors nào, đặt các indices về None
            faiss_index = None
            tfidf_matrix = None
            logger.warning("Không còn vector nào sau khi xóa file, đã đặt indices về None")
        
        # Lưu trạng thái
        save_state()
        
        logger.info(f"Đã xóa tài liệu '{filename}' khỏi hệ thống thành công")
        return True, f"Đã xóa tài liệu '{filename}' khỏi hệ thống thành công"
    except IndexError as e:
        logger.error(f"Lỗi chỉ mục khi xóa tài liệu '{filename}': {str(e)}")
        return False, f"Lỗi chỉ mục khi xóa tài liệu: {str(e)}"
    except Exception as e:
        logger.error(f"Lỗi khi xóa tài liệu '{filename}': {str(e)}")
        return False, f"Lỗi khi xóa tài liệu: {str(e)}"

# --- API và tích hợp mô hình ---
# --- Hàm đánh giá hiệu suất ---
def evaluate_performance(test_queries, top_k=5, chunking_method=None):
    """Đánh giá hiệu suất của hệ thống với một tập các câu hỏi thử nghiệm"""
    results = []
    
    for query in test_queries:
        start_time = time.time()
        answer, sources = answer_question(query, top_k=top_k, chunking_method=chunking_method)
        total_time = time.time() - start_time
        
        # Lấy thông tin hiệu suất từ global_performance
        query_perf = next((p for p in global_performance if p["query"] == query), None)
        
        if query_perf:
            result = {
                "query": query,
                "answer": answer,
                "sources": sources,
                "retrieval_time": query_perf.get("retrieval_time", 0),
                "answer_time": query_perf.get("answer_time", 0),
                "total_time": total_time,
                "num_chunks": query_perf.get("num_chunks", 0),
                "chunking_method": query_perf.get("chunking_method", chunking_method or "unknown")
            }
            results.append(result)
    
    # Tính toán các chỉ số trung bình
    avg_retrieval_time = sum(r["retrieval_time"] for r in results) / len(results) if results else 0
    avg_answer_time = sum(r["answer_time"] for r in results) / len(results) if results else 0
    avg_total_time = sum(r["total_time"] for r in results) / len(results) if results else 0
    
    summary = {
        "num_queries": len(results),
        "avg_retrieval_time": avg_retrieval_time,
        "avg_answer_time": avg_answer_time,
        "avg_total_time": avg_total_time,
        "chunking_method": chunking_method or "mixed",
        "results": results
    }
    
    return summary

def track_performance(query, retrieval_time, answer_time, num_chunks, chunking_method):
    """Theo dõi hiệu suất của một câu truy vấn"""
    global global_performance
    
    # Kiểm tra xem câu truy vấn đã tồn tại chưa
    existing_query = next((p for p in global_performance if p["query"] == query), None)
    
    if existing_query:
        # Cập nhật thông tin hiệu suất
        existing_query["retrieval_time"] = retrieval_time
        existing_query["answer_time"] = answer_time
        existing_query["total_time"] = retrieval_time + answer_time
        existing_query["num_chunks"] = num_chunks
        existing_query["chunking_method"] = chunking_method
        existing_query["timestamp"] = time.time()
    else:
        # Thêm thông tin hiệu suất mới
        performance_data = {
            "query": query,
            "retrieval_time": retrieval_time,
            "answer_time": answer_time,
            "total_time": retrieval_time + answer_time,
            "num_chunks": num_chunks,
            "chunking_method": chunking_method,
            "timestamp": time.time()
        }
        global_performance.append(performance_data)
    
    # Giới hạn số lượng bản ghi hiệu suất
    if len(global_performance) > 100:
        global_performance = global_performance[-100:]
    
    # Lưu trạng thái
    save_state()

def analyze_performance(chunking_method=None):
    """Phân tích hiệu suất dựa trên dữ liệu đã theo dõi"""
    global global_performance
    
    if not global_performance:
        return {
            "message": "Chưa có dữ liệu hiệu suất",
            "data": {}
        }
    
    # Lọc theo phương pháp chunking nếu được chỉ định
    filtered_data = global_performance
    if chunking_method:
        filtered_data = [p for p in global_performance if p.get("chunking_method") == chunking_method]
    
    if not filtered_data:
        return {
            "message": f"Không có dữ liệu hiệu suất cho phương pháp chunking '{chunking_method}'",
            "data": {}
        }
    
    # Tính toán các chỉ số trung bình
    avg_retrieval_time = sum(p["retrieval_time"] for p in filtered_data) / len(filtered_data)
    avg_answer_time = sum(p["answer_time"] for p in filtered_data) / len(filtered_data)
    avg_total_time = sum(p["total_time"] for p in filtered_data) / len(filtered_data)
    avg_num_chunks = sum(p["num_chunks"] for p in filtered_data) / len(filtered_data)
    
    # Phân tích theo phương pháp chunking
    chunking_methods = {}
    for p in filtered_data:
        method = p.get("chunking_method", "unknown")
        if method not in chunking_methods:
            chunking_methods[method] = {
                "count": 0,
                "total_retrieval_time": 0,
                "total_answer_time": 0,
                "total_chunks": 0
            }
        
        chunking_methods[method]["count"] += 1
        chunking_methods[method]["total_retrieval_time"] += p["retrieval_time"]
        chunking_methods[method]["total_answer_time"] += p["answer_time"]
        chunking_methods[method]["total_chunks"] += p["num_chunks"]
    
    # Tính toán trung bình cho từng phương pháp
    for method, data in chunking_methods.items():
        if data["count"] > 0:
            data["avg_retrieval_time"] = data["total_retrieval_time"] / data["count"]
            data["avg_answer_time"] = data["total_answer_time"] / data["count"]
            data["avg_total_time"] = (data["total_retrieval_time"] + data["total_answer_time"]) / data["count"]
            data["avg_chunks"] = data["total_chunks"] / data["count"]
    
    # Kết quả phân tích
    analysis = {
        "overall": {
            "count": len(filtered_data),
            "avg_retrieval_time": avg_retrieval_time,
            "avg_answer_time": avg_answer_time,
            "avg_total_time": avg_total_time,
            "avg_num_chunks": avg_num_chunks
        },
        "by_chunking_method": chunking_methods,
        "recent_queries": [
            {
                "query": p["query"],
                "retrieval_time": p["retrieval_time"],
                "answer_time": p["answer_time"],
                "total_time": p["total_time"],
                "num_chunks": p["num_chunks"],
                "chunking_method": p.get("chunking_method", "unknown")
            }
            for p in sorted(filtered_data, key=lambda x: x.get("timestamp", 0), reverse=True)[:5]
        ]
    }
    
    return {
        "message": "Phân tích hiệu suất thành công",
        "data": analysis
    }

# --- Hàm theo dõi hiệu suất ---
def track_performance_metrics(query, retrieval_time, answer_time, chunk_count, chunking_method):
    """Theo dõi metrics hiệu suất của hệ thống RAG"""
    global performance_metrics
    
    performance_metrics["queries"].append(query)
    performance_metrics["retrieval_times"].append(retrieval_time)
    performance_metrics["answer_times"].append(answer_time)
    performance_metrics["chunk_counts"].append(chunk_count)
    performance_metrics["chunking_methods"].append(chunking_method)
    
    # Lưu metrics vào file
    try:
        df = pd.DataFrame(performance_metrics)
        df.to_csv("performance_metrics.csv", index=False)
        logger.info(f"Đã lưu metrics hiệu suất, tổng số queries: {len(performance_metrics['queries'])}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu metrics hiệu suất: {str(e)}")

# --- Hàm phân tích hiệu suất ---
def analyze_performance():
    """Phân tích hiệu suất của hệ thống RAG"""
    try:
        if os.path.exists("performance_metrics.csv"):
            df = pd.read_csv("performance_metrics.csv")
            
            # Tính toán các thống kê
            stats = {
                "total_queries": len(df),
                "avg_retrieval_time": df["retrieval_times"].mean(),
                "avg_answer_time": df["answer_times"].mean(),
                "total_time": df["retrieval_times"].sum() + df["answer_times"].sum(),
                "avg_chunk_count": df["chunk_counts"].mean(),
                "chunking_methods": dict(Counter(df["chunking_methods"]))
            }
            
            # Phân tích theo phương pháp chunking
            chunking_stats = df.groupby("chunking_methods").agg({
                "retrieval_times": "mean",
                "answer_times": "mean",
                "chunk_counts": "mean"
            }).reset_index()
            
            return stats, chunking_stats
        else:
            return {"error": "Không tìm thấy file metrics"}, None
    except Exception as e:
        logger.error(f"Lỗi khi phân tích hiệu suất: {str(e)}")
        return {"error": str(e)}, None

# --- Hàm tối ưu hóa context ---
def build_optimized_context(chunks, metadata, question, max_tokens=4000):
    """
    Xây dựng ngữ cảnh tối ưu từ các chunks đã truy xuất
    
    Hàm này chọn lọc và sắp xếp các chunks liên quan nhất để tạo ra
    ngữ cảnh tối ưu cho việc trả lời câu hỏi, đảm bảo không vượt quá
    giới hạn token của mô hình.
    
    Args:
        chunks (list): Danh sách các đoạn văn bản đã truy xuất
        metadata (list): Metadata của các đoạn
        question (str): Câu hỏi cần trả lời
        max_tokens (int): Số token tối đa cho ngữ cảnh
        
    Returns:
        str: Ngữ cảnh đã tối ưu hóa
    """
    if not chunks:
        return "", []
    
    # Phát hiện số điều luật cụ thể trong câu hỏi
    article_number = None
    article_match = re.search(r'điều\s+(\d+)', question.lower())
    if article_match:
        article_number = article_match.group(1)
    
    # Phát hiện câu hỏi ngắn gọn
    is_short_question = len(question.split()) <= 4
    
    # Phát hiện câu hỏi về điều khoản luật
    is_law_question = False
    law_article_patterns = [
        r'điều\s+\d+', r'khoản\s+\d+', r'điểm\s+\d+',
        r'chính sách', r'nguyên tắc', r'quy định', r'luật'
    ]
    
    # Khởi tạo article_numbers ở đây để đảm bảo nó luôn tồn tại
    article_numbers = []
    
    for pattern in law_article_patterns:
        if re.search(pattern, question.lower()):
            is_law_question = True
            if article_match:
                article_numbers.append(article_match.group(1))
            break
    
    # Phát hiện câu hỏi yêu cầu trả lời dài
    requires_long_answer = False
    long_answer_patterns = [
        r'giải thích chi tiết', r'trình bày đầy đủ', r'liệt kê tất cả',
        r'nêu rõ', r'phân tích', r'so sánh', r'đánh giá', r'tổng hợp',
        r'tóm tắt', r'kết luận', r'tổng kết', r'gồm những gì', r'bao gồm',
        r'các loại', r'các hình thức', r'các yếu tố', r'các nguyên nhân',
        r'các đặc điểm', r'các đặc trưng', r'các bước', r'quy trình', r'cách thức'
    ]
    
    for pattern in long_answer_patterns:
        if re.search(pattern, question.lower()):
            requires_long_answer = True
            # Tăng giới hạn token cho câu hỏi yêu cầu trả lời dài
            max_tokens = max(max_tokens, 8000)  # Tăng lên 8000 tokens cho câu trả lời dài
            break
    
    # Phát hiện câu hỏi yêu cầu trả lời chính xác và đầy đủ
    requires_exact_answer = False
    exact_answer_patterns = [
        r'chính xác', r'đầy đủ', r'chi tiết', r'cụ thể',
        r'toàn bộ', r'tất cả', r'đúng', r'chính xác và đầy đủ'
    ]
    
    for pattern in exact_answer_patterns:
        if re.search(pattern, question.lower()):
            requires_exact_answer = True
            # Tăng giới hạn token cho câu hỏi yêu cầu trả lời chính xác và đầy đủ
            max_tokens = max(max_tokens, 10000)  # Tăng lên 10000 tokens cho câu trả lời chính xác và đầy đủ
            break
    
    # Phát hiện câu hỏi yêu cầu tổng quan
    requires_overview = False
    overview_patterns = [
        r'tổng quan', r'tổng quát', r'tổng thể', r'khái quát',
        r'sơ lược', r'giới thiệu', r'tóm tắt', r'tổng kết'
    ]
    
    for pattern in overview_patterns:
        if re.search(pattern, question.lower()):
            requires_overview = True
            logger.info(f"API: Phát hiện câu hỏi yêu cầu tổng quan: {question}")
            break
    
    # Tối ưu hóa cho câu trả lời dài
    if requires_long_answer or requires_exact_answer:
        # Tìm các chunk có liên quan đến nhau
        related_chunks = []
        used_chunks = set()
        
        for i, chunk in enumerate(chunks):
            if i in used_chunks:
                continue
                
            current_group = [chunk]
            used_chunks.add(i)
            
            # Tìm các chunk liền kề có nội dung liên quan
            for j, next_chunk in enumerate(chunks[i+1:], start=i+1):
                if j in used_chunks:
                    continue
                    
                # Kiểm tra sự liên quan giữa các chunk
                if any(term in next_chunk.lower() for term in chunk.lower().split()):
                    current_group.append(next_chunk)
                    used_chunks.add(j)
                
                # Kiểm tra xem chunk hiện tại có kết thúc ở cuối trang và chunk tiếp theo có bắt đầu ở đầu trang tiếp theo không
                current_page_match = re.search(r'\[Trang (\d+)\]', chunk)
                next_page_match = re.search(r'\[Trang (\d+)\]', next_chunk)
                
                if current_page_match and next_page_match:
                    current_page = int(current_page_match.group(1))
                    next_page = int(next_page_match.group(1))
                    
                    # Nếu trang tiếp theo liền kề với trang hiện tại, có thể là nội dung liên tục
                    if next_page == current_page + 1:
                        current_group.append(next_chunk)
                        used_chunks.add(j)
                    
            related_chunks.append(current_group)
        
        # Kết hợp các nhóm chunk liên quan
        optimized_chunks = []
        for group in related_chunks:
            combined_chunk = " ".join(group)
            optimized_chunks.append(combined_chunk)
            
        chunks = optimized_chunks
    
    # Sắp xếp chunks theo độ liên quan (đã được sắp xếp từ hàm gọi)
    sorted_chunks_with_metadata = list(zip(chunks, metadata))
    
    # Phân loại chunks theo granularity
    document_chunks = []
    section_chunks = []
    paragraph_chunks = []
    sentence_chunks = []
    other_chunks = []
    
    for chunk, meta in sorted_chunks_with_metadata:
        if "[Granularity: document]" in chunk:
            document_chunks.append((chunk, meta))
        elif "[Granularity: section]" in chunk:
            section_chunks.append((chunk, meta))
        elif "[Granularity: paragraph]" in chunk:
            paragraph_chunks.append((chunk, meta))
        elif "[Granularity: sentence]" in chunk:
            sentence_chunks.append((chunk, meta))
        else:
            other_chunks.append((chunk, meta))
    
    # Sắp xếp lại dựa trên loại câu hỏi
    if requires_overview:
        # Ưu tiên document và section chunks cho câu hỏi tổng quan
        sorted_chunks_with_metadata = document_chunks + section_chunks + paragraph_chunks + other_chunks + sentence_chunks
    elif requires_long_answer:
        # Ưu tiên section và paragraph chunks cho câu hỏi yêu cầu trả lời dài
        sorted_chunks_with_metadata = section_chunks + paragraph_chunks + document_chunks + other_chunks + sentence_chunks
    elif requires_exact_answer:
        # Ưu tiên paragraph và sentence chunks cho câu hỏi yêu cầu trả lời chính xác
        sorted_chunks_with_metadata = paragraph_chunks + sentence_chunks + section_chunks + other_chunks + document_chunks
    elif is_law_question:
        # Tìm các điều khoản cụ thể được đề cập trong câu hỏi
        article_match = re.search(r'điều\s+(\d+)', question.lower())
        if article_match:
            article_numbers.append(article_match.group(1))
        
        # Sắp xếp lại để ưu tiên các chunks là điều khoản luật
        law_chunks = []
        other_chunks = []
        
        for chunk, meta in sorted_chunks_with_metadata:
            # Ưu tiên các điều khoản được đề cập trực tiếp trong câu hỏi
            if meta.get("is_law_article", False) and meta.get("article_number") in article_numbers:
                law_chunks.insert(0, (chunk, meta))
            # Sau đó đến các điều khoản luật khác
            elif meta.get("is_law_article", False):
                law_chunks.append((chunk, meta))
            # Cuối cùng là các chunks thông thường
            else:
                other_chunks.append((chunk, meta))
        
        sorted_chunks_with_metadata = law_chunks + other_chunks
    else:
        # Cho các câu hỏi thông thường, ưu tiên paragraph và sentence chunks
        sorted_chunks_with_metadata = paragraph_chunks + sentence_chunks + section_chunks + other_chunks + document_chunks
    
    # Tạo context với thông tin trích dẫn
    context_parts = []
    used_files = []
    total_tokens = 0
    token_limit_reached = False
    
    # Phân tích từ khóa trong câu hỏi để tìm các chunks liên quan
    question_keywords = set(word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 3)
    
    # Tính điểm liên quan cho mỗi chunk dựa trên từ khóa
    chunk_scores = []
    for i, (chunk, meta) in enumerate(sorted_chunks_with_metadata):
        chunk_text = chunk.lower()
        keyword_matches = sum(1 for keyword in question_keywords if keyword in chunk_text)
        
        # Tính điểm dựa trên số lượng từ khóa khớp và vị trí trong danh sách
        relevance_score = keyword_matches * 10 - i * 0.1  # Ưu tiên các chunks có nhiều từ khóa và xuất hiện sớm
        
        # Tăng điểm cho các đoạn quan trọng
        if meta.get("is_important_section", False):
            relevance_score += 5
        
        # Tăng điểm cho các điều khoản luật nếu là câu hỏi về luật
        if is_law_question and meta.get("is_law_article", False):
            relevance_score += 10
            # Tăng thêm điểm nếu điều khoản được đề cập trực tiếp
            article_match = re.search(r'điều\s+(\d+)', question.lower())
            if article_match and meta.get("article_number") == article_match.group(1):
                relevance_score += 20
        
        # Tăng điểm cho các chunks có chứa thông tin ngữ cảnh
        if "[Ngữ cảnh:" in chunk:
            relevance_score += 8
        
        # Tăng điểm cho các chunks có chứa định nghĩa
        if re.search(r'(là|được định nghĩa là|được hiểu là|có nghĩa là|được gọi là)', chunk):
            relevance_score += 6
        
        # Tăng điểm cho các chunks có chứa ví dụ
        if re.search(r'(ví dụ|thí dụ|minh họa|chẳng hạn)', chunk):
            relevance_score += 4
        
        # Tăng điểm cho các chunks có chứa danh sách
        if re.search(r'(các|những)[^:]+:', chunk) or re.search(r'(\d+\.|[a-z]\)|\([a-z]\))', chunk):
            relevance_score += 7
            
        # Điều chỉnh điểm dựa trên granularity
        if "[Granularity: document]" in chunk and requires_overview:
            relevance_score += 15  # Ưu tiên cao nhất cho document-level chunks khi cần tổng quan
        elif "[Granularity: section]" in chunk and requires_overview:
            relevance_score += 10  # Ưu tiên cao cho section-level chunks khi cần tổng quan
        elif "[Granularity: section]" in chunk and requires_long_answer:
            relevance_score += 12  # Ưu tiên cao cho section-level chunks khi cần trả lời dài
        elif "[Granularity: paragraph]" in chunk and requires_exact_answer:
            relevance_score += 8   # Ưu tiên cho paragraph-level chunks khi cần trả lời chính xác
        elif "[Granularity: sentence]" in chunk and not requires_overview:
            relevance_score += 5   # Ưu tiên nhẹ cho sentence-level chunks cho câu hỏi thông thường
        
        # Tăng điểm cho các chunks có nội dung trải dài qua nhiều trang
        if re.search(r'\[Trang \d+\].*\[Trang \d+\]', chunk):
            relevance_score += 10  # Ưu tiên cao cho chunks có nội dung trải dài qua nhiều trang
        
        chunk_scores.append((i, relevance_score))
    
    # Sắp xếp lại các chunks dựa trên điểm liên quan
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in chunk_scores]
    
    # Tạo context từ các chunks đã sắp xếp lại
    for rank, idx in enumerate(sorted_indices):
        chunk, meta = sorted_chunks_with_metadata[idx]
        
        # Ước tính số token trong chunk
        estimated_tokens = len(chunk.split()) * 1.3  # Hệ số ước tính
        
        # Xử lý các trường hợp đặc biệt
        if is_law_question and meta.get("is_law_article", False) and meta.get("article_number") in article_numbers:
            # Đảm bảo không vượt quá giới hạn token tối đa
            if total_tokens + estimated_tokens > max_tokens * 1.5:  # Cho phép vượt 50% giới hạn cho điều khoản quan trọng
                token_limit_reached = True
                break
        elif requires_long_answer and meta.get("is_important_section", False) and rank < 5:
            # Cho phép vượt 30% giới hạn cho các đoạn quan trọng trong câu hỏi yêu cầu trả lời dài
            if total_tokens + estimated_tokens > max_tokens * 1.3:
                token_limit_reached = True
                break
        elif requires_exact_answer:
            # Cho phép vượt 100% giới hạn cho câu hỏi yêu cầu trả lời chính xác và đầy đủ
            if total_tokens + estimated_tokens > max_tokens * 2:
                token_limit_reached = True
                break
        elif requires_overview and ("[Granularity: document]" in chunk or "[Granularity: section]" in chunk):
            # Cho phép vượt 50% giới hạn cho document/section chunks trong câu hỏi tổng quan
            if total_tokens + estimated_tokens > max_tokens * 1.5:
                token_limit_reached = True
                break
        # Nếu không, áp dụng giới hạn token thông thường
        elif total_tokens + estimated_tokens > max_tokens and rank > 0:
            token_limit_reached = True
            break
        
        # Thêm thông tin trích dẫn
        filename = meta.get("filename", "Unknown")
        page = meta.get("page", "")
        page_info = f" (trang {page})" if page else ""
        
        # Thêm thông tin đoạn văn gốc nếu có
        original_text = meta.get("original_text", chunk)
        
        # Loại bỏ thông tin granularity nếu có
        if "[Granularity:" in original_text:
            original_text = re.sub(r'\[Granularity: [^\]]+\]\s*', '', original_text)
        
        # Định dạng trích dẫn
        citation = f"[{rank+1}] Từ: {filename}{page_info}"
        
        # Đảm bảo giữ nguyên định dạng liệt kê trong văn bản
        # Bảo vệ các định dạng liệt kê như a), b), c), (a), (b), (c), v.v.
        original_text = re.sub(r'(\s[a-z]\)|\s\([a-z]\))', lambda m: m.group(0), original_text)
        
        # Thêm vào context
        context_part = f"{citation}\n{original_text}\n\n"
        context_parts.append(context_part)
        
        # Cập nhật số token và danh sách file đã sử dụng
        total_tokens += estimated_tokens
        if filename not in used_files:
            used_files.append(filename)
    
    # Thêm thông báo nếu đã cắt bớt context
    if token_limit_reached:
        context_parts.append("(Đã cắt bớt một số nội dung do giới hạn độ dài)")
    
    # Sắp xếp context cho câu hỏi điều luật
    if is_law_question and article_number:
        # Đưa các đoạn văn chứa điều luật cụ thể lên đầu context
        reordered_context_parts = []
        remaining_parts = []
        
        for part in context_parts:
            part_lower = part.lower()
            if f"điều {article_number}" in part_lower or f"điều {article_number}." in part_lower:
                reordered_context_parts.append(part)
            else:
                remaining_parts.append(part)
        
        # Nếu tìm thấy đoạn văn có điều luật cụ thể, kết hợp với các đoạn còn lại
        if reordered_context_parts:
            context_parts = reordered_context_parts + remaining_parts
            logger.info(f"Đã sắp xếp lại context, đưa {len(reordered_context_parts)} đoạn văn chứa điều {article_number} lên đầu")
    
    # Kết hợp tất cả các phần
    context = "\n".join(context_parts)
    
    return context, used_files

# --- Các hàm mới cho Query Transformation và Reranking ---

def transform_query_for_vietnamese(query, model="gemini-2.0-flash"):
    """
    Chuẩn hóa truy vấn tiếng Việt
    
    Hàm này thực hiện chuẩn hóa cơ bản cho câu truy vấn tiếng Việt.
    Không còn tạo các biến thể truy vấn.
    
    Args:
        query (str): Truy vấn gốc
        model (str): Không còn sử dụng, giữ lại để tương thích
        
    Returns:
        dict: Kết quả chứa truy vấn gốc và đã chuẩn hóa
    """
    # Chuẩn hóa truy vấn
    normalized_query = query
    if VIETNAMESE_NLP_AVAILABLE:
        try:
            normalized_query = text_normalize(query)
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn hóa truy vấn: {str(e)}")
    
    # Trả về kết quả đơn giản
    return {
        "original_query": query,
        "normalized_query": normalized_query,
        "query_variants": [query],
        "analysis": {}
    }

def rerank_results_for_vietnamese(question, chunks, metadata, top_k=10):
    """
    Sắp xếp lại kết quả truy xuất cho tiếng Việt
    
    Hàm này áp dụng các kỹ thuật reranking đặc biệt cho tiếng Việt
    để cải thiện thứ tự của các đoạn văn bản đã truy xuất, giúp
    đưa các đoạn liên quan nhất lên đầu.
    
    Args:
        question (str): Câu hỏi gốc
        chunks (list): Danh sách các đoạn văn bản đã truy xuất
        metadata (list): Metadata của các đoạn
        top_k (int): Số lượng kết quả trả về
        
    Returns:
        tuple: (chunks đã sắp xếp lại, metadata tương ứng)
    """
    if not chunks or len(chunks) == 0:
        return [], []
    
    # Chuẩn bị dữ liệu
    chunk_texts = [chunk for chunk in chunks]
    
    # Tokenize câu hỏi và các đoạn văn bằng ViTokenizer nếu có
    tokenized_question = question
    tokenized_chunks = chunk_texts
    
    if VIETNAMESE_NLP_AVAILABLE:
        try:
            tokenized_question = ViTokenizer.tokenize(question)
            tokenized_chunks = [ViTokenizer.tokenize(chunk) for chunk in chunk_texts]
        except Exception as e:
            logger.error(f"Lỗi khi tokenize văn bản tiếng Việt: {str(e)}")
    
    # 1. Tính điểm BM25
    try:
        # Tokenize và tách từ
        tokenized_corpus = [doc.split() for doc in tokenized_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = tokenized_question.split()
        bm25_scores = bm25.get_scores(tokenized_query)
    except Exception as e:
        logger.error(f"Lỗi khi tính điểm BM25: {str(e)}")
        bm25_scores = np.ones(len(chunk_texts))
    
    # 2. Tính điểm dựa trên sự xuất hiện của từ khóa
    keyword_scores = []
    # Trích xuất từ khóa từ câu hỏi (từ có độ dài > 1)
    keywords = [word for word in re.findall(r'\b\w+\b', question.lower()) if len(word) > 1]
    
    for chunk in chunk_texts:
        chunk_lower = chunk.lower()
        # Đếm số lần xuất hiện của từ khóa trong đoạn văn
        keyword_count = sum(1 for keyword in keywords if keyword in chunk_lower)
        # Tính tỷ lệ từ khóa trên tổng số từ khóa
        keyword_ratio = keyword_count / len(keywords) if keywords else 0
        keyword_scores.append(keyword_ratio)
    
    # 3. Tính điểm dựa trên vị trí và độ dài
    position_scores = []
    length_scores = []
    
    # Chuẩn hóa độ dài đoạn văn (ưu tiên đoạn văn có độ dài vừa phải)
    lengths = [len(chunk.split()) for chunk in chunk_texts]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    
    for i, length in enumerate(lengths):
        # Điểm vị trí (ưu tiên các đoạn xuất hiện sớm)
        position_scores.append(1.0 / (i + 1))
        
        # Điểm độ dài (ưu tiên đoạn có độ dài gần với độ dài trung bình)
        length_diff = abs(length - avg_length) / avg_length if avg_length > 0 else 0
        length_scores.append(1.0 / (1.0 + length_diff))
    
    # 4. Tính điểm dựa trên granularity phù hợp với loại câu hỏi
    granularity_scores = []
    
    # Phát hiện loại câu hỏi
    is_overview_question = any(pattern in question.lower() for pattern in [
        'tổng quan', 'tổng quát', 'tổng thể', 'khái quát', 'sơ lược', 'giới thiệu'
    ])
    
    is_detailed_question = any(pattern in question.lower() for pattern in [
        'chi tiết', 'cụ thể', 'đầy đủ', 'chính xác', 'toàn bộ', 'tất cả'
    ])
    
    is_long_answer_question = any(pattern in question.lower() for pattern in [
        'giải thích', 'trình bày', 'liệt kê', 'phân tích', 'so sánh', 'đánh giá'
    ])
    
    for chunk in chunk_texts:
        # Mặc định điểm granularity
        granularity_score = 0.5
        
        # Điều chỉnh điểm dựa trên loại câu hỏi và granularity của chunk
        if is_overview_question:
            if "[Granularity: document]" in chunk:
                granularity_score = 1.0  # Điểm cao nhất cho document-level chunks
            elif "[Granularity: section]" in chunk:
                granularity_score = 0.8  # Điểm cao cho section-level chunks
            else:
                granularity_score = 0.3  # Điểm thấp cho các loại chunks khác
        elif is_detailed_question:
            if "[Granularity: paragraph]" in chunk:
                granularity_score = 1.0  # Điểm cao nhất cho paragraph-level chunks
            elif "[Granularity: sentence]" in chunk:
                granularity_score = 0.8  # Điểm cao cho sentence-level chunks
            else:
                granularity_score = 0.4  # Điểm trung bình cho các loại chunks khác
        elif is_long_answer_question:
            if "[Granularity: section]" in chunk:
                granularity_score = 1.0  # Điểm cao nhất cho section-level chunks
            elif "[Granularity: paragraph]" in chunk:
                granularity_score = 0.8  # Điểm cao cho paragraph-level chunks
            else:
                granularity_score = 0.5  # Điểm trung bình cho các loại chunks khác
        
        granularity_scores.append(granularity_score)
    
    # 5. Kết hợp các điểm số
    combined_scores = []
    for i in range(len(chunk_texts)):
        # Trọng số cho từng loại điểm
        bm25_weight = 0.4
        keyword_weight = 0.3
        position_weight = 0.1
        length_weight = 0.1
        granularity_weight = 0.1
        
        # Chuẩn hóa điểm BM25
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        normalized_bm25 = bm25_scores[i] / max_bm25
        
        # Tính điểm tổng hợp
        score = (bm25_weight * normalized_bm25 +
                keyword_weight * keyword_scores[i] +
                position_weight * position_scores[i] +
                length_weight * length_scores[i] +
                granularity_weight * granularity_scores[i])
        
        # Tăng điểm cho các đoạn văn quan trọng
        if metadata[i].get("is_important_section", False):
            score *= 1.5
        
        # Tăng điểm cho các điều khoản luật nếu câu hỏi liên quan đến luật
        law_patterns = [r'điều\s+\d+', r'khoản\s+\d+', r'điểm\s+\d+', r'chính sách', r'nguyên tắc', r'quy định', r'luật']
        is_law_question = any(re.search(pattern, question.lower()) for pattern in law_patterns)
        
        if is_law_question and metadata[i].get("is_law_article", False):
            score *= 2.0
            
            # Tăng điểm gấp 3 lần cho điều khoản được đề cập trực tiếp
            article_match = re.search(r'điều\s+(\d+)', question.lower())
            if article_match and metadata[i].get("article_number") == article_match.group(1):
                score *= 1.5
        
        combined_scores.append((i, score))
    
    # Sắp xếp theo điểm số giảm dần
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy top_k kết quả
    top_indices = [idx for idx, _ in combined_scores[:top_k]]
    reranked_chunks = [chunk_texts[idx] for idx in top_indices]
    reranked_metadata = [metadata[idx] for idx in top_indices]
    
    return reranked_chunks, reranked_metadata

def split_multiple_questions(question):
    """
    Phân tách nhiều câu hỏi trong một câu đầu vào
    
    Args:
        question (str): Câu hỏi đầu vào có thể chứa nhiều câu hỏi
        
    Returns:
        list: Danh sách các câu hỏi đã tách
    """
    # Đảm bảo question không phải None và là chuỗi
    if question is None:
        return [""]
        
    # Tạo danh sách để lưu tất cả các câu hỏi tìm thấy
    all_questions = []
    
    # Bước 1: Tách câu hỏi dựa trên từ khóa kết nối
    connectors = [r'và', r'cùng với', r'đồng thời', r'ngoài ra', r'bên cạnh đó', r'thêm vào đó', r',']
    connector_pattern = r'|'.join([fr'({c})' for c in connectors])
    
    # Tìm tất cả câu kết thúc bằng dấu hỏi và phân tách theo từ kết nối
    question_segments = re.split(f'({connector_pattern})\\s+', question)
    
    # Xử lý các phân đoạn và cố gắng xây dựng lại các câu hỏi hoàn chỉnh
    current_segment = ""
    for i, segment in enumerate(question_segments):
        # Segment có thể None khi phân tách regex
        if segment is None:
            continue
            
        # Bỏ qua nếu segment là từ kết nối
        if any(re.match(f'^{c}$', segment.strip(), re.IGNORECASE) for c in connectors):
            # Nếu segment hiện tại có nội dung và kết thúc bằng dấu hỏi, thêm vào danh sách
            if current_segment and current_segment.strip().endswith('?'):
                all_questions.append(current_segment.strip())
                current_segment = ""
            continue
        
        # Thêm segment vào câu hỏi hiện tại
        current_segment += segment
        
        # Nếu kết thúc bằng dấu hỏi, thêm vào danh sách và reset
        if segment.strip().endswith('?'):
            all_questions.append(current_segment.strip())
            current_segment = ""
    
    # Thêm phần còn lại nếu có
    if current_segment.strip():
        # Thêm dấu hỏi nếu chưa có
        if not current_segment.strip().endswith('?'):
            current_segment += '?'
        all_questions.append(current_segment.strip())
    
    # Bước 2: Tách câu hỏi dựa trên dấu chấm câu
    if not all_questions:
        sentences = re.split(r'[.!?]\s+', question)
        
        for sentence in sentences:
            # Chỉ thêm câu có dấu hỏi hoặc có từ khóa hỏi
            if ('?' in sentence or 
                any(keyword in sentence.lower() for keyword in 
                    ['là gì', 'như thế nào', 'ra sao', 'thế nào', 'ở đâu', 'khi nào', 
                     'tại sao', 'vì sao', 'bao nhiêu', 'như nào', 'làm sao', 'làm thế nào'])):
                # Thêm dấu hỏi nếu không có
                if not sentence.strip().endswith('?'):
                    sentence += '?'
                all_questions.append(sentence.strip())
    
    # Bước 3: Tìm kiếm các câu hỏi dạng đánh số hoặc dấu chấm
    numbered_questions = re.findall(r'(?:\d+[\.\)]\s*|\([a-zA-Z]\)\s*|[a-zA-Z]\)\s*)([^.!?]+\??)', question)
    if numbered_questions:
        for q in numbered_questions:
            clean_q = q.strip()
            if not clean_q.endswith('?'):
                clean_q += '?'
            if clean_q not in all_questions:
                all_questions.append(clean_q)
    
    # Nếu không tìm thấy câu hỏi nào, trả về câu hỏi gốc
    if not all_questions:
        return [question]
    
    # Loại bỏ câu quá ngắn và câu trùng lặp
    filtered_questions = []
    for q in all_questions:
        if len(q.split()) >= 2 and q not in filtered_questions:
            filtered_questions.append(q)
    
    logger.info(f"Đã tách thành {len(filtered_questions)} câu hỏi: {filtered_questions}")
    return filtered_questions if filtered_questions else [question]

# --- Hàm trả lời câu hỏi với nhiều cải tiến ---
def answer_question(question, top_k=20, threshold=5.0, model="gemini-2.0-flash", chunking_method=None):
    """
    Trả lời câu hỏi dựa trên tài liệu đã tải lên
    
    Đây là hàm chính của hệ thống RAG, thực hiện toàn bộ quy trình:
    1. Biến đổi truy vấn để tối ưu cho tiếng Việt
    2. Truy xuất các đoạn văn bản liên quan nhất
    3. Sắp xếp lại kết quả truy xuất
    4. Xây dựng ngữ cảnh tối ưu
    5. Tạo câu trả lời từ mô hình ngôn ngữ
    
    Args:
        question (str): Câu hỏi cần trả lời
        top_k (int): Số lượng đoạn văn bản truy xuất
        threshold (float): Ngưỡng điểm số để chọn đoạn văn bản
        model (str): Mô hình ngôn ngữ sử dụng
        chunking_method (str): Phương pháp chunking sử dụng
        
    Returns:
        dict: Kết quả bao gồm câu trả lời, nguồn tài liệu, và các thông tin khác
    """
    global faiss_index, global_metadata, tfidf_vectorizer, tfidf_matrix
    
    # Nạp trạng thái hệ thống của người dùng đang đăng nhập
    load_state()
    
    if faiss_index is None or len(global_metadata) == 0:
        return {
            "answer": "Chưa có tài liệu nào để tra cứu.",
            "sources": [],
            "suggested_questions": []
        }
    
    # Phân tách nhiều câu hỏi nếu có
    questions = split_multiple_questions(question)
    
    # Nếu có nhiều câu hỏi, xử lý từng câu và kết hợp kết quả
    if len(questions) > 1:
        logger.info(f"Xử lý {len(questions)} câu hỏi riêng biệt")
        
        all_answers = []
        all_sources = []
        all_suggested_questions = {}  # Sử dụng dict để lưu gợi ý cho từng câu hỏi không tìm thấy thông tin
        
        for i, single_question in enumerate(questions):
            logger.info(f"Xử lý câu hỏi {i+1}/{len(questions)}: {single_question}")
            
            # Gọi đệ quy để xử lý từng câu hỏi
            result = answer_question(single_question, top_k, threshold, model, chunking_method)
            
            # Thêm câu hỏi vào đầu câu trả lời
            answer_with_question = f"{single_question}\n\n{result['answer']}"
            all_answers.append(answer_with_question)
            
            # Thêm nguồn tài liệu
            all_sources.extend(result['sources'])
            
            # Lưu gợi ý cho câu hỏi không tìm thấy thông tin
            if "Không có thông tin liên quan trong tài liệu." in result['answer'] and result['suggested_questions']:
                all_suggested_questions[single_question] = result['suggested_questions']
        
        # Kết hợp các câu trả lời
        combined_answer = "\n\n---\n\n".join(all_answers)
        
        # Loại bỏ các nguồn trùng lặp
        unique_sources = []
        for source in all_sources:
            if source not in unique_sources:
                unique_sources.append(source)
        
        return {
            "answer": combined_answer,
            "sources": unique_sources,
            "suggested_questions": all_suggested_questions  # Trả về dict gợi ý cho từng câu hỏi
        }
    
    # Bắt đầu đo thời gian truy xuất
    start_retrieval = time.time()
    
    # Phát hiện câu hỏi về điều khoản luật
    is_law_question = False
    article_number = None
    law_article_patterns = [
        r'điều\s+\d+', r'khoản\s+\d+', r'điểm\s+\d+',
        r'chính sách', r'nguyên tắc', r'quy định', r'luật'
    ]
    
    # Phát hiện số điều luật cụ thể
    article_match = re.search(r'điều\s+(\d+)', question.lower())
    if article_match:
        article_number = article_match.group(1)
        is_law_question = True
        
        # Phát hiện câu hỏi ngắn gọn về điều luật như "Điều 33 gồm?" hoặc "Điều 33?"
        if len(question.split()) <= 4:
            # Mở rộng câu hỏi để tăng khả năng tìm kiếm
            original_question = question
            expanded_questions = [
                f"Điều {article_number} bao gồm những gì?",
                f"Nội dung của Điều {article_number} là gì?",
                f"Điều {article_number} quy định những gì?",
                f"Điều {article_number} nói về vấn đề gì?"
            ]
            
            # Lưu câu hỏi gốc
            logger.info(f"Phát hiện câu hỏi ngắn gọn về điều luật: {question}")
            logger.info(f"Mở rộng thành: {expanded_questions}")
            
            # Thử từng câu hỏi mở rộng cho đến khi tìm được kết quả
            for expanded_question in expanded_questions:
                # Thực hiện tìm kiếm với câu hỏi mở rộng
                # Mã hóa câu hỏi mở rộng
                q_vec = embedder.encode(expanded_question, convert_to_tensor=False).astype('float32')
                q_vec = np.expand_dims(q_vec, axis=0)
                
                # Tìm kiếm với FAISS
                distances, indices = faiss_index.search(q_vec, min(top_k, faiss_index.ntotal))
                
                # Kiểm tra kết quả
                if min(distances[0]) < threshold:
                    # Nếu tìm thấy kết quả tốt, sử dụng câu hỏi mở rộng này
                    logger.info(f"Tìm thấy kết quả tốt với câu hỏi mở rộng: {expanded_question}")
                    question = expanded_question
                    break
    
    # Kiểm tra các mẫu câu hỏi ngắn gọn khác (không chỉ về điều luật)
    if len(question.split()) <= 4 and not is_law_question:
        # Kiểm tra xem có phải định dạng câu hỏi "X là gì?" hoặc "X gồm?" không
        term_match = re.match(r'^([^?]+)(\s+(là gì|gồm|bao gồm|nghĩa là gì|định nghĩa|gồm những gì))?\??$', question)
        
        if term_match:
            term = term_match.group(1).strip()
            if len(term) > 0:
                # Mở rộng câu hỏi
                original_question = question
                expanded_questions = [
                    f"{term} là gì?",
                    f"{term} được định nghĩa như thế nào?",
                    f"{term} bao gồm những gì?",
                    f"{term} có ý nghĩa gì?"
                ]
                
                logger.info(f"Phát hiện câu hỏi ngắn gọn: {question}")
                logger.info(f"Mở rộng thành: {expanded_questions}")
                
                # Thử từng câu hỏi mở rộng
                for expanded_question in expanded_questions:
                    # Mã hóa câu hỏi mở rộng
                    q_vec = embedder.encode(expanded_question, convert_to_tensor=False).astype('float32')
                    q_vec = np.expand_dims(q_vec, axis=0)
                    
                    # Tìm kiếm với FAISS
                    distances, indices = faiss_index.search(q_vec, min(top_k, faiss_index.ntotal))
                    
                    # Kiểm tra kết quả
                    if min(distances[0]) < threshold:
                        # Nếu tìm thấy kết quả tốt, sử dụng câu hỏi mở rộng này
                        logger.info(f"Tìm thấy kết quả tốt với câu hỏi mở rộng: {expanded_question}")
                        question = expanded_question
                        break
    
    # Tiếp tục kiểm tra các mẫu điều luật
    for pattern in law_article_patterns:
        if re.search(pattern, question.lower()):
            is_law_question = True
            logger.info(f"API: Phát hiện câu hỏi về điều khoản luật: {question}")
            # Tăng top_k cho câu hỏi về luật
            top_k = max(top_k, 30)
            break
    
    # Phát hiện câu hỏi yêu cầu trả lời dài
    requires_long_answer = False
    long_answer_patterns = [
        r'giải thích chi tiết', r'trình bày đầy đủ', r'liệt kê tất cả',
        r'nêu rõ', r'phân tích', r'so sánh', r'đánh giá', r'tổng hợp',
        r'tóm tắt', r'kết luận', r'tổng kết', r'gồm những gì', r'bao gồm',
        r'các loại', r'các hình thức', r'các yếu tố', r'các nguyên nhân',
        r'các đặc điểm', r'các đặc trưng', r'các bước', r'quy trình', r'cách thức'
    ]
    
    for pattern in long_answer_patterns:
        if re.search(pattern, question.lower()):
            requires_long_answer = True
            logger.info(f"API: Phát hiện câu hỏi yêu cầu trả lời dài: {question}")
            # Tăng top_k cho câu hỏi yêu cầu trả lời dài
            top_k = max(top_k, 25)
            break
    
    # Phát hiện câu hỏi yêu cầu trả lời chính xác và đầy đủ
    requires_exact_answer = False
    exact_answer_patterns = [
        r'chính xác', r'đầy đủ', r'chi tiết', r'cụ thể',
        r'toàn bộ', r'tất cả', r'đúng', r'chính xác và đầy đủ'
    ]
    
    for pattern in exact_answer_patterns:
        if re.search(pattern, question.lower()):
            requires_exact_answer = True
            logger.info(f"API: Phát hiện câu hỏi yêu cầu trả lời chính xác và đầy đủ: {question}")
            # Tăng top_k cho câu hỏi yêu cầu trả lời chính xác và đầy đủ
            top_k = max(top_k, 30)
            break
    
    # Phát hiện câu hỏi tổng quan
    requires_overview = False
    overview_patterns = [
        r'tổng quan', r'tổng quát', r'tổng thể', r'khái quát',
        r'sơ lược', r'giới thiệu', r'tóm tắt', r'tổng kết'
    ]
    
    for pattern in overview_patterns:
        if re.search(pattern, question.lower()):
            requires_overview = True
            logger.info(f"API: Phát hiện câu hỏi yêu cầu tổng quan: {question}")
            break
    
    # Điều chỉnh top_k dựa trên loại câu hỏi
    if is_law_question:
        top_k = max(top_k, 30)  # Tăng số lượng chunks cho câu hỏi về luật
    elif requires_long_answer:
        top_k = max(top_k, 25)  # Tăng số lượng chunks cho câu hỏi yêu cầu trả lời dài
    elif requires_exact_answer:
        top_k = max(top_k, 30)  # Tăng số lượng chunks cho câu hỏi yêu cầu trả lời chính xác và đầy đủ
    
    # --- QUERY TRANSFORMATION ---
    # Không sử dụng biến đổi truy vấn nữa, chỉ dùng câu hỏi gốc
    query_variants = [question]
    logger.info(f"Sử dụng câu hỏi gốc để tìm kiếm: {question}")
    
    # Tìm kiếm với câu hỏi gốc
    all_indices = []
    all_distances = []
    article_specific_indices = []  # Để lưu trữ các chỉ mục điều luật cụ thể
    article_specific_found = False  # Biến cờ để đánh dấu đã tìm thấy điều luật cụ thể chưa
    
    try:
        # Mã hóa câu hỏi
        q_vec = embedder.encode(question, convert_to_tensor=False).astype('float32')
        q_vec = np.expand_dims(q_vec, axis=0)
        
        # Tìm kiếm với FAISS
        distances, indices = faiss_index.search(q_vec, min(top_k, faiss_index.ntotal))
        
        # Thêm vào kết quả
        all_indices.extend(indices[0])
        all_distances.extend(distances[0])
        
        # Nếu là câu hỏi về điều luật cụ thể
        if is_law_question and article_number:
            # Tìm trực tiếp các đoạn văn chứa điều luật cụ thể
            for idx, meta in enumerate(global_metadata):
                file_text = meta.get("text", "").lower()
                meta_article_number = meta.get("article_number")
                
                # Kiểm tra bằng article_number trong metadata
                if meta_article_number == article_number:
                    article_specific_indices.append(idx)
                    article_specific_found = True
                    logger.info(f"Tìm thấy điều luật {article_number} trong metadata")
                # Hoặc kiểm tra qua nội dung văn bản
                elif f"điều {article_number}" in file_text or f"điều {article_number}." in file_text:
                    article_specific_indices.append(idx)
                    article_specific_found = True
                    logger.info(f"Tìm thấy điều luật {article_number} trong nội dung")
            
            # Nếu tìm thấy điều luật cụ thể, thêm vào kết quả
            if article_specific_found:
                logger.info(f"Đã tìm thấy {len(article_specific_indices)} đoạn văn chứa điều luật {article_number}")
                # Thêm vào đầu kết quả
                for idx in article_specific_indices:
                    if idx not in all_indices:
                        all_indices.insert(0, idx)
                        # Gán một khoảng cách rất nhỏ để đảm bảo nó được ưu tiên cao nhất
                        all_distances.insert(0, 0.01)
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm với câu hỏi '{question}': {str(e)}")
    
    # Loại bỏ các kết quả trùng lặp và giữ khoảng cách tốt nhất
    unique_indices = {}
    for idx, dist in zip(all_indices, all_distances):
        if idx not in unique_indices or dist < unique_indices[idx]:
            unique_indices[idx] = dist
    
    # Kiểm tra ngưỡng khoảng cách
    if unique_indices:
        best_distance = min(unique_indices.values())
        if best_distance > threshold * 2:  # Nới lỏng ngưỡng cho hybrid search
            logger.info(f"Không tìm thấy chunks phù hợp. Khoảng cách tốt nhất: {best_distance}")
            # Tạo câu hỏi gợi ý khi không tìm thấy thông tin
            suggested_questions = generate_similar_questions(question, global_metadata)
            return {
                "answer": "Không có thông tin liên quan trong tài liệu.",
                "sources": [],
                "suggested_questions": suggested_questions
            }
    else:
        logger.info("Không tìm thấy kết quả nào từ tìm kiếm ngữ nghĩa")
        # Tạo câu hỏi gợi ý khi không tìm thấy thông tin
        suggested_questions = generate_similar_questions(question, global_metadata)
        return {
            "answer": "Không có thông tin liên quan trong tài liệu.",
            "sources": [],
            "suggested_questions": suggested_questions
        }
    
    # 2. Tìm kiếm từ khóa với TF-IDF nếu có
    keyword_scores = {}
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        try:
            # Chuyển câu hỏi thành vector TF-IDF
            q_tfidf = tfidf_vectorizer.transform([question])
            
            # Tính độ tương đồng cosine
            cosine_similarities = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
            
            # Lấy top k kết quả từ TF-IDF
            top_tfidf_indices = cosine_similarities.argsort()[-top_k*2:][::-1]
            
            # Lưu điểm số
            for idx in top_tfidf_indices:
                keyword_scores[idx] = cosine_similarities[idx]
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm TF-IDF: {str(e)}")
    
    # 3. Kết hợp kết quả từ cả hai phương pháp
    combined_scores = {}
    
    # Thêm kết quả từ tìm kiếm ngữ nghĩa
    for idx, dist in unique_indices.items():
        if idx < len(global_metadata):
            # Chuẩn hóa khoảng cách thành điểm (càng gần càng cao)
            semantic_score = 1.0 / (1.0 + dist)
            
            # Tăng điểm cho các điều khoản luật nếu là câu hỏi về luật
            if is_law_question and global_metadata[idx].get("is_law_article", False):
                # Tìm các điều khoản cụ thể được đề cập trong câu hỏi
                article_match = re.search(r'điều\s+(\d+)', question.lower())
                article_number = None
                if article_match:
                    article_number = article_match.group(1)
                
                if article_number and global_metadata[idx].get("article_number") == article_number:
                    # Tăng điểm gấp 3 lần cho điều khoản được đề cập trực tiếp
                    semantic_score *= 3.0
                else:
                    # Tăng điểm gấp 2 lần cho các điều khoản luật khác
                    semantic_score *= 2.0
            
            # Tăng điểm cho các đoạn văn quan trọng nếu là câu hỏi yêu cầu trả lời dài
            if requires_long_answer and global_metadata[idx].get("is_important_section", False):
                semantic_score *= 1.5
            
            # Tăng điểm cho các đoạn văn quan trọng nếu là câu hỏi yêu cầu trả lời chính xác và đầy đủ
            if requires_exact_answer and global_metadata[idx].get("is_important_section", False):
                semantic_score *= 2.0
                
            # Điều chỉnh điểm dựa trên granularity cho multi-granularity chunks
            chunk_text = global_metadata[idx].get("text", "")
            if "[Granularity:" in chunk_text:
                # Ưu tiên các chunks có granularity phù hợp với loại câu hỏi
                if requires_overview and "document" in chunk_text:
                    semantic_score *= 3.0  # Ưu tiên cao nhất cho document-level chunks khi cần tổng quan
                elif requires_overview and "section" in chunk_text:
                    semantic_score *= 2.0  # Ưu tiên cao cho section-level chunks khi cần tổng quan
                elif requires_long_answer and "section" in chunk_text:
                    semantic_score *= 2.0  # Ưu tiên cao cho section-level chunks khi cần trả lời dài
                elif requires_exact_answer and "paragraph" in chunk_text:
                    semantic_score *= 1.5  # Ưu tiên cho paragraph-level chunks khi cần trả lời chính xác
                elif not requires_overview and "sentence" in chunk_text:
                    semantic_score *= 1.2  # Ưu tiên nhẹ cho sentence-level chunks cho câu hỏi thông thường
            
            combined_scores[idx] = semantic_score * 0.7  # Trọng số 70% cho tìm kiếm ngữ nghĩa
    
    # Thêm kết quả từ tìm kiếm từ khóa
    for idx, score in keyword_scores.items():
        if idx in combined_scores:
            combined_scores[idx] += score * 0.3  # Trọng số 30% cho tìm kiếm từ khóa
        else:
            combined_scores[idx] = score * 0.3
    
    # Sắp xếp kết quả theo điểm số
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in sorted_results[:top_k*2]]  # Lấy nhiều hơn để rerank
    
    # Lấy các chunks liên quan
    retrieved_chunks = []
    retrieved_metadata = []
    
    for idx in top_indices:
        if idx < len(global_metadata):
            retrieved_chunks.append(global_metadata[idx]["text"])
            retrieved_metadata.append(global_metadata[idx])
    
    # --- RERANKING ---
    # Sắp xếp lại kết quả để tăng độ chính xác
    try:
        reranked_chunks, reranked_metadata = rerank_results_for_vietnamese(
            question, retrieved_chunks, retrieved_metadata, top_k=top_k
        )
        logger.info(f"Đã rerank {len(retrieved_chunks)} chunks xuống còn {len(reranked_chunks)} chunks")
        
        # Sử dụng kết quả đã rerank
        retrieved_chunks = reranked_chunks
        retrieved_metadata = reranked_metadata
    except Exception as e:
        logger.error(f"Lỗi khi rerank kết quả: {str(e)}")
        # Giữ nguyên kết quả ban đầu nếu có lỗi, nhưng giới hạn số lượng
        retrieved_chunks = retrieved_chunks[:top_k]
        retrieved_metadata = retrieved_metadata[:top_k]
    
    # Kết thúc đo thời gian truy xuất
    end_retrieval = time.time()
    retrieval_time = end_retrieval - start_retrieval
    
    # Bắt đầu đo thời gian trả lời
    start_answer = time.time()
    
    # Tối ưu hóa context
    context, used_files = build_optimized_context(retrieved_chunks, retrieved_metadata, question)
    
    # Xây dựng prompt tối ưu
    system_prompt = """Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu. Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, đầy đủ và trực tiếp dựa trên thông tin từ các đoạn văn bản được cung cấp.

Hướng dẫn:
1. Trả lời trực tiếp vào câu hỏi, sử dụng CHÍNH XÁC cách diễn đạt trong tài liệu gốc.
2. KHÔNG thêm các từ mở đầu thừa như "Theo tài liệu:", "Dựa trên thông tin cung cấp:", v.v.
3. Nếu không có thông tin liên quan, hãy nói "Không có thông tin liên quan trong tài liệu."
4. KHÔNG sử dụng kiến thức bên ngoài, chỉ sử dụng thông tin từ các đoạn văn bản được cung cấp.
5. KHÔNG cắt ngắn câu trả lời gốc, đảm bảo trích dẫn ĐẦY ĐỦ nội dung liên quan.
6. Nếu câu hỏi yêu cầu liệt kê các điểm, nguyên tắc, chính sách, quy định, v.v., hãy đảm bảo trích dẫn TOÀN BỘ nội dung liên quan, không bỏ sót bất kỳ điểm nào.
7. Nếu thông tin không đầy đủ, hãy trả lời phần bạn biết và nêu rõ những thông tin còn thiếu.
8. Nếu không có thông tin liên quan, hãy nói "Không có thông tin liên quan trong tài liệu."
9. QUAN TRỌNG: Chỉ sử dụng thông tin từ các đoạn văn bản được cung cấp. KHÔNG sử dụng kiến thức bên ngoài.
10. Đối với các câu hỏi yêu cầu trả lời dài hoặc chi tiết: hãy trả lời ĐẦY ĐỦ và CHI TIẾT, bao gồm tất cả các điểm, ví dụ, giải thích được đề cập trong văn bản.
11. Sử dụng định dạng có cấu trúc (như đánh số, gạch đầu dòng) khi trả lời các câu hỏi yêu cầu liệt kê nhiều điểm.
12. Nếu văn bản có nhiều phần, hãy tổng hợp thông tin từ tất cả các phần liên quan để đưa ra câu trả lời đầy đủ nhất.
13. Luôn ưu tiên sử dụng CHÍNH XÁC cách diễn đạt trong tài liệu gốc, không thêm bớt hay diễn giải lại.
14. Khi câu hỏi là "X là gì?", hãy trả lời trực tiếp "X là..." mà không thêm bất kỳ từ ngữ nào khác ở đầu câu.
15. KHÔNG thêm các từ như "Theo tài liệu", "Dựa trên thông tin", "Theo thông tin cung cấp" vào bất kỳ phần nào của câu trả lời.
16. KHÔNG thêm các kết luận, tóm tắt hoặc ý kiến cá nhân vào cuối câu trả lời.
17. Nếu câu hỏi yêu cầu một định nghĩa, hãy cung cấp định nghĩa chính xác từ tài liệu mà không thêm từ "Định nghĩa:" vào đầu câu trả lời.
18. QUAN TRỌNG: Nếu câu hỏi yêu cầu trả lời "chính xác và đầy đủ", hãy đảm bảo bao gồm TẤT CẢ thông tin liên quan từ tài liệu, không bỏ sót bất kỳ chi tiết nào.
19. Nếu câu hỏi yêu cầu "liệt kê tất cả", hãy đảm bảo liệt kê TOÀN BỘ các mục được đề cập trong tài liệu, không bỏ sót bất kỳ mục nào.
20. Nếu câu hỏi yêu cầu "giải thích chi tiết", hãy cung cấp giải thích ĐẦY ĐỦ và CHI TIẾT nhất có thể dựa trên thông tin trong tài liệu.
21. QUAN TRỌNG: Giữ nguyên định dạng xuống dòng và cách trình bày của văn bản gốc. Nếu văn bản gốc có xuống dòng, hãy giữ nguyên các dòng đó trong câu trả lời.
22. QUAN TRỌNG: Đảm bảo không có lỗi dấu cách trong các từ tiếng Việt. Ví dụ, "k ỹ thuật" phải được viết là "kỹ thuật", "ph ụ tùng" phải được viết là "phụ tùng".
23. ĐẶC BIỆT QUAN TRỌNG: KHÔNG tách các từ tiếng Việt. Ví dụ, viết "gồm" thay vì "g ồm", "rơ moóc" thay vì "r ơ mo óc", "sơ mi rơ moóc" thay vì "s ơ mi r ơ mo óc".
24. TUYỆT ĐỐI QUAN TRỌNG: Giữ nguyên định dạng liệt kê như a), b), c) hoặc (a), (b), (c) trong văn bản gốc. KHÔNG chuyển đổi sang định dạng khác.
25. Đảm bảo các từ tiếng Việt được viết liền mạch, không bị tách ra thành các âm tiết riêng biệt với dấu cách ở giữa.
26. ĐẶC BIỆT QUAN TRỌNG ĐỐI VỚI ĐIỀU LUẬT: Khi câu hỏi liên quan đến điều luật cụ thể (ví dụ: "Điều 33 gồm?", "Điều 5 là gì?"), hãy trả lời đầy đủ toàn bộ nội dung của điều luật đó, bao gồm tất cả các khoản, điểm, mục. KHÔNG rút gọn nội dung.
27. Đối với các câu hỏi rất ngắn gọn (ví dụ: "Heartbeat?", "Điều 33?", "Chức năng X?"), hãy hiểu là người dùng đang hỏi về định nghĩa, nội dung hoặc toàn bộ thông tin liên quan đến khái niệm đó.
    """

    
    user_prompt = f"""Dựa trên các thông tin dưới đây, hãy trả lời câu hỏi một cách đầy đủ và chính xác:

==== THÔNG TIN ====
{context}
==== HẾT THÔNG TIN ====

Câu hỏi: {question}

Lưu ý: 
1. Nếu không tìm thấy thông tin liên quan để trả lời, vui lòng cho biết "Không có thông tin liên quan trong tài liệu."
2. Chỉ sử dụng thông tin từ các đoạn văn bản được cung cấp, KHÔNG sử dụng kiến thức bên ngoài.
3. Trả lời trực tiếp vào câu hỏi, sử dụng CHÍNH XÁC cách diễn đạt trong tài liệu gốc.
4. KHÔNG thêm các từ mở đầu thừa như "Định nghĩa:", "Theo tài liệu:", v.v.
5. Nếu câu hỏi yêu cầu liệt kê các điểm, nguyên tắc, chính sách, quy định, v.v., hãy đảm bảo trích dẫn TOÀN BỘ nội dung liên quan, không bỏ sót bất kỳ điểm nào.
6. Trả lời một cách có cấu trúc, sử dụng đánh số hoặc gạch đầu dòng khi cần thiết để liệt kê các điểm.
7. Nếu câu hỏi yêu cầu giải thích chi tiết hoặc trình bày đầy đủ, hãy đảm bảo câu trả lời của bạn bao gồm tất cả các khía cạnh được đề cập trong văn bản.
8. Khi câu hỏi là "X là gì?", hãy trả lời trực tiếp "X là..." mà không thêm bất kỳ từ ngữ nào khác ở đầu câu.
9. KHÔNG thêm các kết luận, tóm tắt hoặc ý kiến cá nhân vào cuối câu trả lời.
10. QUAN TRỌNG: Nếu câu hỏi yêu cầu trả lời "chính xác và đầy đủ", hãy đảm bảo bao gồm TẤT CẢ thông tin liên quan từ tài liệu, không bỏ sót bất kỳ chi tiết nào.
11. QUAN TRỌNG: Giữ nguyên định dạng gốc của văn bản. Nếu trong tài liệu các bước được liệt kê theo a), b), c) thì câu trả lời phải giữ nguyên định dạng này (không chuyển thành 1., 2., 3., …).
12. QUAN TRỌNG: Nếu nội dung trải dài qua nhiều trang, hãy đảm bảo trích dẫn đầy đủ nội dung từ tất cả các trang liên quan.
13. QUAN TRỌNG: Giữ nguyên định dạng xuống dòng và cách trình bày của văn bản gốc. Nếu văn bản gốc có xuống dòng, hãy giữ nguyên các dòng đó trong câu trả lời.
14. QUAN TRỌNG: Đảm bảo không có lỗi dấu cách trong các từ tiếng Việt. Ví dụ, "k ỹ thuật" phải được viết là "kỹ thuật", "ph ụ tùng" phải được viết là "phụ tùng".
15. ĐẶC BIỆT QUAN TRỌNG: KHÔNG tách các từ tiếng Việt. Ví dụ, viết "gồm" thay vì "g ồm", "rơ moóc" thay vì "r ơ mo óc", "sơ mi rơ moóc" thay vì "s ơ mi r ơ mo óc".
16. TUYỆT ĐỐI QUAN TRỌNG: Giữ nguyên định dạng liệt kê như a), b), c) hoặc (a), (b), (c) trong văn bản gốc. KHÔNG chuyển đổi sang định dạng khác.
17. Đảm bảo các từ tiếng Việt được viết liền mạch, không bị tách ra thành các âm tiết riêng biệt với dấu cách ở giữa.
18. Nếu câu hỏi rất ngắn gọn (như "Điều 33?" hoặc "Điều 33 gồm?"), hiểu rằng người dùng muốn biết toàn bộ nội dung của điều luật đó.
"""
    
    # Log để debug
    logger.info(f"Câu hỏi: {question}")
    logger.info(f"Số chunks tìm thấy: {len(retrieved_chunks)}")
    logger.info(f"Khoảng cách tốt nhất: {best_distance}")
    logger.info(f"Sử dụng mô hình: {model}")
    
    # Gọi Gemini API để sinh câu trả lời
    full_prompt = system_prompt + "\n\n" + user_prompt
    
    # Điều chỉnh cấu hình dựa trên loại câu hỏi
    current_config = gemini_config.copy()
    
    # Nếu là câu hỏi về luật, tăng giới hạn token đầu ra và giảm temperature
    if is_law_question:
        current_config["max_output_tokens"] = 8192  # Tăng giới hạn token đầu ra
        current_config["temperature"] = 0.1  # Giảm temperature để tăng độ chính xác
    # Nếu là câu hỏi yêu cầu trả lời dài
    elif requires_long_answer:
        current_config["max_output_tokens"] = 8192  # Tăng giới hạn token đầu ra
        current_config["temperature"] = 0.2  # Giữ temperature ở mức trung bình
    # Nếu là câu hỏi yêu cầu trả lời chính xác và đầy đủ
    elif requires_exact_answer:
        current_config["max_output_tokens"] = 10240  # Tăng giới hạn token đầu ra tối đa
        current_config["temperature"] = 0.1  # Giảm temperature để tăng độ chính xác
    
    try:
        response = generate_with_retry(full_prompt, current_config)
        # Lấy text từ response
        answer = response
        
        # Đảm bảo giữ nguyên định dạng liệt kê trong câu trả lời
        # Bảo vệ các định dạng liệt kê như a), b), c), (a), (b), (c), v.v.
        answer = re.sub(r'(\s[a-z]\)|\s\([a-z]\))', lambda m: m.group(0), answer)
        
        # Sửa lỗi dấu cách trong từ tiếng Việt
        # Mẫu regex để tìm các từ tiếng Việt bị tách bởi dấu cách
        vietnamese_patterns = [
            # Các nguyên âm bị tách
            (r'([aàáạảãăằắặẳẵâầấậẩẫ])\s+([aàáạảãăằắặẳẵâầấậẩẫ])', r'\1\2'),
            (r'([eèéẹẻẽêềếệểễ])\s+([eèéẹẻẽêềếệểễ])', r'\1\2'),
            (r'([iìíịỉĩ])\s+([iìíịỉĩ])', r'\1\2'),
            (r'([oòóọỏõôồốộổỗơờớợởỡ])\s+([oòóọỏõôồốộổỗơờớợởỡ])', r'\1\2'),
            (r'([uùúụủũưừứựửữ])\s+([uùúụủũưừứựửữ])', r'\1\2'),
            (r'([yỳýỵỷỹ])\s+([yỳýỵỷỹ])', r'\1\2'),
            
            # Phụ âm đầu bị tách với nguyên âm
            (r'([bcdđghklmnpqrstvxBCDĐGHKLMNPQRSTVX])\s+([aàáạảãăằắặẳẵâầấậẩẫeèéẹẻẽêềếệểễiìíịỉĩoòóọỏõôồốộổỗơờớợởỡuùúụủũưừứựửữyỳýỵỷỹ])', r'\1\2'),
            
            # Các trường hợp phổ biến
            (r'g\s+ồ', r'gồ'),
            (r'g\s+ồm', r'gồm'),
            (r'k\s+é', r'ké'),
            (r'k\s+éo', r'kéo'),
            (r'r\s+ơ', r'rơ'),
            (r'r\s+ơ\s+mo', r'rơ mo'),
            (r'r\s+ơ\s+moóc', r'rơ moóc'),
            (r's\s+ơ', r'sơ'),
            (r's\s+ơ\s+mi', r'sơ mi'),
            (r's\s+ơ\s+mi\s+r', r'sơ mi r'),
            (r's\s+ơ\s+mi\s+rơ', r'sơ mi rơ'),
            (r's\s+ơ\s+mi\s+rơ\s+mo', r'sơ mi rơ mo'),
            (r's\s+ơ\s+mi\s+rơ\s+moóc', r'sơ mi rơ moóc'),
            (r'ho\s+ặc', r'hoặc'),
            (r'c\s+ông', r'công'),
            (r'd\s+ụng', r'dụng'),
            (r'kh\s+ối', r'khối'),
            (r'l\s+ượng', r'lượng'),
            (r'b\s+ản', r'bản'),
            (r'th\s+ân', r'thân'),
            (r'ch\s+ở', r'chở'),
            (r'ng\s+ười', r'người'),
            (r'h\s+àng', r'hàng'),
            (r'h\s+óa', r'hóa'),
            (r'g\s+ắn', r'gắn'),
            (r'đ\s+ộng', r'động'),
            (r'c\s+ơ', r'cơ'),
            (r'ch\s+ức', r'chức'),
            (r'n\s+ăng', r'năng'),
            (r'đ\s+ặc', r'đặc'),
            (r'bi\s+ệt', r'biệt'),
            (r'đ\s+ường', r'đường'),
            (r'b\s+ộ', r'bộ'),
            (r'ch\s+ạy', r'chạy'),
            (r'thi\s+ết', r'thiết'),
            (r'k\s+ế', r'kế'),
            (r's\s+ản', r'sản'),
            (r'xu\s+ất', r'xuất'),
            (r'ho\s+ạt', r'hoạt'),
            (r'đ\s+ộng', r'động'),
            (r'r\s+ay', r'ray'),
            (r'd\s+ẫn', r'dẫn'),
            (r'đi\s+ện', r'điện'),
            (r'b\s+ánh', r'bánh'),
            (r'l\s+ớn', r'lớn'),
            (r'h\s+ơn', r'hơn'),
            (r'b\s+ao', r'bao'),
            
            # Sửa lỗi cho từ "rơ moóc" và "sơ mi rơ moóc"
            (r'r\s*ơ\s*mo\s*óc', r'rơ moóc'),
            (r's\s*ơ\s*mi\s*r\s*ơ\s*mo\s*óc', r'sơ mi rơ moóc'),
            
            # Bảo vệ các định dạng liệt kê
            (r'([a-z])\s*\)', r'\1)'),
            (r'\(\s*([a-z])\s*\)', r'(\1)'),
        ]
        
        # Áp dụng các mẫu regex để sửa lỗi dấu cách
        for pattern, replacement in vietnamese_patterns:
            answer = re.sub(pattern, replacement, answer)
            
        # Thêm một bước kiểm tra cuối cùng cho các từ phổ biến
        common_words = {
            "g ồm": "gồm",
            "k éo": "kéo",
            "r ơ": "rơ",
            "s ơ": "sơ",
            "ho ặc": "hoặc",
            "c ông": "công",
            "d ụng": "dụng",
            "kh ối": "khối",
            "l ượng": "lượng",
            "b ản": "bản",
            "th ân": "thân",
            "ch ở": "chở",
            "ng ười": "người",
            "h àng": "hàng",
            "h óa": "hóa",
            "g ắn": "gắn",
            "đ ộng": "động",
            "c ơ": "cơ",
            "ch ức": "chức",
            "n ăng": "năng",
            "đ ặc": "đặc",
            "bi ệt": "biệt",
            "đ ường": "đường",
            "b ộ": "bộ",
            "ch ạy": "chạy",
            "thi ết": "thiết",
            "k ế": "kế",
            "s ản": "sản",
            "xu ất": "xuất",
            "ho ạt": "hoạt",
            "đ ộng": "động",
            "r ay": "ray",
            "d ẫn": "dẫn",
            "đi ện": "điện",
            "b ánh": "bánh",
            "l ớn": "lớn",
            "h ơn": "hơn",
            "b ao": "bao",
            "rơ mo óc": "rơ moóc",
            "sơ mi rơ mo óc": "sơ mi rơ moóc"
        }
        
        for wrong, correct in common_words.items():
            answer = answer.replace(wrong, correct)
            
        # Sử dụng Gemini để sửa lỗi tách từ tiếng Việt
        answer = fix_vietnamese_spacing(answer)
            
    except Exception as e:
        error_message = str(e)
        logger.error(f"Lỗi khi xử lý câu trả lời: {error_message}")
        return {
            "answer": f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {error_message}",
            "sources": [],
            "suggested_questions": []
        }
    
    # Loại bỏ các từ mở đầu thừa nếu có
    common_prefixes = [
        "Định nghĩa: ", "Theo tài liệu: ", "Dựa trên thông tin cung cấp: ",
        "Theo thông tin được cung cấp, ", "Dựa vào tài liệu, ",
        "Theo các đoạn văn bản được cung cấp, ", "Trả lời: ",
        "Dựa trên các đoạn văn bản, ", "Từ thông tin được cung cấp, ",
        "Căn cứ vào tài liệu, ", "Theo nội dung tài liệu, ",
        "Dựa vào thông tin trong tài liệu, ", "Theo đoạn văn, ",
        "Theo thông tin trong tài liệu, ", "Dựa vào các đoạn văn, ",
        "Theo các tài liệu, ", "Dựa trên tài liệu, ",
        "Căn cứ theo tài liệu, ", "Theo đoạn trích, ",
        "Dựa trên đoạn văn, ", "Theo văn bản, "
    ]
    
    for prefix in common_prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
            break
    
    # Loại bỏ các kết luận hoặc tóm tắt thừa ở cuối câu trả lời
    common_suffixes = [
        "\n\nTóm lại,", "\n\nNhư vậy,", "\n\nKết luận,", "\n\nTổng kết,",
        "\n\nTóm tắt,", "\n\nNói tóm lại,", "\n\nNói chung,", "\n\nNhìn chung,",
        "\n\nTổng cộng,", "\n\nTổng quát,", "\n\nTổng thể,", "\n\nTóm tắt lại,",
        "\nTóm lại,", "\nNhư vậy,", "\nKết luận,", "\nTổng kết,",
        "\nTóm tắt,", "\nNói tóm lại,", "\nNói chung,", "\nNhìn chung,",
        "\nTổng cộng,", "\nTổng quát,", "\nTổng thể,", "\nTóm tắt lại,"
    ]
    
    for suffix in common_suffixes:
        if suffix in answer:
            answer = answer.split(suffix)[0]
            break
    
    # Đảm bảo giữ nguyên định dạng liệt kê trong câu trả lời
    # Bảo vệ các định dạng liệt kê như a), b), c), (a), (b), (c), v.v.
    answer = re.sub(r'(\s[a-z]\)|\s\([a-z]\))', lambda m: m.group(0), answer)
    
    # Đảm bảo nội dung trải dài qua nhiều trang được xử lý đúng
    # Loại bỏ các dấu hiệu trang không cần thiết trong câu trả lời cuối cùng
    # answer = re.sub(r'\[Trang \d+\]\s*', '', answer)
    
    # Thay vì xóa hoàn toàn, ta chỉ làm gọn các thông tin trang liên tiếp
    answer = re.sub(r'\[Trang \d+\]\s*\[Trang \d+\]', lambda m: m.group(0).split(']')[0] + ']', answer)
    
    # Thêm thông tin nguồn tài liệu
    if used_files:
        # Phân tích câu trả lời để xác định nguồn thật sự được sử dụng
        real_sources = []
        source_scores = {}  # Lưu điểm số cho mỗi nguồn
        
        # Nếu tìm thấy thông tin liên quan
        if "Không có thông tin liên quan trong tài liệu." not in answer:
            # Chuẩn bị câu trả lời để so khớp
            clean_answer = re.sub(r'\n+', ' ', answer)
            clean_answer = re.sub(r'\s+', ' ', clean_answer)
            
            # Chia câu trả lời thành các câu
            answer_sentences = re.split(r'(?<=[.!?])\s+', clean_answer)
            
            # Phân tích context để tìm nguồn thực sự được sử dụng
            chunks_and_sources = {}
            
            for idx, chunk_meta in enumerate(retrieved_metadata):
                if idx >= len(retrieved_chunks):
                    continue
                    
                chunk_text = retrieved_chunks[idx]
                file_name = chunk_meta.get("filename", "Unknown")
                
                # Chuẩn bị chunk để so khớp - Giữ lại thông tin trang
                # Chỉ loại bỏ các thông tin không liên quan đến nội dung
                clean_chunk = re.sub(r'\[Granularity: [^\]]+\]\s*', '', chunk_text)
                clean_chunk = re.sub(r'\n+', ' ', clean_chunk)
                clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
                
                if file_name not in chunks_and_sources:
                    chunks_and_sources[file_name] = []
                chunks_and_sources[file_name].append(clean_chunk)
                
                # Khởi tạo điểm số cho nguồn nếu chưa có
                if file_name not in source_scores:
                    source_scores[file_name] = 0
            
            # Cải thiện thuật toán so khớp nguồn
            for file_name, chunks in chunks_and_sources.items():
                file_score = 0  # Điểm số cho mỗi file
                content_matched = False  # Đánh dấu xem có nội dung khớp quan trọng không
                exact_sentences_found = 0  # Số câu khớp hoàn toàn
                
                for chunk in chunks:
                    # Chia chunk thành các câu
                    chunk_sentences = re.split(r'(?<=[.!?])\s+', chunk)
                    
                    # 1: Tìm kiếm câu hoàn chỉnh - Ưu tiên cao nhất
                    for sentence in answer_sentences:
                        # Chỉ kiểm tra các câu có ý nghĩa
                        if len(sentence) < 15:  # Bỏ qua câu quá ngắn
                            continue
                            
                        # Loại bỏ các thông tin về trang khi so khớp nội dung
                        cleaned_sentence = re.sub(r'\[Trang \d+\]\s*', '', sentence)
                        
                        # Kiểm tra câu trong chunk
                        # Sử dụng full_match để đánh dấu trường hợp khớp chính xác
                        full_match = False
                        
                        # Tạo phiên bản cleaned của chunk để so sánh chính xác nội dung
                        chunk_content_only = re.sub(r'\[Trang \d+\]\s*', '', chunk)
                        
                        if cleaned_sentence in chunk_content_only:
                            file_score += 25  # Tăng điểm cho câu khớp hoàn toàn
                            exact_sentences_found += 1
                            content_matched = True
                            full_match = True
                        
                        # Nếu không tìm thấy khớp hoàn toàn, kiểm tra khớp một phần
                        if not full_match and len(cleaned_sentence) > 30:
                            # Phân tích từng từ trong câu
                            words = re.findall(r'\b\w+\b', cleaned_sentence.lower())
                            chunk_lower = chunk_content_only.lower()
                            
                            # Đếm số từ khớp
                            matched_words = sum(1 for word in words if word in chunk_lower)
                            
                            # Tính tỷ lệ khớp
                            if len(words) > 0:
                                match_ratio = matched_words / len(words)
                                
                                # Nếu tỷ lệ khớp cao
                                if match_ratio > 0.85:  # Tăng ngưỡng từ 0.8 lên 0.85
                                    file_score += 20  # Tăng điểm từ 15 lên 20
                                    content_matched = True
                                elif match_ratio > 0.7:  # Tăng ngưỡng từ 0.6 lên 0.7
                                    file_score += 12  # Tăng điểm từ 8 lên 12
                    
                    # 2: Tìm kiếm cụm từ có ý nghĩa - Giúp tìm ra các trích dẫn một phần
                    if not content_matched:  # Chỉ thực hiện nếu chưa tìm thấy khớp hoàn chỉnh
                        meaningful_phrases_found = 0
                        
                        for i in range(len(chunk_sentences)):
                            for j in range(i, min(i + 3, len(chunk_sentences))):
                                phrase = ' '.join(chunk_sentences[i:j+1])
                                if len(phrase) < 30:  # Bỏ qua cụm quá ngắn
                                    continue
                                
                                # Kiểm tra cụm từ trong câu trả lời
                                if phrase in clean_answer:
                                    file_score += 10  # Điểm cho cụm khớp
                                    meaningful_phrases_found += 1
                                    content_matched = True
                                    
                                    # Giới hạn số điểm từ phương pháp này
                                    if meaningful_phrases_found >= 3:
                                        break
                    
                    # 3: Tìm kiếm từ khóa đặc biệt - Dùng cho các định nghĩa, thuật ngữ
                    # Chỉ áp dụng nếu chưa tìm thấy khớp hoàn chỉnh hoặc cụm từ
                    if not content_matched:
                        special_terms_found = 0
                        
                        # Tìm các thuật ngữ đặc biệt 
                        special_terms = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|"[^"]+"|\'[^\']+\')', chunk)
                        special_terms += re.findall(r'(\b[A-Za-z]+\b\s+là\s+[^.!?]+)', chunk)  # Mẫu định nghĩa "X là Y"
                        
                        for term in special_terms:
                            if len(term) < 5:  # Bỏ qua thuật ngữ quá ngắn
                                continue
                                
                            if term.lower() in clean_answer.lower():
                                file_score += 5  # Điểm cho thuật ngữ khớp
                                special_terms_found += 1
                                
                                # Giới hạn số điểm từ phương pháp này
                                if special_terms_found >= 5:
                                    break
                
                # Bổ sung: Kiểm tra xem có câu trả lời chính xác nằm trong file không
                # Rất hữu ích cho các câu trả lời ngắn như định nghĩa
                if len(answer_sentences) <= 3 and len(answer_sentences) > 0:
                    main_sentence = answer_sentences[0]  # Câu đầu tiên thường là câu trả lời chính
                    
                    for chunk in chunks:
                        # Nếu câu trả lời chính nằm hoàn toàn trong chunk
                        if main_sentence in chunk and len(main_sentence) > 20:
                            file_score += 50  # Điểm rất cao cho việc tìm thấy câu trả lời chính xác
                            content_matched = True
                
                # Điều chỉnh điểm dựa trên số lượng câu khớp hoàn toàn
                if exact_sentences_found > 0:
                    # Tỷ lệ câu khớp hoàn toàn so với tổng số câu trong câu trả lời
                    exact_ratio = exact_sentences_found / len(answer_sentences)
                    
                    # Nếu hầu hết các câu đều từ file này
                    if exact_ratio > 0.7:  # Trên 70% câu từ file này
                        file_score *= 1.5  # Tăng 50% điểm
                    elif exact_ratio > 0.4:  # Trên 40% câu từ file này
                        file_score *= 1.2  # Tăng 20% điểm
                
                # Lưu điểm số cho nguồn
                source_scores[file_name] = file_score
                
                # Nếu đạt ngưỡng điểm và có nội dung khớp, thêm vào danh sách nguồn thật sự
                if file_score >= 10 and content_matched and file_name not in real_sources:
                    real_sources.append(file_name)
            
            # Loại bỏ các nguồn có điểm thấp nếu có nguồn điểm cao
            if real_sources:
                # Tìm điểm cao nhất
                max_score = max(source_scores.get(source, 0) for source in real_sources)
                
                # Chỉ giữ lại các nguồn có điểm đủ cao (ít nhất 40% điểm cao nhất)
                filtered_sources = [source for source in real_sources 
                                   if source_scores.get(source, 0) >= max_score * 0.4]
                
                # Nếu lọc còn ít nhất một nguồn, sử dụng danh sách đã lọc
                if filtered_sources:
                    real_sources = filtered_sources
            
            # Sắp xếp nguồn theo điểm số từ cao xuống thấp
            sorted_sources = sorted([(s, source_scores.get(s, 0)) for s in real_sources], 
                                   key=lambda x: x[1], reverse=True)
            
            # Debug log
            logger.info(f"Điểm số nguồn: {sorted_sources}")
            
            # Bổ sung: Kiểm tra mức độ trùng lặp chi tiết hơn cho các nguồn có điểm thấp
            for file_name, file_score in list(source_scores.items()):
                # Nếu file đã được xác định là nguồn thực sự và có điểm cao, bỏ qua
                if file_name in real_sources and file_score > 50:
                    continue
                    
                if file_name in chunks_and_sources:
                    # Kiểm tra xem có bao nhiêu phần trăm câu trả lời xuất hiện trong tài liệu
                    all_content = ' '.join(chunks_and_sources[file_name])
                    
                    # Loại bỏ thông tin trang khi so sánh nội dung
                    clean_answer_for_compare = re.sub(r'\[Trang \d+\]\s*', '', clean_answer)
                    clean_all_content = re.sub(r'\[Trang \d+\]\s*', '', all_content)
                    
                    # Kiểm tra từng đoạn văn lớn trong câu trả lời
                    paragraphs = re.split(r'\n{2,}', clean_answer_for_compare)
                    matched_paragraphs = 0
                    
                    for para in paragraphs:
                        if len(para) < 40:  # Bỏ qua đoạn quá ngắn
                            continue
                            
                        if para in clean_all_content:
                            matched_paragraphs += 1
                            # Tăng điểm dựa trên số đoạn văn khớp hoàn toàn
                            source_scores[file_name] += len(para) / 10  # Điểm tỷ lệ với độ dài đoạn khớp
                    
                    # Nếu có nhiều đoạn văn khớp, tăng điểm mạnh
                    if matched_paragraphs > 1:
                        source_scores[file_name] += matched_paragraphs * 15
                        if file_name not in real_sources:
                            real_sources.append(file_name)
            
            # Cập nhật lại danh sách nguồn đã sắp xếp sau khi đánh giá lại
            sorted_sources = sorted([(s, source_scores.get(s, 0)) for s in real_sources], 
                                   key=lambda x: x[1], reverse=True)
            
            # Chỉ giữ lại các nguồn có điểm số
            real_sources = [s for s, score in sorted_sources if score > 0]
            
            # Nếu không tìm thấy nguồn thật sự, sử dụng tất cả các nguồn đã truy xuất
            if not real_sources:
                real_sources = used_files
        
        # Hiển thị thông tin nguồn
        unique_files = list(set(real_sources))
        
        # Phân tích sâu hơn để tìm đúng vị trí của thông tin
        # Sử dụng hàm identify_most_relevant_pages để xác định chính xác các trang chứa câu trả lời
        relevant_pages_dict = identify_most_relevant_pages(answer, retrieved_chunks, retrieved_metadata)
        
        # Chuẩn bị để tạo thông tin trích dẫn chi tiết
        citation_info = {}
        
        # Nếu tìm được trang có liên quan
        if relevant_pages_dict:
            for file_name, pages in relevant_pages_dict.items():
                if file_name in unique_files:
                    # Chuyển từ set sang list và sắp xếp
                    sorted_pages = sorted([int(p) for p in pages])
                    
                    # Tìm các section, heading hoặc thông tin vị trí từ metadata
                    section_info = []
                    
                    # Thu thập thông tin về section từ metadata
                    for meta in retrieved_metadata:
                        if meta.get("filename") == file_name and str(meta.get("primary_page")) in pages:
                            # Kiểm tra các thông tin cụ thể
                            article_number = meta.get("article_number")
                            
                            if article_number:
                                section_info.append(f"Điều {article_number}")
                    
                    # Tạo chuỗi thông tin trích dẫn
                    citation = ""
                    
                    # Thêm thông tin về số trang - chỉ lấy trang đầu tiên
                    if sorted_pages:
                        citation += f"trang {sorted_pages[0]}"
                    
                    # Thêm thông tin về section nếu có
                    if section_info:
                        unique_sections = list(set(section_info))
                        if citation:
                            citation += ", "
                        citation += ", ".join(unique_sections)
                    
                    # Lưu thông tin trích dẫn cho file này
                    citation_info[file_name] = citation
        else:
            # Nếu không tìm được trang cụ thể, sử dụng phương pháp cũ
            source_with_pages = {}
            
            for idx, chunk_meta in enumerate(retrieved_metadata):
                if idx >= len(retrieved_chunks):
                    continue
                        
                file_name = chunk_meta.get("filename", "Unknown")
                if file_name in unique_files:
                    # Thu thập thông tin trang
                    primary_page = chunk_meta.get("primary_page")
                    pages = chunk_meta.get("pages", [])
                    
                    if not pages:
                        # Nếu không có trong metadata, trích xuất từ text
                        chunk_text = retrieved_chunks[idx]
                        page_info = re.findall(r'\[Trang (\d+)\]', chunk_text)
                        pages = [int(p) for p in page_info]
                    
                    if file_name not in source_with_pages:
                        source_with_pages[file_name] = set()
                    
                    # Thêm số trang vào tập hợp
                    for page in pages:
                        source_with_pages[file_name].add(str(page))
            
            # Tạo thông tin trích dẫn đơn giản - chỉ lấy trang đầu tiên
            for file_name, pages in source_with_pages.items():
                if pages:
                    sorted_pages = sorted([int(p) for p in pages])
                    citation_info[file_name] = f"trang {sorted_pages[0]}"
                else:
                    citation_info[file_name] = ""
        
        # Tạo chuỗi nguồn với thông tin trích dẫn đầy đủ
        source_strings = []
        for file_name in unique_files:
            citation = citation_info.get(file_name, "")
            if citation:
                # Thêm một dấu hiệu để đánh dấu nguồn chứa câu trả lời (ví dụ: dấu *)
                extension = file_name.split('.')[-1].lower() if '.' in file_name else ""
                
                # Đặt dấu * cho nguồn chính xác
                if extension in ['pdf', 'docx', 'txt'] and citation.startswith("trang"):
                    # Đây là nguồn chính, thêm dấu *
                    source_strings.append(f"{file_name} ({citation})*")
                else:
                    source_strings.append(f"{file_name} ({citation})")
            else:
                source_strings.append(file_name)
        
        # Sắp xếp để nguồn chính xác (có dấu *) lên đầu
        source_strings.sort(key=lambda s: 0 if s.endswith('*') else 1)
        
        # Loại bỏ dấu * trước khi hiển thị
        formatted_sources = [s[:-1] if s.endswith('*') else s for s in source_strings]
        
        sources_info = f"\n\nNguồn tài liệu: {', '.join(formatted_sources)}"
        answer += sources_info
    else:
        # Nếu không tìm thấy nguồn thật sự, vẫn hiển thị tất cả các nguồn đã truy xuất
        sources_info = f"\n\nNguồn tài liệu: {', '.join(used_files)}"
        answer += sources_info
    
    # Kết thúc đo thời gian trả lời
    end_answer = time.time()
    answer_time = end_answer - start_answer
    
    # Ghi log thời gian xử lý
    logger.info(f"Thời gian truy xuất: {retrieval_time:.2f}s, Thời gian trả lời: {answer_time:.2f}s")
    
    # Theo dõi hiệu suất
    track_performance(question, retrieval_time, answer_time, len(retrieved_chunks), chunking_method)
    
    # Tạo câu hỏi gợi ý cho cả trường hợp tìm thấy câu trả lời
    suggested_questions = []
    if "Không có thông tin liên quan trong tài liệu." in answer:
        # Tạo câu hỏi gợi ý khi không tìm thấy thông tin
        suggested_questions = generate_similar_questions(question, global_metadata)
    
    # Trả về kết quả
    return {
        "answer": answer,
        "sources": used_files,
        "suggested_questions": suggested_questions
    }

def generate_similar_questions(question, metadata):
    """
    Tạo các câu hỏi gợi ý dựa trên câu hỏi gốc và metadata của tài liệu
    
    Args:
        question (str): Câu hỏi gốc
        metadata (list): Metadata của các chunks trong tài liệu
        
    Returns:
        list: Danh sách các câu hỏi gợi ý
    """
    try:
        # Tạo prompt để sinh câu hỏi gợi ý tương tự với câu hỏi gốc
        prompt = f"""Tạo 3 câu hỏi có ý nghĩa tương tự với câu hỏi gốc: "{question}"

                Yêu cầu:
                1. Các câu hỏi được tạo ra phải gần nghĩa nhất, tương tự nhất và liên quan nhất đối với câu hỏi gốc
                2. Chỉ liệt kê 3 câu hỏi theo định dạng số thứ tự
                3. Không thêm bất kỳ giải thích, phân tích hay nhận xét nào
                4. Không đề cập đến việc phân tích lĩnh vực hay chuyên môn
                5. Không có lời mở đầu hay kết luận
                6. Câu hỏi gốc nếu có các từ khóa tiếng anh thì giữ nguyên tiếng anh không được tự ý dịch lại tiếng việt

                Trả lời theo định dạng chính xác sau:
                1. [Câu hỏi 1]
                2. [Câu hỏi 2]
                3. [Câu hỏi 3]"""
        
        # Sử dụng hàm generate_with_retry đã có
        response = generate_with_retry(prompt, {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        })
        
        # Tách các câu hỏi thành danh sách
        suggested_questions = []
        for line in response.strip().split('\n'):
            # Loại bỏ số thứ tự và khoảng trắng
            clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
            if clean_line:
                suggested_questions.append(clean_line)
        
        # Đảm bảo có đúng 3 câu hỏi
        if len(suggested_questions) < 3:
            # Thêm câu hỏi mặc định nếu thiếu
            default_questions = [
                f"Bạn có thể cung cấp thêm thông tin về {question}?",
                f"Tài liệu có đề cập đến {question} không?",
                f"Có thông tin nào liên quan đến {question} trong tài liệu không?"
            ]
            suggested_questions.extend(default_questions[:(3 - len(suggested_questions))])
        
        # Giới hạn số lượng câu hỏi là 3
        return suggested_questions[:3]
    except Exception as e:
        logger.error(f"Lỗi khi tạo câu hỏi gợi ý: {str(e)}")
        # Trả về câu hỏi mặc định nếu có lỗi
        return [
            f"Bạn có thể cung cấp thêm thông tin về {question}?",
            f"Tài liệu có đề cập đến {question} không?",
            f"Có thông tin nào liên quan đến {question} trong tài liệu không?"
        ]

# --- API routes cho hiệu suất và đánh giá ---
@app.route('/api/performance', methods=['GET'])
def get_performance():
    """API endpoint để lấy thông tin hiệu suất"""
    # Yêu cầu đăng nhập trước khi truy cập API
    from supabase_modules.auth import verify_session
    if not verify_session():
        return jsonify({"error": "Unauthorized"}), 401
    
    # Nạp trạng thái hệ thống của người dùng đã đăng nhập
    load_state()
    
    chunking_method = request.args.get('chunking_method')
    analysis = analyze_performance(chunking_method)
    return jsonify(analysis)

@app.route('/api/embeddings', methods=['GET'])
def view_embeddings():
    """Endpoint để xem thông tin về embeddings"""
    # Yêu cầu đăng nhập trước khi truy cập API
    from supabase_modules.auth import verify_session
    if not verify_session():
        return jsonify({"error": "Unauthorized"}), 401
    
    # Nạp trạng thái hệ thống của người dùng đã đăng nhập
    load_state()
    
    try:
        response = {
            'total_documents': len(global_all_files),
            'total_chunks': len(global_metadata),
            'documents': {},
            'embeddings_stats': {}
        }
        
        # Thông tin về từng document
        for filename in global_all_files:
            doc_chunks = [meta for meta in global_metadata if meta.get('source') == filename]
            response['documents'][filename] = {
                'total_chunks': len(doc_chunks),
                'chunking_method': doc_chunks[0].get('chunking_method') if doc_chunks else 'unknown',
                'last_modified': doc_chunks[0].get('timestamp') if doc_chunks else None
            }
        
        # Thống kê về embeddings
        if global_vector_list:
            vectors = np.vstack(global_vector_list)
            response['embeddings_stats'] = {
                'total_vectors': len(global_vector_list),
                'vector_dimension': vectors.shape[1],
                'memory_usage_mb': vectors.nbytes / (1024 * 1024),
                'index_type': 'FAISS' if faiss_index is not None else 'None'
            }
            
            if faiss_index is not None:
                response['embeddings_stats']['faiss_total_vectors'] = faiss_index.ntotal
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_system():
    """API endpoint để đánh giá hệ thống với một tập câu hỏi"""
    # Yêu cầu đăng nhập trước khi truy cập API
    from supabase_modules.auth import verify_session
    if not verify_session():
        return jsonify({"error": "Unauthorized"}), 401
    
    # Nạp trạng thái hệ thống của người dùng đã đăng nhập
    load_state()
    
    data = request.json
    if not data or 'queries' not in data:
        return jsonify({"error": "Cần cung cấp danh sách câu hỏi"}), 400
    
    queries = data.get('queries', [])
    top_k = data.get('top_k', 5)
    chunking_method = data.get('chunking_method')
    
    if not queries:
        return jsonify({"error": "Danh sách câu hỏi không được để trống"}), 400
    
    results = evaluate_performance(queries, top_k, chunking_method)
    return jsonify(results)

@app.route('/api/chunking_methods', methods=['GET'])
def get_chunking_methods():
    """Trả về danh sách các phương pháp chunking được hỗ trợ"""
    # Yêu cầu đăng nhập trước khi truy cập API
    from supabase_modules.auth import verify_session
    if not verify_session():
        return jsonify({"error": "Unauthorized"}), 401
        
    methods = {
        "sentence_windows": "Cửa sổ câu - Chia văn bản thành các cửa sổ trượt của các câu liên tiếp",
        "paragraph": "Đoạn văn - Chia văn bản thành các đoạn văn riêng biệt",
        "semantic": "Ngữ nghĩa - Chia văn bản dựa trên ranh giới ngữ nghĩa",
        "token": "Token - Chia văn bản thành các đoạn có số lượng token cố định",
        "hybrid": "Kết hợp - Kết hợp nhiều phương pháp chunking khác nhau",
        "adaptive": "Thích ứng - Tự động điều chỉnh kích thước chunk dựa trên nội dung",
        "hierarchical": "Phân cấp - Tạo chunks theo cấu trúc phân cấp của văn bản",
        "contextual": "Ngữ cảnh - Tạo chunks dựa trên ngữ cảnh và mối quan hệ ngữ nghĩa",
        "multi_granularity": "Multi-granularity - Tạo chunks với nhiều mức độ chi tiết khác nhau (tài liệu, phần, đoạn văn, câu) để tối ưu cho nhiều loại câu hỏi"
    }
    
    return jsonify(methods)

@app.route('/api/answer', methods=['POST'])
def api_answer_question():
    """API endpoint để trả lời câu hỏi"""
    # Yêu cầu đăng nhập trước khi truy cập API
    from supabase_modules.auth import verify_session
    if not verify_session():
        return jsonify({"error": "Unauthorized"}), 401
    
    # Nạp trạng thái hệ thống của người dùng đã đăng nhập
    load_state()
    
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Cần cung cấp câu hỏi"}), 400
    
    question = data.get('question')
    top_k = data.get('top_k', 10)
    threshold = data.get('threshold', 5.0)
    model = data.get('model', "gemini-2.0-flash")
    chunking_method = data.get('chunking_method')
    
    # Phát hiện câu hỏi về điều khoản luật
    is_law_question = False
    law_article_patterns = [
        r'điều\s+\d+', r'khoản\s+\d+', r'điểm\s+\d+',
        r'chính sách', r'nguyên tắc', r'quy định', r'luật'
    ]
    
    for pattern in law_article_patterns:
        if re.search(pattern, question.lower()):
            is_law_question = True
            logger.info(f"API: Phát hiện câu hỏi về điều khoản luật: {question}")
            # Tăng top_k cho câu hỏi về luật
            top_k = max(top_k, 30)
            break
    
    try:
        result = answer_question(question, top_k, threshold, model, chunking_method)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "suggested_questions": result["suggested_questions"]
        })
    except Exception as e:
        error_message = str(e)
        logger.error(f"API: Lỗi khi trả lời câu hỏi: {error_message}")
        
        # Xử lý lỗi token length
        if "Token indices sequence length is longer than the specified maximum sequence length" in error_message:
            # Giảm top_k và thử lại
            reduced_top_k = max(3, top_k // 3)
            logger.info(f"API: Giảm top_k từ {top_k} xuống {reduced_top_k} và thử lại")
            
            try:
                result = answer_question(question, reduced_top_k, threshold, model, chunking_method)
                return jsonify({
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "suggested_questions": result["suggested_questions"],
                    "warning": f"Đã giảm số lượng đoạn văn bản từ {top_k} xuống {reduced_top_k} do giới hạn token"
                })
            except Exception as inner_e:
                return jsonify({
                    "error": f"Lỗi khi trả lời câu hỏi (sau khi giảm top_k): {str(inner_e)}"
                }), 500
        
        return jsonify({
            "error": f"Lỗi khi trả lời câu hỏi: {error_message}"
        }), 500

# --- Hàm tải cài đặt ---
def load_settings():
    """Tải cài đặt từ file hoặc sử dụng giá trị mặc định"""
    default_settings = {
        "sentence_windows": {
            "window_size": 3,
            "step": 1
        },
        "paragraph": {
            "max_chars": 1000,
            "overlap": 200
        },
        "semantic": {
            "min_words": 50,
            "max_words": 200
        },
        "token": {
            "max_tokens": 500,
            "overlap_tokens": 50
        },
        "faiss_index_type": "flat"
    }
    
    try:
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as f:
                settings = json.load(f)
                
                # Đảm bảo tất cả các cài đặt mặc định đều có
                for key, value in default_settings.items():
                    if key not in settings:
                        settings[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in settings[key]:
                                settings[key][sub_key] = sub_value
                
                return settings
    except Exception as e:
        logger.error(f"Lỗi khi tải cài đặt: {str(e)}")
    
    return default_settings

# Route cho trang chủ
@app.route('/')
def index():
    """
    Trang chủ của ứng dụng RAG
    
    Kiểm tra session người dùng và hiển thị trang chính
    """
    # Yêu cầu đăng nhập trước khi truy cập hệ thống
    from supabase_modules.auth import verify_session
    if not verify_session():
        return redirect(url_for('login'))
    
    # Nạp danh sách tài liệu của người dùng đã đăng nhập
    from supabase_modules.helpers import get_user_id_from_session, get_user_files_with_metadata
    user_id = get_user_id_from_session()
    
    # Nạp trạng thái hệ thống của người dùng đã đăng nhập
    global global_all_files
    load_state()  # Đảm bảo nạp dữ liệu của người dùng hiện tại
    
    return render_template_string(
        index_html, 
        files=global_all_files,
        settings=load_settings(),
        answer=None,
        sources=None
    )

# Route cho truy vấn
@app.route('/query', methods=['POST'])
def query():
    """
    Xử lý câu hỏi từ người dùng
    """
    # Yêu cầu đăng nhập trước khi truy cập hệ thống
    from supabase_modules.auth import verify_session
    if not verify_session():
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': 'Vui lòng đăng nhập để sử dụng hệ thống'
            }), 401
        return redirect(url_for('login'))
    
    # Nạp trạng thái hệ thống của người dùng đã đăng nhập
    load_state()  # Đảm bảo nạp dữ liệu của người dùng hiện tại
    
    # Tiếp tục với logic xử lý câu hỏi
    question = request.form.get('question')
    top_k = int(request.form.get('top_k', 20))
    threshold = float(request.form.get('threshold', 5.0))
    model = request.form.get('model', 'gemini-2.0-flash')
    chunking_method = request.form.get('chunking_method', None)
    
    # Kiểm tra xem có phải là AJAX request không
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    # Log thông tin request để debug
    logger.info(f"Query request - Question: '{question}', AJAX: {is_ajax}")
    logger.info(f"Form data: {dict(request.form)}")
    
    if not question:
        logger.warning("Empty question received")
        if is_ajax:
            logger.info("Returning AJAX empty question response")
            return render_template_string(
                """<div class="chat-message bot-message"><p class="text-gray-800 dark:text-gray-200 text-sm sm:text-base">Vui lòng nhập câu hỏi.</p></div>"""
            )
        else:
            return render_template_string(
                index_html,
                files=global_all_files,
                settings=load_settings(),
                answer="Vui lòng nhập câu hỏi.",
                sources=[]
            )
    
    try:
        # Đảm bảo question không phải None và là chuỗi trước khi xử lý
        if question is None:
            question = ""
        
        result = answer_question(question, top_k, threshold, model)
        answer = result["answer"] if result.get("answer") is not None else ""
        sources = result.get("sources", [])
        suggested_questions = result.get("suggested_questions", [])
        
        # Đảm bảo luôn có câu hỏi gợi ý khi không tìm thấy thông tin
        if answer and "Không có thông tin liên quan trong tài liệu." in answer and not suggested_questions:
            # Tạo câu hỏi gợi ý từ Gemini nếu không có
            suggested_questions = generate_similar_questions(question, global_metadata)
            logger.info(f"Tạo câu hỏi gợi ý từ Gemini: {suggested_questions}")
        
        # Nếu là AJAX request, chỉ trả về phần tin nhắn bot
        if is_ajax:
            logger.info(f"Returning AJAX response with answer length: {len(answer)}")
            
            # Tạo HTML cho câu hỏi gợi ý nếu có
            suggested_questions_html = ""
            if "Không có thông tin liên quan trong tài liệu." in answer:
                # Kiểm tra xem suggested_questions có phải là dict không (trường hợp nhiều câu hỏi)
                if isinstance(suggested_questions, dict):
                    # Xử lý trường hợp nhiều câu hỏi - chỉ hiển thị gợi ý cho câu hỏi không có thông tin
                    for q, suggestions in suggested_questions.items():
                        if suggestions:
                            suggested_questions_html += f"""
                            <div class="suggested-questions">
                                <p class="suggested-questions-title">Gợi ý cho câu hỏi "{q}":</p>
                                <div class="space-y-2">
                            """
                            for suggestion in suggestions:
                                # Escape dấu nháy đơn trong câu hỏi để tránh lỗi JavaScript
                                escaped_suggestion = suggestion.replace("'", "\\'")
                                suggested_questions_html += f"""
                                    <button onclick="submitQuestion('{escaped_suggestion}')" 
                                            class="suggested-question-btn">
                                        {suggestion}
                                    </button>
                                """
                            suggested_questions_html += """
                                </div>
                            </div>
                            """
                # Trường hợp một câu hỏi đơn
                elif suggested_questions:
                    suggested_questions_html = """
                    <div class="suggested-questions">
                        <p class="suggested-questions-title">Bạn có thể quan tâm đến các câu hỏi sau:</p>
                        <div class="space-y-2">
                    """
                    for question in suggested_questions:
                        # Escape dấu nháy đơn trong câu hỏi để tránh lỗi JavaScript
                        escaped_question = question.replace("'", "\\'")
                        suggested_questions_html += f"""
                            <button onclick="submitQuestion('{escaped_question}')" 
                                    class="suggested-question-btn">
                                {question}
                            </button>
                        """
                    suggested_questions_html += """
                        </div>
                    </div>
                    """
            
            return render_template_string(
                """<div class="chat-message bot-message"><div class="text-gray-800 dark:text-gray-200 whitespace-pre-wrap text-sm sm:text-base" style="display: inline;">{{ answer }}</div>{{ suggested_questions_html | safe }}</div>""", 
                answer=answer,
                suggested_questions_html=suggested_questions_html
            )
        else:
            # Nếu không phải AJAX, trả về toàn bộ trang
            return render_template_string(
                index_html,
                files=global_all_files,
                settings=load_settings(),
                answer=answer,
                sources=sources,
                question=question,
                suggested_questions=suggested_questions if "Không có thông tin liên quan trong tài liệu." in answer else []
            )
            
    except Exception as e:
        error_message = str(e)
        logger.error(f"Lỗi khi trả lời câu hỏi: {error_message}")
        
        if is_ajax:
            logger.info("Returning AJAX error response")
            return render_template_string(
                """<div class="chat-message bot-message"><p class="text-red-500 dark:text-red-400 text-sm sm:text-base">Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {{ error }}</p></div>""", error=error_message
            )
        else:
            return render_template_string(
                index_html,
                files=global_all_files,
                settings=load_settings(),
                answer=f"Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {error_message}",
                sources=[],
                suggested_questions=[]
            )

# Route cho upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Xử lý tải lên tài liệu
    """
    # Sử dụng phiên bản nâng cao của hàm upload_file để hỗ trợ xác thực và phân chia theo người dùng
    from supabase_integration import enhanced_upload_file
    return enhanced_upload_file(
        upload_folder, 
            index_html,
        global_all_files, 
        load_settings, 
        extract_text_pdf, 
        extract_text_docx, 
        extract_text_txt, 
        add_document, 
        save_state
        )

# Route cho xóa file
@app.route('/remove', methods=['POST'])
def remove_file():
    """
    Xử lý xóa tài liệu
    """
    # Sử dụng phiên bản nâng cao của hàm remove_file để hỗ trợ xác thực và phân chia theo người dùng
    from supabase_integration import enhanced_remove_file
    return enhanced_remove_file(index_html, global_all_files, load_settings, remove_document)

# Route cho lưu cài đặt
@app.route('/settings', methods=['POST'])
def save_settings():
    """
    Xử lý lưu cài đặt hệ thống
    """
    # Yêu cầu đăng nhập trước khi thay đổi cài đặt
    from supabase_modules.auth import verify_session
    if not verify_session():
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"error": "Unauthorized", "message": "Vui lòng đăng nhập để thay đổi cài đặt"}), 401
        else:
            flash("Vui lòng đăng nhập để thay đổi cài đặt", "error")
            return redirect(url_for('login'))
    
    # Nạp trạng thái hệ thống của người dùng đã đăng nhập
    load_state()
    
    # Lấy các cài đặt từ form
    settings = {
        "sentence_windows": {
            "window_size": int(request.form.get('window_size', 3)),
            "step": int(request.form.get('step', 1))
        },
        "paragraph": {
            "max_chars": int(request.form.get('max_chars', 1000)),
            "overlap": int(request.form.get('overlap', 200))
        },
        "semantic": {
            "min_words": int(request.form.get('min_words', 50)),
            "max_words": int(request.form.get('max_words', 200))
        },
        "token": {
            "max_tokens": int(request.form.get('max_tokens', 500)),
            "overlap_tokens": int(request.form.get('overlap_tokens', 50))
        }
    }
    
    # Cập nhật cài đặt mặc định
    global default_configs
    default_configs = settings
    
    # Lưu cài đặt vào file
    try:
        with open(os.path.join(upload_folder, 'settings.json'), 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Lỗi khi lưu cài đặt: {str(e)}")
    
    return render_template_string(
        index_html,
        files=global_all_files,
        settings=settings,
        upload_result="Đã lưu cài đặt thành công.",
        answer=None,
        sources=None
    )

# --- Khởi động server và ngrok ---
from pyngrok import ngrok

# Hàm tự động tải lại dữ liệu khi khởi động
def initialize_system():
    """
    Khởi tạo hệ thống và tải lại dữ liệu khi khởi động
    
    Hàm này kiểm tra xem các file cần thiết đã tồn tại hay chưa và tải lại dữ liệu
    từ các file đã lưu.
    """
    global global_metadata, global_all_files, global_vector_list, faiss_index, tfidf_vectorizer, tfidf_matrix
    
    # Kiểm tra và tạo thư mục uploads nếu chưa có
    os.makedirs(upload_folder, exist_ok=True)
    
    # Tải dữ liệu dựa trên người dùng đăng nhập
    success = load_state()
    
    if not success:
        logger.info("Không tìm thấy dữ liệu hiện có, khởi tạo hệ thống mới")
        
        # Khởi tạo biến global - phải khởi tạo đúng kiểu dữ liệu
        global_metadata = []  # Danh sách metadata
        global_all_files = {}  # Dictionary tài liệu, key là tên file
        global_vector_list = []
        
        # Khởi tạo index FAISS
        initialize_faiss_index()
        
        # Khởi tạo TF-IDF
        initialize_tfidf()
        
        # Lưu trạng thái
        save_state()
    
    logger.info(f"Hệ thống đã được khởi tạo với {len(global_all_files)} tài liệu")

def identify_most_relevant_pages(answer, retrieved_chunks, retrieved_metadata):
    """
    Xác định chính xác trang bắt đầu của câu trả lời
    
    Phân tích câu trả lời để tìm trang bắt đầu chứa thông tin liên quan,
    thay vì liệt kê tất cả các trang.
    
    Args:
        answer (str): Câu trả lời được tạo bởi LLM
        retrieved_chunks (list): Danh sách các đoạn văn bản được truy xuất
        retrieved_metadata (list): Metadata tương ứng với các chunks
        
    Returns:
        dict: Từ điển với khóa là filename và giá trị là trang bắt đầu của câu trả lời
    """
    # Tách câu trả lời thành các câu riêng biệt và loại bỏ thẻ trang
    clean_answer = re.sub(r'\[Trang \d+\]\s*', '', answer)
    
    # Tách thành các câu và đoạn văn
    paragraphs = clean_answer.split('\n\n')
    all_sentences = []
    
    # Tách câu trả lời thành câu và sắp xếp theo thứ tự xuất hiện trong câu trả lời
    for para in paragraphs:
        if not para.strip():
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        all_sentences.extend(s.strip() for s in sentences if len(s.strip()) > 20)
    
    # Loại bỏ trùng lặp nhưng giữ thứ tự ban đầu
    unique_sentences = []
    seen = set()
    for s in all_sentences:
        if s not in seen and len(s) > 20:
            seen.add(s)
            unique_sentences.append(s)
    
    # Dictionary lưu trang bắt đầu cho mỗi file
    start_pages = {}
    # Dictionary lưu các câu đã tìm thấy trong mỗi file
    found_sentences = defaultdict(list)
    # Dictionary lưu vị trí đầu tiên của mỗi câu trong câu trả lời
    first_positions = {}
    
    # Tìm vị trí xuất hiện đầu tiên của mỗi câu trong tất cả các chunks
    for sentence_idx, sentence in enumerate(unique_sentences):
        # Lưu vị trí câu trong câu trả lời - ưu tiên câu xuất hiện sớm hơn
        first_positions[sentence] = sentence_idx
        
        for chunk_idx, chunk in enumerate(retrieved_chunks):
            if chunk_idx >= len(retrieved_metadata):
                continue
                
            metadata = retrieved_metadata[chunk_idx]
            filename = metadata.get('filename', 'Unknown')
            
            # Loại bỏ thẻ trang để so sánh nội dung
            clean_chunk = re.sub(r'\[Trang \d+\]\s*', '', chunk)
            
            # Kiểm tra xem câu có trong chunk không
            if sentence in clean_chunk:
                # Lấy số trang từ chunk
                page_markers = re.findall(r'\[Trang (\d+)\]', chunk)
                
                # Ưu tiên sử dụng trang từ metadata nếu có
                if "primary_page" in metadata and metadata["primary_page"] is not None:
                    page = str(metadata["primary_page"])
                elif page_markers:
                    page = page_markers[0]  # Lấy trang đầu tiên trong chunk
                else:
                    continue  # Không tìm thấy trang
                
                # Lưu sentence vào found_sentences với vị trí câu trong câu trả lời
                found_sentences[filename].append((sentence, int(page), sentence_idx))
                
                # Đã tìm thấy câu này trong chunk này, không cần kiểm tra tiếp các chunk khác
                break
    
    # Xác định trang bắt đầu cho mỗi file dựa trên các câu đã tìm thấy
    for filename, sentences in found_sentences.items():
        if not sentences:
            continue
        
        # Sắp xếp các câu theo vị trí xuất hiện trong câu trả lời (câu đầu tiên là quan trọng nhất)
        sorted_sentences = sorted(sentences, key=lambda x: x[2])
        
        # Lấy trang của câu đầu tiên làm trang bắt đầu
        if sorted_sentences:
            start_pages[filename] = str(sorted_sentences[0][1])
    
    # Nếu không tìm thấy trang nào, thử phương pháp dựa trên từ khóa
    if not start_pages:
        return _identify_pages_by_keywords_start(unique_sentences, retrieved_chunks, retrieved_metadata)
    
    # Chuyển đổi thành định dạng tương thích với phần còn lại của code
    result = {}
    for filename, page in start_pages.items():
        result[filename] = {page}
    
    return result

def _identify_pages_by_keywords_start(sentences, retrieved_chunks, retrieved_metadata):
    """
    Phương pháp dự phòng để xác định trang bắt đầu dựa trên từ khóa trong câu trả lời
    
    Args:
        sentences (list): Các câu trong câu trả lời
        retrieved_chunks (list): Các đoạn văn bản được truy xuất
        retrieved_metadata (list): Metadata tương ứng
        
    Returns:
        dict: Từ điển với khóa là filename và giá trị là tập hợp chứa trang bắt đầu
    """
    # Trích xuất các từ khóa từ các câu đầu tiên (quan trọng nhất)
    important_keywords = set()
    for sentence in sentences[:3]:  # Chỉ xem xét 3 câu đầu tiên
        words = sentence.split()
        # Sử dụng độ dài từ để lọc các từ ngắn
        keywords = [w for w in words if len(w) > 4]
        important_keywords.update(keywords)
    
    # Trích xuất thêm từ khóa từ các câu còn lại
    all_keywords = important_keywords.copy()
    for sentence in sentences[3:]:  # Các câu tiếp theo
        words = sentence.split()
        keywords = [w for w in words if len(w) > 4]
        all_keywords.update(keywords)
    
    # Lưu trữ điểm số cho từng trang của mỗi file
    file_pages = {}
    page_scores = defaultdict(lambda: defaultdict(int))
    
    # Tìm các từ khóa trong các chunks
    for idx, chunk in enumerate(retrieved_chunks):
        if idx >= len(retrieved_metadata):
            continue
            
        metadata = retrieved_metadata[idx]
        filename = metadata.get('filename', 'Unknown')
        
        # Lấy thông tin trang
        if "primary_page" in metadata and metadata["primary_page"] is not None:
            page = str(metadata["primary_page"])
        else:
            page_markers = re.findall(r'\[Trang (\d+)\]', chunk)
            if not page_markers:
                continue
            page = page_markers[0]  # Lấy trang đầu tiên
        
        # Tính điểm dựa trên số lượng từ khóa quan trọng trùng khớp
        important_matches = sum(1 for keyword in important_keywords if keyword in chunk)
        all_matches = sum(1 for keyword in all_keywords if keyword in chunk)
        
        # Trang có nhiều từ khóa quan trọng nhất có khả năng cao là trang bắt đầu
        page_scores[filename][page] = important_matches * 3 + all_matches
        
        # Lưu trang vào file_pages
        if filename not in file_pages:
            file_pages[filename] = set()
        file_pages[filename].add(page)
    
    # Chọn trang có điểm cao nhất cho mỗi file làm trang bắt đầu
    start_pages = {}
    for filename, pages in file_pages.items():
        if not pages:
            continue
            
        # Tìm trang có điểm cao nhất
        best_page = max(pages, key=lambda p: page_scores[filename][p])
        start_pages[filename] = {best_page}
    
    return start_pages

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Xử lý đăng nhập người dùng
    """
    if verify_session():
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Vui lòng nhập đầy đủ thông tin', 'error')
            return render_template('login.html')
        
        success, message, user_data = login_user(email, password)
        
        if success:
            flash(message, 'success')
            
            # Nếu có localStorage data trong form, chuyển đến khởi tạo dữ liệu người dùng
            if request.form.get('chat_history_json'):
                return redirect(url_for('init_user_data'))
            
            return redirect(url_for('index'))
        else:
            flash(message, 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    Xử lý đăng ký người dùng mới
    """
    if verify_session():
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not email or not password or not confirm_password:
            flash('Vui lòng nhập đầy đủ thông tin', 'error')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Mật khẩu và xác nhận mật khẩu không khớp', 'error')
            return render_template('register.html')
        
        success, message = register_user(email, password)
        
        if success:
            flash(message, 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """
    Đăng xuất người dùng hiện tại
    """
    if logout_user():
        flash('Đăng xuất thành công', 'success')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET'])
@require_auth
def profile():
    """
    Hiển thị trang hồ sơ người dùng
    """
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    # Lấy danh sách file của người dùng
    files = get_user_files_with_metadata(user['id'])
    
    return render_template('profile.html', user=user, files=files)

@app.route('/change-password', methods=['POST'])
@require_auth
def change_pwd():
    """
    Xử lý thay đổi mật khẩu
    """
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_new_password = request.form.get('confirm_new_password')
    
    if not current_password or not new_password or not confirm_new_password:
        flash('Vui lòng nhập đầy đủ thông tin', 'error')
        return redirect(url_for('profile'))
        
    if new_password != confirm_new_password:
        flash('Mật khẩu mới và xác nhận mật khẩu mới không khớp', 'error')
        return redirect(url_for('profile'))
    
    success, message = change_password(user['id'], current_password, new_password)
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
        
    return redirect(url_for('profile'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """
    Xử lý yêu cầu đặt lại mật khẩu
    """
    if verify_session():
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash('Vui lòng nhập địa chỉ email', 'error')
            return render_template('forgot_password.html')
        
        success, message = reset_password_request(email)
        
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
    
    return render_template('forgot_password.html')

@app.route('/init-user-data', methods=['POST'])
@require_auth
def init_user_data():
    """
    Khởi tạo dữ liệu người dùng (di chuyển từ localStorage sang Supabase)
    """
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    # Lấy dữ liệu từ form
    chat_history_json = request.form.get('chat_history_json', '{}')
    
    # Khởi tạo dữ liệu người dùng
    result = initialize_user_data(user['id'], chat_history_json)
    
    if result['success']:
        flash(f"Khởi tạo dữ liệu thành công! Đã di chuyển {result['migrated_chats']} cuộc trò chuyện và {result['migrated_files']} tệp tin.", 'success')
    else:
        flash(f"Có lỗi xảy ra khi khởi tạo dữ liệu: {result['message']}", 'error')
    
    return redirect(url_for('index'))

@app.route('/delete-file', methods=['POST'])
@require_auth
def delete_user_file_route():
    """
    Xóa file của người dùng
    """
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    file_id = request.form.get('file_id')
    if not file_id:
        flash('Thiếu thông tin file cần xóa', 'error')
        return redirect(url_for('profile'))
    
    success = delete_user_file(user['id'], file_id)
    
    if success:
        flash('Đã xóa file thành công', 'success')
    else:
        flash('Có lỗi xảy ra khi xóa file', 'error')
    
    return redirect(url_for('profile'))

# --- Hàm khởi tạo indices ---
def initialize_faiss_index():
    """
    Khởi tạo FAISS index mới
    """
    global faiss_index
    
    # Khởi tạo FAISS index nếu chưa có
    if faiss_index is None:
        # Tạo index với metric=L2 (Euclidean distance) và kích thước vector 768 (mặc định cho embedding model)
        faiss_index = faiss.IndexFlatL2(768)  
        logger.info(f"Đã khởi tạo FAISS index mới")

def initialize_tfidf():
    """
    Khởi tạo TF-IDF vectorizer và matrix mới
    """
    global tfidf_vectorizer, tfidf_matrix
    
    # Khởi tạo TF-IDF vectorizer mới nếu chưa có
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        tfidf_matrix = None
        logger.info(f"Đã khởi tạo TF-IDF vectorizer mới")

# Thiết lập routes xác thực
# setup_auth_routes(app)
# Thiết lập routes xác thực chính
# setup_auth_routes_main = supabase_integration.setup_auth_routes
# setup_auth_routes_main(app)
# Chỉ sử dụng một cách đăng ký auth routes để tránh trùng lặp
from supabase_modules.auth import setup_auth_routes
setup_auth_routes(app)

# Thiết lập các route API cho chat history và tương tác với Supabase
from supabase_integration import setup_chat_routes, api_update_profile, api_change_password
setup_chat_routes(app)

# Đăng ký các route API liên quan đến profile và đổi mật khẩu từ supabase_integration
app.add_url_rule('/api/user/profile', view_func=api_update_profile, methods=['POST'])
app.add_url_rule('/api/user/change-password', view_func=api_change_password, methods=['POST'])

# Thêm route riêng để kiểm tra kết nối Supabase
@app.route('/api/supabase-check', methods=['GET'])
def check_supabase_connection():
    """
    Kiểm tra kết nối với Supabase
    """
    from supabase_modules.config import get_supabase_client
    import json
    
    supabase = get_supabase_client()
    results = {
        "connection": False,
        "chats_table": False,
        "messages_table": False,
        "auth": False,
        "details": {}
    }
    
    try:
        # Kiểm tra kết nối và bảng chats
        chats_query = supabase.table("chats").select("*").limit(5).execute()
        results["connection"] = True
        results["chats_table"] = True
        results["details"]["chats"] = f"Found {len(chats_query.data or [])} records"
        
        # Kiểm tra bảng messages
        messages_query = supabase.table("messages").select("*").limit(5).execute()
        results["messages_table"] = True
        results["details"]["messages"] = f"Found {len(messages_query.data or [])} records"
        
        # Kiểm tra xác thực
        from supabase_modules.auth import verify_session, get_current_user
        user = get_current_user()
        results["auth"] = user is not None
        results["details"]["auth"] = "Logged in as " + user["email"] if user else "Not logged in"
        
        return jsonify(results)
    except Exception as e:
        results["details"]["error"] = str(e)
        logger.error(f"Lỗi khi kiểm tra kết nối Supabase: {str(e)}")
        return jsonify(results), 500

@app.route('/clear-flash-messages', methods=['POST'])
def clear_flash_messages():
    """
    Xóa flash messages khỏi session
    """
    session.pop('_flashes', None)
    return '', 204

if __name__ == "__main__":
    # Khởi tạo hệ thống và tải lại dữ liệu
    initialize_system()
    
    # logger.info("Khởi tạo kết nối với ngrok và khởi động server Flask")
    # ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))  # Lấy token từ biến môi trường
    # public_url = ngrok.connect(int(os.getenv("PORT", 5000)))
    # print("Public URL:", public_url)
    app.run()