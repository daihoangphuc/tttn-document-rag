# Cài đặt các gói cần thiết
# Chạy lệnh sau trong cell riêng:
#!pip install flask pyngrok google-generativeai transformers sentence-transformers PyPDF2 python-docx nltk faiss-cpu scikit-learn pandas matplotlib tqdm underthesea pyvi rank_bm25

# Thông báo: Dự án này đã được cải tiến với các tính năng tối ưu hóa quá trình truy xuất (retrieval) cho tiếng Việt:
# 1. Query transformation: Sử dụng các kỹ thuật biến đổi truy vấn để tăng tính liên quan giữa truy vấn và tài liệu
# 2. Reranking: Sau khi truy xuất các đoạn tài liệu ban đầu, áp dụng mô hình rerank để sắp xếp lại các kết quả theo mức độ liên quan thực sự
# Các thư viện NLP tiếng Việt (underthesea, pyvi, rank_bm25) được sử dụng để tối ưu hóa cho tiếng Việt

# Import Flask và các thư viện cần thiết
from flask import Flask, request, render_template_string, redirect, url_for, jsonify
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
# Thư viện NLP cho tiếng Việt
try:
    from underthesea import word_tokenize, text_normalize
    from pyvi import ViTokenizer
    from rank_bm25 import BM25Okapi
    VIETNAMESE_NLP_AVAILABLE = True
except ImportError:
    VIETNAMESE_NLP_AVAILABLE = False
    print("Thư viện NLP tiếng Việt không khả dụng. Một số tính năng có thể bị hạn chế.")

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('rag_system')

# Khởi tạo Flask app
app = Flask(__name__)

# Tải biến môi trường từ file .env
load_dotenv()

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
    "max_output_tokens": 4096,  # Độ dài tối đa của câu trả lời
    "response_mime_type": "text/plain"  # Định dạng phản hồi
}

def generate_with_retry(prompt, config, max_retries=MAX_RETRIES):
    """
    Thử gọi API Gemini với cơ chế retry và chuyển đổi API key khi cần
    
    Hàm này sẽ thử gọi API Gemini và tự động chuyển sang API key khác
    nếu gặp lỗi về quota hoặc resource exhausted. Nó sẽ thử lại tối đa
    số lần bằng số lượng API key có sẵn.
    
    Args:
        prompt (str): Nội dung prompt gửi đến Gemini
        config (dict): Cấu hình cho việc sinh nội dung
        max_retries (int): Số lần thử tối đa
        
    Returns:
        str: Nội dung phản hồi từ Gemini
        
    Raises:
        Exception: Nếu không thể tạo câu trả lời sau khi thử tất cả API keys
    """
    global gemini_model
    
    for attempt in range(max_retries):
        try:
            if gemini_model is None:
                gemini_model = initialize_gemini()
                if gemini_model is None:
                    raise Exception("Không thể khởi tạo model Gemini")
            
            # Gọi API Gemini để sinh câu trả lời
            response = gemini_model.generate_content(
                contents=prompt,
                generation_config=config
            )
            
            # Trả về text từ response
            return response.text
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Lần thử {attempt + 1}/{max_retries}: Lỗi khi gọi Gemini API: {error_message}")
            
            if "Resource has been exhausted" in error_message or "quota" in error_message.lower():
                logger.info(f"Chuyển sang API key tiếp theo (lần thử {attempt + 1})")
                gemini_model = switch_api_key()
                time.sleep(1)  # Đợi 1 giây trước khi thử lại
                continue
                
            if attempt == max_retries - 1:
                raise Exception(f"Đã thử {max_retries} lần nhưng không thành công: {error_message}")
    
    raise Exception("Không thể tạo câu trả lời sau khi thử tất cả API keys")

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
    
    Args:
        file_path (str): Đường dẫn đến file DOCX
        
    Returns:
        str: Nội dung văn bản đã trích xuất
    """
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        
        # Đảm bảo trả về một chuỗi văn bản
        if isinstance(text, list):
            text = "\n".join(text)
        return text
    except Exception as e:
        logger.error(f"Lỗi khi đọc file DOCX {file_path}: {str(e)}")
        return ""

def extract_text_txt(file_path):
    """
    Đọc nội dung từ file văn bản thuần túy (TXT)
    
    Args:
        file_path (str): Đường dẫn đến file TXT
        
    Returns:
        str: Nội dung văn bản đã đọc
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
            # Đảm bảo trả về một chuỗi văn bản
            if isinstance(text, list):
                text = "\n".join(text)
            return text
    except UnicodeDecodeError:
        try:
            # Thử lại với encoding khác
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
                
                # Đảm bảo trả về một chuỗi văn bản
                if isinstance(text, list):
                    text = "\n".join(text)
                return text
        except Exception as e:
            logger.error(f"Lỗi khi đọc file TXT {file_path}: {str(e)}")
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
embedder = SentenceTransformer('all-mpnet-base-v2')  # Nâng cấp từ 'all-MiniLM-L6-v2'
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
    state = {
        'metadata': global_metadata,
        'all_files': global_all_files,
        'date': datetime.now().isoformat()
    }
    # Lưu metadata và danh sách file
    with open(os.path.join(upload_folder, 'rag_state.json'), 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    
    # Lưu vectors
    if global_vector_list:
        vectors = np.vstack(global_vector_list).astype('float32')
        with open(os.path.join(upload_folder, 'vectors.pkl'), 'wb') as f:
            pickle.dump(vectors, f)
    
    # Lưu index FAISS nếu đã có
    if faiss_index is not None:
        faiss.write_index(faiss_index, os.path.join(upload_folder, 'faiss_index.bin'))
    
    # Lưu TF-IDF vectorizer và matrix
    if tfidf_vectorizer is not None and tfidf_matrix is not None:
        with open(os.path.join(upload_folder, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(os.path.join(upload_folder, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(tfidf_matrix, f)

def load_state():
    """Nạp trạng thái từ file lưu trữ"""
    global global_metadata, global_all_files, global_vector_list, faiss_index, tfidf_vectorizer, tfidf_matrix
    
    metadata_path = os.path.join(upload_folder, 'rag_state.json')
    vectors_path = os.path.join(upload_folder, 'vectors.pkl')
    index_path = os.path.join(upload_folder, 'faiss_index.bin')
    tfidf_vectorizer_path = os.path.join(upload_folder, 'tfidf_vectorizer.pkl')
    tfidf_matrix_path = os.path.join(upload_folder, 'tfidf_matrix.pkl')
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                global_metadata = state['metadata']
                global_all_files = state.get('all_files', {})
                logger.info(f"Nạp metadata: {len(global_metadata)} chunks, {len(global_all_files)} files")
        except Exception as e:
            logger.error(f"Lỗi khi nạp metadata: {str(e)}")
    
    if os.path.exists(vectors_path):
        try:
            with open(vectors_path, 'rb') as f:
                vectors = pickle.load(f)
                # Tạo lại danh sách vector từ matrix
                global_vector_list = [vectors[i] for i in range(vectors.shape[0])]
                logger.info(f"Nạp vectors: {len(global_vector_list)} vectors")
        except Exception as e:
            logger.error(f"Lỗi khi nạp vectors: {str(e)}")
    
    if os.path.exists(index_path):
        try:
            faiss_index = faiss.read_index(index_path)
            logger.info(f"Nạp FAISS index: {faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Lỗi khi nạp FAISS index: {str(e)}")

    # Nạp TF-IDF vectorizer và matrix
    if os.path.exists(tfidf_vectorizer_path) and os.path.exists(tfidf_matrix_path):
        try:
            with open(tfidf_vectorizer_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            with open(tfidf_matrix_path, 'rb') as f:
                tfidf_matrix = pickle.load(f)
            logger.info(f"Nạp TF-IDF matrix: {tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Lỗi khi nạp TF-IDF: {str(e)}")

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
    global global_vector_list, global_metadata, global_all_files, faiss_index
    
    if filename not in global_all_files:
        return False, f"Không tìm thấy tài liệu '{filename}'"
    
    try:
        # Tìm các chunks thuộc tài liệu
        new_vectors = []
        new_metadata = []
        indices_to_keep = []
        
        for i, meta in enumerate(global_metadata):
            if meta["filename"] != filename:
                new_metadata.append(meta)
                new_vectors.append(global_vector_list[i])
                indices_to_keep.append(i)
        
        # Cập nhật danh sách vector và metadata
        global_vector_list = new_vectors
        global_metadata = new_metadata
        
        # Xóa khỏi danh sách tài liệu
        del global_all_files[filename]
        
        # Tạo lại index FAISS
        if global_vector_list:
            rebuild_indices()
        else:
            faiss_index = None
        
        # Lưu trạng thái
        save_state()
        
        logger.info(f"Đã xóa tài liệu '{filename}' khỏi hệ thống")
        return True, f"Đã xóa tài liệu '{filename}' khỏi hệ thống thành công"
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
            break
    
    # Phát hiện câu hỏi yêu cầu trả lời dài
    requires_long_answer = False
    long_answer_patterns = [
        r'giải thích chi tiết', r'trình bày đầy đủ', r'liệt kê tất cả',
        r'nêu rõ', r'phân tích', r'so sánh', r'đánh giá', r'tổng hợp',
        r'tóm tắt', r'kết luận', r'tổng kết', r'tại sao', r'lý do',
        r'ưu nhược điểm', r'các bước', r'quy trình', r'cách thức',
        r'gồm những gì', r'bao gồm', r'các loại', r'các hình thức',
        r'các yếu tố', r'các nguyên nhân', r'các đặc điểm', r'các đặc trưng'
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
        r'tổng quan', r'tổng quát', r'khái quát', r'sơ lược',
        r'giới thiệu', r'tóm tắt', r'tổng thể', r'toàn cảnh'
    ]
    
    for pattern in overview_patterns:
        if re.search(pattern, question.lower()):
            requires_overview = True
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
    
    # Kết hợp tất cả các phần
    context = "\n".join(context_parts)
    
    return context, used_files

# --- Các hàm mới cho Query Transformation và Reranking ---

def transform_query_for_vietnamese(query, model="gemini-2.0-flash"):
    """
    Biến đổi truy vấn để tối ưu cho tiếng Việt
    
    Hàm này áp dụng các kỹ thuật xử lý ngôn ngữ tự nhiên đặc biệt cho
    tiếng Việt để cải thiện chất lượng truy vấn, bao gồm mở rộng truy vấn,
    phân tích từ khóa, và chuẩn hóa.
    
    Args:
        query (str): Truy vấn gốc
        model (str): Mô hình sử dụng để biến đổi truy vấn
        
    Returns:
        str: Truy vấn đã được tối ưu hóa
    """
    # Chuẩn hóa truy vấn
    normalized_query = query
    if VIETNAMESE_NLP_AVAILABLE:
        try:
            normalized_query = text_normalize(query)
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn hóa truy vấn: {str(e)}")
    
    # Sử dụng LLM để mở rộng truy vấn
    expansion_prompt = f"""
    Hãy phân tích và mở rộng câu truy vấn sau đây để tăng khả năng tìm kiếm thông tin liên quan trong tài liệu tiếng Việt.

    Truy vấn gốc: "{query}"

    Hãy trả về kết quả theo định dạng sau:
    1. Phân tích cấu trúc câu hỏi:
       - Loại câu hỏi (what, how, why, etc.)
       - Chủ đề chính
       - Các yếu tố quan trọng cần tìm
       - Mức độ chi tiết yêu cầu (tổng quan/chi tiết)

    2. Các từ khóa chính:
       - Danh từ quan trọng
       - Động từ quan trọng
       - Tính từ quan trọng
       - Cụm từ quan trọng

    3. Các từ đồng nghĩa và liên quan:
       - Từ đồng nghĩa
       - Từ liên quan ngữ nghĩa
       - Từ trong cùng phạm trù

    4. Các cách diễn đạt khác:
       - Cách diễn đạt trang trọng
       - Cách diễn đạt thông thường
       - Cách diễn đạt chuyên ngành

    5. Các câu hỏi liên quan:
       - Câu hỏi mở rộng
       - Câu hỏi chi tiết
       - Câu hỏi bổ sung

    6. Các truy vấn con:
       - Truy vấn định nghĩa
       - Truy vấn ví dụ
       - Truy vấn giải thích
       - Truy vấn so sánh

    Chỉ trả về kết quả theo định dạng trên, không thêm bất kỳ giải thích nào.
    """
    
    try:
        # Sử dụng hàm generate_with_retry đã có
        expansion_result = generate_with_retry(expansion_prompt, {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        })
        
        # Phân tích kết quả
        analysis = {}
        current_section = None
        current_subsection = None
        
        for line in expansion_result.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Xác định section chính
            if line.startswith('1. '):
                current_section = 'analysis'
                analysis[current_section] = {}
            elif line.startswith('2. '):
                current_section = 'keywords'
                analysis[current_section] = {}
            elif line.startswith('3. '):
                current_section = 'synonyms'
                analysis[current_section] = {}
            elif line.startswith('4. '):
                current_section = 'paraphrases'
                analysis[current_section] = {}
            elif line.startswith('5. '):
                current_section = 'related_questions'
                analysis[current_section] = {}
            elif line.startswith('6. '):
                current_section = 'sub_queries'
                analysis[current_section] = {}
            # Xác định subsection
            elif line.startswith('- '):
                if current_section:
                    key = line.replace('- ', '').split(':')[0].strip()
                    value = line.split(':')[1].strip() if ':' in line else ''
                    if value:
                        analysis[current_section][key] = [v.strip() for v in value.split(',')]
                    else:
                        current_subsection = key
                        analysis[current_section][current_subsection] = []
            # Thêm giá trị vào subsection hiện tại
            elif current_section and current_subsection and line:
                analysis[current_section][current_subsection].append(line)
        
        # Tạo các biến thể truy vấn dựa trên phân tích
        query_variants = [query]  # Bắt đầu với truy vấn gốc
        
        # Thêm các cách diễn đạt khác
        if 'paraphrases' in analysis:
            for style, phrases in analysis['paraphrases'].items():
                query_variants.extend(phrases)
        
        # Tạo truy vấn mở rộng với từ khóa
        if 'keywords' in analysis:
            keywords = []
            for key_type, words in analysis['keywords'].items():
                keywords.extend(words)
            expanded_query = f"{query} {' '.join(keywords)}"
            query_variants.append(expanded_query)
        
        # Thêm các truy vấn con
        if 'sub_queries' in analysis:
            for query_type, queries in analysis['sub_queries'].items():
                query_variants.extend(queries)
        
        # Loại bỏ các biến thể trống hoặc quá ngắn
        query_variants = [q for q in query_variants if len(q.strip()) > 5]
        
        # Loại bỏ các biến thể trùng lặp
        query_variants = list(set(query_variants))
        
        # Sắp xếp các biến thể theo độ dài (ưu tiên các truy vấn ngắn và súc tích)
        query_variants.sort(key=len)
        
        logger.info(f"Đã tạo {len(query_variants)} biến thể truy vấn cho: '{query}'")
        
        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "query_variants": query_variants,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Lỗi khi biến đổi truy vấn: {str(e)}")
        # Trả về phiên bản đơn giản nếu có lỗi
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
    
    if faiss_index is None or len(global_metadata) == 0:
        return "Chưa có tài liệu nào để tra cứu.", []
    
    # Bắt đầu đo thời gian truy xuất
    start_retrieval = time.time()
    
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
    # Biến đổi truy vấn để tăng tính liên quan
    try:
        transformed_query = transform_query_for_vietnamese(question, model)
        query_variants = transformed_query["query_variants"]
        logger.info(f"Đã tạo {len(query_variants)} biến thể truy vấn")
    except Exception as e:
        logger.error(f"Lỗi khi biến đổi truy vấn: {str(e)}")
        query_variants = [question]
    
    # Tìm kiếm với tất cả các biến thể truy vấn
    all_indices = []
    all_distances = []
    
    for variant in query_variants:
        try:
            # Mã hóa biến thể truy vấn
            q_vec = embedder.encode(variant, convert_to_tensor=False).astype('float32')
            q_vec = np.expand_dims(q_vec, axis=0)
            
            # Tìm kiếm với FAISS
            distances, indices = faiss_index.search(q_vec, min(top_k, faiss_index.ntotal))
            
            # Thêm vào kết quả tổng hợp
            all_indices.extend(indices[0])
            all_distances.extend(distances[0])
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm với biến thể '{variant}': {str(e)}")
    
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
            return "Không có thông tin liên quan trong tài liệu.", []
    else:
        logger.info("Không tìm thấy kết quả nào từ tìm kiếm ngữ nghĩa")
        return "Không có thông tin liên quan trong tài liệu.", []
    
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
    system_prompt = """Bạn là trợ lý AI có khả năng trả lời câu hỏi dựa trên thông tin được cung cấp. 
    Nhiệm vụ của bạn là:
    1. Phân tích thông tin từ các đoạn văn bản được cung cấp.
    2. Trả lời câu hỏi một cách CHÍNH XÁC và ĐẦY ĐỦ, sử dụng CHÍNH XÁC cách diễn đạt trong tài liệu gốc.
    3. KHÔNG thêm các từ mở đầu thừa như "Định nghĩa:", "Theo tài liệu:", "Dựa trên thông tin cung cấp:", v.v.
    4. Trả lời trực tiếp vào câu hỏi, giữ nguyên cách diễn đạt trong tài liệu gốc.
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
    except Exception as e:
        error_message = str(e)
        logger.error(f"Lỗi khi xử lý câu trả lời: {error_message}")
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {error_message}", []
    
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
    answer = re.sub(r'\[Trang \d+\]\s*', '', answer)
    
    # Thêm thông tin nguồn tài liệu
    if used_files:
        unique_files = list(set(used_files))
        sources_info = f"\n\nNguồn tài liệu: {', '.join(unique_files)}"
        answer += sources_info
    
    # Kết thúc đo thời gian trả lời
    end_answer = time.time()
    answer_time = end_answer - start_answer
    
    # Theo dõi hiệu suất
    if chunking_method is None:
        # Xác định phương pháp chunking phổ biến nhất trong các chunks được sử dụng
        chunking_methods = [meta.get("chunking_method", "unknown") for meta in retrieved_metadata if "chunking_method" in meta]
        if chunking_methods:
            chunking_method = Counter(chunking_methods).most_common(1)[0][0]
        else:
            chunking_method = "unknown"
    
    # Theo dõi hiệu suất với cả hai hàm
    track_performance(question, retrieval_time, answer_time, len(retrieved_chunks), chunking_method)
    track_performance_metrics(question, retrieval_time, answer_time, len(retrieved_chunks), chunking_method)
    
    return answer, used_files

# --- API routes cho hiệu suất và đánh giá ---
@app.route('/api/performance', methods=['GET'])
def get_performance():
    """API endpoint để lấy thông tin hiệu suất"""
    chunking_method = request.args.get('chunking_method')
    analysis = analyze_performance(chunking_method)
    return jsonify(analysis)

@app.route('/api/embeddings', methods=['GET'])
def view_embeddings():
    """Endpoint để xem thông tin về embeddings"""
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
        answer, sources = answer_question(question, top_k, threshold, model, chunking_method)
    except Exception as e:
        error_message = str(e)
        logger.error(f"API: Lỗi khi trả lời câu hỏi: {error_message}")
        
        # Xử lý lỗi token length
        if "Token indices sequence length is longer than the specified maximum sequence length" in error_message:
            # Giảm top_k và thử lại
            reduced_top_k = max(3, top_k // 3)
            logger.info(f"API: Giảm top_k từ {top_k} xuống {reduced_top_k} và thử lại")
            
            try:
                answer, sources = answer_question(question, reduced_top_k, threshold, model, chunking_method)
            except Exception as e2:
                logger.error(f"API: Vẫn lỗi sau khi giảm top_k: {str(e2)}")
                return jsonify({
                    "error": "Lỗi khi xử lý câu hỏi do văn bản quá dài",
                    "details": str(e),
                    "question": question,
                    "is_law_question": is_law_question
                }), 500
        else:
            return jsonify({
                "error": "Lỗi khi xử lý câu hỏi",
                "details": error_message,
                "question": question,
                "is_law_question": is_law_question
            }), 500
    
    return jsonify({
        "question": question,
        "answer": answer,
        "sources": sources,
        "is_law_question": is_law_question
    })

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
    return render_template_string(
        index_html, 
        files=global_all_files,
        settings=load_settings(),
        answer=None,
        sources=None,
        upload_result=None
    )

# Route cho truy vấn
@app.route('/query', methods=['POST'])
def query():
    question = request.form.get('question', '')
    model = request.form.get('model', 'gemini-2.0-flash')
    top_k = int(request.form.get('top_k', 10))
    threshold = float(request.form.get('threshold', 5.0))
    
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
                """
                <div class="chat-message bot-message">
                    <p class="text-gray-800 dark:text-gray-200 text-sm sm:text-base">Vui lòng nhập câu hỏi.</p>
                </div>
                """
            )
        else:
            return render_template_string(
                index_html,
                files=global_all_files,
                settings=load_settings(),
                answer="Vui lòng nhập câu hỏi.",
                sources=[]
            )
    
    # Phát hiện câu hỏi về điều khoản luật
    is_law_question = False
    law_article_patterns = [
        r'điều\s+\d+', r'khoản\s+\d+', r'điểm\s+\d+',
        r'chính sách', r'nguyên tắc', r'quy định', r'luật'
    ]
    
    for pattern in law_article_patterns:
        if re.search(pattern, question.lower()):
            is_law_question = True
            logger.info(f"Web UI: Phát hiện câu hỏi về điều khoản luật: {question}")
            # Tăng top_k cho câu hỏi về luật
            top_k = max(top_k, 30)
            break
    
    try:
        answer, sources = answer_question(question, top_k, threshold, model)
    except Exception as e:
        error_message = str(e)
        logger.error(f"Lỗi khi trả lời câu hỏi: {error_message}")
        
        # Xử lý lỗi token length
        if "Token indices sequence length is longer than the specified maximum sequence length" in error_message:
            # Giảm top_k và thử lại
            reduced_top_k = max(3, top_k // 3)
            logger.info(f"Giảm top_k từ {top_k} xuống {reduced_top_k} và thử lại")
            
            try:
                answer, sources = answer_question(question, reduced_top_k, threshold, model)
            except Exception as e2:
                error_message = str(e2)
                logger.error(f"Vẫn lỗi sau khi giảm top_k: {error_message}")
                
                if is_ajax:
                    logger.info("Returning AJAX error response")
                    return render_template_string(
                        """
                        <div class="chat-message bot-message">
                            <p class="text-red-500 dark:text-red-400 text-sm sm:text-base">Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {{ error }}</p>
                        </div>
                        """, error=error_message
                    )
                else:
                    return render_template_string(
                        index_html,
                        files=global_all_files,
                        settings=load_settings(),
                        answer=f"Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {error_message}",
                        sources=[]
                    )
        else:
            if is_ajax:
                logger.info("Returning AJAX error response")
                return render_template_string(
                    """
                    <div class="chat-message bot-message">
                        <p class="text-red-500 dark:text-red-400 text-sm sm:text-base">Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {{ error }}</p>
                    </div>
                    """, error=error_message
                )
            else:
                return render_template_string(
                    index_html,
                    files=global_all_files,
                    settings=load_settings(),
                    answer=f"Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {error_message}",
                    sources=[]
                )
    
    # Nếu là AJAX request, chỉ trả về phần tin nhắn bot
    if is_ajax:
        logger.info(f"Returning AJAX response with answer length: {len(answer)}")
        return render_template_string(
            """
            <div class="chat-message bot-message">
                <div class="text-gray-800 dark:text-gray-200 whitespace-pre-wrap text-sm sm:text-base">{{ answer }}</div>
            </div>
            """, answer=answer
        )
    else:
        # Nếu không phải AJAX, trả về toàn bộ trang
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            answer=answer,
            sources=sources,
            question=question
        )

# Route cho upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    # Lấy phương pháp chunking từ form
    chunking_method = request.form.get('chunking_method', 'sentence_windows')
    
    # Kiểm tra xem có file được gửi lên không
    if 'file[]' not in request.files:
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result="Không có file nào được chọn.",
            answer=None,
            sources=None
        )
    
    files = request.files.getlist('file[]')
    
    # Nếu không có file nào được chọn
    if len(files) == 0 or files[0].filename == '':
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result="Không có file nào được chọn.",
            answer=None,
            sources=None
        )
    
    # Định dạng file được phép
    allowed_extensions = {'txt', 'pdf', 'docx'}
    
    # Biến lưu kết quả xử lý
    processed_files = []
    failed_files = []
    total_chunks = 0
    
    # Xử lý từng file
    for file in files:
        # Kiểm tra phần mở rộng file
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            failed_files.append(f"{file.filename} (định dạng không được hỗ trợ)")
            continue
        
        try:
            # Lưu file
            filename = file.filename
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            
            # Trích xuất văn bản từ file
            text = ""
            if file_ext == 'pdf':
                text = extract_text_pdf(file_path)
            elif file_ext == 'docx':
                text = extract_text_docx(file_path)
            elif file_ext == 'txt':
                text = extract_text_txt(file_path)
            
            # Kiểm tra xem text có phải là chuỗi không
            if not isinstance(text, str):
                logger.error(f"Lỗi: text không phải là chuỗi mà là {type(text)}")
                if isinstance(text, list):
                    text = "\n".join(text)
                else:
                    text = str(text) if text is not None else ""
            
            # Thêm tài liệu vào hệ thống RAG
            chunk_count = add_document(text, filename, chunking_method)
            total_chunks += chunk_count
            processed_files.append(filename)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file {file.filename}: {str(e)}")
            failed_files.append(f"{file.filename} (lỗi: {str(e)})")
    
    # Lưu trạng thái
    save_state()
    
    # Tạo thông báo kết quả
    if len(processed_files) > 0 and len(failed_files) == 0:
        if len(processed_files) == 1:
            result_message = f"Đã tải lên và xử lý thành công file {processed_files[0]} với {total_chunks} chunks."
        else:
            result_message = f"Đã tải lên và xử lý thành công {len(processed_files)} file với tổng cộng {total_chunks} chunks."
    elif len(processed_files) > 0 and len(failed_files) > 0:
        result_message = f"Đã xử lý {len(processed_files)} file thành công với {total_chunks} chunks. {len(failed_files)} file thất bại: {', '.join(failed_files)}"
    else:
        result_message = f"Không thể xử lý file: {', '.join(failed_files)}"
    
    # Kiểm tra nếu request là AJAX (có header X-Requested-With)
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if is_ajax:
        # Trả về JSON nếu là AJAX request
        return jsonify({
            'success': len(processed_files) > 0,
            'message': result_message,
            'files': global_all_files
        })
    else:
        # Trả về HTML như bình thường
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result=result_message,
            answer=None,
            sources=None
        )

# Route cho xóa file
@app.route('/remove', methods=['POST'])
def remove_file():
    filename = request.form.get('filename')
    if not filename:
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result="Lỗi: Không có tên tài liệu được cung cấp.",
            answer=None,
            sources=None
        )
    
    if filename not in global_all_files:
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result=f"Lỗi: Không tìm thấy tài liệu '{filename}' trong hệ thống.",
            answer=None,
            sources=None
        )
    
    try:
        success, message = remove_document(filename)
        
        if not success:
            return render_template_string(
                index_html,
                files=global_all_files,
                settings=load_settings(),
                upload_result=f"Lỗi khi xóa tài liệu: {message}",
                answer=None,
                sources=None
            )
        
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result=message,
            answer=None,
            sources=None
        )
    except Exception as e:
        logger.error(f"Lỗi khi xóa tài liệu '{filename}': {str(e)}")
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result=f"Đã xảy ra lỗi khi xóa tài liệu: {str(e)}",
            answer=None,
            sources=None
        )

# Route cho lưu cài đặt
@app.route('/settings', methods=['POST'])
def save_settings():
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
if __name__ == "__main__":
    logger.info("Khởi tạo kết nối với ngrok và khởi động server Flask")
    ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))  # Lấy token từ biến môi trường
    public_url = ngrok.connect(int(os.getenv("PORT", 5000)))
    print("Public URL:", public_url)
    app.run()