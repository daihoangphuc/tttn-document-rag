"""
Mô-đun tích hợp Supabase vào hệ thống RAG
"""
import os
import logging
from flask import redirect, url_for, render_template, flash, request, jsonify, session
from functools import wraps
from supabase_modules.config import get_supabase_client
from supabase_modules.auth import register_user, login_user, logout_user, get_current_user, verify_session, require_auth, change_password, reset_password_request, update_user_profile
from supabase_modules.chat_history import create_chat, get_chats, get_chat, update_chat_title, delete_chat, add_message, get_messages, migrate_chat_history, validate_and_format_uuid
from supabase_modules.file_manager import ensure_user_upload_dir, save_user_file, get_user_files, delete_user_file, migrate_files_to_user_directory, get_file_path
from supabase_modules.helpers import get_user_id_from_session, migrate_localStorage_to_supabase, initialize_user_data, format_chat_history_for_frontend, get_user_files_with_metadata
import json

# Thiết lập logging
logger = logging.getLogger('RAG_System')

@require_auth
def api_update_profile():
    """
    API cập nhật thông tin cá nhân của người dùng
    """
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
    
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "Không có dữ liệu cần cập nhật"
            }), 400
        
        # Cập nhật thông tin người dùng
        success, message, updated_user = update_user_profile(user['id'], data)
        
        if success and updated_user:
            # Cập nhật session nếu cần
            if 'email' in data and data['email'] != user['email']:
                session['email'] = data['email']
            
            return jsonify({
                "success": True,
                "message": message,
                "user": {
                    "id": updated_user.id,
                    "email": updated_user.email,
                    "full_name": updated_user.user_metadata.get('full_name') if updated_user.user_metadata else None
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": message
            }), 400
            
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật thông tin cá nhân: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Có lỗi xảy ra: {str(e)}"
        }), 500

@require_auth
def api_change_password():
    """
    API đổi mật khẩu người dùng
    """
    user = get_current_user()
    if not user:
        return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
        
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return jsonify({
                "success": False,
                "message": "Thiếu thông tin mật khẩu"
            }), 400
            
        # Đổi mật khẩu
        success, message = change_password(user['id'], current_password, new_password)
        
        if success:
            return jsonify({
                "success": True,
                "message": message
            })
        else:
            return jsonify({
                "success": False,
                "message": message
            }), 400
            
    except Exception as e:
        logger.error(f"Lỗi khi đổi mật khẩu: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Có lỗi xảy ra: {str(e)}"
        }), 500

# Setup routes cho API Chat
def setup_chat_routes(app):
    """
    Thiết lập các route API liên quan đến chat cho ứng dụng
    
    Args:
        app: Flask app
    """
    @app.route('/api/chat/history', methods=['GET'])
    @require_auth
    def api_get_chat_history():
        """
        API trả về lịch sử chat của người dùng
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
            
        try:
            # Lấy danh sách chat của người dùng
            chats = get_chats(user['id'])
            
            # Bổ sung thông tin tin nhắn cho mỗi cuộc trò chuyện
            for chat in chats:
                if 'id' in chat:
                    # Lấy tin nhắn cho mỗi cuộc trò chuyện
                    chat['messages'] = get_messages(chat['id'], user['id'])
            
            # Format lại data để trả về cho frontend
            chat_history = format_chat_history_for_frontend(chats)
            
            return jsonify({
                "success": True,
                "chat_history": chat_history
            })
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy lịch sử chat: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra: {str(e)}"
            }), 500
    
    @app.route('/api/chat/sync', methods=['POST'])
    @require_auth
    def api_sync_chat_history():
        """
        API đồng bộ hóa lịch sử chat từ localStorage lên Supabase
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
        
        try:
            # Lấy dữ liệu từ request
            data = request.get_json()
            if not data or not data.get('chat_history'):
                return jsonify({
                    "success": False,
                    "message": "Thiếu dữ liệu lịch sử chat"
                }), 400
                
            chat_history_json = json.dumps(data['chat_history'])
            
            # Đồng bộ hóa lịch sử chat
            synced_count, errors = migrate_chat_history(user['id'], chat_history_json)
            
            if errors:
                return jsonify({
                    "success": True,
                    "synced_count": synced_count,
                    "message": f"Đã đồng bộ {synced_count} cuộc trò chuyện, có {len(errors)} lỗi",
                    "errors": errors
                })
            else:
                return jsonify({
                    "success": True,
                    "synced_count": synced_count,
                    "message": f"Đã đồng bộ {synced_count} cuộc trò chuyện thành công"
                })
                
        except Exception as e:
            logger.error(f"Lỗi khi đồng bộ lịch sử chat: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra: {str(e)}"
            }), 500
    
    @app.route('/api/chat', methods=['POST'])
    @require_auth
    def api_create_chat():
        """
        API tạo cuộc trò chuyện mới
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
            
        try:
            # In thông tin người dùng để debug
            logger.info(f"Tạo chat mới cho user_id: {user['id']}")
            
            # Lấy dữ liệu từ request
            data = request.get_json()
            title = data.get('title', 'Cuộc trò chuyện mới')
            
            # Tạo cuộc trò chuyện mới
            new_chat = create_chat(user['id'], title)
            
            if new_chat:
                logger.info(f"Đã tạo chat mới thành công: {new_chat['id']}")
                return jsonify({
                    "success": True,
                    "chat": new_chat,
                    "message": "Đã tạo cuộc trò chuyện mới"
                })
            else:
                logger.error(f"Không thể tạo cuộc trò chuyện mới cho user_id: {user['id']}")
                return jsonify({
                    "success": False,
                    "message": "Không thể tạo cuộc trò chuyện mới"
                }), 500
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo cuộc trò chuyện: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra: {str(e)}"
            }), 500
    
    @app.route('/api/chat/<chat_id>/message', methods=['POST'])
    @require_auth
    def api_add_message(chat_id):
        """
        API thêm tin nhắn vào cuộc trò chuyện
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
            
        try:
            # Kiểm tra và chuẩn hóa UUID
            validated_chat_id = validate_and_format_uuid(chat_id, "chat_id")
            if not validated_chat_id:
                return jsonify({
                    "success": False, 
                    "message": f"ID cuộc trò chuyện không hợp lệ: {chat_id}"
                }), 400
            
            # Lấy dữ liệu từ request
            data = request.get_json()
            role = data.get('role', 'user')
            content = data.get('content')
            
            if not content:
                return jsonify({
                    "success": False,
                    "message": "Thiếu nội dung tin nhắn"
                }), 400
                
            # Thêm tin nhắn
            message = add_message(validated_chat_id, user['id'], role, content)
            
            if message:
                return jsonify({
                    "success": True,
                    "message": message,
                    "message_text": "Đã thêm tin nhắn thành công"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Không thể thêm tin nhắn"
                }), 500
                
        except Exception as e:
            logger.error(f"Lỗi khi thêm tin nhắn: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra: {str(e)}"
            }), 500
    
    @app.route('/api/chat/<chat_id>/messages', methods=['GET'])
    @require_auth
    def api_get_messages(chat_id):
        """
        API lấy danh sách tin nhắn của một cuộc trò chuyện
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
            
        try:
            # Kiểm tra và chuẩn hóa UUID trước khi truy vấn
            validated_chat_id = validate_and_format_uuid(chat_id, "chat_id")
            if not validated_chat_id:
                return jsonify({
                    "success": False, 
                    "message": f"ID cuộc trò chuyện không hợp lệ: {chat_id}"
                }), 400
            
            # Sử dụng ID đã được chuẩn hóa
            messages = get_messages(validated_chat_id, user['id'])
            
            return jsonify({
                "success": True,
                "messages": messages
            })
                
        except Exception as e:
            logger.error(f"Lỗi khi lấy tin nhắn: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra: {str(e)}"
            }), 500
    
    @app.route('/api/chat/<chat_id>', methods=['DELETE'])
    @require_auth
    def api_delete_chat(chat_id):
        """
        API xóa một cuộc trò chuyện và tất cả tin nhắn liên quan
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
            
        try:
            logger.info(f"Bắt đầu xóa cuộc trò chuyện ID: {chat_id} bởi user_id: {user['id']}")
            
            # Kiểm tra và chuẩn hóa UUID
            validated_chat_id = validate_and_format_uuid(chat_id, "chat_id")
            if not validated_chat_id:
                logger.warning(f"Yêu cầu xóa với chat_id không hợp lệ: {chat_id}")
                return jsonify({
                    "success": False, 
                    "message": f"ID cuộc trò chuyện không hợp lệ: {chat_id}"
                }), 400
            
            # Ghi log thông tin
            logger.info(f"Đã validate chat_id: {validated_chat_id}, đang tiến hành xóa...")
            
            # Xóa cuộc trò chuyện và tất cả tin nhắn liên quan
            success = delete_chat(validated_chat_id, user['id'])
            
            if success:
                logger.info(f"Đã xóa thành công cuộc trò chuyện ID: {validated_chat_id} của user_id: {user['id']}")
                return jsonify({
                    "success": True,
                    "message": "Đã xóa cuộc trò chuyện và tất cả tin nhắn thành công"
                })
            else:
                logger.error(f"Không thể xóa cuộc trò chuyện ID: {validated_chat_id} của user_id: {user['id']}")
                return jsonify({
                    "success": False,
                    "message": "Không thể xóa cuộc trò chuyện. Vui lòng thử lại sau."
                }), 500
                
        except Exception as e:
            logger.error(f"Lỗi khi xóa cuộc trò chuyện ID {chat_id}: {str(e)}")
            import traceback
            logger.error(f"Chi tiết lỗi: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra khi xóa cuộc trò chuyện: {str(e)}"
            }), 500

# Các route xác thực
def setup_auth_routes(app, index_html=None):
    """
    Thiết lập các route xác thực cho ứng dụng
    
    Args:
        app: Flask app
        index_html: Template HTML cho trang chủ (nếu cần)
    """
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
                
                # Chuyển hướng đến @index.html thay vì url_for('index')
                return redirect('/')
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

    @app.route('/api/user/profile', methods=['POST'])
    @require_auth
    def api_update_profile():
        """
        API cập nhật thông tin cá nhân của người dùng
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
        
        try:
            # Lấy dữ liệu từ request
            data = request.get_json()
            if not data:
                return jsonify({
                    "success": False,
                    "message": "Không có dữ liệu cần cập nhật"
                }), 400
            
            # Cập nhật thông tin người dùng
            success, message, updated_user = update_user_profile(user['id'], data)
            
            if success and updated_user:
                # Cập nhật session nếu cần
                if 'email' in data and data['email'] != user['email']:
                    session['email'] = data['email']
                
                return jsonify({
                    "success": True,
                    "message": message,
                    "user": {
                        "id": updated_user.id,
                        "email": updated_user.email,
                        "full_name": updated_user.user_metadata.get('full_name') if updated_user.user_metadata else None
                    }
                })
            else:
                return jsonify({
                    "success": False,
                    "message": message
                }), 400
                
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật thông tin cá nhân: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra: {str(e)}"
            }), 500

    @app.route('/api/user/change-password', methods=['POST'])
    @require_auth
    def api_change_password():
        """
        API đổi mật khẩu người dùng
        """
        user = get_current_user()
        if not user:
            return jsonify({"success": False, "message": "Chưa đăng nhập"}), 401
            
        try:
            # Lấy dữ liệu từ request
            data = request.get_json()
            current_password = data.get('current_password')
            new_password = data.get('new_password')
            
            if not current_password or not new_password:
                return jsonify({
                    "success": False,
                    "message": "Thiếu thông tin mật khẩu"
                }), 400
                
            # Đổi mật khẩu
            success, message = change_password(user['id'], current_password, new_password)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": message
                })
            else:
                return jsonify({
                    "success": False,
                    "message": message
                }), 400
                
        except Exception as e:
            logger.error(f"Lỗi khi đổi mật khẩu: {str(e)}")
            return jsonify({
                "success": False,
                "message": f"Có lỗi xảy ra: {str(e)}"
            }), 500

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

# Các hàm xử lý file
def enhanced_upload_file(upload_folder, index_html, global_all_files, load_settings, extract_text_pdf, extract_text_docx, extract_text_txt, add_document, save_state, load_state=None):
    """
    Phiên bản nâng cao của route upload_file với xác thực người dùng
    """
    # Nạp trạng thái hệ thống của người dùng đang đăng nhập
    if load_state:
        load_state()
    
    # Xác thực người dùng
    user = get_current_user()
    if not user:
        # Nếu không có người dùng đăng nhập, chuyển hướng đến trang đăng nhập
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': "Bạn cần đăng nhập để tải lên tài liệu"
            })
        flash("Bạn cần đăng nhập để tải lên tài liệu", "error")
        return redirect(url_for('login'))
    
    # Lấy phương pháp chunking từ form
    chunking_method = request.form.get('chunking_method', 'sentence_windows')
    
    # Kiểm tra xem có file được gửi lên không
    if 'file[]' not in request.files:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': "Không có file nào được chọn"
            })
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
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': "Không có file nào được chọn"
            })
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
    duplicate_files = []
    
    # Đảm bảo thư mục người dùng tồn tại
    user_upload_dir = ensure_user_upload_dir(user['id'])
    
    # Xử lý từng file
    for file in files:
        # Kiểm tra phần mở rộng file
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            failed_files.append(f"{file.filename} (định dạng không được hỗ trợ)")
            continue
        
        # Kiểm tra xem file đã tồn tại trong hệ thống chưa
        if file.filename in global_all_files:
            duplicate_files.append(file.filename)
            continue
        
        try:
            # Lưu file bằng module file_manager
            success, file_path = save_user_file(user['id'], file)
            
            if not success:
                failed_files.append(f"{file.filename} (lỗi khi lưu: {file_path})")
                continue
            
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
            chunk_count = add_document(text, file.filename, chunking_method)
            
            if chunk_count > 0:
                processed_files.append(file.filename)
                total_chunks += chunk_count
            else:
                failed_files.append(f"{file.filename} (không thể trích xuất văn bản)")
                
        except Exception as e:
            failed_files.append(f"{file.filename} (lỗi: {str(e)})")
            logger.error(f"Lỗi khi xử lý file '{file.filename}': {str(e)}")
    
    # Lưu trạng thái
    save_state()
    
    # Tạo thông báo kết quả
    result_parts = []
    
    if processed_files:
        result_parts.append(f"Đã xử lý {len(processed_files)} tài liệu thành công ({total_chunks} chunks).")
    
    if duplicate_files:
        result_parts.append(f"Có {len(duplicate_files)} tài liệu đã tồn tại trong hệ thống.")
    
    if failed_files:
        result_parts.append(f"Không thể xử lý {len(failed_files)} tài liệu: {', '.join(failed_files)}")
    
    result_message = " ".join(result_parts)
    
    # Trả về response tùy theo loại request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
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

def enhanced_remove_file(index_html, global_all_files, load_settings, remove_document):
    """
    Phiên bản nâng cao của route remove_file với xác thực người dùng
    """
    # Xác thực người dùng
    user = get_current_user()
    if not user:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': "Bạn cần đăng nhập để xóa tài liệu"
            }), 401
        flash("Bạn cần đăng nhập để xóa tài liệu", "error")
        return redirect(url_for('login'))
    
    filename = request.form.get('filename')
    if not filename:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': "Không có tên tài liệu được cung cấp"
            }), 400
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result="Lỗi: Không có tên tài liệu được cung cấp.",
            answer=None,
            sources=None
        )
    
    if filename not in global_all_files:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': f"Không tìm thấy tài liệu '{filename}' trong hệ thống"
            }), 404
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result=f"Lỗi: Không tìm thấy tài liệu '{filename}' trong hệ thống.",
            answer=None,
            sources=None
        )
    
    try:
        # Xóa file trong Supabase
        supabase_delete_success = delete_user_file(user['id'], filename)
        if not supabase_delete_success:
            logger.warning(f"Không thể xóa tệp '{filename}' từ Supabase cho user_id: {user['id']}")
        
        # Xóa tài liệu khỏi hệ thống RAG
        success, message = remove_document(filename)
        
        if not success:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': False,
                    'message': message
                }), 500
            return render_template_string(
                index_html,
                files=global_all_files,
                settings=load_settings(),
                upload_result=f"Lỗi khi xóa tài liệu: {message}",
                answer=None,
                sources=None
            )
        
        # Trả về response thành công với script để xóa localStorage
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'message': message,
                'filename': filename,
                'clearLocalStorage': True
            })
        
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
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'message': f"Đã xảy ra lỗi khi xóa tài liệu: {str(e)}"
            }), 500
        
        return render_template_string(
            index_html,
            files=global_all_files,
            settings=load_settings(),
            upload_result=f"Đã xảy ra lỗi khi xóa tài liệu: {str(e)}",
            answer=None,
            sources=None
        )

# Cập nhật homepage để bao gồm thông tin người dùng và đăng nhập
def get_enhanced_index_html():
    """
    Phiên bản nâng cao của index_html bao gồm thông tin người dùng và đăng nhập
    """
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System with Supabase</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.1/flowbite.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.1/flowbite.min.js"></script>
    <!-- Bổ sung các style và script của hệ thống hiện tại -->
</head>
<body class="bg-gray-50 dark:bg-gray-900">
    <!-- Navbar -->
    <nav class="bg-white border-b border-gray-200 px-4 py-2.5 dark:bg-gray-800 dark:border-gray-700 fixed left-0 right-0 top-0 z-50">
        <div class="flex flex-wrap justify-between items-center">
            <div class="flex justify-start items-center">
                <a href="/" class="flex items-center">
                    <span class="self-center text-xl font-semibold whitespace-nowrap dark:text-white">RAG System</span>
                </a>
            </div>
            <div class="flex items-center lg:order-2">
                {% if session.get('user_id') %}
                <button type="button" class="flex mr-3 text-sm bg-gray-800 rounded-full md:mr-0 focus:ring-4 focus:ring-gray-300 dark:focus:ring-gray-600" id="user-menu-button" aria-expanded="false" data-dropdown-toggle="user-dropdown" data-dropdown-placement="bottom">
                    <span class="sr-only">Mở menu</span>
                    <div class="relative w-10 h-10 overflow-hidden bg-gray-100 rounded-full dark:bg-gray-600">
                        <svg class="absolute w-12 h-12 text-gray-400 -left-1" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"></path></svg>
                    </div>
                </button>
                <!-- Dropdown menu -->
                <div class="hidden z-50 my-4 text-base list-none bg-white divide-y divide-gray-100 rounded-lg shadow dark:bg-gray-700 dark:divide-gray-600" id="user-dropdown">
                    <div class="px-4 py-3">
                        <span class="block text-sm text-gray-900 dark:text-white">{{ session.get('email') }}</span>
                    </div>
                    <ul class="py-2" aria-labelledby="user-menu-button">
                        <li>
                            <a href="/profile" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white">Hồ sơ</a>
                        </li>
                        <li>
                            <a href="/logout" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white">Đăng xuất</a>
                        </li>
                    </ul>
                </div>
                {% else %}
                <a href="/login" class="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 dark:bg-blue-600 dark:hover:bg-blue-700 focus:outline-none dark:focus:ring-blue-800">Đăng nhập</a>
                {% endif %}
            </div>
        </div>
    </nav>

    <!-- Nội dung chính -->
    <div class="pt-20">
        <!-- Flash messages -->
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="p-4 mb-4 mx-4 text-sm rounded-lg
                        {% if category == 'error' %}text-red-800 bg-red-50 dark:bg-gray-800 dark:text-red-400{% endif %}
                        {% if category == 'success' %}text-green-800 bg-green-50 dark:bg-gray-800 dark:text-green-400{% endif %}
                        {% if category == 'warning' %}text-yellow-800 bg-yellow-50 dark:bg-gray-800 dark:text-yellow-300{% endif %}
                        {% if category == 'info' %}text-blue-800 bg-blue-50 dark:bg-gray-800 dark:text-blue-400{% endif %}"
                        role="alert">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <!-- Đặt nội dung của giao diện hiện tại ở đây -->

    </div>
</body>
</html>
""" 

# Thêm hàm lấy toàn bộ lịch sử chat cho người dùng
def get_full_chat_history(user_id):
    """
    Lấy toàn bộ lịch sử chat của người dùng từ Supabase và định dạng nó cho frontend
    
    Args:
        user_id (str): ID người dùng
    
    Returns:
        dict: Dictionary chứa lịch sử chat theo định dạng phù hợp với frontend
    """
    try:
        # Lấy danh sách chat
        chats = get_chats(user_id)
        
        # Dictionary để lưu kết quả
        result = {}
        
        # Duyệt qua từng chat
        for chat in chats:
            chat_id = chat['id']
            
            # Lấy danh sách tin nhắn của chat này
            messages = get_messages(chat_id, user_id)
            
            # Định dạng chat để phù hợp với frontend
            result[chat_id] = {
                'id': chat_id,
                'title': chat.get('title', 'Cuộc trò chuyện mới'),
                'messages': messages or [],
                'timestamp': chat.get('created_at') or chat.get('updated_at', '')
            }
        
        return result
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử chat đầy đủ: {str(e)}")
        return {}

# Hàm đồng bộ tất cả chat từ localStorage lên Supabase
def sync_all_chats_with_local_storage(user_id, chat_history_json):
    """
    Đồng bộ hóa tất cả chat từ localStorage lên Supabase
    
    Args:
        user_id (str): ID người dùng
        chat_history_json (str): JSON chuỗi chứa lịch sử chat từ localStorage
    
    Returns:
        tuple: (số chat đồng bộ thành công, danh sách lỗi)
    """
    try:
        # Parse JSON
        chat_history = json.loads(chat_history_json)
        
        # Đếm số chat đồng bộ thành công
        synced_count = 0
        errors = []
        
        # Duyệt qua từng chat
        for chat_id, chat_data in chat_history.items():
            try:
                # Kiểm tra xem chat đã tồn tại chưa
                existing_chat = get_chat(chat_id, user_id)
                
                if existing_chat:
                    # Cập nhật tiêu đề nếu cần
                    update_chat_title(chat_id, user_id, chat_data.get('title', 'Cuộc trò chuyện không có tiêu đề'))
                else:
                    # Tạo chat mới với ID từ localStorage nếu có thể
                    new_chat = create_chat(user_id, chat_data.get('title', 'Cuộc trò chuyện không có tiêu đề'))
                    if not new_chat:
                        errors.append(f"Không thể tạo chat {chat_id}")
                        continue
                
                # Thêm tin nhắn vào chat
                if 'messages' in chat_data and isinstance(chat_data['messages'], list):
                    for msg in chat_data['messages']:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if content:
                            add_message(chat_id, user_id, role, content)
                
                synced_count += 1
            except Exception as e:
                errors.append(f"Lỗi khi đồng bộ chat {chat_id}: {str(e)}")
        
        return synced_count, errors
    except Exception as e:
        logger.error(f"Lỗi khi đồng bộ tất cả chat: {str(e)}")
        return 0, [f"Lỗi tổng thể: {str(e)}"] 