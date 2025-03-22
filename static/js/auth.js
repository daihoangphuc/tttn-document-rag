/**
 * File JavaScript cho các chức năng xác thực người dùng
 */

// Hàm thực thi khi DOM đã sẵn sàng
document.addEventListener('DOMContentLoaded', function() {
    // Khởi tạo các chức năng xác thực
    initAuthFunctions();
});

/**
 * Khởi tạo các chức năng xác thực người dùng
 */
function initAuthFunctions() {
    // Xử lý sự kiện form đăng nhập
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', function(event) {
            if (!validateLoginForm()) {
                event.preventDefault();
            }
        });
    }

    // Xử lý sự kiện form đăng ký
    const registerForm = document.getElementById('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', function(event) {
            if (!validateRegisterForm()) {
                event.preventDefault();
            }
        });
    }

    // Khởi tạo sự kiện hiển thị/ẩn mật khẩu
    initPasswordToggles();
}

/**
 * Kiểm tra biểu mẫu đăng nhập
 * @returns {boolean} Kết quả kiểm tra hợp lệ
 */
function validateLoginForm() {
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;

    // Kiểm tra email
    if (!email) {
        showAlert('Vui lòng nhập địa chỉ email', 'error');
        return false;
    }

    // Kiểm tra định dạng email
    if (!isValidEmail(email)) {
        showAlert('Địa chỉ email không hợp lệ', 'error');
        return false;
    }

    // Kiểm tra mật khẩu
    if (!password) {
        showAlert('Vui lòng nhập mật khẩu', 'error');
        return false;
    }

    return true;
}

/**
 * Kiểm tra biểu mẫu đăng ký
 * @returns {boolean} Kết quả kiểm tra hợp lệ
 */
function validateRegisterForm() {
    const username = document.getElementById('username').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm_password').value;
    const terms = document.getElementById('terms');

    // Kiểm tra tên người dùng
    if (!username) {
        showAlert('Vui lòng nhập tên người dùng', 'error');
        return false;
    }

    // Kiểm tra email
    if (!email) {
        showAlert('Vui lòng nhập địa chỉ email', 'error');
        return false;
    }

    // Kiểm tra định dạng email
    if (!isValidEmail(email)) {
        showAlert('Địa chỉ email không hợp lệ', 'error');
        return false;
    }

    // Kiểm tra mật khẩu
    if (!password) {
        showAlert('Vui lòng nhập mật khẩu', 'error');
        return false;
    }

    // Kiểm tra độ dài mật khẩu
    if (password.length < 6) {
        showAlert('Mật khẩu phải có ít nhất 6 ký tự', 'error');
        return false;
    }

    // Kiểm tra mật khẩu xác nhận
    if (password !== confirmPassword) {
        showAlert('Mật khẩu xác nhận không khớp', 'error');
        return false;
    }

    // Kiểm tra điều khoản sử dụng
    if (terms && !terms.checked) {
        showAlert('Vui lòng đồng ý với điều khoản sử dụng', 'error');
        return false;
    }

    return true;
}

/**
 * Kiểm tra định dạng email hợp lệ
 * @param {string} email - Địa chỉ email cần kiểm tra
 * @returns {boolean} Kết quả kiểm tra
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Hiển thị thông báo cho người dùng
 * @param {string} message - Nội dung thông báo
 * @param {string} type - Loại thông báo (success, error, warning, info)
 */
function showAlert(message, type = 'info') {
    // Tìm container thông báo hoặc tạo mới
    let alertContainer = document.querySelector('.alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.className = 'alert-container';
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '20px';
        alertContainer.style.right = '20px';
        alertContainer.style.zIndex = '1000';
        document.body.appendChild(alertContainer);
    }

    // Tạo thông báo
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        <span class="font-medium">${message}</span>
        <button type="button" class="close" aria-label="Close" 
            style="position: absolute; right: 10px; top: 10px; cursor: pointer; background: none; border: none;">
            <span aria-hidden="true">&times;</span>
        </button>
    `;
    alertContainer.appendChild(alert);

    // Thêm sự kiện đóng
    alert.querySelector('.close').addEventListener('click', function() {
        alert.remove();
    });

    // Tự động đóng sau 5 giây
    setTimeout(() => {
        if (alert && alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

/**
 * Khởi tạo các nút hiển thị/ẩn mật khẩu
 */
function initPasswordToggles() {
    const toggleButtons = document.querySelectorAll('.password-toggle');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const passwordInput = document.getElementById(targetId);
            
            if (passwordInput.type === 'password') {
                passwordInput.type = 'text';
                this.querySelector('i').classList.remove('fa-eye');
                this.querySelector('i').classList.add('fa-eye-slash');
            } else {
                passwordInput.type = 'password';
                this.querySelector('i').classList.remove('fa-eye-slash');
                this.querySelector('i').classList.add('fa-eye');
            }
        });
    });
}

/**
 * Thay đổi ngôn ngữ
 * @param {string} lang - Mã ngôn ngữ (vi, en, etc.)
 */
function changeLanguage(lang) {
    // Lưu ngôn ngữ vào localStorage
    localStorage.setItem('language', lang);
    // Thực hiện thay đổi ngôn ngữ tại đây (ví dụ: tải lại trang với tham số ngôn ngữ)
    window.location.search = `?lang=${lang}`;
}

/**
 * Chuyển đổi giữa chế độ sáng và tối
 */
function toggleDarkMode() {
    const htmlElement = document.documentElement;
    
    if (htmlElement.classList.contains('dark')) {
        htmlElement.classList.remove('dark');
        localStorage.setItem('darkMode', 'false');
    } else {
        htmlElement.classList.add('dark');
        localStorage.setItem('darkMode', 'true');
    }
}

// Kiểm tra và áp dụng chế độ tối từ localStorage khi trang tải
(function() {
    const darkMode = localStorage.getItem('darkMode');
    
    if (darkMode === 'true') {
        document.documentElement.classList.add('dark');
    } else if (darkMode === null) {
        // Kiểm tra nếu người dùng ưa thích chế độ tối của hệ thống
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('darkMode', 'true');
        }
    }
})(); 