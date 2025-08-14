# Image Processing Web App

## 📌 Mô tả dự án
Web app này được xây dựng bằng **Flask** và **OpenCV** cho phép người dùng:

- Upload ảnh từ máy tính
- Xử lý ảnh trực tiếp trên web, ví dụ:
  - Chuyển sang grayscale
  - Làm mờ (blur)
  - Phát hiện cạnh (edge detection)
- Hiển thị cả **ảnh gốc** và **ảnh đã xử lý**
- Có thể mở rộng thêm nhiều chức năng xử lý ảnh khác

Web app sử dụng **HTML + Bootstrap** để tạo giao diện thân thiện, responsive.

---

## 📁 Cấu trúc thư mục

- **image_app/**
  - `app.py` → File chính chạy Flask
  - `requirements.txt` → Liệt kê các thư viện cần thiết
  - **static/** → Chứa file tĩnh (CSS, JS, ảnh)
    - **css/**
    - **js/**
    - **img/**
    - **uploads/** → Lưu ảnh người dùng upload
    - **processed/** → Lưu ảnh đã xử lý
  - **templates/** → HTML templates
  - **utils/** → Chứa các file Python xử lý logic

