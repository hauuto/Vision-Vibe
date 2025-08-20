# Image Processing Web App

## 📌 Mô tả
Web app dùng **Flask** và **OpenCV** để:
- Upload ảnh
- Xử lý ảnh (grayscale, blur, edge detection)
- Hiển thị ảnh gốc và ảnh đã xử lý
- Giao diện responsive với **HTML + Bootstrap**

## 🚀 Cài đặt
1. Cài đặt Python 3.8+
2. Tạo môi trường ảo và cài đặt thư viện:
   ```
   run.bat
   ```
   or
   ```
   .\run.bat
   ```

## 📁 Cấu trúc thư mục
```
image_app/
├── app.py               # File chính chạy Flask
├── requirements.txt     # Thư viện cần thiết
├── static/              # File tĩnh
│   ├── css/
│   ├── js/
│   ├── img/
│   ├── uploads/        # Ảnh upload
│   └── processed/      # Ảnh đã xử lý
├── templates/           # HTML templates
└── utils/              # Logic xử lý Python
```

## 📁 Cấu trúc JavaScript
```
static/js/
├── data.js        # Mô tả hình ảnh
├── modal.js       # Modal gallery
├── navigation.js  # Responsive menu
├── api.js         # Giao tiếp Flask API
├── header.js      # Quản lý header
└── main.js        # Khởi tạo chính
```

### 🔧 Chi tiết JS modules
- **data.js**: Quản lý mô tả ảnh
- **modal.js**: Class `ModalGallery` (show/close modal, animations)
- **navigation.js**: Class `Navigation` (toggle/close sidebar, mobile support)
- **api.js**: Class `FlaskAPI` (async API calls, error handling)
- **header.js**: Class `HeaderComponent` (scroll effects, active menu)
- **main.js**: Entry point, debug, global config


### 🚀 Thêm tính năng JS
1. Tạo file JS mới trong `static/js/`
2. Tạo class cho tính năng
3. Import vào `index.html`
4. Khởi tạo trong `main.js`

Ví dụ:
```javascript
class NewFeature {
    constructor() { this.init(); }
    init() { /* Code */ }
}
const newFeature = new NewFeature();
```

## 📁 Cấu trúc Components
```
templates/
├── index.html
└── components/
    └── header.html  # Navbar + Sidebar
```

### 🧩 Header Component
- **header.html**: Sticky navbar, responsive sidebar, nav links, logo, hamburger menu
- **header.js**: Class `HeaderComponent` (scroll effects, active menu, toggle navbar)

### ✨ Sử dụng Header
```html
{% include 'components/header.html' %}
```
```javascript
headerComponent.setActiveMenuItem('about');
headerComponent.toggleNavbar(true); // Show
```

### 🔧 Tính năng Header
- Sticky nav, responsive, smooth scroll
- Active menu highlight, scroll effects
- Mobile sidebar

### 🚀 Mở rộng Header
- Thêm menu item:
```html
<a href="#contact" class="w3-bar-item w3-button">CONTACT</a>
```
- Thêm logo:
```html
<a href="#home" class="w3-bar-item w3-button w3-wide">
    <img src="/static/img/logo.png" alt="Logo" style="height:30px;">
</a>
```
- Custom CSS trong `index.html` hoặc file riêng
