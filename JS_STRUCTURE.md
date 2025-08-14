# Vision Vibe - JavaScript Structure Documentation

## 📁 Cấu trúc JavaScript mới

```
static/js/
├── data.js        # Dữ liệu mô tả hình ảnh
├── modal.js       # Xử lý modal gallery
├── navigation.js  # Xử lý navigation
├── api.js         # Giao tiếp với Flask API
└── main.js        # File khởi tạo chính
```

## 🔧 Chi tiết các module

### data.js
- Chứa object `imageDescriptions` với thông tin chi tiết cho từng hình ảnh
- Dễ dàng thêm/sửa/xóa mô tả hình ảnh
- Tách biệt data khỏi logic code

### modal.js
- Class `ModalGallery` quản lý modal image gallery
- Methods: `showModal()`, `closeModal()`, `initEventListeners()`
- Animation slide với timeout và transition
- Backward compatibility với function `onClick()`

### navigation.js  
- Class `Navigation` quản lý responsive menu
- Methods: `toggleSidebar()`, `closeSidebar()`
- Support mobile hamburger menu
- Backward compatibility với `w3_open()` và `w3_close()`

### api.js
- Class `FlaskAPI` xử lý giao tiếp với Flask backend
- Methods: `callHelloAPI()`, `showMessage()`
- Async/await pattern cho API calls
- Error handling và user feedback

### main.js
- Entry point cho toàn bộ application
- Console log để debug
- Có thể thêm global configurations

## ✨ Ưu điểm của cấu trúc mới

- **Modular**: Tách biệt concerns, dễ maintain
- **Reusable**: Các class có thể tái sử dụng
- **Scalable**: Dễ dàng thêm tính năng mới
- **Clean**: Code sạch, organized
- **Modern**: Sử dụng ES6+ features
- **Compatible**: Vẫn support onclick attributes cũ

## 🚀 Cách thêm tính năng mới

1. Tạo file JavaScript mới trong `static/js/`
2. Tạo class cho tính năng đó
3. Import file vào `index.html` 
4. Initialize trong `main.js` nếu cần

Ví dụ:
```javascript
// static/js/newfeature.js
class NewFeature {
    constructor() {
        this.init();
    }
    
    init() {
        // Initialize code here
    }
}

const newFeature = new NewFeature();
```
