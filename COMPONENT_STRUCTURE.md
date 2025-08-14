# Component Structure Documentation

## 📁 Cấu trúc Components

```
templates/
├── index.html              # Main template
└── components/
    └── header.html         # Header component (Navbar + Sidebar)
```

## 🧩 Header Component

### HTML Template: `templates/components/header.html`
- Chứa navbar sticky (w3-top)
- Responsive sidebar cho mobile
- Navigation links (ABOUT, TEAM, WORK)
- Logo placeholder
- Hamburger menu icon

### JavaScript: `static/js/header.js`
- Class `HeaderComponent` để quản lý header
- Scroll effects (navbar opacity)
- Active menu highlighting
- Navbar toggle methods

## ✨ Sử dụng Component

### Trong template:
```html
<!-- Include Header Component -->
{% include 'components/header.html' %}
```

### Trong JavaScript:
```javascript
// Highlight active menu item
headerComponent.setActiveMenuItem('about');

// Toggle navbar visibility
headerComponent.toggleNavbar(false); // Hide
headerComponent.toggleNavbar(true);  // Show
```

## 🔧 Tính năng Header Component

1. **Sticky Navigation**: Luôn hiển thị ở top khi scroll
2. **Responsive**: Tự động chuyển sang hamburger menu trên mobile
3. **Smooth Scroll**: Hỗ trợ scroll mượt đến các section
4. **Active State**: Highlight menu item hiện tại
5. **Scroll Effects**: Opacity effect khi scroll
6. **Mobile Sidebar**: Overlay menu cho thiết bị nhỏ

## 🚀 Mở rộng Component

### Thêm menu item mới:
1. Thêm vào `header.html`:
```html
<a href="#contact" class="w3-bar-item w3-button">CONTACT</a>
```

2. Thêm vào sidebar:
```html
<a href="#contact" onclick="w3_close()" class="w3-bar-item w3-button">CONTACT</a>
```

### Thêm logo/brand:
```html
<a href="#home" class="w3-bar-item w3-button w3-wide">
    <img src="/static/img/logo.png" alt="Logo" style="height:30px;">
</a>
```

### Custom styling:
Thêm CSS trong `index.html` hoặc tạo file CSS riêng cho component.

## 📈 Lợi ích

- **Reusable**: Có thể dùng lại trong nhiều page
- **Maintainable**: Dễ sửa đổi và bảo trì
- **Scalable**: Dễ thêm tính năng mới
- **Organized**: Code structure rõ ràng
- **Testable**: Có thể test riêng từng component
