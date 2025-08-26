// Header Component JavaScript
class HeaderComponent {
    constructor() {
        this.navbar = null;
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.navbar = document.getElementById("myNavbar");
            this.initScrollEffect();
            this.initNavbarBehavior();
        });
    }

    // Thêm hiệu ứng scroll cho navbar (tùy chọn)
    initScrollEffect() {
        if (this.navbar) {
            window.addEventListener('scroll', () => {
                if (window.scrollY > 50) {
                    this.navbar.classList.add('w3-opacity');
                } else {
                    this.navbar.classList.remove('w3-opacity');
                }
            });
        }
    }

    // Khởi tạo các behavior khác cho navbar
    initNavbarBehavior() {
        // Có thể thêm các chức năng khác như:
        // - Active menu highlighting
        // - Smooth scroll enhancement
        // - Mobile menu improvements
        console.log('Header component initialized');
    }

    // Method để highlight menu item hiện tại
    setActiveMenuItem(sectionId) {
        const menuItems = document.querySelectorAll('#myNavbar a[href^="#"]');
        menuItems.forEach(item => {
            item.classList.remove('w3-theme');
            if (item.getAttribute('href') === `#${sectionId}`) {
                item.classList.add('w3-theme');
            }
        });
    }

    // Method để ẩn/hiện navbar (tùy chọn)
    toggleNavbar(show = true) {
        if (this.navbar) {
            const topDiv = this.navbar.parentElement;
            topDiv.style.display = show ? 'block' : 'none';
        }
    }
}

// Initialize header component
const headerComponent = new HeaderComponent();
