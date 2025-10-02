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

    // Add scroll effect for navbar
    initScrollEffect() {
        if (this.navbar) {
            window.addEventListener('scroll', () => {
                if (window.scrollY > 50) {
                    this.navbar.classList.add('shadow-md');
                } else {
                    this.navbar.classList.remove('shadow-md');
                }
            });
        }
    }

    // Initialize extra behaviors
    initNavbarBehavior() {
        // You can add: active menu highlighting, smooth scroll enhancements, etc.
        console.log('Header component initialized');
    }

    // Highlight current menu item
    setActiveMenuItem(sectionId) {
        const menuItems = document.querySelectorAll('#myNavbar a[href^="#"]');
        menuItems.forEach(item => {
            item.classList.remove('text-[#7fa4a4]', 'font-semibold', 'border-b-2', 'border-[#7fa4a4]');
            if (item.getAttribute('href') === `#${sectionId}`) {
                item.classList.add('text-[#7fa4a4]', 'font-semibold');
            }
        });
    }

    // Show/Hide navbar (optional)
    toggleNavbar(show = true) {
        if (this.navbar) {
            const topDiv = this.navbar.parentElement;
            topDiv.style.display = show ? 'block' : 'none';
        }
    }
}

// Initialize header component
const headerComponent = new HeaderComponent();
