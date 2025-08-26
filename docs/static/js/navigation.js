// Navigation Functions
class Navigation {
    constructor() {
        this.sidebar = null;
        this.initNavigation();
    }

    initNavigation() {
        document.addEventListener('DOMContentLoaded', () => {
            this.sidebar = document.getElementById("mySidebar");
        });
    }

    // Toggle between showing and hiding the sidebar when clicking the menu icon
    toggleSidebar() {
        if (this.sidebar) {
            if (this.sidebar.style.display === 'block') {
                this.sidebar.style.display = 'none';
            } else {
                this.sidebar.style.display = 'block';
            }
        }
    }

    // Close the sidebar with the close button
    closeSidebar() {
        if (this.sidebar) {
            this.sidebar.style.display = "none";
        }
    }
}

// Global functions for onclick attributes (backward compatibility)
function w3_open() {
    navigation.toggleSidebar();
}

function w3_close() {
    navigation.closeSidebar();
}

// Initialize navigation
const navigation = new Navigation();
