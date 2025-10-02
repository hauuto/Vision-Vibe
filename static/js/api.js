// Flask API Functions
class FlaskAPI {
    constructor() {
        this.initHelloButton();
    }

    // Initialize Hello World button
    initHelloButton() {
        document.addEventListener('DOMContentLoaded', () => {
            const helloBtn = document.getElementById("helloBtn");
            if (helloBtn) {
                helloBtn.addEventListener("click", () => {
                    this.callHelloAPI();
                });
            }
        });
    }

    // Call Flask Hello API
    async callHelloAPI() {
        try {
            const response = await fetch('/api/hello/');
            const data = await response.json();

            this.showMessage(data.message, 'success');
        } catch (error) {
            console.error('Error:', error);
            this.showMessage("Error connecting to server!", 'error');
        }
    }

    // Show message to user
    showMessage(message, type = 'success') {
        const messageDiv = document.getElementById("helloMessage");
        const messageText = document.getElementById("messageText");

        if (messageDiv && messageText) {
            messageText.textContent = message;

            // Tailwind-based styles
            if (type === 'error') {
                messageText.className = "text-white text-lg"; // error as white text for contrast on dark bg
            } else {
                messageText.className = "text-[#7fa4a4] text-lg"; // success/info with accent color
            }

            messageDiv.style.display = "block";

            // Hide message after 3 seconds
            setTimeout(() => {
                messageDiv.style.display = "none";
            }, 3000);
        }
    }
}

// Initialize Flask API
const flaskAPI = new FlaskAPI();
