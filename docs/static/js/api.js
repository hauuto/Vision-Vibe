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
            const response = await fetch('/hello');
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

            // Add different styles based on message type
            if (type === 'error') {
                messageText.className = "w3-text-red w3-large";
            } else {
                messageText.className = "w3-text-yellow w3-large";
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
