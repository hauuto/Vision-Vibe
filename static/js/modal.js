// Modal Image Gallery Functions
class ModalGallery {
    constructor() {
        this.initEventListeners();
    }

    // Show modal with image and description
    showModal(element) {
        const modal = document.getElementById("modal01");
        const img = document.getElementById("img01");
        const modalText = document.getElementById("modal-text");
        const title = document.getElementById("modal-title");
        const description = document.getElementById("modal-description");

        // Get filename from src
        const filename = element.src.split('/').pop();
        const imageData = typeof imageDescriptions !== 'undefined' ? imageDescriptions[filename] : null;

        // Set content
        img.src = element.src;
        title.textContent = imageData ? imageData.title : (element.alt || '');
        description.innerHTML = imageData ? imageData.description : '';

        // Show modal using Tailwind classes
        modal.classList.remove('hidden');
        modal.classList.add('flex');

        // Reset animations
        img.classList.remove("show");
        modalText.classList.remove("show");

        // Trigger animations after a brief delay
        setTimeout(() => {
            img.classList.add("show");
            modalText.classList.add("show");
        }, 50);
    }

    // Close modal and reset animations
    closeModal() {
        const modal = document.getElementById("modal01");
        const img = document.getElementById("img01");
        const modalText = document.getElementById("modal-text");

        img.classList.remove("show");
        modalText.classList.remove("show");

        setTimeout(() => {
            modal.classList.add('hidden');
            modal.classList.remove('flex');
        }, 300);
    }

    // Initialize event listeners
    initEventListeners() {
        // Add click event to modal background
        document.addEventListener('DOMContentLoaded', () => {
            const modal = document.getElementById("modal01");
            if (modal) {
                modal.addEventListener("click", (e) => {
                    if (e.target === modal) {
                        this.closeModal();
                    }
                });
            }

            // Add Escape key handler to close modal when open
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' || e.key === 'Esc') {
                    const m = document.getElementById('modal01');
                    if (m && m.classList.contains('flex')) {
                        this.closeModal();
                    }
                }
            });
        });
    }
}

// Global function for onclick attribute (backward compatibility)
function onClick(element) {
    modalGallery.showModal(element);
}

// Initialize modal gallery
const modalGallery = new ModalGallery();
