#!/usr/bin/env python3
"""
VisionVibe - Advanced Photo Editor with Live Filter Processing

A comprehensive image processing application with real-time filter previews
and an intuitive, modern interface design.

Author: VisionVibe Team
Version: 3.0
"""

import sys
import os
import warnings
from typing import Optional, Tuple, Any
from functools import partial

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QGroupBox, QScrollArea,
    QFrame, QFileDialog, QMessageBox, QStatusBar, QMenuBar, QStyle,
    QSizePolicy
)
from PySide6.QtCore import Qt, QPoint, Signal, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QAction, QWheelEvent, QMouseEvent, QPainter

# Suppress OpenCV warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'

# Import custom image processing modules with fallback
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vision_core.myself_apply import ImageFilters, NoiseGenerator, ImageEnhancements
except ImportError:
    print("Warning: Custom vision_core module not found, using OpenCV fallbacks")


    class ImageFilters:
        """Fallback image filters using OpenCV"""

        @staticmethod
        def mean_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
            """Apply mean blur filter"""
            return cv2.blur(image, (kernel_size, kernel_size))

        @staticmethod
        def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
            """Apply Gaussian blur filter"""
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        @staticmethod
        def median_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
            """Apply median filter"""
            return cv2.medianBlur(image, kernel_size)

        @staticmethod
        def bilateral_filter(image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
            """Apply bilateral filter"""
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        @staticmethod
        def sharpen_filter(image: np.ndarray) -> np.ndarray:
            """Apply sharpening filter"""
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            return cv2.filter2D(image, -1, kernel)

        @staticmethod
        def unsharp_masking(image: np.ndarray, sigma: float, strength: float) -> np.ndarray:
            """Apply unsharp masking"""
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            mask = image.astype(np.float32) - blurred.astype(np.float32)
            result = image.astype(np.float32) + strength * mask
            return np.clip(result, 0, 255).astype(np.uint8)

        @staticmethod
        def canny_edge(image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
            """Apply Canny edge detection"""
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        @staticmethod
        def sobel_magnitude(image: np.ndarray) -> np.ndarray:
            """Apply Sobel edge detection"""
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
            magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
            return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


    class NoiseGenerator:
        """Fallback noise generator"""

        @staticmethod
        def add_gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
            """Add Gaussian noise to image"""
            noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
            return cv2.add(image, noise)


    class ImageEnhancements:
        """Fallback image enhancements"""

        @staticmethod
        def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
            """Adjust image brightness"""
            return cv2.convertScaleAbs(image, alpha=1, beta=value)

        @staticmethod
        def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
            """Adjust image contrast"""
            return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


class ZoomableImageViewer(QLabel):
    """Enhanced image viewer with proper zoom functionality"""

    coordinates_changed = Signal(QPoint)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the zoomable image viewer"""
        super().__init__(parent)
        self._setup_ui()
        self._setup_properties()

    def _setup_ui(self) -> None:
        """Setup UI properties"""
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #5a6570;
                border-radius: 10px;
                background-color: #3a4248;
                color: #888888;
                font-size: 14pt;
            }
        """)
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _setup_properties(self) -> None:
        """Initialize viewer properties"""
        self.original_pixmap: Optional[QPixmap] = None
        self.scaled_pixmap: Optional[QPixmap] = None
        self.scale_factor: float = 1.0
        self.min_scale: float = 0.1
        self.max_scale: float = 10.0
        self.pan_offset: QPoint = QPoint(0, 0)
        self.last_mouse_pos: QPoint = QPoint()
        self.is_panning: bool = False

        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

    def set_image(self, image: Optional[np.ndarray]) -> None:
        """Set image and reset view"""
        if image is None:
            self.original_pixmap = None
            self.scaled_pixmap = None
            self.clear()
            self.setText("Drop an image here or use Open Image\n\nSupported: JPG, PNG, BMP, TIFF, WebP")
            return

        try:
            # Validate and convert image
            if image.size == 0:
                raise ValueError("Empty image provided")

            # Convert to RGB format
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image[:, :, :3]
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            height, width, channels = rgb_image.shape
            bytes_per_line = channels * width

            qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            if qt_image.isNull():
                raise ValueError("Failed to create QImage")

            self.original_pixmap = QPixmap.fromImage(qt_image)
            if self.original_pixmap.isNull():
                raise ValueError("Failed to create QPixmap")

            # Reset view and fit to window
            self.scale_factor = 1.0
            self.pan_offset = QPoint(0, 0)
            self.fit_to_window()

        except Exception as e:
            print(f"Error setting image: {e}")
            self.setText(f"Error loading image: {str(e)}")

    def fit_to_window(self) -> None:
        """Fit image to window maintaining aspect ratio"""
        if not self.original_pixmap:
            return

        available_size = self.size()
        margin = 20  # Leave some margin
        available_size = QSize(available_size.width() - margin, available_size.height() - margin)

        if available_size.width() <= 0 or available_size.height() <= 0:
            return

        # Calculate scale to fit
        scale_x = available_size.width() / self.original_pixmap.width()
        scale_y = available_size.height() / self.original_pixmap.height()
        self.scale_factor = min(scale_x, scale_y)

        # Ensure scale is within bounds
        self.scale_factor = max(self.min_scale, min(self.max_scale, self.scale_factor))

        self.pan_offset = QPoint(0, 0)
        self._update_display()

    def zoom_in(self) -> None:
        """Zoom in by 25%"""
        self._scale_image(1.25)

    def zoom_out(self) -> None:
        """Zoom out by 20%"""
        self._scale_image(0.8)

    def _scale_image(self, factor: float) -> None:
        """Scale image by factor around center"""
        if not self.original_pixmap:
            return

        old_scale = self.scale_factor
        self.scale_factor *= factor
        self.scale_factor = max(self.min_scale, min(self.max_scale, self.scale_factor))

        if abs(self.scale_factor - old_scale) < 0.001:
            return  # No significant change

        # Keep image centered when zooming
        self.pan_offset = QPoint(0, 0)

        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed image"""
        if not self.original_pixmap:
            return

        # Create scaled pixmap
        scaled_size = self.original_pixmap.size() * self.scale_factor
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Create display pixmap with proper positioning
        display_size = self.size()
        display_pixmap = QPixmap(display_size)
        display_pixmap.fill(Qt.transparent)

        painter = QPainter(display_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate position (center + pan offset)
        image_pos = QPoint(
            (display_size.width() - self.scaled_pixmap.width()) // 2 + self.pan_offset.x(),
            (display_size.height() - self.scaled_pixmap.height()) // 2 + self.pan_offset.y()
        )

        painter.drawPixmap(image_pos, self.scaled_pixmap)
        painter.end()

        self.setPixmap(display_pixmap)

    def resizeEvent(self, event) -> None:
        """Handle resize events"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self._update_display()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming"""
        if self.original_pixmap:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start panning"""
        if event.button() == Qt.LeftButton and self.original_pixmap:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle panning and coordinate tracking"""
        if self.is_panning and self.original_pixmap:
            delta = event.position().toPoint() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.position().toPoint()
            self._update_display()

        # Emit coordinates for status bar
        self.coordinates_changed.emit(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Stop panning"""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)


class FilterHistoryManager:
    """Manage filter history for undo/redo functionality"""

    def __init__(self, max_history: int = 15) -> None:
        """Initialize history manager"""
        self.history: list[dict[str, Any]] = []
        self.current_index: int = -1
        self.max_history: int = max_history

    def add_state(self, image: np.ndarray, operation_name: str = "") -> None:
        """Add new state to history"""
        if image is None or image.size == 0:
            return

        # Remove any states after current index
        self.history = self.history[:self.current_index + 1]

        # Add new state
        self.history.append({
            'image': image.copy(),
            'operation': operation_name
        })

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1

    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current_index > 0

    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current_index < len(self.history) - 1

    def undo(self) -> Optional[np.ndarray]:
        """Undo to previous state"""
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index]['image'].copy()
        return None

    def redo(self) -> Optional[np.ndarray]:
        """Redo to next state"""
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index]['image'].copy()
        return None

    def get_current_operation(self) -> str:
        """Get current operation name"""
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index]['operation']
        return ""

    def clear(self) -> None:
        """Clear history"""
        self.history = []
        self.current_index = -1


class VisionVibeEditor(QMainWindow):
    """Main VisionVibe photo editor application"""

    def __init__(self) -> None:
        """Initialize VisionVibe editor"""
        super().__init__()
        self._setup_properties()
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._setup_timers()

    def _setup_properties(self) -> None:
        """Initialize application properties"""
        self.setWindowTitle("VisionVibe - Advanced Photo Editor")
        self.setMinimumSize(1400, 900)

        # Image management
        self.original_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None

        # Filter history
        self.history = FilterHistoryManager()

        # Live update timer
        self.live_update_timer = QTimer()
        self.live_update_timer.setSingleShot(True)
        self.live_update_timer.timeout.connect(self._apply_live_filters)

        # UI components
        self.image_viewer: Optional[ZoomableImageViewer] = None
        self.save_btn: Optional[QPushButton] = None
        self.save_as_btn: Optional[QPushButton] = None
        self.undo_btn: Optional[QPushButton] = None
        self.redo_btn: Optional[QPushButton] = None

        # Current filter states
        self.current_filters = {
            'blur_type': 'None',
            'blur_kernel': 5,
            'blur_sigma': 1.0,
            'blur_space': 75,
            'sharpen_type': 'None',
            'sharpen_sigma': 1.0,
            'sharpen_strength': 1.5,
            'edge_type': 'None',
            'edge_low': 50,
            'edge_high': 100,
            'brightness': 0,
            'contrast': 1.0
        }

    def _setup_timers(self) -> None:
        """Setup timers for live updates"""
        pass  # Timer already initialized in _setup_properties

    def _setup_ui(self) -> None:
        """Setup main user interface"""
        self._setup_styles()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Image panel (left side)
        image_panel = self._create_image_panel()

        # Control panel (right side)
        control_panel = self._create_control_panel()

        # Add to main layout
        main_layout.addWidget(image_panel, 3)
        main_layout.addWidget(control_panel, 1)

    def _setup_styles(self) -> None:
        """Setup VisionVibe styles with original color scheme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b3038;
                color: white;
            }
            QWidget {
                background-color: #2b3038;
                color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #5a6570;
                border-radius: 10px;
                margin: 10px 0px;
                padding-top: 15px;
                background-color: #3a4248;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #ffffff;
                font-size: 11pt;
                font-weight: bold;
            }
            QPushButton {
                background-color: #4a5560;
                border: 1px solid #5a6570;
                border-radius: 8px;
                padding: 10px 15px;
                color: white;
                font-weight: 500;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #5a6570;
                border-color: #6a7580;
            }
            QPushButton:pressed {
                background-color: #3a4248;
            }
            QPushButton:disabled {
                background-color: #35393f;
                color: #6a6a6a;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #4a5560;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                border: 2px solid #007acc;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1e90ff;
            }
            QComboBox {
                background-color: #4a5560;
                border: 1px solid #5a6570;
                border-radius: 8px;
                padding: 8px 15px;
                min-width: 150px;
                color: white;
                font-weight: 500;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox QAbstractItemView {
                background-color: #4a5560;
                border: 1px solid #5a6570;
                selection-background-color: #007acc;
            }
            QLabel {
                color: #e0e0e0;
                padding: 4px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #4a5560;
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #5a6570;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6a7580;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)

    def _create_image_panel(self) -> QWidget:
        """Create enhanced image display panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        # Toolbar
        toolbar = self._create_image_toolbar()
        layout.addWidget(toolbar)

        # Image viewer
        self.image_viewer = ZoomableImageViewer()
        self.image_viewer.coordinates_changed.connect(self._update_coordinates)
        layout.addWidget(self.image_viewer)

        return widget

    def _create_image_toolbar(self) -> QWidget:
        """Create image toolbar"""
        toolbar = QWidget()
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #4a5560;
                border-radius: 10px;
                padding: 5px;
            }
        """)

        layout = QHBoxLayout(toolbar)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 10, 15, 10)

        # File operations
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self.open_image)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)

        self.save_as_btn = QPushButton("Save As")
        self.save_as_btn.clicked.connect(self.save_as_image)
        self.save_as_btn.setEnabled(False)

        # View controls
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(lambda: self.image_viewer.zoom_in() if self.image_viewer else None)

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(lambda: self.image_viewer.zoom_out() if self.image_viewer else None)

        fit_btn = QPushButton("Fit Window")
        fit_btn.clicked.connect(lambda: self.image_viewer.fit_to_window() if self.image_viewer else None)

        # History controls
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)

        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        reset_btn.clicked.connect(self.reset_image)

        # Add widgets to layout
        widgets = [
            open_btn, self.save_btn, self.save_as_btn,
            self._create_separator(),
            zoom_in_btn, zoom_out_btn, fit_btn,
            self._create_separator(),
            self.undo_btn, self.redo_btn, reset_btn
        ]

        for widget in widgets:
            layout.addWidget(widget)

        layout.addStretch()
        return toolbar

    def _create_separator(self) -> QFrame:
        """Create a modern vertical separator"""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("""
            QFrame {
                color: rgba(255, 255, 255, 0.2);
                background: rgba(255, 255, 255, 0.1);
                max-width: 2px;
                border-radius: 1px;
            }
        """)
        return separator

    def _create_control_panel(self) -> QScrollArea:
        """Create modern control panel"""
        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(420)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.05),
                    stop:1 rgba(255, 255, 255, 0.02));
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)

        # Main control widget
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("VisionVibe")
        title.setStyleSheet("""
            QLabel {
                font-size: 18pt;
                font-weight: bold;
                color: #ffffff;
                padding: 15px;
                background-color: #4a5560;
                border-radius: 10px;
                border: 2px solid #5a6570;
            }
        """)
        title.setAlignment(Qt.AlignCenter)

        # Create filter groups
        blur_group = self._create_blur_group()
        sharpen_group = self._create_sharpen_group()
        edge_group = self._create_edge_group()
        basic_group = self._create_basic_adjustments_group()

        # Add widgets
        for widget in [title, blur_group, sharpen_group, edge_group, basic_group]:
            layout.addWidget(widget)

        layout.addStretch()
        scroll_area.setWidget(control_widget)
        return scroll_area

    def _create_blur_group(self) -> QGroupBox:
        """Create live blur operations group"""
        group = QGroupBox("Blur & Smoothing")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)

        # Filter selection
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self.blur_combo = QComboBox()
        self.blur_combo.addItems(["None", "Mean Blur", "Gaussian Blur", "Median Filter", "Bilateral Filter"])
        self.blur_combo.currentTextChanged.connect(self._on_blur_type_changed)
        filter_layout.addWidget(self.blur_combo)
        layout.addLayout(filter_layout)

        # Parameters container
        self.blur_params = QWidget()
        self.blur_params_layout = QVBoxLayout(self.blur_params)
        self.blur_params_layout.setContentsMargins(0, 10, 0, 0)
        layout.addWidget(self.blur_params)

        self._update_blur_parameters()
        return group

    def _create_sharpen_group(self) -> QGroupBox:
        """Create live sharpening operations group"""
        group = QGroupBox("Sharpening & Enhancement")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)

        # Filter selection
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self.sharpen_combo = QComboBox()
        self.sharpen_combo.addItems(["None", "Basic Sharpen", "Unsharp Masking", "High-pass Filter"])
        self.sharpen_combo.currentTextChanged.connect(self._on_sharpen_type_changed)
        filter_layout.addWidget(self.sharpen_combo)
        layout.addLayout(filter_layout)

        # Parameters container
        self.sharpen_params = QWidget()
        self.sharpen_params_layout = QVBoxLayout(self.sharpen_params)
        self.sharpen_params_layout.setContentsMargins(0, 10, 0, 0)
        layout.addWidget(self.sharpen_params)

        self._update_sharpen_parameters()
        return group

    def _create_edge_group(self) -> QGroupBox:
        """Create live edge detection group"""
        group = QGroupBox("Edge Detection")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))

        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["None", "Canny Edge", "Sobel", "Laplacian", "Prewitt"])
        self.edge_combo.currentTextChanged.connect(self._on_edge_type_changed)
        method_layout.addWidget(self.edge_combo)
        layout.addLayout(method_layout)

        # Parameters container
        self.edge_params = QWidget()
        self.edge_params_layout = QVBoxLayout(self.edge_params)
        self.edge_params_layout.setContentsMargins(0, 10, 0, 0)
        layout.addWidget(self.edge_params)

        self._update_edge_parameters()
        return group

    def _create_basic_adjustments_group(self) -> QGroupBox:
        """Create live basic adjustments group"""
        group = QGroupBox("Basic Adjustments")
        layout = QVBoxLayout(group)
        layout.setSpacing(15)

        # Brightness
        bright_layout = self._create_parameter_layout(
            "Brightness", -100, 100, 0, self._on_brightness_changed
        )
        layout.addLayout(bright_layout)

        # Contrast
        contrast_layout = self._create_parameter_layout(
            "Contrast", 1, 30, 10, self._on_contrast_changed,
            value_transform=lambda v: f"{v / 10:.1f}"
        )
        layout.addLayout(contrast_layout)

        return group

    def _create_parameter_layout(self, name: str, min_val: int, max_val: int,
                                 default_val: int, callback, value_transform=None):
        """Create a parameter layout with slider and label"""
        layout = QVBoxLayout()

        # Header with name and value
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel(name + ":"))

        value_label = QLabel(str(default_val) if not value_transform else value_transform(default_val))
        value_label.setStyleSheet("color: #4fc3f7; font-weight: 600;")
        header_layout.addWidget(value_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)

        def on_change(value):
            if value_transform:
                value_label.setText(value_transform(value))
            else:
                value_label.setText(str(value))
            callback(value)

        slider.valueChanged.connect(on_change)
        layout.addWidget(slider)

        # Store references
        setattr(self, f"{name.lower().replace(' ', '_')}_slider", slider)
        setattr(self, f"{name.lower().replace(' ', '_')}_label", value_label)

        return layout

    def _update_blur_parameters(self) -> None:
        """Update blur parameters based on selected filter"""
        self._clear_layout(self.blur_params_layout)

        filter_type = self.blur_combo.currentText() if self.blur_combo else "None"

        if filter_type == "None":
            return

        if filter_type in ["Mean Blur", "Gaussian Blur", "Median Filter"]:
            # Kernel size
            layout = self._create_parameter_layout(
                "Kernel Size", 3, 21, 5, self._on_blur_kernel_changed,
                value_transform=lambda v: str(v if v % 2 == 1 else v + 1)
            )
            self.blur_params_layout.addLayout(layout)

        if filter_type in ["Gaussian Blur", "Bilateral Filter"]:
            # Sigma
            layout = self._create_parameter_layout(
                "Sigma", 1, 50, 10, self._on_blur_sigma_changed,
                value_transform=lambda v: f"{v / 10:.1f}"
            )
            self.blur_params_layout.addLayout(layout)

        if filter_type == "Bilateral Filter":
            # Sigma Space
            layout = self._create_parameter_layout(
                "Sigma Space", 10, 200, 75, self._on_blur_space_changed
            )
            self.blur_params_layout.addLayout(layout)

    def _update_sharpen_parameters(self) -> None:
        """Update sharpen parameters based on selected filter"""
        self._clear_layout(self.sharpen_params_layout)

        filter_type = self.sharpen_combo.currentText() if self.sharpen_combo else "None"

        if filter_type == "None":
            return

        if filter_type == "Unsharp Masking":
            # Sigma
            sigma_layout = self._create_parameter_layout(
                "Sigma", 1, 50, 10, self._on_sharpen_sigma_changed,
                value_transform=lambda v: f"{v / 10:.1f}"
            )
            self.sharpen_params_layout.addLayout(sigma_layout)

            # Strength
            strength_layout = self._create_parameter_layout(
                "Strength", 1, 30, 15, self._on_sharpen_strength_changed,
                value_transform=lambda v: f"{v / 10:.1f}"
            )
            self.sharpen_params_layout.addLayout(strength_layout)

    def _update_edge_parameters(self) -> None:
        """Update edge detection parameters based on selected method"""
        self._clear_layout(self.edge_params_layout)

        method = self.edge_combo.currentText() if self.edge_combo else "None"

        if method == "None":
            return

        if method == "Canny Edge":
            # Low threshold
            low_layout = self._create_parameter_layout(
                "Low Threshold", 1, 255, 50, self._on_edge_low_changed
            )
            self.edge_params_layout.addLayout(low_layout)

            # High threshold
            high_layout = self._create_parameter_layout(
                "High Threshold", 1, 255, 100, self._on_edge_high_changed
            )
            self.edge_params_layout.addLayout(high_layout)

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        """Clear all widgets from layout"""
        for i in reversed(range(layout.count())):
            child = layout.itemAt(i)
            if child and child.widget():
                child.widget().deleteLater()
            elif child and child.layout():
                self._clear_layout(child.layout())

    # Live update callbacks
    def _on_blur_type_changed(self, filter_type: str) -> None:
        """Handle blur type change"""
        self.current_filters['blur_type'] = filter_type
        self._update_blur_parameters()
        self._trigger_live_update()

    def _on_blur_kernel_changed(self, value: int) -> None:
        """Handle blur kernel size change"""
        # Ensure odd kernel size
        if value % 2 == 0:
            value += 1
        self.current_filters['blur_kernel'] = value
        self._trigger_live_update()

    def _on_blur_sigma_changed(self, value: int) -> None:
        """Handle blur sigma change"""
        self.current_filters['blur_sigma'] = value / 10.0
        self._trigger_live_update()

    def _on_blur_space_changed(self, value: int) -> None:
        """Handle blur space change"""
        self.current_filters['blur_space'] = value
        self._trigger_live_update()

    def _on_sharpen_type_changed(self, filter_type: str) -> None:
        """Handle sharpen type change"""
        self.current_filters['sharpen_type'] = filter_type
        self._update_sharpen_parameters()
        self._trigger_live_update()

    def _on_sharpen_sigma_changed(self, value: int) -> None:
        """Handle sharpen sigma change"""
        self.current_filters['sharpen_sigma'] = value / 10.0
        self._trigger_live_update()

    def _on_sharpen_strength_changed(self, value: int) -> None:
        """Handle sharpen strength change"""
        self.current_filters['sharpen_strength'] = value / 10.0
        self._trigger_live_update()

    def _on_edge_type_changed(self, method: str) -> None:
        """Handle edge type change"""
        self.current_filters['edge_type'] = method
        self._update_edge_parameters()
        self._trigger_live_update()

    def _on_edge_low_changed(self, value: int) -> None:
        """Handle edge low threshold change"""
        self.current_filters['edge_low'] = value
        self._trigger_live_update()

    def _on_edge_high_changed(self, value: int) -> None:
        """Handle edge high threshold change"""
        self.current_filters['edge_high'] = value
        self._trigger_live_update()

    def _on_brightness_changed(self, value: int) -> None:
        """Handle brightness change"""
        self.current_filters['brightness'] = value
        self._trigger_live_update()

    def _on_contrast_changed(self, value: int) -> None:
        """Handle contrast change"""
        self.current_filters['contrast'] = value / 10.0
        self._trigger_live_update()

    def _trigger_live_update(self) -> None:
        """Trigger live filter update with debouncing"""
        if self.current_image is None:
            return

        # Stop any existing timer and restart with 150ms delay
        self.live_update_timer.stop()
        self.live_update_timer.start(150)

    def _apply_live_filters(self) -> None:
        """Apply all current filters to the image"""
        if self.original_image is None:
            return

        try:
            # Start with original image
            result = self.original_image.copy()

            # Apply filters in order
            result = self._apply_blur_filter(result)
            result = self._apply_sharpen_filter(result)
            result = self._apply_edge_filter(result)
            result = self._apply_basic_adjustments(result)

            # Update display
            self.current_image = result
            if self.image_viewer:
                self.image_viewer.set_image(self.current_image)

        except Exception as e:
            print(f"Error applying filters: {e}")

    def _apply_blur_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply blur filter based on current settings"""
        filter_type = self.current_filters['blur_type']

        if filter_type == "None":
            return image

        try:
            if filter_type == "Mean Blur":
                kernel_size = self.current_filters['blur_kernel']
                return ImageFilters.mean_blur(image, kernel_size)

            elif filter_type == "Gaussian Blur":
                kernel_size = self.current_filters['blur_kernel']
                sigma = self.current_filters['blur_sigma']
                return ImageFilters.gaussian_blur(image, kernel_size, sigma)

            elif filter_type == "Median Filter":
                kernel_size = self.current_filters['blur_kernel']
                return ImageFilters.median_filter(image, kernel_size)

            elif filter_type == "Bilateral Filter":
                sigma_color = self.current_filters['blur_sigma'] * 50
                sigma_space = self.current_filters['blur_space']
                return ImageFilters.bilateral_filter(image, 9, sigma_color, sigma_space)

        except Exception as e:
            print(f"Error in blur filter: {e}")

        return image

    def _apply_sharpen_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpen filter based on current settings"""
        filter_type = self.current_filters['sharpen_type']

        if filter_type == "None":
            return image

        try:
            if filter_type == "Basic Sharpen":
                return ImageFilters.sharpen_filter(image)

            elif filter_type == "Unsharp Masking":
                sigma = self.current_filters['sharpen_sigma']
                strength = self.current_filters['sharpen_strength']
                return ImageFilters.unsharp_masking(image, sigma, strength)

            elif filter_type == "High-pass Filter":
                return ImageFilters.sharpen_filter(image)

        except Exception as e:
            print(f"Error in sharpen filter: {e}")

        return image

    def _apply_edge_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply edge detection based on current settings"""
        method = self.current_filters['edge_type']

        if method == "None":
            return image

        try:
            if method == "Canny Edge":
                low_thresh = self.current_filters['edge_low']
                high_thresh = self.current_filters['edge_high']
                return ImageFilters.canny_edge(image, low_thresh, high_thresh)

            elif method in ["Sobel", "Laplacian", "Prewitt"]:
                return ImageFilters.sobel_magnitude(image)

        except Exception as e:
            print(f"Error in edge filter: {e}")

        return image

    def _apply_basic_adjustments(self, image: np.ndarray) -> np.ndarray:
        """Apply basic brightness and contrast adjustments"""
        try:
            brightness = self.current_filters['brightness']
            contrast = self.current_filters['contrast']

            if brightness != 0:
                image = ImageEnhancements.adjust_brightness(image, brightness)

            if abs(contrast - 1.0) > 0.01:
                image = ImageEnhancements.adjust_contrast(image, contrast)

        except Exception as e:
            print(f"Error in basic adjustments: {e}")

        return image

    # File Operations
    def open_image(self) -> None:
        """Open image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image - VisionVibe", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)"
        )

        if not file_path:
            return

        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not load image")

            self.original_image = image
            self.current_image = image.copy()
            self.image_path = file_path

            # Clear history and add initial state
            self.history.clear()
            self.history.add_state(self.original_image, "Original")

            # Reset all filters
            self._reset_filters()

            # Update UI
            if self.image_viewer:
                self.image_viewer.set_image(self.current_image)

            if self.save_btn:
                self.save_btn.setEnabled(True)
            if self.save_as_btn:
                self.save_as_btn.setEnabled(True)

            self._update_history_buttons()
            self.statusBar().showMessage(f"Opened: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image: {str(e)}")

    def save_image(self) -> None:
        """Save current image to original path"""
        if self.current_image is None or self.image_path is None:
            self.save_as_image()
            return

        try:
            success = cv2.imwrite(self.image_path, self.current_image)
            if success:
                self.statusBar().showMessage(f"Saved: {os.path.basename(self.image_path)}")
            else:
                raise ValueError("Failed to save image")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def save_as_image(self) -> None:
        """Save current image with new name"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As - VisionVibe", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp);;All Files (*)"
        )

        if not file_path:
            return

        try:
            success = cv2.imwrite(file_path, self.current_image)
            if success:
                self.image_path = file_path
                self.statusBar().showMessage(f"Saved as: {os.path.basename(file_path)}")
            else:
                raise ValueError("Failed to save image")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    # History Operations
    def undo(self) -> None:
        """Undo last operation"""
        # Add current state before undoing
        if self.current_image is not None:
            self.history.add_state(self.current_image, "Current State")

        previous_image = self.history.undo()
        if previous_image is not None:
            self.original_image = previous_image
            self.current_image = previous_image.copy()
            self._reset_filters()

            if self.image_viewer:
                self.image_viewer.set_image(self.current_image)
            self._update_history_buttons()

            operation = self.history.get_current_operation()
            self.statusBar().showMessage(f"Undone to: {operation}")

    def redo(self) -> None:
        """Redo next operation"""
        next_image = self.history.redo()
        if next_image is not None:
            self.original_image = next_image
            self.current_image = next_image.copy()
            self._reset_filters()

            if self.image_viewer:
                self.image_viewer.set_image(self.current_image)
            self._update_history_buttons()

            operation = self.history.get_current_operation()
            self.statusBar().showMessage(f"Redone to: {operation}")

    def reset_image(self) -> None:
        """Reset image to original state"""
        if self.original_image is None:
            return

        # Add current state to history before reset
        if self.current_image is not None:
            self.history.add_state(self.current_image, "Before Reset")

        self.current_image = self.original_image.copy()

        if self.image_viewer:
            self.image_viewer.set_image(self.current_image)

        self._reset_filters()
        self._update_history_buttons()
        self.statusBar().showMessage("Reset to original image")

    def _reset_filters(self) -> None:
        """Reset all filter controls to default values"""
        # Reset combo boxes
        if self.blur_combo:
            self.blur_combo.setCurrentText("None")
        if self.sharpen_combo:
            self.sharpen_combo.setCurrentText("None")
        if self.edge_combo:
            self.edge_combo.setCurrentText("None")

        # Reset sliders
        if hasattr(self, 'brightness_slider'):
            self.brightness_slider.setValue(0)
        if hasattr(self, 'contrast_slider'):
            self.contrast_slider.setValue(10)

        # Reset filter states
        self.current_filters = {
            'blur_type': 'None',
            'blur_kernel': 5,
            'blur_sigma': 1.0,
            'blur_space': 75,
            'sharpen_type': 'None',
            'sharpen_sigma': 1.0,
            'sharpen_strength': 1.5,
            'edge_type': 'None',
            'edge_low': 50,
            'edge_high': 100,
            'brightness': 0,
            'contrast': 1.0
        }

    def _update_history_buttons(self) -> None:
        """Update undo/redo button states"""
        if self.undo_btn:
            self.undo_btn.setEnabled(self.history.can_undo())
        if self.redo_btn:
            self.redo_btn.setEnabled(self.history.can_redo())

    def _update_coordinates(self, pos: QPoint) -> None:
        """Update coordinates in status bar"""
        if self.current_image is not None:
            self.statusBar().showMessage(f"Position: ({pos.x()}, {pos.y()})")

    def _setup_menu_bar(self) -> None:
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_as_image)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        reset_action = QAction("Reset to Original", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.reset_image)
        edit_menu.addAction(reset_action)

        # View menu
        view_menu = menubar.addMenu("View")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(lambda: self.image_viewer.zoom_in() if self.image_viewer else None)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(lambda: self.image_viewer.zoom_out() if self.image_viewer else None)
        view_menu.addAction(zoom_out_action)

        fit_window_action = QAction("Fit to Window", self)
        fit_window_action.setShortcut("Ctrl+0")
        fit_window_action.triggered.connect(lambda: self.image_viewer.fit_to_window() if self.image_viewer else None)
        view_menu.addAction(fit_window_action)

    def _setup_status_bar(self) -> None:
        """Setup status bar"""
        self.statusBar().showMessage("Ready - Open an image to start editing with VisionVibe")


