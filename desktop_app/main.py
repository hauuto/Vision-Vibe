#!/usr/bin/env python3
"""
Photo Editor Application
Advanced image processing with separate operation groups for blur, sharpen, and edge detection
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ui.main_window import VisionVibeEditor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def main() -> None:
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application properties
    app.setApplicationName("VisionVibe")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("VisionVibe Team")

    # Create and show main window
    try:
        editor = VisionVibeEditor()
        editor.show()

        # Center window on screen
        screen = app.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            editor.move(
                (screen_geometry.width() - editor.width()) // 2,
                (screen_geometry.height() - editor.height()) // 2
            )

        sys.exit(app.exec())

    except Exception as e:
        print(f"Error starting VisionVibe: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
