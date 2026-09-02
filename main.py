"""
AMS2 AI Livery Studio — Application Entry Point.
Automated custom racing livery creation and installation for Automobilista 2.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
import qdarktheme

from src.ui.main_window import MainWindow
from src.utils.constants import APP_NAME, APP_ORGANIZATION, APP_VERSION


def main():
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORGANIZATION)

    # Apply modern dark theme
    additional_qss = """
        QGroupBox {
            font-weight: bold;
            font-size: 11px;
            color: #d0d4dc;
            border: 1px solid #2e323b;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 6px;
            background-color: #1f2229;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 4px;
            background-color: #1f2229;
        }
    """

    if hasattr(qdarktheme, "setup_theme"):
        qdarktheme.setup_theme(
            theme="dark",
            custom_colors={"primary": "#0078d4", "background": "#181a1f"},
            additional_qss=additional_qss,
        )
    elif hasattr(qdarktheme, "load_stylesheet"):
        base_style = qdarktheme.load_stylesheet("dark")
        app.setStyleSheet(base_style + "\n" + additional_qss)
    else:
        app.setStyleSheet(additional_qss)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
