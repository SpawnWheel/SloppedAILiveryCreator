"""Photoshop / GIMP-style dark chrome for the paint editor (and the main window tabs)."""

ACCENT = "#2f8ff0"
ACCENT_DARK = "#1c6fc4"
BG = "#232427"          # app background behind docks
PANEL = "#2b2d31"       # dock / panel body
PANEL_ALT = "#26282b"
HEADER = "#1f2023"      # dock title bars, option bar
BORDER = "#141517"
TEXT = "#d7d9dd"
TEXT_DIM = "#8c9099"
FIELD = "#1b1c1f"
PASTEBOARD = "#1c1d20"

EDITOR_QSS = f"""
QMainWindow#PaintEditor, QWidget#EditorRoot {{ background: {BG}; }}
QMainWindow::separator {{ background: {BORDER}; width: 3px; height: 3px; }}
QMainWindow::separator:hover {{ background: {ACCENT}; }}

QWidget {{ color: {TEXT}; font-size: 11px; }}
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel[dim="true"] {{ color: {TEXT_DIM}; }}
QLabel[section="true"] {{ color: {TEXT_DIM}; font-size: 10px; font-weight: bold; letter-spacing: 1px; }}

/* ---- docks ---- */
QDockWidget {{ color: {TEXT_DIM}; font-size: 10px; font-weight: bold; }}
QDockWidget::title {{ background: {HEADER}; padding: 3px 8px; border-bottom: 1px solid {BORDER};
                       text-align: left; }}
QDockWidget::close-button, QDockWidget::float-button {{ background: transparent; border: none; padding: 0px;
                       icon-size: 10px; }}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{ background: #3a3d43; border-radius: 2px; }}
QDockWidget > QWidget {{ background: {PANEL}; }}
QTabBar {{ background: {HEADER}; }}
QTabBar::tab {{ background: {HEADER}; color: {TEXT_DIM}; padding: 5px 12px; border: none; min-width: 64px;
                border-right: 1px solid {BORDER}; font-size: 10px; font-weight: bold; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

/* ---- controls ---- */
QPushButton {{ background: #36393f; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 3px;
               padding: 4px 9px; }}
QPushButton:hover {{ background: #41454c; }}
QPushButton:pressed {{ background: #2a2c30; }}
QPushButton:checked {{ background: {ACCENT_DARK}; color: white; }}
QPushButton:disabled {{ color: #5d6168; background: #2c2e32; }}
QPushButton[primary="true"] {{ background: {ACCENT}; color: white; font-weight: bold; border: 1px solid {ACCENT_DARK};
                               padding: 5px 14px; }}
QPushButton[primary="true"]:hover {{ background: #4aa0f5; }}
QPushButton[apply="true"] {{ background: #2e8b47; color: white; font-weight: bold; border: 1px solid #1e6533;
                             padding: 5px 14px; }}
QPushButton[apply="true"]:hover {{ background: #36a254; }}
QPushButton[apply="true"]:disabled {{ background: #2c2e32; color: #5d6168; border-color: {BORDER}; }}
QToolButton {{ background: transparent; color: {TEXT}; border: 1px solid transparent; border-radius: 3px;
               padding: 3px; }}
QToolButton:hover {{ background: #3a3d43; border-color: #45484f; }}
QToolButton:checked {{ background: {ACCENT_DARK}; border-color: {ACCENT}; }}
QToolButton:pressed {{ background: #2a2c30; }}
QToolButton:disabled {{ color: #5d6168; }}
QToolButton::menu-indicator {{ image: none; }}

QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox, QLineEdit, QPlainTextEdit {{
    background: {FIELD}; border: 1px solid {BORDER}; border-radius: 2px; color: {TEXT}; padding: 2px 4px;
    selection-background-color: {ACCENT}; }}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; selection-background-color: {ACCENT}; }}
QCheckBox {{ color: {TEXT}; spacing: 5px; }}
QCheckBox::indicator {{ width: 12px; height: 12px; border: 1px solid #5a5e66; background: {FIELD}; border-radius: 2px; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {FIELD}; border: 1px solid {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #5b7fa8; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #e6e7ea; border: 1px solid #111; width: 10px; height: 10px;
                              margin: -5px 0; border-radius: 5px; }}

QListWidget {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; outline: none; }}
QListWidget::item {{ padding: 3px 4px; border-bottom: 1px solid #222326; color: {TEXT}; }}
QListWidget::item:selected {{ background: #34507a; color: white; }}
QListWidget::item:hover:!selected {{ background: #313338; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {PANEL_ALT}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #4a4d54; min-height: 24px; border-radius: 4px; margin: 2px; }}
QScrollBar:horizontal {{ background: {PANEL_ALT}; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #4a4d54; min-width: 24px; border-radius: 4px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

/* ---- option bar / status bar ---- */
QToolBar#OptionsBar {{ background: {HEADER}; border: none; border-bottom: 1px solid {BORDER}; spacing: 6px;
                       padding: 3px 6px; }}
QToolBar#OptionsBar QLabel {{ color: {TEXT_DIM}; }}
QToolBar::separator {{ background: #3a3d43; width: 1px; margin: 4px 6px; }}
QStatusBar {{ background: {HEADER}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ color: {TEXT_DIM}; padding: 0 6px; }}
QStatusBar::item {{ border: none; }}
QMenuBar {{ background: {HEADER}; color: {TEXT}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 4px 9px; background: transparent; }}
QMenuBar::item:selected {{ background: #3a3d43; }}
QMenu {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 0; }}
QMenu::item {{ padding: 5px 26px 5px 22px; }}
QMenu::item:selected {{ background: {ACCENT_DARK}; color: white; }}
QMenu::item:disabled {{ color: #5d6168; }}
QMenu::separator {{ height: 1px; background: #3a3d43; margin: 4px 8px; }}
QToolTip {{ background: #111214; color: {TEXT}; border: 1px solid #3a3d43; padding: 4px; }}

/* ---- floating toolbox ---- */
QFrame#Toolbox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; }}
QFrame#ToolboxGrip {{ background: {HEADER}; border-top-left-radius: 4px; border-top-right-radius: 4px;
                       border-bottom: 1px solid {BORDER}; }}
QFrame#Toolbox QToolButton {{ padding: 4px; }}
QFrame#EmptyState {{ background: rgba(32, 33, 36, 235); border: 1px solid #3a3d43; border-radius: 8px; }}
"""

# Main-window document tabs (the AI Generation / Paint Editor / Wheels / Install bar)
MAIN_TABS_QSS = f"""
QTabWidget#MainTabs::pane {{ border: none; border-top: 1px solid {BORDER}; top: -1px; background: {BG}; }}
QTabWidget#MainTabs > QTabBar {{ background: {HEADER}; }}
QTabWidget#MainTabs > QTabBar::tab {{ background: {HEADER}; color: {TEXT_DIM}; padding: 8px 20px 7px 20px;
    border: none; border-right: 1px solid {BORDER}; border-top: 2px solid transparent;
    font-size: 12px; font-weight: bold; min-width: 120px; }}
QTabWidget#MainTabs > QTabBar::tab:selected {{ background: {BG}; color: #ffffff; border-top: 2px solid {ACCENT}; }}
QTabWidget#MainTabs > QTabBar::tab:hover:!selected {{ color: {TEXT}; background: #2a2c30; }}
"""
