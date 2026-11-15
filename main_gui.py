"""
GUI-приложение для анализа отзывов Steam.
Запуск:
    pip install PyQt6 markdown
    python main_gui.py
"""

import os
import sqlite3
import sys
import traceback
import uuid
from datetime import datetime
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QTextCursor, QFont, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QTextBrowser,
    QTextEdit, QVBoxLayout, QWidget
)

try:
    import markdown as md_lib
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# ==========================================================
# Константы и стили
# ==========================================================
DB_PATH = "local_storage.db"

C_BG_DEEP      = "#171a21"   # Глубокий фон
C_BG_PANEL     = "#1b2838"   # Панели (темные сообщения ИИ)
C_BG_BLOCK     = "#2a475e"   # Блоки (светлые сообщения пользователя)
C_BORDER       = "#3d5a73"   # Границы (сделана заметнее)
C_TEXT         = "#e8e8e8"   # Основной текст
C_TEXT_DIM     = "#8f9baa"   # Второстепенный текст
C_AI           = "#00a4ff"   # Яркий Steam-синий
C_ERROR        = "#ff5555"
C_ERROR_BG     = "#3a1a1a"

# ==========================================================
# Кастомная кнопка отправки
# ==========================================================
class SendButton(QPushButton):
    """Кнопка с идеально отрисованной стрелкой по центру."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        is_hovered = self.underMouse()
        is_enabled = self.isEnabled()

        # Фон и обводка круга
        if is_enabled:
            if is_hovered:
                painter.setBrush(QColor(C_AI))
                painter.setPen(Qt.PenStyle.NoPen)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(C_AI), 2))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(C_BORDER), 2))

        painter.drawEllipse(2, 2, 36, 36)

        # Цвет стрелки
        if is_enabled:
            arrow_color = QColor(C_BG_DEEP) if is_hovered else QColor(C_AI)
        else:
            arrow_color = QColor(C_TEXT_DIM)

        # Рисуем стрелку (толстую и короткую)
        painter.setPen(QPen(arrow_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        
        cx, cy = 20, 20
        # Вертикальная линия
        painter.drawLine(cx, cy - 7, cx, cy + 7)
        # Наконечники
        painter.drawLine(cx, cy - 7, cx - 6, cy - 1)
        painter.drawLine(cx, cy - 7, cx + 6, cy - 1)

# ==========================================================
# Поля ввода и потоки
# ==========================================================
class ChatInput(QTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
        else:
            super().keyPressEvent(event)

class InitWorker(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            from dialog_manager import DialogManager
            
            class GuiDialogManager:
                def __init__(self, db_path):
                    self.db_path = db_path
                    self._init_db_session()
                    self._dm = DialogManager(db_path=db_path)
                    self.session_id = str(uuid.uuid4()) 

                def _init_db_session(self):
                    with sqlite3.connect(self.db_path) as conn:
                        try:
                            conn.execute("ALTER TABLE journal ADD COLUMN session_id TEXT")
                        except sqlite3.OperationalError:
                            pass

                def handle_request(self, query, history=None):
                    original_save = self._dm._save_to_journal
                    def patched_save(user_query, game, response):
                        try:
                            with sqlite3.connect(self.db_path) as conn:
                                conn.execute(
                                    "INSERT INTO journal (session_id, timestamp, user_query, game, response) VALUES (?, ?, ?, ?, ?)",
                                    (self.session_id, datetime.now().isoformat(), user_query, game, response)
                                )
                        except Exception as e:
                            import logging
                            logging.getLogger(__name__).error(f"Ошибка записи: {e}")
                            
                    self._dm._save_to_journal = patched_save
                    try:
                        return self._dm.handle_request(query, history=history)
                    finally:
                        self._dm._save_to_journal = original_save

                @property
                def ollama(self):
                    return self._dm.ollama

                def close(self):
                    """Освобождает ресурсы DialogManager при закрытии приложения."""
                    try:
                        self._dm.close()
                    except Exception:
                        pass

            dm = GuiDialogManager(db_path=DB_PATH)
            self.ready.emit(dm)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")

class QueryWorker(QThread):
    finished_ok = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(self, dm, query: str, history: list):
        super().__init__()
        self.dm = dm
        self.query = query
        self.history = history

    def run(self):
        try:
            response = self.dm.handle_request(self.query, history=self.history)
            self.finished_ok.emit(self.query, response)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")

# ==========================================================
# Главное окно
# ==========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор отзывов Steam")
        self.resize(1280, 850)
        self.setMinimumSize(950, 650)

        self.dm = None
        self.history = []
        self.current_worker = None
        self.init_worker = None

        self._loading_base = ""
        self._loading_dots = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self._animate_loading)

        self._build_ui()
        self._apply_style()
        self._show_welcome()
        self._init_backend()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(16, 16, 16, 16)
        sb_layout.setSpacing(12)

        title = QLabel("Диалоги")
        title.setObjectName("AppTitle")
        sb_layout.addWidget(title)

        self.new_chat_btn = QPushButton("Новый диалог")
        self.new_chat_btn.setFixedHeight(36)
        self.new_chat_btn.clicked.connect(self._new_chat)
        sb_layout.addWidget(self.new_chat_btn)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_click)
        sb_layout.addWidget(self.history_list, 1)

        root.addWidget(sidebar)

        # Main Area
        main = QWidget()
        main.setObjectName("MainArea")
        mv = QVBoxLayout(main)
        mv.setContentsMargins(24, 24, 24, 24)
        mv.setSpacing(16)

        self.chat_view = QTextBrowser()
        self.chat_view.setObjectName("ChatView")
        self.chat_view.setOpenExternalLinks(False)
        mv.addWidget(self.chat_view, 1)

        self.loading_label = QLabel("")
        self.loading_label.setObjectName("LoadingLabel")
        self.loading_label.setVisible(False)
        mv.addWidget(self.loading_label)

        # Input Area
        input_container = QFrame()
        input_container.setObjectName("InputContainer")
        ic_layout = QHBoxLayout(input_container)
        ic_layout.setContentsMargins(12, 12, 12, 12)
        ic_layout.setSpacing(12)

        self.input_field = ChatInput()
        self.input_field.setPlaceholderText(
            "Напишите запрос... (Enter — отправить, Shift+Enter — новая строка)"
        )
        self.input_field.setFixedHeight(60)
        self.input_field.setEnabled(False)
        self.input_field.send_requested.connect(self._send_message)
        ic_layout.addWidget(self.input_field, 1)

        self.send_btn = SendButton()
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._send_message)
        ic_layout.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignBottom)

        mv.addWidget(input_container)
        root.addWidget(main, 1)

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DEEP}; }}
            QWidget {{ font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 15px; }}
            
            QWidget#Sidebar {{
                background-color: {C_BG_PANEL};
                border-right: 1px solid {C_BORDER};
            }}
            QWidget#MainArea {{ background-color: {C_BG_DEEP}; }}
            
            QLabel#AppTitle {{
                color: {C_AI};
                font-size: 22px;
                font-weight: bold;
                padding: 4px 0px;
            }}
            QLabel#LoadingLabel {{
                color: {C_AI};
                font-style: italic;
                font-size: 14px;
                padding: 4px 8px;
            }}
            
            QListWidget {{
                background-color: transparent;
                border: none;
                color: {C_TEXT};
                outline: none;
            }}
            QListWidget::item {{
                padding: 12px 10px;
                border-bottom: 1px solid {C_BORDER};
                border-radius: 6px;
                margin-bottom: 2px;
            }}
            QListWidget::item:hover {{ background-color: {C_BG_BLOCK}; }}
            QListWidget::item:selected {{
                background-color: {C_BG_BLOCK};
                color: {C_AI};
                border-left: 3px solid {C_AI};
            }}
            
            QPushButton {{
                background-color: {C_BG_BLOCK};
                color: {C_TEXT};
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {C_AI};
                color: {C_BG_DEEP};
            }}
            QPushButton:disabled {{
                color: {C_TEXT_DIM};
                background-color: {C_BG_PANEL};
            }}
            
            /* Чат и поля ввода */
            QTextBrowser#ChatView {{
                background-color: {C_BG_DEEP};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
                padding: 8px;
                color: {C_TEXT};
                font-size: 16px;
            }}
            QFrame#InputContainer {{
                background-color: {C_BG_PANEL};
                border: 1px solid {C_BORDER};
                border-radius: 12px;
            }}
            QTextEdit {{
                background-color: transparent;
                color: {C_TEXT};
                border: none;
                font-size: 16px;
                padding: 4px;
                selection-background-color: {C_AI};
            }}
            
            /* Скроллбар: полностью прозрачный фон, круглый ползунок */
            QScrollBar:vertical {{
                background: transparent;
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BG_BLOCK};
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {C_AI}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)

    def _init_backend(self):
        self._start_loading("Инициализация нейросети. Загрузка моделей может занять некоторое время")
        self.init_worker = InitWorker()
        self.init_worker.ready.connect(self._on_backend_ready)
        self.init_worker.failed.connect(self._on_backend_failed)
        self.init_worker.start()

    def _on_backend_ready(self, dm):
        self.dm = dm
        self._stop_loading()
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        self._load_journal()

    def _on_backend_failed(self, err):
        self._stop_loading()
        self._append_chat_html(self._error_html(f"Ошибка бэкенда:\n{err}"))

    def _start_loading(self, text: str):
        self._loading_base = text
        self._loading_dots = 0
        self.loading_label.setText(text)
        self.loading_label.setVisible(True)
        self._loading_timer.start(500)

    def _animate_loading(self):
        self._loading_dots = (self._loading_dots + 1) % 4
        self.loading_label.setText(self._loading_base + "." * self._loading_dots)

    def _stop_loading(self):
        self._loading_timer.stop()
        self.loading_label.setVisible(False)

    # ------------------------------------------------------
    # Рендеринг (вложенные таблицы для идеальной ширины и отступов)
    # ------------------------------------------------------
    def _append_chat_html(self, html: str):
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(html)
        cursor.insertBlock() # Создает новый абзац, отделяя от следующего сообщения
        
        sb = self.chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _escape(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace("\n", "<br>"))

    def _user_html(self, text: str) -> str:
        # Внешняя таблица задает 60% пустоты слева. Внутренняя - аккуратный пузырь.
        return f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 12px;">
            <tr>
                <td width="60%"></td>
                <td width="40%" align="right">
                    <table align="right" cellpadding="10" cellspacing="0" style="background-color: {C_BG_BLOCK}; border-radius: 12px;">
                        <tr>
                            <td align="left" style="color: {C_TEXT}; font-size: 15px; line-height: 1.3;">
                                {self._escape(text)}
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        """

    def _ai_html(self, markdown_text: str) -> str:
        if HAS_MARKDOWN:
            content = md_lib.markdown(markdown_text, extensions=["tables", "fenced_code", "nl2br"])
        else:
            content = self._escape(markdown_text)

        # Внешняя таблица задает 20% пустоты справа. Внутренняя - пузырь с рамкой.
        return f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 12px;">
            <tr>
                <td width="80%" align="left">
                    <table align="left" cellpadding="12" cellspacing="0" style="background-color: {C_BG_PANEL}; border: 1px solid {C_BORDER}; border-radius: 12px;">
                        <tr>
                            <td align="left" style="color: {C_TEXT}; font-size: 15px; line-height: 1.4;">
                                {content}
                            </td>
                        </tr>
                    </table>
                </td>
                <td width="20%"></td>
            </tr>
        </table>
        """

    def _error_html(self, text: str) -> str:
        return f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 12px;">
            <tr>
                <td width="80%" align="left">
                    <table align="left" cellpadding="12" cellspacing="0" style="background-color: {C_ERROR_BG}; border: 1px solid {C_ERROR}; border-radius: 12px;">
                        <tr>
                            <td align="left" style="color: #ffaaaa; font-size: 15px;">
                                {self._escape(text)}
                            </td>
                        </tr>
                    </table>
                </td>
                <td width="20%"></td>
            </tr>
        </table>
        """

    def _show_welcome(self):
        text = (
            "Здравствуйте! Напишите название игры — я соберу отзывы, выделю ключевые "
            "особенности и сформирую сводку. Дополнительные параметры (язык, период, "
            "характеристики ПК) распознаются автоматически.\n\n"
            "Например: «Составь сводку по Elden Ring за последний месяц»."
        )
        self._append_chat_html(self._ai_html(text))

    # ------------------------------------------------------
    # Логика отправки
    # ------------------------------------------------------
    def _send_message(self):
        if self.dm is None or self.current_worker is not None:
            return

        query = self.input_field.toPlainText().strip()
        if not query:
            return

        self.input_field.clear()
        self._append_chat_html(self._user_html(query))
        hist_snapshot = list(self.history)

        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.history_list.setEnabled(False)
        self.new_chat_btn.setEnabled(False)

        self._start_loading("Нейросеть анализирует отзывы, это может занять некоторое время")

        self.current_worker = QueryWorker(self.dm, query, hist_snapshot)
        self.current_worker.finished_ok.connect(self._on_query_ok)
        self.current_worker.failed.connect(self._on_query_err)
        self.current_worker.finished.connect(self._on_query_done)
        self.current_worker.start()

    def _on_query_ok(self, query: str, response: str):
        self._stop_loading()
        self._append_chat_html(self._ai_html(response))
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        self._load_journal()

    def _on_query_err(self, err: str):
        self._stop_loading()
        self._append_chat_html(self._error_html(err))

    def _on_query_done(self):
        self.current_worker = None
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.history_list.setEnabled(True)
        self.new_chat_btn.setEnabled(True)
        self.input_field.setFocus()

    # ------------------------------------------------------
    # Журнал (сессии)
    # ------------------------------------------------------
    def _load_journal(self):
        self.history_list.clear()
        if not os.path.exists(DB_PATH):
            return
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT session_id, MIN(timestamp) as ts, game, 
                           (SELECT user_query FROM journal WHERE session_id = j.session_id ORDER BY id ASC LIMIT 1) as first_query
                    FROM journal j
                    GROUP BY session_id
                    ORDER BY ts DESC LIMIT 100
                """).fetchall()
        except Exception as e:
            print(f"[GUI] Ошибка чтения журнала: {e}")
            return

        for session_id, ts, game, first_query in rows:
            label = self._format_sidebar_label(first_query, game)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            item.setToolTip(f"{ts}\n\n{first_query}")
            self.history_list.addItem(item)

    @staticmethod
    def _format_sidebar_label(query: str, game: str | None) -> str:
        q = (query or "").strip().replace("\n", " ")
        if len(q) > 32:
            q = q[:32] + "..."
        game_str = game if game and game != "None" else ""
        if game_str:
            return f"[{game_str}] {q}"
        return q or "(без названия)"

    def _on_history_click(self, item: QListWidgetItem):
        if self.current_worker is not None:
            return
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if not session_id:
            return

        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT user_query, game, response FROM journal WHERE session_id=? ORDER BY id ASC",
                    (session_id,)
                ).fetchall()
        except Exception as e:
            print(f"[GUI] Ошибка загрузки сессии: {e}")
            return
            
        if not rows:
            return

        self.chat_view.clear()
        self.history = []
        
        for query, game, response in rows:
            self._append_chat_html(self._user_html(query))
            self._append_chat_html(self._ai_html(response))
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": response})
            
        self.dm.session_id = session_id

    def _new_chat(self):
        if self.current_worker is not None:
            return
        self.chat_view.clear()
        self.history = []
        if self.dm:
            self.dm.session_id = str(uuid.uuid4())
        self._show_welcome()

    def closeEvent(self, event):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.quit()
            self.current_worker.wait(2000)
        if self.init_worker and self.init_worker.isRunning():
            self.init_worker.quit()
            self.init_worker.wait(2000)

        # Выгружаем модель из памяти Ollama и (если сервер был запущен нами)
        # полностью останавливаем его через минуту после закрытия приложения.
        if self.dm:
            try:
                if hasattr(self.dm, "close"):
                    self.dm.close()
                elif hasattr(self.dm, "ollama"):
                    self.dm.ollama.unload_model()
            except Exception:
                pass

            ollama_mgr = getattr(self.dm, "ollama", None)
            proc = ollama_mgr.server_process() if hasattr(ollama_mgr, "server_process") else None
            if proc is not None:
                QTimer.singleShot(60_000, lambda: self._shutdown_ollama(ollama_mgr))

        event.accept()

    @staticmethod
    def _shutdown_ollama(ollama_mgr):
        """Через минуту после выхода: выгрузить модель и остановить `ollama serve`."""
        try:
            ollama_mgr.stop_server()
        except Exception:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())