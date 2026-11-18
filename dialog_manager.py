import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from openai import OpenAI

from intent import IntentExtractor, OllamaManager, OLLAMA_MODEL_NAME
from data_manager import DataManager
from review_processor import ReviewProcessor

logger = logging.getLogger(__name__)

class DialogManager:
    """
    Оркестратор пайплайна (согласно Sequence Diagram).
    Управляет единым экземпляром OllamaManager (NFR-006), координирует 
    работу Intent, DataManager и ReviewProcessor, ведет журнал (FR-005).
    """
    def __init__(self, db_path: str = "local_storage.db"):
        logger.info("Инициализация DialogManager...")
        
        # 1. Инициализируем OllamaManager ОДИН РАЗ для всей сессии
        self.ollama = OllamaManager(model_name=OLLAMA_MODEL_NAME)
        
        # 2. Передаем его во все модули (Dependency Injection)
        self.intent_extractor = IntentExtractor(ollama_manager=self.ollama)
        self.data_manager = DataManager()
        self.review_processor = ReviewProcessor(ollama_manager=self.ollama)
        
        # 3. Инициализация локального хранилища (журнал диалогов)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Создает таблицу для журнала, если её нет (FR-005)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_query TEXT,
                    game TEXT,
                    response TEXT
                )
            """)

    def _save_to_journal(self, user_query: str, game: str, response: str):
        """Сохраняет сессию в локальную БД."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO journal (timestamp, user_query, game, response) VALUES (?, ?, ?, ?)",
                    (datetime.now().isoformat(), user_query, game, response)
                )
            logger.info("Сессия успешно сохранена в журнал.")
        except Exception as e:
            logger.error(f"Ошибка записи в журнал: {e}")

    def handle_request(self, user_query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Главная точка входа. Выполняет запрос пользователя от начала до конца.
        """
        logger.info(f"Новый запрос: '{user_query}'")
        
        try:
            # Шаг 1: Распознавание намерения (Intent & NLU)
            intent_data = self.intent_extractor.extract_intent(user_query, history=history)
            logger.debug(f"Извлечен intent: {intent_data}")

            text_intent = intent_data.get("intent")
            if text_intent in ("free_talk", "clarify", "unknown"):
                logger.info("Обработка текстового intent=%s", text_intent)
                final_response = self._generate_text_response(
                    intent_data,
                    user_query,
                    history=history,
                )
                self._save_to_journal(user_query, "None", final_response)
                return final_response

            # Шаг 2: Получение данных (Data Manager)
            # Внутри DataManager происходит проверка кэша, запрос к Steam API и семантическая фильтрация
            dm_output = self.data_manager.process(intent_data)

            # Если это просто болтовня или ошибка — обрабатываем сразу
            if dm_output.get("status") in ("text_only", "stub", "error"):
                final_response = dm_output.get("message", "Не удалось обработать запрос.")
                self._save_to_journal(user_query, "None", final_response)
                return final_response

            # Шаг 3: Анализ и суммаризация (Review Processing)
            # Передаем оригинальный запрос пользователя для контекста!
            logger.info("Запуск Map-Reduce суммаризации...")
            review_result = self.review_processor.process(dm_output, user_query=user_query)

            # Шаг 4: Форматирование ответа для UI
            final_markdown = self._format_response(review_result)
            
            # Шаг 5: Запись в журнал
            game_name = dm_output.get("game", "Неизвестная игра")
            self._save_to_journal(user_query, game_name, final_markdown)

            return final_markdown

        except Exception as e:
            logger.exception("Критическая ошибка в DialogManager")
            return f"Произошла внутренняя ошибка при обработке запроса: {e}"
            
        finally:
            # NFR-006: Выгружаем модель ТОЛЬКО в самом конце, после формирования ответа
            self.ollama.unload_model()

    def close(self) -> None:
        """Выгружает модель из памяти Ollama (вызывается при закрытии приложения)."""
        try:
            self.ollama.unload_model()
        except Exception as e:
            logger.warning(f"Не удалось выгрузить модель при закрытии: {e}")

    def _generate_text_response(
        self,
        intent_data: Dict[str, Any],
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        intent = intent_data.get("intent")
        missing_slots = intent_data.get("missing_slots") or []
        missing_text = ", ".join(str(slot) for slot in missing_slots)

        if intent == "clarify":
            system_prompt = (
                "Ты — помощник приложения для анализа отзывов Steam. "
                "Кратко и по-человечески попроси пользователя уточнить недостающие данные."
            )
            user_prompt = (
                f"Запрос пользователя: {user_query}\n"
                f"Не заполнены поля: {missing_text or 'название игры'}\n"
                "Задай только необходимые уточнения. Не анализируй отзывы и не выдумывай название игры."
            )
        elif intent == "unknown":
            system_prompt = (
                "Ты — помощник приложения для анализа отзывов Steam. "
                "Вежливо сообщи, что запрос не понят, и предложи конкретные примеры анализа отзывов."
            )
            user_prompt = (
                f"Запрос пользователя: {user_query}\n"
                "Ответь кратко на русском: объясни, что запрос нужно сформулировать яснее."
            )
        else:
            system_prompt = (
                "Ты — дружелюбный ассистент приложения для анализа отзывов Steam. "
                "Отвечай естественно на русском, кратко и без выдуманных данных."
            )
            user_prompt = (
                f"Запрос пользователя: {user_query}\n"
                "Это свободный разговор. Ответь на запрос пользователя по смыслу."
            )

        cloud = getattr(self.review_processor, "cloud", None)
        if cloud is not None:
            try:
                return cloud.chat_text(
                    system_prompt,
                    user_prompt,
                    history=history,
                )
            except Exception as e:
                logger.warning("Облачная генерация текстового ответа недоступна: %s", e)

        try:
            client = OpenAI(
                base_url=self.ollama.client_url,
                api_key="ollama",
                timeout=25.0,
            )
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-3:])
            messages.append({"role": "user", "content": user_prompt})
            response = client.chat.completions.create(
                model=self.ollama.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=700,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("Локальная генерация текстового ответа недоступна: %s", e)

        if intent == "clarify":
            return "Уточните, пожалуйста, название игры, отзывы которой нужно проанализировать."
        if intent == "unknown":
            return (
                "Я не смог понять запрос. Укажите название игры, например: "
                "«Составь сводку по отзывам Elden Ring за последний месяц»."
            )
        return "Здравствуйте! Чем могу помочь?"
    
    def _format_response(self, review_result: Dict[str, Any]) -> str:
        """Собирает финальный Markdown-ответ для UI."""
        if review_result.get("status") != "success":
            return f"Не удалось сформировать сводку. Причина: {review_result.get('message')}"

        output = []
        
        # 1. Предупреждения (FR-008, FR-009)
        warnings = review_result.get("warnings", [])
        if warnings:
            output.append("### ⚠️ Важные примечания")
            for w in warnings:
                output.append(f"- {w}")
            output.append("")

        # 2. Основная сводка (FR-010)
        output.append(review_result.get("summary", ""))

        # 3. Таблица (FR-006)
        table = review_result.get("table")
        if table:
            output.append("\n" + table)

        # 4. Мета-информация (количество проанализированных отзывов)
        meta = review_result.get("meta", {})
        if meta:
            raw_cnt = meta.get("raw_reviews_count", 0)
            final_cnt = meta.get("final_reviews_count", 0)
            output.append(f"\n---\n*Проанализировано {final_cnt} отзывов из {raw_cnt} загруженных.*")

        return "\n".join(output)