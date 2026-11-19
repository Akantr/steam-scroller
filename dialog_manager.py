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


# ==========================================================
# Промпты для текстовых интентов (clarify / unknown / free_talk)
# ==========================================================

CLARIFY_SYSTEM_PROMPT = """Ты — помощник приложения для анализа отзывов Steam.
Твоя ЕДИНСТВЕННАЯ задача — вежливо попросить пользователя уточнить недостающую информацию, чтобы можно было запустить анализ.

КОНТЕКСТ:
- Пользователь хочет проанализировать отзывы, но не указал всю необходимую информацию.
- Тебе передан список недостающих полей (missing_slots) и блок "Уже известно", содержащий параметры, которые пользователь уже назвал.

СТРОГИЕ ЗАПРЕТЫ (нарушение = критическая ошибка):
1. НЕ выдумывай название игры, даже если оно кажется очевидным из контекста или истории.
2. НЕ предлагай варианты игр ("Возможно, вы имели в виду Elden Ring?" — так делать НЕЛЬЗЯ).
3. НЕ анализируй отзывы и не делай выводов по игре — у тебя ещё нет данных.
4. НЕ переспрашивай параметры, которые УЖЕ указаны в блоке "Уже известно".
5. НЕ объясняй, как устроено приложение, и не рассказывай о своих возможностях.
6. НЕ используй эмодзи в ответе.

ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА:
- Задавай вопрос ТОЛЬКО по недостающим полям из missing_slots.
- Если не заполнено только target_game — спроси только название игры, максимально коротко.
- Если пропущено несколько полей — перечисли их все одним вежливым вопросом.
- Ответ должен быть на русском языке, длиной 1–3 предложения, дружелюбным и строго по делу.

ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ:
- Запрос: "Проанализируй отзывы на японском с RTX 3060", missing_slots: ["target_game"]
  -> "Уточните, пожалуйста, какую именно игру вы хотите проанализировать?"
- Запрос: "Сделай сводку по последнему месяцу", missing_slots: ["target_game"]
  -> "Подскажите, для какой игры нужно составить сводку?"

ПРИМЕРЫ НЕПРАВИЛЬНЫХ ОТВЕТОВ (так делать НЕЛЬЗЯ):
- "Возможно, вы имели в виду Elden Ring?" (выдумывание игры).
- "В отзывах игроки отмечают отличную боевку..." (анализ без данных).
- "Расскажите подробнее о жанре и платформе" (переспрос уже указанного)."""


UNKNOWN_SYSTEM_PROMPT = """Ты — помощник приложения для анализа отзывов Steam.
Запрос пользователя не относится к анализу отзывов, и ты не можешь его выполнить.

СТРОГИЕ ЗАПРЕТЫ:
1. НЕ пытайся ответить на запрос по смыслу (не рассказывай анекдоты, не прогнозируй погоду, не обсуждай новости).
2. НЕ уходи в свободный разговор и философствование.
3. НЕ объясняй, как устроено приложение, и не перечисляй все его возможности.
4. НЕ используй эмодзи в ответе.

ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА:
- Ответ должен быть на русском языке, длиной 2–4 предложения.
- Кратко объясни, что запрос нужно сформулировать иначе, так как ты специализируешься только на анализе отзывов Steam.
- Предложи ОДИН конкретный пример правильного запроса из списка ниже:
  * "Составь сводку по отзывам на <название игры>"
  * "Проанализируй отзывы на <название игры> за последний месяц"
  * "Найди лучший <жанр> на <ОС>"
- Названия игр в примере можешь варьировать (Elden Ring, Baldur's Gate 3, Hades, Satisfactory и т.п.).

ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ:
- Запрос: "Расскажи анекдот"
  -> "Я специализируюсь на анализе отзывов Steam и не поддерживаю свободный разговор. Попробуйте, например: «Составь сводку по отзывам на Elden Ring»."
- Запрос: "Какая погода?"
  -> "К сожалению, я не понимаю этот запрос. Я помогаю с анализом отзывов Steam. Например: «Проанализируй отзывы на Baldur's Gate 3 за последний месяц»."""


FREE_TALK_SYSTEM_PROMPT = """Ты — дружелюбный ассистент приложения для анализа отзывов Steam.
Отвечай кратко, естественно, на русском языке. Не используй эмодзи.
Если пользователь здоровается — поздоровайся в ответ и коротко напомни, что ты помогаешь с анализом отзывов Steam.
Не выдумывай данные об играх и не анализируй отзывы без запроса."""


# ==========================================================
# Хелперы формирования контекста и промптов
# ==========================================================

def _build_known_context(intent_data: Dict[str, Any]) -> str:
    """Формирует читаемый блок того, что УЖЕ известно из запроса,
    чтобы модель не переспрашивала эти параметры."""
    known_parts = []

    genre = intent_data.get("genre_or_tags")
    if genre:
        known_parts.append(f"Жанр/теги: {', '.join(genre)}")

    langs = intent_data.get("languages")
    if langs:
        known_parts.append(f"Язык отзывов: {', '.join(langs)}")

    hw_parts = []
    if intent_data.get("hardware_os"):
        hw_parts.append(f"ОС: {intent_data['hardware_os']}")
    if intent_data.get("hardware_cpu"):
        hw_parts.append(f"CPU: {intent_data['hardware_cpu']}")
    if intent_data.get("hardware_gpu"):
        hw_parts.append(f"GPU: {intent_data['hardware_gpu']}")
    if hw_parts:
        known_parts.append(f"Железо: {', '.join(hw_parts)}")

    time_f = intent_data.get("time_filter_text")
    if time_f:
        known_parts.append(f"Период: {time_f}")

    review_type = intent_data.get("review_type")
    if review_type:
        known_parts.append(f"Тип отзывов: {review_type}")

    sort_by = intent_data.get("sort_by")
    if sort_by:
        known_parts.append(f"Сортировка: {sort_by}")

    if not known_parts:
        return "Из запроса пока ничего не известно, кроме намерения проанализировать отзывы."

    return "Из запроса уже известно:\n- " + "\n- ".join(known_parts)


def _build_clarify_user_prompt(
    user_query: str,
    missing_slots: List[str],
    intent_data: Dict[str, Any],
) -> str:
    missing_text = ", ".join(missing_slots) if missing_slots else "название игры"
    known_context = _build_known_context(intent_data)
    return (
        f"Запрос пользователя: {user_query}\n\n"
        f"{known_context}\n\n"
        f"Не заполнены поля: {missing_text}\n\n"
        "Сформулируй короткий уточняющий вопрос СТРОГО по missing_slots. "
        "Не переспрашивай то, что указано в блоке 'Из запроса уже известно'."
    )


def _build_unknown_user_prompt(user_query: str) -> str:
    return (
        f"Запрос пользователя: {user_query}\n\n"
        "Этот запрос не относится к анализу отзывов Steam. "
        "Вежливо откажи и предложи один конкретный пример правильного запроса из правил."
    )


def _build_free_talk_user_prompt(user_query: str) -> str:
    return (
        f"Запрос пользователя: {user_query}\n\n"
        "Это свободный разговор (приветствие или болтовня). Ответь коротко и по смыслу."
    )


# ==========================================================
# DialogManager
# ==========================================================

class DialogManager:
    """
    Оркестратор пайплайна (согласно Sequence Diagram).
    Управляет единым экземпляром OllamaManager (NFR-006), координирует
    работу Intent, DataManager и ReviewProcessor, ведет журнал (FR-005).
    """

    def __init__(self, db_path: str = "local_storage.db"):
        logger.info("Инициализация DialogManager...")
        self.ollama = OllamaManager(model_name=OLLAMA_MODEL_NAME)
        self.intent_extractor = IntentExtractor(ollama_manager=self.ollama)
        self.data_manager = DataManager()
        self.review_processor = ReviewProcessor(ollama_manager=self.ollama)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
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
        logger.info(f"Новый запрос: '{user_query}'")
        try:
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

            dm_output = self.data_manager.process(intent_data)

            if dm_output.get("status") in ("text_only", "stub", "error"):
                final_response = dm_output.get("message", "Не удалось обработать запрос.")
                self._save_to_journal(user_query, "None", final_response)
                return final_response

            logger.info("Запуск Map-Reduce суммаризации...")
            review_result = self.review_processor.process(dm_output, user_query=user_query)

            final_markdown = self._format_response(review_result)

            game_name = dm_output.get("game", "Неизвестная игра")
            self._save_to_journal(user_query, game_name, final_markdown)
            return final_markdown
        except Exception as e:
            logger.exception("Критическая ошибка в DialogManager")
            return f"Произошла внутренняя ошибка при обработке запроса: {e}"
        finally:
            self.ollama.unload_model()

    def close(self) -> None:
        try:
            self.ollama.unload_model()
        except Exception as e:
            logger.warning(f"Не удалось выгрузить модель при закрытии: {e}")

    # ----------------------------------------------------------
    # Генерация текстового ответа для clarify / unknown / free_talk
    # ----------------------------------------------------------
    def _generate_text_response(
        self,
        intent_data: Dict[str, Any],
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        intent = intent_data.get("intent")
        missing_slots = intent_data.get("missing_slots") or []

        # 1. Подбираем system_prompt и user_prompt под интент
        if intent == "clarify":
            system_prompt = CLARIFY_SYSTEM_PROMPT
            user_prompt = _build_clarify_user_prompt(user_query, missing_slots, intent_data)
            fallback = "Уточните, пожалуйста, название игры, отзывы которой нужно проанализировать."
        elif intent == "unknown":
            system_prompt = UNKNOWN_SYSTEM_PROMPT
            user_prompt = _build_unknown_user_prompt(user_query)
            fallback = (
                "Я не смог понять запрос. Укажите название игры, например: "
                "«Составь сводку по отзывам Elden Ring за последний месяц»."
            )
        else:  # free_talk
            system_prompt = FREE_TALK_SYSTEM_PROMPT
            user_prompt = _build_free_talk_user_prompt(user_query)
            fallback = "Здравствуйте! Чем могу помочь?"

        # --- ЛОГИРОВАНИЕ: что уходит в LLM ---
        logger.info(
            "[text-gen] intent=%s | missing_slots=%s | user_query=%r",
            intent, missing_slots, user_query,
        )
        logger.debug(
            "[text-gen] system_prompt (len=%d):\n%s",
            len(system_prompt), system_prompt,
        )
        logger.debug("[text-gen] user_prompt:\n%s", user_prompt)
        if history:
            logger.debug(
                "[text-gen] history (последние %d сообщений): %s",
                len(history), history[-3:],
            )
        else:
            logger.debug("[text-gen] history: пусто")

        # 2. Пытаемся сгенерировать через облако
        cloud = getattr(self.review_processor, "cloud", None)
        if cloud is not None:
            try:
                text = cloud.chat_text(
                    system_prompt,
                    user_prompt,
                    history=history,
                )
                if text:
                    logger.info(
                        "[text-gen] backend=cloud | intent=%s | response=%r",
                        intent, text,
                    )
                    return text
            except Exception as e:
                logger.warning("[text-gen] Облачная генерация недоступна: %s", e)

        # 3. Fallback на локальную модель
        try:
            client = OpenAI(
                base_url=self.ollama.client_url,
                api_key="ollama",
                timeout=25.0,
            )
            messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
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
                logger.info(
                    "[text-gen] backend=local | intent=%s | response=%r",
                    intent, text,
                )
                return text
        except Exception as e:
            logger.warning("[text-gen] Локальная генерация недоступна: %s", e)

        # 4. Жёсткий шаблон, если всё упало
        logger.info(
            "[text-gen] backend=fallback | intent=%s | response=%r",
            intent, fallback,
        )
        return fallback

    def _format_response(self, review_result: Dict[str, Any]) -> str:
        """Собирает финальный Markdown-ответ для UI."""
        if review_result.get("status") != "success":
            return f"Не удалось сформировать сводку. Причина: {review_result.get('message')}"

        output = []

        # 1. Предупреждения (без эмодзи)
        warnings = review_result.get("warnings", [])
        if warnings:
            output.append("### Важные примечания")
            for w in warnings:
                output.append(f"- {w}")
            output.append("")

        # 2. Основная сводка
        output.append(review_result.get("summary", ""))

        # График (если запрошен и удалось отрендерить)
        chart_png = review_result.get("chart_png")
        if chart_png:
            output.append("### Динамика отзывов по времени")
            output.append("")
            output.append(
                f'<p><img src="data:image/png;base64,{chart_png}" '
                f'style="max-width: 100%;" alt="График отзывов по времени" /></p>'
            )
            output.append("")

        # 3. Таблица
        table = review_result.get("table")
        if table:
            output.append("\n" + table)

        # 4. Мета-информация
        meta = review_result.get("meta", {})
        if meta:
            raw_cnt = meta.get("raw_reviews_count", 0)
            final_cnt = meta.get("final_reviews_count", 0)
            output.append(f"\n---\n*Проанализировано {final_cnt} отзывов из {raw_cnt} загруженных.*")

        return "\n".join(output)