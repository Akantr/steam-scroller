import json
import logging
import platform
import re
import subprocess
import time
from typing import Any, Dict, List, Literal, Optional
import sys

import requests
from openai import OpenAI
import openai
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ==========================================
# Константы
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "qwen3.8-4b"
SERVER_START_TIMEOUT_SECONDS = 10
UNLOAD_TIMEOUT_SECONDS = 5
RETRY_BACKOFF_BASE_SECONDS = 1.5

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# ==========================================
# 1. Управление сервером Ollama
# ==========================================
class OllamaManager:
    """Управляет локальным сервером Ollama и памятью модели."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_name: str = OLLAMA_MODEL_NAME):
        self.base_url = base_url.rstrip("/")
        self.client_url = f"{self.base_url}/v1"
        self.model_name = model_name

    # ---- Проверка/запуск ----
    def is_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_server(self) -> None:
        """Запускает Ollama в фоне, если он не запущен."""
        if self.is_running():
            return

        logger.info("Запуск сервера Ollama...")

        popen_kwargs: Dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        try:
            subprocess.Popen(["ollama", "serve"], **popen_kwargs)
        except FileNotFoundError as e:
            raise RuntimeError(
                "Бинарник 'ollama' не найден в PATH. Установите Ollama: "
                "https://ollama.com/download"
            ) from e

        for _ in range(SERVER_START_TIMEOUT_SECONDS):
            if self.is_running():
                logger.info("Ollama успешно запущена.")
                return
            time.sleep(1)

        raise RuntimeError(
            f"Не удалось запустить Ollama за {SERVER_START_TIMEOUT_SECONDS} сек."
        )

    # ---- Выгрузка модели ----
    def unload_model(self) -> None:
        """Выгружает модель из памяти (освобождает VRAM/RAM)."""
        if sys.is_finalizing():
            return

        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=UNLOAD_TIMEOUT_SECONDS,
            )
            logger.debug(f"Модель {self.model_name} выгружена из памяти.")
        except Exception as e:
            logger.warning(f"Не удалось выгрузить модель {self.model_name}: {e}")

    def close(self) -> None:
        """Освобождает ресурсы (выгружает модель)."""
        self.unload_model()

    def __enter__(self) -> "OllamaManager":
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ==========================================
# 2. Pydantic-схема
# ==========================================
class IntentOutput(BaseModel):
    intent: Literal["analyze_reviews", "rank_games", "free_talk", "clarify", "unknown"]
    confidence: float = Field(..., ge=0.0, le=1.0)

    target_game: Optional[str] = None
    genre_or_tags: List[str] = Field(default_factory=list)
    sub_tasks: List[Literal["table", "graph"]] = Field(default_factory=list)

    languages: Optional[List[str]] = None
    review_type: Optional[Literal["positive", "negative", "all"]] = None
    purchase_type: Optional[Literal["steam", "non_steam_purchase", "all"]] = None
    sort_by: Optional[Literal["helpful", "recent", "updated"]] = None

    time_filter_text: Optional[str] = None
    min_playtime_hours: Optional[int] = None
    max_playtime_hours: Optional[int] = None

    hardware_os: Optional[Literal["windows", "mac", "linux"]] = None
    hardware_cpu: Optional[str] = None
    hardware_gpu: Optional[str] = None

    missing_slots: List[str] = Field(default_factory=list)
    data_retrieval_needed: bool


# ==========================================
# 3. Класс экстрактора намерений
# ==========================================
class IntentExtractor:
    SYSTEM_PROMPT = """
Ты — высокоточный парсер запросов для приложения анализа отзывов Steam.
Твоя единственная задача — извлечь параметры из запроса пользователя и вернуть СТРОГО валидный JSON. Никакого текста вокруг, никакого markdown.

ПРАВИЛА КЛАССИФИКАЦИИ (intent):
- "analyze_reviews": Пользователь просит сводку, плюсы/минусы, анализ для КОНКРЕТНОЙ игры (например, "проанализируй Elden Ring").
- "rank_games": Пользователь ищет игру по критериям, просит топ или рейтинг (например, "лучший рогалик за месяц").
- "free_talk": Приветствия, болтовня.
- "clarify": Используй, если intent = analyze_reviews, но target_game не указан (добавь "target_game" в missing_slots).
- "unknown": Не удалось понять запрос.

ПРАВИЛА ЗАПОЛНЕНИЯ (КРИТИЧЕСКИ ВАЖНО):
1. Если пользователь НЕ упомянул параметр явно, ставь null. НЕ ИСПОЛЬЗУЙ дефолтные значения.
2. target_game: название игры. Если игра не названа — null.
3. genre_or_tags: извлекай жанры/теги ТОЛЬКО текстом на английском (например, ["roguelike", "soulslike"]).
4. sub_tasks: добавь "table" (если просят таблицу), "graph" (если просят график).
5. languages: массив языков на английском (например, ["russian"], ["japanese", "english"]). Если не указано — null.
6. review_type: строго одно из ["positive", "negative", "all"] или null.
7. purchase_type: строго одно из ["steam", "non_steam_purchase", "all"] или null.
8. sort_by: строго одно из ["helpful", "recent", "updated"] или null.
9. time_filter_text: извлекай время текстом, как в запросе (например, "last month", "2023", "last week"). Если времени нет — null.
10. hardware_os: если просят игру для ПК/Windows ставь "windows", для Mac — "mac", для Linux — "linux". Иначе null.
11. hardware_cpu / hardware_gpu: если явно указана модель (например, "RTX 4090") — запиши текстом, иначе null.
12. data_retrieval_needed: true, если intent = analyze_reviews или rank_games. false для free_talk, clarify, unknown.
13. confidence: число от 0.0 до 1.0. Насколько ты уверен в определении intent.

ПРИМЕРЫ:
Запрос: "Сделай сводку по русским отзывам Elden Ring за последний месяц и выведи таблицу"
Ответ: {"intent": "analyze_reviews", "confidence": 0.99, "target_game": "Elden Ring", "genre_or_tags": [], "sub_tasks": ["table"], "languages": ["russian"], "review_type": null, "purchase_type": null, "sort_by": null, "time_filter_text": "last month", "min_playtime_hours": null, "max_playtime_hours": null, "hardware_os": null, "hardware_cpu": null, "hardware_gpu": null, "missing_slots": [], "data_retrieval_needed": true}

Запрос: "Найди лучший хоррор на ПК"
Ответ: {"intent": "rank_games", "confidence": 0.95, "target_game": null, "genre_or_tags": ["horror"], "sub_tasks": [], "languages": null, "review_type": null, "purchase_type": null, "sort_by": null, "time_filter_text": null, "min_playtime_hours": null, "max_playtime_hours": null, "hardware_os": "windows", "hardware_cpu": null, "hardware_gpu": null, "missing_slots": [], "data_retrieval_needed": true}

Запрос: "Проанализируй отзывы на японском с RTX 3060"
Ответ: {"intent": "clarify", "confidence": 0.9, "target_game": null, "genre_or_tags": [], "sub_tasks": [], "languages": ["japanese"], "review_type": null, "purchase_type": null, "sort_by": null, "time_filter_text": null, "min_playtime_hours": null, "max_playtime_hours": null, "hardware_os": null, "hardware_cpu": null, "hardware_gpu": "RTX 3060", "missing_slots": ["target_game"], "data_retrieval_needed": false}

Запрос: "AMD Ryzen 5 7535HS, Satisfactory."
Ответ: {"intent": "analyze_reviews", "confidence": 0.98, "target_game": "Satisfactory", "genre_or_tags": [], "sub_tasks": [], "languages": null, "review_type": null, "purchase_type": null, "sort_by": null, "time_filter_text": null, "min_playtime_hours": null, "max_playtime_hours": null, "hardware_os": null, "hardware_cpu": "AMD Ryzen 5 7535HS", "hardware_gpu": null, "missing_slots": [], "data_retrieval_needed": true}

Верни только JSON.
"""

    def __init__(self, model_name: str = OLLAMA_MODEL_NAME, ollama_manager: Optional[OllamaManager] = None):
        # ИСПОЛЬЗУЕМ ПЕРЕДАННЫЙ МЕНЕДЖЕР (синглтон) ИЛИ СОЗДАЕМ НОВЫЙ
        self.ollama = ollama_manager if ollama_manager else OllamaManager(model_name=model_name)
        
        if not self.ollama.is_running():
            self.ollama.start_server()

        self.client = OpenAI(
            base_url=self.ollama.client_url,
            api_key="ollama",
        )
        self.model_name = model_name

    @staticmethod
    def _strip_code_fences(raw: str) -> str:
        return _CODE_FENCE_RE.sub("", raw.strip()).strip()

    @staticmethod
    def _fallback_intent() -> Dict[str, Any]:
        return IntentOutput(
            intent="unknown",
            confidence=0.0,
            data_retrieval_needed=False,
        ).model_dump()

    def extract_intent(
        self,
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        if history:
            messages.extend(history[-3:])
        messages.append({"role": "user", "content": user_query})

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                cleaned = self._strip_code_fences(raw)
                parsed = json.loads(cleaned)
                validated = IntentOutput(**parsed)
                return validated.model_dump()

            except json.JSONDecodeError as e:
                logger.warning(f"Попытка {attempt + 1}/{max_retries}: невалидный JSON. Ошибка: {e}")
                messages.append({
                    "role": "user",
                    "content": "Ошибка: ты вернул невалидный JSON. Верни строго JSON без markdown и без пояснений.",
                })

            except ValidationError as e:
                logger.warning(f"Попытка {attempt + 1}/{max_retries}: JSON не соответствует схеме. Ошибка: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Ошибка: твой JSON не соответствует схеме. Детали: {e}. Исправь.",
                })

            except openai.NotFoundError as e:
                logger.error(f"Модель {self.model_name!r} не найдена. Детали: {e}")
                break

            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                logger.warning(f"Попытка {attempt + 1}/{max_retries}: сетевая ошибка: {e}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt))

            except openai.APIError as e:
                logger.warning(f"Попытка {attempt + 1}/{max_retries}: ошибка API: {e}")
                if attempt < max_retries - 1:
                    time.sleep(RETRY_BACKOFF_BASE_SECONDS * (2 ** attempt))

            except Exception as e:
                logger.exception(f"Попытка {attempt + 1}/{max_retries}: непредвиденная ошибка: {type(e).__name__}: {e}")
                break

        logger.error("Не удалось получить валидный интент от LLM.")
        return self._fallback_intent()

    def close(self) -> None:
        """
        Ничего не делаем. Жизненным циклом OllamaManager управляет DialogManager.
        Если IntentExtractor используется автономно (без DialogManager) - 
        вызывайте self.ollama.unload_model() вручную.
        """
        pass

    def __enter__(self) -> "IntentExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()