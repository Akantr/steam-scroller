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
NUM_CTX = 6144

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
            self._server_proc = subprocess.Popen(["ollama", "serve"], **popen_kwargs)
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

    def server_process(self) -> Optional[subprocess.Popen]:
        """Возвращает Popen-объект запущенного нами сервера `ollama serve` (или None)."""
        return getattr(self, "_server_proc", None)

    def stop_server(self, terminate_timeout: float = 5.0) -> None:
        """
        Полностью останавливает сервер `ollama serve`, если он был запущен
        этим приложением (запущенный вручную/системой сервер не трогается).
        Сначала выгружает модель, затем завершает процесс.
        """
        self.unload_model()

        proc = getattr(self, "_server_proc", None)
        if proc is None or proc.poll() is not None:
            return

        logger.info("Остановка запущенного приложением сервера Ollama...")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=terminate_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=terminate_timeout)
        except Exception as e:
            logger.warning(f"Не удалось остановить сервер Ollama: {e}")

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
    # ↓ добавлен "funny"
    sort_by: Optional[Literal["helpful", "recent", "updated", "funny"]] = None

    time_filter_text: Optional[str] = None
    min_playtime_hours: Optional[int] = None
    max_playtime_hours: Optional[int] = None

    hardware_os: Optional[Literal["windows", "mac", "linux"]] = None
    hardware_cpu: Optional[str] = None
    hardware_gpu: Optional[str] = None

    # ↓ НОВОЕ: сколько сырых отзывов тянем из Steam (до очистки и семантики)
    target_count: Optional[int] = Field(None, ge=1)

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
"analyze_reviews": Пользователь просит сводку, плюсы/минусы, анализ для КОНКРЕТНОЙ игры (например, "проанализируй Elden Ring", "что говорят про боевку в Baldur's Gate 3").
"rank_games": Пользователь ищет игру по критериям, просит топ или рейтинг (например, "лучший рогалик за месяц", "найди хоррор на ПК").
"free_talk": Приветствия ("привет", "здравствуй"), благодарности ("спасибо"), простые вопросы не по теме анализа ("как дела?").
"clarify": Используй, если пользователь явно хочет анализ отзывов, но НЕ указал название игры. Добавь "target_game" в missing_slots. Также используй, если пропущены другие критически важные параметры.
"unknown": Запрос не относится к анализу отзывов Steam И не является приветствием/болтовней. Например: "расскажи анекдот", "какая погода?", "что такое Steam?", "купи мне игру".

ПРАВИЛА ДЛЯ НЕОДНОЗНАЧНЫХ СЛУЧАЕВ:
1. Если пользователь упоминает жанр/тег, но не игру:
   - Если просит НАЙТИ/РЕКОМЕНДОВАТЬ ("найди лучший хоррор") → rank_games
   - Если просит АНАЛИЗ ("расскажи про хорроры") → clarify (нет конкретной игры)
2. Если пользователь называет игру с опечаткой ("Элден Ринг", "Baldurs Gate") → analyze_reviews с target_game = текст из запроса (fuzzy matching сделает остальное).
3. Если пользователь просит анализ НЕСКОЛЬКИХ игр ("проанализируй Elden Ring и Baldur's Gate 3") → clarify с missing_slots = ["target_game"] (нужно уточнить, какую именно).
4. Если в истории диалога уже была игра, а пользователь говорит "а теперь про боевку" → analyze_reviews с target_game из истории (если она есть в контексте).
5. Если запрос слишком абстрактный ("расскажи про игры") → unknown.

КРИТИЧЕСКИЙ ЗАПРЕТ:
ЕСЛИ пользователь НЕ назвал конкретную игру явно, target_game ДОЛЖЕН БЫТЬ null. НЕ ВЫДУМЫВАЙ название игры, даже если оно кажется очевидным из контекста.

ПРАВИЛА ЗАПОЛНЕНИЯ (КРИТИЧЕСКИ ВАЖНО):
Если пользователь НЕ упомянул параметр явно, ставь null. НЕ ИСПОЛЬЗУЙ дефолтные значения.
target_game: название игры. Если игра не названа — null.
genre_or_tags: извлекай жанры/теги ТОЛЬКО текстом на английском (например, ["roguelike", "soulslike"]).
sub_tasks: добавь "table" (если просят таблицу), "graph" (если просят график).
languages: массив языков на английском (например, ["russian"], ["japanese", "english"]). Если не указано — null.
review_type: строго одно из ["positive", "negative", "all"] или null.
purchase_type: строго одно из ["steam", "non_steam_purchase", "all"] или null.
sort_by: строго одно из ["helpful", "recent", "updated", "funny"] или null.
time_filter_text: извлекай время текстом, как в запросе (например, "last month", "2023", "last week"). Если времени нет — null.
hardware_os: если просят игру для ПК/Windows ставь "windows", для Mac — "mac", для Linux — "linux". Иначе null.
hardware_cpu / hardware_gpu: если явно указана модель (например, "RTX 4090") — запиши текстом, иначе null.
data_retrieval_needed: true, если intent = analyze_reviews или rank_games. false для free_talk, clarify, unknown.
confidence: число от 0.0 до 1.0. Насколько ты уверен в определении intent.
target_count: количество отзывов для загрузки — целое число. Указывай ТОЛЬКО если пользователь явно назвал число ("возьми 500 отзывов", "проанализируй 1000 отзывов"), иначе null.

ПРИМЕРЫ:

Запрос: "Сделай сводку по русским отзывам Elden Ring за последний месяц и выведи таблицу"
Ответ: {
  "intent": "analyze_reviews",
  "confidence": 0.99,
  "target_game": "Elden Ring",
  "genre_or_tags": [],
  "sub_tasks": ["table"],
  "languages": ["russian"],
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": "last month",
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": null,
  "missing_slots": [],
  "data_retrieval_needed": true
}

Запрос: "Найди лучший хоррор на ПК"
Ответ: {
  "intent": "rank_games",
  "confidence": 0.95,
  "target_game": null,
  "genre_or_tags": ["horror"],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": "windows",
  "hardware_cpu": null,
  "hardware_gpu": null,
  "missing_slots": [],
  "data_retrieval_needed": true
}

Запрос: "Проанализируй отзывы на японском с RTX 3060"
Ответ: {
  "intent": "clarify",
  "confidence": 0.9,
  "target_game": null,
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": ["japanese"],
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": "RTX 3060",
  "missing_slots": ["target_game"],
  "data_retrieval_needed": false
}

Запрос: "AMD Ryzen 5 7535HS, Satisfactory"
Ответ: {
  "intent": "analyze_reviews",
  "confidence": 0.98,
  "target_game": "Satisfactory",
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": "AMD Ryzen 5 7535HS",
  "hardware_gpu": null,
  "missing_slots": [],
  "data_retrieval_needed": true
}

Запрос: "Проанализируй 500 отзывов Elden Ring, самые смешные"
Ответ: {
  "intent": "analyze_reviews",
  "confidence": 0.97,
  "target_game": "Elden Ring",
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": "funny",
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": null,
  "target_count": 500,
  "missing_slots": [],
  "data_retrieval_needed": true
}

Запрос: "Привет"
Ответ: {
  "intent": "free_talk",
  "confidence": 0.99,
  "target_game": null,
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": null,
  "missing_slots": [],
  "data_retrieval_needed": false
}

Запрос: "Расскажи анекдот"
Ответ: {
  "intent": "unknown",
  "confidence": 0.95,
  "target_game": null,
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": null,
  "missing_slots": [],
  "data_retrieval_needed": false
}

Запрос: "Сделай сводку по последнему месяцу"
Ответ: {
  "intent": "clarify",
  "confidence": 0.85,
  "target_game": null,
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": "last month",
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": null,
  "missing_slots": ["target_game"],
  "data_retrieval_needed": false
}

Запрос: "Что говорят про боевку?"
Ответ: {
  "intent": "clarify",
  "confidence": 0.88,
  "target_game": null,
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": null,
  "missing_slots": ["target_game"],
  "data_retrieval_needed": false
}

Запрос: "Проанализируй Elden Ring и Baldur's Gate 3"
Ответ: {
  "intent": "clarify",
  "confidence": 0.92,
  "target_game": null,
  "genre_or_tags": [],
  "sub_tasks": [],
  "languages": null,
  "review_type": null,
  "purchase_type": null,
  "sort_by": null,
  "time_filter_text": null,
  "min_playtime_hours": null,
  "max_playtime_hours": null,
  "hardware_os": null,
  "hardware_cpu": null,
  "hardware_gpu": null,
  "missing_slots": ["target_game"],
  "data_retrieval_needed": false
}

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
                    # ↓ добавить
                    extra_body={
                        "options": {
                            "num_ctx": NUM_CTX,
                            "temperature": 0.0,
                            "repeat_penalty": 1.3,
                            "repeat_last_n": 256,
                        },
                        "think": False,
                        "keep_alive": "30m",
                    },
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