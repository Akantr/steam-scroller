import json
import logging
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from rapidfuzz import fuzz, process as fuzz_process
import re

import requests

logger = logging.getLogger(__name__)

# ==========================================
# Константы Steam Web API (актуально на октябрь 2026)
# ==========================================
STORE_SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
REVIEWS_API_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"

FILTER_MAP = {
    "helpful": 0,
    "recent": 1,
    "updated": 2,
    "funny": 3,
}

REVIEW_TYPE_MAP = {
    "all": 0,
    "positive": 1,
    "negative": 2,
}

PURCHASE_TYPE_MAP = {
    "steam": 0,
    "all": 1,
    "non_steam_purchase": 2,
}

MAX_REVIEWS_TARGET = 2000
PAGE_SIZE = 100
MAX_PAGES = 20
MAX_DAY_RANGE = 365
DEFAULT_TIMEOUT = 15
RETRY_BACKOFF_SECONDS = 0.6

USER_AGENT = "SteamReviewAnalyzer/1.0 (contact: local-app)"


class GameNotFoundError(Exception):
    """Выбрасывается, если игру не удалось найти в Steam Store."""
    pass


class SteamAPIError(Exception):
    """Выбрасывается при сетевых ошибках или ошибках Steam API."""
    pass


class SteamRateLimitError(SteamAPIError):
    """HTTP 429 — превышен лимит запросов."""
    pass


# ==========================================
# 1. Поиск appid по названию игры
# ==========================================
def _normalize_title(title: str) -> str:
    """Нижний регистр, только буквы и цифры — для сравнения названий."""
    return re.sub(r'[^a-z0-9]', '', (title or '').lower())


def resolve_appid(game_name: str, timeout: int = 10) -> int:
    """
    Ищет appid игры через Steam Store Search API.

    Защита от неоднозначных названий: если по запросу «Hades» первым
    приходит «Hades II», ищем ТОЧНОЕ совпадение названия среди всех
    результатов. Fuzzy-матчинг — только как fallback для опечаток.
    """
    if not game_name or not game_name.strip():
        raise GameNotFoundError("Пустое название игры.")

    params = {
        "term": game_name.strip(),
        "l": "russian",
        "cc": "RU",
    }

    # --- 1. Запрос к Steam Store Search ---
    try:
        response = requests.get(
            STORE_SEARCH_URL,
            params=params,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса к Steam Store Search: {e}")
        raise SteamAPIError(f"Steam Store Search недоступен: {e}")

    items = data.get("items", [])
    if not items:
        raise GameNotFoundError(f"Игра '{game_name}' не найдена в Steam Store.")

    # --- 2. Отбрасываем всё, что не игра (DLC, саундтреки, демо) ---
    games = [it for it in items if it.get("type") == "app"]
    if not games:
        found = [it.get("name") for it in items]
        raise GameNotFoundError(
            f"По запросу '{game_name}' не найдено игр. "
            f"Найденные варианты: {found}. Уточните название."
        )

    # --- 3. Точное совпадение по нормализованному названию ---
    target_norm = _normalize_title(game_name)
    for game in games:
        name = game.get("name", "")
        if _normalize_title(name) == target_norm:
            appid = game.get("id")
            if appid:
                logger.info(
                    f"Точное совпадение: '{game_name}' -> '{name}' "
                    f"(appid={appid})"
                )
                return int(appid)

    # --- 4. Fuzzy-совпадение со строгим порогом ---
    best_ratio = 0.0
    best_game = None
    for game in games:
        if not game.get("id"):
            continue
        ratio = SequenceMatcher(
            None, target_norm, _normalize_title(game.get("name", ""))
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_game = game

    if best_game and best_ratio >= 0.85:
        appid = best_game["id"]
        logger.info(
            f"Fuzzy-совпадение (ratio={best_ratio:.2f}): "
            f"'{game_name}' -> '{best_game.get('name')}' (appid={appid})"
        )
        return int(appid)

    # --- 5. Уверенно определить не удалось ---
    found = [g.get("name") for g in games]
    raise GameNotFoundError(
        f"Не удалось однозначно определить игру '{game_name}'. "
        f"Найденные варианты: {found}. Уточните название."
    )


# ==========================================
# 2. Парсинг временных фильтров
# ==========================================
def parse_time_filter(time_filter_text: Optional[str]) -> Dict[str, Any]:
    """
    Преобразует текстовый временной фильтр в параметры для GetAppReviews.

    Возвращает:
    - {"day_range": int} для относительных периодов;
    - {"date_range_start": int, "date_range_end": int} для конкретного года;
    - {} если фильтр не задан / "all".
    """
    if not time_filter_text:
        return {}

    text = time_filter_text.lower().strip()

    if text in ("all", "все", "все время", "all time"):
        return {}

    if any(word in text for word in ("month", "месяц")):
        return {"day_range": 30}
    if any(word in text for word in ("week", "недел")):
        return {"day_range": 7}
    if any(word in text for word in ("year", "год")):
        return {"day_range": 365}

    try:
        year = int(text)
        now = datetime.now(timezone.utc)
        if year > now.year:
            logger.warning(
                f"time_filter_text='{time_filter_text}' указывает на будущий год. "
                "Фильтр проигнорирован."
            )
            return {}

        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        if year == now.year:
            end = now
        else:
            end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        return {
            "date_range_start": int(start.timestamp()),
            "date_range_end": int(end.timestamp()),
        }
    except ValueError:
        pass

    logger.warning(
        f"Не удалось распарсить time_filter_text='{time_filter_text}'. "
        "Фильтр по времени не применён."
    )
    return {}


def parse_time_to_days(time_filter_text: Optional[str]) -> int:
    """
    Устаревшая обёртка. Для конкретного года возвращает 0.
    Используйте parse_time_filter() для date_range_start/end.
    """
    parsed = parse_time_filter(time_filter_text)
    if "day_range" in parsed:
        return parsed["day_range"]

    if "date_range_start" in parsed:
        logger.warning(
            "parse_time_to_days() не поддерживает фильтр по конкретному году. "
            "Используйте parse_time_filter() для получения date_range_start/end."
        )
        return 0

    return 0


# ==========================================
# Хелпер для нормализации ответа Steam API
# ==========================================
def _extract_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Сервисные эндпоинты Steam Web API (IUserReviewsService и др.)
    оборачивают ответ в {"response": {...}}.
    Возвращает payload независимо от того, есть обёртка или нет.
    """
    if isinstance(data, dict) and isinstance(data.get("response"), dict):
        return data["response"]
    return data


def _read_total_matching(payload: Dict[str, Any]) -> Optional[int]:
    """
    Достаёт общее количество отзывов, соответствующих фильтрам.
    В новом API это query_summary.total_reviews; в старом могло быть total_matching.
    """
    if payload.get("total_matching") is not None:
        return payload["total_matching"]

    summary = payload.get("query_summary") or {}
    return summary.get("total_reviews")


# ==========================================
# 3. Загрузка отзывов через новый API (500 отзывов, пагинация)
# ==========================================
def fetch_reviews_new_api(
    appid: int,
    normalized_params: Dict[str, Any],
    target_count: int = MAX_REVIEWS_TARGET,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Загружает отзывы через IUserReviewsService/GetAppReviews/v1/.

    Возвращает:
        {
            "reviews": List[Dict],
            "hardware_fallback": bool,
            "total_fetched": int,
            "total_matching": int | None,  # из query_summary.total_reviews
            "day_range_used": int | None,
        }
    """
    input_json: Dict[str, Any] = {
        "appid": appid,
        "num_per_page": PAGE_SIZE,
        "filter_offtopic_activity": False,
    }

    # Языки
    languages = normalized_params.get("languages")
    if not languages or languages == ["all"]:
        input_json["languages"] = ["all"]
    else:
        input_json["languages"] = languages

    # Сортировка
    sort_by = normalized_params.get("sort_by", "helpful")
    input_json["filter"] = FILTER_MAP.get(sort_by, 0)

    # Тип отзыва и покупки
    input_json["review_type"] = REVIEW_TYPE_MAP.get(
        normalized_params.get("review_type", "all"), 0
    )
    input_json["purchase_type"] = PURCHASE_TYPE_MAP.get(
        normalized_params.get("purchase_type", "all"), 1
    )

    # Временной фильтр: day_range применяется только к Helpful
    day_range = normalized_params.get("day_range", 0)
    if sort_by == "helpful":
        input_json["day_range"] = min(day_range, MAX_DAY_RANGE) if day_range > 0 else 0
    elif day_range:
        logger.warning(
            f"day_range={day_range} проигнорирован: применяется только к Helpful."
        )

    dr_start = normalized_params.get("date_range_start")
    dr_end = normalized_params.get("date_range_end")
    if dr_start is not None and dr_end is not None:
        input_json["date_range_start"] = int(dr_start)
        input_json["date_range_end"] = int(dr_end)
    elif dr_start is not None or dr_end is not None:
        logger.warning("date_range_start и date_range_end должны быть заданы вместе.")

    if normalized_params.get("min_playtime_hours") is not None:
        input_json["playtime_min_hours"] = int(normalized_params["min_playtime_hours"])
    if normalized_params.get("max_playtime_hours") is not None:
        input_json["playtime_max_hours"] = int(normalized_params["max_playtime_hours"])

    hw_os = normalized_params.get("hardware_os")
    hw_cpu = normalized_params.get("hardware_cpu")
    hw_gpu = normalized_params.get("hardware_gpu")
    has_hardware_filter = any([hw_os, hw_cpu, hw_gpu])
    if hw_os:
        input_json["hardware_os"] = hw_os
    if hw_cpu:
        input_json["hardware_cpu"] = hw_cpu
    if hw_gpu:
        input_json["hardware_gpu"] = hw_gpu

    # --- Пагинация ---
    all_reviews: List[Dict[str, Any]] = []
    cursor: Optional[str] = "*"  # '*' для первой страницы
    page = 0
    hardware_fallback = False
    total_matching: Optional[int] = None
    day_range_used: Optional[int] = None

    while len(all_reviews) < target_count and page < MAX_PAGES:
        page += 1
        input_json["cursor"] = cursor

        encoded_input = urllib.parse.quote(
            json.dumps(input_json, ensure_ascii=False), safe=""
        )
        url = f"{REVIEWS_API_URL}?input_json={encoded_input}"
        logger.debug(f"GET страница {page} (cursor={cursor})")

        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            if response.status_code == 429:
                raise SteamRateLimitError("HTTP 429: rate limit.")
            response.raise_for_status()
            data = response.json()
        except SteamRateLimitError:
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса (страница {page}): {e}")
            if page == 1:
                raise SteamAPIError(f"Не удалось загрузить отзывы: {e}")
            break
        except json.JSONDecodeError as e:
            logger.error(f"Некорректный JSON (страница {page}): {e}")
            if page == 1:
                raise SteamAPIError(f"Steam API вернул некорректный JSON: {e}")
            break

        # Нормализуем ответ: Steam оборачивает в {"response": {...}}
        payload = _extract_payload(data)
        reviews_page = payload.get("reviews") or []

        if page == 1:
            total_matching = _read_total_matching(payload)
            day_range_used = payload.get("day_range_used")

        logger.debug(
            f"Страница {page}: получено {len(reviews_page)} отзывов, "
            f"total_matching={total_matching}, day_range_used={day_range_used}"
        )

        # --- Пустая страница ---
        if not reviews_page:
            # ВАЖНО: hardware-fallback должен срабатывать ДО break.
            # Если API вернул 0 из-за аппаратного фильтра — повторяем без него.
            if has_hardware_filter and page == 1:
                logger.warning(
                    "Hardware-фильтр вернул 0 отзывов. Повторный запрос без железа."
                )
                hardware_fallback = True
                for key in ("hardware_os", "hardware_cpu", "hardware_gpu"):
                    input_json.pop(key, None)
                has_hardware_filter = False
                page = 0
                cursor = "*"
                all_reviews = []
                total_matching = None
                day_range_used = None
                continue

            logger.info(f"Страница {page} пуста, завершаем пагинацию.")
            break

        all_reviews.extend(reviews_page)

        cursor = payload.get("cursor")
        if not cursor:
            logger.info("Курсор отсутствует, пагинация завершена.")
            break

        time.sleep(0.4)

    all_reviews = all_reviews[:target_count]

    logger.info(
        f"Загружено {len(all_reviews)} отзывов (appid={appid}, "
        f"hardware_fallback={hardware_fallback}, total_matching={total_matching})"
    )

    return {
        "reviews": all_reviews,
        "hardware_fallback": hardware_fallback,
        "total_fetched": len(all_reviews),
        "total_matching": total_matching,
        "day_range_used": day_range_used,
    }