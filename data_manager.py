import gc
import logging
from typing import Any, Dict, List
from collections import Counter
import hashlib
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import steam_api
import preprocessor

logger = logging.getLogger(__name__)

# ==========================================
# Константы
# ==========================================
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
DUPLICATE_THRESHOLD = 0.92  # повышено с 0.85 — защита от схлопывания тематических отзывов
DEFAULT_MAX_OUTPUT = 120  # повышено с 80 — больше материала для 7 признаков
MAX_CLUSTERS = 7
DEFAULT_TARGET_REVIEWS = 1000
MIN_REVIEWS_FOR_SUMMARY = 30
MIN_PER_ASPECT = 3  # минимум отзывов на аспект при стратифицированном отборе
LENGTH_CAP = 1500  # дальше рост длины не даёт роста информативности

# Группы intent'ов (источник истины — intent.IntentOutput)
INTENTS_TEXT_ONLY = {"free_talk", "clarify", "unknown"}
INTENTS_STUB = {"rank_games"}
INTENTS_REVIEWS = {"analyze_reviews"}


# ==========================================
# 1. Нормализация данных
# ==========================================
def normalize_intent(intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводит данные от IntentExtractor к машинному формату.
    Заполняет дефолтные значения, конвертирует time_filter_text
    в параметры API (day_range или date_range_start/end).
    """
    if not isinstance(intent_data, dict):
        raise TypeError(
            f"intent_data должен быть dict, получено {type(intent_data).__name__}"
        )
    normalized = intent_data.copy()

    if not normalized.get("languages"):
        normalized["languages"] = ["all"]
    if not normalized.get("review_type"):
        normalized["review_type"] = "all"
    if not normalized.get("purchase_type"):
        normalized["purchase_type"] = "all"
    if not normalized.get("sort_by"):
        normalized["sort_by"] = "helpful"

    time_text = normalized.get("time_filter_text")
    time_params = steam_api.parse_time_filter(time_text)
    normalized["day_range"] = time_params.get("day_range", 0)
    if "date_range_start" in time_params and "date_range_end" in time_params:
        normalized["date_range_start"] = time_params["date_range_start"]
        normalized["date_range_end"] = time_params["date_range_end"]
    return normalized


# ==========================================
# 2. Роутинг запросов
# ==========================================
def route_request(intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Определяет, что делать с запросом на основе intent.
    Возвращает dict с полем "kind" ∈ {"text_only", "stub", "need_reviews", "error"}.
    """
    intent = intent_data.get("intent")
    if intent in INTENTS_TEXT_ONLY:
        return {"kind": "text_only", "intent": intent}
    if intent in INTENTS_STUB:
        return {
            "kind": "stub",
            "message": f"Обработка intent={intent!r} пока не реализована.",
        }
    if intent in INTENTS_REVIEWS:
        return {"kind": "need_reviews"}
    return {"kind": "error", "message": f"Неизвестный intent: {intent!r}"}


# ==========================================
# 3. Семантическая фильтрация
# ==========================================
def _aspect_profile(text: str) -> set:
    """Определяет, какие аспекты игры упомянуты в отзыве.
    Переиспользует регулярки из preprocessor — единый источник истины."""
    return {a for a, rx in preprocessor._ASPECT_REGEXES.items() if rx.search(text)}


def _quality_score(text: str, profile: set) -> float:
    """Оценка качества отзыва: аспекты важнее длины."""
    return 2.0 * len(profile) + min(len(text), LENGTH_CAP) / LENGTH_CAP


def _select_diverse(texts: List[str], max_output: int) -> List[int]:
    """Стратифицированный отбор с квотой на аспект.
    Фаза 1: закрывает квоты, начиная с самых редких аспектов.
    Фаза 2: добирает остаток по качеству (аспекты + длина)."""
    profiles = [_aspect_profile(t) for t in texts]
    freq = Counter(a for p in profiles for a in p)
    selected: List[int] = []
    used: set = set()
    covered: Counter = Counter()

    # Фаза 1: закрываем квоты, начиная с самых редких аспектов
    for aspect in sorted(freq, key=lambda a: freq[a]):
        candidates = sorted(
            (i for i, p in enumerate(profiles) if aspect in p and i not in used),
            key=lambda i: -_quality_score(texts[i], profiles[i]),
        )
        for i in candidates[: max(0, MIN_PER_ASPECT - covered[aspect])]:
            selected.append(i)
            used.add(i)
            covered.update(profiles[i])
            if len(selected) >= max_output:
                return selected

    # Фаза 2: добираем остаток по качеству, а не по длине
    rest = sorted(
        (i for i in range(len(texts)) if i not in used),
        key=lambda i: -_quality_score(texts[i], profiles[i]),
    )
    for i in rest:
        if len(selected) >= max_output:
            break
        selected.append(i)
    return selected


def _enrich_with_metadata(
    texts: List[str], reviews: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Добавляет метаданные из Steam API к отобранным отзывам.
    Это те самые косвенные признаки, которые модель использует как теги."""
    result = []
    for text, review in zip(texts, reviews):
        author = review.get("author") or {}
        playtime_at_review = (author.get("playtime_at_review") or 0) / 60
        playtime_forever = (author.get("playtime_forever") or 0) / 60
        result.append({
            "review_text": text,
            "playtime_at_review": playtime_at_review,
            "playtime_forever": playtime_forever,
            "votes_up": review.get("votes_up") or 0,
            "votes_funny": review.get("votes_funny") or 0,
            "weighted_vote_score": review.get("weighted_vote_score") or 0.0,
            "comment_count": review.get("comment_count") or 0,
            "was_updated": (
                (review.get("timestamp_updated") or 0) >
                (review.get("timestamp_created") or 0)
            ),
            "is_probable_meme": (
                (review.get("_meme_info") or {}).get("is_probable_meme", False)
            ),
        })
    return result


def _log_coverage(enriched: List[Dict[str, Any]]) -> None:
    """Логирует покрытие аспектов в финальной выборке."""
    coverage: Counter = Counter()
    for r in enriched:
        for a in _aspect_profile(r.get("review_text", "")):
            coverage[a] += 1
    logger.info(
        "Покрытие аспектов в финальной выборке: %s",
        dict(coverage.most_common()),
    )
    weak = [a for a, c in coverage.items() if c < MIN_PER_ASPECT]
    if weak:
        logger.warning(
            "Слабо покрытые аспекты (риск склейки признаков): %s", weak
        )


def _deduplicate(
    texts: List[str], embeddings: np.ndarray, threshold: float
) -> List[int]:
    """Greedy-дедупликация. Для каждого отзыва в порядке «лучший представитель
    первым» проверяем: похож ли он на УЖЕ отобранные (а не на любые в цепочке).
    Устраняет проблему single-linkage."""
    sim = np.dot(embeddings, embeddings.T)
    n = len(texts)
    order = sorted(range(n), key=lambda i: (-len(texts[i]), i))
    kept: List[int] = []
    for i in order:
        if not kept:
            kept.append(i)
            continue
        if np.any(sim[i, kept] > threshold):
            continue
        kept.append(i)
    return sorted(kept)


def _semantic_filter_with_embeddings(
    texts: List[str], reviews: List[Dict[str, Any]], max_output: int
) -> List[Dict[str, Any]]:
    """Внутренняя реализация; может бросать исключения — обёрнута в semantic_filter."""
    logger.info("Семантическая фильтрация: %d отзывов", len(texts))
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    try:
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

        # Дедупликация
        unique_indices = _deduplicate(texts, embeddings, DUPLICATE_THRESHOLD)
        unique_texts = [texts[i] for i in unique_indices]
        unique_reviews = [reviews[i] for i in unique_indices]
        logger.info("Дедупликация: %d -> %d", len(texts), len(unique_texts))

        if len(unique_texts) <= max_output:
            result = _enrich_with_metadata(unique_texts, unique_reviews)
            _log_coverage(result)
            return result

        # Стратифицированный отбор по аспектам (вместо KMeans)
        selected_indices = _select_diverse(unique_texts, max_output)
        selected_texts = [unique_texts[i] for i in selected_indices]
        selected_reviews = [unique_reviews[i] for i in selected_indices]

        result = _enrich_with_metadata(selected_texts, selected_reviews)
        _log_coverage(result)
        logger.info(
            "После стратифицированного отбора: %d отзывов", len(result)
        )
        return result
    finally:
        del model
        gc.collect()
        logger.debug("Модель эмбеддингов выгружена из памяти.")


def semantic_filter(
    reviews: List[Dict[str, Any]], max_output: int = DEFAULT_MAX_OUTPUT
) -> List[Dict[str, Any]]:
    """Убирает дубликаты и стратифицированно отбирает отзывы по аспектам.
    Возвращает List[Dict] с метаданными для review_processor."""
    if not reviews:
        return []
    texts = [r.get("review", "") for r in reviews if r.get("review")]
    valid_reviews = [r for r in reviews if r.get("review")]

    if len(texts) <= max_output:
        return _enrich_with_metadata(texts, valid_reviews)

    try:
        return _semantic_filter_with_embeddings(texts, valid_reviews, max_output)
    except Exception as e:
        logger.error(
            "Семантическая фильтрация недоступна (%s: %s). "
            "Fallback: обрезка до %d отзывов.",
            type(e).__name__, e, max_output,
        )
        return _enrich_with_metadata(
            texts[:max_output], valid_reviews[:max_output]
        )


# ==========================================
# 4. Хелперы для консистентного ответа
# ==========================================
def _make_error(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message, "data": None}


def _make_text_only(intent: Any) -> Dict[str, Any]:
    return {"status": "text_only", "intent": intent, "data": None}


def _make_stub(message: str) -> Dict[str, Any]:
    return {"status": "stub", "message": message, "data": None}


# ==========================================
# 5. Главный класс-оркестратор
# ==========================================
class DataManager:
    def __init__(self, max_output: int = DEFAULT_MAX_OUTPUT, cache_dir: str = "cache"):
        self.max_output = max_output
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, appid: int, normalized_params: dict) -> str:
        api_relevant_keys = {
            "languages", "review_type", "purchase_type", "sort_by",
            "day_range", "date_range_start", "date_range_end",
            "min_playtime_hours", "max_playtime_hours",
            "hardware_os", "hardware_cpu", "hardware_gpu",
        }
        params_to_hash = {
            k: v for k, v in normalized_params.items() if k in api_relevant_keys
        }
        params_str = json.dumps(params_to_hash, sort_keys=True)
        return hashlib.md5(f"{appid}_{params_str}".encode("utf-8")).hexdigest()

    def process(self, intent_json: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._process(intent_json)
        except Exception as e:
            logger.exception("Необработанная ошибка в DataManager.process")
            return _make_error(f"Внутренняя ошибка: {type(e).__name__}: {e}")

    def _process(self, intent_json: Dict[str, Any]) -> Dict[str, Any]:
        # Шаг 1. Нормализация
        normalized = normalize_intent(intent_json)

        # Шаг 2. Роутинг
        route = route_request(normalized)
        kind = route["kind"]
        if kind == "text_only":
            return _make_text_only(normalized.get("intent"))
        if kind == "stub":
            return _make_stub(route["message"])
        if kind == "error":
            return _make_error(route["message"])

        # Шаг 3. Поиск игры
        target_game = normalized.get("target_game")
        if not target_game:
            return _make_error("Не указано название игры для анализа.")
        try:
            appid = steam_api.resolve_appid(target_game)
        except steam_api.GameNotFoundError as e:
            return _make_error(str(e))
        except steam_api.SteamAPIError as e:
            return _make_error(f"Ошибка Steam API: {e}")

        # ШАГ 4. ПРОВЕРКА КЭША
        cache_key = self._get_cache_key(appid, normalized)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            logger.info("Кэш найден: %s", cache_file)
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_data["sub_tasks"] = normalized.get("sub_tasks", [])
                return cached_data
            except Exception as e:
                logger.warning("Ошибка чтения кэша (%s). Загружаем данные заново.", e)

        # ШАГ 5. СБОР ОТЗЫВОВ
        try:
            fetch = steam_api.fetch_reviews_new_api(
                appid, normalized, target_count=DEFAULT_TARGET_REVIEWS,
            )
        except steam_api.SteamAPIError as e:
            return _make_error(f"Не удалось загрузить отзывы: {e}")
        raw_reviews = fetch["reviews"]
        api_fallback = fetch.get("hardware_fallback", False)
        raw_count = len(raw_reviews)
        if not raw_reviews:
            return _make_error(
                f"Не удалось найти отзывы для игры '{target_game}'."
            )

        # Шаг 6. Очистка и безопасность
        clean_result = preprocessor.clean_reviews(
            raw_reviews=raw_reviews,
            hardware_fallback=api_fallback,
            hw_gpu=normalized.get("hardware_gpu"),
            hw_cpu=normalized.get("hardware_cpu"),
        )
        cleaned_reviews = clean_result["reviews"]
        clean_stats = clean_result["stats"]
        cleaned_count = len(cleaned_reviews)
        if not cleaned_reviews:
            return _make_error("Все отзывы были отфильтрованы как неинформативные.")

        # Шаг 7. Семантическая обработка (теперь возвращает List[Dict])
        final_reviews = semantic_filter(cleaned_reviews, max_output=self.max_output)
        if not final_reviews:
            return _make_error("Не удалось отобрать отзывы для анализа.")

        # Логирование воронки
        logger.info(
            "Воронка %s: raw=%d -> cleaned=%d -> final=%d (%.1f%% от исходного)",
            target_game, raw_count, cleaned_count, len(final_reviews),
            100.0 * len(final_reviews) / max(raw_count, 1),
        )

        # Формируем словарь для примененных фильтров
        applied_filters = {
            key: normalized.get(key)
            for key in (
                "languages", "review_type", "purchase_type", "sort_by",
                "day_range", "date_range_start", "date_range_end",
                "min_playtime_hours", "max_playtime_hours",
                "hardware_os", "hardware_cpu", "hardware_gpu",
            )
            if normalized.get(key) not in (None, [], 0, "")
        }

        # ШАГ 8. ФОРМИРОВАНИЕ ОТВЕТА
        if len(final_reviews) < MIN_REVIEWS_FOR_SUMMARY:
            result = {
                "status": "insufficient_data",
                "message": (
                    f"Найдено только {len(final_reviews)} отзывов "
                    f"для игры '{target_game}' — summary может быть нерепрезентативным."
                ),
                "game": target_game,
                "appid": appid,
                "reviews_count": len(final_reviews),
                "reviews": final_reviews,
                "sub_tasks": normalized.get("sub_tasks", []),
                "meta": {
                    "raw_reviews_count": raw_count,
                    "cleaned_reviews_count": cleaned_count,
                    "final_reviews_count": len(final_reviews),
                    "cleaning": clean_stats,
                    "insufficient_data": True,
                    "applied_filters": applied_filters,
                },
            }
        else:
            has_hw_filter = bool(
                normalized.get("hardware_os")
                or normalized.get("hardware_gpu")
                or normalized.get("hardware_cpu")
            )
            result = {
                "status": "ready_for_summary",
                "game": target_game,
                "appid": appid,
                "reviews_count": len(final_reviews),
                "reviews": final_reviews,
                "sub_tasks": normalized.get("sub_tasks", []),
                "meta": {
                    "raw_reviews_count": raw_count,
                    "cleaned_reviews_count": cleaned_count,
                    "final_reviews_count": len(final_reviews),
                    "cleaning": clean_stats,
                    "used_hardware_filter": has_hw_filter,
                    "api_fallback_applied": api_fallback,
                    "total_matching": fetch.get("total_matching"),
                    "day_range_used": fetch.get("day_range_used"),
                    "applied_filters": applied_filters,
                },
            }

        # ШАГ 9. СОХРАНЕНИЕ В КЭШ
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info("Данные успешно сохранены в кэш: %s", cache_file)
        except Exception as e:
            logger.error("Не удалось сохранить кэш: %s", e)
        return result


# ==========================================
# Пример использования (для отладки)
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    test_intent = {
        "intent": "analyze_reviews",
        "confidence": 0.98,
        "target_game": "elden ring",
        "genre_or_tags": [],
        "sub_tasks": [],
        "languages": ["all"],
        "review_type": None,
        "purchase_type": None,
        "sort_by": None,
        "time_filter_text": None,
        "min_playtime_hours": None,
        "max_playtime_hours": None,
        "hardware_os": None,
        "hardware_cpu": None,
        "hardware_gpu": None,
        "missing_slots": [],
        "data_retrieval_needed": True,
    }
    manager = DataManager()
    result = manager.process(test_intent)
    print(f"Статус: {result.get('status')}")
    if result.get("status") == "ready_for_summary":
        print(f"Игра: {result['game']} (appid={result['appid']})")
        print(f"Отобрано отзывов: {result['reviews_count']}")
        print(f"Мета: {result['meta']}")
        if result["reviews"]:
            first = result["reviews"][0]
            text = first.get("review_text", "") if isinstance(first, dict) else first
            print(f"Первый отзыв: {text[:100]}...")
    elif result.get("status") == "error":
        print(f"Ошибка: {result['message']}")