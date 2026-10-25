import gc
import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

import steam_api
import preprocessor

logger = logging.getLogger(__name__)

# ==========================================
# Константы
# ==========================================
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
DUPLICATE_THRESHOLD = 0.85
DEFAULT_MAX_OUTPUT = 80
MAX_CLUSTERS = 7
DEFAULT_TARGET_REVIEWS = 500
MIN_REVIEWS_FOR_SUMMARY = 30

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

    # Значения по умолчанию
    if not normalized.get("languages"):
        normalized["languages"] = ["all"]
    if not normalized.get("review_type"):
        normalized["review_type"] = "all"
    if not normalized.get("purchase_type"):
        normalized["purchase_type"] = "all"
    if not normalized.get("sort_by"):
        normalized["sort_by"] = "helpful"

    # Временной фильтр: возвращает либо {"day_range": N},
    # либо {"date_range_start": ts, "date_range_end": ts}, либо {}
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
def _find_connected_components(adjacency: np.ndarray) -> Dict[int, List[int]]:
    """
    Union-Find по верхнему треугольнику матрицы смежности.
    Возвращает {корневой_индекс: [индексы компоненты]}.
    """
    n = adjacency.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows, cols = np.where(adjacency)
    for i, j in zip(rows.tolist(), cols.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return groups


def _deduplicate(
    texts: List[str], embeddings: np.ndarray, threshold: float
) -> List[int]:
    """
    Greedy-дедупликация. Для каждого отзыва в порядке «лучший представитель
    первым» проверяем: похож ли он на УЖЕ отобранные (а не на любые в цепочке).
    Если похож — пропускаем; иначе — добавляем в отобранные.

    Устраняет проблему single-linkage: цепочка A~B~C~D больше не схлопывает
    всё в один представитель, если A и D семантически различны.
    """
    sim = np.dot(embeddings, embeddings.T)
    n = len(texts)

    # Порядок: сначала самые длинные (при равенстве — по индексу).
    # Длинные информативнее коротких «топ» и «10/10».
    order = sorted(range(n), key=lambda i: (-len(texts[i]), i))

    kept: List[int] = []
    for i in order:
        if not kept:
            kept.append(i)
            continue
        # Векторизованная проверка: сходство с любым из уже отобранных
        if np.any(sim[i, kept] > threshold):
            continue
        kept.append(i)

    return sorted(kept)


def _semantic_filter_with_embeddings(
    texts: List[str], max_output: int
) -> List[str]:
    """Внутренняя реализация; может бросать исключения — обёрнута в semantic_filter."""
    logger.info(f"Семантическая фильтрация: {len(texts)} отзывов")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    try:
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

        # Дедупликация
        unique_indices = _deduplicate(texts, embeddings, DUPLICATE_THRESHOLD)
        unique_texts = [texts[i] for i in unique_indices]
        logger.info(f"Дедупликация: {len(texts)} -> {len(unique_texts)}")

        if len(unique_texts) <= max_output:
            return unique_texts

        # Кластеризация
        unique_embeddings = embeddings[unique_indices]
        n_clusters = min(MAX_CLUSTERS, len(unique_texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(unique_embeddings)

        # Равномерный отбор по кластерам
        per_cluster = max(1, max_output // n_clusters)
        selected: List[int] = []

        for cluster_id in range(n_clusters):
            idxs = np.sort(np.where(labels == cluster_id)[0])
            selected.extend(int(x) for x in idxs[:per_cluster])

        # Добираем из оставшихся, если кластеры оказались неполными
        if len(selected) < max_output:
            selected_set = set(selected)
            for idx in range(len(unique_texts)):
                if idx not in selected_set:
                    selected.append(idx)
                    if len(selected) >= max_output:
                        break

        selected = sorted(selected)[:max_output]
        result = [unique_texts[i] for i in selected]
        logger.info(
            f"После кластеризации (K={n_clusters}): {len(result)} отзывов"
        )
        return result

    finally:
        # Жёсткая очистка памяти (NFR-006)
        del model
        gc.collect()
        logger.debug("Модель эмбеддингов выгружена из памяти.")


def semantic_filter(
    reviews: List[Dict[str, Any]], max_output: int = DEFAULT_MAX_OUTPUT
) -> List[str]:
    """
    Убирает дубликаты и сбалансированно отбирает отзывы по кластерам.
    При недоступности модели (нет интернета, OOM, отсутствие пакета) —
    fallback на простую обрезку: качество ниже, но система не падает.
    """
    if not reviews:
        return []

    texts = [r.get("review", "") for r in reviews if r.get("review")]
    if len(texts) <= max_output:
        return texts

    try:
        return _semantic_filter_with_embeddings(texts, max_output)
    except Exception as e:
        logger.error(
            f"Семантическая фильтрация недоступна "
            f"({type(e).__name__}: {e}). "
            f"Fallback: обрезка до {max_output} отзывов."
        )
        return texts[:max_output]


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
    def __init__(self, max_output: int = DEFAULT_MAX_OUTPUT):
        self.max_output = max_output

    def process(self, intent_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Главный метод: от сырого JSON интента до готового списка отзывов.
        Все исключения ловятся и превращаются в {"status": "error", ...}.
        """
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

        # Шаг 4. Сбор отзывов
        try:
            fetch = steam_api.fetch_reviews_new_api(
                appid,
                normalized,
                target_count=DEFAULT_TARGET_REVIEWS,
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

        # Шаг 5. Очистка и безопасность
        # ВНИМАНИЕ: следующие два шага используют текущую сигнатуру
        # preprocessor.clean_reviews -> List[Dict]. При рефакторинге
        # preprocessor стоит возвращать структурированный результат,
        # чтобы явно сообщать о срабатывании текстового hardware-fallback.
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
            return _make_error(
                "Все отзывы были отфильтрованы как неинформативные."
            )

        # Шаг 6. Семантическая обработка
        final_texts = semantic_filter(cleaned_reviews, max_output=self.max_output)
        if not final_texts:
            return _make_error("Не удалось отобрать отзывы для анализа.")
        if len(final_texts) < MIN_REVIEWS_FOR_SUMMARY:
            return {
                "status": "insufficient_data",
                "message": (
                    f"Найдено только {len(final_texts)} отзывов "
                    f"для игры '{target_game}' — summary может быть нерепрезентативным."
                ),
                "game": target_game,
                "appid": appid,
                "reviews_count": len(final_texts),
                "reviews": final_texts,
                "meta": {
                    "raw_reviews_count": raw_count,
                    "cleaned_reviews_count": cleaned_count,
                    "final_reviews_count": len(final_texts),
                    "cleaning": clean_stats,
                    "insufficient_data": True,
                    "applied_filters": applied_filters,
                },
            }

        # Шаг 7. Формирование ответа для Map-этапа
        has_hw_filter = bool(
            normalized.get("hardware_os")
            or normalized.get("hardware_gpu")
            or normalized.get("hardware_cpu")
        )

        applied_filters = {
            key: normalized.get(key)
            for key in (
                "languages",
                "review_type",
                "purchase_type",
                "sort_by",
                "day_range",
                "date_range_start",
                "date_range_end",
                "min_playtime_hours",
                "max_playtime_hours",
                "hardware_os",
                "hardware_cpu",
                "hardware_gpu",
            )
            if normalized.get(key) not in (None, [], 0, "")
        }

        return {
            "status": "ready_for_summary",
            "game": target_game,
            "appid": appid,
            "reviews_count": len(final_texts),
            "reviews": final_texts,
            "meta": {
                # Счётчики пайплайна (для диагностики и логирования)
                "raw_reviews_count": raw_count,
                "cleaned_reviews_count": cleaned_count,
                "final_reviews_count": len(final_texts),
                # Флаги фильтрации
                "cleaning": clean_stats,
                "used_hardware_filter": has_hw_filter,
                "api_fallback_applied": api_fallback,
                # Метаданные из нового API (октябрь 2026)
                "total_matching": fetch.get("total_matching"),
                "day_range_used": fetch.get("day_range_used"),
                # Фактически применённые фильтры
                "cleaning": clean_stats,
                "applied_filters": applied_filters,
            },
        }


# ==========================================
# Пример использования (для отладки)
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO ,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_intent = {
        "intent": "analyze_reviews",
        "confidence": 0.98,
        "target_game": "hades",
        "genre_or_tags": [],
        "sub_tasks": [],
        "languages": ["japanese"],
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
            print(f"Первый отзыв: {result['reviews'][0][:100]}...")
    elif result.get("status") == "error":
        print(f"Ошибка: {result['message']}")