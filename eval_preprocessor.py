"""
Диагностика качества предобработки отзывов Steam.

Запуск:
    python eval_preprocessor.py --game "Elden Ring" --language russian
    python eval_preprocessor.py --game "Hades" --language japanese --year 2023
    python eval_preprocessor.py --game "Satisfactory" --gpu "RTX 4090"
"""
import argparse
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

import steam_api
import preprocessor
import data_manager

logger = logging.getLogger(__name__)

# Топик-словари для быстрой оценки информационного покрытия.
TOPIC_KEYWORDS = {
    "performance": {
        "fps", "fpsов", "лаги", "лагов", "тормоза", "оптимизация", "оптимизирован",
        "stutters", "lag", "optimization", "frame", "drop", "crash", "вылет", "вылетает",
        "краш", "crashes", "fps", "плавно", "60fps", "144fps",
    },
    "bugs": {
        "баг", "баги", "глюк", "глючит", "bug", "bugs", "broken", "сломан",
        "fix", "фикс", "патч", "patch", "вылет", "crash", "freeze", "зависает",
    },
    "story": {
        "сюжет", "история", "story", "narrative", "plot", "персонаж", "character",
        "лор", "lore", "концовка", "ending", "диалог", "dialogue", "quest",
    },
    "mechanics": {
        "бой", "боевка", "combat", "механика", "mechanic", "геймплей", "gameplay",
        "управление", "controls", "bосс", "boss", "battles",
    },
    "content": {
        "dlc", "контент", "content", "концовк", "hour", "часов", "часы", "прохождение",
        "playtime", "реиграбельность", "replayable", "секрет", "secret",
    },
    "price": {
        "цена", "price", "стоит", "деньги", "money", "sale", "скидка", "buy",
    },
    "recommend": {
        "рекомендую", "recommend", "не рекомендую", "not recommend", "10/10", "8/10",
    },
}


def _percentiles(values: List[int], points=(0, 25, 50, 75, 100)) -> List[int]:
    if not values:
        return [0] * len(points)
    values_sorted = sorted(values)
    result = []
    for p in points:
        idx = int(len(values_sorted) * p / 100)
        idx = min(idx, len(values_sorted) - 1)
        result.append(values_sorted[idx])
    return result


def _detect_language(text: str) -> str:
    has_cyr = any('\u0400' <= c <= '\u04FF' for c in text)
    has_lat = any('a' <= c.lower() <= 'z' for c in text)
    has_cjk = any('\u4E00' <= c <= '\u9FFF' or '\u3040' <= c <= '\u30FF' for c in text)
    if has_cjk and not has_cyr and not has_lat:
        return "cjk"
    if has_cyr and not has_lat and not has_cjk:
        return "cyrillic"
    if has_lat and not has_cyr and not has_cjk:
        return "latin"
    if not (has_cyr or has_lat or has_cjk):
        return "other"
    return "mixed"


def _topic_coverage(reviews: List[str]) -> Dict[str, int]:
    counter: Dict[str, int] = defaultdict(int)
    for text in reviews:
        lower = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                counter[topic] += 1
    return dict(counter)


def run(game: str, language: str, year: Optional[int], gpu: Optional[str],
        hardware_fallback: bool) -> None:
    print(f"\n{'=' * 70}")
    print(f"PIPELINE QUALITY REPORT")
    print(f"Game: {game!r}, language: {language!r}, "
          f"year: {year}, gpu: {gpu!r}, hw_fallback: {hardware_fallback}")
    print(f"{'=' * 70}\n")

    # --- Параметры, как будто пришли от intent ---
    time_text = str(year) if year else None
    intent = {
        "intent": "analyze_reviews",
        "confidence": 0.99,
        "target_game": game,
        "languages": [language],
        "review_type": "all",
        "purchase_type": "all",
        "sort_by": "helpful",
        "time_filter_text": time_text,
        "hardware_gpu": gpu,
        "hardware_cpu": None,
        "hardware_os": None,
        "missing_slots": [],
        "data_retrieval_needed": True,
    }
    normalized = data_manager.normalize_intent(intent)

    # --- Stage 1: Steam API ---
    print("── STAGE 1: STEAM API ──────────────────────────────")
    try:
        appid = steam_api.resolve_appid(game)
    except steam_api.SteamAPIError as e:
        print(f"❌ FAIL: {e}")
        sys.exit(1)

    fetch = steam_api.fetch_reviews_new_api(appid, normalized)
    raw = fetch["reviews"]
    print(f"  appid: {appid}")
    print(f"  fetched: {len(raw)}")
    print(f"  total_matching: {fetch.get('total_matching')}")
    print(f"  day_range_used: {fetch.get('day_range_used')}")
    print(f"  hardware_fallback (API): {fetch.get('hardware_fallback')}")
    if not raw:
        print("❌ FAIL: Steam API вернул 0 отзывов. Нечего анализировать.")
        sys.exit(1)

    # --- Stage 2: clean_reviews ---
    print("\n── STAGE 2: PREPROCESS ─────────────────────────────")
    clean = preprocessor.clean_reviews(
        raw_reviews=raw,
        hardware_fallback=hardware_fallback or fetch.get("hardware_fallback", False),
        hw_gpu=gpu,
        hw_cpu=None,
    )
    cleaned = clean["reviews"]
    stats = clean["stats"]
    gb = stats.get("garbage_breakdown", {})
    print(f"  input: {stats['input_count']}")
    print(f"  after hardware_text_filter: {stats['after_hardware_filter']}")
    print(f"  after garbage_filter: {stats['after_garbage_filter']}")
    print(f"  garbage breakdown:")
    print(f"    short (<{preprocessor.MIN_REVIEW_LENGTH}): {gb.get('dropped_short', 0)}")
    print(f"    emoji: {gb.get('dropped_emoji', 0)}")
    print(f"    no alnum: {gb.get('dropped_no_alnum', 0)}")
    print(f"    short_but_hardware_kept: {gb.get('short_but_hardware_kept', 0)}")
    print(f"  anon stripped: {gb.get('anon_stripped', {})}")
    if clean["dropped_examples"]:
        print("  dropped examples:")
        for reason, snippet in clean["dropped_examples"][:5]:
            print(f"    [{reason}] {snippet!r}")
    if not cleaned:
        print("❌ FAIL: clean_reviews вернул 0 отзывов.")
        sys.exit(1)

    # --- Stage 3: semantic_filter ---
    print("\n── STAGE 3: SEMANTIC FILTER ────────────────────────")
    final = data_manager.semantic_filter(cleaned, max_output=80)
    print(f"  after dedup+cluster: {len(final)}")

    # --- Stage 4: Quality metrics ---
    print("\n── QUALITY METRICS ─────────────────────────────────")
    lens = [len(t) for t in final]
    p = _percentiles(lens)
    print(f"  length percentiles (0/25/50/75/100): {p}")

    langs = Counter(_detect_language(t) for t in final)
    print(f"  language mix: {dict(langs)}")

    topics = _topic_coverage(final)
    print(f"  topic coverage:")
    for topic, count in sorted(topics.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(final)
        bar = "█" * (count * 40 // len(final))
        print(f"    {topic:12s} {count:3d}/{len(final)} ({pct:4.0f}%) {bar}")

    # Diversity: сколько уникальных первых 30 символов
    prefixes = {t[:30] for t in final}
    print(f"  unique prefixes: {len(prefixes)}/{len(final)}")

    # --- Warnings ---
    print("\n── WARNINGS ────────────────────────────────────────")
    warnings = []
    short_count = sum(1 for t in final if len(t) < 100)
    if short_count > len(final) * 0.2:
        warnings.append(
            f"⚠ {short_count}/{len(final)} отзывов < 100 симв. "
            f"Могут быть слишком краткими для LLM."
        )
    if len(langs) == 1 and "cjk" in langs:
        warnings.append("⚠ Все отзывы CJK — проверьте MIN_REVIEW_LENGTH (может быть мало).")
    if "performance" in topics and topics["performance"] > len(final) * 0.5:
        warnings.append(
            f"⚠ >{50}% отзывов про performance — кластеризация "
            f"может быть несбалансированной."
        )
    if not topics:
        warnings.append(
            "⚠ Ни один topic-keyword не найден — возможно, "
            "фильтр слишком агрессивен."
        )
    if len(prefixes) < len(final) * 0.7:
        warnings.append(
            f"⚠ Много дублей по префиксам ({len(prefixes)}/{len(final)}). "
            f"Проверьте _deduplicate."
        )
    if not warnings:
        print("  ✅ OK — критических проблем не обнаружено.")
    else:
        for w in warnings:
            print(f"  {w}")

    # --- Samples ---
    print("\n── SAMPLES (первые 5) ──────────────────────────────")
    for i, text in enumerate(final[:5], 1):
        print(f"  [{i}] ({len(text)} симв.) {text[:160].replace(chr(10), ' ')!r}")

    # --- Save for inspection ---
    out_path = f"eval_{game.replace(' ', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "game": game,
                "params": {"language": language, "year": year, "gpu": gpu},
                "stats": stats,
                "final_reviews": final,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nПолный отчёт сохранён: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Диагностика качества препроцессинга отзывов Steam")
    parser.add_argument("--game", required=True, help="Название игры")
    parser.add_argument("--language", default="russian", help="Язык отзывов")
    parser.add_argument("--year", type=int, default=None, help="Конкретный год (опционально)")
    parser.add_argument("--gpu", default=None, help="GPU (например, 'RTX 4090')")
    parser.add_argument("--hardware-fallback", action="store_true",
                        help="Эмулировать hardware-fallback")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args.game, args.language, args.year, args.gpu, args.hardware_fallback)


if __name__ == "__main__":
    main()