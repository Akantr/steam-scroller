import json
import logging
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from rapidfuzz import fuzz
from ollama import Client as _OllamaClient
from intent import OllamaManager, OLLAMA_MODEL_NAME
import time

logger = logging.getLogger(__name__)

# ==========================================
# Конфигурация
# ==========================================
MAP_BATCH_SIZE = 16
MAP_WORKERS = 1
MAP_MAX_FINDINGS = 12        # кап LLM-фолбэка
MAP_NUM_PREDICT = 1024
REDUCE_NUM_PREDICT = 1536
NUM_CTX = 6144
REVIEW_TRUNCATE = 600
MAX_QUOTE_LEN = 180
MIN_QUOTE_LEN = 15
QUOTE_FUZZY_THRESHOLD = 90
TOPIC_FUZZY_THRESHOLD = 82
QUOTE_SIM_THRESHOLD = 0.80
TOPIC_VALIDATE_THRESHOLD = 60
TOPIC_NORMALIZE_SIM_THRESHOLD = 0.82

# --- Extractive QA ---
EXTRACTIVE_QA_MODEL_NAME = "timpal0l/mdeberta-v3-base-squad2"
QA_SCORE_THRESHOLD = 0.10
QA_MAX_ANSWER_TOKENS = 128      # макс. span в токенах (~180–250 симв. для кириллицы)
QA_INPUT_MAX_TOKENS = 512

# --- Map: QA-путь ---
MAP_MAX_QA_PER_BATCH = 32       # сколько цитат берём с батча после дедупа/сортировки
MAP_CATEGORIZE_SUB_BATCH = 12   # саббатч для Qwen-категоризации
MAP_CATEGORIZE_NUM_PREDICT = 2048    

# --- Query reformulation ---
QUERY_MAX_QUESTIONS = 2
QUERY_REFORMULATE_NUM_PREDICT = 512

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array", "minItems": 1, "maxItems": QUERY_MAX_QUESTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "topic":    {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["topic", "question"],
            },
        },
    },
    "required": ["questions"],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_points": {
            "type": "array", "minItems": 5, "maxItems": 7,
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "description": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
                    "finding_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                },
                "required": ["feature", "description", "sentiment", "finding_ids"],
            },
        },
    },
    "required": ["summary_points"],
}

# Единственный «беззапросный» вопрос. topic=None → fallback-категоризация
# в _categorize_sub_batch для этого случая работать не будет — это осознанный
# компромисс (Qwen обычно покрывает 32/32, см. ваш run.log).
GENERIC_QA_QUESTIONS = [
    ("Какая фраза в отзыве описывает сильные стороны игры?", None),
    ("Какая фраза в отзыве описывает слабые стороны игры?", None),
]

"""
CONSTANT_QA_QUESTION = (
    "Какая фраза в отзыве выражает основное впечатление автора об игре?",
    None,
)
"""

UMBRELLA_ANCHORS = {
    "Боевка":       "Боевка: боевая система, комбо, парирование, оружие, сражения",
    "Сюжет":        "Сюжет: история, лор, повествование, нарратив, диалоги",
    "Графика":      "Графика: визуальный стиль, арт, анимации, картинка",
    "Звук":         "Звук: музыка, саундтрек, озвучка",
    "Оптимизация":  "Оптимизация: производительность, fps, стабильность",
    "Сложность":    "Сложность: трудность, челлендж",
    "Прогрессия":   "Прогрессия: прокачка, развитие",
    "Исследование": "Исследование: исследование мира, exploration",
    "Содержание":   "Содержание: контент, объём",
    "Вариативность":"Вариативность: разнообразие, билды",
    "Геймплей":     "Геймплей: механики, управление",
    "Атмосфера":    "Атмосфера: настроение, вайб",
    "Баги":         "Баги: вылеты, краши, глитчи, проблемы",
    "Цена":         "Цена: стоимость, ценник",
    "Сообщество":   "Сообщество: комьюнити",
    "Боссы":        "Боссы: боссфайты, схватки с боссами",
}

UMBRELLA_FEATURES = frozenset({
    "Боевка", "Сюжет", "Графика", "Звук", "Оптимизация",
    "Сложность", "Прогрессия", "Исследование", "Содержание",
    "Вариативность", "Геймплей", "Атмосфера", "Баги",
    "Цена", "Сообщество", "Боссы",
})

FEATURE_CHILDREN = {
    "Боевка": {
        "боевка", "боёвка", "боевая система", "комбо", "парирование",
        "оружие", "уворот", "уклонение", "сражения", "бои", "слешер",
        "combat", "combos", "parry", "weapons", "action",
        "weapon switching", "武器切替",
        "アクション", "戦闘", "战斗", "액션", "전투",
    },
    "Сюжет": {
        "сюжет", "история", "лор", "нарратив", "повествование",
        "диалоги", "записки", "концовка", "персонажи",
        "story", "plot", "scenario", "lore", "narrative", "dialogue",
        "ストーリー", "シナリオ", "故事", "剧情", "스토리",
    },
    "Графика": {
        "графика", "визуал", "визуальный стиль", "арт", "картинка",
        "анимации", "анимация", "модели", "стиль", "пейзажи", "окружение",
        "graphics", "visuals", "art", "animation", "style",
        "グラフィック", "ビジュアル", "画面", "그래픽", "비주얼",
    },
    "Звук": {
        "звук", "музыка", "саундтрек", "ost", "озвучка", "voice",
        "music", "sound", "soundtrack", "voice acting",
        "サウンド", "音楽", "ボイス", "サントラ", "사운드", "음악",
    },
    "Оптимизация": {
        "оптимизация", "производительность", "fps", "стабильность",
        "performance", "optimization", "optimisation", "stuttering",
        "техническое состояние", "техническое", "техническому", "техсостояние",
        "パフォーマンス", "最適化", "パフォ", "최적화", "퍼포먼스",
    },
    "Сложность": {
        "сложность", "трудность", "челлендж", "difficulty", "challenge",
        "challenging", "hard", "easy",
        "難易度", "難しさ", "난이도",
    },
    "Прогрессия": {
        "прогрессия", "прокачка", "развитие", "progression", "upgrade",
        "upgrades", "leveling", "growth",
        "成長", "成長要素", "進歩", "레벨업", "성장",
    },
    "Исследование": {
        "исследование", "мир", "открытый мир", "уровни", "exploration",
        "exploring", "explore", "explored", "level design", "world",
        "探索", "冒険", "맵", "탐험",
    },
    "Содержание": {
        "содержание", "контент", "content",
        "コンテンツ", "アイテム", "内容", "物品", "콘텐츠", "아이템",
    },
    "Вариативность": {
        "вариативность", "разнообразие", "билды", "сборки", "variety",
        "diversity", "builds", "build variety",
        "реиграбельность", "replayability",
        "多様性", "多様", "다양성",
    },
    "Геймплей": {
        "геймплей", "управление", "механики", "контроль",
        "gameplay", "controls", "mechanics", "movement", "traversal",
        "metroidvania",
        "ゲームプレイ", "ゲーム性", "ゲームシステム", "操作",
        "플랫폼요소", "게임플레이", "조작", "游戏性",
    },
    "Атмосфера": {
        "атмосфера", "настроение", "вайб", "atmosphere", "vibe", "mood",
        "雰囲気", "분위기",
    },
    "Баги": {
        "баги", "вылеты", "краши", "глитчи", "застревания", "проблемы",
        "issue", "bugs", "crashes", "glitches", "freezes",
        "バグ", "不具合", "버그", "缺陷",
    },
    "Цена": {
        "цена", "стоимость", "ценник", "price", "worth", "cost",
        "дорого", "дорогая", "переоценено", "высокая цена",
        "expensive", "overpriced", "not worth", "too expensive",
        "не стоит своих денег", "не оправдывает цену",
        "値段が高い", "高すぎる", "비싸다",
        "値段", "価格", "가격",
    },
    "Сообщество": {
        "сообщество", "комьюнити", "community",
        "コミュニティ", "制作側", "커뮤니티",
    },
    "Боссы": {
        "боссы", "боссфайты", "схватки с боссами",
        "bosses", "boss", "boss fight",
        "ボス", "보스",
    },
}

CLOUD_REDUCE_ENABLED = os.getenv("CLOUD_REDUCE_ENABLED", "1") == "1"
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "codestral-2508")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "mstrl_Pl2GQurhDXsQyU3HUZIvdGV117qrpooq_2FEx6A")
CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://api.mistral.ai/v1")
CLOUD_TIMEOUT = float(os.getenv("CLOUD_TIMEOUT", "25"))

CALIBRATION_DIR = Path(os.getenv("CALIBRATION_DIR", "calibration"))
PHASE0_DUMP_ENABLED = os.getenv("PHASE0_DUMP", "1") == "1"

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class LLMJSONError(Exception):
    pass


# ==========================================
# Утилиты
# ==========================================
def _smart_truncate(text: str, max_len: int = REVIEW_TRUNCATE) -> str:
    if len(text) <= max_len:
        return text
    head_len = int(max_len * 0.6)
    tail_len = max_len - head_len - 20
    head = text[:head_len].rstrip()
    tail = text[-tail_len:].lstrip()
    last_space = head.rfind(" ")
    if last_space > head_len - 50:
        head = head[:last_space]
    first_space = tail.find(" ")
    if 0 <= first_space < 50:
        tail = tail[first_space + 1:]
    return f"{head}\n... [пропущено {len(text) - head_len - tail_len} символов] ...\n{tail}"


_HIRAGANA_KATAKANA_RE = re.compile(r'[\u3040-\u30ff]')
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
_HANGUL_RE = re.compile(r'[\uac00-\ud7af]')
_CYRILLIC_RE = re.compile(r'[\u0400-\u04ff]')
_LATIN_RE = re.compile(r'[A-Za-z]')

_RU_STOPWORDS = frozenset({
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между", "это",
    "очень", "вроде", "типа", "просто", "некоторые", "всё", "ведь", "ваш",
})

_EN_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "have", "but", "not",
    "you", "are", "was", "were", "will", "they", "their", "there", "these",
    "those", "which", "what", "when", "where", "while", "into", "very",
    "much", "many", "some", "such", "only", "also", "just", "than", "then",
    "them", "been", "being", "would", "could", "should", "about", "against",
    "between", "during", "through", "because", "before", "after", "above",
    "below", "over", "under", "again", "further", "once", "here", "both",
    "even", "still", "well", "back", "down", "made", "make", "goes", "going",
})

def _coerce_topic(raw: Any) -> str:
        """Приводит topic к строке. Qwen иногда возвращает list вместо str."""
        if raw is None:
            return ""
        if isinstance(raw, list):
            return " ".join(str(x) for x in raw if x is not None).strip()
        return str(raw).strip()

_INVERSION_MARKERS = (
    " but ", " however ", " although ", " though ", " yet ",
    ", but", ", however", ", although", ", though",
    " но ", " однако ", " хотя ", ", но", ", однако", ", хотя",
)


def _has_inversion_right_after(text: str, e_char_rel: int) -> bool:
        """Проверяет, есть ли инверсия сразу после span'а (в пределах 30 символов)."""
        tail = text[e_char_rel:e_char_rel + 30].lower()
        return any(marker in " " + tail for marker in _INVERSION_MARKERS)
    
def _expand_span_to_sentence(context: str, s_char: int, e_char: int,
                                max_extra: int = 200) -> tuple:
        """Расширяет span до границ предложения, но не более max_extra символов с каждой стороны."""
        # влево до точки/!/?/начала
        left = s_char
        lo = max(0, s_char - max_extra)
        while left > lo:
            if context[left - 1] in ".!?\n":
                break
            left -= 1
        # вправо до точки/!/?/конца
        right = e_char
        hi = min(len(context), e_char + max_extra)
        while right < hi:
            if context[right] in ".!?\n":
                right += 1
                break
            right += 1
        return left, right


def _stem_ru(word: str) -> str:
    for suf in (
        "ать", "ять", "еть", "ить", "уть", "ыть",
        "ого", "его", "ому", "ему", "ыми", "ими", "ых", "их", "ой", "ей",
        "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю",
        "ом", "ем", "ам", "ям", "ах", "ях",
        "а", "я", "о", "е", "ы", "и", "у", "ю", "ь",
    ):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def _stem_en(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _content_words(text: str) -> set:
    words = re.findall(r"[\w']+", (text or "").lower(), re.UNICODE)
    result: set = set()
    for w in words:
        if _CYRILLIC_RE.search(w):
            if len(w) < 4 or w in _RU_STOPWORDS:
                continue
            result.add(_stem_ru(w))
        elif _LATIN_RE.search(w):
            if len(w) < 5 or w in _EN_STOPWORDS:
                continue
            result.add(_stem_en(w))
    return result


def _compute_grounding(point: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    description = (point.get("description") or "").strip()
    ids = point.get("finding_ids") or []
    quote_text = " ".join(
        (by_id[i].get("quote") or "") for i in ids if i in by_id
    )
    desc_words = _content_words(description)
    quote_words = _content_words(quote_text)
    if not desc_words:
        return {"desc_size": 0, "covered": 0, "ratio": 1.0, "low_grounding": False}
    covered = len(desc_words & quote_words)
    ratio = covered / len(desc_words)
    return {
        "desc_size": len(desc_words),
        "covered": covered,
        "ratio": ratio,
        "low_grounding": ratio < 0.60,
    }


def _detect_script(text: str) -> str:
    if not text:
        return "unknown"
    if _HIRAGANA_KATAKANA_RE.search(text):
        return "ja"
    counts = {
        "zh": len(_CJK_RE.findall(text)),
        "ko": len(_HANGUL_RE.findall(text)),
        "ru": len(_CYRILLIC_RE.findall(text)),
        "en": len(_LATIN_RE.findall(text)),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "unknown"


def _containment(ids_a: Optional[List[str]], ids_b: Optional[List[str]]) -> float:
    sa, sb = set(ids_a or []), set(ids_b or [])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _canonical_sentiment(value: Any) -> str:
    if not value:
        return "mixed"
    v = str(value).strip().lower()
    if v in ("positive", "положительно"):
        return "positive"
    if v in ("negative", "отрицательно"):
        return "negative"
    return "mixed"


_RU_SENTIMENT = {
    "positive": "Положительно",
    "negative": "Отрицательно",
    "mixed": "Смешанно",
}


def _render_sentiment(canonical: str) -> str:
    return _RU_SENTIMENT.get(canonical, canonical)


def _log_map_findings_batch(findings: List[Dict[str, Any]]) -> None:
    if not findings:
        return
    raw_dist = Counter(f.get("sentiment", "mixed") for f in findings)
    canon_dist = Counter(_canonical_sentiment(f.get("sentiment", "")) for f in findings)
    logger.debug(
        "[0.3] Map sentiment: raw=%s | canonical=%s",
        dict(raw_dist), dict(canon_dist),
    )
    for f in findings:
        topic = f.get("topic", "")
        quote = f.get("quote", "")
        ta = _topic_quote_score(topic, quote) if topic and quote else 0.0
        logger.debug(
            "[0.1] Map finding id=%s lang=%s/%s topic=%r alignment=%.1f weak_topic=%s qa_score=%s",
            f.get("source_id"),
            _detect_script(topic), _detect_script(quote),
            topic, ta, f.get("weak_topic", False),
            f.get("_qa_score"),
        )


def _dump_calibration_findings(findings: List[Dict[str, Any]]) -> None:
    if not PHASE0_DUMP_ENABLED or not findings:
        return
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    sample = random.sample(findings, min(150, len(findings)))
    path = CALIBRATION_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_findings.jsonl"

    quote_sims: List[Optional[float]] = [None] * len(sample)
    try:
        embedder = _get_embedder()
        topics = [f.get("topic", "") for f in sample]
        quotes = [f.get("quote", "") for f in sample]
        if any(topics) and any(quotes):
            qv = embedder.encode(
                [f"query: {t}" for t in topics],
                normalize_embeddings=True, batch_size=32,
            )
            pv = embedder.encode(
                [f"passage: {q}" for q in quotes],
                normalize_embeddings=True, batch_size=32,
            )
            quote_sims = [float(qv[i] @ pv[i]) for i in range(len(sample))]
    except Exception as e:
        logger.warning("Embedder недоступен в _dump_calibration_findings: %s", e)

    with path.open("w", encoding="utf-8") as fh:
        for i, finding in enumerate(sample):
            topic = finding.get("topic", "")
            quote = finding.get("quote", "")
            ta = _topic_quote_score(topic, quote) if topic and quote else 0.0
            row = {
                "id": finding.get("id"),
                "source_id": finding.get("source_id"),
                "topic": topic,
                "topic_ru": finding.get("topic_ru"),
                "weak_topic": finding.get("weak_topic", False),
                "sentiment": finding.get("sentiment"),
                "sentiment_canonical": _canonical_sentiment(finding.get("sentiment")),
                "quote": quote,
                "topic_alignment": ta,
                "quote_sim": quote_sims[i],
                "topic_lang": _detect_script(topic),
                "quote_lang": _detect_script(quote),
                "qa_score": finding.get("_qa_score"),
                "label": None,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("[0.5] Calibration findings → %s (%d строк)", path, len(sample))


def _dump_calibration_pairs(
    points: List[Dict[str, Any]],
    by_id: Dict[str, Dict[str, Any]],
    source: str = "local",
) -> None:
    if not PHASE0_DUMP_ENABLED or not points:
        return
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    sample = points[:50]
    path = CALIBRATION_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_pairs.jsonl"

    with path.open("w", encoding="utf-8") as fh:
        for p in sample:
            ids = p.get("finding_ids", [])
            findings = [by_id[i] for i in ids if i in by_id]
            topics = [x.get("topic", "") for x in findings]
            aligned = sum(1 for x in findings if _matches_feature(x, p.get("feature", "")))
            row = {
                "feature": p.get("feature"),
                "description": p.get("description", ""),
                "finding_ids": ids,
                "finding_topics": topics,
                "finding_topic_ru": [x.get("topic_ru") for x in findings],
                "topic_aligned_count": aligned,
                "topic_aligned_share": aligned / max(len(findings), 1),
                "is_umbrella": p.get("feature") in UMBRELLA_FEATURES,
                "weak_evidence": p.get("weak_evidence", False),
                "label": None,
            }
            g = p.get("_grounding") or _compute_grounding(p, by_id)
            row["low_grounding"] = g["low_grounding"]
            row["grounding_ratio"] = round(g["ratio"], 3)
            row["source"] = source
            row["cloud_model"] = CLOUD_MODEL if source == "cloud" else None
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("[0.5] Calibration pairs → %s (%d строк)", path, len(sample))


def _topic_quote_score(topic: str, quote: str) -> float:
    t = (topic or "").lower().strip()
    q = (quote or "").lower().strip()
    if not t or not q:
        return 0.0
    return float(max(fuzz.partial_ratio(t, q), fuzz.token_set_ratio(t, q)))


_UMBRELLA_ANCHOR_CACHE: Optional[Dict[str, Any]] = None


def _get_umbrella_anchor_vectors() -> Optional[Dict[str, Any]]:
    global _UMBRELLA_ANCHOR_CACHE
    if _UMBRELLA_ANCHOR_CACHE is not None:
        return _UMBRELLA_ANCHOR_CACHE
    try:
        embedder = _get_embedder()
    except Exception as e:
        logger.warning("Embedder недоступен для anchor-vectors: %s", e)
        return None
    anchors = list(UMBRELLA_ANCHORS.keys())
    phrases = [f"passage: {UMBRELLA_ANCHORS[a]}" for a in anchors]
    vecs = embedder.encode(phrases, normalize_embeddings=True, batch_size=32)
    _UMBRELLA_ANCHOR_CACHE = {a: vecs[i] for i, a in enumerate(anchors)}
    return _UMBRELLA_ANCHOR_CACHE


def _normalize_topics_batch(findings: List[Dict[str, Any]]) -> None:
    if not findings:
        return
    anchor_vecs = _get_umbrella_anchor_vectors()
    if anchor_vecs is None:
        for f in findings:
            f["topic_ru"] = None
        return
    try:
        embedder = _get_embedder()
    except Exception as e:
        logger.warning("Embedder недоступен для нормализации topic: %s", e)
        for f in findings:
            f["topic_ru"] = None
        return

    topics = [(f.get("topic") or "") for f in findings]
    if not any(topics):
        for f in findings:
            f["topic_ru"] = None
        return

    query_vecs = embedder.encode(
        [f"query: {t}" for t in topics],
        normalize_embeddings=True, batch_size=32,
    )

    umbrellas = list(anchor_vecs.keys())
    anchor_matrix = np.stack([anchor_vecs[u] for u in umbrellas])

    for i, f in enumerate(findings):
        topic = topics[i]
        sims = anchor_matrix @ query_vecs[i]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        topic_lang = _detect_script(topic)
        threshold = TOPIC_NORMALIZE_SIM_THRESHOLD
        if topic_lang in ("ja", "zh", "ko"):
            threshold = 0.75
        if best_sim < threshold:
            f["topic_ru"] = None
            logger.debug(
                "[R2.2] topic_orig=%r → topic_ru=None (sim=%.2f < %.2f, lang=%s)",
                topic, best_sim, threshold, topic_lang,
            )
            continue

        best_umbrella = umbrellas[best_idx]
        umbrella_lang = _detect_script(best_umbrella)
        if topic_lang == "ru" and umbrella_lang == "ru":
            lex = _topic_quote_score(best_umbrella, topic)
            if lex < 50:
                f["topic_ru"] = None
                logger.debug(
                    "[R2.2] topic_orig=%r → topic_ru=None (lex=%.1f < 50, embedding wanted %r at %.2f)",
                    topic, lex, best_umbrella, best_sim,
                )
                continue

        f["topic_ru"] = best_umbrella
        logger.debug(
            "[R2.2] topic_orig=%r → topic_ru=%r (sim=%.2f, lang=%s)",
            topic, f["topic_ru"], best_sim, topic_lang,
        )


def _format_review_for_llm(idx: int, r: Union[str, Dict[str, Any]]) -> str:
    if isinstance(r, str):
        return f"[Отзыв {idx} | id={idx}]\n{_smart_truncate(r)}"
    tags: List[str] = []
    if r.get("was_updated"):
        tags.append("ОТЗЫВ БЫЛ ДОПОЛНЕН ПОЗЖЕ")
    if r.get("playtime_forever", 0) > 100 and r.get("was_updated"):
        tags.append(f"ГЛУБОКОЕ ПОГРУЖЕНИЕ ({r['playtime_forever']:.0f} ч)")
    playtime_delta = r.get("playtime_forever", 0) - r.get("playtime_at_review", 0)
    if playtime_delta > 50 and not r.get("was_updated"):
        tags.append(f"ПРОДОЛЖИЛ ИГРАТЬ ПОСЛЕ ОТЗЫВА (+{playtime_delta:.0f} ч)")
    if r.get("weighted_vote_score", 0) > 0.8:
        tags.append("ВЫСОКИЙ АВТОРИТЕТ В СООБЩЕСТВЕ")
    elif r.get("weighted_vote_score", 0) < 0.3 and r.get("votes_up", 0) < 5:
        tags.append("НИШЕВЫЙ ОТЗЫВ")
    if r.get("comment_count", 0) > 20:
        tags.append(f"ВЫЗВАЛ ДИСКУССИЮ ({r['comment_count']} комм.)")
    if r.get("is_probable_meme"):
        tags.append("ЮМОРИСТИЧЕСКИЙ/ИРОНИЧНЫЙ")

    playtime = r.get("playtime_forever", 0)
    header = f"[Отзыв {idx} | id={idx} | Наиграно: {playtime:.0f}ч"
    if tags:
        header += " | " + " | ".join(tags)
    header += "]"
    text = r.get("review_text") or r.get("review") or ""
    return f"{header}\n{_smart_truncate(text)}"


def _normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _quote_in_text(quote: str, source_text: str) -> bool:
    cq = _normalize_for_match(quote)
    cs = _normalize_for_match(source_text)
    if not cq or not cs:
        return False
    if cq in cs:
        return True
    return fuzz.partial_ratio(cq, cs) >= QUOTE_FUZZY_THRESHOLD


def _validate_finding(finding: Any, batch: List) -> Optional[Dict[str, Any]]:
    if not isinstance(finding, dict):
        return None
    sid = finding.get("source_id")
    if not isinstance(sid, int) or not (1 <= sid <= len(batch)):
        return None

    quote = (finding.get("quote") or "").strip()
    if not (MIN_QUOTE_LEN <= len(quote) <= MAX_QUOTE_LEN):
        return None
    if any(ch in quote for ch in "☐☑✓✗"):
        return None

    src = batch[sid - 1]
    src_text = (
        (src.get("review_text") or src.get("review") or "")
        if isinstance(src, dict) else str(src)
    )
    if not _quote_in_text(quote, src_text):
        return None

    topic = _coerce_topic(finding.get("topic"))
    if not topic:
        return None

    if topic.lower() in quote.lower():
        weak_topic = False
        reason = "substring"
    else:
        ta = _topic_quote_score(topic, quote)
        weak_topic = ta < TOPIC_VALIDATE_THRESHOLD
        reason = f"fuzz(ta={ta:.0f})"

    if weak_topic:
        logger.debug(
            "Weak topic: id=%s topic=%r quote=%r reason=%s",
            sid, topic, quote[:60], reason,
        )

    return {
        "source_id": sid,
        "topic": topic,
        "sentiment": finding.get("sentiment", "mixed"),
        "quote": quote,
        "weak_topic": weak_topic,
    }


def _topic_aligned(finding_topic: str, feature: str) -> bool:
    return fuzz.token_set_ratio(
        finding_topic.lower().strip(),
        feature.lower().strip(),
    ) >= TOPIC_FUZZY_THRESHOLD


_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("intfloat/multilingual-e5-small")
    return _EMBEDDER


def _matches_feature(finding: Dict[str, Any], feature: str) -> bool:
    if not feature:
        return False
    feature_lower = feature.lower().strip()
    topic_ru = (finding.get("topic_ru") or "").strip()
    topic_orig = (finding.get("topic") or "").strip()
    topic_orig_lang = _detect_script(topic_orig) if topic_orig else "unknown"
    trust_topic_ru = topic_orig_lang not in ("ja", "zh", "ko")

    resolved_umbrella: Optional[str] = None
    for umbrella, children in FEATURE_CHILDREN.items():
        if feature_lower in {c.lower() for c in children}:
            resolved_umbrella = umbrella
            break

    if resolved_umbrella is not None:
        children_lower = {c.lower() for c in FEATURE_CHILDREN.get(resolved_umbrella, set())}
        orig_match = bool(topic_orig and topic_orig.lower() in children_lower)
        if not trust_topic_ru:
            return orig_match
        ru_match = bool(topic_ru and topic_ru.lower() in children_lower)
        return ru_match or orig_match

    if trust_topic_ru and topic_ru and topic_ru.lower() == feature_lower:
        return True
    if topic_orig and topic_orig.lower() == feature_lower:
        return True
    if topic_orig and _topic_quote_score(feature, topic_orig) >= 70:
        return True
    return False


# ==========================================
# ReviewProcessor
# ==========================================
class ReviewProcessor:
    def __init__(
        self,
        model_name: str = OLLAMA_MODEL_NAME,
        batch_size: int = MAP_BATCH_SIZE,
        ollama_manager: Optional[OllamaManager] = None,
    ):
        self.ollama = ollama_manager if ollama_manager else OllamaManager(model_name=model_name)
        self.model_name = model_name
        self.ollama_client = _OllamaClient(host=self.ollama.base_url)
        self.batch_size = batch_size

        from cloud_llm import CloudLLM, CloudLLMError
        self._CloudLLMError = CloudLLMError
        _api_key = CLOUD_API_KEY or os.getenv("MISTRAL_API_KEY", "").strip()
        if CLOUD_REDUCE_ENABLED and _api_key:
            self.cloud = CloudLLM(
                model=CLOUD_MODEL, api_key=_api_key,
                base_url=CLOUD_BASE_URL, timeout=CLOUD_TIMEOUT,
            )
        else:
            if CLOUD_REDUCE_ENABLED and not _api_key:
                logger.warning("CLOUD_REDUCE_ENABLED=1 но API-ключ пуст → local fallback")
            self.cloud = None
        self._last_valid_points: Optional[List[Dict[str, Any]]] = None

        # === Extractive QA (transformers v5+) ===
        logger.info("Загрузка Extractive QA модели: %s", EXTRACTIVE_QA_MODEL_NAME)
        try:
            self.qa_tokenizer = AutoTokenizer.from_pretrained(EXTRACTIVE_QA_MODEL_NAME)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                EXTRACTIVE_QA_MODEL_NAME,
                device_map="auto",
            )
            self.qa_model.eval()
            logger.info("Extractive QA модель успешно загружена.")
        except Exception as e:
            logger.error("Не удалось загрузить Extractive QA модель: %s", e)
            self.qa_model = None
            self.qa_tokenizer = None

    def _embedding_best(self, feature, description, ids, by_id):
        try:
            embedder = _get_embedder()
        except Exception as e:
            logger.warning("Embedder недоступен: %s", e)
            return None
        query = f"query: {feature}. {description}"
        passages = [f"passage: {by_id[i]['quote']}" for i in ids]
        qv = embedder.encode(query, normalize_embeddings=True)
        pv = embedder.encode(passages, normalize_embeddings=True)
        sims = (pv @ qv).tolist()
        best_k = max(range(len(sims)), key=lambda k: sims[k])
        if sims[best_k] < QUOTE_SIM_THRESHOLD:
            return None
        return ids[best_k]

    def process(self, dm_output: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
        if dm_output.get("status") not in ("ready_for_summary", "insufficient_data"):
            return dm_output
        reviews = dm_output.get("reviews", [])
        game = dm_output.get("game", "Неизвестная игра")
        meta = dm_output.get("meta", {})
        if not reviews:
            return {"status": "error", "message": "Нет отзывов для анализа.", "data": None}

        warnings = self._extract_warnings(dm_output)
        try:
            logger.info("Запуск Map-этапа для %d отзывов...", len(reviews))
            mapped_findings = self._map_reviews(reviews, game, user_query)
            if not mapped_findings:
                return {
                    "status": "error",
                    "message": "Не удалось извлечь полезную информацию из отзывов.",
                    "data": None,
                }
            self._last_valid_points = None
            logger.info("Запуск Reduce-этапа (саммари)...")
            summary_md = self._reduce_summary(mapped_findings, game, user_query)
            table_md = None
            sub_tasks = dm_output.get("sub_tasks", [])
            if sub_tasks and "table" in sub_tasks:
                logger.info("Запуск Reduce-этапа (таблица)...")
                table_md = self._reduce_table(mapped_findings, game, user_query)
            return {
                "status": "success",
                "game": game,
                "summary": summary_md,
                "table": table_md,
                "warnings": warnings,
                "meta": meta,
            }
        except Exception as e:
            logger.exception("Критическая ошибка в ReviewProcessor")
            return {
                "status": "error",
                "message": f"Ошибка генерации саммари: {type(e).__name__}: {e}",
                "data": None,
            }

    def _extract_warnings(self, dm_output: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        meta = dm_output.get("meta", {})
        raw_reviews_count = meta.get("raw_reviews_count")
        if raw_reviews_count is not None and raw_reviews_count < 200:
            warnings.append(
                "Внимание: из Steam загружено менее 200 отзывов "
                f"({raw_reviews_count}). Сводка может быть нерепрезентативной."
            )
        if meta.get("days_since_last_review", 0) > 60:
            warnings.append(
                "Внимание: Последний отзыв был написан более 2 месяцев назад. "
                "Информация может быть устаревшей."
            )
        return warnings

    # ---------- Extractive QA ----------
    def _extract_qa_span(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """Извлекает БАЙТ-ТОЧНЫЙ фрагмент context (через offset_mapping),
        а не декодированные токены. Это гарантирует, что _quote_in_text
        пройдёт по подстроке, а не только по fuzz.
        """
        if self.qa_model is None or self.qa_tokenizer is None:
            return None
        try:
            enc = self.qa_tokenizer(
                question, context,
                return_tensors="pt",
                truncation=True,
                max_length=QA_INPUT_MAX_TOKENS,
                return_offsets_mapping=True,
            )
        except Exception as e:
            logger.warning("Токенизация для QA не удалась: %s", e)
            return None

        offset_mapping = enc["offset_mapping"]          # [chunks, seq, 2]
        sequence_ids = [enc.sequence_ids(i) for i in range(enc["input_ids"].shape[0])]
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        device = next(self.qa_model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = self.qa_model(input_ids=input_ids, attention_mask=attention_mask)

        best: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for c in range(input_ids.shape[0]):
            seq_ids = sequence_ids[c]
            offsets = offset_mapping[c]
            ctx_mask = torch.tensor(
                [s == 1 for s in seq_ids], dtype=torch.bool, device=device
            )
            if ctx_mask.sum() == 0:
                continue

            start_logits = out.start_logits[c].masked_fill(~ctx_mask, -1e9)
            end_logits = out.end_logits[c].masked_fill(~ctx_mask, -1e9)
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)

            k = min(20, int(ctx_mask.sum().item()))
            s_top = torch.topk(start_probs, k=k).indices.tolist()
            e_top = torch.topk(end_probs, k=k).indices.tolist()

            for s_idx in s_top:
                for e_idx in e_top:
                    if e_idx < s_idx:
                        continue
                    if e_idx - s_idx + 1 > QA_MAX_ANSWER_TOKENS:
                        continue
                    s_char = offsets[s_idx][0].item()
                    e_char = offsets[e_idx][1].item()
                    if s_char >= e_char:
                        continue
                    answer = context[s_char:e_char].strip()
                    if not (MIN_QUOTE_LEN <= len(answer) <= MAX_QUOTE_LEN):
                        continue
                    score = (start_probs[s_idx] * end_probs[e_idx]).item()
                    if score > best_score:
                        best_score = score
                        best = {
                            "answer": answer,
                            "score": score,
                            "span": (s_char, e_char),
                        }
        if best:
            s_char, e_char = best["span"]
            expanded_s, expanded_e = _expand_span_to_sentence(context, s_char, e_char)
            expanded = context[expanded_s:expanded_e].strip()

            # Если после span'а идёт «but/however/но» — оригинальный span искажает смысл
            if _has_inversion_right_after(context, e_char) and len(expanded) <= MAX_QUOTE_LEN:
                best = {
                    "answer": expanded,
                    "score": best["score"] * 0.9,   # лёгкий штраф за расширение
                    "span": (expanded_s, expanded_e),
                }
                logger.debug(
                    "Span расширен из-за инверсии: %r → %r",
                    context[s_char:e_char], expanded[:80],
                )
                        
        if best and best["score"] >= QA_SCORE_THRESHOLD:
            return best
        return None

    def _collect_qa_quotes(
    self, batch: List, query_questions: List[tuple],
) -> List[Dict[str, Any]]:
        if self.qa_model is None:
            return []
        questions = self._generate_questions_for_qa(query_questions)
        query_topics = {t.lower() for _, t in query_questions if t}
        collected: List[Dict[str, Any]] = []

        review_texts = [
            (r.get("review_text") or r.get("review") or "") if isinstance(r, dict) else str(r)
            for r in batch
        ]

        for q, topic_hint in questions:
            spans = self._extract_qa_spans_batch(q, review_texts)
            for i, r in enumerate(spans, 1):
                if not r:
                    continue

                # буст по теме из запроса (а не по позиции вопроса)
                boost = 1.15 if (topic_hint and topic_hint.lower() in query_topics) else 1.0
                length_factor = (
                    0.85 + 0.30 * min(len(r["answer"]), MAX_QUOTE_LEN) / MAX_QUOTE_LEN
                )

                collected.append({
                    "source_id": i,
                    "quote": r["answer"],
                    "span": r["span"],
                    "_qa_score": r["score"] * boost * length_factor,
                    "_qa_topic": topic_hint,
                })

        # дедуп по (source_id, span) — span теперь координаты внутри конкретного контекста
        dedup: Dict[tuple, Dict[str, Any]] = {}
        for c in collected:
            key = (c["source_id"], c["span"])
            if key not in dedup or c["_qa_score"] > dedup[key]["_qa_score"]:
                dedup[key] = c

        result = sorted(dedup.values(), key=lambda x: -x["_qa_score"])
        return result[:MAP_MAX_QA_PER_BATCH]

    def _categorize_quotes(self, extracted: List[Dict[str, Any]], game: str) -> List[Dict[str, Any]]:
        """Qwen расставляет topic/sentiment по индексам, НЕ пере-эмитит quote.
        Это защищает от искажений и экономит output-токены.
        """
        out: List[Dict[str, Any]] = []
        for offset in range(0, len(extracted), MAP_CATEGORIZE_SUB_BATCH):
            sub = extracted[offset:offset + MAP_CATEGORIZE_SUB_BATCH]
            out.extend(self._categorize_sub_batch(sub, game, offset))
        return out

    def _categorize_sub_batch(
        self, sub: List[Dict[str, Any]], game: str, offset: int,
    ) -> List[Dict[str, Any]]:
        lines = "\n".join(
            f"[{offset + j}] «{e['quote']}»" for j, e in enumerate(sub)
        )
        prompt = f"""
Ты — аналитик отзывов Steam. Для КАЖДОЙ пронумерованной цитаты из отзывов на игру "{game}"
определи ТЕМУ и ТОНАЛЬНОСТЬ. Обязательно покрой ВСЕ индексы от {offset} до {offset + len(sub) - 1}.
Не пропускай ни одного индекса — даже если цитата кажется спорной.

Для каждого индекса верни ровно одну запись:
{{"idx": <целое>, "topic": "<1–3 слова>", "sentiment": "positive|negative|mixed"}}

"topic" — краткое название категории: «Боевка», «Сюжет», «Графика», «Звук», «Оптимизация»,
«Сложность», «Баги», «Цена», «Боссы», «Прогрессия», «Геймплей», «Атмосфера», «Содержание»,
«Исследование», «Сообщество», «Вариативность», «Функции». НЕ цитируй сами цитаты.

Верни строго JSON:
{{"findings": [{{"idx": 0, "topic": "Боевка", "sentiment": "positive"}}]}}

ЦИТАТЫ:
{lines}
"""
        result = self._call_llm(
            prompt, "Map-категоризация",
            num_predict=MAP_CATEGORIZE_NUM_PREDICT,
            repeat_penalty=1.0,   # ← структура повторяется по дизайну, штраф ломает вывод
        )
        by_idx: Dict[int, Dict[str, Any]] = {}
        for f in result.get("findings", []):
            if not isinstance(f, dict):
                continue
            idx = f.get("idx")
            if not isinstance(idx, int):
                continue
            by_idx[idx] = f

        out: List[Dict[str, Any]] = []
        fallback_count = 0
        for j, e in enumerate(sub):
            idx = offset + j
            meta = by_idx.get(idx)
            topic = _coerce_topic(meta.get("topic") if meta else None)
            sentiment = (meta.get("sentiment") if meta else None) or "mixed"

            if not topic:
                # Qwen промолчал или вернул мусор → fallback из вопроса
                fallback = e.get("_qa_topic")
                if not fallback:
                    logger.debug(
                        "Categorize: idx=%d не покрыт Qwen и нет fallback — дроп", idx,
                    )
                    continue
                topic = fallback
                sentiment = "mixed"
                fallback_count += 1

            out.append({
                "source_id": e["source_id"],
                "quote": e["quote"],
                "topic": topic,
                "sentiment": sentiment,
                "_qa_score": e["_qa_score"],
                "_from_fallback": meta is None or not _coerce_topic(meta.get("topic")),
            })
        if fallback_count:
            logger.info(
                "Categorize: %d/%d через fallback (Qwen не покрыл)", fallback_count, len(sub),
            )
        return out


    # ---------- Map ----------
    def _map_reviews(self, reviews, game, user_query):
        query_questions = self._reformulate_query(user_query)
        if query_questions:
            logger.info("Query-reformulated: %s", [t for _, t in query_questions])
        elif user_query:
            logger.info("Query-reformulated: пусто → CONSTANT_QA_QUESTION")

        batches = [
            reviews[i : i + self.batch_size]
            for i in range(0, len(reviews), self.batch_size)
        ]
        all_findings: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=MAP_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._process_batch_safe, b, game, user_query, query_questions,
                ): b
                for b in batches
            }
            for future in as_completed(futures):
                try:
                    batch_findings = future.result()
                    if isinstance(batch_findings, list):
                        all_findings.extend(batch_findings)
                except Exception as e:
                    logger.error("Батч упал целиком: %s", e)

        for i, f in enumerate(all_findings):
            f["id"] = f"f{i}"
        _normalize_topics_batch(all_findings)
        logger.info("Map-этап собрал %d фактов.", len(all_findings))
        _dump_calibration_findings(all_findings)

        # опционально: сигнал «запроса нет в отзывах»
        if query_questions:
            generic_topics = {"", "общее впечатление", "общее"}
            thematic = [(q, t) for q, t in query_questions
                        if (t or "").strip().lower() not in generic_topics]
            if thematic and not any(
                _matches_feature(f, t) for f in all_findings for _, t in thematic
            ):
                logger.info(
                    "Запрос %r: темы %s не найдены среди %d findings",
                    user_query, [t for _, t in thematic], len(all_findings),
                )

        return all_findings

    def _process_batch_safe(
        self, batch: List, game: str, user_query: str, query_questions: List[tuple],
    ) -> List[Dict[str, Any]]:
        try:
            return self._process_batch(batch, game, user_query, query_questions)
        except LLMJSONError as e:
            if len(batch) <= 1:
                logger.warning("Split-retry: единственный отзыв не дал JSON (%s). Дроп.", e)
                return []
            mid = len(batch) // 2
            logger.info("Split-retry: %d -> %d + %d", len(batch), mid, len(batch) - mid)
            return (
                self._process_batch_safe(batch[:mid], game, user_query, query_questions)
                + self._process_batch_safe(batch[mid:], game, user_query, query_questions)
            )

    def _reformulate_query(self, user_query: str) -> List[tuple]:
        """user_query → [(question, topic), ...], ≤ QUERY_MAX_QUESTIONS.
        Пустой список = _generate_questions_for_qa вернёт GENERIC_QA_QUESTIONS."""
        if not user_query or not user_query.strip():
            return []

        prompt = f"""
    Ты — аналитик отзывов Steam. Пользователь задал запрос:
    "{user_query}"

    Преврати запрос в 1–2 вопроса для extractive-QA модели (SQuAD-стиль).

    ПРАВИЛА:
    1. Каждый вопрос ОБЯЗАН начинаться с "Какая фраза в отзыве" и заканчиваться "?".
    Запрещены yes/no ("есть ли...", "хороша ли...", "стоит ли...").
    Запрещены общие вопросы ("что говорят об игре").
    2. "topic" — короткое название темы (1–3 слова), ДОЛЖНО встречаться в "question"
    как подстрока. Тема может быть ЛЮБОЙ (крафт, кооператив, вес оружия,
    интерфейс) — не ограничивайся типовыми категориями.
    3. Если запрос составной ("боевка и оптимизация") — верни 2 вопроса.
    Первым — то, что в запросе важнее.
    4. Если запрос слишком общий — верни ровно один:
    {{"topic": "общее впечатление",
        "question": "Какая фраза в отзыве выражает основное впечатление автора об игре?"}}
    5. Не выдумывай темы, которых в запросе нет.

    ПРИМЕРЫ:

    Запрос: "что с оптимизацией и боевкой?"
    Ответ: {{"questions": [
    {{"topic": "Оптимизация", "question": "Какая фраза в отзыве описывает оптимизацию и FPS?"}},
    {{"topic": "Боевка", "question": "Какая фраза в отзыве описывает боевую систему?"}}
    ]}}

    Запрос: "составь сводку по игре"
    Ответ: {{"questions": [
    {{"topic": "общее впечатление", "question": "Какая фраза в отзыве выражает основное впечатление автора об игре?"}}
    ]}}
    """

        result: Dict[str, Any] = {}

        # 1) cloud — быстрее, не грузит GPU перед QA
        if self.cloud is not None and self.cloud.enabled:
            try:
                result = self.cloud.chat_json(prompt, QUERY_SCHEMA, "query_reformulate")
            except self._CloudLLMError as e:
                logger.warning("Query-reformulate (cloud) failed: %s → local", e)

        # 2) local Qwen
        if not result:
            try:
                result = self._call_llm(
                    prompt, "Query-reformulate",
                    num_predict=QUERY_REFORMULATE_NUM_PREDICT,
                )
            except Exception as e:
                logger.warning("Query-reformulate (local) failed: %s → CONSTANT", e)
                return []

        # 3) пост-валидация формата (SQuAD + topic ⊆ question)
        out: List[tuple] = []
        for q in result.get("questions", [])[:QUERY_MAX_QUESTIONS]:
            if not isinstance(q, dict):
                continue
            topic = _coerce_topic(q.get("topic"))
            question = _coerce_topic(q.get("question"))
            if not topic:
                continue
            if (not question.endswith("?")
                    or "отзыв" not in question.lower()
                    or topic.lower() not in question.lower()):
                question = f"Какая фраза в отзыве описывает {topic}?"
            out.append((question, topic))
        return out
    
    #def _generate_questions_for_qa(self, query_questions: List[tuple]) -> List[tuple]:
    #"""Единая точка. Без запроса — 1 CONSTANT, с запросом — 1–2 из LLM."""
    #if query_questions:
    #    return query_questions
    #return [CONSTANT_QA_QUESTION]

    def _generate_questions_for_qa(self, query_questions: List[tuple]) -> List[tuple]:
        if not query_questions:
            return GENERIC_QA_QUESTIONS
        # LLM вернул fallback «общее впечатление» → тоже 2 catch-all
        generic_topics = {"", "общее впечатление", "общее"}
        if all((t or "").strip().lower() in generic_topics for _, t in query_questions):
            return GENERIC_QA_QUESTIONS
        return query_questions

    def _process_batch(
        self, batch: List, game: str, user_query: str, query_questions: List[tuple],
    ) -> List[Dict[str, Any]]:
        if self.qa_model is not None:
            _t0 = time.perf_counter()
            extracted = self._collect_qa_quotes(batch, query_questions)   # ← изменено
            if extracted:
                _t1 = time.perf_counter()
                findings = self._categorize_quotes(extracted, game)
                _t2 = time.perf_counter()
                logger.info(
                    "batch(len=%d): qa=%.2fs categorize=%.2fs n=%d",
                    len(batch), _t1 - _t0, _t2 - _t1, len(extracted),
                )
                valid: List[Dict[str, Any]] = []
                for f in findings:
                    v = _validate_finding(f, batch)
                    if v:
                        v["_qa_score"] = f.get("_qa_score")
                        valid.append(v)
                if valid:
                    logger.debug(
                        "QA-путь: extracted=%d, categorized=%d, valid=%d",
                        len(extracted), len(findings), len(valid),
                    )
                    _log_map_findings_batch(valid)
                    return valid[:MAP_MAX_QA_PER_BATCH]
            
            logger.info("QA-путь не дал валидных findings → LLM fallback")

        return self._process_batch_llm_only(batch, game, user_query)

    def _process_batch_llm_only(
        self, batch: List, game: str, user_query: str
    ) -> List[Dict[str, Any]]:
        """Fallback: извлечение цитат только LLM (исходная логика)."""
        formatted = [_format_review_for_llm(i, r) for i, r in enumerate(batch, 1)]
        reviews_text = "\n\n---\n\n".join(formatted)

        prompt = f"""
Ты — аналитик отзывов Steam. Проанализируй отзывы на игру "{game}".
Извлеки до {MAP_MAX_FINDINGS} САМЫХ ВАЖНЫХ фактов (сюжет, графика, звук,
оптимизация, баги, цена, сложность и т.д.).

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Каждый finding обязан содержать "source_id" — целое число из заголовка
   [Отзыв N | id=N], откуда взята цитата. Без source_id finding бесполезен.
2. "quote" — ТОЧНАЯ ПОДСТРОКА из отзыва на оригинальном языке с указанным source_id, 15–{MAX_QUOTE_LEN} символов.
   Запрещены: чеклисты (☐, ☑), наборы цифр, эмодзи-списки.
3. Не более {MAP_MAX_FINDINGS} findings на батч. Лучше 3 достоверных, чем 6 притянутых.
4. "topic" — ДОСЛОВНАЯ ПОДСТРОКА ИЗ ЦИТАТЫ (1–4 слова, скопировано символ в символ).
   - Не переводи, не перефразируй, не подбирай синонимы.
   - Если в цитате нет осмысленной фразы-темы — верни "topic": "".
   - Проверь себя: topic должен встречаться в quote как подстрока.
5. "sentiment" — positive / negative / mixed.

ПРАВИЛА РАБОТЫ С ИРОНИЕЙ:
1. ЮМОРИСТИЧЕСКИЙ/ИРОНИЧНЫЙ отзыв — только как дополнительное подтверждение.
2. Ищи ИНВЕРСИЮ СМЫСЛА: если оценка противоречит описанию — верь описанию.
3. Гиперболы — не факты.
4. Отзыв без описания игрового опыта — пропускай. Пустой результат корректен.
5. Только текст отзывов, никаких знаний об игре "из головы".

ПРИМЕРЫ ХОРОШИХ findings (topic — подстрока цитаты):
{{"source_id": 5, "quote": "Боссы требуют заучивания паттернов, но каждый бой честный", "topic": "Боссы", "sentiment": "positive"}}
{{"source_id": 12, "quote": "The combat alone is TOP TIER. The fluidity, the animations...", "topic": "combat", "sentiment": "positive"}}

Верни строго JSON:
{{"findings": [
  {{"source_id": 3, "quote": "Игра летает даже на старом ПК", "topic": "ПК", "sentiment": "positive"}}
]}}

ОТЗЫВЫ ДЛЯ РАБОТЫ:
{reviews_text}
"""
        result = self._call_llm(prompt, "Map-этап", num_predict=MAP_NUM_PREDICT)
        findings = result.get("findings", [])

        valid: List[Dict[str, Any]] = []
        for f in findings:
            v = _validate_finding(f, batch)
            if v:
                valid.append(v)

        rejected = len(findings) - len(valid)
        if rejected:
            logger.debug("Map(LLM): %d/%d findings отброшено валидацией", rejected, len(findings))

        _log_map_findings_batch(valid)
        return valid[:MAP_MAX_FINDINGS]

    def _extract_qa_spans_batch(
        self, question: str, contexts: List[str],
    ) -> List[Optional[Dict[str, Any]]]:
        if not contexts or self.qa_model is None:
            return [None] * len(contexts)

        enc = self.qa_tokenizer(
            [question] * len(contexts),
            contexts,
            padding=True,
            truncation=True,
            max_length=QA_INPUT_MAX_TOKENS,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        device = next(self.qa_model.parameters()).device
        offsets = enc["offset_mapping"]                    # [N, L, 2]
        attn = enc["attention_mask"].to(device)            # [N, L]
        input_ids = enc["input_ids"].to(device)            # [N, L]

        with torch.no_grad():
            out = self.qa_model(input_ids=input_ids, attention_mask=attn)

        results: List[Optional[Dict[str, Any]]] = []

        for b in range(len(contexts)):
            seq_ids_b = enc.sequence_ids(b)
            ctx_mask = torch.tensor(
                [s == 1 for s in seq_ids_b], dtype=torch.bool, device=device,
            )
            if ctx_mask.sum() == 0:
                results.append(None)
                continue

            s_log = out.start_logits[b].masked_fill(~ctx_mask, -1e9)
            e_log = out.end_logits[b].masked_fill(~ctx_mask, -1e9)
            s_p, e_p = torch.softmax(s_log, -1), torch.softmax(e_log, -1)

            best, best_score = None, -1.0
            k = min(20, int(ctx_mask.sum()))
            s_top = torch.topk(s_p, k=k).indices.tolist()
            e_top = torch.topk(e_p, k=k).indices.tolist()

            for s_idx in s_top:
                for e_idx in e_top:
                    if e_idx < s_idx:
                        continue
                    if e_idx - s_idx + 1 > QA_MAX_ANSWER_TOKENS:
                        continue
                    s_char = offsets[b][s_idx][0].item()
                    e_char = offsets[b][e_idx][1].item()
                    if s_char >= e_char:
                        continue
                    answer = contexts[b][s_char:e_char].strip()
                    if not (MIN_QUOTE_LEN <= len(answer) <= MAX_QUOTE_LEN):
                        continue
                    score = (s_p[s_idx] * e_p[e_idx]).item()
                    if score > best_score:
                        best_score = score
                        best = {"answer": answer, "score": score, "span": (s_char, e_char)}

            # инверсия + расширение — та же логика, но на contexts[b]
            if best:
                s_char, e_char = best["span"]
                exp_s, exp_e = _expand_span_to_sentence(contexts[b], s_char, e_char)
                if _has_inversion_right_after(contexts[b], e_char):
                    expanded = contexts[b][exp_s:exp_e].strip()
                    if len(expanded) <= MAX_QUOTE_LEN:
                        best = {"answer": expanded, "score": best["score"] * 0.9,
                                "span": (exp_s, exp_e)}

            results.append(best if best and best["score"] >= QA_SCORE_THRESHOLD else None)

        return results
        

    def _validate_points(self, points, by_id, existing_features=None):
        valid = []
        seen = {f.strip().lower() for f in (existing_features or [])}

        for p in points:
            if not isinstance(p, dict):
                continue
            feature = (p.get("feature") or "").strip()
            if not feature or feature.lower() in seen:
                continue

            raw_ids = p.get("finding_ids") or []
            ids = [i for i in raw_ids if isinstance(i, str) and i in by_id]
            if len(ids) < 2:
                logger.warning("Feature %r: <2 валидных finding_id. Дроп.", feature)
                continue

            matched_ids = [i for i in ids if _matches_feature(by_id[i], feature)]
            weak_evidence = False
            if len(matched_ids) < 2:
                logger.warning(
                    "Feature %r: matched=%d/%d findings → weak_evidence=true (не дроп)",
                    feature, len(matched_ids), len(ids),
                )
                weak_evidence = True
            else:
                ids = matched_ids

            point_sent = _canonical_sentiment(p.get("sentiment"))
            logger.debug(
                "[0.3] Reduce point feature=%r sentiment_raw=%r → canonical=%r",
                feature, p.get("sentiment"), point_sent,
            )

            if point_sent in ("positive", "negative"):
                original_ids = ids
                ids = [
                    i for i in ids
                    if _canonical_sentiment(by_id[i].get("sentiment", "")) in (point_sent, "mixed")
                ]
                if not ids:
                    logger.warning(
                        "sentiment filter emptied point feature=%r (point=%r, had=%s)",
                        feature, point_sent, original_ids,
                    )
                    ids = original_ids

            if ids:
                aligned = sum(1 for i in ids if _matches_feature(by_id[i], feature))
                logger.debug(
                    "[0.2] Point feature=%r umbrella=%s aligned=%d/%d (%.0f%%) weak=%s",
                    feature, feature in UMBRELLA_FEATURES,
                    aligned, len(ids),
                    100 * aligned / max(len(ids), 1),
                    weak_evidence,
                )

            quote_id = self._pick_quote_id(feature, p.get("description", ""), ids, by_id)

            if not matched_ids and quote_id is None:
                logger.warning(
                    "Feature %r: empty point (0 matched, no quote) → drop", feature,
                )
                continue

            seen.add(feature.lower())
            valid.append({
                "feature": feature,
                "description": (p.get("description") or "").strip(),
                "sentiment": point_sent,
                "quote": by_id[quote_id]["quote"] if quote_id else None,
                "finding_ids": ids,
                "weak_evidence": weak_evidence,
            })
        return valid

    def _pick_quote_id(self, feature, description, ids, by_id):
        aligned = [i for i in ids if _matches_feature(by_id[i], feature)]
        if aligned:
            feature_low = feature.lower().strip()

            def _score(i: str) -> tuple:
                f = by_id[i]
                topic_ru = (f.get("topic_ru") or "").strip().lower()
                topic_orig = (f.get("topic") or "").strip().lower()
                exact_ru = 1 if topic_ru == feature_low else 0
                exact_orig = 1 if topic_orig == feature_low else 0
                if topic_orig and _detect_script(topic_orig) == _detect_script(feature):
                    ta = _topic_quote_score(feature, topic_orig)
                else:
                    ta = 0.0
                return (exact_ru, exact_orig, ta, len(f.get("quote", "") or ""))

            return max(aligned, key=_score)

        quote_id = self._embedding_best(feature, description, ids, by_id)
        if quote_id:
            return quote_id

        logger.info("Feature %r: цитата не подобрана (ids=%s)", feature, ids)
        return None

    # ---------- Reduce: саммари ----------
    def _reduce_summary(
        self, mapped_findings: List[Dict[str, Any]],
        game: str, user_query: str,
    ) -> str:
        by_id = {f["id"]: f for f in mapped_findings}

        points: List[Dict[str, Any]] = []
        cloud_used = False
        if self.cloud is not None and self.cloud.enabled:
            try:
                points = self._reduce_summary_cloud(mapped_findings, game, user_query)
                cloud_used = True
                logger.info("[Cloud] summary OK: %d points", len(points))
            except self._CloudLLMError as e:
                logger.warning("[Cloud] summary failed: %s → local fallback", e)

        if not cloud_used:
            points = self._reduce_summary_local(mapped_findings, game, user_query)

        for p in points:
            g = _compute_grounding(p, by_id)
            p["_grounding"] = g
            logger.debug(
                "[R2.5] feature=%r desc_words=%d covered=%d ratio=%.2f low=%s",
                p.get("feature"), g["desc_size"], g["covered"],
                g["ratio"], g["low_grounding"],
            )
        _grounding_low = [p for p in points if p.get("_grounding", {}).get("low_grounding")]
        if _grounding_low:
            logger.info(
                "[R2.5] low_grounding: %d/%d — %s",
                len(_grounding_low), len(points),
                ", ".join(f"{p['feature']}({p['_grounding']['ratio']:.0%})"
                          for p in _grounding_low),
            )
        else:
            logger.info("[R2.5] grounding: all %d points ≥60%%", len(points))

        self._last_valid_points = points
        _dump_calibration_pairs(points, by_id, source=("cloud" if cloud_used else "local"))

        weak_count = sum(1 for p in points if p.get("weak_evidence"))
        umbrella_count = sum(1 for p in points if p.get("feature") in UMBRELLA_FEATURES)
        logger.info(
            "[R2.5] Summary: %d points (%d umbrella), %d weak_evidence (%.0f%%) [%s]",
            len(points), umbrella_count, weak_count,
            100 * weak_count / max(len(points), 1),
            "cloud" if cloud_used else "local",
        )

        md = "### Ключевые особенности игры\n\n"
        if not points:
            return md + "*Не удалось выделить достоверные особенности на основе предоставленных отзывов.*\n"
        for i, p in enumerate(points, 1):
            sent_ru = _render_sentiment(p.get("sentiment", "mixed"))
            md += f"**{i}. {p['feature']}** — _{sent_ru}_\n{p['description']}\n"
            if p.get("quote"):
                md += f"> *{p['quote']}*\n"
            md += "\n"
        return md

    def _reduce_summary_cloud(
        self, mapped_findings: List[Dict[str, Any]],
        game: str, user_query: str,
    ) -> List[Dict[str, Any]]:
        _reduce_view = [
            {
                "id": f["id"],
                "topic": f.get("topic"),
                "topic_ru": f.get("topic_ru"),
                "sentiment": f.get("sentiment"),
                "quote": f.get("quote"),
            }
            for f in mapped_findings
        ]
        combined_data = json.dumps(_reduce_view, ensure_ascii=False)
        by_id = {f["id"]: f for f in mapped_findings}
        focus = (
            f"Пользователь особенно интересуется: '{user_query}'. Поставь в начало."
            if user_query else ""
        )

        prompt = f"""
Ты — профессиональный игровой журналист. По списку фактов из отзывов на "{game}" составь сводку из 5-7 ключевых особенностей.
{focus}

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Пункты не должны дублировать друг друга по теме.
2. description — живое, 1–2 предложения, естественный русский.
3. description = СУММА цитат выбранных finding_ids. Никаких знаний об игре вне списка.
4. SEMANTIC INVERSION: если цитата сравнительная ("easier", "проще", "легче") —
sentiment пункта должен соответствовать смыслу цитаты, а не слову.
5. СОСТАВНЫЕ ТЕМЫ ("Оптимизация и боссы") — раздели на два отдельных пункта.
6. finding_ids — только существующие id из списка, ≥2 на пункт.
7. Если тема не подтверждается минимум 2 findings — не включай.
8. Не включай цитаты или finding_ids непосредственно в description.

Верни строго JSON:
{{
"summary_points": [
{{"feature": "<название>", "description": "<1-2 предложения>",
    "sentiment": "positive|negative|mixed", "finding_ids": ["<id1>", "<id2>"]}}
]
}}

СПИСОК ФАКТОВ:
{combined_data}
"""
        result = self.cloud.chat_json(prompt, SUMMARY_SCHEMA, "reduce_summary")
        return self._validate_points(result.get("summary_points", []), by_id)

    def _reduce_summary_local(
        self, mapped_findings: List[Dict[str, Any]],
        game: str, user_query: str,
    ) -> List[Dict[str, Any]]:
        _reduce_view = [
            {k: f.get(k) for k in ("id", "topic", "sentiment", "quote")}
            for f in mapped_findings
        ]
        combined_data = json.dumps(_reduce_view, ensure_ascii=False)
        by_id = {f["id"]: f for f in mapped_findings}
        focus = (
            f"Пользователь особенно интересуется темой: '{user_query}'. "
            f"Поставь её на первое место."
            if user_query else ""
        )

        prompt_1 = f"""
Ты — профессиональный игровой журналист. Дан список фактов из отзывов на игру "{game}".
{focus}

Составь сводку из 5–7 ключевых особенностей игры.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. ПРИОРИТЕТ РАЗНООБРАЗИЯ: разные категории. Не дроби одну тему.
2. Описание — живое, 1–2 предложения. Следи за падежами.
3. Описание = СУММА выбранных фактов. Не добавляй знаний об игре вне цитат.
4. finding_ids — только существующие id. Мин. 2 на пункт.
5. Если достоверных меньше 5 — верни меньше. Пять честных лучше семи выдуманных.

Верни строго JSON:
{{
"summary_points": [
{{"feature": "...", "description": "...", "sentiment": "positive|negative|mixed", "finding_ids": ["f1", "f2"]}}
]
}}

СПИСОК ФАКТОВ:
{combined_data}
"""
        result = self._call_llm(prompt_1, "Reduce-саммари (local)", num_predict=REDUCE_NUM_PREDICT)
        return self._validate_points(result.get("summary_points", []), by_id)

    # ---------- Reduce: таблица ----------
    def _reduce_table(
        self, mapped_findings: List[Dict[str, Any]],
        game: str, user_query: str,
    ) -> str:
        if self._last_valid_points is not None:
            return self._render_table_from_points(self._last_valid_points)
        logger.warning("_reduce_table: valid_points отсутствуют → пустая таблица")
        return "### Анализ по категориям\n\n*Нет данных.*\n"

    def _render_table_from_points(self, points: List[Dict[str, Any]]) -> str:
        md = "### Анализ по категориям\n\n"
        if not points:
            return md + "*Недостаточно достоверных данных для построения таблицы.*\n"
        md += "| Категория | Тональность | Описание |\n| :--- | :--- | :--- |\n"
        for p in points:
            cat = str(p.get("feature", "")).replace("|", "\\|")
            sent = _render_sentiment(p.get("sentiment", "mixed")).replace("|", "\\|")
            desc = str(p.get("description", "")).replace("\n", " ").replace("|", "\\|")
            md += f"| **{cat}** | {sent} | {desc} |\n"
        return md

    # ---------- LLM-обёртка ----------
    def _call_llm(
        self,
        prompt: str,
        stage_name: str,
        num_predict: int,
        max_retries: int = 2,
        repeat_penalty: float = 1.3,
    ) -> Dict[str, Any]:
        for _ in range(max_retries):
            try:
                options = {
                    "num_ctx": NUM_CTX,
                    "temperature": 0.0,
                    "num_predict": num_predict,
                    "repeat_penalty": repeat_penalty,
                    "repeat_last_n": 256,
                }
                resp = self.ollama_client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Ты возвращаешь только валидный JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    format="json",
                    think=False,
                    keep_alive="30m",
                    options=options,
                )
                raw = resp["message"]["content"] or "{}"
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    logger.error(
                        "[%s] JSON decode error: %s. Хвост ответа: %r",
                        stage_name, e, raw[-200:],
                    )
                    raise LLMJSONError(str(e)) from e
            except LLMJSONError:
                raise
            except Exception as e:
                logger.error("[%s] Непредвиденная ошибка: %s", stage_name, e)
                raise
        raise LLMJSONError(f"{stage_name}: исчерпаны retries")


# ==========================================
# Пример использования (для отладки)
# ==========================================
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename="run.log",
        filemode="w",
        encoding="utf-8",
    )

    CACHE_FILE = "cache/0277ae7107a17da37b37d3c7da99a600.json"
    if len(sys.argv) > 1:
        CACHE_FILE = sys.argv[1]

    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        logger.error("Кэш-файл не найден: %s", cache_path.resolve())
        sys.exit(1)

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            dm_output = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Невалидный JSON в кэш-файле: %s", e)
        sys.exit(1)
    except OSError as e:
        logger.error("Ошибка чтения файла: %s", e)
        sys.exit(1)

    user_query = "«Составь сводку по Elden Ring, сделай упор на крафт"

    processor = ReviewProcessor()
    result = processor.process(dm_output, user_query=user_query)

    print("\n=== РЕЗУЛЬТАТ ===")
    print(f"Статус: {result['status']}")
    print(f"Предупреждения: {result.get('warnings', [])}")

    if result["status"] == "success":
        print(f"\nИгра: {result.get('game')}")
        print("\n--- САММАРИ ---")
        print(result["summary"])
        if result.get("table"):
            print("\n--- ТАБЛИЦА ---")
            print(result["table"])
    elif result["status"] == "error":
        print(f"\nОшибка: {result.get('message')}")
    else:
        print(f"\nНеожиданный статус: {result['status']}")
        print(f"Детали: {result}")