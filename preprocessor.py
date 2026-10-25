import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ==========================================
# Анонимизация (152-ФЗ)
# ==========================================
URL_REGEX = re.compile(
    r'(?:https?://[^\s<>"\'`]+)'
    r'|(?:www\.[a-zA-Z0-9][a-zA-Z0-9\-.]*\.[a-zA-Z]{2,}[^\s<>"\'`]*)',
    re.IGNORECASE,
)
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
STEAMID_REGEX = re.compile(r'\b7656\d{13}\b')

# ==========================================
# Эмодзи — только реальные эмодзи-диапазоны.
# ИСКЛЮЧЕНЫ: \U000024C2-\U0001F251 (там CJK, Hiragana, Hangul, Yi).
# ==========================================
EMOJI_REGEX = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F000-\U0001F0FF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

MEANINGFUL_CHAR_REGEX = re.compile(r'\w', re.UNICODE)

MIN_REVIEW_LENGTH = 40

# ==========================================
# Токенизация для матчинга по железу
# ==========================================
TOKEN_SPLIT_REGEX = re.compile(r'[^\w]+', re.UNICODE)
NON_ALNUM_REGEX = re.compile(r'[^\w]+', re.UNICODE)

HARDWARE_FILLER_TOKENS = frozenset({
    "nvidia", "amd", "intel", "arc",
    "geforce", "radeon", "core", "ryzen", "athlon", "pentium", "celeron",
    "threadripper", "rtx", "gtx", "rx", "series",
    "gpu", "cpu", "processor", "card", "graphics", "video", "videocard",
    "gaming",
    "for", "with", "on", "and", "the",
})


# ==========================================
# 1. Анонимизация
# ==========================================
def anonymize_text(text: str, stats: Optional[Dict[str, int]] = None) -> str:
    """Удаляет персональные данные. Опционально инкрементит счётчики в stats."""
    if not text:
        return ""
    if stats is not None:
        stats["urls"] = stats.get("urls", 0) + len(URL_REGEX.findall(text))
        stats["emails"] = stats.get("emails", 0) + len(EMAIL_REGEX.findall(text))
        stats["steamids"] = stats.get("steamids", 0) + len(STEAMID_REGEX.findall(text))
    text = URL_REGEX.sub("[URL]", text)
    text = EMAIL_REGEX.sub("[EMAIL]", text)
    text = STEAMID_REGEX.sub("[STEAMID]", text)
    return text


# ==========================================
# 2. Фильтрация мусора
# ==========================================
def _short_review_is_hardware_relevant(
    text: str, hw_cpu: Optional[str], hw_gpu: Optional[str]
) -> bool:
    """Короткий отзыв, но совпадает с железом — оставляем (см. ниже)."""
    if not (hw_cpu or hw_gpu):
        return False
    alnum = NON_ALNUM_REGEX.sub("", text.lower())
    if not alnum:
        return False
    return _review_matches_hardware(alnum, hw_cpu) or \
           _review_matches_hardware(alnum, hw_gpu)


def filter_garbage(
    reviews: List[Dict[str, Any]],
    hw_cpu: Optional[str] = None,
    hw_gpu: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Отсеивает неинформативные отзывы. Возвращает:
        {
            "reviews": List[Dict],
            "stats": {...},
            "dropped_examples": List[Tuple[str, str]],
        }
    """
    stats: Dict[str, Any] = {
        "input": len(reviews),
        "output": 0,
        "dropped_short": 0,
        "dropped_emoji": 0,
        "dropped_no_alnum": 0,
        "short_but_hardware_kept": 0,
        "anon_stripped": {"urls": 0, "emails": 0, "steamids": 0},
    }
    cleaned: List[Dict[str, Any]] = []
    dropped_examples: List[tuple] = []

    def _drop(reason: str, text: str) -> None:
        stats[reason] += 1
        if len(dropped_examples) < 10:
            dropped_examples.append((reason, text[:80].replace("\n", " ")))

    for review in reviews:
        raw_text = review.get("review", "") or ""
        anon_text = anonymize_text(raw_text, stats["anon_stripped"])
        stripped = anon_text.strip()

        if len(stripped) < MIN_REVIEW_LENGTH:
            if _short_review_is_hardware_relevant(stripped, hw_cpu, hw_gpu):
                stats["short_but_hardware_kept"] += 1
            else:
                _drop("dropped_short", stripped)
                continue

        emoji_length = sum(len(m) for m in EMOJI_REGEX.findall(stripped))
        if emoji_length > len(stripped) * 0.5:
            _drop("dropped_emoji", stripped)
            continue

        if not MEANINGFUL_CHAR_REGEX.search(stripped):
            _drop("dropped_no_alnum", stripped)
            continue

        cleaned.append({**review, "review": anon_text})

    stats["output"] = len(cleaned)
    logger.info(
        f"filter_garbage: {stats['input']} -> {stats['output']} "
        f"(short={stats['dropped_short']}, emoji={stats['dropped_emoji']}, "
        f"no_alnum={stats['dropped_no_alnum']}, "
        f"hw_kept={stats['short_but_hardware_kept']})"
    )
    return {"reviews": cleaned, "stats": stats, "dropped_examples": dropped_examples}


# ==========================================
# 2.5. Детекция и балансировка мемных отзывов
# ==========================================
# Идея: реальный отзыв упоминает грань игры И оценивает её.
# Мем шутит про игру, но не сообщает «что хорошо, что плохо».
#
# Мемы полностью не выкидываются: обилие шуток — сигнал «живое
# комьюнити», но >20% шума ломает аналитическое summary.
# Балансировка сохраняет самые «информативные» мемы (с наибольшим
# числом аспектов / длиной).

# Грань игры — регулярки по ключевым стемам на разных языках.
ASPECT_KEYWORDS: Dict[str, frozenset] = {
    "performance": frozenset({
        "производительност", "оптимизац", "лаг", "тормоз", "фпс", "fps",
        "просадк", "фриз", "статтер", "stutter", "optimization",
        "performance", "frame drop",
        "パフォーマンス", "最適化", "カクつき", "성능", "최적화", "렉",
    }),
    "graphics": frozenset({
        "график", "визуал", "модел", "текстур", "арт",
        "graphics", "visual", "texture", "art style",
        "グラフィック", "ビジュアル", "그래픽",
    }),
    "story": frozenset({
        "сюжет", "истори", "нарратив", "повествован", "персонаж", "диалог",
        "story", "narrative", "plot", "character", "dialogue", "lore",
        "物語", "ストーリー", "キャラクター", "스토리", "이야기", "캐릭터",
    }),
    "gameplay": frozenset({
        "геймплей", "боёвк", "боевк", "механик", "управлен", "баланс",
        "gameplay", "combat", "mechanic", "controls", "balance", "boss",
        "ゲームプレイ", "戦闘", "ゲーム性", "게임플레이", "전투",
    }),
    "sound": frozenset({
        "музык", "саундтрек", "звук", "озвучк", "аудио",
        "music", "soundtrack", "voice acting", "audio", "ost",
        "音楽", "サウンド", "음악", "사운드",
    }),
    "bugs": frozenset({
        "баг", "глюк", "вылет", "краш", "сломан", "не работает",
        "bug", "crash", "broken", "glitch", "freeze",
        "バグ", "버그", "오류",
    }),
    "price": frozenset({
        "цен", "стоимость", "переплат", "скидк", "деньг",
        "price", "worth", "cost", "sale", "value",
        "価格", "値段", "가격",
    }),
    "content": frozenset({
        "контент", "часов", "прохожден", "длс", "dlc", "реиграбельн",
        "content", "hours", "playtime", "replayability",
        "コンテンツ", "時間", "콘텐츠",
    }),
    "difficulty": frozenset({
        "сложност", "хардкор", "челлендж",
        "difficulty", "challenging", "hardcore",
        "難易度", "難しい", "난이도",
    }),
    "multiplayer": frozenset({
        "мультиплеер", "кооп", "онлайн", "пвп", "pvp",
        "multiplayer", "coop", "online", "network", "netcode",
        "マルチプレイ", "オンライン", "멀티플레이",
    }),
    "recommend": frozenset({
        "рекоменд", "не стоит брать",
        "recommend", "must play", "worth buying",
        "おすすめ", "추천",
    }),
}

# Оценочные маркеры (стемы) — «хорошо/плохо», а не «про что».
EVALUATIVE_STEMS_RU = frozenset({
    "хорош", "плох", "интересн", "скучн", "качествен", "ужасн",
    "отличн", "лучш", "худш", "нравит", "не нрав", "стоит",
    "люблю", "ненавиж", "прекрасн", "классн", "супер", "крут",
    "великолепн", "потрясающ", "неплох", "провальн", "слаб",
    "норм", "дерьм", "уныл", "весел", "забавн", "рекоменд",
    "разочаров", "восхит", "восторг", "отврат",
})

EVALUATIVE_WORDS_EN = frozenset({
    "good", "bad", "great", "terrible", "boring", "fun", "amazing", "awful",
    "best", "worst", "like", "love", "hate", "worth", "recommend",
    "quality", "excellent", "poor", "masterpiece", "disappointing",
    "awesome", "fantastic", "solid", "decent", "mediocre", "trash",
})

EVALUATIVE_CJK = frozenset({
    "良い", "悪い", "面白い", "つまらない", "素晴らしい", "最高", "最低",
    "好き", "嫌い", "おすすめ",
    "좋은", "나쁜", "재미", "재밌", "최고", "최악", "추천",
})

# Числовая оценка: 10/10, 8/10, 8,5/10, 10\10, 8 из 10.
# Покрывает кириллическое «из», обратный слэш, запятую как разделитель.
RATING_REGEX = re.compile(
    r'\b(?:10|[0-9](?:[.,]\d)?)\s*(?:[/\\]|из)\s*10\b',
    re.IGNORECASE,
)

MAX_MEME_RATIO = 0.2

_ASPECT_REGEXES: Dict[str, re.Pattern] = {
    aspect: re.compile("|".join(re.escape(kw) for kw in kws), re.IGNORECASE)
    for aspect, kws in ASPECT_KEYWORDS.items()
}
_EN_EVAL_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in EVALUATIVE_WORDS_EN) + r')\b',
    re.IGNORECASE,
)


def _detect_aspects(text: str) -> List[str]:
    return [a for a, rx in _ASPECT_REGEXES.items() if rx.search(text)]


def _has_evaluation(text: str) -> bool:
    lower = text.lower()
    if any(stem in lower for stem in EVALUATIVE_STEMS_RU):
        return True
    if _EN_EVAL_REGEX.search(text):
        return True
    if any(w in text for w in EVALUATIVE_CJK):
        return True
    return False


def classify_review(text: str) -> Dict[str, Any]:
    """
    Классифицирует отзыв: мем vs реальный.

    Два скора:
    - base_score — только по сигналу (аспекты + оценка/рейтинг).
    - score — base_score + бонус за длину; используется только для
      приоритезации при отборе (длиннее → полезнее).

    Классификация ИДЁТ по base_score, а не по score: длинный мем
    всё равно мем, даже если он 800+ символов.

    Пороги:
    - base_score <= 1 → definite meme (мем без сигнала)
    - base_score 2-3 + длина < 200 → probable meme (слабый сигнал)
    - иначе → real (либо ≥ 4 base_score, либо 2-3 сигнала + длина ≥ 200)

    Примеры (base_score):
      «Желаю Миядзаке…» (0 аспектов, 0 оценок)                → 0 → definite
      «Когда Миядзаки били в детстве…» (0 аспектов, 0 оценок) → 0 → definite
      «Плюсы: ушки Мияби / 10 непонятных сюжетов из 10»       → 4 → real
      «Белый экран и вылет… игра не работоспособна»           → 2, len 220 → real
      «Прерванный стрик» (много аспектов + оценка)            → 8 → real
    """
    if not text:
        return {
            "aspects": [], "has_evaluation": False, "has_rating": False,
            "score": 0, "base_score": 0, "is_probable_meme": True,
        }

    aspects = _detect_aspects(text)
    has_eval = _has_evaluation(text)
    has_rating = bool(RATING_REGEX.search(text))
    length = len(text)

    # Сигнальный скор (без бонуса за длину)
    base_score = min(len(aspects), 3) * 2
    if has_eval or has_rating:
        base_score += 2
    if has_rating and length >= 150:
        base_score += 1

    # Отчётный скор — добавляем длину для приоритезации
    score = base_score
    if length >= 300:
        score += 1
    if length >= 800:
        score += 1

    # Классификация — по base_score
    is_meme = False
    if base_score <= 1:
        is_meme = True
    elif base_score <= 3 and length < 200:
        is_meme = True

    return {
        "aspects": aspects,
        "has_evaluation": has_eval,
        "has_rating": has_rating,
        "score": score,
        "base_score": base_score,
        "is_probable_meme": is_meme,
    }



def annotate_memes(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Прикрепляет _meme_info к каждому отзыву. Не мутирует входные dicts."""
    return [
        {**r, "_meme_info": classify_review(r.get("review", "") or "")}
        for r in reviews
    ]


def balance_meme_reviews(
    reviews: List[Dict[str, Any]],
    max_meme_ratio: float = MAX_MEME_RATIO,
) -> Dict[str, Any]:
    """
    Двухуровневая фильтрация мемов:

    Tier 1 — definite memes (base_score <= 1): всегда дропаются,
             даже если общая доля мемов ниже порога. Такие отзывы
             не содержат никакой информации о игре.
             Предохранитель: если после дропа останется < max(10, total//3)
             отзывов — дропаем не всех, а только до этого минимума.

    Tier 2 — probable memes (base_score 2-3, len < 200): капятся так,
             чтобы их доля в итоговой выборке не превышала max_meme_ratio.
             Сортируются по (аспекты, длина) — более «сигнальные» идут первыми.

    Safety: если >= 90% отзывов — мемы (вероятно, ложная классификация
            или игра-мем), фильтрация пропускается.
    """
    total = len(reviews)
    if total == 0:
        return {
            "reviews": [],
            "stats": {
                "total_before": 0, "real": 0,
                "definite_memes_total": 0, "definite_memes_kept": 0,
                "definite_memes_dropped": 0,
                "probable_memes_total": 0, "probable_memes_kept": 0,
                "probable_memes_dropped": 0,
                "memes_total": 0, "memes_kept": 0, "memes_dropped": 0,
                "meme_ratio_before": 0.0, "meme_ratio_after": 0.0,
            },
            "dropped_memes": [],
        }

    # Partition
    real: List[Dict[str, Any]] = []
    definite: List[Dict[str, Any]] = []
    probable: List[Dict[str, Any]] = []

    for r in reviews:
        info = r.get("_meme_info") or {}
        base = info.get("base_score", info.get("score", 0))
        is_meme = info.get("is_probable_meme", False)
        if not is_meme:
            real.append(r)
        elif base <= 1:
            definite.append(r)
        else:
            probable.append(r)

    memes_total = len(definite) + len(probable)
    before_ratio = memes_total / total

    # Safety: 90%+ мемов
    if before_ratio >= 0.9:
        logger.warning(
            f"90%+ отзывов классифицированы как мемы — вероятно, ложная "
            f"классификация или игра-мем. Балансировка пропущена "
            f"({total} отзывов, {memes_total} мемов)."
        )
        return {
            "reviews": reviews,
            "stats": {
                "total_before": total, "real": len(real),
                "definite_memes_total": len(definite),
                "definite_memes_kept": len(definite),
                "definite_memes_dropped": 0,
                "probable_memes_total": len(probable),
                "probable_memes_kept": len(probable),
                "probable_memes_dropped": 0,
                "memes_total": memes_total, "memes_kept": memes_total,
                "memes_dropped": 0,
                "meme_ratio_before": round(before_ratio, 3),
                "meme_ratio_after": round(before_ratio, 3),
                "filtering_skipped": True,
            },
            "dropped_memes": [],
        }

    # Tier 1: дроп definite-мемов
    min_keep = max(10, total // 3)
    remaining_after_definite = len(real) + len(probable)

    if remaining_after_definite >= min_keep:
        kept_definite: List[Dict[str, Any]] = []
        dropped_definite = definite
    else:
        # Слишком мало останется — оставляем самые «сигнальные» definite
        needed = min_keep - remaining_after_definite
        definite_sorted = sorted(
            definite,
            key=lambda r: (
                -len((r.get("_meme_info") or {}).get("aspects", [])),
                -len(r.get("review", "")),
            ),
        )
        kept_definite = definite_sorted[:needed]
        dropped_definite = definite_sorted[needed:]

    # Tier 2: кап probable-мемов
    base_kept = len(real) + len(kept_definite)
    if probable:
        # prob_kept / (base_kept + prob_kept) <= max_meme_ratio
        if base_kept > 0:
            max_probable = int(max_meme_ratio * base_kept / (1 - max_meme_ratio))
        else:
            max_probable = 0
        max_probable = max(1, max_probable)
        max_probable = min(max_probable, len(probable))

        probable_sorted = sorted(
            probable,
            key=lambda r: (
                -len((r.get("_meme_info") or {}).get("aspects", [])),
                -len(r.get("review", "")),
            ),
        )
        kept_probable = probable_sorted[:max_probable]
        dropped_probable = probable_sorted[max_probable:]
    else:
        kept_probable = []
        dropped_probable = []

    final = real + kept_definite + kept_probable
    dropped = dropped_definite + dropped_probable
    after_ratio = (
        (len(kept_definite) + len(kept_probable)) / len(final) if final else 0.0
    )

    logger.info(
        f"Meme balance: {total} -> {len(final)} | "
        f"real={len(real)}, "
        f"definite {len(definite)} -> {len(kept_definite)}, "
        f"probable {len(probable)} -> {len(kept_probable)} | "
        f"ratio {before_ratio:.1%} -> {after_ratio:.1%}"
    )

    return {
        "reviews": final,
        "stats": {
            "total_before": total,
            "real": len(real),
            "definite_memes_total": len(definite),
            "definite_memes_kept": len(kept_definite),
            "definite_memes_dropped": len(dropped_definite),
            "probable_memes_total": len(probable),
            "probable_memes_kept": len(kept_probable),
            "probable_memes_dropped": len(dropped_probable),
            "memes_total": memes_total,
            "memes_kept": len(kept_definite) + len(kept_probable),
            "memes_dropped": len(dropped),
            "meme_ratio_before": round(before_ratio, 3),
            "meme_ratio_after": round(after_ratio, 3),
        },
        "dropped_memes": dropped,
    }


# ==========================================
# 3. Текстовая фильтрация по железу
# ==========================================
def _normalize_for_match(text: str) -> str:
    return NON_ALNUM_REGEX.sub("", text.lower())


def _extract_hardware_tokens(spec: Optional[str]) -> List[str]:
    if not spec:
        return []
    tokens = TOKEN_SPLIT_REGEX.split(spec.lower().strip())
    tokens = [t for t in tokens if t and t not in HARDWARE_FILLER_TOKENS]
    return [t for t in tokens if len(t) >= 2]


def _review_matches_hardware(review_alnum: str, spec: Optional[str]) -> bool:
    tokens = _extract_hardware_tokens(spec)
    if not tokens:
        return False
    strong = [t for t in tokens if len(t) >= 4]
    if strong:
        return any(t in review_alnum for t in strong)
    return all(t in review_alnum for t in tokens)


def filter_by_hardware_text(
    reviews: List[Dict[str, Any]],
    hardware_fallback: bool,
    hw_cpu: Optional[str] = None,
    hw_gpu: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "reviews": reviews,
        "applied": False,
        "matched": False,
        "fallback_returned_full": False,
    }
    if not hardware_fallback or not (hw_cpu or hw_gpu):
        return result

    result["applied"] = True
    if not (_extract_hardware_tokens(hw_cpu) or _extract_hardware_tokens(hw_gpu)):
        logger.warning(
            f"Нет значимых токенов в hardware_cpu={hw_cpu!r}, "
            f"hardware_gpu={hw_gpu!r}. Пропуск."
        )
        result["fallback_returned_full"] = True
        return result

    matched = [
        r for r in reviews
        if _review_matches_hardware(_normalize_for_match(r.get("review", "")), hw_cpu)
        or _review_matches_hardware(_normalize_for_match(r.get("review", "")), hw_gpu)
    ]

    if matched:
        result["reviews"] = matched
        result["matched"] = True
        logger.info(f"Hardware-text: {len(matched)}/{len(reviews)} совпадений")
        return result

    logger.warning(f"Hardware-text: 0 совпадений для {hw_cpu!r}/{hw_gpu!r}. Полный список.")
    result["fallback_returned_full"] = True
    return result


# ==========================================
# 4. Главный оркестратор
# ==========================================
def clean_reviews(
    raw_reviews: List[Dict[str, Any]],
    hardware_fallback: bool,
    hw_gpu: Optional[str] = None,
    hw_cpu: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Порядок:
      1. hardware-filter (только при fallback) — до length-фильтра,
         чтобы сохранить короткие hardware-релевантные отзывы;
      2. filter_garbage — короткие / эмодзи / no-alnum;
      3. annotate_memes — классификация каждого отзыва;
      4. balance_meme_reviews — кап доли мемов (по умолчанию 20%).

    Возвращает:
        {
            "reviews": List[Dict],           # с _meme_info у каждого
            "stats": {
                "input_count": int,
                "after_hardware_filter": int,
                "after_garbage_filter": int,
                "after_meme_balance": int,
                "hardware_text_filter_applied": bool,
                "hardware_text_filter_matched": bool,
                "hardware_text_fallback_full": bool,
                "garbage_breakdown": {...},
                "meme_analysis": {...},
            },
            "dropped_examples": List[Tuple[str, str]],
            "dropped_memes": List[Dict],     # отсечённые мемы (для диагностики)
        }
    """
    empty = {
        "reviews": [], "stats": {}, "dropped_examples": [], "dropped_memes": [],
    }
    if not raw_reviews:
        return empty

    # Шаг 1: hardware-filter на «сырых» отзывах (только при fallback)
    hw_result = filter_by_hardware_text(
        reviews=raw_reviews,
        hardware_fallback=hardware_fallback,
        hw_cpu=hw_cpu,
        hw_gpu=hw_gpu,
    )
    candidates = hw_result["reviews"]

    # Шаг 2: garbage-filter с сохранением hardware-релевантных коротких
    garbage_result = filter_garbage(candidates, hw_cpu=hw_cpu, hw_gpu=hw_gpu)

    # Шаг 3: аннотировать мемы
    annotated = annotate_memes(garbage_result["reviews"])

    # Шаг 4: балансировка доли мемов
    balance_result = balance_meme_reviews(annotated)

    stats = {
        "input_count": len(raw_reviews),
        "after_hardware_filter": len(candidates),
        "after_garbage_filter": garbage_result["stats"]["output"],
        "after_meme_balance": len(balance_result["reviews"]),
        "hardware_text_filter_applied": hw_result["applied"],
        "hardware_text_filter_matched": hw_result["matched"],
        "hardware_text_fallback_full": hw_result["fallback_returned_full"],
        "garbage_breakdown": garbage_result["stats"],
        "meme_analysis": balance_result["stats"],
    }
    return {
        "reviews": balance_result["reviews"],
        "stats": stats,
        "dropped_examples": garbage_result["dropped_examples"],
        "dropped_memes": balance_result["dropped_memes"],
    }