import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from rapidfuzz import fuzz
import ollama
from intent import OllamaManager, OLLAMA_MODEL_NAME

logger = logging.getLogger(__name__)

# ==========================================
# Конфигурация (шаг 1 для RTX 3060 + qwen3)
# ==========================================
MAP_BATCH_SIZE = 16          # 120 / 16 = 8 батчей
MAP_WORKERS = 1              # без OLLAMA_NUM_PARALLEL>=2 смысла нет
MAP_MAX_FINDINGS = 6         # кап output на батч — ключ к скорости
MAP_NUM_PREDICT = 1024       # 6 findings × ~150 токенов + каркас
REDUCE_NUM_PREDICT = 1536    # 7 пунктов + finding_ids (см. комментарий про qwen3)
NUM_CTX = 6144               # 8192 не нужно, экономит KV-cache
REVIEW_TRUNCATE = 600        # было 900 — режем контекст
MAX_QUOTE_LEN = 180
MIN_QUOTE_LEN = 15
QUOTE_FUZZY_THRESHOLD = 90
TOPIC_FUZZY_THRESHOLD = 82
QUOTE_SIM_THRESHOLD = 0.80


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMJSONError(Exception):
    """LLM вернула обрезанный/невалидный JSON. Ловится в split-retry."""


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


def _format_review_for_llm(idx: int, r: Union[str, Dict[str, Any]]) -> str:
    """id=N в заголовке — на него ссылается Map через source_id."""
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
    """Жёсткая проверка одного finding из Map."""
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

    topic = (finding.get("topic") or "").strip()
    if not topic:
        return None

    return {
        "source_id": sid,
        "topic": topic,
        "sentiment": finding.get("sentiment", "mixed"),
        "quote": quote,
    }

def _topic_aligned(finding_topic: str, feature: str) -> bool:
    """Лексическое совпадение topic ↔ feature. Быстро, без ML."""
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
        self.ollama_client = ollama
        self.batch_size = batch_size

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

    # ---------- Оркестрация ----------
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

    # ---------- Служебное ----------
    def _extract_warnings(self, dm_output: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        meta = dm_output.get("meta", {})
        if dm_output.get("status") == "insufficient_data" or meta.get("insufficient_data"):
            warnings.append(
                "Внимание: Найдено менее 30 отзывов. Сводка может быть нерепрезентативной."
            )
        if meta.get("days_since_last_review", 0) > 60:
            warnings.append(
                "Внимание: Последний отзыв был написан более 2 месяцев назад. "
                "Информация может быть устаревшей."
            )
        return warnings

    # ---------- Map ----------
    def _map_reviews(
        self, reviews: List, game: str, user_query: str
    ) -> List[Dict[str, Any]]:
        batches = [
            reviews[i : i + self.batch_size]
            for i in range(0, len(reviews), self.batch_size)
        ]
        all_findings: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=MAP_WORKERS) as executor:
            futures = {
                executor.submit(self._process_batch_safe, b, game, user_query): b
                for b in batches
            }
            for future in as_completed(futures):
                try:
                    batch_findings = future.result()
                    if isinstance(batch_findings, list):
                        all_findings.extend(batch_findings)
                except Exception as e:
                    logger.error("Батч упал целиком: %s", e)

        # Глобальные id для Reduce
        for i, f in enumerate(all_findings):
            f["id"] = f"f{i}"

        logger.info("Map-этап собрал %d фактов.", len(all_findings))
        return all_findings

    def _process_batch_safe(
        self, batch: List, game: str, user_query: str
    ) -> List[Dict[str, Any]]:
        """При LLMJSONError — делим пополам. Никаких потерь батча."""
        try:
            return self._process_batch(batch, game, user_query)
        except LLMJSONError as e:
            if len(batch) <= 1:
                logger.warning("Split-retry: единственный отзыв не дал JSON (%s). Дроп.", e)
                return []
            mid = len(batch) // 2
            logger.info("Split-retry: %d -> %d + %d", len(batch), mid, len(batch) - mid)
            return (
                self._process_batch_safe(batch[:mid], game, user_query)
                + self._process_batch_safe(batch[mid:], game, user_query)
            )

    def _process_batch(
        self, batch: List, game: str, user_query: str
    ) -> List[Dict[str, Any]]:
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
4. "topic" — короткое название темы (1–3 слова): Оптимизация, Сюжет, Боссы, Баги и т.п.
5. "sentiment" — positive / negative / mixed.

ПРАВИЛА РАБОТЫ С ИРОНИЕЙ:
1. ЮМОРИСТИЧЕСКИЙ/ИРОНИЧНЫЙ отзыв — только как дополнительное подтверждение,
   не единственный источник.
2. Ищи ИНВЕРСИЮ СМЫСЛА: если оценка противоречит описанию — верь описанию.
3. Гиперболы ("пол миллиарда попыток", "10/10 шедевр") — не факты.
4. Отзыв без описания игрового опыта — пропускай. Пустой результат корректен.
5. Только текст отзывов, никаких знаний об игре "из головы".

ПОРЯДОК РАБОТЫ:
1. Сначала найди в отзыве с source_id конкретное предложение — цитату.
2. Потом сформулируй topic ИСХОДЯ ИЗ ЦИТАТЫ, а не из общих знаний об игре.
   Если цитата про баги — topic должен быть про баги, даже если ты знаешь,
   что у игры ещё есть проблемы с оптимизацией.
3. Topic — короткая фраза (1–3 слова), описывающая именно то,
   что утверждает эта цитата.

ПРИМЕРЫ ХОРОШИХ findings (цитата явно про заявленный topic):
✅ {{"source_id": 3, "quote": "Игра летает даже на старом ПК, стабильные 60 фпс", "topic": "Оптимизация", "sentiment": "positive"}}
✅ {{"source_id": 7, "quote": "Музыка в бою с боссом пробирает до мурашек", "topic": "Саундтрек", "sentiment": "positive"}}
✅ {{"source_id": 12, "quote": "После патча 1.04 игра стала заметно стабильнее", "topic": "Оптимизация", "sentiment": "mixed"}}
✅ {{"source_id": 18, "quote": "Боссы требуют заучивания паттернов, но каждый бой честный", "topic": "Боссы", "sentiment": "positive"}}

ПРИМЕРЫ ПЛОХИХ findings (цитата НЕ про этот topic — так НЕЛЬЗЯ):
❌ {{"source_id": 5, "quote": "Elden Ring é um jogo fantástico", "topic": "Оптимизация"}}
   ← цитата про общее впечатление, а не про производительность
❌ {{"source_id": 9, "quote": "I rate this game 10/10", "topic": "Звук"}}
   ← цитата про общую оценку, а не про музыку/звук
❌ {{"source_id": 14, "quote": "Это лучшая игра в моей жизни", "topic": "Сюжет"}}
   ← цитата про общее впечатление, а не про историю

Верни строго JSON:
{{
"findings": [
  {{"source_id": 3, "quote": "Игра летает даже на старом ПК", "topic": "Оптимизация", "sentiment": "positive"}}
]
}}

Отзывы:
{reviews_text}
"""
        result = self._call_llm(
            prompt, "Map-этап",
            num_predict=MAP_NUM_PREDICT,
            enable_thinking=False,  # структурный extraction: thinking только жжёт токены
        )
        findings = result.get("findings", [])

        valid: List[Dict[str, Any]] = []
        for f in findings:
            v = _validate_finding(f, batch)
            if v:
                valid.append(v)

        rejected = len(findings) - len(valid)
        if rejected:
            logger.debug("Map: %d/%d findings отброшено валидацией", rejected, len(findings))

        return valid[:MAP_MAX_FINDINGS]
    
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
                # Единственный случай, когда дропаем: Reduce выдумал id
                logger.warning("Feature %r: <2 валидных finding_id. Дроп.", feature)
                continue

            # Фильтр по sentiment пункта (см. п.2 ниже)
            point_sent = (p.get("sentiment") or "mixed").lower()
            if point_sent in ("positive", "negative"):
                ids = [
                    i for i in ids
                    if by_id[i].get("sentiment", "mixed") in (point_sent, "mixed")
                ] or ids  # не режем до нуля — лучше оставить все

            # Best-effort выбор цитаты; пункт сохраняется всегда
            quote_id = self._pick_quote_id(feature, p.get("description", ""), ids, by_id)

            seen.add(feature.lower())
            valid.append({
                "feature": feature,
                "description": (p.get("description") or "").strip(),
                "quote": by_id[quote_id]["quote"] if quote_id else None,
                "finding_ids": ids,
            })
        return valid

    def _pick_quote_id(self, feature, description, ids, by_id):
    # Ступень 1: topic-aligned
        aligned = [i for i in ids if _topic_aligned(by_id[i]["topic"], feature)]
        if aligned:
            return max(aligned, key=lambda i: len(by_id[i]["quote"]))

        # Ступень 2: embedding rerank внутри ids
        quote_id = self._embedding_best(feature, description, ids, by_id)
        if quote_id:
            return quote_id

        # Ступень 3: ничего не нашли — цитаты не будет, пункт останется
        logger.info("Feature %r: цитата не подобрана (ids=%s)", feature, ids)
        return None

    # ---------- Reduce: саммари ----------
    def _reduce_summary(
        self,
        mapped_findings: List[Dict[str, Any]],
        game: str,
        user_query: str,
    ) -> str:
        combined_data = json.dumps(mapped_findings, ensure_ascii=False)
        by_id = {f["id"]: f for f in mapped_findings}

        focus = (
            f"Пользователь особенно интересуется темой: '{user_query}'. "
            f"Поставь её на первое место."
            if user_query else ""
        )
    

        prompt_1 = f"""
Ты — профессиональный игровой журналист. Дан список фактов из отзывов на игру "{game}".
{focus}

Составь сводку РОВНО ИЗ 7 ключевых особенностей игры.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. ПРИОРИТЕТ РАЗНООБРАЗИЯ: 7 РАЗНЫХ категорий. Не дроби одну тему.
2. Если пользователь спросил про конкретную тему — поставь её первой, но дополни до 7.
3а. Описание — живое, 1–2 предложения. Следи за падежами.
3б. Описание должно быть СУММОЙ того, что говорят выбранные finding'и.
    Не добавляй общие знания об игре, которых нет в цитатах.
    Если в finding'ах нет прямого упоминания багов — не пиши "есть мелкие баги",
    даже если ты знаешь, что они есть.
4. Каждый пункт подкрепляй ТОЛЬКО существующими id из списка фактов.
   Возвращай массив "finding_ids" (например, ["f3", "f17"]), а не текст цитаты.
   Цитата подставится автоматически.
5. Каждый пункт — минимум 2 независимых факта в "finding_ids".
   Если достоверных меньше 7 — верни меньше. Пять честных лучше семи выдуманных.

Для каждого пункта верни:
- finding_ids: все факты, на которых основан пункт (мин. 2)

ПРАВИЛА PLAYTIME:
1. playtime_forever > playtime_at_review НЕ равно "реиграбельность".
2. Сильный сигнал: тег "ОТЗЫВ БЫЛ ДОПОЛНЕН ПОЗЖЕ" + playtime_forever > 100 ч.

ПРАВИЛА ДОСТОВЕРНОСТИ:
1. "ВЫСОКИЙ АВТОРИТЕТ В СООБЩЕСТВЕ" — весомее.
2. "НИШЕВЫЙ ОТЗЫВ" — специфично для конфигурации, отметь в описании.
3. "ВЫЗВАЛ ДИСКУССИЮ" — приоритетно, но проверяй тональность.

ПРИМЕРЫ ПРАВИЛЬНОЙ ПРИВЯЗКИ finding_ids:

✅ ПРАВИЛЬНО:
{{
  "feature": "Оптимизация",
  "description": "Игра стабильно работает на средних системах, но в открытом мире есть просадки FPS.",
  "sentiment": "mixed",
  "finding_ids": ["f2", "f14", "f27"]
}}
← Все три finding'а имеют topic про производительность/оптимизацию.
  Их цитаты: "Игра летает даже на старом ПК", "После патча 1.04 стало стабильнее", "В открытом мире просадки до 45 fps".

✅ ПРАВИЛЬНО:
{{
  "feature": "Сюжет",
  "description": "Лор спрятан в описаниях предметов и диалогах NPC, требует самостоятельной сборки.",
  "sentiment": "mixed",
  "finding_ids": ["f8", "f19"]
}}
← Оба finding'а имеют topic про сюжет/лор.

❌ НЕПРАВИЛЬНО (так делать нельзя):
{{
  "feature": "Оптимизация",
  "description": "Игра стабильно работает на высоких настройках...",
  "finding_ids": ["f5", "f11"]
}}
← f5 = "Elden Ring é um jogo fantástico" (topic: Общее впечатление)
  f11 = "10/10 if you are a Souls player" (topic: Рекомендация)
  Оба finding'а НЕ про оптимизацию.


Верни строго JSON:
{{
"summary_points": [
  {{"feature": "Оптимизация", "description": "Игра стабильно работает...", "finding_ids": ["f3", "f17"]"}}
]
}}

Список фактов:
{combined_data}
"""
        # qwen3: если REDUCE стабильно упирается в num_predict — смените enable_thinking на False
        result = self._call_llm(
            prompt_1, "Reduce-саммари (Pass 1)",
            num_predict=REDUCE_NUM_PREDICT,
            enable_thinking=True,
        )
        valid_points = self._validate_points(result.get("summary_points", []), by_id)

        # Pass 2 — только после валидации
        if len(valid_points) < 7:
            missing = 7 - len(valid_points)
            logger.info("Pass 1 дал %d валидных. Запуск Pass 2 (нужно %d)",
                        len(valid_points), missing)
            existing_features = [p["feature"] for p in valid_points]

            prompt_2 = f"""
Ты — профессиональный игровой журналист. Дан список фактов из отзывов на игру "{game}".
Уже выделены темы: {json.dumps(existing_features, ensure_ascii=False)}.

Найди ЕЩЁ {missing} ДРУГИХ важных аспектов, которых НЕТ в списке выше.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Только новые темы (визуал, звук, сюжет, баги, цена, реиграбельность и т.п.).
2. Живое описание, грамотный русский.
3. Используй ТОЛЬКО id из списка фактов ("finding_ids").
4. Каждый пункт — минимум 2 независимых факта.


Верни строго JSON:
{{
"summary_points": [
  {{"feature": "Новая тема", "description": "Описание", "sentiment": "positive", "finding_ids": ["f5", "f9"]}}
]
}}

ПРИМЕР ПРАВИЛЬНОГО пункта:
✅ {{
  "feature": "Реиграбельность",
  "description": "Множество билдов и оружия делают повторные прохождения интересными.",
  "sentiment": "positive",
  "finding_ids": ["f22", "f41", "f58"]
}}
← Все три finding'а имеют topic про реиграбельность/билды/оружие.

❌ {{
  "feature": "Звук",
  "description": "Хорошая музыка.",
  "finding_ids": ["f7"]
}}
← f7 = "Игра года, 10/10" (topic: Рекомендация). Единственный finding не про звук. Невалидно.

Список фактов:
{combined_data}
"""
            result_2 = self._call_llm(
                prompt_2, "Reduce-саммари (Pass 2)",
                num_predict=REDUCE_NUM_PREDICT,
                enable_thinking=True,
            )
            valid_points.extend(
                self._validate_points(
                    result_2.get("summary_points", []),
                    by_id,
                    existing_features=existing_features,
                )
            )

        # Markdown
        md = "### Ключевые особенности игры\n\n"
        if not valid_points:
            return md + "*Не удалось выделить достоверные особенности на основе предоставленных отзывов.*\n"

        for i, p in enumerate(valid_points, 1):
            md += f"**{i}. {p['feature']}**\n{p['description']}\n"
            if p.get("quote"):
                md += f"> *{p['quote']}*\n"
            md += "\n"
        return md

    # ---------- Reduce: таблица ----------
    def _reduce_table(
        self,
        mapped_findings: List[Dict[str, Any]],
        game: str,
        user_query: str,
    ) -> str:
        combined_data = json.dumps(mapped_findings, ensure_ascii=False)
        focus = (
            f"Обязательно включи аспекты по запросу: '{user_query}'."
            if user_query else ""
        )

        prompt_1 = f"""
Ты — профессиональный игровой журналист. Дан список фактов из отзывов на игру "{game}".
{focus}

Составь таблицу главных категорий (5–7 штук).

ПРАВИЛА:
1. Сгруппируй факты по темам. Категории уникальны.
2. Тональность: Положительно / Отрицательно / Смешанно.
3. Описание — 1–2 предложения, грамотный русский.
4. Только факты из списка. Не выдумывай.

Верни строго JSON:
{{
"categories": [
  {{"category": "Название", "sentiment": "Тональность", "description": "Описание"}}
]
}}

Список фактов:
{combined_data}
"""
        result = self._call_llm(
            prompt_1, "Reduce-таблица",
            num_predict=REDUCE_NUM_PREDICT,
            enable_thinking=True,
        )
        categories = result.get("categories", [])

        # dedup + markdown
        unique: List[Dict[str, Any]] = []
        seen: set = set()
        for c in categories:
            name = (c.get("category") or "").strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            unique.append(c)

        md = "### Анализ по категориям\n\n"
        if not unique:
            return md + "*Недостаточно достоверных данных для построения таблицы.*\n"

        md += "| Категория | Тональность | Описание |\n| :--- | :--- | :--- |\n"
        for c in unique:
            cat = str(c.get("category", "")).replace("|", "\\|")
            sent = str(c.get("sentiment", "")).replace("|", "\\|")
            desc = str(c.get("description", "")).replace("\n", " ").replace("|", "\\|")
            md += f"| **{cat}** | {sent} | {desc} |\n"
        return md

    # ---------- LLM-обёртка ----------
    def _call_llm(
        self,
        prompt: str,
        stage_name: str,
        num_predict: int,
        max_retries: int = 2,
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        for _ in range(max_retries):
            try:
                options = {
                    "num_ctx": NUM_CTX,
                    "temperature": 0.0,
                    "num_predict": num_predict,
                }
                resp = ollama.chat(
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
                if enable_thinking and "think" in str(e).lower():
                    logger.warning("[%s] think не поддерживается, retry без него", stage_name)
                    enable_thinking = False
                    continue
                logger.error("[%s] Непредвиденная ошибка: %s", stage_name, e)
                raise

        raise LLMJSONError(f"{stage_name}: исчерпаны retries")

# ==========================================
# Пример использования (для отладки)
# ==========================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # === ПОДСТАВЬТЕ НАЗВАНИЕ ФАЙЛА ЗДЕСЬ ===
    CACHE_FILE = "cache/4abcb290a2631f3eee48d29b200b42e7.json"
    # =======================================

    # Опционально: можно переопределить через аргумент командной строки
    # python review_processor.py path/to/cache.json
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

    # Запрос пользователя для проверки фокуса модели
    user_query = "Как там с оптимизацией и боссами?"

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