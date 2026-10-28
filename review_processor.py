import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI
from rapidfuzz import fuzz

from intent import OllamaManager, OLLAMA_MODEL_NAME

logger = logging.getLogger(__name__)

# Стрип markdown-оберток ```json ... ```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ReviewProcessor:
    """
    Модуль анализа и суммаризации отзывов (Map-Reduce архитектура).
    Выполняет требования FR-010 (сводка из 7+ черт) и FR-006 (динамическая таблица).
    """

    def __init__(
        self, 
        model_name: str = OLLAMA_MODEL_NAME, 
        batch_size: int = 15,
        ollama_manager: Optional[OllamaManager] = None
    ):
        # Используем переданный менеджер (синглтон) или создаем свой
        self.ollama = ollama_manager if ollama_manager else OllamaManager(model_name=model_name)
        
        self.client = OpenAI(
            base_url=self.ollama.client_url,
            api_key="ollama",
        )
        self.model_name = model_name
        self.batch_size = batch_size

    def process(self, dm_output: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
        """
        Главный метод оркестрации.
        :param dm_output: Результат работы DataManager.
        :param user_query: Оригинальный запрос пользователя (для фокуса модели).
        """
        if dm_output.get("status") not in ("ready_for_summary", "insufficient_data"):
            return dm_output

        reviews = dm_output.get("reviews", [])
        game = dm_output.get("game", "Неизвестная игра")
        meta = dm_output.get("meta", {})

        if not reviews:
            return {"status": "error", "message": "Нет отзывов для анализа.", "data": None}

        # 1. Сбор предупреждений (FR-008, FR-009)
        warnings = self._extract_warnings(dm_output)

        try:
            # 2. Map-этап (Извлечение фактов)
            logger.info(f"Запуск Map-этапа для {len(reviews)} отзывов...")
            mapped_findings = self._map_reviews(reviews, game, user_query)
            
            if not mapped_findings:
                return {
                    "status": "error",
                    "message": "Не удалось извлечь полезную информацию из отзывов.",
                    "data": None
                }

            # 3. Reduce-этап (Генерация текстового саммари)
            logger.info("Запуск Reduce-этапа (Генерация саммари)...")
            summary_md = self._reduce_summary(mapped_findings, game, reviews, user_query)

            # 4. Reduce-этап (Генерация таблицы, если запрошено)
            table_md = None
            sub_tasks = dm_output.get("sub_tasks", []) 
            if sub_tasks and "table" in sub_tasks:
                logger.info("Запуск Reduce-этапа (Генерация таблицы)...")
                table_md = self._reduce_table(mapped_findings, game, reviews, user_query)

            # 5. Формирование финального ответа
            return {
                "status": "success",
                "game": game,
                "summary": summary_md,
                "table": table_md,
                "warnings": warnings,
                "meta": meta
            }

        except Exception as e:
            logger.exception("Критическая ошибка в ReviewProcessor")
            return {
                "status": "error",
                "message": f"Ошибка генерации саммари: {type(e).__name__}: {e}",
                "data": None
            }
        # ВАЖНО: Блок finally с unload_model() убран. 
        # Выгрузкой модели управляет DialogManager, чтобы не делать это на каждом шаге.

    # ==========================================
    # Вспомогательные методы
    # ==========================================

    def _extract_warnings(self, dm_output: Dict[str, Any]) -> List[str]:
        warnings = []
        meta = dm_output.get("meta", {})
        
        if dm_output.get("status") == "insufficient_data" or meta.get("insufficient_data"):
            warnings.append("Внимание: Найдено менее 30 отзывов. Сводка может быть нерепрезентативной.")

        if meta.get("days_since_last_review", 0) > 60:
            warnings.append("Внимание: Последний отзыв был написан более 2 месяцев назад. Информация может быть устаревшей.")
        
        return warnings

    def _map_reviews(self, reviews: List[str], game: str, user_query: str) -> List[Dict[str, Any]]:
        """Разбивает отзывы на батчи и извлекает факты параллельно."""
        batches = [reviews[i:i + self.batch_size] for i in range(0, len(reviews), self.batch_size)]
        all_findings = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, game, user_query): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_findings = future.result()
                    if isinstance(batch_findings, list):
                        all_findings.extend(batch_findings)
                except Exception as e:
                    logger.error(f"Ошибка обработки батча: {e}")
                    
        logger.info(f"Map-этап собрал {len(all_findings)} фактов.")
        return all_findings

    def _process_batch(self, batch: List[str], game: str, user_query: str) -> List[Dict[str, Any]]:
        """Обрабатывает один батч отзывов (Map-этап)."""
        # Убрали focus из Map-этапа. Извлекаем ВСЕ факты.
        prompt = f"""
Ты — аналитик отзывов Steam. Проанализируй предоставленные отзывы на игру "{game}".

Твоя задача — извлечь ВСЕ конкретные факты, которые упоминаются в тексте (сюжет, графика, звук, оптимизация, баги, цена, сложность и т.д.). 
Для КАЖДОГО факта найди точную цитату из текста, которая его подтверждает.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Тема (topic) должна строго соответствовать цитате. Не притягивай цитаты за уши.
2. "quote": ТОЧНАЯ цитата из отзыва (до 250 символов). КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать списки, чеклисты (☐, ☑), наборы цифр. Цитата должна быть осмысленным предложением.
3. Не выдумывай факты, которых нет в тексте.

Верни строго JSON без markdown-оберток:
{{
  "findings": [
    {{
      "topic": "Краткое название темы (например: Оптимизация, Сюжет, Боссы, Баги)",
      "sentiment": "positive/negative/mixed",
      "quote": "ТОЧНАЯ цитата из отзыва"
    }}
  ]
}}

Отзывы:
{json.dumps(batch, ensure_ascii=False)}
"""
        result = self._call_llm(prompt, "Map-этап")
        findings = result.get("findings", [])
        
        # Валидируем структуру
        valid_findings = []
        for f in findings:
            if isinstance(f, dict) and f.get("topic") and f.get("quote"):
                quote = f["quote"]
                # Защита от мусора
                if any(char in quote for char in ["☐", "☑", "✓", "✗"]):
                    continue
                if len(quote.strip()) < 15:
                    continue
                valid_findings.append(f)
        return valid_findings

    def _reduce_summary(self, mapped_findings: List[Dict[str, Any]], game: str, original_reviews: List[str], user_query: str) -> str:
        combined_data = json.dumps(mapped_findings, ensure_ascii=False)
        
        # ПРОХОД 1
        focus = f"Пользователь особенно интересуется темой: '{user_query}'. Поставь этот аспект на первое место." if user_query else ""
        
        prompt_1 = f"""
Ты — профессиональный игровой журналист. Тебе дан список фактов из отзывов на игру "{game}".
{focus}

Составь сводку РОВНО ИЗ 7 ключевых особенностей игры.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Если пользователь спросил про конкретную тему, поставь её на первое место.
2. НО ты ОБЯЗАН дополнить сводку до 7 пунктов, взяв остальные темы из списка фактов. Не останавливайся на 1-2 пунктах!
3. Сгруппируй похожие темы.
4. Напиши интересное, живое описание для каждой особенности. Следи за падежами и склонениями.
5. Используй ТОЛЬКО те цитаты, которые уже есть в списке. Выбирай САМЫЕ ПОНЯТНЫЕ и ОСМЫСЛЕННЫЕ цитаты. Не меняй их.
6. Верни строго JSON без markdown-оберток:
{{
  "summary_points": [
    {{"feature": "Название особенности", "description": "Живое описание", "quote": "Цитата из списка"}}
  ]
}}

Список фактов:
{combined_data}
"""
        result = self._call_llm(prompt_1, "Reduce-саммари (Pass 1)")
        points = result.get("summary_points", [])

        # ПРОХОД 2 (Дополнение)
        if len(points) < 7:
            logger.info(f"Pass 1 дал только {len(points)} пунктов. Запуск Pass 2...")
            existing_features = [p.get("feature", "") for p in points]
            
            prompt_2 = f"""
Ты — профессиональный игровой журналист. Тебе дан список фактов из отзывов на игру "{game}".

Мы уже выделили следующие особенности: {json.dumps(existing_features, ensure_ascii=False)}.

Твоя задача — найти в списке фактов ЕЩЕ {7 - len(points)} ДРУГИХ важных аспектов игры, которых НЕТ в списке выше. НЕ ПОВТОРЯЙ уже найденные темы.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Выдели только новые темы (например: визуал, звук, сюжет, баги, цена).
2. Напиши живое описание. Следи за грамматикой.
3. Используй ТОЛЬКО цитаты из списка фактов. Избегай чеклистов.
4. Верни строго JSON в формате:
{{
  "summary_points": [
    {{"feature": "Название НОВОЙ особенности", "description": "Описание", "quote": "Цитата"}}
  ]
}}

Список фактов:
{combined_data}
"""
            result_2 = self._call_llm(prompt_2, "Reduce-саммари (Pass 2)")
            points.extend(result_2.get("summary_points", []))

        # === ПОСТ-ОБРАБОТКА ===
        unique_points = []
        seen_quotes = set()
        seen_features = set()

        for p in points:
            feature = p.get('feature', '').strip().lower()
            quote = p.get('quote', '')
            quote_normalized = re.sub(r'\s+', ' ', quote).strip().lower()

            if feature in seen_features or quote_normalized in seen_quotes:
                continue

            # Программная защита от чеклистов
            if any(char in quote for char in ["☐", "☑", "✓", "✗"]):
                continue

            if not self._is_quote_valid(quote, original_reviews):
                logger.warning(f"Удален пункт с невалидной цитатой: {p.get('feature')}")
                continue

            seen_features.add(feature)
            seen_quotes.add(quote_normalized)
            unique_points.append(p)

        # Форматируем в Markdown
        md = "### Ключевые особенности игры\n\n"
        if not unique_points:
            md += "*Не удалось выделить достоверные особенности на основе предоставленных отзывов.*\n"
        for i, p in enumerate(unique_points, 1):
            md += f"**{i}. {p.get('feature', 'Особенность')}**\n{p.get('description', '')}\n> *{p.get('quote', '')}*\n\n"
        return md

    def _reduce_table(self, mapped_findings: List[Dict[str, Any]], game: str, original_reviews: List[str], user_query: str) -> str:
        combined_data = json.dumps(mapped_findings, ensure_ascii=False)
        
        # --- ПРОХОД 1 ---
        focus = f"Обязательно включи аспекты, связанные с запросом: '{user_query}'." if user_query else ""
        
        prompt_1 = f"""
Ты — профессиональный игровой журналист и редактор. Тебе дан список фактов из отзывов на игру "{game}".
{focus}

Составь таблицу главных категорий.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Сгруппируй факты по темам.
2. Определи тональность (Положительно/Отрицательно/Смешанно).
3. Напиши грамотное описание (1-2 предложения). Следи за падежами.
4. Верни строго JSON без markdown-оберток:
{{
  "categories": [
    {{"category": "Название", "sentiment": "Тональность", "description": "Описание"}}
  ]
}}

Список фактов:
{combined_data}
"""
        result = self._call_llm(prompt_1, "Reduce-таблица (Pass 1)")
        categories = result.get("categories", [])

        # --- ПРОХОД 2: Дополнение до 5 категорий ---
        if len(categories) < 5 and len(mapped_findings) > len(categories):
            logger.info(f"Pass 1 дал только {len(categories)} категорий. Запуск Pass 2...")
            existing_cats = [c.get("category", "") for c in categories]
            
            prompt_2 = f"""
Ты — профессиональный игровой журналист и редактор. Тебе дан список фактов из отзывов на игру "{game}".

Мы уже выделили категории: {json.dumps(existing_cats, ensure_ascii=False)}.

Добавь ЕЩЕ {5 - len(categories)} НОВЫЕ категории из списка фактов, которых НЕТ выше (например: сюжет, графика, цена, баги). НЕ ПОВТОРЯЙ существующие. Бери категори, исходя из отзывов на игру.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Пиши грамотным, литературным русским языком.
2. Верни строго JSON в формате:
{{
  "categories": [
    {{"category": "Название НОВОЙ категории", "sentiment": "Тональность", "description": "Описание"}}
  ]
}}

Список фактов:
{combined_data}
"""
            result_2 = self._call_llm(prompt_2, "Reduce-таблица (Pass 2)")
            categories.extend(result_2.get("categories", []))

        # Пост-обработка
        unique_categories = []
        seen_categories = set()

        for c in categories:
            cat_name = c.get('category', '').strip()
            if not cat_name or cat_name.lower() in seen_categories:
                continue
            seen_categories.add(cat_name.lower())
            unique_categories.append(c)

        # Форматируем в Markdown-таблицу
        md = "### Анализ по категориям\n\n"
        if not unique_categories:
            return md + "*Недостаточно достоверных данных для построения таблицы.*\n"

        md += "| Категория | Тональность | Описание |\n"
        md += "| :--- | :--- | :--- |\n"
        for c in unique_categories:
            cat = str(c.get('category', '')).replace('|', '\\|')
            sent = str(c.get('sentiment', '')).replace('|', '\\|')
            desc = str(c.get('description', '')).replace('\n', ' ').replace('|', '\\|')
            md += f"| **{cat}** | {sent} | {desc} |\n"

        return md

    def _is_quote_valid(self, quote: str, original_reviews: List[str]) -> bool:
        """
        Проверяет, что цитата реально присутствует в одном из исходных отзывов.
        Использует rapidfuzz для нечеткого поиска (защита от изменения пунктуации LLM).
        """
        if not quote or len(quote.strip()) < 10:
            return False

        def normalize(text: str) -> str:
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)  # Убираем пунктуацию
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        clean_quote = normalize(quote)
        if len(clean_quote) < 10:
            return False

        for review in original_reviews:
            clean_review = normalize(review)
            # 1. Быстрое точное совпадение
            if clean_quote in clean_review:
                return True
            # 2. Нечеткое совпадение (если LLM чуть изменила словоформы)
            if fuzz.partial_ratio(clean_quote, clean_review) >= 90:
                return True
                
        return False

    def _call_llm(self, prompt: str, stage_name: str, max_retries: int = 2) -> Dict[str, Any]:
        """Универсальный вызов LLM с валидацией JSON и ретраями."""
        messages = [{"role": "system", "content": "Ты возвращаешь только валидный JSON."}]
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1500,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                cleaned = _CODE_FENCE_RE.sub("", raw.strip()).strip()
                return json.loads(cleaned)

            except json.JSONDecodeError as e:
                logger.warning(f"[{stage_name}] Попытка {attempt+1}: невалидный JSON. {e}")
                messages.append({
                    "role": "user",
                    "content": "Ошибка: невалидный JSON. Верни строго JSON без markdown."
                })
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                logger.warning(f"[{stage_name}] Попытка {attempt+1}: сетевая ошибка. {e}")
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                logger.error(f"[{stage_name}] Непредвиденная ошибка: {e}")
                raise

        raise ValueError(f"Не удалось получить валидный JSON на этапе {stage_name} после {max_retries} попыток.")


# ==========================================
# Пример использования (для отладки)
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Эмуляция ответа от DataManager
    mock_dm_output = {
        "status": "ready_for_summary",
        "game": "Elden Ring",
        "reviews":  ["Отличное казуальное развлечение для всей семьи. Добрые и яркие персонажи, атмосфера сказки и светлого юмора обязательно понравятся детям и их родителям!",
    "Первое правило игры - не играй\r\nВторое правило игры - можешь поиграть, но под успокоительным, желательно не слабительное, не путайте\r\nТретье правило игры - не играй\r\nЧетвертое правило игры - если играешь, убери всё колющее-режущее, чтобы случайно не порезать себе ужин.\r\nПятое правило игры - не играй\r\nШестое правило игры - поехали на рыбалку пить пиво, посторонитесь этой игры\r\nВсем всего хорошего, люблю вас!",
    " вот и всё… Неплохая получилась история. Интересная, весёлая, порой немного грустная, а главное – поучительная…Она научила нас быть смелыми и не бояться вызовов, которые готовит нам жизнь. Помогала нам добиваться поставленных целей несмотря ни на что. И самое важное – она научила нас по-настоящему любить и не сходить с пути, следуя за своей мечтой. 100% достижений",
    "Спустя несколько часов игры пришёл отец и велел собирать вещи, сказал что не одобряет методы которыми Миадзаки воспитывал его отец. Теперь всю дорогу молчит пока едем куда то в город, кажется началась...поездка в отдел полиции с фиксацией на мне побоев и написанием заявления на того самого японца из Fromsoftware.",
    "(Сложность) - Меня избили ногами похитили и продали в рабство закинули в чулан и кинули гранату туда и в конце повесили на стене входа в грозовой замок 10/10.",
    "Лучшая игра фромов, в которую я играл. Последний раз, что бы я вставал, садился играть, и вылезал следующим утром у меня было только в Зельде Ботве.\n\nОчень интересные загадки, отличный дизайн локаций и особенно разнообразие замков. Куча крутых боссов с хорошими механиками.\n\nЭто апогей того, что можно было сделать",
    "Энд гейм данной игрухи определенного благосклонен к священнослужителям, решившим потратить несколько дней своей жизни на облегчение страданий местных мракобесов, которым уже явно кроме насильного экзорцизма ничего не поможет...",
    "i met kings who lost their kingdoms.\ngods who couldn't save their world.\nand warriors who fought long after they had already lost.\nin a world built on ambition and ruin,\neveryone was trying to preserve a broken order,\nsometimes the only way forward is to let it end.\n\n10/10",
    "На мне перманентный дебафф, каинова печать на лбу. Кажется, я шел или ехал куда-то, прилег отдохнуть, а теперь нету сил подняться. Надо мной алое небо Каэлида. В нем качается красивая голова подсолнуха, проросшего из живота. Я слышу, как по дороге скрипят обозы и огнеметательные машины, плетется солдатчина в рваных сапогах.\nЯ думаю о Кольце. Раскололось — дзынь — и мир затрещал по швам. Даже слова утратили смысл. Войну не зовем войной, зиму признали весной. Ловим ртом снег — оказалось, пепел. \nКолечко, видать, давненько треснуло, да звон только теперь услыхали. Растворился певучий звук — и все будто по-прежнему. Попирают небеса вековые твердыни. Гордо шагают по миру убийцы, ударяют кулаком в железную грудь: не допустим! Повторим! Своих не бросаем! Мужичок торопливо срывает шапку, гаркает троекратное ура.\nЧем утешиться? В ослеплении полубожки врезали по несущей балке своего золотого порядка. Гниль и кровь сочатся сквозь трещины кладки, вот-вот хлынут потоком, и аккуратные кирпичи полетят на головы нам, нечестивцам.\nКто-то утонет, кого-то затопчут, остальные увидят будущее. Новых героев Междуземья — смельчаков и головорезов. Конные и пешие ворвутся они в наше безвременье, побьют без счета зверей и люда, пригвоздят владык к собственным тронам, вскроют сундучки и копилочки, напялят доспехи, снятые с дохлых рыцарей. Со смехом покажут на нас: погляди на этих, в обносках, с рваными хоругвями в куцапых лапах; кряхтя и стеная, упрямо плетутся они, сами не зная куда — полоумные мертвяки!",
    "[h1]Прерванный стрик[/h1]\n\nПрактика последних лет показывает, что чрезмерный хайп вокруг той или иной игры зачастую заканчивается плохо как для самой игры, так и для её создателей. FromSoftware – одна из немногих студий, которой все эти годы удавалось мало того, что оправдать поднимаемый вокруг их игр хайп, так ещё и с каждой новой игрой превосходить самих себя. Но вот выходит Elden Ring, и ты, преданный фанбой Миядзаки и его команды, понимаешь – что-то не так. Поэтому, как обычно, обо всём по порядку.\n\n[b]О сложности.[/b] Пройдя все игры FromSoftware, доступные на ПК, я могу с полной уверенностью сказать, что Elden Ring – самая легкая игра студии, этакий «Souls для самых маленьких». И я не говорю сейчас о пополнении хилок в открытом мире вне костров, об урезании урона со статус-эффектов или упрощении локаций, в которых ловушки и прочие мерзости можно довольно легко вычислить, нет. Я говорю о том, что с появлением открытого мира [u]игра даёт вам карт-бланш на «перекач» и «фейсролл».[/u] При первом прохождении мне буквально пришлось заскипать треть локаций, подземелий и боссов, чтобы не прийти перекаченным к Малении и Радагону, но даже так, к первой я пришел на 110-115 уровнях, а ко второму где-то на 120-125 – и это с учетом того, что я за всю игру практически не ел найденные руны. \n\nВы определенно скажете, что проблема «перекача» существует во всех играх студии, и я с вами полностью соглашусь, с одной лишь оговоркой – в других играх она присутствует не так открыто и преднамеренно. Elden Ring сделана с явной целью заполучить новый приток аудитории, никогда не игравшей в Souls-игры. И это на самом деле не плохо. Плохо то, за счет чего они это делают – манипулируют игроком, заставляя его думать, что прохождением игры он обязан только своему «скиллу». И, знаете, я бы закрыл на это глаза, если бы не та откровенная халтура, с которой я столкнулся в этой игре. \n\n[b]О боевой системе и сражениях.[/b] Если при первом взгляде на Elden Ring боевка показалась вам рескином третьего DS, то вам не показалось. Ничего нового, кроме прыжка, лошади и специальных умений на оружиях добавлено не было. Остальные изменения, вроде контратаки после блокирования даже упоминать не стоит, поскольку всё это не важно – [u]кувырок и грамотная позиционка по-прежнему выигрывают любой файт.[/u] И опять же, я не возникал бы так сильно, поскольку люблю боевку DS3, но ведь они даже поленились сделать нормальных новых мобов. Минимум процентов шестьдесят мобов в игре – это полное или частичное копирование из третьего DS. Просто играешь, а в голове бесконечные флешбэки. \n\nВообще, всю игру, глядя на огромное кол-во оружия с аниме-абилками и на тонны заклинаний самого разного калибра, меня не покидало ощущение, что боевка в Elden Ring должна была быть другой – более темповой, с уклоном в ритм-слешер, и обязательным использованием заклинаний или способностей оружия. И если при сражениях с обычными мобами инородность боевки из DS3 не так бросается в глаза, то на боссах её становится трудно не замечать.\n\n[b]О боссах и боссфайтах.[/b] Для подавляющего большинства любителей Souls-like, этот аспект, я думаю, является главным в играх жанра. И именно он, к сожалению, в Elden Ring получился крайне посредственным – [u]ни сами боссы, ни боссфайты не вызывают какого-то «вау-эффекта»,[/u] как это было в прошлых играх студии. Объяснения этому я вижу в нескольких причинах, начнем с основной: игра повторяет ошибку DS2, где кол-во боссов превалирует над их качеством (все мы помним результат этой ошибки – по-настоящему интересные боссы появились только в DLC). В Элдене этой ошибки, признаем, избежать было трудно – открытый мир нужно чем-то заполнять и повторяющиеся второстепенные боссы, в дальнейшем многие из которых будут выступать в качестве обычных мобов, являются допустимым решением. Да, бить одного и тоже рыцаря или гигантского червя в 5-ый раз надоедает, но на это можно закрыть глаза. Однако, нельзя закрыть глаза на основных боссов.  \n\nДержите рецепт того, как испортить абсолютно любой боссфайт: составляем арсенал мувов у босса процентов на семьдесят из атак с огромной задержкой и делаем в конце большинства атак сплэш/аое таким образом, что если игрок увернулся, то его обязательно накроет уроном – поздравляю, вы дизайнер боссфайтов в Elden Ring. Я сейчас скажу вещь, которую каждый игравший в игры Фромов так или иначе отмечал про себя: в играх этой студии [u]боссфайт – это «танец с определенным ритмом и тактом».[/u] Апогей этой идеи мы увидели с вами в Sekiro, где получили боссфайты невероятного качества, эпичности и пафоса. [u]В Elden Ring же на эту идею решили забить,[/u] и по итогу мы имеем легких и по большей части раздражающих боссов, от сражения с которыми не получаешь ни челленджа, ни удовольствия. В игре можно выделить только нескольких более или менее запоминающихся боссов: Маления – практически идеальный и единственный босс в игре, дающий хоть какой-то челлендж, Маликет, первые две стадии Годфри и Радагон. Всё остальное довольно невзрачное и не вызывает особых эмоций. Может быть, боссфайты играют другими красками, если призывать на них духов, использовать баффы и заклинания, но я человек простой – взял в руки Клык Ищейки и пошел кувыркаться без всяких гейств, поэтому для меня боссфайты – это разочарование.\n\n[b]О сюжете, персонажах и квестах.[/b] Говорить о сюжете в играх FromSoftware самое неблагодарное дело на свете – игроки ещё не один год будут копаться в причинно-следственных связях, находить скрытые намёки и прочие неочевидные вещи в истории Elden Ring, но о паре вещей всё же сказать стоит. Во-первых, [u]система повествования из прошлых игр абсолютно не работает в открытом мире,[/u] и во многом это получается из-за отсутствия хоть каких-либо квест/стори-маркеров в игре – я не говорю сейчас о маркерах на карте, которые бы помогали игроку понять, куда двигаться, но говорю о банальном отсутствии простейшего дневника, куда герой мог бы записывать последние события, вроде: «Встретил того-то в таком-то месте, он сказал то-то и то-то». Чем обуславливается отсутствие заметок? Во вселенной игры что, нет письменности, и никто не умеет ни писать, ни читать? Вы можете сказать: «Ну это же Соулс, хардкор, традиции, все дела», а у меня другая точка зрения – введи они дневник в этой игре, то все сразу бы поняли, что участие Мартина в проекте – это всего лишь пиар ход, что история в первом приближении «звёзд с неба не хватает», что квестов крайне мало, и что они посредственные и состоят из пары предложений (как и в прошлых играх), и что держатся они только из-за системы повествования из прошлых игр, как-то так. \n\nВо-вторых, о персонажах. Если в Соулсах, хотелось разбираться в биографии каждого персонажа, и заодно в его роли в сюжете, то в Elden Ring делать этого абсолютно не хочется – персонажи, кроме разве что членов Круглого стола, абсолютно неинтересные. Персонажи в мире Соулса, несмотря на всю его мрачность и обреченность, чувствуются живыми и понятными из-за того, что их наделили человеческими характерами и мотивацией. В Элдене же персонажи – это какие-то куклы, к которым не испытываешь ни интереса, ни эмпатии. Отдельно хочется [u]передать привет «особо одаренным»,[/u] которые посчитали, что дать персонажам созвучные имена – это невероятно круто и креативно (нет).\n\n[b]Об открытом мире.[/b] Бей мобов, собирай мусор – это всё, что можно сказать. О квестах уже было написано, а случайные события – это ночные боссфайты с одними и теми же боссами. Открытый мир для галочки.\n\n[b]Итог.[/b] Самое забавное, что, несмотря на всё вышенаписанное, Elden Ring выглядит и играется лучше подавляющей части выходящих сегодня игр, поэтому я абсолютно не удивлюсь, когда она возьмёт «игру года» и срубит большинство наград на различных выставках и конференциях. Ну, а для себя, я пока оставляю надежду, что Elden Ring – это просто «проба пера», этакая заготовка перед чем-то большим и невероятно интересным (читайте, не убиваю в себе фанбоя), и ухожу ждать DLC, в которых, может быть, хотя бы будут сложные боссфайты. Положительно.",
    "Великолепная игра.\nЕсли вы хотите поиграть со своим другом в коопе, и получить более общий опыт, чем ежели свой собственный, то КРАЙНЕ советую мод Seamless Co-op. Он позволяет играть с друзьями без постоянных подключений, получать все предметы, как сюжетные, так и нет, одновременно, в одном мире - будто вы играете уже не в souls-like игру от фромов, а в полноценную кооп игру. Бан за него не прилетает.",
    "Nice and smooth relaxing date sim\nQuite recommend for cozy evenening gameplay session",
    "Новые механики придумывает тот, у кого старые плохие. А старые очень даже хорошие.",
    "Диз не игре, а стиму, игра куплена 25 числа, произошел автоматический возврат.\nСредства возвращены: 4 мар в 22:59\nВ зависимости от способа оплаты, который вы использовали, возврат средств может занять до 30 дней.\nСрок действия карты закончится раньше, спасибо блять)))\n\n",
    "Купил игру, чтобы исследовать красивый фэнтези-мир. Вышел из стартовой локации, увидел красивого рыцаря на золотом коне. Подумал: \"Наверное, это сюжетный персонаж, пойду поздороваюсь\". Спустя две секунды я летел в стратосферу вместе со своими надеждами на спокойный вечер. Эта игра обладает уникальной способностью: ты можешь убить стометрового древнего дракона, почувствовать себя королем мира, а через минуту отлететь на костер, потому что тебя загрызла облезлая собака в каком-то овраге. 10 \"Вы погибли\" из 10. Поставил игре палец вверх, но потом увидел надпись на земле \"Прыгай, там сокровище\", прыгнул в обрыв и теперь хочу поставить дизлайк и автору записки, и своей доверчивости.",
    "Александр Кувшин (лучше бы Мелина достала уже со своим треплом, лучше бы коня нормального подогнала, а то езжу на на осле как Дон Кихот) покойся с миром, прости что из за дезинформации мир потерял такого хорошого кувшина, жаль конечно этого добряка.",
    "Игра интересная. Игра весёлая. Ни капельки не сложная. Маления? Кто такая Маления? Пол миллиарда попыток и смен билдов и всё.... Дело сделано. Нервный срыв получен, а Маления по прежнему жива. 100 I am Malenia, blade of Miquella из 10",
    "My first pain in the ass with this game, is that i'm forced by very game to accept shady deals with a \"quirky and enigmatic lady\" to progress (you are literally softlocked if you don't). It sucks, and it really makes me feel like a lap dog, and i don't like that in the slightest. But apparently, an army of simps out there does love being treated like that. (not even ragebaiting, it's merely my observation, from searching a mod to possibly remove this \"quirky woman\" from the game so i could enjoy it).\nSecondly, NPCs are fool-proofed (made unkillable and immortal), which has led them towards losing their value in my eyes, and their personalities becoming a chore to deal with in order to play the game. Make them killable, make me regret my outrages, rather than making me irritated by these dummyheads who impose their dialogues onto me.\nThings were way better back then when FS made souls games, in where you had to cherish your NPCs and thus, they were valuable to you. In Elden, you have a roster of unkillable dummies who feed you their cheeky opinions and are overall obnoxious, and you can't even take it out on them, you have to bear with their \"clever\" personas. Make game beatable without this revulsing Melina, and don't lock content behind \"servitude\" bullshit, I don't want an NPC enforced on me, and why the hell, in the game where i go kill literal gods and get to the top of the world, can't i take what i want, by force? This makes no sense, except for cheap \"duh muh NPCs are invincible so that playews cannot kill them out of stupidity and accidentally softlock themsewwes or miss some quirky dialogues we wrote\". This is just plain lazy game design. Make OTHER methods of getting from A to B, for god's sake. Won't cost you an arm, eh? Also the choice in the dialogue with Melina to NOT accept her \"accord\" is just laughable. Illusion of free choice, as is.\nI don't like NPCs who are enforced on me, and 2 first souls games got that right. Yeah yeah it's a different game, not even in the souls series. And yet it shows, that From Slopware is becoming the masses-feeding belt, protecting even their players from themselves (game journalist opinion farming, no less), stripping freedom of actions in favor of trying to please the masses. This is truly depressing for the old fans like me, makes me think my choices don't even matter in this game, except for the ending, and yet i love how they perfected the mechanical part in ER. Best fantasy sandbox out there, if we choose to ignore the \"genuis-written\" NPCs that i don't want to ever see. I want consequences of my actions, not this degradation of gaming.\nI do recognize, however, that this review probably won't change a thing in the game at current state, but i just had to yap. Lazy game design and writing has to go away, or we're facing the unheard slopwaves in gaming.",
    "Третий сезон seamless забега и 3.5 сезона в виде DLC объявляется закрытым. После Dark Souls - прям что-то новое. Миллисента - моя дворяночка кста. А про DLC - пошло оно всё на..й. \r\n\r\n\"...Но наши охотники за сокровищами в лицах Стамески и Дардавиль не закончили свои путешествия. Благодать повела их к кострам неизведанных земель королевства Дранглик. Но сколько займёт дорога до далёких земель?\"",
    "Как фанат Borne/Souls игр это шедевр игровой индустрии.  Некоторых может отпугнуть масштаб, либо то что тебя не ведут за ручку маркеры на компасе, но преодолевая цель за целью, которые тебе ставят не многочисленные непеси, открывается многочасовое приключение с превосходной графикой, историей и анимациями.",
    "Появление в игре босса \"изъявленный древесный дух\" было основано на реальных воспоминаниях из детства Бабадзаки. Вскоре, после получения пиздюлей от отца с ремнем, Мастер респавнился не на благодати, как заверено, а на чердаке. Прогуливаясь по этому ужасному месту 2 на 2 квадратных метра, откуда не возьмись, на него выпрыгивал из под полы тридцатиметровая елда матери, в форме щупальцу. Учитывая предпочтения его родителей в плотских утехах, было неожиданно увидеть устройство, которое умеет в два счета подлетать над комнатой, и также быстро сокращать дистанцию, пытаясь  оформить граб-атаку. Когда Бабадзаки смог довести полоску до середины, то елда начала изливать литры шпермы на пол, поджигая и взрывая все вокруг для остроты ощущений гнилью со вкусом клубники. А по окончанию гигантским фаллосом в очко гения, он получал одно золотое семечко. Эта история так впечатлила мастера, что для увеселения народа, в игру он добавил десять таких боссов, чтобы его комьюнити полностью ощутила замысел автора. 10 хуев в жопе гения из 10",
    "Огненный Великан — мой бывший.\n\nРекомендую? Да, рекомендую так, как рекомендуют терапию. Больно, дорого, но помогает.\n\nВся игра — это сплошной роман с разными типами боссов. Но лишь с одним из них у нас сложились по-настоящему серьезные, токсичные отношения. Я про того здоровяка с тарелкой-щитом и одной здоровенной опухшей гниющей ногой.\n\nМы встретились на заснеженном свадебном пустыре. Он был большой, я был полон надежд. Сначала всё шло мило: он катился, я бегал. Как в турецком сериале. Но потом пошло по накатанной: он вечно «уходил по делам», я же пытался «поговорить по душам»; наши «семейные сцены» были эпичны: его огненные фаерболы — это невымытые мною сковородки после ужина, летящие мне в висок. А его пинок одной ногой — это классическое «Я не толкался, это ты сам налетел на меня в коридоре».\n\nМы столько времени провели вместе, что я уже мог претендовать на алименты. Я знал каждый его прыщ на спине, каждый узор на том самом сапоге, которым он меня накрывал. Победа не принесла радости. Пришло облегчение, как после развода. Теперь я свободен и могу встречаться с другими боссами. Но до сих пор вздрагиваю, когда кто-то рядом с шумом перекатывается на офисном кресле у меня на работе.\n\nИгра — огонь. Великан — мой личный пыточник, с которым мы прожили совместную жизнь. 10 оторванных ног из 10, не жалею, но повторять не советую.\n\nP.S. Поток был нашим брачным консультантом. Спасибо тебе, мой верный спутник, получавший урон вместо меня, как мама, закрывающая ребёнка от ударов отцовского ремня.\n\n---------------------\n\nThe Fire Giant is my ex.\n\nDo I recommend it? I recommend it the way you'd recommend therapy. It hurts, it's expensive, but it helps.\n\nThe whole game is like dating cast of bosses. But with only one of them did I develop a truly serious, toxic relationship. I'm talking about that big lad with the plate-shield and one massive, swollen, rotting leg.\n\nWe met on a snowy, wedding-waste of a plateau. He was big, I was full of hope. At first, it was cute: he'd roll, I'd run. Like a Turkish soap opera. But then we fell into a rut: he'd always be \"leaving on business\" (rolling to the other end of an arena the size of three football fields), while I tried to \"have a heart-to-heart\" (chopping at his one good ankle like a deranged woodpecker). Our \"domestic scenes\" were epic: his fireballs were like my unwashed frying pans from dinner, hurled straight at my temple. And his one-legged kick? A classic case of \"I didn't push you, you ran into me in the hallway!\"\n\nWe spent so much time together I could've filed for alimony. I knew every pimple on his back, every pattern on the very boot he'd squash me with. Victory didn't bring joy. It brought relief, like after a divorce. Now I'm free to date other bosses. But I still flinch when someone noisily rolls an office chair near me at work.\n\nThe game is fire. The Giant was my personal tormentor, with whom I shared a whole life. 10 severed legs out of 10, don't regret it, but wouldn't do it again.\n\nP.S. Torrent was our marriage counselor. Thank you, my faithful companion, who took hits for me like a mom shielding her kid from dad's belt.\n\n-----------------------\ntwitch: semp4i_chan\ntg: semp4i_diary",
    "For a newcomer to the Souls series, “Elden Ring” made a huge impression on me. The game offers a vast open world that can be explored at your own pace, finding hidden secrets and unique enemies at every turn. The mysterious storyline is engrossing from the first minutes, making you think about every encounter and every artifact you find. Multiple endings, depending on your decisions, add depth and encourage you to go through the game again to see all possible outcomes. The graphics and level design are stunning, and the combat system demands attention and tactics, making each victory especially satisfying. “Elden Ring” is a thrilling adventure that leaves a lasting impression and a desire to return to this wonderful world again and again.\n\nThose stripped of the Grace of Gold shall all meet death. In the embrace of Messmer's flame",
    "Мертвый онлайн в плане пвп, если вторгаешься то бегаешь от двух + людей. Нет ковенантов\nСломанный баланс.\nЧрезмерная любовь к магии и катанам.\nОптимизацию не завезли. Лагает и вылетает на топовых системах.\nВся сложность игры заключается только в низком уровне персонажа, стоит немного перекачаться как любой бой превращается в одностороннее избиение мобов.\nБесконечное повторение данжей, боссов и мобов.",
    "One of my favourite. Good Story, great replayability, cool characters, good music, wonderful views.",
    "DS I and DS III are better in every aspect of gameplay and level design. Anyway, shutting ds servers down in order to use it's server hardware (which still runs as shit) is spit in the face of community. Hope they won't add horrible easy anticheat to ds series when they will finally return servers back",
    "Основная игра была интересной, на второе прохождение полностью прокаченным было неинтересно играть. \nЖелание играть появилось после выхода ДЛС, но увы ДЛС в конечном счёте не принесло тех же ощущений, что были от основной игры.",
    "дикие дропы фпс при отсутствии графона; какая-то бабка на выходе из храма обделась на мою агрессию и 2 часа не давала уйти с костра, методично начищая мне рожу букетом роз и натравливая диких красных пчёл. 10 фпс из 120(",
    "Selling gear when you can find exactly the same thing, just with a different texture – that really is the height of greed. Yes, it’s only $5, but everything this DLC adds can easily be found on Nexus Mods, and even better versions at that.",
    "Типа Zelda Breath of the Wild, но вместо того чтобы под каждым камнем было одно из 900 семян, тут это один из 900 способов, которыми разрабы троллят игроков. Наименее душная из Душ. Рекомендую",
    "Рот болит и попе больно, в дилдо ринг играть прикольно\n\nP.S - Хоть мой отзыв и стал мемом, но я хочу поменять оценку на время, ибо Бандай Намко переплюнули остальных. Они начали отбирать у других игроков копию игры и насильно возвращать деньги за неё, что полностью противоречит правилам рефанда в стиме.",
    "Мой отрицательный отзыв в ночь релиза:\n21:9 Нет. Клавамышь убили окончательно. Античит, с которым хз как быть в случае, если иконки боксового к примеру на сони заменить решишь модом. Пропатчить разрешение тоже не сможем, античит не даст. Прекрасно. В общем пипец, ребята. 3090+3950x - едва заметные, но есть фризы, о которых все говорят. Позорище пока что. Жалею, что на пк купил, надо было на соньку брать. Японцы на пк порт вообще болт забили.\n\nВ общем обновляю отзыв:\nСейчас дела намного лучше у игры. Да, античит и отсутствие ultrawide никуда не делись. Производительность фиксанули почти на 100%. Играю в итоге на телеке в 4k с hdr. 60 fps отлично вывозит моя система. Картинка обалденно четкая, такого на ps 5 не видел. А по поводу самой игры, серьезно, после 20 часов скажу - это нечто. В восторге как ребенок. И не я один. Друзья тоже в детском восторге. Мы делимся находками, впечатлениями, рассказываем истории о побежденных и не побежденных страшных противниках, о пройденных и квестах, о не разгаданных загадках, и у всех эти истории разные. И все это по сути в окрестностях первого босса. ОФИГЕТЬ! Игра заставила раскрыть рот от изумления. Как же давно такого от игр не испытывал. Чувство, напоминающие то, что было при знакомствах с готиками и Risen, только помноженное на 100. Невероятно!",
    "Хидетака Миядзаки(Бабадзаки) в очередной раз доказал свой чемпионский титул и отстоял право остаться на Олимпе игровой индустрии. Оглядываясь назад на все тревоги фанатской базы, хочется вздохнуть с облегчением – маститые японцы не подвели, а мы получили солидного претендента на Игру года с серьезной заявкой на победу.\n\nПлюсы:\n\nЖивой и захватывающий мир\nИнтересный эксплоринг\nБодрая боевка, получившая заряд хорошего допинга\nВысокая сложность здорового человека\nПотрясающая работа художников\nНеплохая локализация\nМинусы:\nДвижку пора на пенсию\n\n-----------------------------\n\nИгра определенно стоит вложенных в них денег и сил. Здесь ты учишься на своих ошибках и двигаешься вперед, при этом получая удовольствие от графики, боссов, локаций.\n",
    "На сегодняшний день доступны различные методы лечения геморроя, выбор конкретного из которых зависят от стадии заболевания и выраженности симптомов.\nКонсервативное лечение осуществляется на ранних стадиях развития заболевания и заключается в коррекции диеты, образа жизнедеятельности и приеме специфических лекарственных препаратов.\n\nТрадиционно рекомендации по диете при геморрое включают следующие пункты:\n\nЕжедневно употребляйте 0,5-1,0 кг овощей или фруктов, добавьте в рацион отруби.\nВыпивайте не менее 1.5-2 литров жидкости в сутки.\nОтдавайте предпочтение хлебу из муки грубого помола.\nИсключите из рациона острую, жареную, копченую пищу, а также алкоголь.\nИ бросьте, наконец, курить!!!\nНе подавляйте позывы к дефекации.\nНе пытайтесь убить босса больше чем 40 раз подряд.",
    "Эта игра действительно ошеломляет своим размахом, художественное направление и структурой мира. На каждом углу чувствуешь необходимость остановить игру и просто полюбоваться пейзажем.\nElden Ring — одна из лучших игр начала 2022 года.\nВ ней есть все. Отличные бои, красивые ландшафты, отличное чувство исследования и, как обычно From Software, отличные битвы с боссами.",
    "Вы получаете море удовольствие от анального тр*ха\nБесплатно! От кого угодно! Темные Духи, Боссы, собаки, ПМС болота.\nЧего еше не хватает?! \nДруга который предлагает пойти на 3-х слабейших боссов!))))\nРазве это не счастье?!\nПриглашайте сюда в [h1] Междуземья! [/h1]  Они будут очень рады.\n12 из 10 замороженных кровавых поносов \n\n[h1] Маления СОСЕТ [/h1] \n[h1] нет она реально сосет твою кровь и нервы  [/h1]",
    "If you wanna play this game with your friend - forget it. In this game is a very disgusting pvp system. When you play in coop, you will be invaded by other aggressive players every 5-10 minutes. And this is not about dying, it's about ruining your play. They will run around, and WASTE your time. I just cannot play the game and I have to kill invaders. Seriously? 13 invaders for 2 hours is too much.",
    "In the PC version, unstable framerate may be caused by third party applications that control mouse behavior. Deactivating these third party applications may improve performance.",
    "Ноут не проходит по минимальным требованиям, но всё равно тянет елдарик на высоких. Ожидал \"оптимизацию\" уровня киберпанка, но, к счастью, опасения не оправдались.",
    "Произведение искусства, особенно вторжения! \n\nПВЕ ++++\nПВП++++++++\n\n\n*** *** ***\n\nAn exceptional work of art! Especially invasions.\n\n\nPVE ++++\nPVP++++++++",
    "Предлагаю на законодательном уровне внедрить эту игру в школьную программу как часть развития терпения, дисциплины и профилактики периферического зрения",
    "Красивый дарк соулс в открытом мире с поломанным балансом и дичайшими само повторами.\n7 стражей древа из 10",
    "This successful project clearly lacks Russian voice acting, as there is a lot to read.",
    "Спасибо разработчикам за массаж простаты в течении 118 часов, это было незабываемо !",
    "Игра прикольная, бывают сложности, ну а так легко проходится, всем советую.",
    "Геймплей в две кнопки)",
    "Что может быть лучше чем шедевр купленный за 2500? Правильно, шедевр купленный за 4000)\nПотраченных денег не жалею, окупает полностью",
    "Лишь когда гнилая женщина насаживает тебя на свой меч - становится интересно"
  ],
        "sub_tasks": ["table"], # Теперь sub_tasks прокидывается и таблица сгенерируется
        "meta": {"insufficient_data": True, "days_since_last_review": 65}
    }

    processor = ReviewProcessor()
    
    # Передаем user_query для проверки фокуса
    result = processor.process(mock_dm_output, user_query="Как там с оптимизацией и боссами?")
    
    print("\n=== РЕЗУЛЬТАТ ===")
    print(f"Статус: {result['status']}")
    print(f"Предупреждения: {result['warnings']}")
    if result['status'] == 'success':
        print("\n--- САММАРИ ---")
        print(result['summary'])
        print("\n--- ТАБЛИЦА ---")
        print(result['table'])