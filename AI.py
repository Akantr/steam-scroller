from openai import OpenAI

from steam_reviewer import (
    resolve_appid,
    fetch_reviews_cached,
    reviews_dataframe,
    analyze_basic,
    analyze_keywords,
    analyze_trends,
)


appid, name = resolve_appid("Hades")
batch, from_cache = fetch_reviews_cached(appid, max_count=500, language="russian")
df = reviews_dataframe(batch)
stats = analyze_basic(df, query_summary=batch.query_summary)
print(stats["positive_ratio"])
print(analyze_keywords(df)["negative"][:5])   # топ-5 ключевых слов жалоб
print(analyze_trends(df, freq="week")["direction"])  # up | down | flat
appid, name = resolve_appid("Hades")
print(batch)

def get_streaming_response(prompt: str, model: str = "qwen2.5"):
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True  # ← Включаем потоковый режим
    )
    
    full_answer = ""
    for chunk in stream:
        # Извлекаем новый кусочек текста
        new_text = chunk.choices[0].delta.content
        if new_text is not None:
            full_answer += new_text
            print(new_text, end="", flush=True) # Выводим сразу на экран
            
    return full_answer

get_streaming_response(f"составь саммари по отзывам, выдели минимум 2 плюса и 2 минуса. Отзывы {batch}")