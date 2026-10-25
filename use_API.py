from steam_reviewer import (
    resolve_appid,
    fetch_reviews_cached,
    reviews_dataframe,
    analyze_basic,
    analyze_keywords,
    analyze_trends,
    summarize_reviews,
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
