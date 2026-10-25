from openai import OpenAI
import os
for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(var, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

client = OpenAI(
    base_url='http://127.0.0.1:11434/v1/',
    api_key='ollama',
    timeout=300.0
)

# Сначала проверим, что сервер жив
try:
    models = client.models.list()
    print("Доступные модели:")
    for m in models.data:
        print(" -", m.id)
except Exception as e:
    print("Не удалось подключиться к Ollama:", e)
    print("Убедитесь, что Ollama запущена и слушает порт 11434.")
    raise SystemExit

# Теперь пробуем запрос
for model_name in ["qwen2.5"]:
    try:
        print(f"\n--- {model_name} ---")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Скажи что ты жив"}],
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Ошибка для {model_name}: {e}")