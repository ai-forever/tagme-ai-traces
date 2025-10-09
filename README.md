# TagMe AI Traces

## Описание

**TagMe AI Traces** — это вспомогательная библиотека для отправки диалогов и описаний функций из приложений на базе LangChain (Gigachain) в сервис TagMe. Пакет предоставляет синхронные и асинхронные HTTP‑клиенты, а также удобные декораторы для автоматического логирования вызовов языковых моделей и передачи их в TagMe для разметки и анализа.

## Основные возможности

- Синхронный (`TagmeIntegrationClientSync`) и асинхронный (`TagmeIntegrationClientAsync`) клиенты для REST API TagMe.
- Декораторы `tagme_trace` и `tagme_trace_async` для автоматической отправки диалогов при вызове языковой модели.
- Поддержка произвольных метаданных, которые передаются вместе с диалогом.
- Утилиты для преобразования разных форматов входных данных LangChain в унифицированный формат TagMe.
- Обработка ошибок отсутствующих функций (`MissingFunctionsError`) с возможностью игнорирования.

## Требования

- Python 3.9+
- Установленные зависимости из `requirements/requirements.txt`
- Установленный пакет tagme_ai_traces: `pip install ./<path_to_repo>/cookbook/tagme_ai_traces`
- Действующий токен доступа к TagMe и адрес сервиса.

## Настройка окружения

Перед использованием установите переменные окружения:

- `TAGME_TOKEN` — токен доступа к TagMe. Если значение не задано, его необходимо передать в конструктор клиента.
- `TAGME_BASE_URL` — базовый URL сервиса TagMe (например, `https://tagme.sberdevices.ru/dev/chatwm/plugin_statistics/trace`). Можно переопределить через аргумент `base_url`.

## Примеры

Jupyter Notebook `examples_tagme_ai_traces.ipynb`, демонстрирует интеграцию с LangChain и передачу диалогов в TagMe.

## Быстрый старт

### Синхронный клиент

```python
from tagme_ai_traces import TagmeIntegrationClientSync, FunctionDef

client = TagmeIntegrationClientSync(ignore_missing_functions=False)

functions = [
    FunctionDef(
        name="weather",
        description="Получение прогноза погоды",
        parameters={"type": "object", "properties": {}},
    )
]
client.send_functions(functions)

response = client.send_dialog({"input": [], "metadata": {"custom_meta": {"source": "demo"}}})
print(response)
```

### Асинхронный пример

```python
import asyncio
from tagme_ai_traces import TagmeIntegrationClientAsync

async def main():
    async_client = TagmeIntegrationClientAsync(ignore_missing_functions=True)
    await async_client.send_dialog({"input": [], "metadata": {}})
    await async_client.close()

asyncio.run(main())
```

## Использование callback handler `TagMeAgentTracer` (рекомендуется)

**Назначение.** Коллбек для LangChain/LangGraph, который автоматически собирает:

- `functions: List[FunctionDef]` — описания tools, переданные в модель (из `invocation_params.tools`);
- `dialog: List[ChatMessage]` — нормализованные сообщения диалога (roles: `system|user|assistant|function`), включая `function_call` и ответы функций;
- `metadata: Dict[str, Any]` — ключевые `invocation_params` (model, temperature, max_tokens, …) + `thread_id`.

**Какие события слушает**

- `on_chat_model_start` — «снимок» всей истории батча и набора функций. Если пришёл **новый `thread_id`**, предыдущая сессия автоматически `flush()`-ится, после чего буфер перезаписывается.
- `on_llm_end` — дописывает последнюю генерацию (`AIMessage`) в `dialog` (и, если есть, сериализованные `tool_calls`).

**Идентификация диалога (`thread_id`)**
Трейсер ищет `thread_id` в:

1. `config.configurable.thread_id` → приоритетно;
2. `config.metadata.thread_id`;
3. если не найден — генерирует случайный.

> Рекомендация: при работе с LangGraph + `MemorySaver` прокидывайте **одинаковый `thread_id`** и в `configurable`, и в `metadata`, чтобы чекпойнтер и трейсер видели одну и ту же сессию.

**Схема сообщений**

- `HumanMessage` → `ChatMessage(role="user", content=...)`
- `SystemMessage` → `ChatMessage(role="system", content=...)`
- `AIMessage.tool_calls` → `ChatMessage(role="assistant", function_call=FunctionCall{name,args})` (один элемент на каждый вызов)
- `AIMessage.content` (текст) → `ChatMessage(role="assistant", content=...)`
- `ToolMessage` → `ChatMessage(role="function", content=...)` (это **результат** вызова функции)

Строкизацию контента безопасна: `content: Union[str, list[str|dict]]` приводится к строке (dict → `json.dumps`).

**Отправка / flush**

- `await tracer.flush(reason="…")` — отправляет текущие `functions` и `dialog` в TagMe через клиент и **очищает** буфер.
- Авто-flush происходит при смене `thread_id` (новый диалог).
- Логи: `verbose_send=True` включает краткие `INFO`-сообщения (`"Sending N chat messages..."`), иначе — `DEBUG`.

**Подключение (минимум)**

```python
from tagme_gigachain.callback_handler import TagMeAgentTracer
from langchain_gigachat.chat_models import GigaChat

tracer = TagMeAgentTracer(verbose_send=True)

model = GigaChat(
    model="GigaChat-2-Max",
    verify_ssl_certs=False,
    scope="GIGACHAT_API_CORP",
    callbacks=[tracer],  # глобально к модели
)

# Простой вызов модели
from langchain_core.messages import HumanMessage
tid = "your-thread-id"
resp = model.invoke(
    [HumanMessage("Привет!")],
    config={
        "callbacks": [tracer],
        "configurable": {"thread_id": tid},
        "metadata": {"thread_id": tid},  # синхронизируем для чекпойнтера/логов
    },
)

# Ручной flush после цепочки сообщений
# await tracer.flush("manual")
```

### Подключение к ReAct-агенту (LangGraph)

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(model, tools=[...], checkpointer=MemorySaver(), prompt="…")

tid = "react-session-1"
cfg = {
  "callbacks": [tracer],
  "metadata": {"env": "local", "thread_id": tid},
  "configurable": {"thread_id": tid},
}
agent.invoke({"messages": [("user","привет")]}, config=cfg)
```

## Использование декораторов

Декораторы помогают перехватывать вызовы языковых моделей и автоматически отправлять диалог в TagMe.

### Асинхронный декоратор

```python
from langchain_core.messages import AIMessage, HumanMessage
from tagme_ai_traces import tagme_trace_async

@tagme_trace_async(metadata={"additional_meta": "example"})
async def run_model(messages):
    return AIMessage(content="Ответ", response_metadata={"finish_reason": "stop"})

result = await run_model([HumanMessage(content="Привет")])
```

### Синхронный декоратор

```python
from langchain_core.messages import AIMessage, HumanMessage
from tagme_ai_traces import tagme_trace

@tagme_trace(metadata={"additional_meta": "example"})
def run_model_sync(messages):
    return AIMessage(content="Ответ", response_metadata={})

run_model_sync([HumanMessage(content="Привет")])
```

Оба декоратора принимают дополнительные аргументы:

- `token`, `trust_env`, `ssl`, `ignore_missing_functions`, `base_url` — проксируются в соответствующий клиент.
- `tagme_client` — уже созданный экземпляр клиента (полезно для повторного использования).
- `metadata` — пользовательские метаданные.
- `dialog_transform_fc` — функция, преобразующая входные данные в `DialogData`. По умолчанию используется `form_dialog_data` из `tagme_ai_traces.utils`.

## Дополнительно

### Формирование данных диалога

Функция `form_dialog_data(model_input, model_response, metadata)` приводит разнообразные структуры LangChain (`PromptValue`, списки сообщений, строки, словари) к формату, который ожидает TagMe. Возвращается объект `DialogData` с полями:

- `input` — список сообщений с ролями `system`, `user`, `assistant`, `function`.
- `metadata` — словарь с метаданными вопроса, ответа и пользовательскими значениями.

### Обработка ошибок

Если TagMe не знает описание функции, встреченной в диалоге, сервер возвращает ошибку `FUNCTIONS_NOT_FOUND`. Библиотека выбрасывает исключение `MissingFunctionsError` с перечнем отсутствующих функций. Поведение можно контролировать флагом `ignore_missing_functions` в клиентах и декораторах.

## Логирование

Библиотека использует стандартный модуль `logging`. Чтобы увидеть отладочные сообщения, настройте уровень логирования в приложении:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
