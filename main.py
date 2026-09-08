import os
import re
import json
import math
import hashlib
import secrets
import time
import random

from typing import Optional
from urllib.parse import urlencode, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

import traceback

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

\# ============================================================
\# CONFIG
\# ============================================================

APP\_NAME = "ASCEND AI"

ADMIN\_PASSWORD = os.getenv("ADMIN\_PASSWORD", "CHANGE\_THIS\_PASSWORD")

SUPABASE\_URL = os.getenv("SUPABASE\_URL", "").rstrip("/")
SUPABASE\_SECRET\_KEY = os.getenv("SUPABASE\_SECRET\_KEY", "")

\# ============================================================
\# LLM SETTINGS (API-ключи, а НЕ логин/пароль от личных аккаунтов)
\# ============================================================

OPENROUTER\_URL = "[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)"
DEEPSEEK\_URL = "[https://api.deepseek.com/chat/completions](https://api.deepseek.com/chat/completions)"
QWEN\_URL = "[https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions](https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions)"
PROVOD\_URL = "[https://api.provod.ai/v1/chat/completions](https://api.provod.ai/v1/chat/completions)"

LLM\_TIMEOUT = 30

OPENROUTER\_FREE\_MODELS = [
    "deepseek/deepseek-chat-v3.1\:free",
    "qwen/qwen3-235b-a22b\:free",
    "deepseek/deepseek-r1-distill-qwen-14b\:free",
    "meta-llama/llama-3.2-3b-instruct\:free",
]

DEEPSEEK\_MODEL = "deepseek-chat"
QWEN\_MODEL = "qwen-plus"

PROVOD\_DEFAULT\_MODEL = os.getenv("PROVOD\_MODEL", "xiaomi/mimo-v2.5")

\_DEFAULT\_SETTINGS = {
    "openrouter\_api\_key": os.getenv("OPENROUTER\_API\_KEY", ""),
    "deepseek\_api\_key": os.getenv("DEEPSEEK\_API\_KEY", ""),
    "qwen\_api\_key": os.getenv("QWEN\_API\_KEY", ""),
    "provod\_api\_key": os.getenv("PROVOD\_API\_KEY", ""),
    "provod\_model": PROVOD\_DEFAULT\_MODEL,
    "llm\_direct\_mode": os.getenv("LLM\_DIRECT\_MODE", "true"),
}

runtime\_settings = dict(\_DEFAULT\_SETTINGS)

def is\_llm\_direct\_mode():
    return get\_setting("llm\_direct\_mode").strip().lower() in ("1", "true", "yes", "on")

def get\_setting(key):
    return runtime\_settings.get(key, "") or ""

def mask\_key(value):
    if not value:
        return ""
    if len(value) <= 8:
        return "\*" \* len(value)
    return value[:4] + "…" + value[-4:]

\# ============================================================
\# SEARCH ENGINES (multi-provider fallback chain)
\# ============================================================

SEARXNG\_INSTANCES = [
    "[https://searx.be](https://searx.be)",
    "[https://searx.tiekoetter.com](https://searx.tiekoetter.com)",
    "[https://searxng.site](https://searxng.site)",
    "[https://search.inetol.net](https://search.inetol.net)",
    "[https://priv.au](https://priv.au)",
    "[https://search.bus-hit.me](https://search.bus-hit.me)",
    "[https://searx.namejeff.xyz](https://searx.namejeff.xyz)",
    "[https://baresearch.org](https://baresearch.org)",
    "[https://opnxng.com](https://opnxng.com)",
    "[https://search.sapti.me](https://search.sapti.me)",
]

SEARXNG\_TIMEOUT = 10
SEARXNG\_MAX\_RETRIES\_PER\_INSTANCE = 2
SEARXNG\_RETRY\_BACKOFF\_BASE = 1.5

\# ============================================================
\# LIMITS
\# ============================================================

MAX\_MEMORY = 30
MAX\_SEARCH\_RESULTS = 6
MAX\_SOURCE\_TEXT = 3500
MAX\_MESSAGE\_LENGTH = 5000
PAGE\_TIMEOUT = 12

\# ============================================================
\# CREDITS / АНТИ-АБУЗ ДЛЯ БЕСПЛАТНОГО ЗАПРОСА
\# ============================================================

FREE\_CREDITS = 1

credits\_cache = {}          # ip\_hash -> {"credits": int, "client\_id": str}
client\_id\_index = {}        # client\_id -> ip\_hash (для админского пополнения)

PRICING\_PLANS = [
    {"id": "start", "title": "Старт", "requests": 50, "price": 79,
     "note": "Для знакомства с нейросетью"},
    {"id": "plus", "title": "Плюс", "requests": 150, "price": 199,
     "note": "Самый популярный вариант", "highlight": True},
    {"id": "pro", "title": "Про", "requests": 400, "price": 449,
     "note": "Для активного использования"},
    {"id": "max", "title": "Макс", "requests": 1000, "price": 899,
     "note": "Максимальная выгода за запрос"},
]

SUPPORT\_TELEGRAM = "[https://t.me/lovnff](https://t.me/lovnff)"
PRIVACY\_POLICY\_URL = "[https://telegra.ph/Politika-konfidencialnosti-09-06-116](https://telegra.ph/Politika-konfidencialnosti-09-06-116)"
TERMS\_OF\_USE\_URL = "[https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54](https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54)"

def hash\_ip(request: Request) -> str:
    ip = (request.client.host if request.client else "") or "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    return hashlib.sha256(("ascend-credits:" + ip).encode()).hexdigest()[:32]

def load\_credit\_record(ip\_hash: str):
    if ip\_hash in credits\_cache:
        return credits\_cache[ip\_hash]

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:
        rows = supabase\_request(
            "GET", "user\_credits",
            params={"select": "ip\_hash,credits,client\_id", "ip\_hash": f"eq.{ip\_hash}", "limit": "1"},
        )
        if rows:
            record = {"credits": rows[0].get("credits", 0), "client\_id": rows[0].get("client\_id", "")}
            credits\_cache[ip\_hash] = record
            if record["client\_id"]:
                client\_id\_index[record["client\_id"]] = ip\_hash
            return record

    return None

def persist\_credit\_record(ip\_hash: str, record: dict):
    credits\_cache[ip\_hash] = record
    if record.get("client\_id"):
        client\_id\_index[record["client\_id"]] = ip\_hash

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:
        existing = supabase\_request(
            "GET", "user\_credits",
            params={"select": "ip\_hash", "ip\_hash": f"eq.{ip\_hash}", "limit": "1"},
        )
        payload = {"ip\_hash": ip\_hash, "credits": record["credits"], "client\_id": record.get("client\_id", "")}
        if existing:
            supabase\_request("PATCH", "user\_credits", payload, params={"ip\_hash": f"eq.{ip\_hash}"})
        else:
            supabase\_request("POST", "user\_credits", payload)

def get\_or\_create\_credit\_record(ip\_hash: str, client\_id: str = ""):
    record = load\_credit\_record(ip\_hash)

    if record is None:
        record = {"credits": FREE\_CREDITS, "client\_id": client\_id or ""}
        persist\_credit\_record(ip\_hash, record)
        return record

    if client\_id and not record.get("client\_id"):
        record["client\_id"] = client\_id
        persist\_credit\_record(ip\_hash, record)

    return record

def consume\_credit(ip\_hash: str):
    record = credits\_cache.get(ip\_hash)
    if record is None:
        return
    record["credits"] = max(0, record["credits"] - 1)
    persist\_credit\_record(ip\_hash, record)

\# ============================================================
\# ПРОМОКОДЫ
\# ============================================================
\# Промокоды намеренно нигде не отображаются в интерфейсе — их
\# нужно сообщать пользователям вручную (партнёрам, по договорённости
\# и т.д.). Раздел "Промокод" в интерфейсе — это просто форма ввода,
\# сам код в UI/документации не публикуется.

PROMO\_CODES = {
    "AIRENT1000": {"credits": 1000},
}

promo\_codes\_cache = {code: dict(value) for code, value in PROMO\_CODES.items()}  # code -> {"credits": int}
promo\_redeemed\_cache = set()   # "{ip\_hash}:{code}" — уже активированные комбинации

def load\_promo\_codes():
    """Подтягивает дополнительные промокоды из Supabase (если настроен),
    не затирая встроенные коды, заданные в PROMO\_CODES."""
    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        return

    rows = supabase\_request("GET", "promo\_codes", params={"select": "code,credits"})

    for row in rows or []:
        code = str(row\.get("code", "")).strip().upper()
        credits = row\.get("credits")
        if code and isinstance(credits, int) and credits > 0:
            promo\_codes\_cache[code] = {"credits": credits}

    print("Promo codes loaded:", len(promo\_codes\_cache))

def get\_promo\_code(code: str):
    return promo\_codes\_cache.get(code.strip().upper())

def has\_redeemed\_promo(ip\_hash: str, code: str) -> bool:
    key = f"{ip\_hash}:{code}"
    if key in promo\_redeemed\_cache:
        return True

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:
        rows = supabase\_request(
            "GET", "promo\_redemptions",
            params={"select": "id", "ip\_hash": f"eq.{ip\_hash}", "code": f"eq.{code}", "limit": "1"},
        )
        if rows:
            promo\_redeemed\_cache.add(key)
            return True

    return False

def mark\_promo\_redeemed(ip\_hash: str, code: str):
    promo\_redeemed\_cache.add(f"{ip\_hash}:{code}")

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:
        supabase\_request("POST", "promo\_redemptions", {"ip\_hash": ip\_hash, "code": code})

\# ============================================================
\# FASTAPI
\# ============================================================

app = FastAPI(title=APP\_NAME, version="2.1.0")

app.add\_middleware(
    CORSMiddleware,
    allow\_origins=["\*"],
    allow\_credentials=False,
    allow\_methods=["\*"],
    allow\_headers=["\*"],
)

@app.exception\_handler(Exception)
async def unhandled\_exception\_handler(request: Request, exc: Exception):
    print("=" \* 60, flush=True)
    print("UNHANDLED EXCEPTION on", request.method, request.url.path, flush=True)
    traceback.print\_exc()
    print("=" \* 60, flush=True)

    return JSONResponse(
        status\_code=500,
        content={"detail": f"Внутренняя ошибка сервера: {type(exc).\_\_name\_\_}"},
    )

@app.exception\_handler(StarletteHTTPException)
async def http\_exception\_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status\_code=exc.status\_code, content={"detail": exc.detail})

@app.exception\_handler(RequestValidationError)
async def validation\_exception\_handler(request: Request, exc: RequestValidationError):
    print("VALIDATION ERROR on", request.method, request.url.path, ":", exc.errors(), flush=True)
    return JSONResponse(
        status\_code=422,
        content={"detail": "Некорректные данные запроса.", "errors": exc.errors()},
    )

\# ============================================================
\# SUPABASE REST CLIENT
\# ============================================================

def supabase\_request(method, table, data=None, params=None):
    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        return []

    url = f"{SUPABASE\_URL}/rest/v1/{table}"

    if params:
        try:
            url += "?" + urlencode(params, doseq=True)
        except Exception as e:
            print("SUPABASE PARAM ERROR:", repr(e))
            return []

    body = None
    if data is not None:
        try:
            body = json.dumps(data, ensure\_ascii=False).encode("utf-8")
        except Exception as e:
            print("SUPABASE JSON ERROR:", repr(e))
            return []

    headers = {
        "apikey": SUPABASE\_SECRET\_KEY,
        "Authorization": "Bearer " + SUPABASE\_SECRET\_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        response = requests.request(
            method=method, url=url, headers=headers, data=body, timeout=20
        )
    except Exception as e:
        print("SUPABASE REQUEST ERROR:", repr(e))
        return []

    if response.status\_code >= 400:
        print("SUPABASE ERROR:", response.status\_code, response.text[:1000])
        return []

    if not response.text:
        return []

    try:
        return response.json()
    except Exception:
        return []

\# ============================================================
\# TEXT
\# ============================================================

RUSSIAN\_STOPWORDS = {
    "и", "а", "но", "или", "да", "в", "во", "на", "за", "из", "к", "ко",
    "с", "со", "у", "о", "об", "от", "до", "по", "для", "при", "над",
    "под", "не", "ни", "же", "ли", "бы", "как", "что", "это", "этот",
    "эта", "эти", "мне", "меня", "моя", "мой", "есть", "можно", "нужно",
    "надо", "ну", "вот",
}

def normalize(text):
    if not text:
        return ""
    text = str(text)
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    words = normalize(text).split()
    return [w for w in words if w not in RUSSIAN\_STOPWORDS and len(w) >= 2]

def stable\_hash(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

\# ============================================================
\# SYNONYMS
\# ============================================================

SYNONYMS = {
    "жирный": ["жирный", "жирная", "жирную", "сальная", "сальный", "себум", "жирность"],
    "прыщи": ["прыщи", "прыщ", "акне", "угри", "угрей", "высыпания"],
    "лицо": ["лицо", "лица", "лицу", "фейс"],
    "волосы": ["волосы", "волос", "волосяной"],
    "питание": ["питание", "еда", "продукты", "рацион", "диета"],
    "сон": ["сон", "спать", "засыпать", "недосып", "бессонница"],
    "тренировки": ["тренировка", "тренировки", "спорт", "мышцы", "зал", "качаться", "упражнение", "упражнения"],
    "мешки": ["мешки", "отеки", "отек", "под глазами", "глазами"],
    "темные круги": ["темные круги", "темные круги под глазами", "синяки под глазами", "круги под глазами", "синяки"],
    "морщины": ["морщины", "морщина", "складки", "старение"],
    "перхоть": ["перхоть", "перхотью", "перхоти", "себорейный", "шелушение", "шелушится", "кожа головы", "шелушение кожи головы", "шелушится кожа головы"],
}

def expand\_query(text):
    normalized\_text = normalize(text)
    words = tokenize(text)
    expanded = set(words)

    for canonical, variants in SYNONYMS.items():
        found = False
        for variant in variants:
            normalized\_variant = normalize(variant)
            if not normalized\_variant:
                continue
            if normalized\_variant in normalized\_text:
                found = True
                break

        if found:
            expanded.add(canonical)
            for variant in variants:
                for word in tokenize(variant):
                    expanded.add(word)

    return list(expanded)

\# ============================================================
\# NEURAL BRAIN
\# ============================================================

class NeuralBrain:
    def \_\_init\_\_(self):
        self.vocabulary = []
        self.word\_index = {}
        self.categories = []
        self.category\_index = {}
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.ready = False

    def build(self, knowledge):
        vocabulary = set()
        categories = set()

        for item in knowledge:
            text = (
                item.get("question", "")
                \+ " "
                \+ item.get("answer", "")
                \+ " "
                \+ " ".join(item.get("tags", []))
            )
            for word in expand\_query(text):
                vocabulary.add(word)

            category = item.get("category")
            if category:
                categories.add(category)

        self.vocabulary = sorted(vocabulary)
        self.word\_index = {w: i for i, w in enumerate(self.vocabulary)}
        self.categories = sorted(categories)
        self.category\_index = {c: i for i, c in enumerate(self.categories)}

        if not self.vocabulary or not self.categories:
            self.ready = False
            return

        input\_size = len(self.vocabulary)
        hidden\_size = min(128, max(16, input\_size // 2))
        output\_size = len(self.categories)

        rng = np.random.default\_rng(42)

        self.W1 = rng.normal(0, np.sqrt(2 / input\_size), (input\_size, hidden\_size))
        self.b1 = np.zeros(hidden\_size)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden\_size), (hidden\_size, output\_size))
        self.b2 = np.zeros(output\_size)

        self.ready = True

    def vectorize(self, text):
        vector = np.zeros(len(self.vocabulary))
        for word in expand\_query(text):
            index = self.word\_index.get(word)
            if index is not None:
                vector[index] += 1

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / (np.sum(exp) + 1e-9)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = self.relu(z1)
        z2 = h @ self.W2 + self.b2
        output = self.softmax(z2)
        return z1, h, output

    def train(self, knowledge, epochs=180, learning\_rate=0.035):
        self.build(knowledge)

        if not self.ready:
            return {"success": False, "epochs": 0}

        dataset = []
        for item in knowledge:
            question = item.get("question", "")
            tags = item.get("tags", [])
            text = question + " " + " ".join(tags)
            vector = self.vectorize(text)

            category = item.get("category")
            label = self.category\_index.get(category)
            if label is None:
                continue

            dataset.append((vector, label))

        if not dataset:
            return {"success": False, "epochs": 0}

        for \_ in range(epochs):
            for x, label in dataset:
                z1, h, prediction = self.forward(x)

                target = np.zeros(len(self.categories))
                target[label] = 1

                error = prediction - target

                dW2 = np.outer(h, error)
                db2 = error

                dh = error @ self.W2.T
                dz1 = dh \* (z1 > 0)

                dW1 = np.outer(x, dz1)
                db1 = dz1

                self.W2 -= learning\_rate \* dW2
                self.b2 -= learning\_rate \* db2
                self.W1 -= learning\_rate \* dW1
                self.b1 -= learning\_rate \* db1

        return {
            "success": True,
            "epochs": epochs,
            "samples": len(dataset),
            "vocabulary": len(self.vocabulary),
            "categories": len(self.categories),
        }

    def predict(self, text):
        if not self.ready:
            return None, 0.0

        x = self.vectorize(text)
        if not np.any(x):
            return None, 0.0

        \_, \_, output = self.forward(x)
        index = int(np.argmax(output))

        return self.categories[index], float(output[index])

\# ============================================================
\# DEFAULT KNOWLEDGE
\# ============================================================

DEFAULT\_KNOWLEDGE = [
    {
        "title": "Жирная кожа",
        "category": "skin",
        "question": "Что делать если у меня жирная кожа?",
        "answer": """
Если кожа быстро становится жирной, не стоит постоянно
и агрессивно обезжиривать её.

Базовый уход:

1\. Умывай лицо мягким очищающим средством утром и вечером.
2\. Не используй агрессивное мыло и спиртовые средства без необходимости.
3\. Рассмотри средства с ниацинамидом или салициловой кислотой,
   если они подходят твоей коже.
4\. Используй лёгкий увлажняющий крем.
5\. Днём используй солнцезащитное средство.
6\. Не выдавливай воспаления.

Если есть выраженное болезненное акне, лучше обратиться к дерматологу.
""",
        "tags": ["жирная кожа", "себум", "кожа", "лицо", "акне", "прыщи"],
    },
    {
        "title": "Прыщи",
        "category": "skin",
        "question": "Как избавиться от прыщей и акне?",
        "answer": """
При склонности к акне лучше выстроить простой регулярный уход.

Утром:
• мягкое очищение;
• увлажнение;
• солнцезащита.

Вечером:
• очищение;
• средство против акне, подходящее твоей коже;
• увлажнение.

Не начинай сразу несколько новых активных средств.

Если акне тяжёлое, болезненное или оставляет рубцы,
стоит обратиться к дерматологу.
""",
        "tags": ["прыщи", "акне", "угри", "кожа", "лицо"],
    },
    {
        "title": "Улучшение внешности",
        "category": "appearance",
        "question": "Как улучшить внешность?",
        "answer": """
На внешний вид влияет сразу несколько факторов.

Полезная база:

• нормальный режим сна;
• регулярная физическая активность;
• сбалансированное питание;
• уход за кожей;
• уход за волосами;
• личная гигиена;
• солнцезащита;
• подходящая одежда и причёска.

Лучше постепенно улучшать несколько направлений,
чем искать одно чудо-средство.
""",
        "tags": ["внешность", "лицо", "красота", "уход"],
    },
    {
        "title": "Питание",
        "category": "nutrition",
        "question": "Что есть чтобы лучше выглядеть?",
        "answer": """
Для внешнего вида обычно важнее сбалансированный рацион,
чем экстремальная диета.

Старайся регулярно получать:

• достаточное количество белка;
• овощи и фрукты;
• цельные продукты;
• полезные жиры;
• достаточное количество жидкости.

Не нужно исключать целые группы продуктов без конкретной причины.
""",
        "tags": ["питание", "еда", "рацион", "диета", "внешность"],
    },
    {
        "title": "Сон",
        "category": "lifestyle",
        "question": "Как сон влияет на внешность?",
        "answer": """
Стабильный режим сна важен для общего самочувствия.

Полезно:

• ложиться примерно в одно время;
• вставать примерно в одно время;
• уменьшить яркий экран перед сном;
• не употреблять много кофеина поздно вечером;
• обеспечить комфортные условия для сна.

Главное — стабильность режима.
""",
        "tags": ["сон", "режим", "внешность", "лицо", "недосып"],
    },
    {
        "title": "Тренировки",
        "category": "fitness",
        "question": "Как тренироваться чтобы улучшить тело?",
        "answer": """
Для улучшения физической формы можно сочетать силовые тренировки
и кардио.

Основные принципы:

• постепенно увеличивай нагрузку;
• соблюдай технику упражнений;
• тренируй основные мышечные группы;
• оставляй время на восстановление;
• следи за питанием и сном.

Не обязательно тренироваться каждый день.
""",
        "tags": ["тренировки", "спорт", "мышцы", "тело", "зал"],
    },
    {
        "title": "Тёмные круги и синяки под глазами",
        "category": "темные круги",
        "question": "Что делать с тёмными кругами / синяками под глазами?",
        "answer": """
Тёмные круги под глазами обычно связаны с несколькими факторами:
тонкая кожа в этой зоне, недосып, обезвоживание, наследственность,
пигментация или расширенные сосуды.

Что может помочь:

1\. Наладь режим сна (7–9 часов, стабильное время отхода ко сну).
2\. Пей достаточно воды в течение дня.
3\. Используй крем для области вокруг глаз с кофеином,
   витамином К или ретинолом (если кожа не чувствительная).
4\. Прикладывай холодный компресс на несколько минут утром.
5\. Используй солнцезащитный крем — УФ усиливает пигментацию.
6\. Высыпайся и ограничь соль и алкоголь вечером — это уменьшает отёки.

Если круги появились резко, сопровождаются отёком, болью
или другими симптомами — стоит показаться врачу, чтобы
исключить, например, аллергию или проблемы с носовыми пазухами.
""",
        "tags": ["темные круги", "синяки", "мешки", "под глазами", "глаза", "недосып"],
    },
]

\# ============================================================
\# GLOBAL STATE
\# ============================================================

knowledge\_cache = []
brain = NeuralBrain()

\# ============================================================
\# LOAD KNOWLEDGE
\# ============================================================

def load\_knowledge():
    global knowledge\_cache

    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        knowledge\_cache = DEFAULT\_KNOWLEDGE.copy()
        brain.train(knowledge\_cache)
        print("Supabase not configured.")
        print("Using default knowledge:", len(knowledge\_cache))
        return

    rows = supabase\_request(
        "GET",
        "knowledge",
        params={"select": "\*", "approved": "eq.true", "order": "created\_at.desc"},
    )

    if rows:
        knowledge\_cache = rows
    else:
        knowledge\_cache = DEFAULT\_KNOWLEDGE.copy()

    brain.train(knowledge\_cache)

    print("Knowledge:", len(knowledge\_cache))
    print("Brain ready:", brain.ready)

\# ============================================================
\# LOCAL KNOWLEDGE SEARCH
\# ============================================================

def similarity(a, b):
    a\_words = set(expand\_query(a))
    b\_words = set(expand\_query(b))

    if not a\_words or not b\_words:
        return 0.0

    intersection = len(a\_words & b\_words)
    union = len(a\_words | b\_words)

    return intersection / max(1, union)

def search\_local\_knowledge(query):
    predicted\_category, confidence = brain.predict(query)

    results = []
    query\_expanded = set(expand\_query(query))

    for item in knowledge\_cache:
        question = item.get("question", "")
        tags = " ".join(item.get("tags", []))
        title = item.get("title", "")

        score\_question = similarity(query, question)
        score\_tags = similarity(query, tags)
        score\_title = similarity(query, title)

        direct\_similarity = max(score\_question, score\_tags, score\_title)

        category\_bonus = 0.0
        if (
            predicted\_category
            and item.get("category") == predicted\_category
            and direct\_similarity >= 0.08
        ):
            category\_bonus = confidence \* 0.15

        score = (
            score\_question \* 0.50
            \+ score\_tags \* 0.25
            \+ score\_title \* 0.15
            \+ category\_bonus
        )

        combined\_text = normalize(question + " " + title + " " + tags)

        exact\_bonus = 0.0
        for word in query\_expanded:
            if len(word) >= 4 and word in combined\_text:
                exact\_bonus += 0.03
        exact\_bonus = min(exact\_bonus, 0.15)

        score += exact\_bonus

        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)

    filtered\_results = [item for item in results if item[0] >= 0.10]

    print("LOCAL QUERY:", query)
    print("LOCAL PREDICTED CATEGORY:", predicted\_category)
    print("LOCAL CATEGORY CONFIDENCE:", confidence)

    if filtered\_results:
        print(
            "LOCAL TOP RESULT:",
            filtered\_results[0][1].get("title", ""),
            "score=",
            filtered\_results[0][0],
        )
    else:
        print("LOCAL RESULT: none")

    return filtered\_results[:5]

\# ============================================================
\# WEB HELPERS
\# ============================================================

def clean\_text(text):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()

def valid\_http\_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False

BROWSER\_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

def \_parse\_searxng\_payload(payload, limit):
    raw\_results = payload.get("results", [])
    if not isinstance(raw\_results, list):
        raw\_results = []

    results = []
    seen\_urls = set()

    for raw in raw\_results:
        if not isinstance(raw, dict):
            continue

        title = clean\_text(raw\.get("title", ""))
        url\_value = str(raw\.get("url", "") or raw\.get("link", ""))
        snippet = clean\_text(raw\.get("content", "") or raw\.get("snippet", "") or "")

        if not title:
            continue
        if not valid\_http\_url(url\_value):
            continue
        if url\_value in seen\_urls:
            continue

        seen\_urls.add(url\_value)

        results.append(
            {
                "title": title[:250],
                "url": url\_value,
                "snippet": snippet[:1500],
                "source": "searxng",
            }
        )

        if len(results) >= limit:
            break

    return results

\# ============================================================
\# SEARCH ENGINE 1: SEARXNG (with retry/backoff per instance)
\# ============================================================

def searxng\_search(query, limit=MAX\_SEARCH\_RESULTS):
    query = query.strip()
    if not query:
        return []

    headers = {
        \*\*BROWSER\_HEADERS,
        "Accept": "application/json",
    }

    for instance in SEARXNG\_INSTANCES:
        base = instance.rstrip("/")
        url = base + "/search"

        print("")
        print("------------------------------------------")
        print("SEARXNG QUERY:", query)
        print("SEARXNG INSTANCE:", base)

        for attempt in range(1, SEARXNG\_MAX\_RETRIES\_PER\_INSTANCE + 1):
            try:
                response = requests.get(
                    url,
                    params={
                        "q": query,
                        "format": "json",
                        "language": "ru-RU",
                        "safesearch": "1",
                        "categories": "general",
                    },
                    headers=headers,
                    timeout=SEARXNG\_TIMEOUT,
                    allow\_redirects=True,
                )

                print(
                    f"SEARXNG HTTP (attempt {attempt}):",
                    response.status\_code,
                )

            except Exception as e:
                print(f"SEARXNG REQUEST ERROR (attempt {attempt}):", repr(e))
                break

            if response.status\_code == 429:
                if attempt < SEARXNG\_MAX\_RETRIES\_PER\_INSTANCE:
                    delay = SEARXNG\_RETRY\_BACKOFF\_BASE \* attempt
                    print(f"SEARXNG 429 — retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    print("SEARXNG BAD STATUS: 429 (giving up on instance)")
                    break

            if response.status\_code >= 400:
                print("SEARXNG BAD STATUS:", response.status\_code)
                break

            content\_type = response.headers.get("content-type", "").lower()
            if "json" not in content\_type:
                print("SEARXNG NOT JSON:", content\_type)
                break

            try:
                payload = response.json()
            except Exception as e:
                print("SEARXNG JSON ERROR:", repr(e))
                break

            results = \_parse\_searxng\_payload(payload, limit)

            print("SEARXNG RESULTS:", len(results))
            for i, r in enumerate(results, start=1):
                print(f"SEARXNG RESULT {i}:", r.get("title", "")[:120], r.get("url", ""))

            if results:
                print("SEARXNG SEARCH SUCCESS via", base)
                print("------------------------------------------")
                return results

            print("SEARXNG EMPTY RESULTS")
            break

    print("SEARXNG SEARCH FAILED: ALL INSTANCES")
    print("------------------------------------------")
    return []

\# ============================================================
\# SEARCH ENGINE 2: DUCKDUCKGO HTML (fallback, no API key needed)
\# ============================================================

def duckduckgo\_html\_search(query, limit=MAX\_SEARCH\_RESULTS):
    query = query.strip()
    if not query:
        return []

    url = "[https://html.duckduckgo.com/html/](https://html.duckduckgo.com/html/)"

    print("")
    print("------------------------------------------")
    print("DUCKDUCKGO QUERY:", query)

    try:
        response = requests.post(
            url,
            data={"q": query, "kl": "ru-ru"},
            headers=BROWSER\_HEADERS,
            timeout=SEARXNG\_TIMEOUT,
        )

        print("DUCKDUCKGO HTTP:", response.status\_code)

    except Exception as e:
        print("DUCKDUCKGO REQUEST ERROR:", repr(e))
        return []

    if response.status\_code >= 400:
        print("DUCKDUCKGO BAD STATUS:", response.status\_code)
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print("DUCKDUCKGO PARSE ERROR:", repr(e))
        return []

    results = []
    seen\_urls = set()

    for result\_div in soup.select(".result"):
        link = result\_div.select\_one("a.result\_\_a")
        if not link:
            continue

        title = clean\_text(link.get\_text(" ", strip=True))
        href = link.get("href", "")

        snippet\_tag = result\_div.select\_one(".result\_\_snippet")
        snippet = clean\_text(snippet\_tag.get\_text(" ", strip=True)) if snippet\_tag else ""

        if not title or not valid\_http\_url(href):
            continue
        if href in seen\_urls:
            continue

        seen\_urls.add(href)

        results.append(
            {
                "title": title[:250],
                "url": href,
                "snippet": snippet[:1500],
                "source": "duckduckgo",
            }
        )

        if len(results) >= limit:
            break

    print("DUCKDUCKGO RESULTS:", len(results))
    for i, r in enumerate(results, start=1):
        print(f"DUCKDUCKGO RESULT {i}:", r.get("title", "")[:120], r.get("url", ""))
    print("------------------------------------------")

    return results

\# ============================================================
\# UNIFIED WEB SEARCH (tries all engines in order)
\# ============================================================

SEARCH\_ENGINES = [
    ("searxng", searxng\_search),
    ("duckduckgo", duckduckgo\_html\_search),
]

\# ============================================================
\# GARBAGE / REDIRECT PAGE DETECTION
\# ============================================================

REDIRECT\_STUB\_MARKERS = (
    "please click here if the page does not redirect",
    "click here if you are not redirected",
    "redirecting you to",
    "javascript is disabled",
)

def is\_redirect\_stub(text):
    if not text:
        return False
    normalized = text.strip().lower()
    if len(normalized) < 400 and any(marker in normalized for marker in REDIRECT\_STUB\_MARKERS):
        return True
    return False

def web\_search\_with\_fallback(query, limit=MAX\_SEARCH\_RESULTS):
    for name, engine\_fn in SEARCH\_ENGINES:
        try:
            results = engine\_fn(query, limit)
        except Exception as e:
            print(f"SEARCH ENGINE '{name}' CRASHED:", repr(e))
            results = []

        if results:
            print(f"WEB SEARCH ENGINE USED: {name}")
            return results, name

    print("WEB SEARCH: all engines failed.")
    return [], None

\# ============================================================
\# FETCH WEB PAGE
\# ============================================================

def fetch\_page\_text(url):
    if not valid\_http\_url(url):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ASCEND-AI/2.1)",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    }

    print("FETCH SOURCE:", url)

    try:
        response = requests.get(
            url, headers=headers, timeout=PAGE\_TIMEOUT, allow\_redirects=True
        )

        print("SOURCE HTTP:", response.status\_code)

        if response.status\_code >= 400:
            print("SOURCE ERROR STATUS")
            return ""

        content\_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content\_type:
            print("SOURCE NOT HTML:", content\_type)
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
            tag.decompose()

        text = clean\_text(soup.get\_text(" ", strip=True))
        text = text[:MAX\_SOURCE\_TEXT]

        print("SOURCE TEXT LENGTH:", len(text))

        return text

    except Exception as e:
        print("SOURCE FETCH ERROR:", repr(e))
        return ""

\# ============================================================
\# COLLECT WEB INFORMATION
\# ============================================================

def collect\_web\_information(query):
    print("")
    print("==========================================")
    print("WEB SEARCH START")
    print("QUERY:", query)

    search\_results, engine\_used = web\_search\_with\_fallback(query)

    if not search\_results:
        print("WEB SEARCH: no engine returned results.")
        print("==========================================")
        return []

    print("WEB SEARCH ENGINE:", engine\_used)
    print("WEB SEARCH RESULTS:", len(search\_results))

    enriched = []
    for index, result in enumerate(search\_results, start=1):
        page\_text = fetch\_page\_text(result["url"])

        if is\_redirect\_stub(page\_text):
            print(f"SOURCE {index}: SKIPPED (redirect stub / no real content)")
            page\_text = ""

        if not page\_text and not clean\_text(result.get("snippet", "")):
            print(f"SOURCE {index}: DROPPED (no content, no snippet)")
            continue

        enriched.append({\*\*result, "page\_text": page\_text})

        print(
            f"SOURCE {index}: title={result.get('title', '')[:100]} "
            f"text={len(page\_text)} snippet={len(result.get('snippet', ''))}"
        )

    print("WEB SEARCH COMPLETE")
    print("==========================================")

    return enriched

\# ============================================================
\# SAVE WEB SOURCES
\# ============================================================

def save\_web\_sources(session\_id, query, results):
    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        return

    for result in results:
        payload = {
            "session\_id": session\_id,
            "query": query,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
            "page\_text": result.get("page\_text", ""),
            "source": result.get("source", "web"),
        }

        supabase\_request("POST", "web\_sources", payload)

\# ============================================================
\# WEB CONTEXT
\# ============================================================

def build\_web\_context(results):
    pieces = []

    for index, item in enumerate(results, start=1):
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        page\_text = item.get("page\_text", "")

        text = page\_text or snippet
        if not text:
            continue

        pieces.append(
            f"\nИСТОЧНИК {index}\nНазвание: {title}\nURL: {url}\nИнформация:\n\n{text}\n"
        )

    return "\n".join(pieces)

def clean\_web\_text(results):
    pieces = []
    for item in results:
        text = item.get("page\_text") or item.get("snippet") or ""
        text = clean\_text(text)
        if text:
            pieces.append(text)

    return "\n\n".join(pieces)

\# ============================================================
\# SENTENCES
\# ============================================================

def split\_sentences(text):
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clean\_text(x) for x in parts if len(clean\_text(x)) > 20]

def rank\_sentences(query, text, limit=8):
    sentences = split\_sentences(text)
    qwords = set(expand\_query(query))

    scored = []
    for sentence in sentences:
        swords = set(expand\_query(sentence))
        overlap = len(qwords & swords)

        if overlap:
            score = overlap / math.sqrt(max(1, len(swords)))
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [sentence for \_, sentence in scored[:limit]]

\# ============================================================
\# SETTINGS PERSISTENCE (Supabase key-value table \`app\_settings\`)
\# ============================================================

SETTINGS\_KEYS = (
    "openrouter\_api\_key",
    "deepseek\_api\_key",
    "qwen\_api\_key",
    "provod\_api\_key",
    "provod\_model",
    "llm\_direct\_mode",
)

def load\_settings():
    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        return

    rows = supabase\_request(
        "GET",
        "app\_settings",
        params={"select": "key,value"},
    )

    for row in rows or []:
        key = row\.get("key")
        value = row\.get("value")
        if key in SETTINGS\_KEYS and value:
            runtime\_settings[key] = value

    print("Settings loaded from Supabase:", [k for k in SETTINGS\_KEYS if runtime\_settings.get(k)])

def save\_setting(key, value):
    if key not in SETTINGS\_KEYS:
        raise ValueError(f"Unknown setting key: {key}")

    runtime\_settings[key] = value

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:
        supabase\_request(
            "POST",
            "app\_settings",
            {"key": key, "value": value},
            params={"on\_conflict": "key"},
        )

\# ============================================================
\# LLM ANSWER SYNTHESIS
\# ============================================================

API\_KEY\_SETTINGS = ("openrouter\_api\_key", "deepseek\_api\_key", "qwen\_api\_key", "provod\_api\_key")

def llm\_available():
    return any(get\_setting(k) for k in API\_KEY\_SETTINGS)

def \_chat\_completion\_request(url, api\_key, model, messages, extra\_headers=None):
    headers = {
        "Authorization": f"Bearer {api\_key}",
        "Content-Type": "application/json",
    }
    if extra\_headers:
        headers.update(extra\_headers)

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.4,
            "max\_tokens": 900,
        },
        timeout=LLM\_TIMEOUT,
    )
    return response

def \_try\_openrouter(messages):
    api\_key = get\_setting("openrouter\_api\_key")
    if not api\_key:
        return None

    for model in OPENROUTER\_FREE\_MODELS:
        print("LLM TRY: openrouter /", model)
        try:
            response = \_chat\_completion\_request(OPENROUTER\_URL, api\_key, model, messages)
            print("LLM HTTP:", response.status\_code, "openrouter /", model)
        except Exception as e:
            print("LLM REQUEST ERROR (openrouter):", repr(e), model)
            continue

        if response.status\_code in (404, 429) or response.status\_code >= 400:
            print("LLM SKIP (openrouter):", response.status\_code, model)
            continue

        try:
            content = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("LLM PARSE ERROR (openrouter):", repr(e))
            continue

        if content:
            print("LLM SUCCESS: openrouter /", model)
            return content

    return None

def \_try\_deepseek\_direct(messages):
    api\_key = get\_setting("deepseek\_api\_key")
    if not api\_key:
        return None

    print("LLM TRY: deepseek direct")
    try:
        response = \_chat\_completion\_request(DEEPSEEK\_URL, api\_key, DEEPSEEK\_MODEL, messages)
        print("LLM HTTP:", response.status\_code, "deepseek")
    except Exception as e:
        print("LLM REQUEST ERROR (deepseek):", repr(e))
        return None

    if response.status\_code >= 400:
        print("LLM BAD STATUS (deepseek):", response.status\_code, response.text[:300])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: deepseek direct")
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (deepseek):", repr(e))
        return None

def \_try\_qwen\_direct(messages):
    api\_key = get\_setting("qwen\_api\_key")
    if not api\_key:
        return None

    print("LLM TRY: qwen direct")
    try:
        response = \_chat\_completion\_request(QWEN\_URL, api\_key, QWEN\_MODEL, messages)
        print("LLM HTTP:", response.status\_code, "qwen")
    except Exception as e:
        print("LLM REQUEST ERROR (qwen):", repr(e))
        return None

    if response.status\_code >= 400:
        print("LLM BAD STATUS (qwen):", response.status\_code, response.text[:300])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: qwen direct")
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (qwen):", repr(e))
        return None

def \_try\_provod\_direct(messages):
    api\_key = get\_setting("provod\_api\_key")
    if not api\_key:
        return None

    model = get\_setting("provod\_model") or PROVOD\_DEFAULT\_MODEL

    print("LLM TRY: provod direct /", model)
    try:
        response = \_chat\_completion\_request(PROVOD\_URL, api\_key, model, messages)
        print("LLM HTTP:", response.status\_code, "provod /", model)
    except Exception as e:
        print("LLM REQUEST ERROR (provod):", repr(e))
        return None

    if response.status\_code >= 400:
        print("LLM BAD STATUS (provod):", response.status\_code, response.text[:500])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: provod direct /", model)
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (provod):", repr(e))
        return None

def build\_history\_messages(history):
    if not history:
        return []

    messages = []
    for item in history[-MAX\_MEMORY:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})

    return messages

def call\_llm(system\_prompt, user\_prompt, history=None):
    messages = [{"role": "system", "content": system\_prompt}]
    messages.extend(build\_history\_messages(history))
    messages.append({"role": "user", "content": user\_prompt})

    providers = (
        \_try\_provod\_direct,
        \_try\_openrouter,
        \_try\_deepseek\_direct,
        \_try\_qwen\_direct,
    )

    for provider\_fn in providers:
        result = provider\_fn(messages)
        if result:
            return result

    print("LLM: all providers failed")
    return None

def llm\_answer\_from\_local(query, knowledge\_answer, web\_results, history=None):
    web\_context = build\_web\_context(web\_results) if web\_results else ""

    system\_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Отвечай на русском языке, простым "
        "разговорным текстом, без списков источников и без вставки URL "
        "в текст ответа. Можешь использовать нумерованные шаги или "
        "маркированные пункты для советов, если это уместно. "
        "Не выдумывай медицинские факты — если сомневаешься, порекомендуй "
        "обратиться к врачу."
    )

    user\_prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"Проверенный ответ из базы знаний (используй как основу):\n"
        f"{knowledge\_answer}\n"
    )

    if web\_context:
        user\_prompt += (
            f"\nДополнительная информация из свежего веб-поиска "
            f"(используй только если она релевантна и не противоречит "
            f"базе знаний; не перечисляй источники и не вставляй ссылки):\n"
            f"{web\_context}\n"
        )

    user\_prompt += (
        "\nПерескажи это связным текстом на русском, дружелюбно и по делу."
    )

    return call\_llm(system\_prompt, user\_prompt, history=history)

def llm\_answer\_from\_web(query, web\_results, history=None):
    web\_context = build\_web\_context(web\_results)
    if not web\_context:
        return None

    system\_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Отвечай на русском языке связным "
        "текстом на основе предоставленной информации из интернета. "
        "НЕ перечисляй источники, НЕ вставляй URL и названия сайтов "
        "в текст ответа — просто дай полезный ответ по существу. "
        "Если информации недостаточно для уверенного ответа, честно "
        "скажи об этом и порекомендуй обратиться к специалисту."
    )

    user\_prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"Информация, найденная в интернете:\n{web\_context}\n\n"
        "Дай связный, дружелюбный ответ по существу вопроса."
    )

    return call\_llm(system\_prompt, user\_prompt, history=history)

def llm\_answer\_general(query, history=None):
    system\_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Веб-поиск сейчас недоступен, поэтому "
        "отвечай на основе своих собственных знаний по теме. Если вопрос "
        "затрагивает несколько разных проблем сразу — ответь по каждой "
        "отдельным пунктом. Пиши на русском, дружелюбно и по делу. "
        "Не выдумывай медицинские факты — если сомневаешься, честно "
        "скажи об этом и порекомендуй обратиться к врачу/дерматологу. "
        "В конце ОБЯЗАТЕЛЬНО одной короткой строкой предупреди, что "
        "ответ дан без сверки со свежими источниками из интернета."
    )

    user\_prompt = f"Вопрос пользователя: {query}\n\nДай полезный ответ по существу."

    return call\_llm(system\_prompt, user\_prompt, history=history)

\# ============================================================
\# WEB ANSWER (extractive fallback — used only if LLM unavailable)
\# ============================================================

def fallback\_web\_answer(query, web\_results):
    web\_text = clean\_web\_text(web\_results)
    if not web\_text:
        return ""

    sentences = rank\_sentences(query, web\_text, limit=8)

    if not sentences:
        sentences = split\_sentences(web\_text)[:5]

    if not sentences:
        return ""

    answer = "Я нашёл информацию по твоему вопросу в интернете.\n\n"

    for sentence in sentences:
        answer += "• " + sentence + "\n"

    return answer

\# ============================================================
\# RESPONSE GENERATOR
\# ============================================================

def generate\_response(query, memory, local\_results, web\_results):
    best\_score = 0.0
    best\_item = None

    if local\_results:
        best\_score, best\_item = local\_results[0]

    print("LOCAL BEST SCORE:", best\_score)
    if best\_item:
        print("LOCAL BEST:", best\_item.get("title", ""))
    print("WEB RESULTS:", len(web\_results))
    print("LLM AVAILABLE:", llm\_available())

    if best\_item and best\_score >= 0.18:
        knowledge\_answer = best\_item.get("answer", "").strip()

        if llm\_available():
            llm\_answer = llm\_answer\_from\_local(query, knowledge\_answer, web\_results, history=memory)
            if llm\_answer:
                return llm\_answer

        answer = knowledge\_answer

        if web\_results:
            web\_text = clean\_web\_text(web\_results)
            sentences = rank\_sentences(query, web\_text, limit=4)

            if sentences:
                answer += "\n\nДополнение из актуального поиска:\n"
                for sentence in sentences:
                    answer += "\n• " + sentence

        return answer

    if web\_results:
        if llm\_available():
            llm\_answer = llm\_answer\_from\_web(query, web\_results, history=memory)
            if llm\_answer:
                return llm\_answer

        web\_answer = fallback\_web\_answer(query, web\_results)
        if web\_answer:
            return web\_answer

    if llm\_available():
        print("Falling back to LLM general knowledge (no local/web data)")
        general\_answer = llm\_answer\_general(query, history=memory)
        if general\_answer:
            return general\_answer

    return (
        "Я не смог получить результаты веб-поиска прямо сейчас.\n\n"
        "Попробуй повторить запрос немного позже или сформулировать его подробнее."
    )

\# ============================================================
\# MEMORY
\# ============================================================

def save\_message(session\_id, role, content):
    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        return None

    rows = supabase\_request(
        "POST",
        "chat\_messages",
        {"session\_id": session\_id, "role": role, "content": content},
    )

    if rows:
        return rows[0]

    return None

def get\_memory(session\_id):
    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        return []

    rows = supabase\_request(
        "GET",
        "chat\_messages",
        params={
            "select": "role,content,created\_at",
            "session\_id": f"eq.{session\_id}",
            "order": "created\_at.desc",
            "limit": str(MAX\_MEMORY),
        },
    )

    rows.reverse()
    return rows

\# ============================================================
\# TRAINING LOG
\# ============================================================

def save\_training\_log(question, answer, category, source):
    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:
        return

    supabase\_request(
        "POST",
        "training\_log",
        {
            "question": question,
            "answer": answer,
            "category": category,
            "source": source,
            "approved": True,
        },
    )

\# ============================================================
\# ADMIN AUTH
\# ============================================================

def create\_admin\_token():
    timestamp = str(int(time.time()))
    raw = ADMIN\_PASSWORD + ":" + timestamp
    signature = hashlib.sha256(raw\.encode()).hexdigest()
    return timestamp + "." + signature

def verify\_admin\_token(token):
    if not token:
        return False

    parts = token.split(".")
    if len(parts) != 2:
        return False

    timestamp, signature = parts

    try:
        timestamp\_int = int(timestamp)
    except Exception:
        return False

    if abs(int(time.time()) - timestamp\_int) > 43200:
        return False

    expected = hashlib.sha256((ADMIN\_PASSWORD + ":" + timestamp).encode()).hexdigest()

    return secrets.compare\_digest(signature, expected)

def check\_admin(request: Request):
    token = request.headers.get("X-Admin-Token", "")
    if not verify\_admin\_token(token):
        raise HTTPException(status\_code=401, detail="Нет доступа.")

\# ============================================================
\# MODELS
\# ============================================================

class HistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session\_id: str
    message: str
    client\_id: Optional[str] = ""
    history: Optional[list[HistoryItem]] = []

class CreditsTopUp(BaseModel):
    client\_id: str
    amount: int

class AdminLogin(BaseModel):
    password: str

class KnowledgeCreate(BaseModel):
    title: str
    category: str
    question: str
    answer: str
    tags: list[str] = []

class FeedbackRequest(BaseModel):
    session\_id: str
    message\_id: Optional[str] = None
    rating: int
    comment: Optional[str] = ""

class SettingsUpdate(BaseModel):
    openrouter\_api\_key: Optional[str] = None
    deepseek\_api\_key: Optional[str] = None
    qwen\_api\_key: Optional[str] = None
    provod\_api\_key: Optional[str] = None
    provod\_model: Optional[str] = None
    llm\_direct\_mode: Optional[bool] = None

class PromoRedeem(BaseModel):
    client\_id: Optional[str] = ""
    code: str

class PromoCreate(BaseModel):
    code: str
    credits: int

\# ============================================================
\# HTML
\# ============================================================

HTML = r"""
\<!DOCTYPE html>
\<html lang="ru">
\<head>
\<meta charset="UTF-8">
\<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,interactive-widget=resizes-content">
\<title>ASCEND AI\</title>
\<style>
\:root {
    \--bg: #000000;
    \--bg-soft: #000000;
    \--panel: transparent;
    \--panel-border: rgba(255,255,255,0.10);
    \--panel-hover: rgba(255,255,255,0.07);
    \--text: #ececec;
    \--text-dim: rgba(236,236,236,0.62);
    \--text-faint: rgba(236,236,236,0.38);
    \--accent-a: #10a37f;
)
    \\--accent-b: #10a37f;

    \\--accent-c: #10a37f;

    \\--gradient: #10a37f;

    \\--radius-lg: 26px;

    \\--radius-md: 20px;

    \\--radius-sm: 15px;

    /\\\* Liquid glass tokens \\\*/

    \\--glass-bg: rgba(255,255,255,0.055);

    \\--glass-bg-strong: rgba(255,255,255,0.09);

    \\--glass-border: rgba(255,255,255,0.14);

    \\--glass-border-soft: rgba(255,255,255,0.08);

    \\--glass-blur: blur(28px) saturate(180%);

    \\--glass-blur-strong: blur(40px) saturate(190%);

    \\--glass-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.10);

    \\--glass-shadow-soft: 0 4px 18px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.07);

    \\--app-height: 100dvh;

}

\\\* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }

html, body { height: 100%; }

body {

    margin: 0;

    background: var(--bg);

    color: var(--text);

    font-family: -apple-system, "Segoe UI", "Inter", Arial, sans-serif;

    overscroll-behavior: none;

    -webkit-font-smoothing: antialiased;

    height: var(--app-height);

    position: fixed;

    inset: 0;

    width: 100%;

}

button, textarea, input { font: inherit; color: inherit; }

button { cursor: pointer; }

a { color: inherit; }

svg.icon { width: 18px; height: 18px; flex-shrink: 0; display: block; }

svg.icon path, svg.icon circle, svg.icon rect, svg.icon line, svg.icon polyline { vector-effect: non-scaling-stroke; }

/\\\* -------- Ambient background -------- \\\*/

.bg-glow {

    position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;

}

.blob {

    position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.28;

}

.b1 { width: 480px; height: 480px; background: radial-gradient(circle, #10a37f, transparent 70%); top: -160px; left: -120px; }

.b2 { width: 380px; height: 380px; background: radial-gradient(circle, #19c37d, transparent 70%); bottom: -140px; right: -100px; }

.b3 { width: 300px; height: 300px; background: radial-gradient(circle, #0d7d63, transparent 70%); top: 40%; left: 60%; opacity: 0.16; }

/\\\* -------- App shell -------- \\\*/

.app { position: relative; z-index: 1; display: flex; height: var(--app-height); }

/\\\* -------- Sidebar (glass panel) -------- \\\*/

.sidebar {

    width: 288px; flex-shrink: 0;

    background: var(--glass-bg);

    -webkit-backdrop-filter: var(--glass-blur);

    backdrop-filter: var(--glass-blur);

    border-right: 1px solid var(--glass-border-soft);

    box-shadow: var(--glass-shadow);

    display: flex; flex-direction: column;

    transform: translateX(-100%); transition: transform .32s cubic-bezier(.4,0,.2,1);

    position: fixed; top: 0; left: 0; bottom: 0; z-index: 40;

    padding-top: env(safe-area-inset-top);

    padding-bottom: env(safe-area-inset-bottom);

}

.sidebar.open { transform: translateX(0); }

.sidebar-head { padding: 18px 18px 14px; display: flex; align-items: center; justify-content: space-between; }

.brand { font-weight: 700; font-size: 19px; letter-spacing: .2px; display: flex; align-items: center; gap: 9px; }

.brand-dot {

    width: 9px; height: 9px; border-radius: 50%; background: var(--accent-a);

    box-shadow: 0 0 10px 2px rgba(16,163,127,0.7);

}

.icon-btn {

    width: 40px; height: 40px; border-radius: 50%; border: 1px solid transparent;

    background: transparent; display: flex; align-items: center; justify-content: center;

    transition: background .18s, border-color .18s, transform .12s;

}

.icon-btn\\\:hover { background: var(--panel-hover); border-color: var(--glass-border-soft); }

.icon-btn\\\:active { transform: scale(0.93); }

.new-chat-btn {

    margin: 6px 14px 12px; padding: 12px 14px; border-radius: var(--radius-md);

    border: 1px solid var(--glass-border-soft);

    background: var(--glass-bg-strong);

    -webkit-backdrop-filter: blur(10px);

    backdrop-filter: blur(10px);

    display: flex; align-items: center; gap: 11px; font-weight: 600; font-size: 14.5px;

    color: var(--text); transition: background .18s, transform .12s;

    box-shadow: var(--glass-shadow-soft);

}

.new-chat-btn\\\:hover { background: rgba(255,255,255,0.13); }

.new-chat-btn\\\:active { transform: scale(0.985); }

.chat-list { flex: 1; overflow-y: auto; padding: 4px 12px 10px; -webkit-overflow-scrolling: touch; }

.chat-item {

    display: flex; align-items: center; gap: 8px; padding: 11px 12px; border-radius: var(--radius-sm);

    margin-bottom: 2px; cursor: pointer; transition: background .18s; position: relative;

}

.chat-item\\\:hover { background: var(--panel-hover); }

.chat-item.active { background: var(--glass-bg-strong); box-shadow: inset 0 0 0 1px var(--glass-border-soft); }

.chat-item .title { flex: 1; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; opacity: .92; font-weight: 400; }

.chat-item .del {

    opacity: 0; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center;

    color: var(--text-faint); transition: opacity .15s, background .15s, color .15s; flex-shrink: 0;

}

.chat-item\\\:hover .del { opacity: 1; }

.chat-item .del\\\:hover { background: rgba(255,92,92,0.18); color: #ff8a8a; }

.chat-item .del svg { width: 13px; height: 13px; }

.sidebar-foot { padding: 10px 12px 16px; border-top: 1px solid var(--glass-border-soft); }

.credits-pill {

    display: flex; align-items: center; justify-content: space-between; gap: 8px;

    background: var(--glass-bg); border: 1px solid var(--glass-border-soft); border-radius: var(--radius-sm);

    padding: 11px 13px; font-size: 13px; margin-bottom: 8px; color: var(--text-dim);

}

.credits-pill .label { display: flex; align-items: center; gap: 8px; }

.credits-pill b { color: var(--accent-a); font-weight: 700; }

.foot-links { display: flex; flex-wrap: wrap; gap: 7px; padding: 0 2px; }

.link-chip {

    font-size: 12.5px; padding: 9px 10px; border-radius: var(--radius-sm); border: 1px solid var(--glass-border-soft);

    background: var(--glass-bg); color: var(--text-dim); transition: background .18s, color .18s;

    text-decoration: none; display: flex; align-items: center; gap: 6px; flex: 1 1 auto; justify-content: center;

}

.link-chip\\\:hover { background: var(--panel-hover); color: var(--text); }

.link-chip svg { width: 14px; height: 14px; }

/\\\* -------- Overlay -------- \\\*/

.overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.55); -webkit-backdrop-filter: blur(2px); backdrop-filter: blur(2px); z-index: 30; opacity: 0; pointer-events: none; transition: opacity .28s; }

.overlay.show { opacity: 1; pointer-events: all; }

/\\\* -------- Main column -------- \\\*/

.main-col { flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0; }

.topbar {

    height: 60px; flex-shrink: 0; display: flex; align-items: center; justify-content: space-between;

    padding: 0 14px; padding-top: env(safe-area-inset-top);

    background: rgba(0,0,0,0.35);

    -webkit-backdrop-filter: var(--glass-blur);

    backdrop-filter: var(--glass-blur);

    border-bottom: 1px solid var(--glass-border-soft);

    position: relative; z-index: 20;

}

.topbar-left { display: flex; align-items: center; gap: 10px; }

.topbar-title { font-weight: 700; font-size: 15px; letter-spacing: .2px; }

.topbar-title span { color: var(--accent-a); }

.topbar-right { display: flex; align-items: center; gap: 8px; }

.pill-btn {

    padding: 9px 15px; border-radius: 30px; border: 1px solid var(--glass-border-soft);

    background: var(--glass-bg); font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 7px;

    transition: background .18s, transform .12s;

    color: var(--text-dim);

    -webkit-backdrop-filter: blur(10px);

    backdrop-filter: blur(10px);

}

.pill-btn\\\:hover { background: var(--panel-hover); }

.pill-btn\\\:active { transform: scale(0.96); }

.pill-btn.gradient {

    background: linear-gradient(135deg, var(--accent-a), #19c37d);

    color: #fff; border: none; font-weight: 700;

    box-shadow: 0 4px 16px rgba(16,163,127,0.35), inset 0 1px 0 rgba(255,255,255,0.3);

}

.pill-btn svg { width: 15px; height: 15px; }

/\\\* -------- Chat area -------- \\\*/

.chat-area { flex: 1; overflow-y: auto; padding: 28px 0 10px; min-height: 0; -webkit-overflow-scrolling: touch; overscroll-behavior: contain; }

.chat-inner { width: min(780px, 92%); margin: 0 auto; }

.message { display: flex; margin-bottom: 22px; opacity: 0; transform: translateY(6px) scale(0.98); animation: rise .34s cubic-bezier(.22,1,.36,1) forwards; }

@keyframes rise { to { opacity: 1; transform: translateY(0) scale(1); } }

.message.user { justify-content: flex-end; }

.message.ai { justify-content: flex-start; }

.bubble { max-width: min(680px, 88%); padding: 14px 18px; border-radius: 22px; line-height: 1.6; white-space: pre-wrap; font-size: 15px; }

.message.ai .bubble {

    background: var(--glass-bg);

    border: 1px solid var(--glass-border-soft);

    -webkit-backdrop-filter: blur(16px);

    backdrop-filter: blur(16px);

    box-shadow: var(--glass-shadow-soft);

    border-radius: 22px;

}

.message.user .bubble {

    background: linear-gradient(135deg, #3a3a3c, #2a2a2c);

    color: var(--text); font-weight: 400;

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 22px;

    box-shadow: var(--glass-shadow-soft);

}

.avatar-row { display: flex; gap: 10px; align-items: flex-start; max-width: 88%; }

.avatar {

    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center;

    background: linear-gradient(135deg, var(--accent-a), #19c37d); margin-top: 2px;

    box-shadow: 0 3px 10px rgba(16,163,127,0.4);

}

/\\\* typing indicator \\\*/

.typing-dots { display: flex; gap: 5px; padding: 6px 2px; }

.typing-dots span {

    width: 7px; height: 7px; border-radius: 50%; background: var(--text-dim);

    animation: bounce 1.1s infinite ease-in-out;

}

.typing-dots span\\\:nth-child(2) { animation-delay: .15s; }

.typing-dots span\\\:nth-child(3) { animation-delay: .3s; }

@keyframes bounce { 0%,60%,100% { transform: translateY(0); opacity: .4; } 30% { transform: translateY(-5px); opacity: 1; } }

/\\\* -------- Composer -------- \\\*/

.composer-wrap { padding: 10px 0 calc(18px + env(safe-area-inset-bottom)); flex-shrink: 0; }

.composer-inner { width: min(780px, 92%); margin: 0 auto; }

.composer-box {

    display: flex; align-items: flex-end; gap: 10px;

    background: var(--glass-bg-strong);

    border: 1px solid var(--glass-border);

    -webkit-backdrop-filter: var(--glass-blur-strong);

    backdrop-filter: var(--glass-blur-strong);

    padding: 12px 12px 12px 20px; border-radius: 30px;

    box-shadow: var(--glass-shadow);

    transition: box-shadow .2s, border-color .2s;

}

.composer-box\\\:focus-within { box-shadow: var(--glass-shadow), 0 0 0 2px rgba(16,163,127,0.45); border-color: rgba(16,163,127,0.5); }

.composer-box textarea {

    flex: 1; border: 0; outline: 0; background: transparent; resize: none;

    min-height: 24px; max-height: 160px; padding: 8px 0; line-height: 1.5; font-size: 15px;

}

.composer-box textarea::placeholder { color: var(--text-faint); }

.send-btn {

    width: 40px; height: 40px; border-radius: 50%; border: 0; flex-shrink: 0;

    background: linear-gradient(135deg, var(--accent-a), #19c37d); color: #fff;

    display: flex; align-items: center; justify-content: center;

    transition: transform .15s, opacity .15s;

    box-shadow: 0 4px 14px rgba(16,163,127,0.45), inset 0 1px 0 rgba(255,255,255,0.35);

}

.send-btn svg { width: 17px; height: 17px; margin-left: 1px; }

.send-btn\\\:hover { transform: scale(1.06); }

.send-btn\\\:disabled { opacity: .35; cursor: not-allowed; transform: none; }

.composer-hint {

    text-align: center;

    font-size: 11px;

    color: var(--text-faint);

    margin-top: 8px;

    letter-spacing: .3px;

}

/\\\* -------- Modals -------- \\\*/

.modal-overlay {

    position: fixed; inset: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);

    display: flex; align-items: center; justify-content: center; z-index: 100;

    opacity: 0; pointer-events: none; transition: opacity .28s;

    padding: 20px;

}

.modal-overlay.show { opacity: 1; pointer-events: all; }

.modal-box {

    width: min(480px, 100%); max-height: 86vh; overflow-y: auto;

    background: rgba(24,24,26,0.72);

    -webkit-backdrop-filter: var(--glass-blur-strong);

    backdrop-filter: var(--glass-blur-strong);

    border: 1px solid var(--glass-border); border-radius: var(--radius-lg);

    padding: 28px; transform: translateY(14px) scale(.98); transition: transform .28s;

    box-shadow: var(--glass-shadow);

}

.modal-overlay.show .modal-box { transform: translateY(0) scale(1); }

.modal-box h2 { margin: 0 0 6px; font-size: 19px; }

.modal-box p { color: var(--text-dim); font-size: 13.5px; line-height: 1.6; }

.doc-links { display: flex; flex-direction: column; gap: 9px; margin: 16px 0; }

.doc-link {

    display: flex; align-items: center; justify-content: space-between; padding: 14px 16px;

    border-radius: var(--radius-sm); background: var(--glass-bg); border: 1px solid var(--glass-border-soft);

    text-decoration: none; font-size: 13.5px; font-weight: 600; transition: background .18s;

}

.doc-link .left { display: flex; align-items: center; gap: 10px; }

.doc-link svg { width: 17px; height: 17px; opacity: .85; }

.doc-link\\\:hover { background: var(--panel-hover); }

.modal-actions { display: flex; gap: 10px; margin-top: 18px; }

.btn-primary {

    flex: 1; padding: 13px; border-radius: var(--radius-sm); border: 0;

    background: linear-gradient(135deg, var(--accent-a), #19c37d);

    color: #fff; font-weight: 700; font-size: 14px;

    box-shadow: 0 4px 16px rgba(16,163,127,0.35);

}

.btn-ghost {

    padding: 13px 16px; border-radius: var(--radius-sm); border: 1px solid var(--glass-border-soft);

    background: var(--glass-bg); color: var(--text-dim); font-size: 14px;

}

.promo-input {

    width: 100%; background: rgba(0,0,0,0.35); color: var(--text);

    border: 1px solid var(--glass-border-soft); border-radius: 12px;

    padding: 13px; outline: none; text-transform: uppercase; letter-spacing: .5px;

}

/\\\* pricing \\\*/

.plan-grid { display: flex; flex-direction: column; gap: 10px; margin-top: 14px; }

.plan-card {

    border: 1px solid var(--glass-border-soft); border-radius: var(--radius-md); padding: 16px 18px;

    background: var(--glass-bg); display: flex; align-items: center; justify-content: space-between; gap: 12px;

    transition: border-color .18s;

    box-shadow: var(--glass-shadow-soft);

}

.plan-card.hl { border-color: rgba(16,163,127,0.55); background: rgba(16,163,127,0.10); }

.plan-name { font-weight: 700; font-size: 14.5px; margin-bottom: 2px; }

.plan-note { font-size: 11.5px; color: var(--text-faint); }

.plan-price { font-weight: 800; font-size: 17px; white-space: nowrap; }

.plan-price small { font-weight: 500; font-size: 11px; color: var(--text-faint); display: block; }

.buy-btn {

    padding: 10px 16px; border-radius: 20px; border: 0;

    background: linear-gradient(135deg, var(--accent-a), #19c37d);

    color: #fff; font-weight: 700; font-size: 12.5px; white-space: nowrap;

    box-shadow: 0 3px 10px rgba(16,163,127,0.35);

}

.client-id-note { font-size: 11.5px; color: var(--text-faint); margin-top: 14px; text-align: center; }

.client-id-note code { background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 6px; }

/\\\* -------- Admin -------- \\\*/

.admin { display: none; }

.admin-wrap { width: min(900px, 94%); margin: 0 auto; padding: 30px 0 60px; overflow-y: auto; height: 100%; -webkit-overflow-scrolling: touch; }

.card {

    background: var(--glass-bg); border: 1px solid var(--glass-border-soft); border-radius: var(--radius-lg);

    padding: 22px; margin-bottom: 18px;

    -webkit-backdrop-filter: blur(18px); backdrop-filter: blur(18px);

    box-shadow: var(--glass-shadow-soft);

}

.card h2 { margin-top: 0; display: flex; align-items: center; gap: 9px; }

.card h2 svg { width: 19px; height: 19px; color: var(--accent-a); }

.field { margin-bottom: 15px; }

.field label { display: block; opacity: .65; margin-bottom: 7px; font-size: 13px; }

.field input, .field textarea { width: 100%; background: rgba(0,0,0,0.35); color: var(--text); border: 1px solid var(--glass-border-soft); border-radius: 12px; padding: 13px; outline: none; }

.field textarea { min-height: 150px; resize: vertical; }

.primary {

    background: linear-gradient(135deg, var(--accent-a), #19c37d); color: #fff; border: 0; border-radius: 12px;

    padding: 12px 18px; font-weight: 700; box-shadow: 0 4px 14px rgba(16,163,127,0.35);

}

.knowledge-item { border-top: 1px solid var(--glass-border-soft); padding: 17px 0; }

.knowledge-item\\\:first-child { border-top: 0; }

.badge { display: inline-flex; align-items: center; gap: 6px; background: var(--glass-bg-strong); padding: 5px 10px; border-radius: 8px; font-size: 12px; opacity: .85; }

.badge svg { width: 12px; height: 12px; }

.status { margin-top: 12px; opacity: .7; font-size: 13px; }

.hidden { display: none !important; }

.admin-back { display: inline-flex; align-items: center; gap: 6px; margin-bottom: 16px; font-size: 13px; color: var(--text-dim); }

/\\\* -------- Extra animation & interactivity layer -------- \\\*/

.blob { animation: blobFloat 16s ease-in-out infinite; will-change: transform; }

.b1 { animation-delay: 0s; }

.b2 { animation-delay: -5s; }

.b3 { animation-delay: -10s; }

@keyframes blobFloat {

    0%, 100% { transform: translate(0, 0) scale(1); }

    33% { transform: translate(26px, -18px) scale(1.08); }

    66% { transform: translate(-18px, 22px) scale(0.94); }

}

.brand-dot { animation: dotGlow 2.6s ease-in-out infinite; }

@keyframes dotGlow {

    0%, 100% { box-shadow: 0 0 10px 2px rgba(16,163,127,0.7); transform: scale(1); }

    50% { box-shadow: 0 0 18px 6px rgba(16,163,127,0.95); transform: scale(1.12); }

}

.icon-btn { transition: background .18s, border-color .18s, transform .18s; }

.icon-btn\\\:hover { transform: rotate(8deg); }

.icon-btn\\\:active { transform: scale(0.88) rotate(0deg); }

.new-chat-btn svg { transition: transform .25s ease; }

.new-chat-btn\\\:hover svg { transform: rotate(90deg); }

.chat-item { animation: itemPop .28s cubic-bezier(.34,1.4,.64,1) both; }

@keyframes itemPop {

    from { opacity: 0; transform: translateX(-10px) scale(0.97); }

    to { opacity: 1; transform: translateX(0) scale(1); }

}

.chat-item .title { transition: transform .18s; }

.chat-item\\\:hover .title { transform: translateX(2px); }

.chat-item .del { transform: scale(0.7) rotate(-90deg); transition: opacity .15s, transform .2s, background .15s, color .15s; }

.chat-item\\\:hover .del { transform: scale(1) rotate(0deg); }

.credits-pill b { display: inline-block; transition: transform .2s; }

.credits-pill b.pulse, .pill-btn.pulse { animation: creditsPulse .55s ease; }

@keyframes creditsPulse {

    0% { transform: scale(1); }

    35% { transform: scale(1.22); color: var(--accent-a); }

    100% { transform: scale(1); }

}

.link-chip, .doc-link { transition: background .18s, color .18s, transform .18s; }

.link-chip\\\:hover { transform: translateY(-2px); }

.doc-link\\\:hover { transform: translateX(3px); }

.pill-btn.gradient, .btn-primary, .buy-btn, .primary {

    position: relative;

    overflow: hidden;

    transition: transform .15s, box-shadow .2s;

}

.pill-btn.gradient::after, .btn-primary::after, .buy-btn::after, .primary::after {

    content: "";

    position: absolute;

    top: 0; left: -60%;

    width: 40%; height: 100%;

    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);

    transform: skewX(-20deg);

    animation: shimmerSweep 3.4s ease-in-out infinite;

    pointer-events: none;

}

@keyframes shimmerSweep {

    0% { left: -60%; }

    45% { left: 130%; }

    100% { left: 130%; }

}

.pill-btn.gradient\\\:hover, .btn-primary\\\:hover, .buy-btn\\\:hover, .primary\\\:hover { transform: translateY(-1px); }

.pill-btn.gradient\\\:active, .btn-primary\\\:active, .buy-btn\\\:active, .primary\\\:active { transform: translateY(0) scale(0.97); }

.send-btn { transition: transform .18s cubic-bezier(.34,1.56,.64,1), opacity .15s, box-shadow .2s; }

.send-btn\\\:active\\\:not(\\\:disabled) { transform: scale(0.9); }

.avatar { transition: transform .2s; }

.message\\\:hover .avatar { transform: scale(1.08) rotate(-4deg); }

.plan-card { transition: border-color .2s, transform .2s, box-shadow .2s; }

.plan-card\\\:hover { transform: translateY(-3px); border-color: rgba(16,163,127,0.5); box-shadow: var(--glass-shadow); }

.buy-btn { transition: transform .15s, box-shadow .2s; }

.buy-btn\\\:hover { box-shadow: 0 5px 16px rgba(16,163,127,0.5); }

.card { transition: border-color .2s, transform .2s; }

.card\\\:hover { border-color: rgba(16,163,127,0.3); }

.knowledge-item, .doc-link, .plan-card { animation: fadeSlideIn .35s ease both; }

@keyframes fadeSlideIn {

    from { opacity: 0; transform: translateY(8px); }

    to { opacity: 1; transform: translateY(0); }

}

\\::-webkit-scrollbar { width: 8px; height: 8px; }

\\::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.14); border-radius: 10px; }

@media (max-width: 860px) {

    .sidebar { width: 84%; max-width: 300px; }

    .chat-inner, .composer-inner { width: 94%; }

    .bubble { max-width: 92%; }

}

\\\</style>

\\\</head>

\\\<body>

\\\<div class="bg-glow">\\\<div class="blob b1">\\\</div>\\\<div class="blob b2">\\\</div>\\\<div class="blob b3">\\\</div>\\\</div>

\\\<div class="app">

  \\\<div id="overlay" class="overlay" onclick="closeSidebar()">\\\</div>

  \\\<aside id="sidebar" class="sidebar">

    \\\<div class="sidebar-head">

      \\\<div class="brand">\\\<span class="brand-dot">\\\</span>ASCEND AI\\\</div>

      \\\<button class="icon-btn" onclick="closeSidebar()" title="Закрыть">

        \\\<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">\\\<line x1="6" y1="6" x2="18" y2="18"/>\\\<line x1="18" y1="6" x2="6" y2="18"/>\\\</svg>

      \\\</button>

    \\\</div>

    \\\<button class="new-chat-btn" onclick="newChat()">

      \\\<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">\\\<line x1="12" y1="5" x2="12" y2="19"/>\\\<line x1="5" y1="12" x2="19" y2="12"/>\\\</svg>

      Новый чат

    \\\</button>

    \\\<div id="chatList" class="chat-list">\\\</div>

    \\\<div class="sidebar-foot">

      \\\<div class="credits-pill">

        \\\<span class="label">

          \\\<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;opacity:.7">\\\<rect x="2" y="6" width="20" height="13" rx="3"/>\\\<path d="M2 10h20"/>\\\<circle cx="17" cy="14.5" r="1.4" fill="currentColor" stroke="none"/>\\\</svg>

          Баланс запросов

        \\\</span>

        \\\<b id="creditsBadgeSide">…\\\</b>

      \\\</div>

      \\\<div class="foot-links">

        \\\<button class="link-chip" onclick="openPricing()">

          \\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<rect x="2" y="5" width="20" height="14" rx="3"/>\\\<path d="M2 10h20"/>\\\</svg>

          Тарифы

        \\\</button>

        \\\<button class="link-chip" onclick="openPromo()">

          \\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M20 12v9H4v-9"/>\\\<path d="M2 7h20v5H2z"/>\\\<path d="M12 22V7"/>\\\<path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"/>\\\<path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"/>\\\</svg>

          Промокод

        \\\</button>

        \\\<button class="link-chip" onclick="openDocs()">

          \\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>\\\<path d="M14 3v5h5"/>\\\</svg>

          Документы

        \\\</button>

        \\\<a class="link-chip" href="[[https://t.me/lovnff\](https://t.me/lovnff)](https://t.me/lovnff]\(https://t.me/lovnff\))" target="\\\_blank" rel="noopener noreferrer">

          \\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>\\\</svg>

          Поддержка

        \\\</a>

      \\\</div>

    \\\</div>

  \\\</aside>

  \\\<div class="main-col">

    \\\<header class="topbar">

      \\\<div class="topbar-left">

        \\\<button class="icon-btn" onclick="openSidebar()" title="Чаты">

          \\\<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">\\\<line x1="4" y1="7" x2="20" y2="7"/>\\\<line x1="4" y1="12" x2="20" y2="12"/>\\\<line x1="4" y1="17" x2="20" y2="17"/>\\\</svg>

        \\\</button>

        \\\<div class="topbar-title">ASCEND \\\<span>AI\\\</span>\\\</div>

      \\\</div>

      \\\<div class="topbar-right">

        \\\<button class="pill-btn" id="creditsBadgeTop" onclick="openPricing()">Баланс: …\\\</button>

        \\\<button class="pill-btn gradient" onclick="openPricing()">

          \\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<rect x="2" y="5" width="20" height="14" rx="3"/>\\\<path d="M2 10h20"/>\\\</svg>

          Тарифы

        \\\</button>

      \\\</div>

    \\\</header>

    \\\<section id="chatSection" style="display\\\:flex; flex-direction\\\:column; flex:1; min-height:0;">

      \\\<div id="chatArea" class="chat-area">

        \\\<div class="chat-inner" id="messages">\\\</div>

      \\\</div>

      \\\<div class="composer-wrap">

        \\\<div class="composer-inner">

          \\\<div class="composer-box">

            \\\<textarea id="messageInput" rows="1" placeholder="Напиши свой вопрос...">\\\</textarea>

            \\\<button id="sendButton" class="send-btn" onclick="sendMessage()" title="Отправить">

              \\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">\\\<line x1="5" y1="12" x2="19" y2="12"/>\\\<polyline points="12 5 19 12 12 19"/>\\\</svg>

            \\\</button>

          \\\</div>

          \\\<div class="composer-hint">mekbuda\\\</div>

        \\\</div>

      \\\</div>

    \\\</section>

    \\\<section id="adminSection" class="admin">

      \\\<div class="admin-wrap">

        \\\<div class="admin-back" onclick="location.hash=''; location.reload();">← Вернуться в чат\\\</div>

        \\\<div id="adminLoginCard" class="card">

          \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<circle cx="12" cy="12" r="3"/>\\\<path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>\\\</svg>Админка\\\</h2>

          \\\<p>Вход в панель управления нейросетью.\\\</p>

          \\\<div class="field">

            \\\<label>Пароль\\\</label>

            \\\<input id="adminPassword" type="password" placeholder="Пароль администратора">

          \\\</div>

          \\\<button class="primary" onclick="loginAdmin()">Войти\\\</button>

          \\\<div id="loginStatus" class="status">\\\</div>

        \\\</div>

        \\\<div id="adminPanel" class="hidden">

          \\\<div class="card">

            \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M9.5 2a5.5 5.5 0 0 0-4.5 8.7A5.5 5.5 0 0 0 9 20a2 2 0 0 0 2-2v-1"/>\\\<path d="M14.5 2A5.5 5.5 0 0 1 19 10.7 5.5 5.5 0 0 1 15 20a2 2 0 0 1-2-2v-1"/>\\\</svg>Состояние нейросети\\\</h2>

            \\\<div id="brainStats">Загрузка...\\\</div>

          \\\</div>

          \\\<div class="card">

            \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<rect x="2" y="5" width="20" height="14" rx="3"/>\\\<path d="M2 10h20"/>\\\</svg>Пополнить баланс пользователя\\\</h2>

            \\\<p style="opacity:.7;font-size:13px;margin-top:-8px;">

              Пользователь присылает свой client\\\_id (виден ему в окне "Тарифы") после оплаты вручную —

              вставь его сюда и укажи, сколько запросов начислить.

            \\\</p>

            \\\<div class="field">

              \\\<label>client\\\_id пользователя\\\</label>

              \\\<input id="topupClientId" placeholder="например, 3f9a1c2b-...">

            \\\</div>

            \\\<div class="field">

              \\\<label>Сколько запросов начислить\\\</label>

              \\\<input id="topupAmount" type="number" placeholder="50">

            \\\</div>

            \\\<button class="primary" onclick="topUpCredits()">Начислить\\\</button>

            \\\<div id="topupStatus" class="status">\\\</div>

          \\\</div>

          \\\<div class="card">

            \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M20 12v9H4v-9"/>\\\<path d="M2 7h20v5H2z"/>\\\<path d="M12 22V7"/>\\\<path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z"/>\\\<path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z"/>\\\</svg>Создать промокод\\\</h2>

            \\\<p style="opacity:.7;font-size:13px;margin-top:-8px;">

              Промокоды нигде не показываются в интерфейсе — сообщай их

              вручную. Каждый пользователь (по IP) может активировать

              конкретный код только один раз.

            \\\</p>

            \\\<div class="field">

              \\\<label>Код\\\</label>

              \\\<input id="promoCode" placeholder="Например, WELCOME50" style="text-transform\\\:uppercase;">

            \\\</div>

            \\\<div class="field">

              \\\<label>Сколько запросов начисляет\\\</label>

              \\\<input id="promoCreditsAmount" type="number" placeholder="50">

            \\\</div>

            \\\<button class="primary" onclick="createPromo()">Создать промокод\\\</button>

            \\\<div id="promoCreateStatus" class="status">\\\</div>

          \\\</div>

          \\\<div class="card">

            \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3-3.5 3.5zm-3 0l-2 2"/>\\\</svg>API-ключи для нейросетей\\\</h2>

            \\\<p style="opacity:.7;font-size:13px;margin-top:-8px;">

              Вставляй сюда официальные API-ключи (не пароль от личного кабинета).

              DeepSeek: platform.deepseek.com → API Keys. Qwen: dashscope.console.aliyun.com.

              OpenRouter (бесплатные модели): openrouter.ai → Keys.

            \\\</p>

            \\\<div id="llmStatus" class="status">\\\</div>

            \\\<div class="field">

              \\\<label>OpenRouter API Key\\\</label>

              \\\<input id="openrouterKey" type="password" placeholder="sk-or-v1-...">

            \\\</div>

            \\\<div class="field">

              \\\<label>DeepSeek API Key\\\</label>

              \\\<input id="deepseekKey" type="password" placeholder="sk-...">

            \\\</div>

            \\\<div class="field">

              \\\<label>Qwen (DashScope) API Key\\\</label>

              \\\<input id="qwenKey" type="password" placeholder="sk-...">

            \\\</div>

            \\\<div class="field">

              \\\<label>provod.ai API Key\\\</label>

              \\\<input id="provodKey" type="password" placeholder="sk-...">

            \\\</div>

            \\\<div class="field">

              \\\<label>provod.ai — имя модели (точно как в личном кабинете)\\\</label>

              \\\<input id="provodModel" placeholder="xiaomi/mimo-v2.5">

            \\\</div>

            \\\<div class="field" style="display\\\:flex;align-items\\\:center;gap:10px;">

              \\\<input id="llmDirectMode" type="checkbox" style="width\\\:auto;">

              \\\<label style="margin:0;">Прямой LLM режим (не ходить в веб-поиск, отвечать сразу через LLM)\\\</label>

            \\\</div>

            \\\<button class="primary" onclick="saveSettings()">Сохранить ключи\\\</button>

            \\\<div id="settingsStatus" class="status">\\\</div>

          \\\</div>

          \\\<div class="card">

            \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>\\\<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>\\\</svg>Добавить знание\\\</h2>

            \\\<div class="field">

              \\\<label>Название\\\</label>

              \\\<input id="title" placeholder="Например: Перхоть">

            \\\</div>

            \\\<div class="field">

              \\\<label>Категория\\\</label>

              \\\<input id="category" placeholder="hair">

            \\\</div>

            \\\<div class="field">

              \\\<label>Пример вопроса пользователя\\\</label>

              \\\<input id="question" placeholder="Что делать с перхотью?">

            \\\</div>

            \\\<div class="field">

              \\\<label>Ответ нейросети\\\</label>

              \\\<textarea id="answer" placeholder="Напиши правильный ответ...">\\\</textarea>

            \\\</div>

            \\\<div class="field">

              \\\<label>Теги через запятую\\\</label>

              \\\<input id="tags" placeholder="перхоть, волосы, кожа головы">

            \\\</div>

            \\\<button class="primary" onclick="addKnowledge()">Обучить нейросеть\\\</button>

            \\\<div id="trainStatus" class="status">\\\</div>

          \\\</div>

          \\\<div class="card">

            \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>\\\<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>\\\</svg>База знаний\\\</h2>

            \\\<div id="knowledgeList">Загрузка...\\\</div>

          \\\</div>

          \\\<div class="card">

            \\\<h2>\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<circle cx="12" cy="12" r="10"/>\\\<line x1="2" y1="12" x2="22" y2="12"/>\\\<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>\\\</svg>Последние поиски\\\</h2>

            \\\<div id="webSearchList">Загрузка...\\\</div>

          \\\</div>

        \\\</div>

      \\\</div>

    \\\</section>

  \\\</div>

\\\</div>

\\\<!-- Consent / documents modal -->

\\\<div id="docsModal" class="modal-overlay">

  \\\<div class="modal-box">

    \\\<h2>Прежде чем начать\\\</h2>

    \\\<p>Используя ASCEND AI, ты соглашаешься с пользовательским соглашением и политикой конфиденциальности. Если возникнут вопросы — поддержка всегда на связи.\\\</p>

    \\\<div class="doc-links">

      \\\<a class="doc-link" href="[[https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54\](https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54)](https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54]\(https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54\))" target="\\\_blank" rel="noopener noreferrer">

        \\\<span class="left">\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>\\\<path d="M14 3v5h5"/>\\\<line x1="9" y1="13" x2="15" y2="13"/>\\\<line x1="9" y1="17" x2="13" y2="17"/>\\\</svg>Пользовательское соглашение\\\</span>

        \\\<span>↗\\\</span>

      \\\</a>

      \\\<a class="doc-link" href="[[https://telegra.ph/Politika-konfidencialnosti-09-06-116\](https://telegra.ph/Politika-konfidencialnosti-09-06-116)](https://telegra.ph/Politika-konfidencialnosti-09-06-116]\(https://telegra.ph/Politika-konfidencialnosti-09-06-116\))" target="\\\_blank" rel="noopener noreferrer">

        \\\<span class="left">\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<rect x="4" y="10" width="16" height="10" rx="2"/>\\\<path d="M8 10V7a4 4 0 0 1 8 0v3"/>\\\</svg>Политика конфиденциальности\\\</span>

        \\\<span>↗\\\</span>

      \\\</a>

      \\\<a class="doc-link" href="[[https://t.me/lovnff\](https://t.me/lovnff)](https://t.me/lovnff]\(https://t.me/lovnff\))" target="\\\_blank" rel="noopener noreferrer">

        \\\<span class="left">\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">\\\<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>\\\</svg>Поддержка в Telegram\\\</span>

        \\\<span>↗\\\</span>

      \\\</a>

    \\\</div>

    \\\<div class="modal-actions">

      \\\<button class="btn-primary" onclick="acceptConsent()">Принять и продолжить\\\</button>

    \\\</div>

  \\\</div>

\\\</div>

\\\<!-- Pricing modal -->

\\\<div id="pricingModal" class="modal-overlay">

  \\\<div class="modal-box">

    \\\<h2>Тарифы\\\</h2>

    \\\<p>1-й запрос — бесплатно. Дальше выбери пакет — оплата по СБП появится совсем скоро, а пока баланс пополняется вручную через поддержку.\\\</p>

    \\\<div id="planGrid" class="plan-grid">Загрузка тарифов...\\\</div>

    \\\<div class="client-id-note">Твой ID для оплаты: \\\<code id="clientIdShow">…\\\</code>\\\<br>Пришли его в поддержку вместе с чеком.\\\</div>

    \\\<div class="modal-actions">

      \\\<button class="btn-ghost" style="flex:1" onclick="closeModal('pricingModal')">Закрыть\\\</button>

    \\\</div>

  \\\</div>

\\\</div>

\\\<!-- Promo code modal -->

\\\<div id="promoModal" class="modal-overlay">

  \\\<div class="modal-box">

    \\\<h2>Промокод\\\</h2>

    \\\<p>Есть промокод? Введи его ниже, чтобы пополнить баланс запросов. Каждый код можно активировать только один раз.\\\</p>

    \\\<div class="field" style="margin-top:16px;">

      \\\<input id="promoInput" class="promo-input" placeholder="Введите промокод" autocomplete="off">

    \\\</div>

    \\\<div id="promoStatus" class="status">\\\</div>

    \\\<div class="modal-actions">

      \\\<button class="btn-primary" onclick="applyPromo()">Активировать\\\</button>

      \\\<button class="btn-ghost" onclick="closeModal('promoModal')">Закрыть\\\</button>

    \\\</div>

  \\\</div>

\\\</div>

\\\<script>

// ============================================================

// VIEWPORT / KEYBOARD FIX (mobile Safari & Chrome)

// ============================================================

// Мобильные браузеры при появлении/скрытии клавиатуры меняют высоту

// visualViewport, но НЕ layout viewport, из-за чего body с

// position: fixed мог "залипать" со сдвигом — верхняя панель с

// кнопками уезжала за пределы экрана и не было возможности

// прокрутить обратно. Решение: держим точную высоту приложения в

// CSS-переменной --app-height, синхронизированной с

// window\\.visualViewport, и всегда сбрасываем скролл документа в 0,

// т.к. сам document скроллиться не должен — скроллится только

// внутренняя область чата (.chat-area).

function setAppHeight() {

    const vv = window\\.visualViewport;

    const height = vv ? vv.height : window\\.innerHeight;

    document.documentElement.style.setProperty("--app-height", height + "px");

    // На iOS после скрытия клавиатуры документ иногда остаётся

    // "прокрученным" вверх на величину бывшей клавиатуры — обнуляем.

    window\\.scrollTo(0, 0);

}

setAppHeight();

window\\.addEventListener("resize", setAppHeight);

window\\.addEventListener("orientationchange", setAppHeight);

if (window\\.visualViewport) {

    window\\.visualViewport.addEventListener("resize", setAppHeight);

    window\\.visualViewport.addEventListener("scroll", setAppHeight);

}

document.addEventListener("focusout", () => setTimeout(setAppHeight, 60));

document.addEventListener("focusin", () => setTimeout(setAppHeight, 60));

// ============================================================

// STATE

// ============================================================

const CLIENT\\\_ID\\\_KEY = "ascend\\\_client\\\_id";

const CHATS\\\_KEY = "ascend\\\_chats";

const CURRENT\\\_KEY = "ascend\\\_current\\\_chat";

const CONSENT\\\_KEY = "ascend\\\_consent\\\_v1";

let clientId = localStorage.getItem(CLIENT\\\_ID\\\_KEY);

if (!clientId) {

    clientId = crypto.randomUUID();

    localStorage.setItem(CLIENT\\\_ID\\\_KEY, clientId);

}

let adminToken = localStorage.getItem("ascend\\\_admin\\\_token");

function loadChats() {

    try { return JSON.parse(localStorage.getItem(CHATS\\\_KEY)) || []; } catch { return []; }

}

function saveChats(chats) { localStorage.setItem(CHATS\\\_KEY, JSON.stringify(chats)); }

function msgsKey(id) { return "ascend\\\_msgs\\\_" + id; }

function loadLocalMsgs(id) {

    try { return JSON.parse(localStorage.getItem(msgsKey(id))) || []; } catch { return []; }

}

function saveLocalMsgs(id, msgs) { localStorage.setItem(msgsKey(id), JSON.stringify(msgs.slice(-60))); }

let chats = loadChats();

let currentChatId = localStorage.getItem(CURRENT\\\_KEY);

if (!chats.length) {

    const id = crypto.randomUUID();

    chats = [{ id, title: "Новый чат", updatedAt: Date.now() }];

    currentChatId = id;

    saveChats(chats);

    localStorage.setItem(CURRENT\\\_KEY, id);

} else if (!currentChatId || !chats.find(c => c.id === currentChatId)) {

    currentChatId = chats[0].id;

    localStorage.setItem(CURRENT\\\_KEY, currentChatId);

}

const GREETING = "Привет! Я ASCEND AI\n\nМогу помочь с вопросами об уходе за кожей, лице, внешности, питании, волосах и тренировках. Если ответа нет в моей базе — поищу актуальную информацию в интернете.\n\nЧто тебя интересует?";

// ============================================================

// SIDEBAR / CHAT LIST

// ============================================================

function openSidebar() {

    document.getElementById("sidebar").classList.add("open");

    document.getElementById("overlay").classList.add("show");

}

function closeSidebar() {

    document.getElementById("sidebar").classList.remove("open");

    document.getElementById("overlay").classList.remove("show");

}

function renderSidebar() {

    const list = document.getElementById("chatList");

    list.innerHTML = "";

    const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);

    sorted.forEach(c => {

        const item = document.createElement("div");

        item.className = "chat-item" + (c.id === currentChatId ? " active" : "");

        item.onclick = () => { switchChat(c.id); closeSidebar(); };

        const title = document.createElement("div");

        title.className = "title";

        title.textContent = c.title || "Новый чат";

        const del = document.createElement("div");

        del.className = "del";

        del.innerHTML = '\\\<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">\\\<line x1="4" y1="4" x2="16" y2="16"/>\\\<line x1="16" y1="4" x2="4" y2="16"/>\\\</svg>';

        del.onclick = (e) => { e.stopPropagation(); deleteChat(c.id); };

        item.appendChild(title);

        item.appendChild(del);

        list.appendChild(item);

    });

}

function newChat() {

    const id = crypto.randomUUID();

    chats.unshift({ id, title: "Новый чат", updatedAt: Date.now() });

    saveChats(chats);

    switchChat(id);

    closeSidebar();

}

async function switchChat(id) {

    currentChatId = id;

    localStorage.setItem(CURRENT\\\_KEY, id);

    renderSidebar();

    await loadChatMessages(id);

}

function deleteChat(id) {

    if (!confirm("Удалить этот чат без возможности восстановления?")) { return; }

    chats = chats.filter(c => c.id !== id);

    localStorage.removeItem(msgsKey(id));

    saveChats(chats);

    fetch("/api/chat/history/" + id, { method: "DELETE" }).catch(() => {});

    if (!chats.length) {

        const newId = crypto.randomUUID();

        chats = [{ id: newId, title: "Новый чат", updatedAt: Date.now() }];

        saveChats(chats);

        currentChatId = newId;

    } else if (currentChatId === id) {

        currentChatId = chats[0].id;

    }

    localStorage.setItem(CURRENT\\\_KEY, currentChatId);

    renderSidebar();

    loadChatMessages(currentChatId);

}

async function loadChatMessages(id) {

    const box = document.getElementById("messages");

    box.innerHTML = "";

    let msgs = [];

    try {

        const r = await fetch("/api/chat/history/" + id);

        if (r.ok) {

            const data = await r.json();

            if (data.messages && data.messages.length) {

                msgs = data.messages.map(m => ({ role: m.role, content: m.content }));

            }

        }

    } catch {}

    if (!msgs.length) { msgs = loadLocalMsgs(id); }

    if (!msgs.length) {

        addMessage("ai", GREETING, false);

        return;

    }

    msgs.forEach(m => addMessage(m.role === "user" ? "user" : "ai", m.content, false));

}

// ============================================================

// MESSAGES UI

// ============================================================

function addMessage(role, text, persist = true) {

    const box = document.getElementById("messages");

    const wrapper = document.createElement("div");

    wrapper.className = "message " + (role === "user" ? "user" : "ai");

    const bubble = document.createElement("div");

    bubble.className = "bubble";

    bubble.textContent = text;

    wrapper.appendChild(bubble);

    box.appendChild(wrapper);

    document.getElementById("chatArea").scrollTop = document.getElementById("chatArea").scrollHeight;

    if (persist) {

        const msgs = loadLocalMsgs(currentChatId);

        msgs.push({ role: role === "user" ? "user" : "assistant", content: text });

        saveLocalMsgs(currentChatId, msgs);

        const chat = chats.find(c => c.id === currentChatId);

        if (chat) {

            chat.updatedAt = Date.now();

            if (role === "user" && (chat.title === "Новый чат" || !chat.title)) {

                chat.title = text.length > 32 ? text.slice(0, 32) + "…" : text;

            }

            saveChats(chats);

            renderSidebar();

        }

    }

    return wrapper;

}

function addTyping() {

    const box = document.getElementById("messages");

    const wrapper = document.createElement("div");

    wrapper.className = "message ai";

    wrapper.id = "typingIndicator";

    const bubble = document.createElement("div");

    bubble.className = "bubble";

    bubble.innerHTML = '\\\<div class="typing-dots">\\\<span>\\\</span>\\\<span>\\\</span>\\\<span>\\\</span>\\\</div>';

    wrapper.appendChild(bubble);

    box.appendChild(wrapper);

    document.getElementById("chatArea").scrollTop = document.getElementById("chatArea").scrollHeight;

}

function removeTyping() {

    const el = document.getElementById("typingIndicator");

    if (el) { el.remove(); }

}

// ============================================================

// SENDING

// ============================================================

async function sendMessage() {

    const input = document.getElementById("messageInput");

    const button = document.getElementById("sendButton");

    const message = input.value.trim();

    if (!message) { return; }

    if (message.length > 5000) { alert("Сообщение слишком длинное."); return; }

    addMessage("user", message);

    input.value = "";

    input.style.height = "auto";

    button.disabled = true;

    addTyping();

    const history = loadLocalMsgs(currentChatId).slice(-20).map(m => ({ role: m.role, content: m.content }));

    try {

        const response = await fetch("/api/chat", {

            method: "POST",

            headers: { "Content-Type": "application/json" },

            body: JSON.stringify({ session\\\_id: currentChatId, message, client\\\_id: clientId, history })

        });

        let data;

        try { data = await response.json(); } catch { data = { detail: "Сервер вернул некорректный ответ." }; }

        removeTyping();

        if (response.status === 402) {

            const detail = data.detail || {};

            addMessage("ai", detail.message || "Бесплатный запрос уже использован. Пополни баланс, чтобы продолжить.");

            setCredits(0);

            openPricing();

        } else if (!response.ok) {

            addMessage("ai", (data.detail && data.detail.message) || data.detail || "Ошибка сервера.");

        } else {

            addMessage("ai", data.answer || "Сервер не вернул ответ.");

            if (typeof data.credits\\\_left === "number") { setCredits(data.credits\\\_left); }

        }

    } catch (error) {

        console.error(error);

        removeTyping();

        addMessage("ai", "Ошибка соединения с сервером.");

    }

    button.disabled = false;

}

document.getElementById("messageInput").addEventListener("keydown", function(event) {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        sendMessage();

    }

});

document.getElementById("messageInput").addEventListener("input", function() {

    this.style.height = "auto";

    this.style.height = Math.min(this.scrollHeight, 160) + "px";

});

// ============================================================

// CREDITS

// ============================================================

let lastCreditsValue = null;

function setCredits(n) {

    const sideEl = document.getElementById("creditsBadgeSide");

    const topEl = document.getElementById("creditsBadgeTop");

    sideEl.textContent = n;

    topEl.textContent = "Баланс: " + n;

    if (lastCreditsValue !== null && lastCreditsValue !== n) {

        [sideEl, topEl].forEach(el => {

            el.classList.remove("pulse");

            void el.offsetWidth;

            el.classList.add("pulse");

        });

    }

    lastCreditsValue = n;

}

async function refreshCredits() {

    try {

        const r = await fetch("/api/credits?client\\\_id=" + encodeURIComponent(clientId));

        if (r.ok) {

            const data = await r.json();

            setCredits(data.credits);

        }

    } catch {}

}

// ============================================================

// MODALS: consent / docs / pricing / promo

// ============================================================

function openModal(id) { document.getElementById(id).classList.add("show"); }

function closeModal(id) { document.getElementById(id).classList.remove("show"); }

function acceptConsent() {

    localStorage.setItem(CONSENT\\\_KEY, "1");

    closeModal("docsModal");

}

function openDocs() { openModal("docsModal"); }

async function openPricing() {

    document.getElementById("clientIdShow").textContent = clientId;

    openModal("pricingModal");

    const grid = document.getElementById("planGrid");

    try {

        const r = await fetch("/api/pricing");

        const data = await r.json();

        grid.innerHTML = "";

        (data.plans || []).forEach(plan => {

            const card = document.createElement("div");

            card.className = "plan-card" + (plan.highlight ? " hl" : "");

            const pricePerReq = (plan.price / plan.requests).toFixed(2);

            card.innerHTML = \\\`

                \\\<div>

                    \\\<div class="plan-name">${plan.title} · ${plan.requests} запросов\\\</div>

                    \\\<div class="plan-note">${plan.note || ""} · \\\~${pricePerReq}₽/запрос\\\</div>

                \\\</div>

                \\\<div style="display\\\:flex;align-items\\\:center;gap:10px;">

                    \\\<div class="plan-price">${plan.price}₽\\\</div>

                    \\\<button class="buy-btn">Купить\\\</button>

                \\\</div>

            \\\`;

            card.querySelector(".buy-btn").onclick = () => buyPlan(plan, data.support);

            grid.appendChild(card);

        });

    } catch {

        grid.innerHTML = "Не удалось загрузить тарифы. Попробуй позже.";

    }

}

function buyPlan(plan, supportUrl) {

    const text = encodeURIComponent(

        \\\`Хочу купить тариф "${plan.title}" (${plan.requests} запросов за ${plan.price}₽). Мой ID: ${clientId}\\\`

    );

    window\\.open((supportUrl || "[[https://t.me/lovnff\](https://t.me/lovnff)](https://t.me/lovnff]\(https://t.me/lovnff\))") + "?text=" + text, "\\\_blank");

}

function openPromo() {

    document.getElementById("promoStatus").textContent = "";

    document.getElementById("promoInput").value = "";

    openModal("promoModal");

    setTimeout(() => document.getElementById("promoInput").focus(), 150);

}

async function applyPromo() {

    const input = document.getElementById("promoInput");

    const status = document.getElementById("promoStatus");

    const code = input.value.trim();

    if (!code) { status.textContent = "Введи промокод."; return; }

    status.textContent = "Проверяю...";

    try {

        const response = await fetch("/api/promo/redeem", {

            method: "POST",

            headers: { "Content-Type": "application/json" },

            body: JSON.stringify({ client\\\_id: clientId, code })

        });

        let data;

        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }

        if (!response.ok) {

            status.textContent = (data.detail && data.detail.message) || data.detail || "Промокод недействителен.";

            return;

        }

        status.textContent = \\\`Готово! Начислено ${data.credits\\\_added} запросов.\\\`;

        setCredits(data.credits);

        input.value = "";

    } catch (error) {

        console.error(error);

        status.textContent = "Ошибка соединения с сервером.";

    }

}

document.getElementById("promoInput") && document.getElementById("promoInput").addEventListener("keydown", function(event) {

    if (event.key === "Enter") {

        event.preventDefault();

        applyPromo();

    }

});

// ============================================================

// ADMIN (доступ только по ссылке вида /#admin — кнопки в интерфейсе

// намеренно нет, чтобы не привлекать внимание к панели управления)

// ============================================================

function checkAdminRoute() {

    if (location.hash.replace("#", "") === "admin") {

        document.getElementById("chatSection").style.display = "none";

        document.getElementById("adminSection").style.display = "block";

        if (adminToken) {

            document.getElementById("adminPanel").classList.remove("hidden");

            loadAdminData();

        }

    } else {

        document.getElementById("chatSection").style.display = "flex";

        document.getElementById("adminSection").style.display = "none";

    }

}

window\\.addEventListener("hashchange", checkAdminRoute);

async function loginAdmin() {

    const password = document.getElementById("adminPassword").value;

    const status = document.getElementById("loginStatus");

    status.textContent = "Проверка...";

    try {

        const response = await fetch("/api/admin/login", {

            method: "POST",

            headers: { "Content-Type": "application/json" },

            body: JSON.stringify({ password })

        });

        let data;

        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }

        if (!response.ok) { status.textContent = data.detail || "Неверный пароль."; return; }

        adminToken = data.token;

        localStorage.setItem("ascend\\\_admin\\\_token", adminToken);

        document.getElementById("adminPanel").classList.remove("hidden");

        status.textContent = "Авторизация успешна.";

        loadAdminData();

    } catch {

        status.textContent = "Ошибка соединения.";

    }

}

function adminHeaders() {

    return { "Content-Type": "application/json", "X-Admin-Token": adminToken };

}

async function topUpCredits() {

    const status = document.getElementById("topupStatus");

    const targetClientId = document.getElementById("topupClientId").value.trim();

    const amount = parseInt(document.getElementById("topupAmount").value, 10);

    if (!targetClientId || !amount) { status.textContent = "Укажи client\\\_id и количество."; return; }

    status.textContent = "Начисляю...";

    try {

        const response = await fetch("/api/admin/credits", {

            method: "POST",

            headers: adminHeaders(),

            body: JSON.stringify({ client\\\_id: targetClientId, amount })

        });

        let data;

        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }

        if (!response.ok) { status.textContent = data.detail || "Ошибка."; return; }

        status.textContent = "Начислено. Новый баланс: " + data.credits;

    } catch (error) {

        console.error(error);

        status.textContent = "Ошибка соединения с сервером.";

    }

}

async function createPromo() {

    const status = document.getElementById("promoCreateStatus");

    const code = document.getElementById("promoCode").value.trim();

    const credits = parseInt(document.getElementById("promoCreditsAmount").value, 10);

    if (!code || !credits) { status.textContent = "Укажи код и количество запросов."; return; }

    status.textContent = "Создаю...";

    try {

        const response = await fetch("/api/admin/promo", {

            method: "POST",

            headers: adminHeaders(),

            body: JSON.stringify({ code, credits })

        });

        let data;

        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }

        if (!response.ok) { status.textContent = data.detail || "Ошибка."; return; }

        status.textContent = \\\`Промокод "${data.code}" создан (+${data.credits} запросов).\\\`;

        document.getElementById("promoCode").value = "";

        document.getElementById("promoCreditsAmount").value = "";

    } catch (error) {

        console.error(error);

        status.textContent = "Ошибка соединения с сервером.";

    }

}

async function addKnowledge() {

    const title = document.getElementById("title").value.trim();

    const category = document.getElementById("category").value.trim();

    const question = document.getElementById("question").value.trim();

    const answer = document.getElementById("answer").value.trim();

    const tags = document.getElementById("tags").value.split(",").map(x => x.trim()).filter(Boolean);

    const status = document.getElementById("trainStatus");

    status.textContent = "Обучаю нейросеть...";

    try {

        const response = await fetch("/api/admin/knowledge", {

            method: "POST",

            headers: adminHeaders(),

            body: JSON.stringify({ title, category, question, answer, tags })

        });

        let data;

        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }

        if (!response.ok) { status.textContent = data.detail || "Ошибка."; return; }

        status.textContent = "Знание добавлено. Нейросеть переобучена.";

        document.getElementById("title").value = "";

        document.getElementById("category").value = "";

        document.getElementById("question").value = "";

        document.getElementById("answer").value = "";

        document.getElementById("tags").value = "";

        loadAdminData();

    } catch (error) {

        console.error(error);

        status.textContent = "Ошибка соединения с сервером.";

    }

}

async function loadAdminData() {

    if (!adminToken) { return; }

    try {

        const statsResponse = await fetch("/api/admin/stats", { headers: adminHeaders() });

        if (statsResponse.ok) {

            const stats = await statsResponse.json();

            document.getElementById("brainStats").innerHTML = \\\`

                \\\<p>Модель: \\\<strong>${stats.brain\\\_ready ? "готова" : "не готова"}\\\</strong>\\\</p>

                \\\<p>Знаний: \\\<strong>${stats.knowledge}\\\</strong>\\\</p>

                \\\<p>Словарь: \\\<strong>${stats.vocabulary}\\\</strong>\\\</p>

                \\\<p>Категорий: \\\<strong>${stats.categories}\\\</strong>\\\</p>

            \\\`;

        }

        await loadSettingsStatus();

        const knowledgeResponse = await fetch("/api/admin/knowledge", { headers: adminHeaders() });

        if (!knowledgeResponse.ok) { return; }

        const knowledge = await knowledgeResponse.json();

        const list = document.getElementById("knowledgeList");

        list.innerHTML = "";

        knowledge.forEach(item => {

            const element = document.createElement("div");

            element.className = "knowledge-item";

            element.innerHTML = \\\`

                \\\<span class="badge">${escapeHtml(item.category || "")}\\\</span>

                \\\<h3>${escapeHtml(item.title || "")}\\\</h3>

                \\\<p>\\\<strong>Вопрос:\\\</strong>\\\<br>${escapeHtml(item.question || "")}\\\</p>

                \\\<p>${escapeHtml(item.answer || "")}\\\</p>

            \\\`;

            list.appendChild(element);

        });

        const webResponse = await fetch("/api/admin/web-sources", { headers: adminHeaders() });

        if (webResponse.ok) {

            const webData = await webResponse.json();

            const webList = document.getElementById("webSearchList");

            webList.innerHTML = "";

            webData.forEach(item => {

                const element = document.createElement("div");

                element.className = "knowledge-item";

                element.innerHTML = \\\`

                    \\\<span class="badge">${escapeHtml(item.source || "web")}\\\</span>

                    \\\<h3>${escapeHtml(item.title || "")}\\\</h3>

                    \\\<p>\\\<strong>Запрос:\\\</strong> ${escapeHtml(item.query || "")}\\\</p>

                    \\\<a href="${escapeHtml(item.url || "#")}" target="\\\_blank" rel="noopener noreferrer">Открыть источник\\\</a>

                \\\`;

                webList.appendChild(element);

            });

        }

    } catch (error) {

        console.error(error);

    }

}

async function loadSettingsStatus() {

    try {

        const response = await fetch("/api/admin/settings", { headers: adminHeaders() });

        if (!response.ok) { return; }

        const data = await response.json();

        const status = document.getElementById("llmStatus");

        const line = (label, set, masked) =>

            \\\`${label}: ${set ? "задан (" + escapeHtml(masked) + ")" : "— не задан"}\\\`;

        status.innerHTML = [

            line("provod.ai", data.provod\\\_api\\\_key\\\_set, data.provod\\\_api\\\_key\\\_masked),

            \\\`provod.ai модель: ${escapeHtml(data.provod\\\_model || "")}\\\`,

            line("OpenRouter", data.openrouter\\\_api\\\_key\\\_set, data.openrouter\\\_api\\\_key\\\_masked),

            line("DeepSeek", data.deepseek\\\_api\\\_key\\\_set, data.deepseek\\\_api\\\_key\\\_masked),

            line("Qwen", data.qwen\\\_api\\\_key\\\_set, data.qwen\\\_api\\\_key\\\_masked),

            \\\`Прямой LLM режим: ${data.llm\\\_direct\\\_mode ? "включён" : "выключен"}\\\`,

            \\\`LLM-ответы: ${data.llm\\\_enabled ? "включены" : "выключены (экстрактивный режим)"}\\\`

        ].join("\\\<br>");

        if (data.provod\\\_model) {

            document.getElementById("provodModel").placeholder = data.provod\\\_model;

        }

        document.getElementById("llmDirectMode").checked = !!data.llm\\\_direct\\\_mode;

    } catch (error) {

        console.error(error);

    }

}

async function saveSettings() {

    const status = document.getElementById("settingsStatus");

    status.textContent = "Сохраняю...";

    const body = {};

    const openrouterKey = document.getElementById("openrouterKey").value.trim();

    const deepseekKey = document.getElementById("deepseekKey").value.trim();

    const qwenKey = document.getElementById("qwenKey").value.trim();

    const provodKey = document.getElementById("provodKey").value.trim();

    const provodModel = document.getElementById("provodModel").value.trim();

    if (openrouterKey) { body.openrouter\\\_api\\\_key = openrouterKey; }

    if (deepseekKey) { body.deepseek\\\_api\\\_key = deepseekKey; }

    if (qwenKey) { body.qwen\\\_api\\\_key = qwenKey; }

    if (provodKey) { body.provod\\\_api\\\_key = provodKey; }

    if (provodModel) { body.provod\\\_model = provodModel; }

    body.llm\\\_direct\\\_mode = document.getElementById("llmDirectMode").checked;

    try {

        const response = await fetch("/api/admin/settings", {

            method: "POST",

            headers: adminHeaders(),

            body: JSON.stringify(body)

        });

        let data;

        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }

        if (!response.ok) { status.textContent = data.detail || "Ошибка."; return; }

        status.textContent = "Сохранено.";

        document.getElementById("openrouterKey").value = "";

        document.getElementById("deepseekKey").value = "";

        document.getElementById("qwenKey").value = "";

        document.getElementById("provodKey").value = "";

        document.getElementById("provodModel").value = "";

        loadSettingsStatus();

    } catch (error) {

        console.error(error);

        status.textContent = "Ошибка соединения с сервером.";

    }

}

function escapeHtml(value) {

    return String(value ?? "")

        .replaceAll("&", "&amp;")

        .replaceAll("<", "&lt;")

        .replaceAll(">", "&gt;")

        .replaceAll('"', "&quot;")

        .replaceAll("'", "&#039;");

}

// ============================================================

// BOOTSTRAP

// ============================================================

renderSidebar();

loadChatMessages(currentChatId);

refreshCredits();

checkAdminRoute();

if (!localStorage.getItem(CONSENT\\\_KEY)) { openModal("docsModal"); }

\\\</script>

\\\</body>

\\\</html>

"""

\\# ============================================================

\\# ROOT

\\# ============================================================

@app.get("/", response\\\_class=HTMLResponse)

async def index():

    return HTML

\\# ============================================================

\\# HEALTH

\\# ============================================================

@app.get("/health")

async def health():

    return {

        "status": "ok",

        "brain\\\_ready": brain.ready,

        "knowledge": len(knowledge\\\_cache),

        "search\\\_engines": [name for name, \\\_ in SEARCH\\\_ENGINES],

        "llm\\\_enabled": llm\\\_available(),

    }

\\# ============================================================

\\# CHAT

\\# ============================================================

GREETING\\\_WORDS = {

    "привет", "здравствуй", "здравствуйте", "приветик", "хай",

    "хеллоу", "хелло", "йо", "ку", "здарова", "здорово",

}

FAREWELL\\\_WORDS = {"пока", "прощай", "досвидания", "бывай", "увидимся"}

THANKS\\\_WORDS = {"спасибо", "благодарю", "спс", "сенкс", "thanks"}

HOWAREYOU\\\_PHRASES = {

    "как дела", "как ты", "как жизнь", "как оно", "че как", "что нового",

}

GREETING\\\_REPLIES = [

    "Привет! Расскажи, что тебя интересует — кожа, внешность, питание, "

    "сон или тренировки?",

    "Привет 👋 С чем помочь сегодня?",

]

FAREWELL\\\_REPLIES = ["Пока! Возвращайся, если появятся вопросы 🙂"]

THANKS\\\_REPLIES = ["Пожалуйста! Обращайся, если будут ещё вопросы."]

HOWAREYOU\\\_REPLIES = [

    "Спасибо, у меня всё в порядке! А у тебя как дела? "

    "И расскажи, чем могу помочь.",

]

def detect\\\_small\\\_talk(message):

    normalized = normalize(message)

    for phrase in HOWAREYOU\\\_PHRASES:

        if phrase in normalized:

            return random.choice(HOWAREYOU\\\_REPLIES)

    words = normalized.split()

    if not words or len(words) > 4:

        return None

    word\\\_set = set(words)

    if word\\\_set & GREETING\\\_WORDS and word\\\_set <= (GREETING\\\_WORDS | {"как", "дела", "там"}):

        return random.choice(GREETING\\\_REPLIES)

    if word\\\_set & FAREWELL\\\_WORDS and word\\\_set <= FAREWELL\\\_WORDS:

        return random.choice(FAREWELL\\\_REPLIES)

    if word\\\_set & THANKS\\\_WORDS and word\\\_set <= THANKS\\\_WORDS:

        return random.choice(THANKS\\\_REPLIES)

    return None

@app.post("/api/chat")

async def chat(data: ChatRequest, request: Request):

    message = data.message.strip()

    if not message:

        raise HTTPException(400, "Пустой запрос.")

    if len(message) > MAX\\\_MESSAGE\\\_LENGTH:

        raise HTTPException(400, "Сообщение слишком длинное.")

    ip\\\_hash = hash\\\_ip(request)

    credit\\\_record = get\\\_or\\\_create\\\_credit\\\_record(ip\\\_hash, data.client\\\_id or "")

    if credit\\\_record["credits"] <= 0:

        raise HTTPException(

            status\\\_code=402,

            detail={

                "code": "NO\\\_CREDITS",

                "message": "Бесплатный запрос уже использован. Пополни баланс, чтобы продолжить общение.",

                "plans": PRICING\\\_PLANS,

                "support": SUPPORT\\\_TELEGRAM,

            },

        )

    print("")

    print("=" \\\* 60)

    print("NEW CHAT REQUEST:", message)

    print("=" \\\* 60)

    memory = get\\\_memory(data.session\\\_id)

    if not memory and data.history:

        memory = [{"role": item.role, "content": item.content} for item in data.history]

    save\\\_message(data.session\\\_id, "user", message)

    small\\\_talk\\\_answer = detect\\\_small\\\_talk(message)

    if small\\\_talk\\\_answer:

        print("SMALL TALK DETECTED — skipping search/LLM")

        assistant\\\_message = save\\\_message(data.session\\\_id, "assistant", small\\\_talk\\\_answer)

        consume\\\_credit(ip\\\_hash)

        print("=" \\\* 60)

        return {

            "answer": small\\\_talk\\\_answer,

            "knowledge\\\_found": False,

            "web\\\_found": False,

            "memory\\\_used": len(memory),

            "message\\\_id": assistant\\\_message.get("id") if assistant\\\_message else None,

            "credits\\\_left": credits\\\_cache.get(ip\\\_hash, {}).get("credits", 0),

        }

    local\\\_results = search\\\_local\\\_knowledge(message)

    direct\\\_mode = is\\\_llm\\\_direct\\\_mode() and llm\\\_available()

    if direct\\\_mode:

        print("LLM DIRECT MODE: skipping web search")

        web\\\_results = []

    else:

        web\\\_results = collect\\\_web\\\_information(message)

        save\\\_web\\\_sources(data.session\\\_id, message, web\\\_results)

    answer = generate\\\_response(message, memory, local\\\_results, web\\\_results)

    assistant\\\_message = save\\\_message(data.session\\\_id, "assistant", answer)

    consume\\\_credit(ip\\\_hash)

    category = None

    if local\\\_results:

        category = local\\\_results[0][1].get("category")

    save\\\_training\\\_log(message, answer, category or "web", "search")

    print("FINAL WEB RESULTS USED:", len(web\\\_results))

    print("=" \\\* 60)

    return {

        "answer": answer,

        "knowledge\\\_found": bool(local\\\_results),

        "web\\\_found": bool(web\\\_results),

        "memory\\\_used": len(memory),

        "message\\\_id": assistant\\\_message.get("id") if assistant\\\_message else None,

        "credits\\\_left": credits\\\_cache.get(ip\\\_hash, {}).get("credits", 0),

    }

\\# ============================================================

\\# CHAT HISTORY

\\# ============================================================

@app.get("/api/chat/history/{session\\\_id}")

async def chat\\\_history(session\\\_id: str):

    return {"messages": get\\\_memory(session\\\_id)}

@app.delete("/api/chat/history/{session\\\_id}")

async def delete\\\_chat\\\_history(session\\\_id: str):

    if SUPABASE\\\_URL and SUPABASE\\\_SECRET\\\_KEY:

        supabase\\\_request("DELETE", "chat\\\_messages", params={"session\\\_id": f"eq.{session\\\_id}"})

        supabase\\\_request("DELETE", "web\\\_sources", params={"session\\\_id": f"eq.{session\\\_id}"})

    return {"success": True}

\\# ============================================================

\\# CREDITS / ТАРИФЫ

\\# ============================================================

@app.get("/api/credits")

async def get\\\_credits(request: Request, client\\\_id: str = ""):

    ip\\\_hash = hash\\\_ip(request)

    record = get\\\_or\\\_create\\\_credit\\\_record(ip\\\_hash, client\\\_id)

    return {"credits": record["credits"], "client\\\_id": record.get("client\\\_id", "")}

@app.get("/api/pricing")

async def get\\\_pricing():

    return {

        "plans": PRICING\\\_PLANS,

        "support": SUPPORT\\\_TELEGRAM,

        "privacy\\\_url": PRIVACY\\\_POLICY\\\_URL,

        "terms\\\_url": TERMS\\\_OF\\\_USE\\\_URL,

    }

@app.post("/api/admin/credits")

async def admin\\\_add\\\_credits(data: CreditsTopUp, request: Request):

    check\\\_admin(request)

    ip\\\_hash = client\\\_id\\\_index.get(data.client\\\_id)

    if not ip\\\_hash and SUPABASE\\\_URL and SUPABASE\\\_SECRET\\\_KEY:

        rows = supabase\\\_request(

            "GET", "user\\\_credits",

            params={"select": "ip\\\_hash", "client\\\_id": f"eq.{data.client\\\_id}", "limit": "1"},

        )

        if rows:

            ip\\\_hash = rows[0].get("ip\\\_hash")

    if not ip\\\_hash:

        raise HTTPException(404, "Пользователь с таким client\\\_id не найден.")

    record = load\\\_credit\\\_record(ip\\\_hash) or {"credits": 0, "client\\\_id": data.client\\\_id}

    record["credits"] = max(0, record["credits"] + data.amount)

    persist\\\_credit\\\_record(ip\\\_hash, record)

    return {"success": True, "credits": record["credits"]}

\\# ============================================================

\\# ПРОМОКОДЫ — ПУБЛИЧНЫЙ ЭНДПОИНТ АКТИВАЦИИ

\\# ============================================================

@app.post("/api/promo/redeem")

async def redeem\\\_promo(data: PromoRedeem, request: Request):

    code = data.code.strip().upper()

    if not code:

        raise HTTPException(400, "Введите промокод.")

    promo = get\\\_promo\\\_code(code)

    if not promo:

        raise HTTPException(

            status\\\_code=404,

            detail={"code": "PROMO\\\_NOT\\\_FOUND", "message": "Промокод не найден или недействителен."},

        )

    ip\\\_hash = hash\\\_ip(request)

    if has\\\_redeemed\\\_promo(ip\\\_hash, code):

        raise HTTPException(

            status\\\_code=409,

            detail={"code": "PROMO\\\_ALREADY\\\_USED", "message": "Этот промокод уже был активирован."},

        )

    record = get\\\_or\\\_create\\\_credit\\\_record(ip\\\_hash, data.client\\\_id or "")

    record["credits"] = record.get("credits", 0) + promo["credits"]

    persist\\\_credit\\\_record(ip\\\_hash, record)

    mark\\\_promo\\\_redeemed(ip\\\_hash, code)

    print("PROMO REDEEMED:", code, "credits\\\_added=", promo["credits"], "ip\\\_hash=", ip\\\_hash)

    return {"success": True, "credits\\\_added": promo["credits"], "credits": record["credits"]}

\\# ============================================================

\\# ПРОМОКОДЫ — АДМИНСКОЕ СОЗДАНИЕ

\\# ============================================================

@app.post("/api/admin/promo")

async def admin\\\_create\\\_promo(data: PromoCreate, request: Request):

    check\\\_admin(request)

    code = data.code.strip().upper()

    if not code:

        raise HTTPException(400, "Код обязателен.")

    if data.credits <= 0:

        raise HTTPException(400, "Количество запросов должно быть больше нуля.")

    promo\\\_codes\\\_cache[code] = {"credits": data.credits}

    if SUPABASE\\\_URL and SUPABASE\\\_SECRET\\\_KEY:

        supabase\\\_request(

            "POST", "promo\\\_codes",

            {"code": code, "credits": data.credits},

            params={"on\\\_conflict": "code"},

        )

    return {"success": True, "code": code, "credits": data.credits}

\\# ============================================================

\\# ADMIN LOGIN

\\# ============================================================

@app.post("/api/admin/login")

async def admin\\\_login(data: AdminLogin):

    print("ADMIN LOGIN ATTEMPT", flush=True)

    try:

        password\\\_ok = secrets.compare\\\_digest(

            data.password.encode("utf-8"),

            ADMIN\\\_PASSWORD.encode("utf-8"),

        )

    except Exception as e:

        print("ADMIN LOGIN compare\\\_digest ERROR:", repr(e), flush=True)

        raise HTTPException(500, f"Ошибка проверки пароля: {e}")

    if not password\\\_ok:

        print("ADMIN LOGIN: wrong password", flush=True)

        raise HTTPException(401, "Неверный пароль.")

    try:

        token = create\\\_admin\\\_token()

    except Exception as e:

        print("ADMIN LOGIN create\\\_admin\\\_token ERROR:", repr(e), flush=True)

        raise HTTPException(500, f"Ошибка создания токена: {e}")

    print("ADMIN LOGIN: success", flush=True)

    return {"success": True, "token": token}

\\# ============================================================

\\# ADMIN STATS

\\# ============================================================

@app.get("/api/admin/stats")

async def admin\\\_stats(request: Request):

    check\\\_admin(request)

    return {

        "brain\\\_ready": brain.ready,

        "knowledge": len(knowledge\\\_cache),

        "vocabulary": len(brain.vocabulary),

        "categories": len(brain.categories),

    }

\\# ============================================================

\\# ADMIN KNOWLEDGE GET

\\# ============================================================


@app.get("/api/admin/knowledge")

async def admin\_knowledge(request: Request):

    check\_admin(request)

    return knowledge\_cache

\# ============================================================

\# ADMIN KNOWLEDGE CREATE

\# ============================================================

@app.post("/api/admin/knowledge")

async def admin\_add\_knowledge(request: Request, data: KnowledgeCreate):

    check\_admin(request)

    title = data.title.strip()

    category = normalize(data.category)

    question = data.question.strip()

    answer = data.answer.strip()

    tags = [x.strip() for x in data.tags if x.strip()]

    if not title:

        raise HTTPException(400, "Название обязательно.")

    if not category:

        raise HTTPException(400, "Категория обязательна.")

    if not question:

        raise HTTPException(400, "Вопрос обязателен.")

    if not answer:

        raise HTTPException(400, "Ответ обязателен.")

    item = {

        "title": title,

        "category": category,

        "question": question,

        "answer": answer,

        "tags": tags,

        "approved": True,

    }

    saved = []

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:

        saved = supabase\_request("POST", "knowledge", item)

    if saved:

        knowledge\_cache.append(saved[0])

    else:

        item["id"] = stable\_hash(title + question + answer)

        knowledge\_cache.append(item)

    result = brain.train(knowledge\_cache)

    save\_training\_log(question, answer, category, "admin")

    return {"success": True, "training": result, "knowledge": len(knowledge\_cache)}

\# ============================================================

\# ADMIN DELETE KNOWLEDGE

\# ============================================================

@app.delete("/api/admin/knowledge/{knowledge\_id}")

async def admin\_delete\_knowledge(knowledge\_id: str, request: Request):

    check\_admin(request)

    global knowledge\_cache

    knowledge\_cache = [

        item for item in knowledge\_cache if str(item.get("id")) != str(knowledge\_id)

    ]

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:

        supabase\_request("DELETE", "knowledge", params={"id": f"eq.{knowledge\_id}"})

    brain.train(knowledge\_cache)

    return {"success": True}

\# ============================================================

\# ADMIN WEB SOURCES

\# ============================================================

@app.get("/api/admin/web-sources")

async def admin\_web\_sources(request: Request):

    check\_admin(request)

    if not SUPABASE\_URL or not SUPABASE\_SECRET\_KEY:

        return []

    rows = supabase\_request(

        "GET",

        "web\_sources",

        params={

            "select": "id,query,title,url,snippet,source,created\_at",

            "order": "created\_at.desc",

            "limit": "50",

        },

    )

    return rows

\# ============================================================

\# ADMIN SETTINGS

\# ============================================================

@app.get("/api/admin/settings")

async def admin\_get\_settings(request: Request):

    check\_admin(request)

    return {

        "openrouter\_api\_key\_set": bool(get\_setting("openrouter\_api\_key")),

        "openrouter\_api\_key\_masked": mask\_key(get\_setting("openrouter\_api\_key")),

        "deepseek\_api\_key\_set": bool(get\_setting("deepseek\_api\_key")),

        "deepseek\_api\_key\_masked": mask\_key(get\_setting("deepseek\_api\_key")),

        "qwen\_api\_key\_set": bool(get\_setting("qwen\_api\_key")),

        "qwen\_api\_key\_masked": mask\_key(get\_setting("qwen\_api\_key")),

        "provod\_api\_key\_set": bool(get\_setting("provod\_api\_key")),

        "provod\_api\_key\_masked": mask\_key(get\_setting("provod\_api\_key")),

        "provod\_model": get\_setting("provod\_model") or PROVOD\_DEFAULT\_MODEL,

        "llm\_direct\_mode": is\_llm\_direct\_mode(),

        "llm\_enabled": llm\_available(),

    }

@app.post("/api/admin/settings")

async def admin\_update\_settings(request: Request, data: SettingsUpdate):

    check\_admin(request)

    updated = []

    if data.openrouter\_api\_key is not None:

        save\_setting("openrouter\_api\_key", data.openrouter\_api\_key.strip())

        updated.append("openrouter\_api\_key")

    if data.deepseek\_api\_key is not None:

        save\_setting("deepseek\_api\_key", data.deepseek\_api\_key.strip())

        updated.append("deepseek\_api\_key")

    if data.qwen\_api\_key is not None:

        save\_setting("qwen\_api\_key", data.qwen\_api\_key.strip())

        updated.append("qwen\_api\_key")

    if data.provod\_api\_key is not None:

        save\_setting("provod\_api\_key", data.provod\_api\_key.strip())

        updated.append("provod\_api\_key")

    if data.provod\_model is not None and data.provod\_model.strip():

        save\_setting("provod\_model", data.provod\_model.strip())

        updated.append("provod\_model")

    if data.llm\_direct\_mode is not None:

        save\_setting("llm\_direct\_mode", "true" if data.llm\_direct\_mode else "false")

        updated.append("llm\_direct\_mode")

    return {"success": True, "updated": updated, "llm\_enabled": llm\_available()}

\# ============================================================

\# FEEDBACK

\# ============================================================

@app.post("/api/feedback")

async def feedback(data: FeedbackRequest):

    if data.rating < 1 or data.rating > 5:

        raise HTTPException(400, "Оценка должна быть от 1 до 5.")

    if SUPABASE\_URL and SUPABASE\_SECRET\_KEY:

        supabase\_request(

            "POST",

            "ai\_feedback",

            {

                "session\_id": data.session\_id,

                "message\_id": data.message\_id,

                "rating": data.rating,

                "comment": data.comment or "",

            },

        )

    return {"success": True}

\# ============================================================

\# STARTUP

\# ============================================================

@app.on\_event("startup")

async def startup():

    try:

        load\_knowledge()

    except Exception as e:

        print("STARTUP ERROR in load\_knowledge:", repr(e), flush=True)

        traceback.print\_exc()

    try:

        load\_settings()

    except Exception as e:

        print("STARTUP ERROR in load\_settings:", repr(e), flush=True)

        traceback.print\_exc()

    try:

        load\_promo\_codes()

    except Exception as e:

        print("STARTUP ERROR in load\_promo\_codes:", repr(e), flush=True)

        traceback.print\_exc()

    print("", flush=True)

    print("=" \* 60, flush=True)

    print("                  ASCEND AI", flush=True)

    print("=" \* 60, flush=True)

    print("Knowledge:", len(knowledge\_cache), flush=True)

    print("Neural brain:", brain.ready, flush=True)

    print("Search engines:", [name for name, \_ in SEARCH\_ENGINES], flush=True)

    print("SearXNG instances:", len(SEARXNG\_INSTANCES), flush=True)

    print("LLM enabled:", llm\_available(), flush=True)

    print("Promo codes loaded:", len(promo\_codes\_cache), flush=True)

    print("=" \* 60, flush=True)

    print(""
