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


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "ASCEND AI"

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "CHANGE_THIS_PASSWORD")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")


# ============================================================
# LLM SETTINGS (API-ключи, а НЕ логин/пароль от личных аккаунтов)
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
PROVOD_URL = "https://api.provod.ai/v1/chat/completions"

LLM_TIMEOUT = 30

OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

DEEPSEEK_MODEL = "deepseek-chat"
QWEN_MODEL = "qwen-plus"

PROVOD_DEFAULT_MODEL = os.getenv("PROVOD_MODEL", "xiaomi/mimo-v2.5")

_DEFAULT_SETTINGS = {
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "qwen_api_key": os.getenv("QWEN_API_KEY", ""),
    "provod_api_key": os.getenv("PROVOD_API_KEY", ""),
    "provod_model": PROVOD_DEFAULT_MODEL,
    "llm_direct_mode": os.getenv("LLM_DIRECT_MODE", "true"),
}

runtime_settings = dict(_DEFAULT_SETTINGS)


def is_llm_direct_mode():
    return get_setting("llm_direct_mode").strip().lower() in ("1", "true", "yes", "on")


def get_setting(key):
    return runtime_settings.get(key, "") or ""


def mask_key(value):
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "…" + value[-4:]


# ============================================================
# SEARCH ENGINES (multi-provider fallback chain)
# ============================================================

SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://searx.tiekoetter.com",
    "https://searxng.site",
    "https://search.inetol.net",
    "https://priv.au",
    "https://search.bus-hit.me",
    "https://searx.namejeff.xyz",
    "https://baresearch.org",
    "https://opnxng.com",
    "https://search.sapti.me",
]

SEARXNG_TIMEOUT = 10
SEARXNG_MAX_RETRIES_PER_INSTANCE = 2
SEARXNG_RETRY_BACKOFF_BASE = 1.5


# ============================================================
# LIMITS
# ============================================================

MAX_MEMORY = 30
MAX_SEARCH_RESULTS = 6
MAX_SOURCE_TEXT = 3500
MAX_MESSAGE_LENGTH = 5000
PAGE_TIMEOUT = 12


# ============================================================
# CREDITS / АНТИ-АБУЗ ДЛЯ БЕСПЛАТНОГО ЗАПРОСА
# ============================================================

FREE_CREDITS = 1

# Промокоды хранятся только на сервере и нигде не выводятся в интерфейсе.
PROMO_CODES = {
    "AIRENT1000": {"credits": 1000, "single_use": True},
}
PROMO_REDEMPTIONS_KEY = "promo_redemptions"
promo_redemptions_cache = None
promo_redemptions_lock = __import__("threading").Lock()

credits_cache = {}          # ip_hash -> {"credits": int, "client_id": str}
client_id_index = {}        # client_id -> ip_hash (для админского пополнения)

PRICING_PLANS = [
    {"id": "start", "title": "Старт", "requests": 50, "price": 79,
     "note": "Для знакомства с нейросетью"},
    {"id": "plus", "title": "Плюс", "requests": 150, "price": 199,
     "note": "Самый популярный вариант", "highlight": True},
    {"id": "pro", "title": "Про", "requests": 400, "price": 449,
     "note": "Для активного использования"},
    {"id": "max", "title": "Макс", "requests": 1000, "price": 899,
     "note": "Максимальная выгода за запрос"},
]

SUPPORT_TELEGRAM = "https://t.me/lovnff"
PRIVACY_POLICY_URL = "https://telegra.ph/Politika-konfidencialnosti-09-06-116"
TERMS_OF_USE_URL = "https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54"


def hash_ip(request: Request) -> str:
    ip = (request.client.host if request.client else "") or "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    return hashlib.sha256(("ascend-credits:" + ip).encode()).hexdigest()[:32]


def load_credit_record(ip_hash: str):
    if ip_hash in credits_cache:
        return credits_cache[ip_hash]

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        rows = supabase_request(
            "GET", "user_credits",
            params={"select": "ip_hash,credits,client_id", "ip_hash": f"eq.{ip_hash}", "limit": "1"},
        )
        if rows:
            record = {"credits": rows[0].get("credits", 0), "client_id": rows[0].get("client_id", "")}
            credits_cache[ip_hash] = record
            if record["client_id"]:
                client_id_index[record["client_id"]] = ip_hash
            return record

    return None


def persist_credit_record(ip_hash: str, record: dict):
    credits_cache[ip_hash] = record
    if record.get("client_id"):
        client_id_index[record["client_id"]] = ip_hash

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        existing = supabase_request(
            "GET", "user_credits",
            params={"select": "ip_hash", "ip_hash": f"eq.{ip_hash}", "limit": "1"},
        )
        payload = {"ip_hash": ip_hash, "credits": record["credits"], "client_id": record.get("client_id", "")}
        if existing:
            supabase_request("PATCH", "user_credits", payload, params={"ip_hash": f"eq.{ip_hash}"})
        else:
            supabase_request("POST", "user_credits", payload)


def get_or_create_credit_record(ip_hash: str, client_id: str = ""):
    record = load_credit_record(ip_hash)

    if record is None:
        record = {"credits": FREE_CREDITS, "client_id": client_id or ""}
        persist_credit_record(ip_hash, record)
        return record

    if client_id and not record.get("client_id"):
        record["client_id"] = client_id
        persist_credit_record(ip_hash, record)

    return record


def consume_credit(ip_hash: str):
    record = credits_cache.get(ip_hash)
    if record is None:
        return
    record["credits"] = max(0, record["credits"] - 1)
    persist_credit_record(ip_hash, record)


def _load_promo_redemptions():
    global promo_redemptions_cache
    if promo_redemptions_cache is not None:
        return promo_redemptions_cache

    promo_redemptions_cache = {}
    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        rows = supabase_request(
            "GET", "app_settings",
            params={"select": "value", "key": f"eq.{PROMO_REDEMPTIONS_KEY}", "limit": "1"},
        )
        if rows:
            try:
                value = rows[0].get("value", "{}")
                parsed = json.loads(value) if isinstance(value, str) else value
                if isinstance(parsed, dict):
                    promo_redemptions_cache = parsed
            except Exception:
                promo_redemptions_cache = {}
    return promo_redemptions_cache


def _save_promo_redemptions():
    if not (SUPABASE_URL and SUPABASE_SECRET_KEY):
        return

    value = json.dumps(promo_redemptions_cache or {}, ensure_ascii=False)
    existing = supabase_request(
        "GET", "app_settings",
        params={"select": "key", "key": f"eq.{PROMO_REDEMPTIONS_KEY}", "limit": "1"},
    )
    if existing:
        supabase_request(
            "PATCH", "app_settings", {"value": value},
            params={"key": f"eq.{PROMO_REDEMPTIONS_KEY}"},
        )
    else:
        supabase_request("POST", "app_settings", {
            "key": PROMO_REDEMPTIONS_KEY,
            "value": value,
        })


def redeem_promo_code(client_id: str, code: str):
    normalized_code = re.sub(r"\s+", "", (code or "").upper())
    client_id = (client_id or "").strip()

    if not client_id:
        raise HTTPException(400, "Не удалось определить ID пользователя.")
    if not normalized_code:
        raise HTTPException(400, "Введи промокод.")

    promo = PROMO_CODES.get(normalized_code)
    if not promo:
        raise HTTPException(400, "Промокод недействителен.")

    with promo_redemptions_lock:
        redemptions = _load_promo_redemptions()
        users = redemptions.setdefault(normalized_code, [])
        if promo.get("single_use", True) and client_id in users:
            raise HTTPException(409, "Этот промокод уже был активирован.")

        ip_hash = client_id_index.get(client_id)
        if not ip_hash:
            for cached_ip, record in credits_cache.items():
                if record.get("client_id") == client_id:
                    ip_hash = cached_ip
                    break

        if not ip_hash:
            raise HTTPException(
                404,
                "Пользователь не найден. Сначала отправь любой запрос в чат.",
            )

        record = load_credit_record(ip_hash) or {
            "credits": 0,
            "client_id": client_id,
        }
        record["credits"] = (
            max(0, int(record.get("credits", 0))) + int(promo["credits"])
        )
        record["client_id"] = client_id
        persist_credit_record(ip_hash, record)

        users.append(client_id)
        _save_promo_redemptions()

        return {
            "success": True,
            "credits_added": int(promo["credits"]),
            "credits": record["credits"],
        }


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title=APP_NAME, version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print("=" * 60, flush=True)
    print("UNHANDLED EXCEPTION on", request.method, request.url.path, flush=True)
    traceback.print_exc()
    print("=" * 60, flush=True)

    return JSONResponse(
        status_code=500,
        content={"detail": f"Внутренняя ошибка сервера: {type(exc).__name__}"},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("VALIDATION ERROR on", request.method, request.url.path, ":", exc.errors(), flush=True)
    return JSONResponse(
        status_code=422,
        content={"detail": "Некорректные данные запроса.", "errors": exc.errors()},
    )


# ============================================================
# SUPABASE REST CLIENT
# ============================================================

def supabase_request(method, table, data=None, params=None):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []

    url = f"{SUPABASE_URL}/rest/v1/{table}"

    if params:
        try:
            url += "?" + urlencode(params, doseq=True)
        except Exception as e:
            print("SUPABASE PARAM ERROR:", repr(e))
            return []

    body = None
    if data is not None:
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            print("SUPABASE JSON ERROR:", repr(e))
            return []

    headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": "Bearer " + SUPABASE_SECRET_KEY,
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

    if response.status_code >= 400:
        print("SUPABASE ERROR:", response.status_code, response.text[:1000])
        return []

    if not response.text:
        return []

    try:
        return response.json()
    except Exception:
        return []


# ============================================================
# TEXT
# ============================================================

RUSSIAN_STOPWORDS = {
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
    return [w for w in words if w not in RUSSIAN_STOPWORDS and len(w) >= 2]


def stable_hash(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


# ============================================================
# SYNONYMS
# ============================================================

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


def expand_query(text):
    normalized_text = normalize(text)
    words = tokenize(text)
    expanded = set(words)

    for canonical, variants in SYNONYMS.items():
        found = False
        for variant in variants:
            normalized_variant = normalize(variant)
            if not normalized_variant:
                continue
            if normalized_variant in normalized_text:
                found = True
                break

        if found:
            expanded.add(canonical)
            for variant in variants:
                for word in tokenize(variant):
                    expanded.add(word)

    return list(expanded)


# ============================================================
# NEURAL BRAIN
# ============================================================

class NeuralBrain:
    def __init__(self):
        self.vocabulary = []
        self.word_index = {}
        self.categories = []
        self.category_index = {}
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
                + " "
                + item.get("answer", "")
                + " "
                + " ".join(item.get("tags", []))
            )
            for word in expand_query(text):
                vocabulary.add(word)

            category = item.get("category")
            if category:
                categories.add(category)

        self.vocabulary = sorted(vocabulary)
        self.word_index = {w: i for i, w in enumerate(self.vocabulary)}
        self.categories = sorted(categories)
        self.category_index = {c: i for i, c in enumerate(self.categories)}

        if not self.vocabulary or not self.categories:
            self.ready = False
            return

        input_size = len(self.vocabulary)
        hidden_size = min(128, max(16, input_size // 2))
        output_size = len(self.categories)

        rng = np.random.default_rng(42)

        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

        self.ready = True

    def vectorize(self, text):
        vector = np.zeros(len(self.vocabulary))
        for word in expand_query(text):
            index = self.word_index.get(word)
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

    def train(self, knowledge, epochs=180, learning_rate=0.035):
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
            label = self.category_index.get(category)
            if label is None:
                continue

            dataset.append((vector, label))

        if not dataset:
            return {"success": False, "epochs": 0}

        for _ in range(epochs):
            for x, label in dataset:
                z1, h, prediction = self.forward(x)

                target = np.zeros(len(self.categories))
                target[label] = 1

                error = prediction - target

                dW2 = np.outer(h, error)
                db2 = error

                dh = error @ self.W2.T
                dz1 = dh * (z1 > 0)

                dW1 = np.outer(x, dz1)
                db1 = dz1

                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1

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

        _, _, output = self.forward(x)
        index = int(np.argmax(output))

        return self.categories[index], float(output[index])


# ============================================================
# DEFAULT KNOWLEDGE
# ============================================================

DEFAULT_KNOWLEDGE = [
    {
        "title": "Жирная кожа",
        "category": "skin",
        "question": "Что делать если у меня жирная кожа?",
        "answer": """
Если кожа быстро становится жирной, не стоит постоянно
и агрессивно обезжиривать её.

Базовый уход:

1. Умывай лицо мягким очищающим средством утром и вечером.
2. Не используй агрессивное мыло и спиртовые средства без необходимости.
3. Рассмотри средства с ниацинамидом или салициловой кислотой,
   если они подходят твоей коже.
4. Используй лёгкий увлажняющий крем.
5. Днём используй солнцезащитное средство.
6. Не выдавливай воспаления.

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

1. Наладь режим сна (7–9 часов, стабильное время отхода ко сну).
2. Пей достаточно воды в течение дня.
3. Используй крем для области вокруг глаз с кофеином,
   витамином К или ретинолом (если кожа не чувствительная).
4. Прикладывай холодный компресс на несколько минут утром.
5. Используй солнцезащитный крем — УФ усиливает пигментацию.
6. Высыпайся и ограничь соль и алкоголь вечером — это уменьшает отёки.

Если круги появились резко, сопровождаются отёком, болью
или другими симптомами — стоит показаться врачу, чтобы
исключить, например, аллергию или проблемы с носовыми пазухами.
""",
        "tags": ["темные круги", "синяки", "мешки", "под глазами", "глаза", "недосып"],
    },
]


# ============================================================
# GLOBAL STATE
# ============================================================

knowledge_cache = []
brain = NeuralBrain()


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def load_knowledge():
    global knowledge_cache

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        knowledge_cache = DEFAULT_KNOWLEDGE.copy()
        brain.train(knowledge_cache)
        print("Supabase not configured.")
        print("Using default knowledge:", len(knowledge_cache))
        return

    rows = supabase_request(
        "GET",
        "knowledge",
        params={"select": "*", "approved": "eq.true", "order": "created_at.desc"},
    )

    if rows:
        knowledge_cache = rows
    else:
        knowledge_cache = DEFAULT_KNOWLEDGE.copy()

    brain.train(knowledge_cache)

    print("Knowledge:", len(knowledge_cache))
    print("Brain ready:", brain.ready)


# ============================================================
# LOCAL KNOWLEDGE SEARCH
# ============================================================

def similarity(a, b):
    a_words = set(expand_query(a))
    b_words = set(expand_query(b))

    if not a_words or not b_words:
        return 0.0

    intersection = len(a_words & b_words)
    union = len(a_words | b_words)

    return intersection / max(1, union)


def search_local_knowledge(query):
    predicted_category, confidence = brain.predict(query)

    results = []
    query_expanded = set(expand_query(query))

    for item in knowledge_cache:
        question = item.get("question", "")
        tags = " ".join(item.get("tags", []))
        title = item.get("title", "")

        score_question = similarity(query, question)
        score_tags = similarity(query, tags)
        score_title = similarity(query, title)

        direct_similarity = max(score_question, score_tags, score_title)

        category_bonus = 0.0
        if (
            predicted_category
            and item.get("category") == predicted_category
            and direct_similarity >= 0.08
        ):
            category_bonus = confidence * 0.15

        score = (
            score_question * 0.50
            + score_tags * 0.25
            + score_title * 0.15
            + category_bonus
        )

        combined_text = normalize(question + " " + title + " " + tags)

        exact_bonus = 0.0
        for word in query_expanded:
            if len(word) >= 4 and word in combined_text:
                exact_bonus += 0.03
        exact_bonus = min(exact_bonus, 0.15)

        score += exact_bonus

        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)

    filtered_results = [item for item in results if item[0] >= 0.10]

    print("LOCAL QUERY:", query)
    print("LOCAL PREDICTED CATEGORY:", predicted_category)
    print("LOCAL CATEGORY CONFIDENCE:", confidence)

    if filtered_results:
        print(
            "LOCAL TOP RESULT:",
            filtered_results[0][1].get("title", ""),
            "score=",
            filtered_results[0][0],
        )
    else:
        print("LOCAL RESULT: none")

    return filtered_results[:5]


# ============================================================
# WEB HELPERS
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def valid_http_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


def _parse_searxng_payload(payload, limit):
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    results = []
    seen_urls = set()

    for raw in raw_results:
        if not isinstance(raw, dict):
            continue

        title = clean_text(raw.get("title", ""))
        url_value = str(raw.get("url", "") or raw.get("link", ""))
        snippet = clean_text(raw.get("content", "") or raw.get("snippet", "") or "")

        if not title:
            continue
        if not valid_http_url(url_value):
            continue
        if url_value in seen_urls:
            continue

        seen_urls.add(url_value)

        results.append(
            {
                "title": title[:250],
                "url": url_value,
                "snippet": snippet[:1500],
                "source": "searxng",
            }
        )

        if len(results) >= limit:
            break

    return results


# ============================================================
# SEARCH ENGINE 1: SEARXNG (with retry/backoff per instance)
# ============================================================

def searxng_search(query, limit=MAX_SEARCH_RESULTS):
    query = query.strip()
    if not query:
        return []

    headers = {
        **BROWSER_HEADERS,
        "Accept": "application/json",
    }

    for instance in SEARXNG_INSTANCES:
        base = instance.rstrip("/")
        url = base + "/search"

        print("")
        print("------------------------------------------")
        print("SEARXNG QUERY:", query)
        print("SEARXNG INSTANCE:", base)

        for attempt in range(1, SEARXNG_MAX_RETRIES_PER_INSTANCE + 1):
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
                    timeout=SEARXNG_TIMEOUT,
                    allow_redirects=True,
                )

                print(
                    f"SEARXNG HTTP (attempt {attempt}):",
                    response.status_code,
                )

            except Exception as e:
                print(f"SEARXNG REQUEST ERROR (attempt {attempt}):", repr(e))
                break

            if response.status_code == 429:
                if attempt < SEARXNG_MAX_RETRIES_PER_INSTANCE:
                    delay = SEARXNG_RETRY_BACKOFF_BASE * attempt
                    print(f"SEARXNG 429 — retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    print("SEARXNG BAD STATUS: 429 (giving up on instance)")
                    break

            if response.status_code >= 400:
                print("SEARXNG BAD STATUS:", response.status_code)
                break

            content_type = response.headers.get("content-type", "").lower()
            if "json" not in content_type:
                print("SEARXNG NOT JSON:", content_type)
                break

            try:
                payload = response.json()
            except Exception as e:
                print("SEARXNG JSON ERROR:", repr(e))
                break

            results = _parse_searxng_payload(payload, limit)

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


# ============================================================
# SEARCH ENGINE 2: DUCKDUCKGO HTML (fallback, no API key needed)
# ============================================================

def duckduckgo_html_search(query, limit=MAX_SEARCH_RESULTS):
    query = query.strip()
    if not query:
        return []

    url = "https://html.duckduckgo.com/html/"

    print("")
    print("------------------------------------------")
    print("DUCKDUCKGO QUERY:", query)

    try:
        response = requests.post(
            url,
            data={"q": query, "kl": "ru-ru"},
            headers=BROWSER_HEADERS,
            timeout=SEARXNG_TIMEOUT,
        )

        print("DUCKDUCKGO HTTP:", response.status_code)

    except Exception as e:
        print("DUCKDUCKGO REQUEST ERROR:", repr(e))
        return []

    if response.status_code >= 400:
        print("DUCKDUCKGO BAD STATUS:", response.status_code)
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print("DUCKDUCKGO PARSE ERROR:", repr(e))
        return []

    results = []
    seen_urls = set()

    for result_div in soup.select(".result"):
        link = result_div.select_one("a.result__a")
        if not link:
            continue

        title = clean_text(link.get_text(" ", strip=True))
        href = link.get("href", "")

        snippet_tag = result_div.select_one(".result__snippet")
        snippet = clean_text(snippet_tag.get_text(" ", strip=True)) if snippet_tag else ""

        if not title or not valid_http_url(href):
            continue
        if href in seen_urls:
            continue

        seen_urls.add(href)

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


# ============================================================
# UNIFIED WEB SEARCH (tries all engines in order)
# ============================================================

SEARCH_ENGINES = [
    ("searxng", searxng_search),
    ("duckduckgo", duckduckgo_html_search),
]


# ============================================================
# GARBAGE / REDIRECT PAGE DETECTION
# ============================================================

REDIRECT_STUB_MARKERS = (
    "please click here if the page does not redirect",
    "click here if you are not redirected",
    "redirecting you to",
    "javascript is disabled",
)


def is_redirect_stub(text):
    if not text:
        return False
    normalized = text.strip().lower()
    if len(normalized) < 400 and any(marker in normalized for marker in REDIRECT_STUB_MARKERS):
        return True
    return False


def web_search_with_fallback(query, limit=MAX_SEARCH_RESULTS):
    for name, engine_fn in SEARCH_ENGINES:
        try:
            results = engine_fn(query, limit)
        except Exception as e:
            print(f"SEARCH ENGINE '{name}' CRASHED:", repr(e))
            results = []

        if results:
            print(f"WEB SEARCH ENGINE USED: {name}")
            return results, name

    print("WEB SEARCH: all engines failed.")
    return [], None


# ============================================================
# FETCH WEB PAGE
# ============================================================

def fetch_page_text(url):
    if not valid_http_url(url):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ASCEND-AI/2.1)",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    }

    print("FETCH SOURCE:", url)

    try:
        response = requests.get(
            url, headers=headers, timeout=PAGE_TIMEOUT, allow_redirects=True
        )

        print("SOURCE HTTP:", response.status_code)

        if response.status_code >= 400:
            print("SOURCE ERROR STATUS")
            return ""

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            print("SOURCE NOT HTML:", content_type)
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
            tag.decompose()

        text = clean_text(soup.get_text(" ", strip=True))
        text = text[:MAX_SOURCE_TEXT]

        print("SOURCE TEXT LENGTH:", len(text))

        return text

    except Exception as e:
        print("SOURCE FETCH ERROR:", repr(e))
        return ""


# ============================================================
# COLLECT WEB INFORMATION
# ============================================================

def collect_web_information(query):
    print("")
    print("==========================================")
    print("WEB SEARCH START")
    print("QUERY:", query)

    search_results, engine_used = web_search_with_fallback(query)

    if not search_results:
        print("WEB SEARCH: no engine returned results.")
        print("==========================================")
        return []

    print("WEB SEARCH ENGINE:", engine_used)
    print("WEB SEARCH RESULTS:", len(search_results))

    enriched = []
    for index, result in enumerate(search_results, start=1):
        page_text = fetch_page_text(result["url"])

        if is_redirect_stub(page_text):
            print(f"SOURCE {index}: SKIPPED (redirect stub / no real content)")
            page_text = ""

        if not page_text and not clean_text(result.get("snippet", "")):
            print(f"SOURCE {index}: DROPPED (no content, no snippet)")
            continue

        enriched.append({**result, "page_text": page_text})

        print(
            f"SOURCE {index}: title={result.get('title', '')[:100]} "
            f"text={len(page_text)} snippet={len(result.get('snippet', ''))}"
        )

    print("WEB SEARCH COMPLETE")
    print("==========================================")

    return enriched


# ============================================================
# SAVE WEB SOURCES
# ============================================================

def save_web_sources(session_id, query, results):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    for result in results:
        payload = {
            "session_id": session_id,
            "query": query,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
            "page_text": result.get("page_text", ""),
            "source": result.get("source", "web"),
        }

        supabase_request("POST", "web_sources", payload)


# ============================================================
# WEB CONTEXT
# ============================================================

def build_web_context(results):
    pieces = []

    for index, item in enumerate(results, start=1):
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        page_text = item.get("page_text", "")

        text = page_text or snippet
        if not text:
            continue

        pieces.append(
            f"\nИСТОЧНИК {index}\nНазвание: {title}\nURL: {url}\nИнформация:\n\n{text}\n"
        )

    return "\n".join(pieces)


def clean_web_text(results):
    pieces = []
    for item in results:
        text = item.get("page_text") or item.get("snippet") or ""
        text = clean_text(text)
        if text:
            pieces.append(text)

    return "\n\n".join(pieces)


# ============================================================
# SENTENCES
# ============================================================

def split_sentences(text):
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(x) for x in parts if len(clean_text(x)) > 20]


def rank_sentences(query, text, limit=8):
    sentences = split_sentences(text)
    qwords = set(expand_query(query))

    scored = []
    for sentence in sentences:
        swords = set(expand_query(sentence))
        overlap = len(qwords & swords)

        if overlap:
            score = overlap / math.sqrt(max(1, len(swords)))
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [sentence for _, sentence in scored[:limit]]


# ============================================================
# SETTINGS PERSISTENCE (Supabase key-value table `app_settings`)
# ============================================================

SETTINGS_KEYS = (
    "openrouter_api_key",
    "deepseek_api_key",
    "qwen_api_key",
    "provod_api_key",
    "provod_model",
    "llm_direct_mode",
)


def load_settings():
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    rows = supabase_request(
        "GET",
        "app_settings",
        params={"select": "key,value"},
    )

    for row in rows or []:
        key = row.get("key")
        value = row.get("value")
        if key in SETTINGS_KEYS and value:
            runtime_settings[key] = value

    print("Settings loaded from Supabase:", [k for k in SETTINGS_KEYS if runtime_settings.get(k)])


def save_setting(key, value):
    if key not in SETTINGS_KEYS:
        raise ValueError(f"Unknown setting key: {key}")

    runtime_settings[key] = value

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request(
            "POST",
            "app_settings",
            {"key": key, "value": value},
            params={"on_conflict": "key"},
        )


# ============================================================
# LLM ANSWER SYNTHESIS
# ============================================================

API_KEY_SETTINGS = ("openrouter_api_key", "deepseek_api_key", "qwen_api_key", "provod_api_key")


def llm_available():
    return any(get_setting(k) for k in API_KEY_SETTINGS)


def _chat_completion_request(url, api_key, model, messages, extra_headers=None):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 900,
        },
        timeout=LLM_TIMEOUT,
    )
    return response


def _try_openrouter(messages):
    api_key = get_setting("openrouter_api_key")
    if not api_key:
        return None

    for model in OPENROUTER_FREE_MODELS:
        print("LLM TRY: openrouter /", model)
        try:
            response = _chat_completion_request(OPENROUTER_URL, api_key, model, messages)
            print("LLM HTTP:", response.status_code, "openrouter /", model)
        except Exception as e:
            print("LLM REQUEST ERROR (openrouter):", repr(e), model)
            continue

        if response.status_code in (404, 429) or response.status_code >= 400:
            print("LLM SKIP (openrouter):", response.status_code, model)
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


def _try_deepseek_direct(messages):
    api_key = get_setting("deepseek_api_key")
    if not api_key:
        return None

    print("LLM TRY: deepseek direct")
    try:
        response = _chat_completion_request(DEEPSEEK_URL, api_key, DEEPSEEK_MODEL, messages)
        print("LLM HTTP:", response.status_code, "deepseek")
    except Exception as e:
        print("LLM REQUEST ERROR (deepseek):", repr(e))
        return None

    if response.status_code >= 400:
        print("LLM BAD STATUS (deepseek):", response.status_code, response.text[:300])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: deepseek direct")
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (deepseek):", repr(e))
        return None


def _try_qwen_direct(messages):
    api_key = get_setting("qwen_api_key")
    if not api_key:
        return None

    print("LLM TRY: qwen direct")
    try:
        response = _chat_completion_request(QWEN_URL, api_key, QWEN_MODEL, messages)
        print("LLM HTTP:", response.status_code, "qwen")
    except Exception as e:
        print("LLM REQUEST ERROR (qwen):", repr(e))
        return None

    if response.status_code >= 400:
        print("LLM BAD STATUS (qwen):", response.status_code, response.text[:300])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: qwen direct")
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (qwen):", repr(e))
        return None


def _try_provod_direct(messages):
    api_key = get_setting("provod_api_key")
    if not api_key:
        return None

    model = get_setting("provod_model") or PROVOD_DEFAULT_MODEL

    print("LLM TRY: provod direct /", model)
    try:
        response = _chat_completion_request(PROVOD_URL, api_key, model, messages)
        print("LLM HTTP:", response.status_code, "provod /", model)
    except Exception as e:
        print("LLM REQUEST ERROR (provod):", repr(e))
        return None

    if response.status_code >= 400:
        print("LLM BAD STATUS (provod):", response.status_code, response.text[:500])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: provod direct /", model)
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (provod):", repr(e))
        return None


def build_history_messages(history):
    if not history:
        return []

    messages = []
    for item in history[-MAX_MEMORY:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content})

    return messages


def call_llm(system_prompt, user_prompt, history=None):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(build_history_messages(history))
    messages.append({"role": "user", "content": user_prompt})

    providers = (
        _try_provod_direct,
        _try_openrouter,
        _try_deepseek_direct,
        _try_qwen_direct,
    )

    for provider_fn in providers:
        result = provider_fn(messages)
        if result:
            return result

    print("LLM: all providers failed")
    return None


def llm_answer_from_local(query, knowledge_answer, web_results, history=None):
    web_context = build_web_context(web_results) if web_results else ""

    system_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Отвечай на русском языке, простым "
        "разговорным текстом, без списков источников и без вставки URL "
        "в текст ответа. Можешь использовать нумерованные шаги или "
        "маркированные пункты для советов, если это уместно. "
        "Не выдумывай медицинские факты — если сомневаешься, порекомендуй "
        "обратиться к врачу."
    )

    user_prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"Проверенный ответ из базы знаний (используй как основу):\n"
        f"{knowledge_answer}\n"
    )

    if web_context:
        user_prompt += (
            f"\nДополнительная информация из свежего веб-поиска "
            f"(используй только если она релевантна и не противоречит "
            f"базе знаний; не перечисляй источники и не вставляй ссылки):\n"
            f"{web_context}\n"
        )

    user_prompt += (
        "\nПерескажи это связным текстом на русском, дружелюбно и по делу."
    )

    return call_llm(system_prompt, user_prompt, history=history)


def llm_answer_from_web(query, web_results, history=None):
    web_context = build_web_context(web_results)
    if not web_context:
        return None

    system_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Отвечай на русском языке связным "
        "текстом на основе предоставленной информации из интернета. "
        "НЕ перечисляй источники, НЕ вставляй URL и названия сайтов "
        "в текст ответа — просто дай полезный ответ по существу. "
        "Если информации недостаточно для уверенного ответа, честно "
        "скажи об этом и порекомендуй обратиться к специалисту."
    )

    user_prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"Информация, найденная в интернете:\n{web_context}\n\n"
        "Дай связный, дружелюбный ответ по существу вопроса."
    )

    return call_llm(system_prompt, user_prompt, history=history)


def llm_answer_general(query, history=None):
    system_prompt = (
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

    user_prompt = f"Вопрос пользователя: {query}\n\nДай полезный ответ по существу."

    return call_llm(system_prompt, user_prompt, history=history)


# ============================================================
# WEB ANSWER (extractive fallback — used only if LLM unavailable)
# ============================================================

def fallback_web_answer(query, web_results):
    web_text = clean_web_text(web_results)
    if not web_text:
        return ""

    sentences = rank_sentences(query, web_text, limit=8)

    if not sentences:
        sentences = split_sentences(web_text)[:5]

    if not sentences:
        return ""

    answer = "Я нашёл информацию по твоему вопросу в интернете.\n\n"

    for sentence in sentences:
        answer += "• " + sentence + "\n"

    return answer


# ============================================================
# RESPONSE GENERATOR
# ============================================================

def generate_response(query, memory, local_results, web_results):
    best_score = 0.0
    best_item = None

    if local_results:
        best_score, best_item = local_results[0]

    print("LOCAL BEST SCORE:", best_score)
    if best_item:
        print("LOCAL BEST:", best_item.get("title", ""))
    print("WEB RESULTS:", len(web_results))
    print("LLM AVAILABLE:", llm_available())

    if best_item and best_score >= 0.18:
        knowledge_answer = best_item.get("answer", "").strip()

        if llm_available():
            llm_answer = llm_answer_from_local(query, knowledge_answer, web_results, history=memory)
            if llm_answer:
                return llm_answer

        answer = knowledge_answer

        if web_results:
            web_text = clean_web_text(web_results)
            sentences = rank_sentences(query, web_text, limit=4)

            if sentences:
                answer += "\n\nДополнение из актуального поиска:\n"
                for sentence in sentences:
                    answer += "\n• " + sentence

        return answer

    if web_results:
        if llm_available():
            llm_answer = llm_answer_from_web(query, web_results, history=memory)
            if llm_answer:
                return llm_answer

        web_answer = fallback_web_answer(query, web_results)
        if web_answer:
            return web_answer

    if llm_available():
        print("Falling back to LLM general knowledge (no local/web data)")
        general_answer = llm_answer_general(query, history=memory)
        if general_answer:
            return general_answer

    return (
        "Я не смог получить результаты веб-поиска прямо сейчас.\n\n"
        "Попробуй повторить запрос немного позже или сформулировать его подробнее."
    )


# ============================================================
# MEMORY
# ============================================================

def save_message(session_id, role, content):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return None

    rows = supabase_request(
        "POST",
        "chat_messages",
        {"session_id": session_id, "role": role, "content": content},
    )

    if rows:
        return rows[0]

    return None


def get_memory(session_id):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []

    rows = supabase_request(
        "GET",
        "chat_messages",
        params={
            "select": "role,content,created_at",
            "session_id": f"eq.{session_id}",
            "order": "created_at.desc",
            "limit": str(MAX_MEMORY),
        },
    )

    rows.reverse()
    return rows


# ============================================================
# TRAINING LOG
# ============================================================

def save_training_log(question, answer, category, source):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    supabase_request(
        "POST",
        "training_log",
        {
            "question": question,
            "answer": answer,
            "category": category,
            "source": source,
            "approved": True,
        },
    )


# ============================================================
# ADMIN AUTH
# ============================================================

def create_admin_token():
    timestamp = str(int(time.time()))
    raw = ADMIN_PASSWORD + ":" + timestamp
    signature = hashlib.sha256(raw.encode()).hexdigest()
    return timestamp + "." + signature


def verify_admin_token(token):
    if not token:
        return False

    parts = token.split(".")
    if len(parts) != 2:
        return False

    timestamp, signature = parts

    try:
        timestamp_int = int(timestamp)
    except Exception:
        return False

    if abs(int(time.time()) - timestamp_int) > 43200:
        return False

    expected = hashlib.sha256((ADMIN_PASSWORD + ":" + timestamp).encode()).hexdigest()

    return secrets.compare_digest(signature, expected)


def check_admin(request: Request):
    token = request.headers.get("X-Admin-Token", "")
    if not verify_admin_token(token):
        raise HTTPException(status_code=401, detail="Нет доступа.")


# ============================================================
# MODELS
# ============================================================

class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    client_id: Optional[str] = ""
    history: Optional[list[HistoryItem]] = []


class CreditsTopUp(BaseModel):
    client_id: str
    amount: int


class PromoRedeemRequest(BaseModel):
    client_id: str
    code: str


class AdminLogin(BaseModel):
    password: str


class KnowledgeCreate(BaseModel):
    title: str
    category: str
    question: str
    answer: str
    tags: list[str] = []


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[str] = None
    rating: int
    comment: Optional[str] = ""


class SettingsUpdate(BaseModel):
    openrouter_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    provod_api_key: Optional[str] = None
    provod_model: Optional[str] = None
    llm_direct_mode: Optional[bool] = None


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover,interactive-widget=resizes-content">
<title>ASCEND AI</title>
<style>
:root {
    --bg: #000000;
    --bg-soft: #000000;
    --panel: transparent;
    --panel-border: rgba(255,255,255,0.10);
    --panel-hover: rgba(255,255,255,0.07);
    --text: #ececec;
    --text-dim: rgba(236,236,236,0.62);
    --text-faint: rgba(236,236,236,0.38);
    --accent-a: #10a37f;
    --accent-b: #10a37f;
    --accent-c: #10a37f;
    --gradient: #10a37f;
    --radius-lg: 26px;
    --radius-md: 20px;
    --radius-sm: 15px;

    /* Liquid glass tokens */
    --glass-bg: rgba(255,255,255,0.055);
    --glass-bg-strong: rgba(255,255,255,0.09);
    --glass-border: rgba(255,255,255,0.14);
    --glass-border-soft: rgba(255,255,255,0.08);
    --glass-blur: blur(28px) saturate(180%);
    --glass-blur-strong: blur(40px) saturate(190%);
    --glass-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.10);
    --glass-shadow-soft: 0 4px 18px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.07);
    --app-height: 100dvh;
}
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
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

/* -------- Ambient background -------- */
.bg-glow {
    position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
}
.blob {
    position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.28;
}
.b1 { width: 480px; height: 480px; background: radial-gradient(circle, #10a37f, transparent 70%); top: -160px; left: -120px; }
.b2 { width: 380px; height: 380px; background: radial-gradient(circle, #19c37d, transparent 70%); bottom: -140px; right: -100px; }
.b3 { width: 300px; height: 300px; background: radial-gradient(circle, #0d7d63, transparent 70%); top: 40%; left: 60%; opacity: 0.16; }

/* -------- App shell -------- */
.app { position: relative; z-index: 1; display: flex; height: var(--app-height); }

/* -------- Sidebar (glass panel) -------- */
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
.icon-btn:hover { background: var(--panel-hover); border-color: var(--glass-border-soft); }
.icon-btn:active { transform: scale(0.93); }
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
.new-chat-btn:hover { background: rgba(255,255,255,0.13); }
.new-chat-btn:active { transform: scale(0.985); }
.chat-list { flex: 1; overflow-y: auto; padding: 4px 12px 10px; -webkit-overflow-scrolling: touch; }
.chat-item {
    display: flex; align-items: center; gap: 8px; padding: 11px 12px; border-radius: var(--radius-sm);
    margin-bottom: 2px; cursor: pointer; transition: background .18s; position: relative;
}
.chat-item:hover { background: var(--panel-hover); }
.chat-item.active { background: var(--glass-bg-strong); box-shadow: inset 0 0 0 1px var(--glass-border-soft); }
.chat-item .title { flex: 1; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; opacity: .92; font-weight: 400; }
.chat-item .del {
    opacity: 0; width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    color: var(--text-faint); transition: opacity .15s, background .15s, color .15s; flex-shrink: 0;
}
.chat-item:hover .del { opacity: 1; }
.chat-item .del:hover { background: rgba(255,92,92,0.18); color: #ff8a8a; }
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
.link-chip:hover { background: var(--panel-hover); color: var(--text); }
.link-chip svg { width: 14px; height: 14px; }

/* -------- Overlay -------- */
.overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.55); -webkit-backdrop-filter: blur(2px); backdrop-filter: blur(2px); z-index: 30; opacity: 0; pointer-events: none; transition: opacity .28s; }
.overlay.show { opacity: 1; pointer-events: all; }

/* -------- Main column -------- */
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
.pill-btn:hover { background: var(--panel-hover); }
.pill-btn:active { transform: scale(0.96); }
.pill-btn.gradient {
    background: linear-gradient(135deg, var(--accent-a), #19c37d);
    color: #fff; border: none; font-weight: 700;
    box-shadow: 0 4px 16px rgba(16,163,127,0.35), inset 0 1px 0 rgba(255,255,255,0.3);
}
.pill-btn svg { width: 15px; height: 15px; }

/* -------- Chat area -------- */
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

/* typing indicator */
.typing-dots { display: flex; gap: 5px; padding: 6px 2px; }
.typing-dots span {
    width: 7px; height: 7px; border-radius: 50%; background: var(--text-dim);
    animation: bounce 1.1s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: .15s; }
.typing-dots span:nth-child(3) { animation-delay: .3s; }
@keyframes bounce { 0%,60%,100% { transform: translateY(0); opacity: .4; } 30% { transform: translateY(-5px); opacity: 1; } }

/* -------- Composer -------- */
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
.composer-box:focus-within { box-shadow: var(--glass-shadow), 0 0 0 2px rgba(16,163,127,0.45); border-color: rgba(16,163,127,0.5); }
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
.send-btn:hover { transform: scale(1.06); }
.send-btn:disabled { opacity: .35; cursor: not-allowed; transform: none; }
.composer-hint {
    text-align: center;
    font-size: 11px;
    color: var(--text-faint);
    margin-top: 8px;
    letter-spacing: .3px;
}

/* -------- Modals -------- */
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
.doc-link:hover { background: var(--panel-hover); }
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

/* pricing */
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

/* -------- Admin -------- */
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
.knowledge-item:first-child { border-top: 0; }
.badge { display: inline-flex; align-items: center; gap: 6px; background: var(--glass-bg-strong); padding: 5px 10px; border-radius: 8px; font-size: 12px; opacity: .85; }
.badge svg { width: 12px; height: 12px; }
.status { margin-top: 12px; opacity: .7; font-size: 13px; }
.hidden { display: none !important; }
.admin-back { display: inline-flex; align-items: center; gap: 6px; margin-bottom: 16px; font-size: 13px; color: var(--text-dim); }

/* -------- Extra animation & interactivity layer -------- */
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
.icon-btn:hover { transform: rotate(8deg); }
.icon-btn:active { transform: scale(0.88) rotate(0deg); }

.new-chat-btn svg { transition: transform .25s ease; }
.new-chat-btn:hover svg { transform: rotate(90deg); }

.chat-item { animation: itemPop .28s cubic-bezier(.34,1.4,.64,1) both; }
@keyframes itemPop {
    from { opacity: 0; transform: translateX(-10px) scale(0.97); }
    to { opacity: 1; transform: translateX(0) scale(1); }
}
.chat-item .title { transition: transform .18s; }
.chat-item:hover .title { transform: translateX(2px); }
.chat-item .del { transform: scale(0.7) rotate(-90deg); transition: opacity .15s, transform .2s, background .15s, color .15s; }
.chat-item:hover .del { transform: scale(1) rotate(0deg); }

.credits-pill b { display: inline-block; transition: transform .2s; }
.credits-pill b.pulse, .pill-btn.pulse { animation: creditsPulse .55s ease; }
@keyframes creditsPulse {
    0% { transform: scale(1); }
    35% { transform: scale(1.22); color: var(--accent-a); }
    100% { transform: scale(1); }
}

.link-chip, .doc-link { transition: background .18s, color .18s, transform .18s; }
.link-chip:hover { transform: translateY(-2px); }
.doc-link:hover { transform: translateX(3px); }

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
.pill-btn.gradient:hover, .btn-primary:hover, .buy-btn:hover, .primary:hover { transform: translateY(-1px); }
.pill-btn.gradient:active, .btn-primary:active, .buy-btn:active, .primary:active { transform: translateY(0) scale(0.97); }

.send-btn { transition: transform .18s cubic-bezier(.34,1.56,.64,1), opacity .15s, box-shadow .2s; }
.send-btn:active:not(:disabled) { transform: scale(0.9); }

.avatar { transition: transform .2s; }
.message:hover .avatar { transform: scale(1.08) rotate(-4deg); }

.plan-card { transition: border-color .2s, transform .2s, box-shadow .2s; }
.plan-card:hover { transform: translateY(-3px); border-color: rgba(16,163,127,0.5); box-shadow: var(--glass-shadow); }
.buy-btn { transition: transform .15s, box-shadow .2s; }
.buy-btn:hover { box-shadow: 0 5px 16px rgba(16,163,127,0.5); }

.card { transition: border-color .2s, transform .2s; }
.card:hover { border-color: rgba(16,163,127,0.3); }

.knowledge-item, .doc-link, .plan-card { animation: fadeSlideIn .35s ease both; }
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.14); border-radius: 10px; }

@media (max-width: 860px) {
    .sidebar { width: 84%; max-width: 300px; }
    .chat-inner, .composer-inner { width: 94%; }
    .bubble { max-width: 92%; }
}
</style>
</head>
<body>

<div class="bg-glow"><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div></div>

<div class="app">

  <div id="overlay" class="overlay" onclick="closeSidebar()"></div>

  <aside id="sidebar" class="sidebar">
    <div class="sidebar-head">
      <div class="brand"><span class="brand-dot"></span>ASCEND AI</div>
      <button class="icon-btn" onclick="closeSidebar()" title="Закрыть">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>
      </button>
    </div>
    <button class="new-chat-btn" onclick="newChat()">
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      Новый чат
    </button>
    <div id="chatList" class="chat-list"></div>
    <div class="sidebar-foot">
      <div class="credits-pill">
        <span class="label">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="width:15px;height:15px;opacity:.7"><rect x="2" y="6" width="20" height="13" rx="3"/><path d="M2 10h20"/><circle cx="17" cy="14.5" r="1.4" fill="currentColor" stroke="none"/></svg>
          Баланс запросов
        </span>
        <b id="creditsBadgeSide">…</b>
      </div>
      <div class="foot-links">
        <button class="link-chip" onclick="openPricing()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="3"/><path d="M2 10h20"/></svg>
          Тарифы
        </button>
        <button class="link-chip" onclick="openDocs()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M14 3v5h5"/></svg>
          Документы
        </button>
        <button class="link-chip" onclick="openPromo()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-7"/><path d="M2 7h20v5H2z"/><path d="M12 7v14"/><path d="M12 7H8.5a2.5 2.5 0 1 1 2.5-2.5V7z"/><path d="M12 7h3.5A2.5 2.5 0 1 0 13 4.5V7z"/></svg>
          Промокод
        </button>
        <a class="link-chip" href="https://t.me/lovnff" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
          Поддержка
        </a>
      </div>
    </div>
  </aside>

  <div class="main-col">
    <header class="topbar">
      <div class="topbar-left">
        <button class="icon-btn" onclick="openSidebar()" title="Чаты">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="4" y1="7" x2="20" y2="7"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="17" x2="20" y2="17"/></svg>
        </button>
        <div class="topbar-title">ASCEND <span>AI</span></div>
      </div>
      <div class="topbar-right">
        <button class="pill-btn" id="creditsBadgeTop" onclick="openPricing()">Баланс: …</button>
        <button class="pill-btn gradient" onclick="openPricing()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="3"/><path d="M2 10h20"/></svg>
          Тарифы
        </button>
      </div>
    </header>

    <section id="chatSection" style="display:flex; flex-direction:column; flex:1; min-height:0;">
      <div id="chatArea" class="chat-area">
        <div class="chat-inner" id="messages"></div>
      </div>
      <div class="composer-wrap">
        <div class="composer-inner">
          <div class="composer-box">
            <textarea id="messageInput" rows="1" placeholder="Напиши свой вопрос..."></textarea>
            <button id="sendButton" class="send-btn" onclick="sendMessage()" title="Отправить">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
            </button>
          </div>
          <div class="composer-hint">mekbuda</div>
        </div>
      </div>
    </section>

    <section id="adminSection" class="admin">
      <div class="admin-wrap">
        <div class="admin-back" onclick="location.hash=''; location.reload();">← Вернуться в чат</div>
        <div id="adminLoginCard" class="card">
          <h2><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>Админка</h2>
          <p>Вход в панель управления нейросетью.</p>
          <div class="field">
            <label>Пароль</label>
            <input id="adminPassword" type="password" placeholder="Пароль администратора">
          </div>
          <button class="primary" onclick="loginAdmin()">Войти</button>
          <div id="loginStatus" class="status"></div>
        </div>
        <div id="adminPanel" class="hidden">
          <div class="card">
            <h2><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2a5.5 5.5 0 0 0-4.5 8.7A5.5 5.5 0 0 0 9 20a2 2 0 0 0 2-2v-1"/><path d="M14.5 2A5.5 5.5 0 0 1 19 10.7 5.5 5.5 0 0 1 15 20a2 2 0 0 1-2-2v-1"/></svg>Состояние нейросети</h2>
            <div id="brainStats">Загрузка...</div>
          </div>
          <div class="card">
            <h2><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="3"/><path d="M2 10h20"/></svg>Пополнить баланс пользователя</h2>
            <p style="opacity:.7;font-size:13px;margin-top:-8px;">
              Пользователь присылает свой client_id (виден ему в окне "Тарифы") после оплаты вручную —
              вставь его сюда и укажи, сколько запросов начислить.
            </p>
            <div class="field">
              <label>client_id пользователя</label>
              <input id="topupClientId" placeholder="например, 3f9a1c2b-...">
            </div>
            <div class="field">
              <label>Сколько запросов начислить</label>
              <input id="topupAmount" type="number" placeholder="50">
            </div>
            <button class="primary" onclick="topUpCredits()">Начислить</button>
            <div id="topupStatus" class="status"></div>
          </div>
          <div class="card">
            <h2><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3-3.5 3.5zm-3 0l-2 2"/></svg>API-ключи для нейросетей</h2>
            <p style="opacity:.7;font-size:13px;margin-top:-8px;">
              Вставляй сюда официальные API-ключи (не пароль от личного кабинета).
              DeepSeek: platform.deepseek.com → API Keys. Qwen: dashscope.console.aliyun.com.
              OpenRouter (бесплатные модели): openrouter.ai → Keys.
            </p>
            <div id="llmStatus" class="status"></div>
            <div class="field">
              <label>OpenRouter API Key</label>
              <input id="openrouterKey" type="password" placeholder="sk-or-v1-...">
            </div>
            <div class="field">
              <label>DeepSeek API Key</label>
              <input id="deepseekKey" type="password" placeholder="sk-...">
            </div>
            <div class="field">
              <label>Qwen (DashScope) API Key</label>
              <input id="qwenKey" type="password" placeholder="sk-...">
            </div>
            <div class="field">
              <label>provod.ai API Key</label>
              <input id="provodKey" type="password" placeholder="sk-...">
            </div>
            <div class="field">
              <label>provod.ai — имя модели (точно как в личном кабинете)</label>
              <input id="provodModel" placeholder="xiaomi/mimo-v2.5">
            </div>
            <div class="field" style="display:flex;align-items:center;gap:10px;">
              <input id="llmDirectMode" type="checkbox" style="width:auto;">
              <label style="margin:0;">Прямой LLM режим (не ходить в веб-поиск, отвечать сразу через LLM)</label>
            </div>
            <button class="primary" onclick="saveSettings()">Сохранить ключи</button>
            <div id="settingsStatus" class="status"></div>
          </div>
          <div class="card">
            <h2><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>Добавить знание</h2>
            <div class="field">
              <label>Название</label>
              <input id="title" placeholder="Например: Перхоть">
            </div>
            <div class="field">
              <label>Категория</label>
              <input id="category" placeholder="hair">
            </div>
            <div class="field">
              <label>Пример вопроса пользователя</label>
              <input id="question" placeholder="Что делать с перхотью?">
            </div>
            <div class="field">
              <label>Ответ нейросети</label>
              <textarea id="answer" placeholder="Напиши правильный ответ..."></textarea>
            </div>
            <div class="field">
              <label>Теги через запятую</label>
              <input id="tags" placeholder="перхоть, волосы, кожа головы">
            </div>
            <button class="primary" onclick="addKnowledge()">Обучить нейросеть</button>
            <div id="trainStatus" class="status"></div>
          </div>
          <div class="card">
            <h2><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>База знаний</h2>
            <div id="knowledgeList">Загрузка...</div>
          </div>
          <div class="card">
            <h2><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>Последние поиски</h2>
            <div id="webSearchList">Загрузка...</div>
          </div>
        </div>
      </div>
    </section>

  </div>
</div>

<!-- Consent / documents modal -->
<div id="docsModal" class="modal-overlay">
  <div class="modal-box">
    <h2>Прежде чем начать</h2>
    <p>Используя ASCEND AI, ты соглашаешься с пользовательским соглашением и политикой конфиденциальности. Если возникнут вопросы — поддержка всегда на связи.</p>
    <div class="doc-links">
      <a class="doc-link" href="https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54" target="_blank" rel="noopener noreferrer">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M14 3v5h5"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="13" y2="17"/></svg>Пользовательское соглашение</span>
        <span>↗</span>
      </a>
      <a class="doc-link" href="https://telegra.ph/Politika-konfidencialnosti-09-06-116" target="_blank" rel="noopener noreferrer">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="10" width="16" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>Политика конфиденциальности</span>
        <span>↗</span>
      </a>
      <a class="doc-link" href="https://t.me/lovnff" target="_blank" rel="noopener noreferrer">
        <span class="left"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>Поддержка в Telegram</span>
        <span>↗</span>
      </a>
    </div>
    <div class="modal-actions">
      <button class="btn-primary" onclick="acceptConsent()">Принять и продолжить</button>
    </div>
  </div>
</div>

<!-- Pricing modal -->
<div id="pricingModal" class="modal-overlay">
  <div class="modal-box">
    <h2>Тарифы</h2>
    <p>1-й запрос — бесплатно. Дальше выбери пакет — оплата по СБП появится совсем скоро, а пока баланс пополняется вручную через поддержку.</p>
    <div id="planGrid" class="plan-grid">Загрузка тарифов...</div>
    <div class="client-id-note">Твой ID для оплаты: <code id="clientIdShow">…</code><br>Пришли его в поддержку вместе с чеком.</div>
    <div class="modal-actions">
      <button class="btn-ghost" style="flex:1" onclick="closeModal('pricingModal')">Закрыть</button>
    </div>
  </div>
</div>

<!-- Promo code modal -->
<div id="promoModal" class="modal-overlay">
  <div class="modal-box">
    <h2>Промокод</h2>
    <p>Введи полученный промокод, чтобы добавить запросы на баланс.</p>
    <div class="field">
      <label>Промокод</label>
      <input id="promoCodeInput" type="text" autocomplete="off" autocapitalize="characters" placeholder="Введи промокод">
    </div>
    <button class="btn-primary" id="promoRedeemButton" onclick="redeemPromo()">Активировать</button>
    <div id="promoStatus" class="status"></div>
    <div class="modal-actions">
      <button class="btn-ghost" style="flex:1" onclick="closeModal('promoModal')">Закрыть</button>
    </div>
  </div>
</div>

<script>
// ============================================================
// VIEWPORT / KEYBOARD FIX (mobile Safari & Chrome)
// ============================================================
// Мобильные браузеры при появлении/скрытии клавиатуры меняют высоту
// visualViewport, но НЕ layout viewport, из-за чего body с
// position: fixed мог "залипать" со сдвигом — верхняя панель с
// кнопками уезжала за пределы экрана и не было возможности
// прокрутить обратно. Решение: держим точную высоту приложения в
// CSS-переменной --app-height, синхронизированной с
// window.visualViewport, и всегда сбрасываем скролл документа в 0,
// т.к. сам document скроллиться не должен — скроллится только
// внутренняя область чата (.chat-area).

function setAppHeight() {
    const vv = window.visualViewport;
    const height = vv ? vv.height : window.innerHeight;
    document.documentElement.style.setProperty("--app-height", height + "px");
    // На iOS после скрытия клавиатуры документ иногда остаётся
    // "прокрученным" вверх на величину бывшей клавиатуры — обнуляем.
    window.scrollTo(0, 0);
}

setAppHeight();
window.addEventListener("resize", setAppHeight);
window.addEventListener("orientationchange", setAppHeight);
if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", setAppHeight);
    window.visualViewport.addEventListener("scroll", setAppHeight);
}
document.addEventListener("focusout", () => setTimeout(setAppHeight, 60));
document.addEventListener("focusin", () => setTimeout(setAppHeight, 60));

// ============================================================
// STATE
// ============================================================
const CLIENT_ID_KEY = "ascend_client_id";
const CHATS_KEY = "ascend_chats";
const CURRENT_KEY = "ascend_current_chat";
const CONSENT_KEY = "ascend_consent_v1";

let clientId = localStorage.getItem(CLIENT_ID_KEY);
if (!clientId) {
    clientId = crypto.randomUUID();
    localStorage.setItem(CLIENT_ID_KEY, clientId);
}

let adminToken = localStorage.getItem("ascend_admin_token");

function loadChats() {
    try { return JSON.parse(localStorage.getItem(CHATS_KEY)) || []; } catch { return []; }
}
function saveChats(chats) { localStorage.setItem(CHATS_KEY, JSON.stringify(chats)); }
function msgsKey(id) { return "ascend_msgs_" + id; }
function loadLocalMsgs(id) {
    try { return JSON.parse(localStorage.getItem(msgsKey(id))) || []; } catch { return []; }
}
function saveLocalMsgs(id, msgs) { localStorage.setItem(msgsKey(id), JSON.stringify(msgs.slice(-60))); }

let chats = loadChats();
let currentChatId = localStorage.getItem(CURRENT_KEY);

if (!chats.length) {
    const id = crypto.randomUUID();
    chats = [{ id, title: "Новый чат", updatedAt: Date.now() }];
    currentChatId = id;
    saveChats(chats);
    localStorage.setItem(CURRENT_KEY, id);
} else if (!currentChatId || !chats.find(c => c.id === currentChatId)) {
    currentChatId = chats[0].id;
    localStorage.setItem(CURRENT_KEY, currentChatId);
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
        del.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="4" y1="4" x2="16" y2="16"/><line x1="16" y1="4" x2="4" y2="16"/></svg>';
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
    localStorage.setItem(CURRENT_KEY, id);
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
    localStorage.setItem(CURRENT_KEY, currentChatId);
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
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
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
            body: JSON.stringify({ session_id: currentChatId, message, client_id: clientId, history })
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
            if (typeof data.credits_left === "number") { setCredits(data.credits_left); }
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
        const r = await fetch("/api/credits?client_id=" + encodeURIComponent(clientId));
        if (r.ok) {
            const data = await r.json();
            setCredits(data.credits);
        }
    } catch {}
}

// ============================================================
// MODALS: consent / docs / pricing
// ============================================================
function openModal(id) { document.getElementById(id).classList.add("show"); }
function closeModal(id) { document.getElementById(id).classList.remove("show"); }

function acceptConsent() {
    localStorage.setItem(CONSENT_KEY, "1");
    closeModal("docsModal");
}
function openDocs() { openModal("docsModal"); }
function openPromo() {
    document.getElementById("promoCodeInput").value = "";
    document.getElementById("promoStatus").textContent = "";
    openModal("promoModal");
    setTimeout(() => document.getElementById("promoCodeInput").focus(), 80);
}

async function redeemPromo() {
    const input = document.getElementById("promoCodeInput");
    const button = document.getElementById("promoRedeemButton");
    const status = document.getElementById("promoStatus");
    const code = input.value.trim();

    if (!code) {
        status.textContent = "Введи промокод.";
        return;
    }

    button.disabled = true;
    status.textContent = "Проверяю...";
    try {
        const response = await fetch("/api/promo/redeem", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ client_id: clientId, code })
        });
        let data;
        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }

        if (!response.ok) {
            status.textContent = data.detail || "Не удалось активировать промокод.";
            return;
        }

        setCredits(data.credits);
        status.textContent = "Промокод активирован. Добавлено запросов: " + data.credits_added + ".";
        input.value = "";
    } catch (error) {
        console.error(error);
        status.textContent = "Ошибка соединения с сервером.";
    } finally {
        button.disabled = false;
    }
}

document.getElementById("promoCodeInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        redeemPromo();
    }
});

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
            card.innerHTML = `
                <div>
                    <div class="plan-name">${plan.title} · ${plan.requests} запросов</div>
                    <div class="plan-note">${plan.note || ""} · ~${pricePerReq}₽/запрос</div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <div class="plan-price">${plan.price}₽</div>
                    <button class="buy-btn">Купить</button>
                </div>
            `;
            card.querySelector(".buy-btn").onclick = () => buyPlan(plan, data.support);
            grid.appendChild(card);
        });
    } catch {
        grid.innerHTML = "Не удалось загрузить тарифы. Попробуй позже.";
    }
}

function buyPlan(plan, supportUrl) {
    const text = encodeURIComponent(
        `Хочу купить тариф "${plan.title}" (${plan.requests} запросов за ${plan.price}₽). Мой ID: ${clientId}`
    );
    window.open((supportUrl || "https://t.me/lovnff") + "?text=" + text, "_blank");
}

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
window.addEventListener("hashchange", checkAdminRoute);

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
        localStorage.setItem("ascend_admin_token", adminToken);
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
    if (!targetClientId || !amount) { status.textContent = "Укажи client_id и количество."; return; }
    status.textContent = "Начисляю...";
    try {
        const response = await fetch("/api/admin/credits", {
            method: "POST",
            headers: adminHeaders(),
            body: JSON.stringify({ client_id: targetClientId, amount })
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
            document.getElementById("brainStats").innerHTML = `
                <p>Модель: <strong>${stats.brain_ready ? "готова" : "не готова"}</strong></p>
                <p>Знаний: <strong>${stats.knowledge}</strong></p>
                <p>Словарь: <strong>${stats.vocabulary}</strong></p>
                <p>Категорий: <strong>${stats.categories}</strong></p>
            `;
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
            element.innerHTML = `
                <span class="badge">${escapeHtml(item.category || "")}</span>
                <h3>${escapeHtml(item.title || "")}</h3>
                <p><strong>Вопрос:</strong><br>${escapeHtml(item.question || "")}</p>
                <p>${escapeHtml(item.answer || "")}</p>
            `;
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
                element.innerHTML = `
                    <span class="badge">${escapeHtml(item.source || "web")}</span>
                    <h3>${escapeHtml(item.title || "")}</h3>
                    <p><strong>Запрос:</strong> ${escapeHtml(item.query || "")}</p>
                    <a href="${escapeHtml(item.url || "#")}" target="_blank" rel="noopener noreferrer">Открыть источник</a>
                `;
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
            `${label}: ${set ? "задан (" + escapeHtml(masked) + ")" : "— не задан"}`;
        status.innerHTML = [
            line("provod.ai", data.provod_api_key_set, data.provod_api_key_masked),
            `provod.ai модель: ${escapeHtml(data.provod_model || "")}`,
            line("OpenRouter", data.openrouter_api_key_set, data.openrouter_api_key_masked),
            line("DeepSeek", data.deepseek_api_key_set, data.deepseek_api_key_masked),
            line("Qwen", data.qwen_api_key_set, data.qwen_api_key_masked),
            `Прямой LLM режим: ${data.llm_direct_mode ? "включён" : "выключен"}`,
            `LLM-ответы: ${data.llm_enabled ? "включены" : "выключены (экстрактивный режим)"}`
        ].join("<br>");
        if (data.provod_model) {
            document.getElementById("provodModel").placeholder = data.provod_model;
        }
        document.getElementById("llmDirectMode").checked = !!data.llm_direct_mode;
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

    if (openrouterKey) { body.openrouter_api_key = openrouterKey; }
    if (deepseekKey) { body.deepseek_api_key = deepseekKey; }
    if (qwenKey) { body.qwen_api_key = qwenKey; }
    if (provodKey) { body.provod_api_key = provodKey; }
    if (provodModel) { body.provod_model = provodModel; }
    body.llm_direct_mode = document.getElementById("llmDirectMode").checked;

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
if (!localStorage.getItem(CONSENT_KEY)) { openModal("docsModal"); }
</script>
</body>
</html>
"""


# ============================================================
# ROOT
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "brain_ready": brain.ready,
        "knowledge": len(knowledge_cache),
        "search_engines": [name for name, _ in SEARCH_ENGINES],
        "llm_enabled": llm_available(),
    }


# ============================================================
# PROMO CODES
# ============================================================

@app.post("/api/promo/redeem")
async def promo_redeem(data: PromoRedeemRequest):
    return redeem_promo_code(data.client_id, data.code)


# ============================================================
# CHAT
# ============================================================

GREETING_WORDS = {
    "привет", "здравствуй", "здравствуйте", "приветик", "хай",
    "хеллоу", "хелло", "йо", "ку", "здарова", "здорово",
}

FAREWELL_WORDS = {"пока", "прощай", "досвидания", "бывай", "увидимся"}

THANKS_WORDS = {"спасибо", "благодарю", "спс", "сенкс", "thanks"}

HOWAREYOU_PHRASES = {
    "как дела", "как ты", "как жизнь", "как оно", "че как", "что нового",
}

GREETING_REPLIES = [
    "Привет! Расскажи, что тебя интересует — кожа, внешность, питание, "
    "сон или тренировки?",
    "Привет 👋 С чем помочь сегодня?",
]

FAREWELL_REPLIES = ["Пока! Возвращайся, если появятся вопросы 🙂"]

THANKS_REPLIES = ["Пожалуйста! Обращайся, если будут ещё вопросы."]

HOWAREYOU_REPLIES = [
    "Спасибо, у меня всё в порядке! А у тебя как дела? "
    "И расскажи, чем могу помочь.",
]


def detect_small_talk(message):
    normalized = normalize(message)

    for phrase in HOWAREYOU_PHRASES:
        if phrase in normalized:
            return random.choice(HOWAREYOU_REPLIES)

    words = normalized.split()

    if not words or len(words) > 4:
        return None

    word_set = set(words)

    if word_set & GREETING_WORDS and word_set <= (GREETING_WORDS | {"как", "дела", "там"}):
        return random.choice(GREETING_REPLIES)

    if word_set & FAREWELL_WORDS and word_set <= FAREWELL_WORDS:
        return random.choice(FAREWELL_REPLIES)

    if word_set & THANKS_WORDS and word_set <= THANKS_WORDS:
        return random.choice(THANKS_REPLIES)

    return None


@app.post("/api/chat")
async def chat(data: ChatRequest, request: Request):
    message = data.message.strip()

    if not message:
        raise HTTPException(400, "Пустой запрос.")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(400, "Сообщение слишком длинное.")

    ip_hash = hash_ip(request)
    credit_record = get_or_create_credit_record(ip_hash, data.client_id or "")

    if credit_record["credits"] <= 0:
        raise HTTPException(
            status_code=402,
            detail={
                "code": "NO_CREDITS",
                "message": "Бесплатный запрос уже использован. Пополни баланс, чтобы продолжить общение.",
                "plans": PRICING_PLANS,
                "support": SUPPORT_TELEGRAM,
            },
        )

    print("")
    print("=" * 60)
    print("NEW CHAT REQUEST:", message)
    print("=" * 60)

    memory = get_memory(data.session_id)
    if not memory and data.history:
        memory = [{"role": item.role, "content": item.content} for item in data.history]

    save_message(data.session_id, "user", message)

    small_talk_answer = detect_small_talk(message)

    if small_talk_answer:
        print("SMALL TALK DETECTED — skipping search/LLM")

        assistant_message = save_message(data.session_id, "assistant", small_talk_answer)
        consume_credit(ip_hash)

        print("=" * 60)

        return {
            "answer": small_talk_answer,
            "knowledge_found": False,
            "web_found": False,
            "memory_used": len(memory),
            "message_id": assistant_message.get("id") if assistant_message else None,
            "credits_left": credits_cache.get(ip_hash, {}).get("credits", 0),
        }

    local_results = search_local_knowledge(message)

    direct_mode = is_llm_direct_mode() and llm_available()

    if direct_mode:
        print("LLM DIRECT MODE: skipping web search")
        web_results = []
    else:
        web_results = collect_web_information(message)
        save_web_sources(data.session_id, message, web_results)

    answer = generate_response(message, memory, local_results, web_results)

    assistant_message = save_message(data.session_id, "assistant", answer)
    consume_credit(ip_hash)

    category = None
    if local_results:
        category = local_results[0][1].get("category")

    save_training_log(message, answer, category or "web", "search")

    print("FINAL WEB RESULTS USED:", len(web_results))
    print("=" * 60)

    return {
        "answer": answer,
        "knowledge_found": bool(local_results),
        "web_found": bool(web_results),
        "memory_used": len(memory),
        "message_id": assistant_message.get("id") if assistant_message else None,
        "credits_left": credits_cache.get(ip_hash, {}).get("credits", 0),
    }


# ============================================================
# CHAT HISTORY
# ============================================================

@app.get("/api/chat/history/{session_id}")
async def chat_history(session_id: str):
    return {"messages": get_memory(session_id)}


@app.delete("/api/chat/history/{session_id}")
async def delete_chat_history(session_id: str):
    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request("DELETE", "chat_messages", params={"session_id": f"eq.{session_id}"})
        supabase_request("DELETE", "web_sources", params={"session_id": f"eq.{session_id}"})
    return {"success": True}


# ============================================================
# CREDITS / ТАРИФЫ
# ============================================================

@app.get("/api/credits")
async def get_credits(request: Request, client_id: str = ""):
    ip_hash = hash_ip(request)
    record = get_or_create_credit_record(ip_hash, client_id)
    return {"credits": record["credits"], "client_id": record.get("client_id", "")}


@app.get("/api/pricing")
async def get_pricing():
    return {
        "plans": PRICING_PLANS,
        "support": SUPPORT_TELEGRAM,
        "privacy_url": PRIVACY_POLICY_URL,
        "terms_url": TERMS_OF_USE_URL,
    }


@app.post("/api/admin/credits")
async def admin_add_credits(data: CreditsTopUp, request: Request):
    check_admin(request)

    ip_hash = client_id_index.get(data.client_id)
    if not ip_hash and SUPABASE_URL and SUPABASE_SECRET_KEY:
        rows = supabase_request(
            "GET", "user_credits",
            params={"select": "ip_hash", "client_id": f"eq.{data.client_id}", "limit": "1"},
        )
        if rows:
            ip_hash = rows[0].get("ip_hash")

    if not ip_hash:
        raise HTTPException(404, "Пользователь с таким client_id не найден.")

    record = load_credit_record(ip_hash) or {"credits": 0, "client_id": data.client_id}
    record["credits"] = max(0, record["credits"] + data.amount)
    persist_credit_record(ip_hash, record)

    return {"success": True, "credits": record["credits"]}


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.post("/api/admin/login")
async def admin_login(data: AdminLogin):
    print("ADMIN LOGIN ATTEMPT", flush=True)

    try:
        password_ok = secrets.compare_digest(
            data.password.encode("utf-8"),
            ADMIN_PASSWORD.encode("utf-8"),
        )
    except Exception as e:
        print("ADMIN LOGIN compare_digest ERROR:", repr(e), flush=True)
        raise HTTPException(500, f"Ошибка проверки пароля: {e}")

    if not password_ok:
        print("ADMIN LOGIN: wrong password", flush=True)
        raise HTTPException(401, "Неверный пароль.")

    try:
        token = create_admin_token()
    except Exception as e:
        print("ADMIN LOGIN create_admin_token ERROR:", repr(e), flush=True)
        raise HTTPException(500, f"Ошибка создания токена: {e}")

    print("ADMIN LOGIN: success", flush=True)

    return {"success": True, "token": token}


# ============================================================
# ADMIN STATS
# ============================================================

@app.get("/api/admin/stats")
async def admin_stats(request: Request):
    check_admin(request)

    return {
        "brain_ready": brain.ready,
        "knowledge": len(knowledge_cache),
        "vocabulary": len(brain.vocabulary),
        "categories": len(brain.categories),
    }


# ============================================================
# ADMIN KNOWLEDGE GET
# ============================================================

@app.get("/api/admin/knowledge")
async def admin_knowledge(request: Request):
    check_admin(request)
    return knowledge_cache


# ============================================================
# ADMIN KNOWLEDGE CREATE
# ============================================================

@app.post("/api/admin/knowledge")
async def admin_add_knowledge(request: Request, data: KnowledgeCreate):
    check_admin(request)

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
    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        saved = supabase_request("POST", "knowledge", item)

    if saved:
        knowledge_cache.append(saved[0])
    else:
        item["id"] = stable_hash(title + question + answer)
        knowledge_cache.append(item)

    result = brain.train(knowledge_cache)

    save_training_log(question, answer, category, "admin")

    return {"success": True, "training": result, "knowledge": len(knowledge_cache)}


# ============================================================
# ADMIN DELETE KNOWLEDGE
# ============================================================

@app.delete("/api/admin/knowledge/{knowledge_id}")
async def admin_delete_knowledge(knowledge_id: str, request: Request):
    check_admin(request)

    global knowledge_cache

    knowledge_cache = [
        item for item in knowledge_cache if str(item.get("id")) != str(knowledge_id)
    ]

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request("DELETE", "knowledge", params={"id": f"eq.{knowledge_id}"})

    brain.train(knowledge_cache)

    return {"success": True}


# ============================================================
# ADMIN WEB SOURCES
# ============================================================

@app.get("/api/admin/web-sources")
async def admin_web_sources(request: Request):
    check_admin(request)

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []

    rows = supabase_request(
        "GET",
        "web_sources",
        params={
            "select": "id,query,title,url,snippet,source,created_at",
            "order": "created_at.desc",
            "limit": "50",
        },
    )

    return rows


# ============================================================
# ADMIN SETTINGS
# ============================================================

@app.get("/api/admin/settings")
async def admin_get_settings(request: Request):
    check_admin(request)

    return {
        "openrouter_api_key_set": bool(get_setting("openrouter_api_key")),
        "openrouter_api_key_masked": mask_key(get_setting("openrouter_api_key")),
        "deepseek_api_key_set": bool(get_setting("deepseek_api_key")),
        "deepseek_api_key_masked": mask_key(get_setting("deepseek_api_key")),
        "qwen_api_key_set": bool(get_setting("qwen_api_key")),
        "qwen_api_key_masked": mask_key(get_setting("qwen_api_key")),
        "provod_api_key_set": bool(get_setting("provod_api_key")),
        "provod_api_key_masked": mask_key(get_setting("provod_api_key")),
        "provod_model": get_setting("provod_model") or PROVOD_DEFAULT_MODEL,
        "llm_direct_mode": is_llm_direct_mode(),
        "llm_enabled": llm_available(),
    }


@app.post("/api/admin/settings")
async def admin_update_settings(request: Request, data: SettingsUpdate):
    check_admin(request)

    updated = []

    if data.openrouter_api_key is not None:
        save_setting("openrouter_api_key", data.openrouter_api_key.strip())
        updated.append("openrouter_api_key")

    if data.deepseek_api_key is not None:
        save_setting("deepseek_api_key", data.deepseek_api_key.strip())
        updated.append("deepseek_api_key")

    if data.qwen_api_key is not None:
        save_setting("qwen_api_key", data.qwen_api_key.strip())
        updated.append("qwen_api_key")

    if data.provod_api_key is not None:
        save_setting("provod_api_key", data.provod_api_key.strip())
        updated.append("provod_api_key")

    if data.provod_model is not None and data.provod_model.strip():
        save_setting("provod_model", data.provod_model.strip())
        updated.append("provod_model")

    if data.llm_direct_mode is not None:
        save_setting("llm_direct_mode", "true" if data.llm_direct_mode else "false")
        updated.append("llm_direct_mode")

    return {"success": True, "updated": updated, "llm_enabled": llm_available()}


# ============================================================
# FEEDBACK
# ============================================================

@app.post("/api/feedback")
async def feedback(data: FeedbackRequest):
    if data.rating < 1 or data.rating > 5:
        raise HTTPException(400, "Оценка должна быть от 1 до 5.")

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request(
            "POST",
            "ai_feedback",
            {
                "session_id": data.session_id,
                "message_id": data.message_id,
                "rating": data.rating,
                "comment": data.comment or "",
            },
        )

    return {"success": True}


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():
    try:
        load_knowledge()
    except Exception as e:
        print("STARTUP ERROR in load_knowledge:", repr(e), flush=True)
        traceback.print_exc()

    try:
        load_settings()
        _load_promo_redemptions()
    except Exception as e:
        print("STARTUP ERROR in load_settings:", repr(e), flush=True)
        traceback.print_exc()

    print("", flush=True)
    print("=" * 60, flush=True)
    print("                  ASCEND AI", flush=True)
    print("=" * 60, flush=True)
    print("Knowledge:", len(knowledge_cache), flush=True)
    print("Neural brain:", brain.ready, flush=True)
    print("Search engines:", [name for name, _ in SEARCH_ENGINES], flush=True)
    print("SearXNG instances:", len(SEARXNG_INSTANCES), flush=True)
    print("LLM enabled:", llm_available(), flush=True)
    print("=" * 60, flush=True)
    print("")
