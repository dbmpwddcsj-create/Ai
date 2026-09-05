import os
import re
import json
import math
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "ASCEND AI"

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

MAX_MEMORY = 20


# ============================================================
# APP
# ============================================================

app = FastAPI(title=APP_NAME)


# ============================================================
# SIMPLE SUPABASE CLIENT
# Без AI API. Supabase используется только как БД.
# ============================================================

import urllib.request
import urllib.error


def supabase_request(method, table, data=None, params=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    url = f"{SUPABASE_URL}/rest/v1/{table}"

    if params:
        parts = []
        for k, v in params.items():
            parts.append(f"{k}={v}")
        url += "?" + "&".join(parts)

    body = None

    if data is not None:
        body = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8")

            if not raw:
                return []

            return json.loads(raw)

    except Exception as e:
        print("Supabase error:", e)
        return []


# ============================================================
# TEXT PROCESSING
# ============================================================

STOP_WORDS = {
    "и", "а", "но", "в", "во", "на", "с", "со",
    "у", "к", "по", "для", "из", "от", "до",
    "как", "что", "это", "мне", "у меня",
    "ли", "же", "же", "или", "бы", "я",
    "ты", "он", "она", "они", "мы"
}


def normalize(text: str):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    words = normalize(text).split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def make_hash(text):
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


# ============================================================
# KNOWLEDGE
# ============================================================

DEFAULT_KNOWLEDGE = [

    {
        "title": "Жирная кожа",
        "category": "skin",
        "question": "Что делать если у меня жирная кожа?",
        "answer": """
Если кожа быстро становится жирной, обычно стоит сосредоточиться не на
агрессивном обезжиривании, а на мягком и регулярном уходе.

Базовый вариант:

1. Умывай лицо мягким очищающим средством 2 раза в день.
2. Не используй слишком агрессивное мыло и не мой лицо десятки раз в день.
3. Ищи в уходе компоненты вроде ниацинамида или салициловой кислоты,
   если они тебе подходят.
4. Используй лёгкий некомедогенный увлажняющий крем.
5. Днём используй солнцезащиту.
6. Не выдавливай воспаления.

Если есть выраженное или болезненное акне, лучше обратиться к дерматологу.
        """,
        "tags": ["жирная кожа", "кожа", "себум", "акне", "прыщи"]
    },

    {
        "title": "Прыщи",
        "category": "skin",
        "question": "Как избавиться от прыщей?",
        "answer": """
При склонности к акне полезно выстроить простой постоянный уход.

Утром:
• мягкое очищение;
• увлажняющий крем;
• SPF.

Вечером:
• очищение;
• подходящее средство против акне;
• увлажнение.

Не стоит одновременно вводить большое количество активных средств.
Начинай постепенно и наблюдай за реакцией кожи.

При тяжёлом, болезненном или оставляющем рубцы акне желательно обратиться
к дерматологу.
        """,
        "tags": ["прыщи", "акне", "лицо", "кожа"]
    },

    {
        "title": "Уход за лицом",
        "category": "face",
        "question": "Как улучшить внешний вид лица?",
        "answer": """
На внешний вид лица обычно влияет не один фактор.

Полезная база:

• стабильный режим сна;
• регулярная физическая активность;
• достаточное количество воды;
• сбалансированное питание;
• мягкий уход за кожей;
• солнцезащита;
• аккуратная стрижка и уход за волосами;
• поддержание чистоты кожи;
• отказ от постоянного выдавливания прыщей.

Не существует одного упражнения или продукта, который мгновенно
изменит внешность.
        """,
        "tags": ["лицо", "внешность", "уход", "красота"]
    },

    {
        "title": "Сон",
        "category": "lifestyle",
        "question": "Как сон влияет на внешность?",
        "answer": """
Недостаток сна может отражаться на общем самочувствии и внешнем виде.

Для улучшения режима:

• старайся ложиться примерно в одно время;
• держи стабильное время подъёма;
• уменьши яркий экран перед сном;
• не употребляй большое количество кофеина поздно вечером;
• сделай спальню прохладной и комфортной.

Важна не одна идеальная ночь, а стабильный режим.
        """,
        "tags": ["сон", "внешность", "режим", "лицо"]
    },

    {
        "title": "Питание",
        "category": "nutrition",
        "question": "Что есть чтобы лучше выглядеть?",
        "answer": """
Для внешнего вида полезнее всего не экстремальная диета, а нормальное
сбалансированное питание.

Старайся регулярно получать:

• белок;
• овощи и фрукты;
• цельные источники углеводов;
• полезные жиры;
• достаточное количество жидкости.

Не нужно полностью исключать какую-либо группу продуктов без причины.
        """,
        "tags": ["питание", "еда", "внешность", "диета"]
    },

    {
        "title": "Тренировки",
        "category": "fitness",
        "question": "Как тренироваться чтобы улучшить внешность?",
        "answer": """
Для улучшения физической формы полезно сочетать силовые тренировки,
кардио и нормальное восстановление.

Основная идея:

• постепенно увеличивать нагрузку;
• тренировать основные мышечные группы;
• соблюдать технику;
• давать организму восстановиться;
• не пытаться компенсировать плохой сон чрезмерными тренировками.

Программа должна соответствовать твоему уровню подготовки.
        """,
        "tags": ["тренировки", "спорт", "мышцы", "форма"]
    },

]


# ============================================================
# LOCAL NEURAL NETWORK
#
# Это не ChatGPT-подобная LLM.
# Это собственное маленькое обучаемое нейронное ядро:
#
# text -> bag-of-words -> hidden layer -> intent/category
#
# Оно обучается непосредственно на твоей базе.
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

        all_words = set()
        categories = set()

        for item in knowledge:
            categories.add(item["category"])

            text = (
                item.get("question", "") + " " +
                item.get("answer", "") + " " +
                " ".join(item.get("tags", []))
            )

            for word in tokenize(text):
                all_words.add(word)

        self.vocabulary = sorted(all_words)
        self.word_index = {
            word: i for i, word in enumerate(self.vocabulary)
        }

        self.categories = sorted(categories)
        self.category_index = {
            c: i for i, c in enumerate(self.categories)
        }

        if not self.vocabulary or not self.categories:
            self.ready = False
            return

        input_size = len(self.vocabulary)
        hidden_size = min(128, max(16, input_size // 2))
        output_size = len(self.categories)

        rng = np.random.default_rng(42)

        self.W1 = rng.normal(
            0,
            np.sqrt(2 / input_size),
            (input_size, hidden_size)
        )

        self.b1 = np.zeros(hidden_size)

        self.W2 = rng.normal(
            0,
            np.sqrt(2 / hidden_size),
            (hidden_size, output_size)
        )

        self.b2 = np.zeros(output_size)

        self.ready = True

    def vectorize(self, text):

        x = np.zeros(len(self.vocabulary))

        for word in tokenize(text):
            if word in self.word_index:
                x[self.word_index[word]] += 1

        if np.sum(x) > 0:
            x /= np.linalg.norm(x) + 1e-8

        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):

        x = x - np.max(x)
        e = np.exp(x)

        return e / (np.sum(e) + 1e-8)

    def forward(self, x):

        z1 = x @ self.W1 + self.b1
        h = self.relu(z1)

        z2 = h @ self.W2 + self.b2
        y = self.softmax(z2)

        return z1, h, y

    def train(self, knowledge, epochs=250, lr=0.03):

        self.build(knowledge)

        if not self.ready:
            return

        X = []
        Y = []

        for item in knowledge:

            text = (
                item.get("question", "") + " " +
                " ".join(item.get("tags", []))
            )

            X.append(self.vectorize(text))
            Y.append(self.category_index[item["category"]])

        X = np.array(X)

        Y = np.array(Y)

        for epoch in range(epochs):

            for x, label in zip(X, Y):

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

                self.W2 -= lr * dW2
                self.b2 -= lr * db2

                self.W1 -= lr * dW1
                self.b1 -= lr * db1

        self.ready = True

    def predict(self, text):

        if not self.ready:
            return None, 0

        x = self.vectorize(text)

        if np.sum(x) == 0:
            return None, 0

        _, _, probabilities = self.forward(x)

        index = int(np.argmax(probabilities))

        return (
            self.categories[index],
            float(probabilities[index])
        )


brain = NeuralBrain()


# ============================================================
# KNOWLEDGE STORAGE
# ============================================================

knowledge_cache = []


def load_knowledge():

    global knowledge_cache

    remote = supabase_request(
        "GET",
        "knowledge",
        params={
            "select": "*",
            "approved": "eq.true",
            "order": "created_at.desc"
        }
    )

    if remote:
        knowledge_cache = remote
    else:
        knowledge_cache = DEFAULT_KNOWLEDGE.copy()

    brain.train(knowledge_cache)

    print(
        f"Knowledge loaded: {len(knowledge_cache)} items"
    )


# ============================================================
# SIMILARITY
# ============================================================

def text_similarity(a, b):

    a_words = set(tokenize(a))
    b_words = set(tokenize(b))

    if not a_words or not b_words:
        return 0

    intersection = len(a_words & b_words)
    union = len(a_words | b_words)

    return intersection / union


def search_knowledge(query):

    results = []

    predicted_category, confidence = brain.predict(query)

    for item in knowledge_cache:

        score = text_similarity(
            query,
            item.get("question", "")
        )

        tag_score = text_similarity(
            query,
            " ".join(item.get("tags", []))
        )

        category_bonus = 0

        if (
            predicted_category and
            item.get("category") == predicted_category
        ):
            category_bonus = confidence * 0.25

        final_score = (
            score * 0.65 +
            tag_score * 0.35 +
            category_bonus
        )

        results.append(
            (final_score, item)
        )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:5]


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query, memory=None):

    results = search_knowledge(query)

    if not results:
        return (
            "Пока я не нашёл достаточно информации в своей базе знаний. "
            "Добавь знания через админку, и я смогу использовать их "
            "в следующих ответах."
        )

    best_score, best = results[0]

    # Если совпадение достаточно хорошее,
    # используем найденное знание.
    if best_score >= 0.12:

        answer = best["answer"].strip()

        return answer

    # Если совпадение слабое,
    # но категория определилась.
    predicted_category, confidence = brain.predict(query)

    if predicted_category and confidence > 0.45:

        category_items = [
            item for item in knowledge_cache
            if item.get("category") == predicted_category
        ]

        if category_items:

            combined = category_items[0]["answer"].strip()

            return (
                "Я определил тему твоего вопроса как "
                f"«{predicted_category}».\n\n"
                + combined
            )

    return (
        "Я пока не уверен в ответе на этот вопрос. "
        "Попробуй сформулировать его немного подробнее. "
        "Например: «У меня жирная кожа, как уменьшить жирность?»"
    )


# ============================================================
# MEMORY
# ============================================================

def save_message(
    session_id,
    role,
    content
):

    if not SUPABASE_URL or not SUPABASE_KEY:
        return

    supabase_request(
        "POST",
        "chat_messages",
        {
            "session_id": session_id,
            "role": role,
            "content": content
        }
    )


def get_memory(session_id):

    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    rows = supabase_request(
        "GET",
        "chat_messages",
        params={
            "select": "role,content,created_at",
            "session_id": f"eq.{session_id}",
            "order": "created_at.desc",
            "limit": str(MAX_MEMORY)
        }
    )

    rows.reverse()

    return rows


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    session_id: str
    message: str


class AdminLogin(BaseModel):

    password: str


class KnowledgeCreate(BaseModel):

    title: str
    category: str
    question: str
    answer: str
    tags: list[str] = []


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="ru">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>ASCEND AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #09090b;
    color: #fff;
}

.container {
    width: min(1000px, 94%);
    margin: auto;
}

header {
    padding: 25px 0;
    border-bottom: 1px solid #222;
}

.logo {
    font-size: 25px;
    font-weight: 800;
}

.logo span {
    opacity: .5;
}

.chat {
    height: calc(100vh - 160px);
    min-height: 500px;
    display: flex;
    flex-direction: column;
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 25px 0;
}

.message {
    margin-bottom: 18px;
    max-width: 75%;
}

.message.user {
    margin-left: auto;
    text-align: right;
}

.bubble {
    display: inline-block;
    padding: 14px 17px;
    border-radius: 18px;
    line-height: 1.5;
    white-space: pre-wrap;
}

.ai .bubble {
    background: #18181b;
    border: 1px solid #29292d;
}

.user .bubble {
    background: #f7d45b;
    color: #111;
}

.input-area {
    display: flex;
    gap: 10px;
    padding: 15px 0 25px;
}

textarea {
    flex: 1;
    resize: none;
    min-height: 55px;
    max-height: 150px;
    border: 1px solid #29292d;
    background: #111113;
    color: #fff;
    border-radius: 15px;
    padding: 16px;
    font-size: 15px;
    outline: none;
}

button {
    border: 0;
    border-radius: 14px;
    padding: 0 22px;
    cursor: pointer;
    background: #f7d45b;
    color: #111;
    font-weight: 700;
}

button:disabled {
    opacity: .5;
}

.admin {
    display: none;
    padding: 30px 0;
}

.card {
    background: #111113;
    border: 1px solid #29292d;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
}

input,
select {
    width: 100%;
    padding: 13px;
    margin: 7px 0 12px;
    background: #09090b;
    color: #fff;
    border: 1px solid #29292d;
    border-radius: 10px;
}

.admin textarea {
    width: 100%;
}

.knowledge {
    margin-top: 20px;
}

.knowledge-item {
    border-top: 1px solid #29292d;
    padding: 15px 0;
}

.small {
    opacity: .55;
    font-size: 13px;
}

.top-actions {
    display: flex;
    gap: 10px;
}

.top-actions button {
    padding: 10px 15px;
}

</style>

</head>

<body>

<header>

<div class="container">

<div class="top-actions">

<div class="logo">
ASCEND <span>AI</span>
</div>

<button onclick="showAdmin()">
⚙ Админка
</button>

<button onclick="showChat()">
💬 Чат
</button>

</div>

</div>

</header>


<main class="container">


<section id="chatSection">

<div class="chat">

<div id="messages"
     class="messages">

<div class="message ai">

<div class="bubble">
Привет! Я ASCEND AI.

Я помогаю разбираться с вопросами,
связанными с внешностью, кожей,
лицом, волосами, питанием и тренировками.

Напиши свой вопрос 👇
</div>

</div>

</div>


<div class="input-area">

<textarea
id="messageInput"
placeholder="Напиши свой вопрос..."
></textarea>

<button id="sendButton"
onclick="sendMessage()">

Отправить

</button>

</div>

</div>

</section>


<section id="adminSection"
         class="admin">

<div class="card">

<h2>🧠 Обучение ASCEND AI</h2>

<p class="small">
Здесь ты добавляешь знания, которыми
будет пользоваться твоя нейросеть.
</p>

<input id="adminPassword"
       type="password"
       placeholder="Пароль администратора">

<button onclick="loginAdmin()">
Войти
</button>

</div>


<div id="adminPanel"
     style="display:none">


<div class="card">

<h2>➕ Добавить знание</h2>

<input id="title"
       placeholder="Название">

<input id="category"
       placeholder="Категория">

<input id="question"
       placeholder="Пример вопроса">

<textarea id="answer"
          placeholder="Правильный ответ..."
          style="min-height:150px">
</textarea>

<input id="tags"
       placeholder="Теги через запятую">

<button onclick="addKnowledge()">
Обучить
</button>

</div>


<div class="card">

<h2>📚 База знаний</h2>

<div id="knowledgeList">
Загрузка...
</div>

</div>

</div>

</section>

</main>


<script>

const sessionKey = "ascend_session";

let sessionId =
localStorage.getItem(sessionKey);

if (!sessionId) {

    sessionId =
        crypto.randomUUID();

    localStorage.setItem(
        sessionKey,
        sessionId
    );
}


function addMessage(
    role,
    text
) {

    const messages =
        document.getElementById(
            "messages"
        );

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message " +
        (role === "user"
            ? "user"
            : "ai");

    const bubble =
        document.createElement("div");

    bubble.className = "bubble";

    bubble.textContent = text;

    wrapper.appendChild(bubble);

    messages.appendChild(wrapper);

    messages.scrollTop =
        messages.scrollHeight;
}


async function sendMessage() {

    const input =
        document.getElementById(
            "messageInput"
        );

    const button =
        document.getElementById(
            "sendButton"
        );

    const message =
        input.value.trim();

    if (!message) return;

    addMessage(
        "user",
        message
    );

    input.value = "";

    button.disabled = true;

    addMessage(
        "ai",
        "🧠 Думаю..."
    );

    try {

        const response =
            await fetch("/api/chat", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    session_id:
                        sessionId,

                    message:
                        message

                })

            });


        const data =
            await response.json();


        const messages =
            document.getElementById(
                "messages"
            );

        messages.lastElementChild
            .remove();


        if (!response.ok) {

            addMessage(
                "ai",
                data.detail ||
                "Произошла ошибка."
            );

        } else {

            addMessage(
                "ai",
                data.answer
            );

        }

    } catch (error) {

        const messages =
            document.getElementById(
                "messages"
            );

        messages.lastElementChild
            .remove();

        addMessage(
            "ai",
            "Не удалось связаться с сервером."
        );

    }

    button.disabled = false;
}


document
    .getElementById("messageInput")
    .addEventListener(
        "keydown",
        function(e) {

            if (
                e.key === "Enter" &&
                !e.shiftKey
            ) {

                e.preventDefault();

                sendMessage();
            }

        }
    );


function showAdmin() {

    document.getElementById(
        "chatSection"
    ).style.display = "none";

    document.getElementById(
        "adminSection"
    ).style.display = "block";
}


function showChat() {

    document.getElementById(
        "adminSection"
    ).style.display = "none";

    document.getElementById(
        "chatSection"
    ).style.display = "block";
}


async function loginAdmin() {

    const password =
        document.getElementById(
            "adminPassword"
        ).value;

    const response =
        await fetch(
            "/api/admin/login",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    password
                })

            }
        );

    const data =
        await response.json();

    if (!response.ok) {

        alert(
            data.detail ||
            "Неверный пароль"
        );

        return;
    }

    document.getElementById(
        "adminPanel"
    ).style.display = "block";

    loadKnowledge();
}


async function addKnowledge() {

    const title =
        document.getElementById(
            "title"
        ).value.trim();

    const category =
        document.getElementById(
            "category"
        ).value.trim();

    const question =
        document.getElementById(
            "question"
        ).value.trim();

    const answer =
        document.getElementById(
            "answer"
        ).value.trim();

    const tags =
        document.getElementById(
            "tags"
        ).value
        .split(",")
        .map(x => x.trim())
        .filter(Boolean);


    const password =
        document.getElementById(
            "adminPassword"
        ).value;


    const response =
        await fetch(
            "/api/admin/knowledge",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json",
                    "X-Admin-Password":
                        password
                },

                body: JSON.stringify({

                    title,
                    category,
                    question,
                    answer,
                    tags

                })

            }
        );


    const data =
        await response.json();


    if (!response.ok) {

        alert(
            data.detail ||
            "Ошибка"
        );

        return;
    }


    alert(
        "Знание добавлено. Нейросеть обучена."
    );


    document.getElementById(
        "title"
    ).value = "";

    document.getElementById(
        "category"
    ).value = "";

    document.getElementById(
        "question"
    ).value = "";

    document.getElementById(
        "answer"
    ).value = "";

    document.getElementById(
        "tags"
    ).value = "";


    loadKnowledge();
}


async function loadKnowledge() {

    const password =
        document.getElementById(
            "adminPassword"
        ).value;

    const response =
        await fetch(
            "/api/admin/knowledge",
            {
                headers: {
                    "X-Admin-Password":
                        password
                }
            }
        );


    if (!response.ok) return;


    const data =
        await response.json();


    const container =
        document.getElementById(
            "knowledgeList"
        );


    container.innerHTML = "";


    data.forEach(item => {

        const element =
            document.createElement(
                "div"
            );

        element.className =
            "knowledge-item";


        element.innerHTML = `
            <strong>${escapeHtml(item.title)}</strong>
            <div class="small">
                ${escapeHtml(item.category)}
            </div>
            <p>
                ${escapeHtml(item.question)}
            </p>
            <p>
                ${escapeHtml(item.answer)}
            </p>
        `;


        container.appendChild(
            element
        );

    });

}


function escapeHtml(text) {

    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

</script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():

    return HTML


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "knowledge": len(knowledge_cache),
        "brain_ready": brain.ready
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
async def chat(data: ChatRequest):

    message = data.message.strip()

    if not message:
        raise HTTPException(
            400,
            "Пустой запрос."
        )

    if len(message) > 5000:
        raise HTTPException(
            400,
            "Сообщение слишком длинное."
        )


    memory = get_memory(
        data.session_id
    )


    save_message(
        data.session_id,
        "user",
        message
    )


    answer = generate_answer(
        message,
        memory
    )


    save_message(
        data.session_id,
        "assistant",
        answer
    )


    return {
        "answer": answer,
        "memory_used": len(memory),
        "brain": {
            "ready": brain.ready
        }
    }


# ============================================================
# ADMIN AUTH
# ============================================================

@app.post("/api/admin/login")
async def admin_login(
    data: AdminLogin
):

    if not secrets.compare_digest(
        data.password,
        ADMIN_PASSWORD
    ):
        raise HTTPException(
            401,
            "Неверный пароль."
        )

    return {
        "success": True
    }


def check_admin(request: Request):

    password =
        request.headers.get(
            "X-Admin-Password",
            ""
        )

    if not secrets.compare_digest(
        password,
        ADMIN_PASSWORD
    ):
        raise HTTPException(
            401,
            "Нет доступа."
        )


# ============================================================
# ADMIN KNOWLEDGE
# ============================================================

@app.get("/api/admin/knowledge")
async def get_admin_knowledge(
    request: Request
):

    check_admin(request)

    return knowledge_cache


@app.post("/api/admin/knowledge")
async def create_knowledge(
    request: Request,
    data: KnowledgeCreate
):

    check_admin(request)

    if not data.title:
        raise HTTPException(
            400,
            "Укажи название."
        )

    if not data.category:
        raise HTTPException(
            400,
            "Укажи категорию."
        )

    if not data.question:
        raise HTTPException(
            400,
            "Укажи пример вопроса."
        )

    if not data.answer:
        raise HTTPException(
            400,
            "Укажи ответ."
        )


    item = {

        "title": data.title,

        "category":
            normalize(data.category),

        "question":
            data.question,

        "answer":
            data.answer,

        "tags":
            data.tags,

        "approved":
            True

    }


    # Сохраняем в Supabase

    saved = supabase_request(
        "POST",
        "knowledge",
        item
    )


    # Если Supabase недоступен,
    # временно добавляем в RAM.

    if saved:

        knowledge_cache.append(
            saved[0]
        )

    else:

        item["id"] = make_hash(
            data.title +
            data.question
        )

        knowledge_cache.append(
            item
        )


    # Переобучаем собственную нейросеть.

    brain.train(
        knowledge_cache
    )


    return {
        "success": True,
        "knowledge_count":
            len(knowledge_cache)
    }


# ============================================================
# DELETE KNOWLEDGE
# ============================================================

@app.delete("/api/admin/knowledge/{knowledge_id}")
async def delete_knowledge(
    knowledge_id: str,
    request: Request
):

    check_admin(request)

    global knowledge_cache

    knowledge_cache = [
        x for x in knowledge_cache
        if str(x.get("id")) != knowledge_id
    ]


    if SUPABASE_URL and SUPABASE_KEY:

        supabase_request(
            "DELETE",
            "knowledge",
            params={
                "id":
                    f"eq.{knowledge_id}"
            }
        )


    brain.train(
        knowledge_cache
    )


    return {
        "success": True
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():

    load_knowledge()

    print("")
    print("================================")
    print("       ASCEND AI STARTED")
    print("================================")
    print(
        "Knowledge:",
        len(knowledge_cache)
    )
    print(
        "Neural brain:",
        brain.ready
    )
    print("================================")
