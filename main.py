import os
import re
import json
import time
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


# ============================================================
# AI CARE V3
# СОБСТВЕННАЯ НЕЙРОСЕТЬ + ГЕНЕРАЦИЯ + ПАМЯТЬ ЧАТА
# БЕЗ OPENAI / GEMINI / CLAUDE И ДРУГИХ AI API
# ============================================================

APP_NAME = "AI Care v3"

BASE_DIR = Path(__file__).resolve().parent

DB_FILE = BASE_DIR / "ai_care_memory.db"
MODEL_FILE = BASE_DIR / "ai_care_rnn.npz"
DATASET_FILE = BASE_DIR / "ai_care_dataset.json"

# ------------------------------------------------------------
# Настройки модели
# ------------------------------------------------------------

HIDDEN_SIZE = 128
LEARNING_RATE = 0.005
CLIP_VALUE = 5.0

# Сколько последних символов модель видит при генерации
MAX_CONTEXT_CHARS = 900

# Сколько последних сообщений сохраняем в контекст
MAX_MESSAGES = 12

# Ограничение длины одного сообщения
MAX_MESSAGE_LENGTH = 1200

# ------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------

app = FastAPI(title=APP_NAME)


# ============================================================
# DATABASE
# ============================================================

db_lock = threading.Lock()


def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            created_at REAL,
            updated_at REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            text TEXT,
            created_at REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            memory_key TEXT,
            memory_value TEXT,
            created_at REAL,
            updated_at REAL,
            UNIQUE(user_id, memory_key)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            response TEXT,
            created_at REAL
        )
    """)

    conn.commit()
    return conn


def ensure_user(user_id: str):
    now = time.time()

    with db_lock:
        conn = get_db()

        row = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if not row:
            conn.execute(
                """
                INSERT INTO users(user_id, name, created_at, updated_at)
                VALUES (?, '', ?, ?)
                """,
                (user_id, now, now)
            )
            conn.commit()

        conn.close()


def save_message(user_id: str, role: str, text: str):
    ensure_user(user_id)

    with db_lock:
        conn = get_db()

        conn.execute(
            """
            INSERT INTO messages(user_id, role, text, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, role, text, time.time())
        )

        conn.commit()
        conn.close()


def get_messages(user_id: str, limit: int = MAX_MESSAGES):
    ensure_user(user_id)

    with db_lock:
        conn = get_db()

        rows = conn.execute(
            """
            SELECT role, text
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

        conn.close()

    return list(reversed(rows))


def clear_messages(user_id: str):
    with db_lock:
        conn = get_db()

        conn.execute(
            "DELETE FROM messages WHERE user_id = ?",
            (user_id,)
        )

        conn.commit()
        conn.close()


# ============================================================
# USER MEMORY
# ============================================================

def save_memory(user_id: str, key: str, value: str):
    ensure_user(user_id)

    now = time.time()

    with db_lock:
        conn = get_db()

        conn.execute(
            """
            INSERT INTO memories(
                user_id,
                memory_key,
                memory_value,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, memory_key)
            DO UPDATE SET
                memory_value = excluded.memory_value,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                key,
                value,
                now,
                now
            )
        )

        conn.commit()
        conn.close()


def get_memories(user_id: str):
    ensure_user(user_id)

    with db_lock:
        conn = get_db()

        rows = conn.execute(
            """
            SELECT memory_key, memory_value
            FROM memories
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,)
        ).fetchall()

        conn.close()

    return {
        row["memory_key"]: row["memory_value"]
        for row in rows
    }


def delete_memories(user_id: str):
    with db_lock:
        conn = get_db()

        conn.execute(
            "DELETE FROM memories WHERE user_id = ?",
            (user_id,)
        )

        conn.commit()
        conn.close()


# ============================================================
# AUTOMATIC MEMORY EXTRACTION
# ============================================================

def extract_memory(user_id: str, text: str):
    """
    Очень простой локальный анализ сообщений.
    Никаких внешних AI API.
    """

    original = text.strip()
    lower = original.lower()

    # Имя
    patterns = [
        r"\bменя зовут\s+([а-яёa-z-]{2,30})",
        r"\bмоё имя\s+([а-яёa-z-]{2,30})",
        r"\bмое имя\s+([а-яёa-z-]{2,30})",
    ]

    for pattern in patterns:
        match = re.search(pattern, lower)

        if match:
            name = match.group(1).strip().capitalize()

            save_memory(
                user_id,
                "name",
                name
            )

            with db_lock:
                conn = get_db()

                conn.execute(
                    """
                    UPDATE users
                    SET name = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (name, time.time(), user_id)
                )

                conn.commit()
                conn.close()

            break

    # Любимые вещи
    favorite_match = re.search(
        r"(?:мне нравится|я люблю|моя любимая игра|мой любимый фильм)\s+(.+)",
        original,
        re.IGNORECASE
    )

    if favorite_match:
        value = favorite_match.group(1).strip()

        if len(value) < 150:
            save_memory(
                user_id,
                "preference",
                value
            )

    # Явная просьба запомнить
    remember_match = re.search(
        r"(?:запомни|запиши|сохрани)\s*:?\s*(.+)",
        original,
        re.IGNORECASE
    )

    if remember_match:
        value = remember_match.group(1).strip()

        if len(value) <= 250:
            save_memory(
                user_id,
                "user_note",
                value
            )


# ============================================================
# DATASET
# ============================================================

DEFAULT_DATASET = [
    {
        "prompt": "Привет",
        "response": "Привет! Я AI Care. Чем могу помочь?"
    },
    {
        "prompt": "Здравствуй",
        "response": "Привет! Рад тебя видеть. Рассказывай, что тебя интересует."
    },
    {
        "prompt": "Как дела?",
        "response": "У меня всё хорошо. Готов помочь тебе с вопросом."
    },
    {
        "prompt": "Что ты умеешь?",
        "response": "Я могу отвечать на вопросы, запоминать важный контекст нашего общения и помогать с уходом за кожей, волосами, гигиеной и полезными привычками."
    },
    {
        "prompt": "Что делать если кожа жирная?",
        "response": "При жирной коже обычно помогает мягкое очищение утром и вечером, лёгкий увлажняющий крем и отказ от слишком агрессивного умывания."
    },
    {
        "prompt": "У меня жирная кожа",
        "response": "Если кожа быстро становится жирной, попробуй мягкое очищение два раза в день и лёгкий некомедогенный увлажняющий крем."
    },
    {
        "prompt": "У меня сухая кожа",
        "response": "При сухости лучше использовать мягкое очищение и простой увлажняющий крем без слишком большого количества раздражающих компонентов."
    },
    {
        "prompt": "Как ухаживать за волосами?",
        "response": "Уход зависит от типа волос. Обычно важно регулярно мыть кожу головы подходящим шампунем и не перегружать волосы большим количеством средств."
    },
    {
        "prompt": "Как улучшить уход за лицом?",
        "response": "Начни с базовой схемы: мягкое очищение, увлажнение и защита кожи от солнца. Сложный уход не всегда нужен."
    },
    {
        "prompt": "Как избавиться от чёрных точек?",
        "response": "Чёрные точки часто связаны с работой сальных желез и забитыми порами. Мягкое очищение и подходящий уход могут уменьшить их выраженность."
    },
    {
        "prompt": "Как часто нужно умываться?",
        "response": "Для большинства людей достаточно мягко умываться утром и вечером. Слишком частое агрессивное очищение может раздражать кожу."
    },
    {
        "prompt": "Что делать если кожа раздражена?",
        "response": "На время лучше убрать агрессивные средства и оставить мягкое очищение и простой увлажняющий уход."
    },
    {
        "prompt": "Как лучше спать?",
        "response": "Полезно стараться ложиться и вставать примерно в одно время и давать организму достаточно сна."
    },
    {
        "prompt": "Почему я устаю?",
        "response": "Усталость может быть связана со сном, нагрузкой, стрессом, питанием и многими другими причинами. Если это длится долго или сильно мешает жизни, стоит поговорить со взрослым и врачом."
    },
    {
        "prompt": "Как стать более активным?",
        "response": "Можно начать с обычной ходьбы, прогулок или другого приятного движения и постепенно увеличивать активность без перегрузок."
    },
    {
        "prompt": "Как улучшить внешний вид?",
        "response": "Безопасный базовый вариант — ухаживать за кожей и волосами, соблюдать гигиену, нормально спать, двигаться и выбирать одежду, в которой тебе комфортно."
    },
    {
        "prompt": "Как ухаживать за кожей?",
        "response": "База ухода — мягкое очищение, увлажнение и защита от солнца. Дополнительные средства лучше добавлять постепенно."
    },
    {
        "prompt": "Что делать с сухими губами?",
        "response": "Можно использовать простой бальзам для губ и стараться не облизывать губы постоянно."
    },
    {
        "prompt": "Что делать если волосы быстро становятся жирными?",
        "response": "Можно подобрать шампунь для кожи головы и мыть волосы по мере необходимости. Слишком редкое мытьё не обязательно полезнее."
    },
    {
        "prompt": "Спасибо",
        "response": "Пожалуйста! Если появится ещё вопрос — обращайся."
    },
    {
        "prompt": "Пока",
        "response": "Пока! Удачи!"
    },
]


def ensure_dataset():
    if not DATASET_FILE.exists():
        DATASET_FILE.write_text(
            json.dumps(
                DEFAULT_DATASET,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )


def load_dataset():
    ensure_dataset()

    try:
        data = json.loads(
            DATASET_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return DEFAULT_DATASET.copy()


def save_dataset(data):
    DATASET_FILE.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# CHARACTER VOCABULARY
# ============================================================

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]


def normalize_text(text):
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # убираем повторяющиеся пробелы
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_vocab(dataset):
    chars = set()

    for item in dataset:
        chars.update(
            normalize_text(item["prompt"])
        )

        chars.update(
            normalize_text(item["response"])
        )

    # Добавляем часто используемые символы
    chars.update(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "0123456789"
        " .,!?;:-—()[]{}\"'«»/\\+=_*%#@"
    )

    chars = sorted(chars)

    vocab = SPECIAL_TOKENS + chars

    char_to_id = {
        char: i
        for i, char in enumerate(vocab)
    }

    id_to_char = {
        i: char
        for i, char in enumerate(vocab)
    }

    return char_to_id, id_to_char


# ============================================================
# RNN MODEL
# ============================================================

class SimpleRNN:
    """
    Небольшая vanilla RNN.

    Формула:

        h_t = tanh(Wxh*x_t + Whh*h_(t-1) + bh)

        y_t = softmax(Why*h_t + by)

    Обучение:
        BPTT + gradient descent

    Это настоящая обучаемая генеративная модель,
    но очень маленькая.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=HIDDEN_SIZE
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        scale = 0.08

        self.Wxh = (
            np.random.randn(
                hidden_size,
                vocab_size
            ) * scale
        )

        self.Whh = (
            np.random.randn(
                hidden_size,
                hidden_size
            ) * scale
        )

        self.Why = (
            np.random.randn(
                vocab_size,
                hidden_size
            ) * scale
        )

        self.bh = np.zeros(
            (hidden_size, 1),
            dtype=np.float32
        )

        self.by = np.zeros(
            (vocab_size, 1),
            dtype=np.float32
        )

    def softmax(self, x):
        x = x - np.max(x)

        exp_x = np.exp(x)

        return exp_x / (
            np.sum(exp_x) + 1e-12
        )

    def forward(self, sequence):
        h = np.zeros(
            (self.hidden_size, 1),
            dtype=np.float32
        )

        hs = {-1: h}

        probs = {}

        loss = 0.0

        for t in range(len(sequence) - 1):
            current_id = sequence[t]
            target_id = sequence[t + 1]

            x = np.zeros(
                (self.vocab_size, 1),
                dtype=np.float32
            )

            x[current_id, 0] = 1.0

            h = np.tanh(
                self.Wxh @ x
                + self.Whh @ hs[t - 1]
                + self.bh
            )

            y = (
                self.Why @ h
                + self.by
            )

            p = self.softmax(y)

            probs[t] = p
            hs[t] = h

            loss -= np.log(
                p[target_id, 0] + 1e-12
            )

        return loss, hs, probs

    def train_sequence(
        self,
        sequence,
        learning_rate=LEARNING_RATE
    ):
        if len(sequence) < 2:
            return 0.0

        loss, hs, probs = self.forward(
            sequence
        )

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)

        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(
            hs[0]
        )

        for t in reversed(
            range(len(sequence) - 1)
        ):
            target_id = sequence[t + 1]

            dy = probs[t].copy()

            dy[target_id, 0] -= 1.0

            dWhy += (
                dy @ hs[t].T
            )

            dby += dy

            dh = (
                self.Why.T @ dy
                + dh_next
            )

            dh_raw = (
                (1 - hs[t] ** 2)
                * dh
            )

            dbh += dh_raw

            current_id = sequence[t]

            x = np.zeros(
                (self.vocab_size, 1),
                dtype=np.float32
            )

            x[current_id, 0] = 1.0

            dWxh += (
                dh_raw @ x.T
            )

            previous_h = (
                hs[t - 1]
                if t - 1 in hs
                else np.zeros_like(hs[0])
            )

            dWhh += (
                dh_raw @ previous_h.T
            )

            dh_next = (
                self.Whh.T @ dh_raw
            )

        # Gradient clipping
        for grad in (
            dWxh,
            dWhh,
            dWhy,
            dbh,
            dby
        ):
            np.clip(
                grad,
                -CLIP_VALUE,
                CLIP_VALUE,
                out=grad
            )

        self.Wxh -= (
            learning_rate * dWxh
        )

        self.Whh -= (
            learning_rate * dWhh
        )

        self.Why -= (
            learning_rate * dWhy
        )

        self.bh -= (
            learning_rate * dbh
        )

        self.by -= (
            learning_rate * dby
        )

        return float(loss)

    def generate(
        self,
        prompt_ids,
        id_to_char,
        char_to_id,
        max_new_chars=500,
        temperature=0.75
    ):
        if not prompt_ids:
            return ""

        h = np.zeros(
            (self.hidden_size, 1),
            dtype=np.float32
        )

        last_id = prompt_ids[0]

        # Прогоняем контекст
        for current_id in prompt_ids:
            x = np.zeros(
                (self.vocab_size, 1),
                dtype=np.float32
            )

            x[current_id, 0] = 1.0

            h = np.tanh(
                self.Wxh @ x
                + self.Whh @ h
                + self.bh
            )

            last_id = current_id

        result = []

        for _ in range(max_new_chars):
            x = np.zeros(
                (self.vocab_size, 1),
                dtype=np.float32
            )

            x[last_id, 0] = 1.0

            h = np.tanh(
                self.Wxh @ x
                + self.Whh @ h
                + self.bh
            )

            logits = (
                self.Why @ h
                + self.by
            ).reshape(-1)

            logits /= max(
                temperature,
                0.05
            )

            logits -= np.max(logits)

            probabilities = np.exp(logits)

            probabilities /= (
                np.sum(probabilities)
                + 1e-12
            )

            # Нельзя генерировать специальные токены
            for token in SPECIAL_TOKENS:
                token_id = char_to_id.get(token)

                if token_id is not None:
                    probabilities[token_id] = 0.0

            total = np.sum(probabilities)

            if total <= 0:
                break

            probabilities /= total

            next_id = np.random.choice(
                len(probabilities),
                p=probabilities
            )

            char = id_to_char.get(
                next_id,
                ""
            )

            if char == "":
                break

            result.append(char)

            last_id = next_id

            # Останавливаемся после завершённого ответа
            if len(result) > 30:
                current = "".join(result)

                if (
                    current.endswith(".")
                    or current.endswith("!")
                    or current.endswith("?")
                ) and len(result) > 100:
                    break

        return "".join(result)


# ============================================================
# MODEL MANAGER
# ============================================================

model_lock = threading.Lock()

MODEL = None
CHAR_TO_ID = {}
ID_TO_CHAR = {}


def create_model():
    global MODEL
    global CHAR_TO_ID
    global ID_TO_CHAR

    dataset = load_dataset()

    CHAR_TO_ID, ID_TO_CHAR = build_vocab(
        dataset
    )

    MODEL = SimpleRNN(
        vocab_size=len(CHAR_TO_ID),
        hidden_size=HIDDEN_SIZE
    )

    if MODEL_FILE.exists():
        try:
            data = np.load(
                MODEL_FILE,
                allow_pickle=False
            )

            saved_vocab = json.loads(
                str(data["vocab_json"])
            )

            # Загружаем только если словарь совпадает
            if saved_vocab == CHAR_TO_ID:
                MODEL.Wxh = data["Wxh"]
                MODEL.Whh = data["Whh"]
                MODEL.Why = data["Why"]
                MODEL.bh = data["bh"]
                MODEL.by = data["by"]

                print("AI model loaded.")

                return

        except Exception as e:
            print(
                "Model loading error:",
                e
            )

    print(
        "New AI model created."
    )


def save_model():
    if MODEL is None:
        return

    np.savez_compressed(
        MODEL_FILE,
        Wxh=MODEL.Wxh,
        Whh=MODEL.Whh,
        Why=MODEL.Why,
        bh=MODEL.bh,
        by=MODEL.by,
        vocab_json=json.dumps(
            CHAR_TO_ID,
            ensure_ascii=False
        )
    )


# ============================================================
# TRAINING
# ============================================================

training_status = {
    "running": False,
    "epoch": 0,
    "epochs": 0,
    "loss": 0.0,
    "message": "Не обучается"
}


def make_training_sequence(prompt, response):
    """
    Формируем последовательность:

    Пользователь: привет
    AI: привет! чем могу помочь?
    """

    text = (
        "Пользователь: "
        + normalize_text(prompt)
        + "\n"
        + "AI: "
        + normalize_text(response)
    )

    text = text[:MAX_CONTEXT_CHARS]

    sequence = []

    for char in text:
        if char in CHAR_TO_ID:
            sequence.append(
                CHAR_TO_ID[char]
            )
        else:
            sequence.append(
                CHAR_TO_ID.get(
                    "<UNK>",
                    1
                )
            )

    return sequence


def train_model(
    epochs=5,
    learning_rate=LEARNING_RATE
):
    global training_status

    if training_status["running"]:
        return

    training_status["running"] = True
    training_status["epoch"] = 0
    training_status["epochs"] = epochs
    training_status["loss"] = 0
    training_status["message"] = "Обучение началось"

    try:
        dataset = load_dataset()

        if not dataset:
            training_status["message"] = (
                "Датасет пуст"
            )
            return

        with model_lock:
            # Если модель не существует
            if MODEL is None:
                create_model()

            total_examples = len(dataset)

            for epoch in range(epochs):
                total_loss = 0.0
                count = 0

                # Перемешивание
                indices = np.random.permutation(
                    total_examples
                )

                for index in indices:
                    item = dataset[index]

                    prompt = str(
                        item.get("prompt", "")
                    )

                    response = str(
                        item.get("response", "")
                    )

                    if not prompt or not response:
                        continue

                    sequence = make_training_sequence(
                        prompt,
                        response
                    )

                    if len(sequence) < 3:
                        continue

                    loss = MODEL.train_sequence(
                        sequence,
                        learning_rate
                    )

                    total_loss += loss
                    count += 1

                avg_loss = (
                    total_loss / max(count, 1)
                )

                training_status["epoch"] = (
                    epoch + 1
                )

                training_status["loss"] = (
                    round(avg_loss, 4)
                )

                training_status["message"] = (
                    f"Эпоха {epoch + 1}/{epochs}"
                )

                print(
                    f"[TRAIN] "
                    f"epoch={epoch + 1}/{epochs} "
                    f"loss={avg_loss:.4f}"
                )

            save_model()

            training_status["message"] = (
                "Обучение завершено"
            )

    except Exception as e:
        training_status["message"] = (
            f"Ошибка: {e}"
        )

        print(
            "Training error:",
            e
        )

    finally:
        training_status["running"] = False


def start_training(
    epochs=5,
    learning_rate=LEARNING_RATE
):
    if training_status["running"]:
        return False

    thread = threading.Thread(
        target=train_model,
        args=(
            epochs,
            learning_rate
        ),
        daemon=True
    )

    thread.start()

    return True


# ============================================================
# CONTEXT
# ============================================================

def build_chat_context(
    user_id: str,
    current_message: str
):
    messages = get_messages(
        user_id,
        MAX_MESSAGES
    )

    memories = get_memories(
        user_id
    )

    parts = []

    # Память
    if memories:
        parts.append(
            "Память пользователя:"
        )

        for key, value in memories.items():
            parts.append(
                f"{key}: {value}"
            )

        parts.append("")

    # История
    for message in messages:
        role = message["role"]
        text = message["text"]

        if role == "user":
            parts.append(
                f"Пользователь: {text}"
            )
        else:
            parts.append(
                f"AI: {text}"
            )

    parts.append(
        f"Пользователь: {current_message}"
    )

    parts.append(
        "AI:"
    )

    context = "\n".join(parts)

    return context[-MAX_CONTEXT_CHARS:]


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    user_id: str,
    message: str
):
    if MODEL is None:
        create_model()

    context = build_chat_context(
        user_id,
        message
    )

    # Вводим только известные символы
    ids = []

    for char in context:
        ids.append(
            CHAR_TO_ID.get(
                char,
                CHAR_TO_ID.get("<UNK>", 1)
            )
        )

    if not ids:
        return (
            "Я пока не знаю, что ответить."
        )

    # Чтобы генерация не была огромной
    ids = ids[-MAX_CONTEXT_CHARS:]

    with model_lock:
        answer = MODEL.generate(
            ids,
            ID_TO_CHAR,
            CHAR_TO_ID,
            max_new_chars=500,
            temperature=0.72
        )

    # Чистим ответ
    answer = answer.strip()

    # Если модель начала новый диалог
    stop_words = [
        "\nПользователь:",
        "\nAI:",
        "Пользователь:",
    ]

    for stop in stop_words:
        if stop in answer:
            answer = answer.split(
                stop,
                1
            )[0]

    answer = answer.strip()

    # Убираем странные повторы
    answer = clean_generated_text(
        answer
    )

    if len(answer) < 2:
        return fallback_answer(
            message
        )

    return answer


def clean_generated_text(text):
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Удаляем слишком длинные повторы слов
    words = text.split()

    if len(words) >= 8:
        cleaned = []

        for word in words:
            if (
                len(cleaned) >= 4
                and cleaned[-1].lower()
                == word.lower()
                and cleaned[-2].lower()
                == word.lower()
            ):
                continue

            cleaned.append(word)

        text = " ".join(cleaned)

    # Ограничиваем безумные повторы фраз
    for _ in range(3):
        half = len(text) // 2

        if half > 30:
            a = text[:half]
            b = text[half:]

            if a == b:
                text = a

    return text[:1000]


def fallback_answer(message):
    lower = message.lower()

    if any(
        word in lower
        for word in [
            "привет",
            "здравствуй",
            "хай",
            "hello"
        ]
    ):
        return (
            "Привет! Рассказывай, чем могу помочь."
        )

    if "спасибо" in lower:
        return (
            "Пожалуйста!"
        )

    if any(
        word in lower
        for word in [
            "жирн",
            "сальная",
            "сало"
        ]
    ):
        return (
            "При жирной коже обычно помогает "
            "мягкое очищение утром и вечером "
            "и лёгкий увлажняющий уход."
        )

    if any(
        word in lower
        for word in [
            "сух",
            "сушит"
        ]
    ):
        return (
            "Если кожа сухая, лучше использовать "
            "мягкое очищение и простой "
            "увлажняющий крем."
        )

    if any(
        word in lower
        for word in [
            "волос",
            "шампун"
        ]
    ):
        return (
            "Для волос важно подобрать уход "
            "под кожу головы и тип волос."
        )

    return (
        "Я пока недостаточно хорошо знаю этот "
        "вопрос. Добавь больше похожих примеров "
        "в датасет и переобучи модель."
    )


# ============================================================
# API MODELS
# ============================================================

class ChatRequest(BaseModel):
    user_id: str = "demo_user"
    message: str


class TrainingRequest(BaseModel):
    epochs: int = 5
    learning_rate: float = LEARNING_RATE


class ExampleRequest(BaseModel):
    prompt: str
    response: str


class MemoryRequest(BaseModel):
    user_id: str
    key: str
    value: str


# ============================================================
# CHAT API
# ============================================================

@app.post("/api/chat")
def api_chat(data: ChatRequest):
    user_id = (
        data.user_id.strip()
        or "demo_user"
    )

    message = (
        data.message.strip()
    )

    if not message:
        return JSONResponse(
            {
                "error": "Сообщение пустое"
            },
            status_code=400
        )

    if len(message) > MAX_MESSAGE_LENGTH:
        return JSONResponse(
            {
                "error": (
                    "Сообщение слишком длинное"
                )
            },
            status_code=400
        )

    ensure_user(user_id)

    # Запоминаем факты
    extract_memory(
        user_id,
        message
    )

    # Сохраняем сообщение
    save_message(
        user_id,
        "user",
        message
    )

    # Генерируем ответ
    answer = generate_answer(
        user_id,
        message
    )

    # Сохраняем ответ
    save_message(
        user_id,
        "assistant",
        answer
    )

    return {
        "answer": answer,
        "memory": get_memories(user_id)
    }


# ============================================================
# MEMORY API
# ============================================================

@app.get("/api/memory/{user_id}")
def api_memory(user_id: str):
    return {
        "user_id": user_id,
        "memory": get_memories(user_id)
    }


@app.post("/api/memory")
def api_save_memory(
    data: MemoryRequest
):
    save_memory(
        data.user_id,
        data.key,
        data.value
    )

    return {
        "ok": True
    }


@app.delete("/api/memory/{user_id}")
def api_delete_memory(user_id: str):
    delete_memories(user_id)

    return {
        "ok": True
    }


@app.delete("/api/chat/{user_id}")
def api_clear_chat(user_id: str):
    clear_messages(user_id)

    return {
        "ok": True
    }


# ============================================================
# DATASET API
# ============================================================

@app.get("/api/dataset")
def api_dataset():
    data = load_dataset()

    return {
        "count": len(data),
        "dataset": data
    }


@app.post("/api/add-example")
def api_add_example(
    data: ExampleRequest
):
    prompt = data.prompt.strip()
    response = data.response.strip()

    if not prompt or not response:
        return JSONResponse(
            {
                "error": (
                    "Prompt и response "
                    "не должны быть пустыми"
                )
            },
            status_code=400
        )

    dataset = load_dataset()

    dataset.append(
        {
            "prompt": prompt,
            "response": response
        }
    )

    save_dataset(dataset)

    # Обновляем словарь модели
    create_model()

    return {
        "ok": True,
        "count": len(dataset)
    }


# ============================================================
# TRAIN API
# ============================================================

@app.post("/api/train")
def api_train(
    data: TrainingRequest
):
    epochs = max(
        1,
        min(
            int(data.epochs),
            100
        )
    )

    learning_rate = max(
        0.00001,
        min(
            float(data.learning_rate),
            0.1
        )
    )

    if training_status["running"]:
        return {
            "ok": False,
            "message": "Обучение уже идёт"
        }

    started = start_training(
        epochs,
        learning_rate
    )

    return {
        "ok": started,
        "message": (
            "Обучение запущено"
            if started
            else "Не удалось запустить обучение"
        )
    }


@app.get("/api/train/status")
def api_train_status():
    return training_status


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL is not None,
        "dataset": len(load_dataset()),
        "memory": "SQLite"
    }


# ============================================================
# MAIN CHAT PAGE
# ============================================================

CHAT_HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AI Care</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background:
        radial-gradient(
            circle at top,
            #18254a 0,
            #080b16 40%,
            #05060b 100%
        );
    color: white;
    min-height: 100vh;
}

.app {
    max-width: 900px;
    margin: 0 auto;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.header {
    padding: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,.1);
    backdrop-filter: blur(20px);
}

.logo {
    font-size: 22px;
    font-weight: 700;
}

.status {
    font-size: 12px;
    opacity: .65;
}

.chat {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
}

.message {
    margin-bottom: 14px;
    display: flex;
}

.message.user {
    justify-content: flex-end;
}

.bubble {
    max-width: 80%;
    padding: 13px 16px;
    border-radius: 18px;
    line-height: 1.45;
    white-space: pre-wrap;
}

.assistant .bubble {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.08);
}

.user .bubble {
    background: #3867ff;
}

.bottom {
    padding: 15px;
    display: flex;
    gap: 10px;
    border-top: 1px solid rgba(255,255,255,.1);
    background: rgba(0,0,0,.2);
    backdrop-filter: blur(20px);
}

input {
    flex: 1;
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(255,255,255,.06);
    color: white;
    outline: none;
    border-radius: 15px;
    padding: 14px;
    font-size: 15px;
}

button {
    border: 0;
    border-radius: 15px;
    padding: 0 20px;
    background: #3867ff;
    color: white;
    font-weight: 600;
    cursor: pointer;
}

button:hover {
    opacity: .9;
}

.info {
    padding: 8px 20px 0;
    font-size: 11px;
    opacity: .45;
}

</style>
</head>

<body>

<div class="app">

    <div class="header">
        <div class="logo">
            AI Care
        </div>

        <div class="status">
            собственная нейросеть • память включена
        </div>
    </div>

    <div class="info">
        ID пользователя:
        <span id="userId"></span>
    </div>

    <div class="chat" id="chat"></div>

    <div class="bottom">
        <input
            id="message"
            placeholder="Напиши сообщение..."
            autocomplete="off"
        >

        <button onclick="sendMessage()">
            Отправить
        </button>
    </div>

</div>

<script>

const chat = document.getElementById("chat");
const input = document.getElementById("message");

let userId = localStorage.getItem("ai_care_user_id");

if (!userId) {
    userId =
        "user_" +
        Math.random()
            .toString(36)
            .substring(2, 12);

    localStorage.setItem(
        "ai_care_user_id",
        userId
    );
}

document.getElementById(
    "userId"
).textContent = userId;


function addMessage(
    role,
    text
) {
    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message " + role;

    const bubble =
        document.createElement("div");

    bubble.className =
        "bubble";

    bubble.textContent = text;

    wrapper.appendChild(bubble);

    chat.appendChild(wrapper);

    chat.scrollTop =
        chat.scrollHeight;
}


async function sendMessage() {

    const message =
        input.value.trim();

    if (!message) {
        return;
    }

    addMessage(
        "user",
        message
    );

    input.value = "";

    addMessage(
        "assistant",
        "Думаю..."
    );

    const thinking =
        chat.lastElementChild;

    try {

        const response =
            await fetch(
                "/api/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        user_id: userId,
                        message: message
                    })
                }
            );

        const data =
            await response.json();

        thinking.remove();

        if (data.error) {
            addMessage(
                "assistant",
                "Ошибка: " +
                data.error
            );

            return;
        }

        addMessage(
            "assistant",
            data.answer
        );

    } catch (error) {

        thinking.remove();

        addMessage(
            "assistant",
            "Не удалось связаться с сервером."
        );
    }
}


input.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter"
            && !event.shiftKey
        ) {
            event.preventDefault();

            sendMessage();
        }
    }
);


addMessage(
    "assistant",
    "Привет! Я AI Care. Я могу общаться, запоминать контекст и обучаться на новых примерах."
);

</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return CHAT_HTML


# ============================================================
# ADMIN PANEL
# ============================================================

ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="ru">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>AI Care Admin</title>

<style>

body {
    margin: 0;
    padding: 30px;
    background: #080a10;
    color: white;
    font-family: Arial;
}

.container {
    max-width: 900px;
    margin: auto;
}

.card {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
}

h1 {
    margin-top: 0;
}

input,
textarea,
button {
    width: 100%;
    padding: 13px;
    margin-top: 8px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,.15);
    box-sizing: border-box;
    font-family: inherit;
}

input,
textarea {
    background: #11141e;
    color: white;
}

textarea {
    min-height: 120px;
    resize: vertical;
}

button {
    background: #3867ff;
    color: white;
    cursor: pointer;
    border: none;
}

pre {
    white-space: pre-wrap;
    background: #05060a;
    padding: 15px;
    border-radius: 12px;
    overflow: auto;
}

.status {
    opacity: .75;
}

</style>

</head>

<body>

<div class="container">

<h1>AI Care — Admin</h1>

<div class="card">

<h2>Обучение</h2>

<p class="status" id="status">
Проверка...
</p>

<label>
Количество эпох
</label>

<input
    id="epochs"
    type="number"
    value="5"
    min="1"
    max="100"
>

<label>
Learning rate
</label>

<input
    id="lr"
    type="number"
    value="0.005"
    step="0.001"
>

<button onclick="train()">
Переобучить нейросеть
</button>

</div>


<div class="card">

<h2>Добавить обучающий пример</h2>

<p>
Формат:
<br>
вопрос пользователя → правильный ответ AI
</p>

<input
    id="prompt"
    placeholder="Например: Как ухаживать за кожей?"
>

<textarea
    id="response"
    placeholder="Например: Начни с мягкого очищения и увлажнения..."
></textarea>

<button onclick="addExample()">
Добавить в датасет
</button>

</div>


<div class="card">

<h2>Датасет</h2>

<p id="count">
Загрузка...
</p>

<pre id="dataset">
</pre>

</div>

</div>


<script>

async function loadDataset() {

    const response =
        await fetch(
            "/api/dataset"
        );

    const data =
        await response.json();

    document.getElementById(
        "count"
    ).textContent =
        "Примеров: " +
        data.count;

    document.getElementById(
        "dataset"
    ).textContent =
        JSON.stringify(
            data.dataset,
            null,
            2
        );
}


async function addExample() {

    const prompt =
        document.getElementById(
            "prompt"
        ).value.trim();

    const response =
        document.getElementById(
            "response"
        ).value.trim();

    if (!prompt || !response) {
        alert(
            "Заполни оба поля"
        );

        return;
    }

    const result =
        await fetch(
            "/api/add-example",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    prompt: prompt,
                    response: response
                })
            }
        );

    const data =
        await result.json();

    if (data.error) {
        alert(data.error);
        return;
    }

    document.getElementById(
        "prompt"
    ).value = "";

    document.getElementById(
        "response"
    ).value = "";

    alert(
        "Пример добавлен!"
    );

    loadDataset();
}


async function train() {

    const epochs =
        Number(
            document.getElementById(
                "epochs"
            ).value
        );

    const lr =
        Number(
            document.getElementById(
                "lr"
            ).value
        );

    const response =
        await fetch(
            "/api/train",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    epochs: epochs,
                    learning_rate: lr
                })
            }
        );

    const data =
        await response.json();

    alert(
        data.message
    );
}


async function updateStatus() {

    const response =
        await fetch(
            "/api/train/status"
        );

    const data =
        await response.json();

    document.getElementById(
        "status"
    ).textContent =
        data.message +
        " | эпоха: " +
        data.epoch +
        "/" +
        data.epochs +
        " | loss: " +
        data.loss;
}


loadDataset();

setInterval(
    updateStatus,
    1000
);

updateStatus();

</script>

</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
def admin():
    return ADMIN_HTML


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    ensure_dataset()

    create_model()

    print("=" * 60)
    print(APP_NAME)
    print("Model:", MODEL_FILE)
    print("Dataset:", DATASET_FILE)
    print("Memory:", DB_FILE)
    print("=" * 60)
