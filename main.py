import os
import re
import json
import time
import base64
import threading
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from supabase import create_client, Client


# ============================================================
# AI CARE V5
# СОБСТВЕННАЯ RNN + ПАМЯТЬ + RETRIEVAL + CHECKPOINTS
# SUPABASE DATABASE
# БЕЗ SUPABASE STORAGE
# ============================================================

APP_NAME = "AI Care v5"

BASE_DIR = Path(__file__).resolve().parent

LOCAL_MODEL = BASE_DIR / "model.npz"
LOCAL_DATASET = BASE_DIR / "dataset.json"

HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "96"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
GRADIENT_CLIP = float(os.getenv("GRADIENT_CLIP", "5.0"))

MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "8"))
MAX_MESSAGE_LENGTH = 1000
MAX_RESPONSE_LENGTH = 100

DEFAULT_TEMPERATURE = 0.8
RETRIEVAL_THRESHOLD = 0.55


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY,
    )


def require_supabase():
    if supabase is None:
        raise RuntimeError(
            "SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY не настроены."
        )


# ============================================================
# APP
# ============================================================

app = FastAPI(title=APP_NAME)


# ============================================================
# LOCKS
# ============================================================

model_lock = threading.RLock()
db_lock = threading.RLock()


# ============================================================
# TOKENIZER
# ============================================================

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]


def tokenize(text: str):
    text = str(text).lower().strip()

    return re.findall(
        r"[a-zа-яё0-9]+|[.,!?;:()\-]",
        text,
        flags=re.IGNORECASE,
    )


def detokenize(tokens):
    result = ""

    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue

        if not result:
            result = token
            continue

        if token in ".,!?;:)":
            result += token
        elif token == "-":
            result += " -"
        else:
            result += " " + token

    return result.strip()


# ============================================================
# DATASET
# ============================================================

DEFAULT_DATASET = [
    {
        "prompt": "привет",
        "response": "Привет! Я AI Care. Чем могу помочь?",
        "category": "greeting",
    },
    {
        "prompt": "здравствуй",
        "response": "Привет! Рад тебя видеть.",
        "category": "greeting",
    },
    {
        "prompt": "как дела",
        "response": "У меня всё хорошо. Готов общаться и помогать.",
        "category": "greeting",
    },
    {
        "prompt": "что ты умеешь",
        "response": "Я умею отвечать на вопросы, поддерживать разговор, использовать память и учиться на подготовленных примерах.",
        "category": "capabilities",
    },
    {
        "prompt": "кто ты",
        "response": "Я AI Care — экспериментальная нейросеть с собственной RNN, памятью и системой обучения.",
        "category": "capabilities",
    },
    {
        "prompt": "что такое нейросеть",
        "response": "Нейросеть — это математическая модель, которая учится находить закономерности в данных.",
        "category": "education",
    },
    {
        "prompt": "что такое rnn",
        "response": "RNN — рекуррентная нейронная сеть, которая может учитывать последовательность информации.",
        "category": "education",
    },
    {
        "prompt": "что такое python",
        "response": "Python — популярный язык программирования, который используют для сайтов, автоматизации, анализа данных и искусственного интеллекта.",
        "category": "education",
    },
    {
        "prompt": "мне скучно",
        "response": "Можно придумать небольшую игру, изучить что-нибудь новое или заняться своим проектом.",
        "category": "conversation",
    },
    {
        "prompt": "я устал",
        "response": "Если устал, полезно сделать небольшой перерыв, попить воды и немного отдохнуть.",
        "category": "wellbeing",
    },
    {
        "prompt": "я не понимаю код",
        "response": "Давай разберём код по частям: сначала посмотрим, что делает каждая функция, а затем разберём логику программы.",
        "category": "programming",
    },
    {
        "prompt": "помоги с программированием",
        "response": "Конечно. Покажи код или опиши задачу, и мы разберём её по шагам.",
        "category": "programming",
    },
    {
        "prompt": "как изучать программирование",
        "response": "Лучше всего регулярно писать небольшие программы и постепенно переходить к более сложным проектам.",
        "category": "programming",
    },
    {
        "prompt": "как улучшить сон",
        "response": "Старайся придерживаться стабильного режима сна и уменьшать использование телефона перед сном.",
        "category": "wellbeing",
    },
    {
        "prompt": "что делать если не могу уснуть",
        "response": "Попробуй спокойно полежать без телефона, приглушить свет и дать организму время расслабиться.",
        "category": "wellbeing",
    },
    {
        "prompt": "как ухаживать за кожей",
        "response": "Обычно полезно мягко очищать кожу и использовать подходящий увлажняющий крем. Если проблема сильная или не проходит, лучше обратиться к врачу.",
        "category": "skincare",
    },
    {
        "prompt": "у меня чёрные точки",
        "response": "Чёрные точки часто связаны с закупоркой пор. Не стоит агрессивно выдавливать их или травмировать кожу.",
        "category": "skincare",
    },
    {
        "prompt": "как убрать жирность кожи",
        "response": "Лучше использовать мягкое очищение и не пересушивать кожу слишком агрессивными средствами.",
        "category": "skincare",
    },
    {
        "prompt": "расскажи шутку",
        "response": "Почему программист любит тёмную тему? Потому что свет притягивает баги.",
        "category": "fun",
    },
    {
        "prompt": "расскажи факт",
        "response": "У осьминога три сердца.",
        "category": "facts",
    },
    {
        "prompt": "что можно изучить",
        "response": "Можно изучить Python, веб-разработку, базы данных, алгоритмы или основы машинного обучения.",
        "category": "education",
    },
    {
        "prompt": "как создать сайт",
        "response": "Для простого сайта достаточно HTML, CSS и JavaScript. Затем его можно разместить на хостинге.",
        "category": "programming",
    },
    {
        "prompt": "как создать бота",
        "response": "Сначала нужно выбрать платформу, затем создать обработчики сообщений и подключить базу данных или другие нужные сервисы.",
        "category": "programming",
    },
    {
        "prompt": "спасибо",
        "response": "Пожалуйста! Рад помочь.",
        "category": "conversation",
    },
]


dataset = []


def normalize_example(item):
    return {
        "prompt": str(item.get("prompt", "")).strip(),
        "response": str(item.get("response", "")).strip(),
        "category": str(item.get("category", "general")).strip() or "general",
    }


def deduplicate_dataset(items):
    result = []
    seen = set()

    for raw in items:
        item = normalize_example(raw)

        if not item["prompt"] or not item["response"]:
            continue

        key = (
            item["prompt"].lower(),
            item["response"].lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def load_dataset():
    global dataset

    if supabase is not None:
        try:
            response = (
                supabase
                .table("ai_dataset")
                .select("id,prompt,response,category")
                .order("id")
                .execute()
            )

            rows = response.data or []

            if rows:
                dataset = deduplicate_dataset(rows)
                save_local_dataset()
                return

        except Exception as exc:
            print("Supabase dataset load error:", exc)

    if LOCAL_DATASET.exists():
        try:
            dataset = deduplicate_dataset(
                json.loads(
                    LOCAL_DATASET.read_text(
                        encoding="utf-8"
                    )
                )
            )

            if dataset:
                return

        except Exception as exc:
            print("Local dataset load error:", exc)

    dataset = deduplicate_dataset(DEFAULT_DATASET)
    save_local_dataset()


def save_local_dataset():
    try:
        LOCAL_DATASET.write_text(
            json.dumps(
                dataset,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print("Local dataset save error:", exc)


def save_dataset_to_supabase():
    require_supabase()

    with db_lock:
        rows = []

        for item in dataset:
            rows.append(
                {
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "category": item.get("category", "general"),
                }
            )

        if rows:
            supabase.table("ai_dataset").upsert(
                rows,
                on_conflict="prompt,response",
            ).execute()


# ============================================================
# VOCABULARY
# ============================================================

vocab = {}
id_to_token = []


def build_vocab(items):
    global vocab, id_to_token

    tokens = []

    for special in SPECIAL_TOKENS:
        if special not in tokens:
            tokens.append(special)

    for item in items:
        tokens.extend(tokenize(item["prompt"]))
        tokens.extend(tokenize(item["response"]))

    unique = []

    for token in tokens:
        if token not in unique:
            unique.append(token)

    id_to_token = unique
    vocab = {
        token: index
        for index, token in enumerate(id_to_token)
    }


def token_id(token):
    return vocab.get(token, vocab["<UNK>"])


# ============================================================
# MODEL
# ============================================================

class RNNModel:
    def __init__(
        self,
        hidden_size,
        vocab_size,
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        scale = 0.05

        self.Wxh = (
            np.random.randn(
                hidden_size,
                vocab_size,
            )
            * scale
        )

        self.Whh = (
            np.random.randn(
                hidden_size,
                hidden_size,
            )
            * scale
        )

        self.Why = (
            np.random.randn(
                vocab_size,
                hidden_size,
            )
            * scale
        )

        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(vocab_size)

    def copy(self):
        model = RNNModel(
            self.hidden_size,
            self.vocab_size,
        )

        model.Wxh = self.Wxh.copy()
        model.Whh = self.Whh.copy()
        model.Why = self.Why.copy()
        model.bh = self.bh.copy()
        model.by = self.by.copy()

        return model

    def forward(
        self,
        inputs,
        targets=None,
        h0=None,
    ):
        if h0 is None:
            h = np.zeros(self.hidden_size)
        else:
            h = h0.copy()

        hs = [h]
        ps = []
        loss = 0.0

        for i, input_id in enumerate(inputs):
            x = np.zeros(self.vocab_size)
            x[input_id] = 1.0

            h = np.tanh(
                self.Wxh @ x
                + self.Whh @ h
                + self.bh
            )

            logits = self.Why @ h + self.by

            logits -= np.max(logits)

            exp_logits = np.exp(logits)
            probs = exp_logits / (
                np.sum(exp_logits) + 1e-12
            )

            ps.append(probs)
            hs.append(h)

            if targets is not None:
                target = targets[i]

                loss -= np.log(
                    probs[target] + 1e-12
                )

        return loss, hs, ps, h

    def train_example(
        self,
        inputs,
        targets,
        learning_rate,
    ):
        loss, hs, ps, _ = self.forward(
            inputs,
            targets,
        )

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros(self.hidden_size)

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0

            dWhy += np.outer(
                dy,
                hs[t + 1],
            )

            dby += dy

            dh = (
                self.Why.T @ dy
                + dh_next
            )

            dh_raw = (
                1.0 - hs[t + 1] ** 2
            ) * dh

            dbh += dh_raw

            dWhh += np.outer(
                dh_raw,
                hs[t],
            )

            x = np.zeros(self.vocab_size)
            x[inputs[t]] = 1.0

            dWxh += np.outer(
                dh_raw,
                x,
            )

            dh_next = (
                self.Whh.T @ dh_raw
            )

        for grad in (
            dWxh,
            dWhh,
            dWhy,
            dbh,
            dby,
        ):
            np.clip(
                grad,
                -GRADIENT_CLIP,
                GRADIENT_CLIP,
                out=grad,
            )

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        return float(loss)


# ============================================================
# MODEL VOCAB MIGRATION
# ============================================================

def expand_model_vocabulary(
    old_model,
    old_vocab,
):
    new_size = len(vocab)

    if old_model is None:
        return RNNModel(
            HIDDEN_SIZE,
            new_size,
        )

    if old_model.hidden_size != HIDDEN_SIZE:
        return RNNModel(
            HIDDEN_SIZE,
            new_size,
        )

    if old_model.vocab_size == new_size:
        return old_model

    new_model = RNNModel(
        HIDDEN_SIZE,
        new_size,
    )

    for token, old_id in old_vocab.items():
        if token not in vocab:
            continue

        new_id = vocab[token]

        if (
            old_id < old_model.Wxh.shape[1]
            and new_id < new_model.Wxh.shape[1]
        ):
            new_model.Wxh[:, new_id] = (
                old_model.Wxh[:, old_id]
            )

        if (
            old_id < old_model.Why.shape[0]
            and new_id < new_model.Why.shape[0]
        ):
            new_model.Why[new_id, :] = (
                old_model.Why[old_id, :]
            )

        if (
            old_id < old_model.by.shape[0]
            and new_id < new_model.by.shape[0]
        ):
            new_model.by[new_id] = (
                old_model.by[old_id]
            )

    new_model.Whh = old_model.Whh.copy()
    new_model.bh = old_model.bh.copy()

    return new_model


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_model(model):
    old_vocab_json = json.dumps(
        vocab,
        ensure_ascii=False,
    )

    with tempfile.NamedTemporaryFile(
        suffix=".npz",
        delete=False,
    ) as tmp:
        path = tmp.name

    try:
        np.savez_compressed(
            path,
            Wxh=model.Wxh,
            Whh=model.Whh,
            Why=model.Why,
            bh=model.bh,
            by=model.by,
            hidden_size=np.array(
                [model.hidden_size],
                dtype=np.int64,
            ),
            vocab=np.array(
                [old_vocab_json],
                dtype=object,
            ),
        )

        data = Path(path).read_bytes()

        return data

    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass


def deserialize_model(data):
    with tempfile.NamedTemporaryFile(
        suffix=".npz",
        delete=False,
    ) as tmp:
        path = tmp.name
        tmp.write(data)

    try:
        loaded = np.load(
            path,
            allow_pickle=True,
        )

        saved_hidden = int(
            loaded["hidden_size"][0]
        )

        saved_vocab = json.loads(
            str(loaded["vocab"][0])
        )

        saved_vocab = {
            str(k): int(v)
            for k, v in saved_vocab.items()
        }

        model = RNNModel(
            saved_hidden,
            len(saved_vocab),
        )

        model.Wxh = loaded["Wxh"]
        model.Whh = loaded["Whh"]
        model.Why = loaded["Why"]
        model.bh = loaded["bh"]
        model.by = loaded["by"]

        return model, saved_vocab

    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass


# ============================================================
# MODEL STATE
# ============================================================

model = None

training = {
    "running": False,
    "epoch": 0,
    "target_epoch": 0,
    "loss": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def load_local_model():
    if not LOCAL_MODEL.exists():
        return None

    try:
        data = LOCAL_MODEL.read_bytes()
        return deserialize_model(data)
    except Exception as exc:
        print("Local model load error:", exc)
        return None


def save_local_model(current_model):
    try:
        data = serialize_model(current_model)
        LOCAL_MODEL.write_bytes(data)
    except Exception as exc:
        print("Local model save error:", exc)


def save_model_to_supabase(
    current_model,
    trained_epochs,
    loss,
):
    require_supabase()

    data = serialize_model(current_model)

    encoded = base64.b64encode(data).decode(
        "ascii"
    )

    payload = {
        "id": 1,
        "trained_epochs": int(trained_epochs),
        "loss": float(loss),
        "hidden_size": int(
            current_model.hidden_size
        ),
        "vocab": vocab,
        "model_blob": encoded,
        "updated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
    }

    with db_lock:
        supabase.table(
            "ai_model_state"
        ).upsert(
            payload,
            on_conflict="id",
        ).execute()

    save_local_model(current_model)


def load_model_from_supabase():
    if supabase is None:
        return None, {}, 0, None

    try:
        response = (
            supabase
            .table("ai_model_state")
            .select(
                "trained_epochs,loss,hidden_size,"
                "vocab,model_blob"
            )
            .eq("id", 1)
            .limit(1)
            .execute()
        )

        rows = response.data or []

        if not rows:
            return None, {}, 0, None

        row = rows[0]

        blob = row.get("model_blob")

        if not blob:
            return None, {}, 0, None

        data = base64.b64decode(blob)

        loaded_model, saved_vocab = (
            deserialize_model(data)
        )

        return (
            loaded_model,
            saved_vocab,
            int(row.get("trained_epochs") or 0),
            row.get("loss"),
        )

    except Exception as exc:
        print("Supabase model load error:", exc)

        return None, {}, 0, None


trained_epochs = 0
last_loss = None


def initialize_model():
    global model
    global vocab
    global id_to_token
    global trained_epochs
    global last_loss

    build_vocab(dataset)

    loaded_model = None
    saved_vocab = {}
    saved_epochs = 0
    saved_loss = None

    if supabase is not None:
        (
            loaded_model,
            saved_vocab,
            saved_epochs,
            saved_loss,
        ) = load_model_from_supabase()

    if loaded_model is None:
        local = load_local_model()

        if local is not None:
            (
                loaded_model,
                saved_vocab,
            ) = local

    if loaded_model is not None:
        old_vocab = saved_vocab.copy()

        loaded_model = expand_model_vocabulary(
            loaded_model,
            old_vocab,
        )

        model = loaded_model

    else:
        model = RNNModel(
            HIDDEN_SIZE,
            len(vocab),
        )

    trained_epochs = saved_epochs
    last_loss = saved_loss

    save_local_model(model)


# ============================================================
# MEMORY
# ============================================================

def get_memories(user_id):
    if supabase is None:
        return []

    try:
        response = (
            supabase
            .table("ai_memories")
            .select("memory_key,memory_value")
            .eq("user_id", user_id)
            .order("id")
            .execute()
        )

        return response.data or []

    except Exception as exc:
        print("Memory load error:", exc)
        return []


def save_memory(
    user_id,
    memory_key,
    memory_value,
):
    require_supabase()

    supabase.table(
        "ai_memories"
    ).upsert(
        {
            "user_id": user_id,
            "memory_key": memory_key,
            "memory_value": memory_value,
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        },
        on_conflict="user_id,memory_key",
    ).execute()


def extract_memory(
    user_id,
    text,
):
    lowered = text.lower().strip()

    match = re.search(
        r"(?:меня зовут|моё имя|мое имя)\s+([а-яёa-z0-9_-]{2,30})",
        lowered,
    )

    if match:
        save_memory(
            user_id,
            "name",
            match.group(1),
        )

    match = re.search(
        r"(?:я люблю|мне нравится|моя любимая игра|мой любимый фильм)\s+(.{2,150})",
        lowered,
    )

    if match:
        save_memory(
            user_id,
            "preference",
            match.group(1).strip(),
        )

    if (
        lowered.startswith("запомни ")
        or lowered.startswith("запиши ")
        or lowered.startswith("сохрани ")
    ):
        value = re.sub(
            r"^(запомни|запиши|сохрани)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if value:
            save_memory(
                user_id,
                "user_note",
                value[:500],
            )


# ============================================================
# CHAT HISTORY
# ============================================================

def save_message(
    user_id,
    role,
    text,
):
    if supabase is None:
        return

    try:
        supabase.table(
            "ai_messages"
        ).insert(
            {
                "user_id": user_id,
                "role": role,
                "text": text[:MAX_MESSAGE_LENGTH],
            }
        ).execute()

    except Exception as exc:
        print("Message save error:", exc)


def get_history(user_id):
    if supabase is None:
        return []

    try:
        response = (
            supabase
            .table("ai_messages")
            .select("role,text,created_at")
            .eq("user_id", user_id)
            .order("id", desc=False)
            .limit(MAX_CONTEXT_MESSAGES * 2)
            .execute()
        )

        return response.data or []

    except Exception as exc:
        print("History load error:", exc)
        return []


# ============================================================
# RETRIEVAL
# ============================================================

def word_set(text):
    return set(
        token
        for token in tokenize(text)
        if token not in SPECIAL_TOKENS
    )


def similarity(a, b):
    a_set = word_set(a)
    b_set = word_set(b)

    if not a_set or not b_set:
        return 0.0

    intersection = len(
        a_set.intersection(b_set)
    )

    union = len(
        a_set.union(b_set)
    )

    if union == 0:
        return 0.0

    jaccard = intersection / union

    contains_bonus = 0.0

    if (
        a.lower() in b.lower()
        or b.lower() in a.lower()
    ):
        contains_bonus = 0.25

    return min(
        1.0,
        jaccard + contains_bonus,
    )


def retrieve_response(text):
    best_item = None
    best_score = 0.0

    for item in dataset:
        score = similarity(
            text,
            item["prompt"],
        )

        if score > best_score:
            best_score = score
            best_item = item

    return best_item, best_score


# ============================================================
# TRAINING DATA
# ============================================================

def make_training_sequence(item):
    prompt_tokens = tokenize(
        item["prompt"]
    )

    response_tokens = tokenize(
        item["response"]
    )

    input_tokens = (
        prompt_tokens
        + ["<BOS>"]
        + response_tokens
    )

    target_tokens = (
        prompt_tokens[1:]
        + ["<BOS>"]
        + response_tokens[1:]
        + ["<EOS>"]
    )

    if len(target_tokens) < len(
        input_tokens
    ):
        target_tokens.append("<EOS>")

    if len(target_tokens) > len(
        input_tokens
    ):
        target_tokens = target_tokens[
            :len(input_tokens)
        ]

    inputs = [
        token_id(token)
        for token in input_tokens
    ]

    targets = [
        token_id(token)
        for token in target_tokens
    ]

    return inputs, targets


# ============================================================
# TRAINING
# ============================================================

def train_worker(additional_epochs):
    global model
    global trained_epochs
    global last_loss

    training["running"] = True
    training["error"] = None
    training["started_at"] = time.time()
    training["finished_at"] = None
    training["target_epoch"] = (
        trained_epochs + additional_epochs
    )

    try:
        for _ in range(additional_epochs):
            with model_lock:
                current_model = model

            total_loss = 0.0
            count = 0

            shuffled = list(dataset)
            np.random.shuffle(shuffled)

            for item in shuffled:
                inputs, targets = (
                    make_training_sequence(item)
                )

                if not inputs:
                    continue

                with model_lock:
                    loss = current_model.train_example(
                        inputs,
                        targets,
                        LEARNING_RATE,
                    )

                total_loss += loss
                count += 1

            epoch_loss = (
                total_loss / max(count, 1)
            )

            trained_epochs += 1
            last_loss = epoch_loss

            with model_lock:
                save_model_to_supabase(
                    current_model,
                    trained_epochs,
                    epoch_loss,
                )

            training["epoch"] = trained_epochs
            training["loss"] = epoch_loss

            print(
                f"Epoch {trained_epochs}: "
                f"loss={epoch_loss:.6f}"
            )

    except Exception as exc:
        training["error"] = str(exc)
        print("Training error:", exc)

    finally:
        training["running"] = False
        training["finished_at"] = time.time()


def start_training(epochs):
    if training["running"]:
        raise HTTPException(
            status_code=409,
            detail="Обучение уже запущено.",
        )

    if epochs < 1 or epochs > 10000:
        raise HTTPException(
            status_code=400,
            detail="Количество эпох должно быть от 1 до 10000.",
        )

    if not dataset:
        raise HTTPException(
            status_code=400,
            detail="Датасет пуст.",
        )

    thread = threading.Thread(
        target=train_worker,
        args=(epochs,),
        daemon=True,
    )

    thread.start()


# ============================================================
# GENERATION
# ============================================================

def generate_response(
    text,
    user_id,
    temperature=DEFAULT_TEMPERATURE,
):
    global model

    retrieved, score = retrieve_response(
        text
    )

    # Точное/очень похожее совпадение.
    if (
        retrieved is not None
        and score >= RETRIEVAL_THRESHOLD
    ):
        return retrieved["response"]

    memories = get_memories(user_id)
    history = get_history(user_id)

    context_parts = []

    for memory in memories:
        context_parts.append(
            f"{memory['memory_key']}: "
            f"{memory['memory_value']}"
        )

    for message in history[-MAX_CONTEXT_MESSAGES:]:
        context_parts.append(
            message["text"]
        )

    context_parts.append(text)

    context = " ".join(
        context_parts
    )

    context_tokens = tokenize(context)

    if not context_tokens:
        return "Я пока не знаю, что ответить."

    with model_lock:
        current_model = model.copy()

    hidden = np.zeros(
        current_model.hidden_size
    )

    # Прогоняем контекст.
    for token in context_tokens:
        input_id = token_id(token)

        x = np.zeros(
            current_model.vocab_size
        )

        x[input_id] = 1.0

        hidden = np.tanh(
            current_model.Wxh @ x
            + current_model.Whh @ hidden
            + current_model.bh
        )

    # BOS запускает генерацию ответа.
    current_id = token_id("<BOS>")

    generated = []
    seen = {}

    for _ in range(MAX_RESPONSE_LENGTH):
        x = np.zeros(
            current_model.vocab_size
        )

        x[current_id] = 1.0

        hidden = np.tanh(
            current_model.Wxh @ x
            + current_model.Whh @ hidden
            + current_model.bh
        )

        logits = (
            current_model.Why @ hidden
            + current_model.by
        )

        temperature = max(
            0.2,
            min(temperature, 2.0),
        )

        logits = logits / temperature
        logits -= np.max(logits)

        probs = np.exp(logits)

        # PAD и BOS не должны генерироваться.
        for special in (
            "<PAD>",
            "<BOS>",
        ):
            idx = vocab.get(special)

            if idx is not None:
                probs[idx] = 0.0

        # Слегка уменьшаем вероятность UNK.
        unk_idx = vocab.get("<UNK>")

        if unk_idx is not None:
            probs[unk_idx] *= 0.15

        # EOS разрешён.
        eos_idx = vocab.get("<EOS>")

        total = probs.sum()

        if total <= 0:
            break

        probs /= total

        next_id = np.random.choice(
            len(probs),
            p=probs,
        )

        next_token = id_to_token[next_id]

        if next_token == "<EOS>":
            break

        if next_token in (
            "<PAD>",
            "<BOS>",
        ):
            break

        seen[next_token] = (
            seen.get(next_token, 0) + 1
        )

        # Защита от бесконечного повторения.
        if seen[next_token] >= 4:
            break

        generated.append(next_token)
        current_id = next_id

    answer = detokenize(
        generated
    )

    if not answer:
        return (
            "Я пока не уверен в ответе. "
            "Добавь похожий пример в датасет, "
            "и я смогу научиться отвечать лучше."
        )

    return answer[:MAX_MESSAGE_LENGTH]


# ============================================================
# ADMIN
# ============================================================

def check_admin(
    x_admin_token: Optional[str],
):
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=500,
            detail=(
                "ADMIN_TOKEN не настроен "
                "в переменных окружения Render."
            ),
        )

    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Неверный admin token.",
        )


# ============================================================
# API MODELS
# ============================================================

class ChatRequest(BaseModel):
    user_id: str
    text: str
    temperature: float = DEFAULT_TEMPERATURE


class MemoryRequest(BaseModel):
    user_id: str
    key: str
    value: str


class DatasetRequest(BaseModel):
    prompt: str
    response: str
    category: str = "general"


class TrainRequest(BaseModel):
    epochs: int


class ApproveRequest(BaseModel):
    user_id: str
    prompt: str
    response: str
    category: str = "approved"


# ============================================================
# CHAT API
# ============================================================

@app.post("/api/chat")
def chat(request: ChatRequest):
    user_id = request.user_id.strip()[:100]
    text = request.text.strip()

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id обязателен.",
        )

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Сообщение пустое.",
        )

    if len(text) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Сообщение слишком длинное.",
        )

    extract_memory(
        user_id,
        text,
    )

    answer = generate_response(
        text,
        user_id,
        request.temperature,
    )

    save_message(
        user_id,
        "user",
        text,
    )

    save_message(
        user_id,
        "assistant",
        answer,
    )

    return {
        "answer": answer,
        "trained_epochs": trained_epochs,
        "retrieval_available": True,
    }


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/history/{user_id}")
def history(user_id: str):
    return {
        "messages": get_history(
            user_id[:100]
        )
    }


# ============================================================
# MEMORY
# ============================================================

@app.get("/api/memory/{user_id}")
def memory(user_id: str):
    return {
        "memories": get_memories(
            user_id[:100]
        )
    }


@app.post("/api/memory")
def create_memory(request: MemoryRequest):
    if not request.key.strip():
        raise HTTPException(
            status_code=400,
            detail="Ключ памяти пуст.",
        )

    save_memory(
        request.user_id[:100],
        request.key[:100],
        request.value[:500],
    )

    return {
        "ok": True
    }


@app.delete("/api/memory/{user_id}/{key}")
def delete_memory(
    user_id: str,
    key: str,
):
    require_supabase()

    supabase.table(
        "ai_memories"
    ).delete().eq(
        "user_id",
        user_id[:100],
    ).eq(
        "memory_key",
        key[:100],
    ).execute()

    return {
        "ok": True
    }


# ============================================================
# DELETE CHAT
# ============================================================

@app.delete("/api/chat/{user_id}")
def delete_chat(user_id: str):
    require_supabase()

    supabase.table(
        "ai_messages"
    ).delete().eq(
        "user_id",
        user_id[:100],
    ).execute()

    return {
        "ok": True
    }


# ============================================================
# ADMIN DATASET
# ============================================================

@app.get("/api/admin/dataset")
def admin_dataset(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):
    check_admin(x_admin_token)

    return {
        "dataset": dataset,
        "count": len(dataset),
    }


@app.post("/api/admin/dataset")
def admin_add_dataset(
    request: DatasetRequest,
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):
    check_admin(x_admin_token)

    prompt = request.prompt.strip()
    response = request.response.strip()
    category = request.category.strip() or "general"

    if not prompt or not response:
        raise HTTPException(
            status_code=400,
            detail="Prompt и response обязательны.",
        )

    if len(prompt) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Prompt слишком длинный.",
        )

    if len(response) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Response слишком длинный.",
        )

    new_item = {
        "prompt": prompt,
        "response": response,
        "category": category,
    }

    key = (
        prompt.lower(),
        response.lower(),
    )

    for item in dataset:
        if (
            item["prompt"].lower(),
            item["response"].lower(),
        ) == key:
            return {
                "ok": True,
                "message": "Такой пример уже есть.",
                "dataset_count": len(dataset),
            }

    dataset.append(new_item)
    dataset[:] = deduplicate_dataset(
        dataset
    )

    save_local_dataset()

    if supabase is not None:
        try:
            supabase.table(
                "ai_dataset"
            ).upsert(
                new_item,
                on_conflict="prompt,response",
            ).execute()
        except Exception as exc:
            print(
                "Dataset Supabase save error:",
                exc,
            )

    # Очень важно:
    # не создаём модель заново.
    # Просто расширяем словарь.
    global vocab
    global id_to_token
    global model

    old_vocab = vocab.copy()

    build_vocab(dataset)

    with model_lock:
        model = expand_model_vocabulary(
            model,
            old_vocab,
        )

    save_local_model(model)

    return {
        "ok": True,
        "dataset_count": len(dataset),
        "vocab_size": len(vocab),
    }


# ============================================================
# APPROVED TRAINING EXAMPLES
# ============================================================

@app.post("/api/admin/approve")
def approve_example(
    request: ApproveRequest,
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):
    check_admin(x_admin_token)

    item = {
        "prompt": request.prompt.strip(),
        "response": request.response.strip(),
        "category": request.category.strip()
        or "approved",
    }

    if not item["prompt"] or not item["response"]:
        raise HTTPException(
            status_code=400,
            detail="Пустой пример.",
        )

    # В дальнейшем здесь можно сделать
    # отдельную очередь одобрения.
    dataset.append(item)

    dataset[:] = deduplicate_dataset(
        dataset
    )

    save_local_dataset()

    if supabase is not None:
        supabase.table(
            "ai_dataset"
        ).upsert(
            item,
            on_conflict="prompt,response",
        ).execute()

    old_vocab = vocab.copy()

    build_vocab(dataset)

    with model_lock:
        global model
        model = expand_model_vocabulary(
            model,
            old_vocab,
        )

    return {
        "ok": True,
        "message": "Пример добавлен.",
        "dataset_count": len(dataset),
    }


# ============================================================
# TRAIN
# ============================================================

@app.post("/api/admin/train")
def admin_train(
    request: TrainRequest,
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):
    check_admin(x_admin_token)

    start_training(
        request.epochs
    )

    return {
        "ok": True,
        "message": (
            f"Запущено +{request.epochs} эпох."
        ),
        "current_epoch": trained_epochs,
    }


@app.get("/api/admin/train/status")
def admin_train_status(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):
    check_admin(x_admin_token)

    return {
        "running": training["running"],
        "epoch": trained_epochs,
        "target_epoch": training[
            "target_epoch"
        ],
        "loss": last_loss,
        "error": training["error"],
        "dataset_count": len(dataset),
        "vocab_size": len(vocab),
        "hidden_size": HIDDEN_SIZE,
    }


# ============================================================
# TEST / EVALUATION
# ============================================================

@app.get("/api/admin/evaluate")
def evaluate(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):
    check_admin(x_admin_token)

    tests = [
        "привет",
        "что ты умеешь",
        "что такое нейросеть",
        "помоги с программированием",
        "мне скучно",
        "как создать сайт",
        "расскажи факт",
    ]

    results = []

    for question in tests:
        answer = generate_response(
            question,
            "evaluation_user",
            0.7,
        )

        retrieved, score = retrieve_response(
            question
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "retrieval_score": round(
                    score,
                    3,
                ),
                "expected_available": (
                    retrieved is not None
                ),
            }
        )

    return {
        "results": results
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "supabase": supabase is not None,
        "dataset_count": len(dataset),
        "vocab_size": len(vocab),
        "trained_epochs": trained_epochs,
        "training": training["running"],
    }


# ============================================================
# ADMIN HTML
# ============================================================

ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Care v5 — Admin</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #080b12;
    color: #eef2ff;
    font-family: Arial, sans-serif;
}

.container {
    width: min(1100px, 94%);
    margin: 30px auto;
}

.card {
    background: #101522;
    border: 1px solid #20283b;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
}

h1, h2 {
    margin-top: 0;
}

input, textarea, select {
    width: 100%;
    background: #080c15;
    color: white;
    border: 1px solid #2a344c;
    border-radius: 12px;
    padding: 12px;
    margin: 6px 0 12px;
}

textarea {
    min-height: 100px;
    resize: vertical;
}

button {
    background: #4d7cff;
    color: white;
    border: 0;
    padding: 12px 16px;
    border-radius: 12px;
    cursor: pointer;
    font-weight: bold;
    margin-right: 8px;
}

button:hover {
    opacity: .9;
}

.stat {
    display: inline-block;
    padding: 10px 14px;
    background: #080c15;
    border-radius: 12px;
    margin: 4px;
}

.example {
    background: #080c15;
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
}

.small {
    color: #8f9ab3;
    font-size: 13px;
}

pre {
    white-space: pre-wrap;
    word-break: break-word;
}
</style>
</head>

<body>

<div class="container">

<div class="card">
<h1>🧠 AI Care v5</h1>

<input
    id="token"
    type="password"
    placeholder="ADMIN_TOKEN"
/>

<button onclick="saveToken()">
Сохранить токен
</button>

<div id="stats"></div>
</div>


<div class="card">
<h2>🚀 Обучение</h2>

<label>Дополнительные эпохи</label>

<input
    id="epochs"
    type="number"
    value="10"
    min="1"
    max="10000"
/>

<button onclick="train()">
Начать обучение
</button>

<button onclick="status()">
Обновить статус
</button>

<pre id="trainingStatus"></pre>

</div>


<div class="card">

<h2>➕ Добавить обучающий пример</h2>

<input
    id="prompt"
    placeholder="Вопрос пользователя"
/>

<textarea
    id="response"
    placeholder="Правильный ответ AI"
></textarea>

<input
    id="category"
    value="general"
    placeholder="Категория"
/>

<button onclick="addExample()">
Добавить
</button>

</div>


<div class="card">

<h2>🧪 Проверка качества</h2>

<button onclick="evaluateAI()">
Проверить ответы
</button>

<pre id="evaluation"></pre>

</div>


<div class="card">

<h2>📚 Датасет</h2>

<button onclick="loadDataset()">
Обновить
</button>

<div id="dataset"></div>

</div>

</div>


<script>

function getToken() {
    return sessionStorage.getItem(
        "ai_admin_token"
    ) || "";
}


function saveToken() {
    const token =
        document.getElementById("token").value;

    sessionStorage.setItem(
        "ai_admin_token",
        token
    );

    alert("Токен сохранён для этой вкладки.");
    status();
    loadDataset();
}


async function api(
    url,
    options = {}
) {
    options.headers = {
        ...(options.headers || {}),
        "X-Admin-Token": getToken(),
        "Content-Type": "application/json"
    };

    const response =
        await fetch(url, options);

    const data =
        await response.json();

    if (!response.ok) {
        throw new Error(
            data.detail || "Ошибка"
        );
    }

    return data;
}


async function status() {
    try {
        const data =
            await api(
                "/api/admin/train/status"
            );

        document.getElementById(
            "trainingStatus"
        ).textContent =
            JSON.stringify(
                data,
                null,
                2
            );

        document.getElementById(
            "stats"
        ).innerHTML = `
            <div class="stat">
                Эпох: ${data.epoch}
            </div>
            <div class="stat">
                Датасет: ${data.dataset_count}
            </div>
            <div class="stat">
                Словарь: ${data.vocab_size}
            </div>
            <div class="stat">
                Loss: ${data.loss ?? "-"}
            </div>
        `;

    } catch (e) {
        document.getElementById(
            "trainingStatus"
        ).textContent = e.message;
    }
}


async function train() {
    const epochs =
        Number(
            document.getElementById(
                "epochs"
            ).value
        );

    try {
        const data =
            await api(
                "/api/admin/train",
                {
                    method: "POST",
                    body: JSON.stringify({
                        epochs
                    })
                }
            );

        alert(data.message);
        status();

    } catch (e) {
        alert(e.message);
    }
}


async function addExample() {

    const prompt =
        document.getElementById(
            "prompt"
        ).value;

    const response =
        document.getElementById(
            "response"
        ).value;

    const category =
        document.getElementById(
            "category"
        ).value;

    try {
        const data =
            await api(
                "/api/admin/dataset",
                {
                    method: "POST",
                    body: JSON.stringify({
                        prompt,
                        response,
                        category
                    })
                }
            );

        alert(
            "Добавлено. "
            + "Словарь: "
            + data.vocab_size
        );

        document.getElementById(
            "prompt"
        ).value = "";

        document.getElementById(
            "response"
        ).value = "";

        loadDataset();

    } catch (e) {
        alert(e.message);
    }
}


async function loadDataset() {
    try {
        const data =
            await api(
                "/api/admin/dataset"
            );

        const root =
            document.getElementById(
                "dataset"
            );

        root.innerHTML = "";

        data.dataset.forEach(
            (item, index) => {

                const div =
                    document.createElement(
                        "div"
                    );

                div.className =
                    "example";

                div.innerHTML = `
                    <b>#${index + 1}
                    [${escapeHtml(
                        item.category
                    )}]</b>

                    <div class="small">
                    Вопрос
                    </div>

                    <div>
                    ${escapeHtml(
                        item.prompt
                    )}
                    </div>

                    <div class="small">
                    Ответ
                    </div>

                    <div>
                    ${escapeHtml(
                        item.response
                    )}
                    </div>
                `;

                root.appendChild(div);
            }
        );

    } catch (e) {
        alert(e.message);
    }
}


async function evaluateAI() {
    try {
        const data =
            await api(
                "/api/admin/evaluate"
            );

        document.getElementById(
            "evaluation"
        ).textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    } catch (e) {
        alert(e.message);
    }
}


function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


setInterval(
    () => {
        if (getToken()) {
            status();
        }
    },
    5000
);

</script>

</body>
</html>
"""


@app.get(
    "/admin",
    response_class=HTMLResponse,
)
def admin():
    return ADMIN_HTML


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    print(
        f"Starting {APP_NAME}"
    )

    load_dataset()

    initialize_model()

    print(
        "Dataset:",
        len(dataset),
    )

    print(
        "Vocabulary:",
        len(vocab),
    )

    print(
        "Trained epochs:",
        trained_epochs,
    )

    print(
        "Supabase:",
        supabase is not None,
    )
