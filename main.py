import os
import re
import json
import time
import threading
from pathlib import Path
from io import BytesIO

import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None


# ============================================================
# AI CARE V5
# СОБСТВЕННАЯ RNN + SUPABASE + ПАМЯТЬ + CHECKPOINTS
# ============================================================

APP_NAME = "AI Care v5"

BASE_DIR = Path(__file__).resolve().parent

LOCAL_MODEL_FILE = BASE_DIR / "model.npz"
LOCAL_DATASET_FILE = BASE_DIR / "dataset.json"


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

SUPABASE_BUCKET = os.getenv(
    "SUPABASE_BUCKET",
    "ai-care-models"
)

supabase = None

if (
    create_client is not None
    and SUPABASE_URL
    and SUPABASE_KEY
):

    try:

        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

        print(
            "Supabase подключён."
        )

    except Exception as error:

        print(
            "Supabase connection error:",
            error
        )


# ============================================================
# НАСТРОЙКИ
# ============================================================

HIDDEN_SIZE = 128

LEARNING_RATE = 0.005

MAX_CONTEXT_MESSAGES = 10

MAX_MESSAGE_LENGTH = 1000

MAX_RESPONSE_LENGTH = 100

GRADIENT_CLIP = 5.0

TEMPERATURE = 0.75

TOP_K = 8

REPETITION_PENALTY = 1.25

MIN_KNOWN_TOKENS = 2

CHECKPOINT_EVERY_EPOCH = 1

MAX_DATASET_EXAMPLES = 10000


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME
)


# ============================================================
# LOCKS
# ============================================================

db_lock = threading.Lock()

model_lock = threading.RLock()

training_lock = threading.Lock()


# ============================================================
# DEFAULT DATASET
# ============================================================

DEFAULT_DATASET = [

    {
        "prompt": "привет",
        "response": "Привет! Я AI Care. Чем могу помочь?"
    },

    {
        "prompt": "здравствуй",
        "response": "Привет! Рад тебя видеть. Рассказывай, что тебя интересует."
    },

    {
        "prompt": "хай",
        "response": "Привет! Чем могу помочь?"
    },

    {
        "prompt": "добрый день",
        "response": "Привет! Чем могу помочь?"
    },

    {
        "prompt": "как дела",
        "response": "Всё хорошо! Готов помочь тебе."
    },

    {
        "prompt": "что ты умеешь",
        "response": "Я умею отвечать на вопросы, запоминать контекст общения и помогать с уходом, гигиеной, волосами и полезными привычками."
    },

    {
        "prompt": "кто ты",
        "response": "Я AI Care — небольшая собственная нейросеть с памятью, обучением и базой знаний."
    },

    {
        "prompt": "как тебя зовут",
        "response": "Меня зовут AI Care."
    },

    {
        "prompt": "спасибо",
        "response": "Пожалуйста! Рад помочь."
    },

    {
        "prompt": "спасибо большое",
        "response": "Не за что!"
    },

    {
        "prompt": "пока",
        "response": "Пока! Удачи!"
    },

    {
        "prompt": "до свидания",
        "response": "До встречи! Хорошего дня."
    },

    {
        "prompt": "у меня жирная кожа",
        "response": "При жирной коже обычно помогает мягкое очищение утром и вечером и лёгкий увлажняющий крем."
    },

    {
        "prompt": "что делать если кожа жирная",
        "response": "Попробуй мягкое очищение утром и вечером и лёгкий увлажняющий уход. Не стоит сильно пересушивать кожу."
    },

    {
        "prompt": "как ухаживать за лицом",
        "response": "Базовый уход — мягкое очищение, увлажнение и защита кожи от солнца."
    },

    {
        "prompt": "у меня сухая кожа",
        "response": "При сухой коже лучше использовать мягкое очищение и простой увлажняющий крем."
    },

    {
        "prompt": "кожа раздражена",
        "response": "Если кожа раздражена, лучше временно убрать агрессивные средства и использовать мягкое очищение и простой увлажняющий уход."
    },

    {
        "prompt": "как ухаживать за волосами",
        "response": "Подбери шампунь под кожу головы, мой волосы по необходимости и не перегружай их большим количеством средств."
    },

    {
        "prompt": "волосы быстро жирнятся",
        "response": "Можно подобрать подходящий шампунь для кожи головы и мыть волосы по мере необходимости."
    },

    {
        "prompt": "как часто умываться",
        "response": "Для большинства людей достаточно мягкого умывания утром и вечером."
    },

    {
        "prompt": "как улучшить внешний вид",
        "response": "Полезно уделять внимание коже, волосам, гигиене, сну, активности и одежде, в которой тебе комфортно."
    },

    {
        "prompt": "как лучше спать",
        "response": "Старайся ложиться и вставать примерно в одно время и давать организму достаточно сна."
    },

    {
        "prompt": "я устал",
        "response": "Попробуй отдохнуть, нормально поспать и не перегружать себя. Если сильная усталость долго не проходит, стоит рассказать взрослому и обратиться к врачу."
    },

    {
        "prompt": "что делать если чёрные точки",
        "response": "Мягкое очищение и подходящий уход могут уменьшить выраженность чёрных точек. Не стоит постоянно выдавливать их."
    },

    {
        "prompt": "как быть здоровым",
        "response": "Полезная база — достаточно сна, обычная разнообразная еда, движение, отдых и нормальная гигиена."
    },

    {
        "prompt": "мне скучно",
        "response": "Можно заняться игрой, прогулкой, небольшим проектом или попробовать изучить что-нибудь новое."
    },

    {
        "prompt": "что делать если появились прыщи",
        "response": "Мягкое очищение и подходящий уход могут помочь. Не стоит постоянно трогать или выдавливать воспаления."
    },

    {
        "prompt": "как избавиться от чёрных точек",
        "response": "Начни с мягкого очищения и подходящего ухода. Не стоит агрессивно скрабировать кожу или постоянно выдавливать точки."
    },

    {
        "prompt": "как ухаживать за кожей",
        "response": "Простой уход обычно состоит из мягкого очищения, увлажнения и защиты кожи от солнца."
    },

    {
        "prompt": "что делать если волосы сухие",
        "response": "Используй мягкий уход, не мой волосы слишком горячей водой и не перегружай их большим количеством средств."
    },

    {
        "prompt": "как улучшить сон",
        "response": "Помогает стабильный режим сна, спокойный вечер и достаточное количество времени для отдыха."
    },

    {
        "prompt": "что делать если нет энергии",
        "response": "Попробуй отдохнуть, нормально поесть, попить воды и выспаться. Если состояние долго не проходит, стоит обратиться к взрослому и врачу."
    },

    {
        "prompt": "что делать когда скучно",
        "response": "Можно поиграть, погулять, заняться программированием, музыкой или попробовать изучить новую тему."
    }

]


# ============================================================
# LOCAL FALLBACK DATASET
# ============================================================

def local_load_dataset():

    if not LOCAL_DATASET_FILE.exists():

        LOCAL_DATASET_FILE.write_text(
            json.dumps(
                DEFAULT_DATASET,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    try:

        data = json.loads(
            LOCAL_DATASET_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):

            return data

    except Exception:

        pass

    return DEFAULT_DATASET.copy()


def local_save_dataset(data):

    LOCAL_DATASET_FILE.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# SUPABASE DATASET
# ============================================================

def load_dataset():

    if supabase is not None:

        try:

            result = (
                supabase
                .table("ai_care_dataset")
                .select("id,prompt,response")
                .order("id")
                .limit(MAX_DATASET_EXAMPLES)
                .execute()
            )

            rows = result.data or []

            if rows:

                return [
                    {
                        "prompt": str(
                            row.get(
                                "prompt",
                                ""
                            )
                        ),
                        "response": str(
                            row.get(
                                "response",
                                ""
                            )
                        )
                    }

                    for row in rows
                ]

        except Exception as error:

            print(
                "Supabase dataset load error:",
                error
            )

    return local_load_dataset()


def save_dataset(data):

    data = data[
        :MAX_DATASET_EXAMPLES
    ]

    local_save_dataset(
        data
    )

    if supabase is None:
        return

    try:

        existing = (
            supabase
            .table("ai_care_dataset")
            .select("id")
            .limit(MAX_DATASET_EXAMPLES)
            .execute()
        )

        for row in (
            existing.data or []
        ):

            try:

                supabase \
                    .table("ai_care_dataset") \
                    .delete() \
                    .eq(
                        "id",
                        row["id"]
                    ) \
                    .execute()

            except Exception:
                pass

        if data:

            rows = [

                {
                    "prompt":
                        str(
                            item.get(
                                "prompt",
                                ""
                            )
                        ).strip(),

                    "response":
                        str(
                            item.get(
                                "response",
                                ""
                            )
                        ).strip()
                }

                for item in data

                if str(
                    item.get(
                        "prompt",
                        ""
                    )
                ).strip()
                and str(
                    item.get(
                        "response",
                        ""
                    )
                ).strip()
            ]

            if rows:

                supabase \
                    .table(
                        "ai_care_dataset"
                    ) \
                    .insert(
                        rows
                    ) \
                    .execute()

        print(
            "Dataset сохранён в Supabase."
        )

    except Exception as error:

        print(
            "Supabase dataset save error:",
            error
        )


def ensure_dataset():

    dataset = load_dataset()

    if dataset:
        return

    save_dataset(
        DEFAULT_DATASET.copy()
    )


# ============================================================
# SUPABASE USERS
# ============================================================

def ensure_user(user_id):

    if supabase is None:
        return

    try:

        result = (
            supabase
            .table("ai_care_users")
            .select("user_id")
            .eq(
                "user_id",
                user_id
            )
            .limit(1)
            .execute()
        )

        if result.data:
            return

        supabase \
            .table("ai_care_users") \
            .insert(
                {
                    "user_id":
                        user_id,

                    "created_at":
                        time.time(),

                    "updated_at":
                        time.time()
                }
            ) \
            .execute()

    except Exception as error:

        print(
            "ensure_user error:",
            error
        )


# ============================================================
# MESSAGES
# ============================================================

def save_message(
    user_id,
    role,
    text
):

    ensure_user(
        user_id
    )

    if supabase is None:
        return

    try:

        supabase \
            .table("ai_care_messages") \
            .insert(
                {
                    "user_id":
                        user_id,

                    "role":
                        role,

                    "text":
                        text,

                    "created_at":
                        time.time()
                }
            ) \
            .execute()

    except Exception as error:

        print(
            "save_message error:",
            error
        )


def get_messages(
    user_id,
    limit=MAX_CONTEXT_MESSAGES
):

    ensure_user(
        user_id
    )

    if supabase is None:
        return []

    try:

        result = (
            supabase
            .table("ai_care_messages")
            .select("role,text")
            .eq(
                "user_id",
                user_id
            )
            .order(
                "id",
                desc=True
            )
            .limit(limit)
            .execute()
        )

        rows = result.data or []

        return list(
            reversed(rows)
        )

    except Exception as error:

        print(
            "get_messages error:",
            error
        )

        return []


def clear_chat(user_id):

    if supabase is None:
        return

    try:

        supabase \
            .table(
                "ai_care_messages"
            ) \
            .delete() \
            .eq(
                "user_id",
                user_id
            ) \
            .execute()

    except Exception as error:

        print(
            "clear_chat error:",
            error
        )


# ============================================================
# MEMORY
# ============================================================

def save_memory(
    user_id,
    key,
    value
):

    ensure_user(
        user_id
    )

    if supabase is None:
        return

    try:

        supabase \
            .table(
                "ai_care_memories"
            ) \
            .upsert(
                {
                    "user_id":
                        user_id,

                    "memory_key":
                        key,

                    "memory_value":
                        value,

                    "created_at":
                        time.time(),

                    "updated_at":
                        time.time()
                },
                on_conflict=
                    "user_id,memory_key"
            ) \
            .execute()

    except Exception as error:

        print(
            "save_memory error:",
            error
        )


def get_memories(user_id):

    ensure_user(
        user_id
    )

    if supabase is None:
        return {}

    try:

        result = (
            supabase
            .table(
                "ai_care_memories"
            )
            .select(
                "memory_key,memory_value"
            )
            .eq(
                "user_id",
                user_id
            )
            .order(
                "updated_at",
                desc=True
            )
            .execute()
        )

        return {
            row["memory_key"]:
                row["memory_value"]

            for row in (
                result.data or []
            )
        }

    except Exception as error:

        print(
            "get_memories error:",
            error
        )

        return {}


def delete_memories(user_id):

    if supabase is None:
        return

    try:

        supabase \
            .table(
                "ai_care_memories"
            ) \
            .delete() \
            .eq(
                "user_id",
                user_id
            ) \
            .execute()

    except Exception as error:

        print(
            "delete_memories error:",
            error
        )


# ============================================================
# АВТОМАТИЧЕСКАЯ ПАМЯТЬ
# ============================================================

def extract_memory(
    user_id,
    text
):

    text = text.strip()

    lower = text.lower()

    patterns = [

        r"меня зовут\s+([а-яёa-z-]{2,30})",

        r"моё имя\s+([а-яёa-z-]{2,30})",

        r"мое имя\s+([а-яёa-z-]{2,30})"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            lower
        )

        if match:

            name = (
                match.group(1)
                .capitalize()
            )

            save_memory(
                user_id,
                "name",
                name
            )

            break

    patterns = [

        r"я люблю\s+(.+)",

        r"мне нравится\s+(.+)",

        r"моя любимая игра\s+(.+)",

        r"мой любимый фильм\s+(.+)"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = (
                match.group(1)
                .strip()
            )

            if 2 <= len(value) <= 200:

                save_memory(
                    user_id,
                    "preference",
                    value
                )

            break

    match = re.search(
        r"(?:запомни|запиши|сохрани)\s*:?\s*(.+)",
        text,
        re.IGNORECASE
    )

    if match:

        value = (
            match.group(1)
            .strip()
        )

        if 2 <= len(value) <= 300:

            save_memory(
                user_id,
                "user_note",
                value
            )


# ============================================================
# TOKENIZER
# ============================================================

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]


def tokenize(text):

    text = text.lower()

    return re.findall(
        r"[а-яёa-z0-9]+|[.,!?;:()\-—]",
        text,
        flags=re.IGNORECASE
    )


def build_vocab(dataset):

    counter = {}

    for item in dataset:

        text = (
            str(
                item.get(
                    "prompt",
                    ""
                )
            )
            + " "
            +
            str(
                item.get(
                    "response",
                    ""
                )
            )
        )

        for token in tokenize(text):

            counter[token] = (
                counter.get(
                    token,
                    0
                )
                + 1
            )

    words = sorted(
        counter.keys()
    )

    vocab = (
        SPECIAL_TOKENS
        + words
    )

    token_to_id = {
        token: i
        for i, token in enumerate(
            vocab
        )
    }

    id_to_token = {
        i: token
        for token, i in (
            token_to_id.items()
        )
    }

    return (
        token_to_id,
        id_to_token
    )


# ============================================================
# RNN
# ============================================================

class RNN:

    def __init__(
        self,
        vocab_size,
        hidden_size=HIDDEN_SIZE
    ):

        self.vocab_size = vocab_size

        self.hidden_size = hidden_size

        scale = 0.05

        self.Wxh = (
            np.random.randn(
                hidden_size,
                vocab_size
            ) * scale
        ).astype(
            np.float32
        )

        self.Whh = (
            np.random.randn(
                hidden_size,
                hidden_size
            ) * scale
        ).astype(
            np.float32
        )

        self.Why = (
            np.random.randn(
                vocab_size,
                hidden_size
            ) * scale
        ).astype(
            np.float32
        )

        self.bh = np.zeros(
            (
                hidden_size,
                1
            ),
            dtype=np.float32
        )

        self.by = np.zeros(
            (
                vocab_size,
                1
            ),
            dtype=np.float32
        )

    def softmax(self, x):

        x = x - np.max(x)

        exp = np.exp(
            np.clip(
                x,
                -50,
                50
            )
        )

        return (
            exp /
            (
                np.sum(exp)
                + 1e-12
            )
        )

    def forward(
        self,
        sequence
    ):

        h = np.zeros(
            (
                self.hidden_size,
                1
            ),
            dtype=np.float32
        )

        hs = {
            -1: h
        }

        probabilities = {}

        loss = 0.0

        for t in range(
            len(sequence) - 1
        ):

            current_id = (
                sequence[t]
            )

            target_id = (
                sequence[t + 1]
            )

            x = np.zeros(
                (
                    self.vocab_size,
                    1
                ),
                dtype=np.float32
            )

            x[
                current_id,
                0
            ] = 1.0

            h = np.tanh(
                self.Wxh @ x
                +
                self.Whh @ hs[t - 1]
                +
                self.bh
            )

            logits = (
                self.Why @ h
                +
                self.by
            )

            p = self.softmax(
                logits
            )

            hs[t] = h

            probabilities[t] = p

            loss -= np.log(
                p[
                    target_id,
                    0
                ]
                + 1e-12
            )

        return (
            loss,
            hs,
            probabilities
        )

    def train(
        self,
        sequence,
        learning_rate
    ):

        if len(sequence) < 3:
            return 0.0

        loss, hs, probs = (
            self.forward(
                sequence
            )
        )

        dWxh = np.zeros_like(
            self.Wxh
        )

        dWhh = np.zeros_like(
            self.Whh
        )

        dWhy = np.zeros_like(
            self.Why
        )

        dbh = np.zeros_like(
            self.bh
        )

        dby = np.zeros_like(
            self.by
        )

        dh_next = np.zeros_like(
            self.bh
        )

        for t in reversed(
            range(
                len(sequence) - 1
            )
        ):

            target_id = (
                sequence[t + 1]
            )

            dy = probs[t].copy()

            dy[
                target_id,
                0
            ] -= 1.0

            dWhy += (
                dy
                @
                hs[t].T
            )

            dby += dy

            dh = (
                self.Why.T
                @
                dy
                +
                dh_next
            )

            dh_raw = (
                (
                    1
                    -
                    hs[t] ** 2
                )
                *
                dh
            )

            dbh += dh_raw

            current_id = (
                sequence[t]
            )

            x = np.zeros(
                (
                    self.vocab_size,
                    1
                ),
                dtype=np.float32
            )

            x[
                current_id,
                0
            ] = 1.0

            dWxh += (
                dh_raw
                @
                x.T
            )

            previous_h = (
                hs[t - 1]
            )

            dWhh += (
                dh_raw
                @
                previous_h.T
            )

            dh_next = (
                self.Whh.T
                @
                dh_raw
            )

        gradients = [

            dWxh,
            dWhh,
            dWhy,
            dbh,
            dby

        ]

        for gradient in gradients:

            np.clip(
                gradient,
                -GRADIENT_CLIP,
                GRADIENT_CLIP,
                out=gradient
            )

        self.Wxh -= (
            learning_rate
            *
            dWxh
        )

        self.Whh -= (
            learning_rate
            *
            dWhh
        )

        self.Why -= (
            learning_rate
            *
            dWhy
        )

        self.bh -= (
            learning_rate
            *
            dbh
        )

        self.by -= (
            learning_rate
            *
            dby
        )

        return float(loss)


# ============================================================
# MODEL GLOBALS
# ============================================================

MODEL = None

TOKEN_TO_ID = {}

ID_TO_TOKEN = {}

MODEL_VERSION = 0

TRAINED_EPOCHS = 0


# ============================================================
# MODEL SERIALIZATION
# ============================================================

def serialize_model():

    if MODEL is None:
        return None

    buffer = BytesIO()

    np.savez_compressed(
        buffer,

        Wxh=MODEL.Wxh,

        Whh=MODEL.Whh,

        Why=MODEL.Why,

        bh=MODEL.bh,

        by=MODEL.by,

        vocab=json.dumps(
            TOKEN_TO_ID,
            ensure_ascii=False
        ),

        hidden_size=np.array(
            [
                MODEL.hidden_size
            ],
            dtype=np.int32
        ),

        model_version=np.array(
            [
                MODEL_VERSION
            ],
            dtype=np.int64
        ),

        trained_epochs=np.array(
            [
                TRAINED_EPOCHS
            ],
            dtype=np.int64
        )
    )

    return buffer.getvalue()


def deserialize_model(blob):

    global MODEL
    global TOKEN_TO_ID
    global ID_TO_TOKEN
    global MODEL_VERSION
    global TRAINED_EPOCHS

    data = np.load(
        BytesIO(blob),
        allow_pickle=False
    )

    vocab = json.loads(
        str(
            data["vocab"]
        )
    )

    hidden_size = int(
        data[
            "hidden_size"
        ][0]
    )

    MODEL_VERSION = int(
        data[
            "model_version"
        ][0]
    )

    TRAINED_EPOCHS = int(
        data[
            "trained_epochs"
        ][0]
    )

    TOKEN_TO_ID = vocab

    ID_TO_TOKEN = {
        int(i): token
        for token, i
        in vocab.items()
    }

    MODEL = RNN(
        vocab_size=len(
            vocab
        ),
        hidden_size=hidden_size
    )

    MODEL.Wxh = (
        data["Wxh"]
        .astype(
            np.float32
        )
    )

    MODEL.Whh = (
        data["Whh"]
        .astype(
            np.float32
        )
    )

    MODEL.Why = (
        data["Why"]
        .astype(
            np.float32
        )
    )

    MODEL.bh = (
        data["bh"]
        .astype(
            np.float32
        )
    )

    MODEL.by = (
        data["by"]
        .astype(
            np.float32
        )
    )


# ============================================================
# MODEL EXPANSION
# ============================================================

def expand_model_for_vocab(
    new_token_to_id,
    new_id_to_token
):

    global MODEL
    global TOKEN_TO_ID
    global ID_TO_TOKEN

    if MODEL is None:

        MODEL = RNN(
            vocab_size=len(
                new_token_to_id
            )
        )

        TOKEN_TO_ID = (
            new_token_to_id
        )

        ID_TO_TOKEN = (
            new_id_to_token
        )

        return

    old_vocab = TOKEN_TO_ID

    old_model = MODEL

    new_vocab_size = len(
        new_token_to_id
    )

    if (
        new_vocab_size
        == old_model.vocab_size
        and
        old_vocab
        == new_token_to_id
    ):

        return

    new_model = RNN(
        vocab_size=new_vocab_size,
        hidden_size=old_model.hidden_size
    )

    for token, new_id in (
        new_token_to_id.items()
    ):

        old_id = old_vocab.get(
            token
        )

        if old_id is None:
            continue

        new_model.Wxh[
            :,
            new_id
        ] = old_model.Wxh[
            :,
            old_id
        ]

        new_model.Why[
            new_id,
            :
        ] = old_model.Why[
            old_id,
            :
        ]

        new_model.by[
            new_id,
            0
        ] = old_model.by[
            old_id,
            0
        ]

    new_model.Whh[:] = (
        old_model.Whh
    )

    new_model.bh[:] = (
        old_model.bh
    )

    MODEL = new_model

    TOKEN_TO_ID = (
        new_token_to_id
    )

    ID_TO_TOKEN = (
        new_id_to_token
    )


# ============================================================
# LOCAL MODEL SAVE
# ============================================================

def save_model_local():

    blob = serialize_model()

    if blob is None:
        return

    with open(
        LOCAL_MODEL_FILE,
        "wb"
    ) as file:

        file.write(
            blob
        )


# ============================================================
# SUPABASE MODEL SAVE
# ============================================================

def upload_model(
    epoch=None,
    loss=None
):

    if supabase is None:
        return False

    blob = serialize_model()

    if blob is None:
        return False

    timestamp = int(
        time.time()
    )

    if epoch is None:
        epoch = TRAINED_EPOCHS

    latest_path = (
        "latest/model.npz"
    )

    checkpoint_path = (
        f"checkpoints/"
        f"epoch_{epoch}_"
        f"{timestamp}.npz"
    )

    try:

        supabase.storage \
            .from_(
                SUPABASE_BUCKET
            ) \
            .upload(
                latest_path,
                blob,
                {
                    "content-type":
                        "application/octet-stream",
                    "upsert":
                        "true"
                }
            )

        supabase.storage \
            .from_(
                SUPABASE_BUCKET
            ) \
            .upload(
                checkpoint_path,
                blob,
                {
                    "content-type":
                        "application/octet-stream",
                    "upsert":
                        "false"
                }
            )

        try:

            supabase \
                .table(
                    "ai_care_model_versions"
                ) \
                .insert(
                    {
                        "epoch":
                            epoch,

                        "loss":
                            float(
                                loss or 0
                            ),

                        "storage_path":
                            checkpoint_path,

                        "model_version":
                            MODEL_VERSION,

                        "created_at":
                            time.time()
                    }
                ) \
                .execute()

        except Exception as error:

            print(
                "Model metadata error:",
                error
            )

        print(
            "Checkpoint сохранён:",
            checkpoint_path
        )

        return True

    except Exception as error:

        print(
            "Supabase model upload error:",
            error
        )

        return False


# ============================================================
# MODEL LOAD
# ============================================================

def download_latest_model():

    if supabase is None:
        return None

    try:

        blob = (
            supabase.storage
            .from_(
                SUPABASE_BUCKET
            )
            .download(
                "latest/model.npz"
            )
        )

        if blob:

            print(
                "Последняя модель "
                "загружена из Supabase."
            )

            return blob

    except Exception as error:

        print(
            "Supabase model download:",
            error
        )

    return None


# ============================================================
# CREATE / LOAD MODEL
# ============================================================

def create_model():

    global MODEL
    global TOKEN_TO_ID
    global ID_TO_TOKEN
    global MODEL_VERSION
    global TRAINED_EPOCHS

    dataset = load_dataset()

    (
        new_token_to_id,
        new_id_to_token
    ) = build_vocab(
        dataset
    )

    # --------------------------------------------------------
    # Сначала пытаемся получить модель из Supabase
    # --------------------------------------------------------

    blob = (
        download_latest_model()
    )

    if blob:

        try:

            deserialize_model(
                blob
            )

            # ------------------------------------------------
            # Если dataset получил новые слова,
            # расширяем модель, сохраняя старые веса.
            # ------------------------------------------------

            expand_model_for_vocab(
                new_token_to_id,
                new_id_to_token
            )

            save_model_local()

            print(
                "Модель восстановлена."
            )

            print(
                "Обученных эпох:",
                TRAINED_EPOCHS
            )

            return

        except Exception as error:

            print(
                "Supabase model invalid:",
                error
            )

    # --------------------------------------------------------
    # Локальная модель
    # --------------------------------------------------------

    if LOCAL_MODEL_FILE.exists():

        try:

            blob = (
                LOCAL_MODEL_FILE
                .read_bytes()
            )

            deserialize_model(
                blob
            )

            expand_model_for_vocab(
                new_token_to_id,
                new_id_to_token
            )

            print(
                "Локальная модель загружена."
            )

            return

        except Exception as error:

            print(
                "Local model error:",
                error
            )

    # --------------------------------------------------------
    # Новая модель
    # --------------------------------------------------------

    MODEL = RNN(
        vocab_size=len(
            new_token_to_id
        )
    )

    TOKEN_TO_ID = (
        new_token_to_id
    )

    ID_TO_TOKEN = (
        new_id_to_token
    )

    MODEL_VERSION = 1

    TRAINED_EPOCHS = 0

    print(
        "Создана новая модель."
    )


# ============================================================
# SAVE EVERYTHING
# ============================================================

def save_model(
    epoch=None,
    loss=None
):

    save_model_local()

    if supabase is not None:

        upload_model(
            epoch=epoch,
            loss=loss
        )


# ============================================================
# TRAINING
# ============================================================

training = {

    "running":
        False,

    "epoch":
        0,

    "epochs":
        0,

    "loss":
        0,

    "message":
        "Готово",

    "total_dataset":
        0,

    "trained_epochs":
        0

}


def make_sequence(
    prompt,
    response
):

    tokens = []

    tokens.extend(
        tokenize(
            prompt
        )
    )

    tokens.append(
        "<BOS>"
    )

    tokens.extend(
        tokenize(
            response
        )
    )

    tokens.append(
        "<EOS>"
    )

    return [

        TOKEN_TO_ID.get(
            token,
            TOKEN_TO_ID[
                "<UNK>"
            ]
        )

        for token in tokens

    ]


def train_model(
    epochs,
    learning_rate
):

    global MODEL
    global TRAINED_EPOCHS
    global MODEL_VERSION

    training["running"] = True

    training["epoch"] = 0

    training["epochs"] = epochs

    training["loss"] = 0

    training["message"] = (
        "Подготовка..."
    )

    try:

        dataset = load_dataset()

        if not dataset:

            training["message"] = (
                "Датасет пуст."
            )

            return

        training[
            "total_dataset"
        ] = len(dataset)

        # ----------------------------------------------------
        # Обновляем словарь перед обучением
        # ----------------------------------------------------

        (
            new_token_to_id,
            new_id_to_token
        ) = build_vocab(
            dataset
        )

        with model_lock:

            expand_model_for_vocab(
                new_token_to_id,
                new_id_to_token
            )

            for local_epoch in range(
                epochs
            ):

                indices = (
                    np.random.permutation(
                        len(dataset)
                    )
                )

                total_loss = 0.0

                count = 0

                for index in indices:

                    item = (
                        dataset[index]
                    )

                    prompt = str(
                        item.get(
                            "prompt",
                            ""
                        )
                    ).strip()

                    response = str(
                        item.get(
                            "response",
                            ""
                        )
                    ).strip()

                    if (
                        not prompt
                        or not response
                    ):
                        continue

                    sequence = (
                        make_sequence(
                            prompt,
                            response
                        )
                    )

                    if len(sequence) < 3:
                        continue

                    loss = MODEL.train(
                        sequence,
                        learning_rate
                    )

                    total_loss += (
                        loss
                    )

                    count += 1

                average_loss = (
                    total_loss
                    /
                    max(
                        count,
                        1
                    )
                )

                TRAINED_EPOCHS += 1

                MODEL_VERSION += 1

                training[
                    "epoch"
                ] = (
                    local_epoch
                    + 1
                )

                training[
                    "loss"
                ] = round(
                    average_loss,
                    4
                )

                training[
                    "trained_epochs"
                ] = (
                    TRAINED_EPOCHS
                )

                training[
                    "message"
                ] = (
                    f"Обучение: "
                    f"{local_epoch + 1}/"
                    f"{epochs}"
                )

                print(
                    training[
                        "message"
                    ],
                    "loss=",
                    average_loss
                )

                # ------------------------------------------------
                # CHECKPOINT ПОСЛЕ КАЖДОЙ ЭПОХИ
                # ------------------------------------------------

                save_model(
                    epoch=
                        TRAINED_EPOCHS,
                    loss=
                        average_loss
                )

            training[
                "message"
            ] = (
                "Обучение завершено."
            )

    except Exception as error:

        training[
            "message"
        ] = (
            "Ошибка обучения: "
            +
            str(error)
        )

        print(
            "TRAIN ERROR:",
            error
        )

    finally:

        training[
            "running"
        ] = False


def start_training(
    epochs,
    learning_rate
):

    if training[
        "running"
    ]:

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
# SEARCH KNOWLEDGE
# ============================================================

def similarity_score(
    query,
    prompt
):

    query_tokens = set(
        tokenize(
            query
        )
    )

    prompt_tokens = set(
        tokenize(
            prompt
        )
    )

    if not query_tokens:
        return 0.0

    intersection = (
        query_tokens
        &
        prompt_tokens
    )

    union = (
        query_tokens
        |
        prompt_tokens
    )

    if not union:
        return 0.0

    return (
        len(intersection)
        /
        len(union)
    )


def retrieve_knowledge(
    message,
    limit=TOP_K
):

    dataset = load_dataset()

    scored = []

    for item in dataset:

        prompt = str(
            item.get(
                "prompt",
                ""
            )
        )

        response = str(
            item.get(
                "response",
                ""
            )
        )

        score = similarity_score(
            message,
            prompt
        )

        if score > 0:

            scored.append(
                (
                    score,
                    prompt,
                    response
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored[:limit]


# ============================================================
# CONTEXT
# ============================================================

def build_context(
    user_id,
    current_message
):

    memories = get_memories(
        user_id
    )

    messages = get_messages(
        user_id
    )

    knowledge = retrieve_knowledge(
        current_message
    )

    parts = []

    if memories:

        parts.append(
            "Память:"
        )

        for key, value in (
            memories.items()
        ):

            parts.append(
                f"{key}: {value}"
            )

    if knowledge:

        parts.append(
            "Подходящие знания:"
        )

        for (
            score,
            prompt,
            response
        ) in knowledge:

            parts.append(
                f"Вопрос: {prompt}"
            )

            parts.append(
                f"Ответ: {response}"
            )

    for message in messages:

        if (
            message["role"]
            == "user"
        ):

            parts.append(
                "Пользователь: "
                +
                message["text"]
            )

        else:

            parts.append(
                "AI: "
                +
                message["text"]
            )

    parts.append(
        "Пользователь: "
        +
        current_message
    )

    parts.append(
        "AI:"
    )

    return "\n".join(
        parts
    )


# ============================================================
# FALLBACK
# ============================================================

def fallback_answer(
    message,
    memories
):

    lower = message.lower()

    name = memories.get(
        "name"
    )

    greeting = (
        f"Привет, {name}!"
        if name
        else "Привет!"
    )

    if any(
        word in lower
        for word in [
            "привет",
            "здравствуй",
            "хай",
            "добрый день"
        ]
    ):

        return (
            greeting
            +
            " Чем могу помочь?"
        )

    if (
        "как дела"
        in lower
    ):

        return (
            "Всё хорошо! "
            "Готов помочь тебе."
        )

    if (
        "кто ты"
        in lower
        or
        "что ты умеешь"
        in lower
    ):

        return (
            "Я AI Care — "
            "собственная небольшая "
            "нейросеть с памятью, "
            "обучением и базой знаний."
        )

    if (
        "спасибо"
        in lower
    ):

        return (
            "Пожалуйста!"
        )

    if (
        "пока"
        in lower
    ):

        return (
            "Пока! Удачи!"
        )

    if any(
        word in lower
        for word in [
            "жирная кожа",
            "жирнится",
            "жирную кожу"
        ]
    ):

        return (
            "При жирной коже обычно "
            "помогает мягкое очищение "
            "утром и вечером и лёгкий "
            "увлажняющий уход."
        )

    if any(
        word in lower
        for word in [
            "сухая кожа",
            "сухую кожу"
        ]
    ):

        return (
            "При сухой коже лучше "
            "использовать мягкое "
            "очищение и простой "
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
            "Для волос важно подобрать "
            "уход под кожу головы "
            "и мыть волосы "
            "по необходимости."
        )

    if any(
        word in lower
        for word in [
            "чёрные точки",
            "черные точки"
        ]
    ):

        return (
            "Мягкое очищение и "
            "подходящий уход могут "
            "помочь уменьшить "
            "выраженность чёрных точек."
        )

    return (
        "Я пока не знаю достаточно "
        "хорошего ответа на этот "
        "вопрос. Этот вопрос можно "
        "добавить в мои знания "
        "через админ-панель."
    )


# ============================================================
# GENERATION
# ============================================================

def generate_with_model(
    context
):

    if MODEL is None:
        return None

    tokens = tokenize(
        context
    )

    if not tokens:
        return None

    ids = [

        TOKEN_TO_ID.get(
            token,
            TOKEN_TO_ID[
                "<UNK>"
            ]
        )

        for token in tokens

    ]

    known = sum(
        1
        for token in tokens
        if token in TOKEN_TO_ID
    )

    if known < MIN_KNOWN_TOKENS:
        return None

    h = np.zeros(
        (
            MODEL.hidden_size,
            1
        ),
        dtype=np.float32
    )

    for token_id in ids:

        x = np.zeros(
            (
                MODEL.vocab_size,
                1
            ),
            dtype=np.float32
        )

        x[
            token_id,
            0
        ] = 1.0

        h = np.tanh(
            MODEL.Wxh @ x
            +
            MODEL.Whh @ h
            +
            MODEL.bh
        )

    result = []

    previous_tokens = []

    for _ in range(
        MAX_RESPONSE_LENGTH
    ):

        logits = (
            MODEL.Why @ h
            +
            MODEL.by
        ).reshape(-1)

        # ----------------------------------------------------
        # Температура
        # ----------------------------------------------------

        logits = (
            logits
            /
            max(
                TEMPERATURE,
                0.05
            )
        )

        # ----------------------------------------------------
        # Repetition penalty
        # ----------------------------------------------------

        for token_id in set(
            previous_tokens
        ):

            if (
                0 <= token_id
                < len(logits)
            ):

                if logits[
                    token_id
                ] > 0:

                    logits[
                        token_id
                    ] /= (
                        REPETITION_PENALTY
                    )

                else:

                    logits[
                        token_id
                    ] *= (
                        REPETITION_PENALTY
                    )

        logits -= np.max(
            logits
        )

        probabilities = np.exp(
            np.clip(
                logits,
                -50,
                50
            )
        )

        # ----------------------------------------------------
        # PAD / UNK / BOS запрещаем
        # EOS разрешён
        # ----------------------------------------------------

        for token in [
            "<PAD>",
            "<UNK>",
            "<BOS>"
        ]:

            token_id = (
                TOKEN_TO_ID.get(
                    token
                )
            )

            if token_id is not None:

                probabilities[
                    token_id
                ] = 0.0

        eos_id = (
            TOKEN_TO_ID.get(
                "<EOS>"
            )
        )

        # ----------------------------------------------------
        # TOP-K
        # ----------------------------------------------------

        if TOP_K > 0:

            valid_indices = np.where(
                probabilities > 0
            )[0]

            if len(
                valid_indices
            ) > TOP_K:

                top_indices = (
                    valid_indices[
                        np.argsort(
                            probabilities[
                                valid_indices
                            ]
                        )[
                            -TOP_K:
                        ]
                    ]
                )

                mask = np.zeros(
                    len(probabilities),
                    dtype=bool
                )

                mask[
                    top_indices
                ] = True

                probabilities[
                    ~mask
                ] = 0.0

        total = np.sum(
            probabilities
        )

        if total <= 0:
            break

        probabilities /= total

        next_id = np.random.choice(
            len(probabilities),
            p=probabilities
        )

        if (
            eos_id is not None
            and
            next_id == eos_id
        ):

            break

        token = ID_TO_TOKEN.get(
            next_id,
            ""
        )

        if not token:
            break

        # ----------------------------------------------------
        # Повторы
        # ----------------------------------------------------

        if (
            len(previous_tokens)
            >= 3
            and
            previous_tokens[-1]
            == next_id
            and
            previous_tokens[-2]
            == next_id
        ):

            break

        result.append(
            token
        )

        previous_tokens.append(
            next_id
        )

        x = np.zeros(
            (
                MODEL.vocab_size,
                1
            ),
            dtype=np.float32
        )

        x[
            next_id,
            0
        ] = 1.0

        h = np.tanh(
            MODEL.Wxh @ x
            +
            MODEL.Whh @ h
            +
            MODEL.bh
        )

    if not result:
        return None

    text = ""

    for token in result:

        if token in [
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            ")"
        ]:

            text += token

        elif token == "(":

            if text:
                text += " "

            text += token

        elif token == "—":

            text += " " + token + " "

        else:

            if text:
                text += " "

            text += token

    text = re.sub(
        r"\s+([.,!?;:)])",
        r"\1",
        text
    )

    text = re.sub(
        r"\(\s+",
        "(",
        text
    )

    text = text.strip()

    if len(text) < 3:
        return None

    words = text.split()

    if len(words) >= 5:

        unique_ratio = (
            len(set(words))
            /
            len(words)
        )

        if unique_ratio < 0.3:

            return None

    return text


# ============================================================
# ANSWER QUALITY
# ============================================================

def answer_quality(
    question,
    answer
):

    if not answer:
        return 0.0

    if len(answer) < 3:
        return 0.0

    score = 0.5

    question_words = set(
        tokenize(
            question
        )
    )

    answer_words = set(
        tokenize(
            answer
        )
    )

    if question_words:

        overlap = (
            len(
                question_words
                &
                answer_words
            )
            /
            len(
                question_words
            )
        )

        score += min(
            overlap,
            0.4
        )

    words = answer.split()

    if words:

        unique_ratio = (
            len(set(words))
            /
            len(words)
        )

        if unique_ratio < 0.35:

            score -= 0.4

    return max(
        0.0,
        min(
            score,
            1.0
        )
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    user_id,
    message
):

    memories = get_memories(
        user_id
    )

    # --------------------------------------------------------
    # Быстрый поиск точного / похожего знания
    # --------------------------------------------------------

    knowledge = retrieve_knowledge(
        message,
        limit=3
    )

    if knowledge:

        best = knowledge[0]

        if best[0] >= 0.6:

            return best[2]

    context = build_context(
        user_id,
        message
    )

    try:

        with model_lock:

            answer = (
                generate_with_model(
                    context
                )
            )

        if answer:

            quality = (
                answer_quality(
                    message,
                    answer
                )
            )

            if quality >= 0.45:

                return answer

    except Exception as error:

        print(
            "GENERATION ERROR:",
            error
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return fallback_answer(
        message,
        memories
    )


# ============================================================
# API MODELS
# ============================================================

class ChatRequest(BaseModel):

    user_id: str = "demo_user"

    message: str


class TrainRequest(BaseModel):

    epochs: int = 10

    learning_rate: float = (
        LEARNING_RATE
    )


class ExampleRequest(BaseModel):

    prompt: str

    response: str


class MemoryRequest(BaseModel):

    user_id: str

    key: str

    value: str


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(
    data: ChatRequest
):

    user_id = (
        data.user_id.strip()
        or
        "demo_user"
    )

    message = (
        data.message.strip()
    )

    if not message:

        return JSONResponse(
            {
                "error":
                    "Сообщение пустое"
            },
            status_code=400
        )

    if len(message) > (
        MAX_MESSAGE_LENGTH
    ):

        return JSONResponse(
            {
                "error":
                    "Сообщение слишком длинное"
            },
            status_code=400
        )

    ensure_user(
        user_id
    )

    extract_memory(
        user_id,
        message
    )

    save_message(
        user_id,
        "user",
        message
    )

    answer = generate_answer(
        user_id,
        message
    )

    save_message(
        user_id,
        "assistant",
        answer
    )

    return {

        "answer":
            answer,

        "memory":
            get_memories(
                user_id
            )

    }


# ============================================================
# MEMORY API
# ============================================================

@app.get(
    "/api/memory/{user_id}"
)
def memory(
    user_id: str
):

    return {

        "memory":
            get_memories(
                user_id
            )

    }


@app.post("/api/memory")
def add_memory(
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


@app.delete(
    "/api/memory/{user_id}"
)
def delete_memory(
    user_id: str
):

    delete_memories(
        user_id
    )

    return {
        "ok": True
    }


@app.delete(
    "/api/chat/{user_id}"
)
def delete_chat(
    user_id: str
):

    clear_chat(
        user_id
    )

    return {
        "ok": True
    }


# ============================================================
# DATASET API
# ============================================================

@app.get("/api/dataset")
def dataset():

    data = load_dataset()

    return {

        "count":
            len(data),

        "dataset":
            data

    }


@app.post("/api/add-example")
def add_example(
    data: ExampleRequest
):

    prompt = (
        data.prompt.strip()
    )

    response = (
        data.response.strip()
    )

    if not prompt or not response:

        return JSONResponse(
            {
                "error":
                    "Оба поля обязательны"
            },
            status_code=400
        )

    dataset = load_dataset()

    dataset.append(
        {
            "prompt":
                prompt,

            "response":
                response
        }
    )

    save_dataset(
        dataset
    )

    # --------------------------------------------------------
    # Очень важно:
    # НЕ удаляем старую модель.
    #
    # Новые слова добавляются в словарь,
    # старые веса сохраняются.
    # --------------------------------------------------------

    with model_lock:

        (
            new_token_to_id,
            new_id_to_token
        ) = build_vocab(
            dataset
        )

        expand_model_for_vocab(
            new_token_to_id,
            new_id_to_token
        )

        save_model()

    return {

        "ok":
            True,

        "count":
            len(dataset)

    }


# ============================================================
# TRAIN
# ============================================================

@app.post("/api/train")
def train(
    data: TrainRequest
):

    epochs = max(
        1,
        min(
            int(
                data.epochs
            ),
            100
        )
    )

    learning_rate = max(
        0.0001,
        min(
            float(
                data.learning_rate
            ),
            0.1
        )
    )

    if training[
        "running"
    ]:

        return {

            "ok":
                False,

            "message":
                "Обучение уже идёт."

        }

    started = start_training(
        epochs,
        learning_rate
    )

    return {

        "ok":
            started,

        "message":
            (
                "Обучение запущено."
                if started
                else
                "Не удалось запустить."
            )

    }


@app.get(
    "/api/train/status"
)
def train_status():

    return {

        **training,

        "supabase":
            supabase is not None,

        "model_version":
            MODEL_VERSION,

        "trained_epochs":
            TRAINED_EPOCHS

    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "model":
            MODEL is not None,

        "trained":
            TRAINED_EPOCHS > 0,

        "dataset":
            len(
                load_dataset()
            ),

        "memory":
            (
                "Supabase"
                if supabase is not None
                else
                "local"
            ),

        "model_storage":
            (
                "Supabase Storage"
                if supabase is not None
                else
                "local filesystem"
            ),

        "trained_epochs":
            TRAINED_EPOCHS,

        "model_version":
            MODEL_VERSION

    }


# ============================================================
# CHAT UI
# ============================================================

CHAT_HTML = r"""
<!DOCTYPE html>

<html lang="ru">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>AI Care</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;

    background:
        radial-gradient(
            circle at top,
            #182653,
            #080b15 45%,
            #05060a
        );

    color: white;

    font-family:
        Arial,
        sans-serif;
}

.app {

    max-width:
        900px;

    min-height:
        100vh;

    margin:
        auto;

    display:
        flex;

    flex-direction:
        column;
}

.header {

    padding:
        20px;

    border-bottom:
        1px solid
        rgba(
            255,
            255,
            255,
            .1
        );

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    backdrop-filter:
        blur(20px);
}

.logo {

    font-size:
        22px;

    font-weight:
        bold;
}

.status {

    font-size:
        11px;

    opacity:
        .55;
}

.chat {

    flex:
        1;

    padding:
        20px;

    overflow-y:
        auto;
}

.message {

    display:
        flex;

    margin-bottom:
        14px;
}

.message.user {

    justify-content:
        flex-end;
}

.bubble {

    max-width:
        82%;

    padding:
        13px 16px;

    border-radius:
        18px;

    line-height:
        1.45;

    white-space:
        pre-wrap;
}

.message.assistant
.bubble {

    background:
        rgba(
            255,
            255,
            255,
            .07
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            .08
        );
}

.message.user
.bubble {

    background:
        #3867ff;
}

.bottom {

    display:
        flex;

    gap:
        10px;

    padding:
        15px;

    border-top:
        1px solid
        rgba(
            255,
            255,
            255,
            .1
        );

    background:
        rgba(
            0,
            0,
            0,
            .25
        );

    backdrop-filter:
        blur(20px);
}

input {

    flex:
        1;

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            .12
        );

    background:
        rgba(
            255,
            255,
            255,
            .06
        );

    color:
        white;

    border-radius:
        15px;

    outline:
        none;

    padding:
        14px;

    font-size:
        15px;
}

button {

    border:
        0;

    border-radius:
        15px;

    padding:
        0 20px;

    background:
        #3867ff;

    color:
        white;

    cursor:
        pointer;

    font-weight:
        bold;
}

button:hover {
    opacity: .9;
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
своя нейросеть • память • обучение
</div>

</div>

<div
    class="chat"
    id="chat"
>
</div>

<div class="bottom">

<input
    id="input"
    placeholder="Напиши сообщение..."
    autocomplete="off"
>

<button
    onclick="sendMessage()"
>
Отправить
</button>

</div>

</div>

<script>

const chat =
    document.getElementById(
        "chat"
    );

const input =
    document.getElementById(
        "input"
    );

let userId =
    localStorage.getItem(
        "ai_care_user_id"
    );

if (!userId) {

    userId =
        "user_" +
        Math.random()
            .toString(36)
            .substring(
                2,
                12
            );

    localStorage.setItem(
        "ai_care_user_id",
        userId
    );
}

function addMessage(
    role,
    text
) {

    const message =
        document.createElement(
            "div"
        );

    message.className =
        "message " +
        role;

    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        "bubble";

    bubble.textContent =
        text;

    message.appendChild(
        bubble
    );

    chat.appendChild(
        message
    );

    chat.scrollTop =
        chat.scrollHeight;
}


async function loadHistory() {

    /*
        История находится в Supabase.
        Последние сообщения используются
        сервером как контекст.
    */

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
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            user_id:
                                userId,

                            message:
                                message
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
            "Ошибка соединения с сервером."
        );
    }
}


input.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter"
            &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }

    }
);


addMessage(
    "assistant",
    "Привет! Я AI Care. У меня есть память, база знаний и собственное обучение."
);

</script>

</body>

</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return CHAT_HTML


# ============================================================
# ADMIN UI
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

    padding: 25px;

    background:
        #080a10;

    color:
        white;

    font-family:
        Arial,
        sans-serif;
}

.container {

    max-width:
        900px;

    margin:
        auto;
}

.card {

    padding:
        20px;

    margin-bottom:
        18px;

    border-radius:
        18px;

    background:
        rgba(
            255,
            255,
            255,
            .06
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            .1
        );
}

h1 {
    margin-top: 0;
}

input,
textarea,
button {

    width:
        100%;

    padding:
        13px;

    margin-top:
        8px;

    border-radius:
        10px;

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            .15
        );

    box-sizing:
        border-box;

    font-family:
        inherit;
}

input,
textarea {

    background:
        #11141e;

    color:
        white;
}

textarea {

    min-height:
        120px;

    resize:
        vertical;
}

button {

    border:
        0;

    background:
        #3867ff;

    color:
        white;

    cursor:
        pointer;

    font-weight:
        bold;
}

.status {

    padding:
        12px;

    background:
        rgba(
            255,
            255,
            255,
            .05
        );

    border-radius:
        10px;

    line-height:
        1.6;
}

pre {

    white-space:
        pre-wrap;

    background:
        #05060a;

    padding:
        15px;

    border-radius:
        12px;

    max-height:
        500px;

    overflow:
        auto;
}

</style>

</head>

<body>

<div class="container">

<h1>
AI Care — Admin
</h1>


<div class="card">

<h2>
Статус
</h2>

<div
    class="status"
    id="status"
>
Загрузка...
</div>

</div>


<div class="card">

<h2>
Обучение
</h2>

<label>
Дополнительные эпохи
</label>

<input
    id="epochs"
    type="number"
    value="10"
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

<button
    onclick="train()"
>
Обучить нейросеть
</button>

</div>


<div class="card">

<h2>
Добавить знания
</h2>

<input
    id="prompt"
    placeholder="Вопрос пользователя"
>

<textarea
    id="response"
    placeholder="Правильный ответ AI"
></textarea>

<button
    onclick="addExample()"
>
Добавить пример
</button>

</div>


<div class="card">

<h2>
Датасет
</h2>

<div
    id="count"
>
Загрузка...
</div>

<pre
    id="dataset"
></pre>

</div>

</div>


<script>

async function updateStatus() {

    try {

        const response =
            await fetch(
                "/api/train/status"
            );

        const data =
            await response.json();

        document.getElementById(
            "status"
        ).innerHTML =

            "Статус: " +

            data.message +

            "<br>" +

            "Эпоха запуска: " +

            data.epoch +

            "/" +

            data.epochs +

            "<br>" +

            "Loss: " +

            data.loss +

            "<br>" +

            "Всего обученных эпох: " +

            data.trained_epochs +

            "<br>" +

            "Версия модели: " +

            data.model_version +

            "<br>" +

            "Supabase: " +

            (
                data.supabase
                ? "подключён"
                : "не подключён"
            );

    } catch (error) {

        document.getElementById(
            "status"
        ).textContent =
            "Ошибка соединения";
    }
}


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
        "Количество примеров: "
        +
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

    if (
        !prompt
        ||
        !response
    ) {

        alert(
            "Заполни оба поля."
        );

        return;
    }

    const result =
        await fetch(
            "/api/add-example",
            {
                method:
                    "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        prompt:
                            prompt,

                        response:
                            response
                    })
            }
        );

    const data =
        await result.json();

    if (data.error) {

        alert(
            data.error
        );

        return;
    }

    document.getElementById(
        "prompt"
    ).value = "";

    document.getElementById(
        "response"
    ).value = "";

    alert(
        "Пример добавлен и сохранён."
    );

    loadDataset();

    updateStatus();
}


async function train() {

    const epochs =
        Number(
            document.getElementById(
                "epochs"
            ).value
        );

    const learningRate =
        Number(
            document.getElementById(
                "lr"
            ).value
        );

    const result =
        await fetch(
            "/api/train",
            {
                method:
                    "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        epochs:
                            epochs,

                        learning_rate:
                            learningRate
                    })
            }
        );

    const data =
        await result.json();

    alert(
        data.message
    );

    updateStatus();
}


loadDataset();

updateStatus();

setInterval(
    updateStatus,
    1000
);

</script>

</body>

</html>
"""


@app.get(
    "/admin",
    response_class=HTMLResponse
)
def admin():

    return ADMIN_HTML


# ============================================================
# STARTUP
# ============================================================

@app.on_event(
    "startup"
)
def startup():

    ensure_dataset()

    create_model()

    print(
        "=" * 60
    )

    print(
        APP_NAME
    )

    print(
        "Chat: /"
    )

    print(
        "Admin: /admin"
    )

    print(
        "Dataset:",
        (
            "Supabase"
            if supabase is not None
            else LOCAL_DATASET_FILE
        )
    )

    print(
        "Model:",
        (
            "Supabase Storage"
            if supabase is not None
            else LOCAL_MODEL_FILE
        )
    )

    print(
        "Memory:",
        (
            "Supabase"
            if supabase is not None
            else "local"
        )
    )

    print(
        "Trained epochs:",
        TRAINED_EPOCHS
    )

    print(
        "=" * 60
    )
