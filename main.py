# ============================================================
# AI CARE v2
# ============================================================
# Один файл: main.py
#
# Умеет:
#   /          — веб-чат
#   /admin     — админ-панель
#   /train     — страница обучения
#   /api/chat  — API чата
#   /api/train — переобучение
#
# НЕТ:
#   OpenAI API
#   Gemini API
#   Claude API
#   других AI API
#   обработки фотографий
#
# Используется:
#   Python
#   NumPy
#   FastAPI
#   собственная MLP
#   собственный backpropagation
#
# Render Start Command:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
#
# requirements.txt:
#   fastapi
#   uvicorn
#   numpy
# ============================================================

import os
import json
import random
import re
import math
from pathlib import Path

import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "AI Care"

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "ai_care_model.npz"
DATASET_FILE = BASE_DIR / "ai_care_dataset.json"

app = FastAPI(title=APP_NAME)


# ============================================================
# КЛАССЫ
# ============================================================

INTENTS = [
    "skin",
    "oily_skin",
    "dry_skin",
    "sensitive_skin",
    "hair",
    "sleep",
    "activity",
    "hygiene",
    "sun",
    "general",
]


INTENT_NAMES = {
    "skin": "Уход за кожей",
    "oily_skin": "Жирная кожа",
    "dry_skin": "Сухая кожа",
    "sensitive_skin": "Чувствительная кожа",
    "hair": "Уход за волосами",
    "sleep": "Сон",
    "activity": "Физическая активность",
    "hygiene": "Гигиена",
    "sun": "Защита от солнца",
    "general": "Общие рекомендации",
}


# ============================================================
# НАШИ ОТВЕТЫ
# ============================================================

RESPONSES = {

    "skin": [
        "Для базового ухода за кожей лучше начать с мягкого очищения и подходящего увлажняющего средства.",
        "Не стоит перегружать уход большим количеством средств. Простая регулярная схема обычно удобнее.",
        "Если кожа реагирует на новое средство, лучше прекратить его использование и не вводить сразу несколько новых продуктов.",
    ],

    "oily_skin": [
        "При жирной коже не стоит пытаться полностью пересушить лицо. Лучше использовать мягкое очищение и лёгкое увлажнение.",
        "Для жирной кожи обычно удобнее лёгкие средства, которые не создают ощущение сильной тяжести на коже.",
        "Если кожа становится раздражённой после ухода, сократи количество активных средств и вернись к базовой схеме.",
    ],

    "dry_skin": [
        "При сухости кожи особенно важны мягкое очищение и регулярное увлажнение.",
        "Старайся не использовать слишком горячую воду и не тереть кожу слишком сильно.",
        "Если новое средство вызывает жжение или выраженное раздражение, лучше его не использовать.",
    ],

    "sensitive_skin": [
        "Для чувствительной кожи лучше выбирать простой уход и вводить новые средства постепенно.",
        "Не стоит одновременно использовать много активных средств — так сложнее понять реакцию кожи.",
        "При выраженном или длительном раздражении лучше обратиться к врачу или дерматологу.",
    ],

    "hair": [
        "Для волос важно учитывать прежде всего состояние кожи головы. Подбирай средство для мытья по её потребностям.",
        "Не обязательно использовать большое количество средств для волос. Начни с подходящего шампуня и базового ухода.",
        "Если кожа головы постоянно раздражается или появляются необычные симптомы, лучше обратиться к специалисту.",
    ],

    "sleep": [
        "Для восстановления полезен стабильный режим сна и примерно одинаковое время отхода ко сну.",
        "Попробуй уменьшить яркий экран и активные занятия непосредственно перед сном.",
        "Если постоянно не получается нормально спать, стоит обсудить это с родителями или врачом.",
    ],

    "activity": [
        "Физическую активность лучше увеличивать постепенно и оставлять организму время на восстановление.",
        "Регулярная умеренная активность может быть хорошей частью здорового режима.",
        "Не нужно тренироваться через боль. Если появляется сильная боль или плохое самочувствие, остановись и сообщи взрослому.",
    ],

    "hygiene": [
        "Базовая гигиена включает регулярный душ, чистую одежду, уход за зубами и чистые руки.",
        "После активной тренировки полезно принять душ и переодеться в чистую одежду.",
        "Регулярный, но не чрезмерный уход обычно лучше постоянного использования большого количества средств.",
    ],

    "sun": [
        "При длительном нахождении на солнце полезно использовать подходящую защиту от UV-излучения.",
        "Старайся не проводить слишком много времени под сильным солнцем без защиты.",
        "Для ежедневного ухода можно рассмотреть солнцезащитное средство с подходящим SPF.",
    ],

    "general": [
        "Лучше менять привычки постепенно: базовый уход, сон, гигиена и комфортная физическая активность.",
        "Не нужно пытаться изменить всё одновременно. Выбери несколько простых привычек и поддерживай их регулярно.",
        "Если вопрос связан с выраженной проблемой кожи, волос или самочувствием, лучше обратиться к соответствующему специалисту.",
    ],
}


# ============================================================
# DATASET
# ============================================================
#
# Формат:
#
# {
#     "text": "...",
#     "intent": "skin"
# }
#
# Это НАШИ обучающие данные.
# Их можно постепенно увеличивать до тысяч примеров.
# ============================================================

DEFAULT_DATASET = [

    # --------------------------------------------------------
    # SKIN
    # --------------------------------------------------------

    {
        "text": "как ухаживать за кожей",
        "intent": "skin",
    },

    {
        "text": "что делать для ухода за лицом",
        "intent": "skin",
    },

    {
        "text": "хочу нормальный уход за кожей",
        "intent": "skin",
    },

    {
        "text": "как правильно умываться",
        "intent": "skin",
    },

    {
        "text": "какой базовый уход нужен",
        "intent": "skin",
    },

    # --------------------------------------------------------
    # OILY
    # --------------------------------------------------------

    {
        "text": "у меня жирная кожа",
        "intent": "oily_skin",
    },

    {
        "text": "лицо быстро становится жирным",
        "intent": "oily_skin",
    },

    {
        "text": "кожа очень жирная",
        "intent": "oily_skin",
    },

    {
        "text": "что делать если кожа жирная",
        "intent": "oily_skin",
    },

    {
        "text": "как ухаживать за жирным лицом",
        "intent": "oily_skin",
    },

    # --------------------------------------------------------
    # DRY
    # --------------------------------------------------------

    {
        "text": "у меня сухая кожа",
        "intent": "dry_skin",
    },

    {
        "text": "кожа постоянно сухая",
        "intent": "dry_skin",
    },

    {
        "text": "лицо сушит",
        "intent": "dry_skin",
    },

    {
        "text": "как ухаживать за сухой кожей",
        "intent": "dry_skin",
    },

    {
        "text": "что делать если кожа сухая",
        "intent": "dry_skin",
    },

    # --------------------------------------------------------
    # SENSITIVE
    # --------------------------------------------------------

    {
        "text": "у меня чувствительная кожа",
        "intent": "sensitive_skin",
    },

    {
        "text": "кожа раздражается от косметики",
        "intent": "sensitive_skin",
    },

    {
        "text": "лицо краснеет после средства",
        "intent": "sensitive_skin",
    },

    {
        "text": "кожа реагирует на уход",
        "intent": "sensitive_skin",
    },

    # --------------------------------------------------------
    # HAIR
    # --------------------------------------------------------

    {
        "text": "как ухаживать за волосами",
        "intent": "hair",
    },

    {
        "text": "какой уход нужен волосам",
        "intent": "hair",
    },

    {
        "text": "что делать с волосами",
        "intent": "hair",
    },

    {
        "text": "как правильно мыть волосы",
        "intent": "hair",
    },

    {
        "text": "хочу улучшить уход за волосами",
        "intent": "hair",
    },

    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    {
        "text": "как лучше спать",
        "intent": "sleep",
    },

    {
        "text": "я плохо сплю",
        "intent": "sleep",
    },

    {
        "text": "не могу нормально высыпаться",
        "intent": "sleep",
    },

    {
        "text": "как улучшить сон",
        "intent": "sleep",
    },

    {
        "text": "у меня мало сна",
        "intent": "sleep",
    },

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    {
        "text": "как начать заниматься спортом",
        "intent": "activity",
    },

    {
        "text": "хочу больше двигаться",
        "intent": "activity",
    },

    {
        "text": "какие упражнения делать",
        "intent": "activity",
    },

    {
        "text": "как начать тренироваться",
        "intent": "activity",
    },

    {
        "text": "хочу стать активнее",
        "intent": "activity",
    },

    # --------------------------------------------------------
    # HYGIENE
    # --------------------------------------------------------

    {
        "text": "как правильно соблюдать гигиену",
        "intent": "hygiene",
    },

    {
        "text": "как ухаживать за собой",
        "intent": "hygiene",
    },

    {
        "text": "что входит в базовую гигиену",
        "intent": "hygiene",
    },

    {
        "text": "как правильно принимать душ",
        "intent": "hygiene",
    },

    # --------------------------------------------------------
    # SUN
    # --------------------------------------------------------

    {
        "text": "нужен ли крем от солнца",
        "intent": "sun",
    },

    {
        "text": "как защищать кожу от солнца",
        "intent": "sun",
    },

    {
        "text": "что такое spf",
        "intent": "sun",
    },

    {
        "text": "как пользоваться солнцезащитным кремом",
        "intent": "sun",
    },

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    {
        "text": "как улучшить свой режим",
        "intent": "general",
    },

    {
        "text": "дай советы по уходу за собой",
        "intent": "general",
    },

    {
        "text": "как начать ухаживать за собой",
        "intent": "general",
    },

    {
        "text": "дай общие советы",
        "intent": "general",
    },

    {
        "text": "хочу улучшить свои привычки",
        "intent": "general",
    },
]


# ============================================================
# СОЗДАНИЕ DATASET.JSON
# ============================================================

def load_dataset():

    if not DATASET_FILE.exists():

        with open(
            DATASET_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                DEFAULT_DATASET,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return DEFAULT_DATASET

    try:

        with open(
            DATASET_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list) and data:

            return data

    except Exception:

        pass

    return DEFAULT_DATASET


DATASET = load_dataset()


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    text = text.lower()

    # Русские + английские слова + числа

    words = re.findall(
        r"[а-яёa-z0-9]+",
        text,
    )

    return words


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "и",
    "в",
    "во",
    "на",
    "за",
    "с",
    "со",
    "по",
    "для",
    "как",
    "что",
    "это",
    "у",
    "мне",
    "меня",
    "я",
    "хочу",
    "можно",
    "ли",
    "а",
    "но",
    "из",
    "к",
    "же",
    "бы",
    "не",
    "да",
    "есть",
    "мой",
    "моя",
    "мне",
    "очень",
}


# ============================================================
# VOCABULARY
# ============================================================

def build_vocabulary(dataset):

    vocabulary = set()

    for item in dataset:

        for word in tokenize(
            item["text"]
        ):

            if word not in STOP_WORDS:

                vocabulary.add(word)

    return sorted(vocabulary)


VOCABULARY = build_vocabulary(DATASET)

WORD_TO_INDEX = {
    word: index
    for index, word in enumerate(VOCABULARY)
}


# ============================================================
# TEXT → VECTOR
# ============================================================

def text_to_vector(text):

    vector = np.zeros(
        len(VOCABULARY),
        dtype=np.float32,
    )

    words = tokenize(text)

    for word in words:

        if word in WORD_TO_INDEX:

            vector[
                WORD_TO_INDEX[word]
            ] += 1.0

    # Нормализация

    if len(words) > 0:

        vector /= len(words)

    return vector


# ============================================================
# TARGET
# ============================================================

def intent_to_vector(intent):

    result = np.zeros(
        len(INTENTS),
        dtype=np.float32,
    )

    if intent in INTENTS:

        result[
            INTENTS.index(intent)
        ] = 1.0

    return result


# ============================================================
# НАША НЕЙРОСЕТЬ
# ============================================================

class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden1=64,
        hidden2=32,
        output_size=len(INTENTS),
    ):

        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

        # Xavier / He initialization

        self.W1 = (
            np.random.randn(
                input_size,
                hidden1,
            )
            * math.sqrt(
                2 / max(input_size, 1)
            )
        )

        self.b1 = np.zeros(hidden1)

        self.W2 = (
            np.random.randn(
                hidden1,
                hidden2,
            )
            * math.sqrt(
                2 / hidden1
            )
        )

        self.b2 = np.zeros(hidden2)

        self.W3 = (
            np.random.randn(
                hidden2,
                output_size,
            )
            * math.sqrt(
                2 / hidden2
            )
        )

        self.b3 = np.zeros(output_size)

    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    @staticmethod
    def relu(x):

        return np.maximum(
            0,
            x,
        )

    @staticmethod
    def relu_derivative(x):

        return (
            x > 0
        ).astype(float)

    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    @staticmethod
    def softmax(x):

        x = x - np.max(
            x,
            axis=1,
            keepdims=True,
        )

        exp_x = np.exp(x)

        return (
            exp_x
            / np.sum(
                exp_x,
                axis=1,
                keepdims=True,
            )
        )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(self, X):

        self.z1 = (
            X @ self.W1
            + self.b1
        )

        self.a1 = self.relu(
            self.z1
        )

        self.z2 = (
            self.a1 @ self.W2
            + self.b2
        )

        self.a2 = self.relu(
            self.z2
        )

        self.z3 = (
            self.a2 @ self.W3
            + self.b3
        )

        self.a3 = self.softmax(
            self.z3
        )

        return self.a3

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    def train(
        self,
        X,
        Y,
        epochs=1800,
        learning_rate=0.025,
    ):

        for epoch in range(
            epochs
        ):

            predictions = self.forward(X)

            batch_size = len(X)

            # ----------------------------------------------
            # LOSS
            # ----------------------------------------------

            eps = 1e-9

            loss = -np.mean(
                np.sum(
                    Y
                    * np.log(
                        predictions
                        + eps
                    ),
                    axis=1,
                )
            )

            # ----------------------------------------------
            # BACKPROP
            # ----------------------------------------------

            dz3 = (
                predictions - Y
            ) / batch_size

            dW3 = (
                self.a2.T
                @ dz3
            )

            db3 = np.sum(
                dz3,
                axis=0,
            )

            da2 = (
                dz3
                @ self.W3.T
            )

            dz2 = (
                da2
                * self.relu_derivative(
                    self.z2
                )
            )

            dW2 = (
                self.a1.T
                @ dz2
            )

            db2 = np.sum(
                dz2,
                axis=0,
            )

            da1 = (
                dz2
                @ self.W2.T
            )

            dz1 = (
                da1
                * self.relu_derivative(
                    self.z1
                )
            )

            dW1 = (
                X.T
                @ dz1
            )

            db1 = np.sum(
                dz1,
                axis=0,
            )

            # ----------------------------------------------
            # GRADIENT DESCENT
            # ----------------------------------------------

            self.W3 -= (
                learning_rate
                * dW3
            )

            self.b3 -= (
                learning_rate
                * db3
            )

            self.W2 -= (
                learning_rate
                * dW2
            )

            self.b2 -= (
                learning_rate
                * db2
            )

            self.W1 -= (
                learning_rate
                * dW1
            )

            self.b1 -= (
                learning_rate
                * db1
            )

            if epoch % 200 == 0:

                print(
                    f"[AI] epoch={epoch} "
                    f"loss={loss:.5f}"
                )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(self, X):

        return self.forward(X)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(self, filename):

        np.savez(
            filename,

            W1=self.W1,
            b1=self.b1,

            W2=self.W2,
            b2=self.b2,

            W3=self.W3,
            b3=self.b3,
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load(self, filename):

        data = np.load(
            filename
        )

        self.W1 = data["W1"]
        self.b1 = data["b1"]

        self.W2 = data["W2"]
        self.b2 = data["b2"]

        self.W3 = data["W3"]
        self.b3 = data["b3"]


# ============================================================
# СОЗДАЁМ МОДЕЛЬ
# ============================================================

model = NeuralNetwork(
    input_size=len(VOCABULARY),
    hidden1=64,
    hidden2=32,
    output_size=len(INTENTS),
)


# ============================================================
# TRAINING
# ============================================================

def prepare_training_data():

    X = []
    Y = []

    for item in DATASET:

        X.append(
            text_to_vector(
                item["text"]
            )
        )

        Y.append(
            intent_to_vector(
                item["intent"]
            )
        )

    return (
        np.array(
            X,
            dtype=np.float32,
        ),
        np.array(
            Y,
            dtype=np.float32,
        ),
    )


def train_model():

    global model

    print()
    print("=" * 60)
    print("AI CARE — TRAINING")
    print("=" * 60)

    # После изменения датасета
    # словарь также обновляется.

    global VOCABULARY
    global WORD_TO_INDEX

    VOCABULARY = build_vocabulary(
        DATASET
    )

    WORD_TO_INDEX = {
        word: index
        for index, word
        in enumerate(VOCABULARY)
    }

    X, Y = prepare_training_data()

    model = NeuralNetwork(
        input_size=len(VOCABULARY),
        hidden1=64,
        hidden2=32,
        output_size=len(INTENTS),
    )

    model.train(
        X,
        Y,
        epochs=1800,
        learning_rate=0.025,
    )

    model.save(
        MODEL_FILE
    )

    print(
        "Модель сохранена."
    )


# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

if MODEL_FILE.exists():

    try:

        model.load(
            MODEL_FILE
        )

        print(
            "[AI] модель загружена"
        )

    except Exception:

        print(
            "[AI] модель повреждена, обучение..."
        )

        train_model()

else:

    train_model()


# ============================================================
# ДОПОЛНИТЕЛЬНЫЕ ПОДСКАЗКИ
# ============================================================
#
# Это не другая нейросеть.
#
# Это небольшая страховка для очевидных слов.
# В дальнейшем её можно убрать, когда датасет станет большим.
# ============================================================

KEYWORD_HINTS = {

    "oily_skin": [
        "жирная кожа",
        "жирное лицо",
        "жирнит",
        "жирн",
    ],

    "dry_skin": [
        "сухая кожа",
        "сухое лицо",
        "сухость кожи",
        "сушит лицо",
    ],

    "sensitive_skin": [
        "чувствительная кожа",
        "раздражение",
        "раздражается кожа",
        "краснеет кожа",
    ],

    "hair": [
        "волосы",
        "волос",
        "шампунь",
        "прическа",
        "причёска",
    ],

    "sleep": [
        "сон",
        "спать",
        "выспаться",
        "не высыпаюсь",
    ],

    "activity": [
        "спорт",
        "тренировка",
        "тренироваться",
        "упражнения",
    ],

    "hygiene": [
        "гигиена",
        "душ",
        "зубы",
        "чистота",
    ],

    "sun": [
        "солнце",
        "spf",
        "ультрафиолет",
    ],

    "skin": [
        "кожа",
        "лицо",
        "умывание",
        "умыться",
    ],
}


# ============================================================
# ОПРЕДЕЛЕНИЕ INTENT
# ============================================================

def detect_keyword_intent(text):

    text_lower = text.lower()

    scores = {
        intent: 0
        for intent in INTENTS
    }

    for intent, words in KEYWORD_HINTS.items():

        for word in words:

            if word in text_lower:

                scores[intent] += 1

    best_intent = max(
        scores,
        key=scores.get,
    )

    if scores[best_intent] > 0:

        return (
            best_intent,
            1.0,
        )

    return (
        None,
        0.0,
    )


def predict_intent(text):

    keyword_intent, keyword_score = (
        detect_keyword_intent(text)
    )

    vector = text_to_vector(
        text
    )

    # Если вообще нет знакомых слов,
    # модель не сможет нормально классифицировать запрос.

    if np.sum(vector) == 0:

        return (
            "general",
            0.05,
        )

    prediction = model.predict(
        vector.reshape(1, -1)
    )[0]

    index = int(
        np.argmax(prediction)
    )

    neural_intent = INTENTS[index]

    neural_confidence = float(
        prediction[index]
    )

    # Для очевидных запросов
    # используем собственный keyword layer.

    if (
        keyword_intent
        and keyword_score > 0
    ):

        return (
            keyword_intent,
            max(
                keyword_score,
                neural_confidence,
            ),
        )

    return (
        neural_intent,
        neural_confidence,
    )


# ============================================================
# ГЕНЕРАЦИЯ ОТВЕТА
# ============================================================

def generate_answer(
    text,
    intent,
    confidence,
):

    options = RESPONSES.get(
        intent,
        RESPONSES["general"],
    )

    answer = random.choice(
        options
    )

    # Если модель не уверена,
    # честно сообщаем об этом.

    if confidence < 0.25:

        answer = (
            "Я пока не совсем уверен, "
            "что правильно понял запрос.\n\n"
            + answer
            + "\n\n"
            "Попробуй написать подробнее, например: "
            "«У меня жирная кожа, как построить базовый уход?»"
        )

    return answer


# ============================================================
# API MODELS
# ============================================================

class ChatRequest(BaseModel):

    message: str


class AddExampleRequest(BaseModel):

    text: str
    intent: str


# ============================================================
# /api/chat
# ============================================================

@app.post("/api/chat")
def api_chat(
    request: ChatRequest
):

    message = (
        request.message
        .strip()
    )

    if not message:

        return JSONResponse(
            {
                "success": False,
                "error": "Пустое сообщение.",
            },
            status_code=400,
        )

    if len(message) > 1000:

        return JSONResponse(
            {
                "success": False,
                "error": "Сообщение слишком длинное.",
            },
            status_code=400,
        )

    intent, confidence = (
        predict_intent(message)
    )

    answer = generate_answer(
        message,
        intent,
        confidence,
    )

    return {
        "success": True,

        "answer": answer,

        "intent": intent,

        "intent_name":
            INTENT_NAMES.get(
                intent,
                "Общее",
            ),

        "confidence":
            round(
                confidence * 100,
                1,
            ),
    }


# ============================================================
# /api/train
# ============================================================

@app.post("/api/train")
def api_train():

    try:

        train_model()

        return {
            "success": True,
            "message":
                "Нейросеть успешно переобучена.",
            "dataset_size":
                len(DATASET),
        }

    except Exception as error:

        return JSONResponse(
            {
                "success": False,
                "error": str(error),
            },
            status_code=500,
        )


# ============================================================
# /api/dataset
# ============================================================

@app.get("/api/dataset")
def api_dataset():

    return {
        "success": True,
        "count": len(DATASET),
        "data": DATASET,
    }


# ============================================================
# /api/add-example
# ============================================================

@app.post("/api/add-example")
def api_add_example(
    request: AddExampleRequest
):

    text = request.text.strip()

    intent = request.intent.strip()

    if not text:

        return JSONResponse(
            {
                "success": False,
                "error": "Текст пустой.",
            },
            status_code=400,
        )

    if intent not in INTENTS:

        return JSONResponse(
            {
                "success": False,
                "error": "Неизвестный intent.",
            },
            status_code=400,
        )

    example = {
        "text": text,
        "intent": intent,
    }

    DATASET.append(
        example
    )

    with open(
        DATASET_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            DATASET,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "success": True,
        "message":
            "Пример добавлен. Теперь переобучи модель.",
        "dataset_size":
            len(DATASET),
    }


# ============================================================
# ГЛАВНАЯ СТРАНИЦА
# ============================================================

HOME_HTML = r"""
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

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color: white;

    background:
        radial-gradient(
            circle at top left,
            #172554 0,
            #070b18 40%,
            #02030a 100%
        );
}

.wrapper {

    width:
        min(
            850px,
            calc(100% - 24px)
        );

    margin: auto;

    padding:
        25px 0 40px;
}

.header {

    text-align: center;

    padding:
        20px 0 25px;
}

.logo {

    font-size: 35px;

    font-weight: 800;

    letter-spacing: -1px;
}

.description {

    color: #94a3b8;

    margin-top: 7px;
}

.chat {

    height: 650px;

    display: flex;

    flex-direction: column;

    background:
        rgba(
            15,
            23,
            42,
            0.76
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.08
        );

    border-radius: 25px;

    overflow: hidden;

    backdrop-filter:
        blur(20px);

    box-shadow:
        0 20px 80px
        rgba(0,0,0,.35);
}

.messages {

    flex: 1;

    overflow-y: auto;

    padding: 20px;
}

.message {

    display: flex;

    margin-bottom: 14px;
}

.message.user {

    justify-content: flex-end;
}

.bubble {

    max-width: 78%;

    padding:
        13px 16px;

    border-radius: 18px;

    line-height: 1.5;

    white-space: pre-line;
}

.ai .bubble {

    background:
        rgba(
            30,
            41,
            59,
            0.9
        );

    border-bottom-left-radius: 5px;
}

.user .bubble {

    background:
        #2563eb;

    border-bottom-right-radius: 5px;
}

.info {

    font-size: 11px;

    color: #64748b;

    margin-top: 6px;
}

.input-area {

    display: flex;

    gap: 10px;

    padding: 15px;

    border-top:
        1px solid
        rgba(
            255,
            255,
            255,
            0.07
        );
}

textarea {

    flex: 1;

    resize: none;

    height: 52px;

    border-radius: 15px;

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.1
        );

    background:
        rgba(
            2,
            6,
            23,
            0.8
        );

    color: white;

    padding:
        14px;

    outline: none;

    font-family: inherit;

    font-size: 15px;
}

button {

    border: 0;

    border-radius: 15px;

    padding:
        0 20px;

    background:
        #2563eb;

    color: white;

    font-weight: 700;

    cursor: pointer;
}

button:hover {

    opacity: .9;
}

.status {

    text-align: center;

    color: #64748b;

    font-size: 12px;

    margin-top: 12px;
}

@media(max-width:600px) {

    .chat {

        height: 75vh;

        min-height: 520px;
    }

    .bubble {

        max-width: 88%;
    }

    .input-area {

        padding: 10px;
    }

    button {

        padding:
            0 14px;
    }
}

</style>

</head>


<body>

<div class="wrapper">

    <div class="header">

        <div class="logo">
            AI Care
        </div>

        <div class="description">
            Собственная нейросеть • без внешних AI API
        </div>

    </div>


    <div class="chat">

        <div
            class="messages"
            id="messages"
        >

            <div class="message ai">

                <div class="bubble">

                    Привет! Я AI Care.

                    Я работаю на собственной небольшой
                    нейросети, обученной на локальном датасете.

                    Напиши, что тебя интересует: кожа,
                    волосы, сон, гигиена или физическая активность.

                </div>

            </div>

        </div>


        <div class="input-area">

            <textarea
                id="input"
                placeholder="Напиши сообщение..."
                onkeydown="handleKey(event)"
            ></textarea>

            <button
                onclick="sendMessage()"
            >
                ➤
            </button>

        </div>

    </div>


    <div class="status">

        AI Care работает без OpenAI, Gemini и других
        внешних AI API.

    </div>

</div>


<script>

const input =
    document.getElementById(
        "input"
    );

const messages =
    document.getElementById(
        "messages"
    );


function addMessage(
    text,
    type,
    info = ""
) {

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        "message " + type;


    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        "bubble";


    bubble.textContent =
        text;


    if (info) {

        const small =
            document.createElement(
                "div"
            );

        small.className =
            "info";

        small.textContent =
            info;

        bubble.appendChild(
            small
        );
    }


    wrapper.appendChild(
        bubble
    );

    messages.appendChild(
        wrapper
    );


    messages.scrollTop =
        messages.scrollHeight;
}


function handleKey(event) {

    if (
        event.key === "Enter"
        &&
        !event.shiftKey
    ) {

        event.preventDefault();

        sendMessage();
    }
}


async function sendMessage() {

    const text =
        input.value.trim();


    if (!text) {

        return;
    }


    addMessage(
        text,
        "user"
    );


    input.value = "";


    addMessage(
        "Думаю...",
        "ai"
    );


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

                    body:
                        JSON.stringify({
                            message: text
                        })
                }
            );


        const data =
            await response.json();


        // Удаляем "Думаю..."

        const all =
            messages.querySelectorAll(
                ".message.ai"
            );

        const last =
            all[all.length - 1];


        if (
            last
            &&
            last
                .querySelector(
                    ".bubble"
                )
                ?.textContent
                .startsWith(
                    "Думаю"
                )
        ) {

            last.remove();
        }


        if (
            data.success
        ) {

            addMessage(

                data.answer,

                "ai",

                data.intent_name
                +
                " • уверенность "
                +
                data.confidence
                +
                "%"

            );

        } else {

            addMessage(
                data.error ||
                "Произошла ошибка.",
                "ai"
            );
        }

    }

    catch(error) {

        const all =
            messages.querySelectorAll(
                ".message.ai"
            );

        const last =
            all[all.length - 1];

        if (last) {

            last.remove();
        }

        addMessage(
            "Не удалось подключиться к серверу.",
            "ai"
        );

        console.error(
            error
        );
    }
}

</script>

</body>

</html>
"""


# ============================================================
# ADMIN HTML
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

    background: #050816;

    color: white;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

.container {

    max-width: 900px;

    margin: auto;
}

.card {

    background:
        #0f172a;

    border:
        1px solid
        #1e293b;

    border-radius: 18px;

    padding: 20px;

    margin-bottom: 18px;
}

input,
select {

    width: 100%;

    padding: 12px;

    margin:
        6px 0 12px;

    border-radius: 10px;

    border:
        1px solid #334155;

    background: #020617;

    color: white;

    box-sizing: border-box;
}

button {

    padding:
        12px 18px;

    border: 0;

    border-radius: 10px;

    background: #2563eb;

    color: white;

    cursor: pointer;

    font-weight: 700;
}

pre {

    white-space: pre-wrap;

    max-height: 400px;

    overflow: auto;

    background: #020617;

    padding: 15px;

    border-radius: 10px;
}

</style>

</head>

<body>

<div class="container">

<h1>AI Care — Admin</h1>


<div class="card">

<h2>Обучение</h2>

<p>
Количество примеров:
<span id="count">...</span>
</p>

<button onclick="train()">
Переобучить нейросеть
</button>

<p id="trainResult"></p>

</div>


<div class="card">

<h2>Добавить обучающий пример</h2>

<input
    id="example"
    placeholder="Например: как ухаживать за жирной кожей"
/>


<select id="intent">

<option value="skin">
Уход за кожей
</option>

<option value="oily_skin">
Жирная кожа
</option>

<option value="dry_skin">
Сухая кожа
</option>

<option value="sensitive_skin">
Чувствительная кожа
</option>

<option value="hair">
Волосы
</option>

<option value="sleep">
Сон
</option>

<option value="activity">
Активность
</option>

<option value="hygiene">
Гигиена
</option>

<option value="sun">
Солнце
</option>

<option value="general">
Общее
</option>

</select>


<button onclick="addExample()">
Добавить
</button>

<p id="addResult"></p>

</div>


<div class="card">

<h2>Датасет</h2>

<pre id="dataset">
Загрузка...
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
        data.count;

    document.getElementById(
        "dataset"
    ).textContent =
        JSON.stringify(
            data.data,
            null,
            2
        );
}


async function train() {

    const result =
        document.getElementById(
            "trainResult"
        );

    result.textContent =
        "Обучение...";


    const response =
        await fetch(
            "/api/train",
            {
                method: "POST"
            }
        );


    const data =
        await response.json();


    result.textContent =
        data.message ||
        data.error;

    loadDataset();
}


async function addExample() {

    const text =
        document.getElementById(
            "example"
        ).value;

    const intent =
        document.getElementById(
            "intent"
        ).value;


    const response =
        await fetch(
            "/api/add-example",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        text,
                        intent
                    })
            }
        );


    const data =
        await response.json();


    document.getElementById(
        "addResult"
    ).textContent =
        data.message ||
        data.error;


    loadDataset();
}


loadDataset();

</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():

    return HOME_HTML


@app.get(
    "/admin",
    response_class=HTMLResponse,
)
def admin():

    return ADMIN_HTML


@app.get(
    "/train",
    response_class=HTMLResponse,
)
def train_page():

    return ADMIN_HTML


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "ai": "AI Care",
        "model": "custom MLP",
        "external_ai_api": False,
        "photo_processing": False,
        "dataset_examples":
            len(DATASET),
    }


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
