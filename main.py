# ============================================================
# AI CARE — СОБСТВЕННАЯ НЕЙРОСЕТЬ С НУЛЯ
# Один файл: main.py
#
# Запуск:
#   pip install fastapi uvicorn numpy
#   python main.py
#
# Открой:
#   http://127.0.0.1:8000
#
# Render:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# ============================================================

import os
import json
import random
import math
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "ai_model.npz"

app = FastAPI(title="AI Care")


# ============================================================
# НАШ ДАТАСЕТ
# ============================================================
#
# Мы сами задаём примеры, на которых модель учится.
#
# В дальнейшем сюда можно добавить тысячи собственных примеров.
#
# labels:
# 0 = базовый уход за кожей
# 1 = уход при жирной коже
# 2 = уход при сухой коже
# 3 = базовый уход за волосами
# 4 = сон и восстановление
# 5 = безопасная физическая активность
# 6 = гигиена
# 7 = защита от солнца
# 8 = общий уход
# 9 = запрос о нескольких направлениях
# ============================================================

LABELS = [
    "skin_basic",
    "skin_oily",
    "skin_dry",
    "hair_basic",
    "sleep",
    "activity",
    "hygiene",
    "sun",
    "general",
    "combined",
]

LABEL_TEXT = {
    "skin_basic": {
        "title": "Базовый уход за кожей",
        "text": (
            "Начни с простого ухода: мягкое очищение, подходящий "
            "увлажняющий крем и защита кожи от солнца."
        ),
    },

    "skin_oily": {
        "title": "Уход за жирной кожей",
        "text": (
            "Для жирной кожи лучше не пересушивать лицо. Используй "
            "мягкое очищение и лёгкий увлажняющий крем. Если кожа "
            "раздражается, уменьши количество активных средств."
        ),
    },

    "skin_dry": {
        "title": "Уход за сухой кожей",
        "text": (
            "При сухости особенно важно мягкое очищение и регулярное "
            "увлажнение. Избегай слишком горячей воды и агрессивного "
            "скрабирования."
        ),
    },

    "hair_basic": {
        "title": "Уход за волосами",
        "text": (
            "Подбирай шампунь по состоянию кожи головы, не мой волосы "
            "слишком агрессивно и следи за тем, чтобы кожа головы "
            "не пересыхала."
        ),
    },

    "sleep": {
        "title": "Сон и восстановление",
        "text": (
            "Регулярный сон помогает восстановлению организма. "
            "Старайся придерживаться примерно одинакового времени "
            "отхода ко сну и пробуждения."
        ),
    },

    "activity": {
        "title": "Физическая активность",
        "text": (
            "Регулярная умеренная активность полезна для общего "
            "самочувствия. Нагрузку лучше увеличивать постепенно, "
            "оставляя время на восстановление."
        ),
    },

    "hygiene": {
        "title": "Гигиена",
        "text": (
            "Базовая гигиена — регулярное умывание, душ после активной "
            "нагрузки, чистая одежда и уход за зубами."
        ),
    },

    "sun": {
        "title": "Защита от солнца",
        "text": (
            "При длительном нахождении на солнце используй подходящую "
            "защиту от UV-излучения и старайся не находиться под "
            "сильным солнцем слишком долго."
        ),
    },

    "general": {
        "title": "Общий уход",
        "text": (
            "Лучший вариант — не пытаться изменить всё сразу. "
            "Сначала наладь базовый уход, сон, гигиену и комфортную "
            "физическую активность."
        ),
    },

    "combined": {
        "title": "Комплексная рекомендация",
        "text": (
            "Можно работать сразу над несколькими направлениями: "
            "базовый уход за кожей, волосы, сон, гигиена и регулярная "
            "умеренная активность."
        ),
    },
}


# ============================================================
# ВХОДНЫЕ ПРИЗНАКИ
# ============================================================

SKIN = [
    "unknown",
    "oily",
    "dry",
    "combination",
    "normal",
]

SENSITIVITY = [
    "unknown",
    "normal",
    "sensitive",
]

HAIR = [
    "unknown",
    "normal",
    "oily",
    "dry",
]

SLEEP = [
    "unknown",
    "less7",
    "7to9",
    "more9",
]

ACTIVITY = [
    "unknown",
    "low",
    "medium",
    "high",
]

GOAL = [
    "general",
    "skin",
    "hair",
    "sleep",
    "activity",
    "hygiene",
    "sun",
]


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def one_hot(value, values):
    result = [0.0] * len(values)

    if value not in values:
        value = values[0]

    result[values.index(value)] = 1.0
    return result


def encode_input(data):
    """
    Превращает анкету пользователя в числовой вектор.
    """

    vector = []

    vector += one_hot(data.get("skin", "unknown"), SKIN)
    vector += one_hot(data.get("sensitivity", "unknown"), SENSITIVITY)
    vector += one_hot(data.get("hair", "unknown"), HAIR)
    vector += one_hot(data.get("sleep", "unknown"), SLEEP)
    vector += one_hot(data.get("activity", "unknown"), ACTIVITY)
    vector += one_hot(data.get("goal", "general"), GOAL)

    return np.array(vector, dtype=np.float32)


INPUT_SIZE = (
    len(SKIN)
    + len(SENSITIVITY)
    + len(HAIR)
    + len(SLEEP)
    + len(ACTIVITY)
    + len(GOAL)
)

OUTPUT_SIZE = len(LABELS)


# ============================================================
# НЕЙРОСЕТЬ
# ============================================================
#
# Архитектура:
#
# INPUT
#   ↓
# 32 нейрона
#   ↓
# 32 нейрона
#   ↓
# OUTPUT
#
# Обучение:
#   собственный forward pass
#   собственный backpropagation
#   собственный gradient descent
#
# Никакого готового AI API.
# ============================================================

class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden1=32,
        hidden2=32,
        output_size=10,
    ):

        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

        # Xavier initialization

        self.W1 = (
            np.random.randn(input_size, hidden1)
            * math.sqrt(2 / input_size)
        )

        self.b1 = np.zeros(hidden1)

        self.W2 = (
            np.random.randn(hidden1, hidden2)
            * math.sqrt(2 / hidden1)
        )

        self.b2 = np.zeros(hidden2)

        self.W3 = (
            np.random.randn(hidden2, output_size)
            * math.sqrt(2 / hidden2)
        )

        self.b3 = np.zeros(output_size)

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    # --------------------------------------------------------
    # Sigmoid
    # --------------------------------------------------------

    @staticmethod
    def sigmoid(x):

        x = np.clip(x, -50, 50)

        return 1 / (1 + np.exp(-x))

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(self, X):

        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    def train(
        self,
        X,
        Y,
        epochs=2500,
        learning_rate=0.03,
    ):

        for epoch in range(epochs):

            predictions = self.forward(X)

            # Binary cross entropy

            eps = 1e-8

            loss = -np.mean(
                Y * np.log(predictions + eps)
                + (1 - Y)
                * np.log(1 - predictions + eps)
            )

            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            batch_size = len(X)

            dz3 = (
                predictions - Y
            ) / batch_size

            dW3 = self.a2.T @ dz3
            db3 = np.sum(dz3, axis=0)

            da2 = dz3 @ self.W3.T

            dz2 = (
                da2
                * self.relu_derivative(self.z2)
            )

            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0)

            da1 = dz2 @ self.W2.T

            dz1 = (
                da1
                * self.relu_derivative(self.z1)
            )

            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)

            # ------------------------------------------------
            # GRADIENT DESCENT
            # ------------------------------------------------

            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3

            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

            if epoch % 250 == 0:

                print(
                    f"Epoch {epoch}/{epochs} "
                    f"| Loss: {loss:.5f}"
                )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(self, X):

        output = self.forward(X)

        return output

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

        data = np.load(filename)

        self.W1 = data["W1"]
        self.b1 = data["b1"]

        self.W2 = data["W2"]
        self.b2 = data["b2"]

        self.W3 = data["W3"]
        self.b3 = data["b3"]


# ============================================================
# СОЗДАНИЕ ОБУЧАЮЩИХ ДАННЫХ
# ============================================================

def make_example(
    skin="unknown",
    sensitivity="unknown",
    hair="unknown",
    sleep="unknown",
    activity="unknown",
    goal="general",
    labels=None,
):

    return {
        "input": {
            "skin": skin,
            "sensitivity": sensitivity,
            "hair": hair,
            "sleep": sleep,
            "activity": activity,
            "goal": goal,
        },
        "labels": labels or ["general"],
    }


def build_dataset():

    data = []

    # --------------------------------------------------------
    # SKIN
    # --------------------------------------------------------

    for skin in ["normal", "combination"]:

        data.append(
            make_example(
                skin=skin,
                goal="skin",
                labels=["skin_basic", "sun"],
            )
        )

    for skin in ["oily"]:

        data.append(
            make_example(
                skin=skin,
                goal="skin",
                labels=["skin_oily", "skin_basic"],
            )
        )

    for skin in ["dry"]:

        data.append(
            make_example(
                skin=skin,
                goal="skin",
                labels=["skin_dry", "skin_basic"],
            )
        )

    # --------------------------------------------------------
    # SENSITIVE
    # --------------------------------------------------------

    data.append(
        make_example(
            sensitivity="sensitive",
            goal="skin",
            labels=["skin_basic"],
        )
    )

    data.append(
        make_example(
            skin="dry",
            sensitivity="sensitive",
            goal="skin",
            labels=["skin_dry"],
        )
    )

    data.append(
        make_example(
            skin="oily",
            sensitivity="sensitive",
            goal="skin",
            labels=["skin_oily"],
        )
    )

    # --------------------------------------------------------
    # HAIR
    # --------------------------------------------------------

    data.append(
        make_example(
            hair="normal",
            goal="hair",
            labels=["hair_basic"],
        )
    )

    data.append(
        make_example(
            hair="oily",
            goal="hair",
            labels=["hair_basic"],
        )
    )

    data.append(
        make_example(
            hair="dry",
            goal="hair",
            labels=["hair_basic"],
        )
    )

    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    data.append(
        make_example(
            sleep="less7",
            goal="sleep",
            labels=["sleep"],
        )
    )

    data.append(
        make_example(
            sleep="7to9",
            goal="sleep",
            labels=["sleep"],
        )
    )

    data.append(
        make_example(
            sleep="more9",
            goal="sleep",
            labels=["sleep"],
        )
    )

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    data.append(
        make_example(
            activity="low",
            goal="activity",
            labels=["activity"],
        )
    )

    data.append(
        make_example(
            activity="medium",
            goal="activity",
            labels=["activity"],
        )
    )

    data.append(
        make_example(
            activity="high",
            goal="activity",
            labels=["activity"],
        )
    )

    # --------------------------------------------------------
    # HYGIENE
    # --------------------------------------------------------

    data.append(
        make_example(
            goal="hygiene",
            labels=["hygiene"],
        )
    )

    # --------------------------------------------------------
    # SUN
    # --------------------------------------------------------

    data.append(
        make_example(
            goal="sun",
            labels=["sun"],
        )
    )

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    data.append(
        make_example(
            goal="general",
            labels=["general"],
        )
    )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    data.append(
        make_example(
            skin="oily",
            hair="oily",
            sleep="less7",
            activity="low",
            goal="general",
            labels=[
                "skin_oily",
                "hair_basic",
                "sleep",
                "activity",
            ],
        )
    )

    data.append(
        make_example(
            skin="dry",
            hair="dry",
            sleep="less7",
            goal="general",
            labels=[
                "skin_dry",
                "hair_basic",
                "sleep",
            ],
        )
    )

    data.append(
        make_example(
            skin="combination",
            hair="normal",
            sleep="7to9",
            activity="medium",
            goal="general",
            labels=[
                "skin_basic",
                "hair_basic",
                "general",
            ],
        )
    )

    return data


DATASET = build_dataset()


# ============================================================
# DATASET → NUMPY
# ============================================================

def prepare_dataset():

    X = []
    Y = []

    for item in DATASET:

        x = encode_input(item["input"])

        y = np.zeros(OUTPUT_SIZE)

        for label in item["labels"]:

            if label in LABELS:

                index = LABELS.index(label)

                y[index] = 1.0

        X.append(x)
        Y.append(y)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
    )


# ============================================================
# СОЗДАНИЕ / ОБУЧЕНИЕ МОДЕЛИ
# ============================================================

model = NeuralNetwork(
    input_size=INPUT_SIZE,
    hidden1=32,
    hidden2=32,
    output_size=OUTPUT_SIZE,
)


def train_model():

    print()
    print("=" * 60)
    print("ОБУЧЕНИЕ СОБСТВЕННОЙ НЕЙРОСЕТИ")
    print("=" * 60)

    X, Y = prepare_dataset()

    model.train(
        X,
        Y,
        epochs=2500,
        learning_rate=0.03,
    )

    model.save(MODEL_FILE)

    print()
    print("Модель сохранена:", MODEL_FILE)
    print()


if os.path.exists(MODEL_FILE):

    try:

        model.load(MODEL_FILE)

        print("Загружена обученная модель.")

    except Exception:

        print("Не удалось загрузить модель.")
        train_model()

else:

    train_model()


# ============================================================
# TEXT → FEATURES
# ============================================================
#
# Никакого внешнего NLP API.
#
# Это наш простой обработчик текста.
#
# В дальнейшем его тоже можно заменить собственной
# маленькой языковой нейросетью.
# ============================================================

def understand_text(text):

    text = text.lower()

    result = {
        "skin": "unknown",
        "sensitivity": "unknown",
        "hair": "unknown",
        "sleep": "unknown",
        "activity": "unknown",
        "goal": "general",
    }

    # --------------------------------------------------------
    # SKIN
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "жирная кожа",
            "жирную кожу",
            "жирн",
            "сальный",
            "сальная",
        ]
    ):
        result["skin"] = "oily"

    elif any(
        word in text
        for word in [
            "сухая кожа",
            "сухую кожу",
            "сухость",
        ]
    ):
        result["skin"] = "dry"

    elif any(
        word in text
        for word in [
            "комбинирован",
        ]
    ):
        result["skin"] = "combination"

    elif any(
        word in text
        for word in [
            "нормальная кожа",
            "нормальную кожу",
        ]
    ):
        result["skin"] = "normal"

    # --------------------------------------------------------
    # SENSITIVITY
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "чувствительная",
            "чувствительную",
            "раздражается",
            "раздражение",
        ]
    ):
        result["sensitivity"] = "sensitive"

    # --------------------------------------------------------
    # HAIR
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "жирные волосы",
            "жирные волос",
        ]
    ):
        result["hair"] = "oily"

    elif any(
        word in text
        for word in [
            "сухие волосы",
            "сухие волос",
        ]
    ):
        result["hair"] = "dry"

    elif any(
        word in text
        for word in [
            "волосы",
            "волос",
        ]
    ):
        result["hair"] = "normal"

    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "мало сплю",
            "не высыпаюсь",
            "не высыпаю",
            "мало сна",
            "сон плохой",
        ]
    ):
        result["sleep"] = "less7"

    elif any(
        word in text
        for word in [
            "хорошо сплю",
            "нормально сплю",
        ]
    ):
        result["sleep"] = "7to9"

    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "не тренируюсь",
            "не занимаюсь",
            "мало двигаюсь",
        ]
    ):
        result["activity"] = "low"

    elif any(
        word in text
        for word in [
            "тренируюсь",
            "тренировки",
            "спорт",
            "занимаюсь спортом",
        ]
    ):
        result["activity"] = "medium"

    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "кожа",
            "лицо",
            "умыван",
            "уход за лицом",
        ]
    ):
        result["goal"] = "skin"

    elif any(
        word in text
        for word in [
            "волос",
            "причёск",
            "прическ",
        ]
    ):
        result["goal"] = "hair"

    elif any(
        word in text
        for word in [
            "сон",
            "спать",
            "высып",
        ]
    ):
        result["goal"] = "sleep"

    elif any(
        word in text
        for word in [
            "спорт",
            "трениров",
            "физическ",
        ]
    ):
        result["goal"] = "activity"

    elif any(
        word in text
        for word in [
            "гигиен",
            "зуб",
            "душ",
        ]
    ):
        result["goal"] = "hygiene"

    elif any(
        word in text
        for word in [
            "солнц",
            "spf",
            "ультрафиолет",
        ]
    ):
        result["goal"] = "sun"

    return result


# ============================================================
# ПОЛУЧЕНИЕ РЕКОМЕНДАЦИЙ
# ============================================================

def get_recommendations(data):

    x = encode_input(data)

    probabilities = model.predict(
        x.reshape(1, -1)
    )[0]

    # Сортируем результаты от наиболее вероятного
    # к наименее вероятному.

    indexes = np.argsort(probabilities)[::-1]

    recommendations = []

    for index in indexes:

        probability = float(
            probabilities[index]
        )

        label = LABELS[index]

        if probability >= 0.30:

            recommendations.append(
                {
                    "label": label,
                    "title": LABEL_TEXT[label]["title"],
                    "text": LABEL_TEXT[label]["text"],
                    "confidence": round(
                        probability * 100,
                        1,
                    ),
                }
            )

    # Если модель ничего не выбрала

    if not recommendations:

        best = indexes[0]

        label = LABELS[best]

        recommendations.append(
            {
                "label": label,
                "title": LABEL_TEXT[label]["title"],
                "text": LABEL_TEXT[label]["text"],
                "confidence": round(
                    float(probabilities[best]) * 100,
                    1,
                ),
            }
        )

    # Не отдаём слишком много пунктов

    return recommendations[:4]


# ============================================================
# API MODELS
# ============================================================

class UserData(BaseModel):

    skin: str = "unknown"
    sensitivity: str = "unknown"
    hair: str = "unknown"
    sleep: str = "unknown"
    activity: str = "unknown"
    goal: str = "general"


class ChatData(BaseModel):

    message: str


# ============================================================
# API — АНКЕТА
# ============================================================

@app.post("/api/recommend")
def recommend(data: UserData):

    recommendations = get_recommendations(
        data.model_dump()
    )

    return {
        "success": True,
        "recommendations": recommendations,
    }


# ============================================================
# API — ЧАТ
# ============================================================

@app.post("/api/chat")
def chat(data: ChatData):

    features = understand_text(
        data.message
    )

    recommendations = get_recommendations(
        features
    )

    # --------------------------------------------------------
    # Формируем ответ
    # --------------------------------------------------------

    if len(recommendations) == 1:

        answer = (
            recommendations[0]["text"]
        )

    else:

        parts = []

        for item in recommendations[:3]:

            parts.append(
                f"**{item['title']}**\n"
                f"{item['text']}"
            )

        answer = "\n\n".join(parts)

    return {
        "success": True,
        "message": answer,
        "detected": features,
        "recommendations": recommendations,
    }


# ============================================================
# API — ПЕРЕОБУЧЕНИЕ
# ============================================================

@app.post("/api/train")
def api_train():

    train_model()

    return {
        "success": True,
        "message": "Нейросеть переобучена.",
    }


# ============================================================
# FRONTEND
# ============================================================

HTML = r"""
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

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top,
            #172554,
            #050816 45%,
            #02030a
        );

    color: white;

    min-height: 100vh;
}

.container {

    width: min(
        900px,
        calc(100% - 30px)
    );

    margin: auto;

    padding: 40px 0 60px;
}

.logo {

    font-size: 34px;

    font-weight: 800;

    margin-bottom: 8px;
}

.subtitle {

    color: #94a3b8;

    margin-bottom: 30px;
}

.card {

    background:
        rgba(15, 23, 42, 0.72);

    border:
        1px solid
        rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 24px;

    margin-bottom: 20px;

    backdrop-filter:
        blur(20px);
}

h2 {

    margin-top: 0;
}

.grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 15px;
}

label {

    display: block;

    color: #cbd5e1;

    font-size: 14px;

    margin-bottom: 7px;
}

select,
textarea {

    width: 100%;

    padding: 13px;

    border-radius: 14px;

    border:
        1px solid
        rgba(255,255,255,0.1);

    background: #0f172a;

    color: white;

    outline: none;

    font-size: 15px;
}

textarea {

    min-height: 110px;

    resize: vertical;
}

button {

    border: 0;

    border-radius: 14px;

    padding: 14px 20px;

    background: #2563eb;

    color: white;

    font-size: 15px;

    font-weight: 700;

    cursor: pointer;

    margin-top: 15px;
}

button:hover {

    opacity: 0.9;
}

.result {

    margin-top: 20px;
}

.recommendation {

    padding: 18px;

    border-radius: 18px;

    background:
        rgba(30,41,59,0.7);

    border:
        1px solid
        rgba(255,255,255,0.07);

    margin-top: 12px;
}

.recommendation h3 {

    margin:
        0 0 8px;
}

.confidence {

    color: #60a5fa;

    font-size: 13px;

    margin-top: 8px;
}

.message {

    white-space: pre-line;

    line-height: 1.6;
}

.small {

    color: #64748b;

    font-size: 12px;

    margin-top: 25px;

    line-height: 1.5;
}

@media(max-width: 650px) {

    .grid {

        grid-template-columns: 1fr;
    }

    .container {

        padding-top: 25px;
    }

    .card {

        padding: 18px;
    }

    .logo {

        font-size: 28px;
    }
}

</style>

</head>

<body>

<div class="container">

    <div class="logo">
        AI Care
    </div>

    <div class="subtitle">
        Собственная нейросеть для персональных рекомендаций
    </div>


    <!-- ================================================== -->
    <!-- АНКЕТА -->
    <!-- ================================================== -->

    <div class="card">

        <h2>
            Персональная рекомендация
        </h2>

        <div class="grid">

            <div>

                <label>
                    Тип кожи
                </label>

                <select id="skin">

                    <option value="unknown">
                        Не знаю
                    </option>

                    <option value="oily">
                        Жирная
                    </option>

                    <option value="dry">
                        Сухая
                    </option>

                    <option value="combination">
                        Комбинированная
                    </option>

                    <option value="normal">
                        Нормальная
                    </option>

                </select>

            </div>


            <div>

                <label>
                    Чувствительность кожи
                </label>

                <select id="sensitivity">

                    <option value="unknown">
                        Не знаю
                    </option>

                    <option value="normal">
                        Обычная
                    </option>

                    <option value="sensitive">
                        Чувствительная
                    </option>

                </select>

            </div>


            <div>

                <label>
                    Волосы
                </label>

                <select id="hair">

                    <option value="unknown">
                        Не знаю
                    </option>

                    <option value="normal">
                        Обычные
                    </option>

                    <option value="oily">
                        Склонные к жирности
                    </option>

                    <option value="dry">
                        Сухие
                    </option>

                </select>

            </div>


            <div>

                <label>
                    Сон
                </label>

                <select id="sleep">

                    <option value="unknown">
                        Не знаю
                    </option>

                    <option value="less7">
                        Меньше 7 часов
                    </option>

                    <option value="7to9">
                        7–9 часов
                    </option>

                    <option value="more9">
                        Больше 9 часов
                    </option>

                </select>

            </div>


            <div>

                <label>
                    Активность
                </label>

                <select id="activity">

                    <option value="unknown">
                        Не знаю
                    </option>

                    <option value="low">
                        Низкая
                    </option>

                    <option value="medium">
                        Средняя
                    </option>

                    <option value="high">
                        Высокая
                    </option>

                </select>

            </div>


            <div>

                <label>
                    Что интересует?
                </label>

                <select id="goal">

                    <option value="general">
                        Общее
                    </option>

                    <option value="skin">
                        Кожа
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
                        Защита от солнца
                    </option>

                </select>

            </div>

        </div>

        <button onclick="getRecommendation()">
            Получить рекомендацию
        </button>

        <div id="result"></div>

    </div>


    <!-- ================================================== -->
    <!-- ЧАТ -->
    <!-- ================================================== -->

    <div class="card">

        <h2>
            AI Care Chat
        </h2>

        <p class="subtitle">
            Напиши, что тебя интересует обычным текстом.
        </p>

        <textarea
            id="chatInput"
            placeholder="Например: у меня жирная кожа и я хочу подобрать базовый уход..."
        ></textarea>

        <button onclick="sendMessage()">
            Спросить AI
        </button>

        <div id="chatResult"></div>

    </div>


    <div class="small">

        Эта версия использует собственную небольшую нейросеть,
        обученную на локальном датасете. Она не использует
        OpenAI, Gemini или другие AI API и не анализирует фотографии.

        Рекомендации являются общими и не заменяют консультацию
        врача или другого квалифицированного специалиста.

    </div>

</div>


<script>

async function getRecommendation() {

    const result =
        document.getElementById("result");

    result.innerHTML =
        "<p>Нейросеть думает...</p>";

    const data = {

        skin:
            document.getElementById("skin").value,

        sensitivity:
            document.getElementById("sensitivity").value,

        hair:
            document.getElementById("hair").value,

        sleep:
            document.getElementById("sleep").value,

        activity:
            document.getElementById("activity").value,

        goal:
            document.getElementById("goal").value
    };


    try {

        const response =
            await fetch(
                "/api/recommend",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)
                }
            );


        const json =
            await response.json();


        result.innerHTML = "";


        json.recommendations
            .forEach(item => {

                const div =
                    document.createElement("div");

                div.className =
                    "recommendation";

                div.innerHTML = `

                    <h3>
                        ${item.title}
                    </h3>

                    <div class="message">
                        ${item.text}
                    </div>

                    <div class="confidence">
                        Уверенность модели:
                        ${item.confidence}%
                    </div>

                `;

                result.appendChild(div);

            });

    }

    catch(error) {

        result.innerHTML =
            "<p>Ошибка соединения с сервером.</p>";

        console.error(error);
    }
}


async function sendMessage() {

    const input =
        document.getElementById("chatInput");

    const result =
        document.getElementById("chatResult");

    const message =
        input.value.trim();


    if (!message) {

        result.innerHTML =
            "<p>Напиши сообщение.</p>";

        return;
    }


    result.innerHTML =
        "<p>Нейросеть анализирует запрос...</p>";


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
                            message: message
                        })
                }
            );


        const json =
            await response.json();


        result.innerHTML = `

            <div class="recommendation">

                <h3>
                    AI Care
                </h3>

                <div class="message">
                    ${json.message}
                </div>

            </div>

        `;

    }

    catch(error) {

        result.innerHTML =
            "<p>Ошибка соединения с сервером.</p>";

        console.error(error);
    }
}

</script>

</body>

</html>
"""


# ============================================================
# ГЛАВНАЯ СТРАНИЦА
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():

    return HTML


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
