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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from supabase import create_client, Client


# ============================================================
# AI CARE V6
# СОБСТВЕННАЯ RNN + ADAM + ПАМЯТЬ + RETRIEVAL
# + CHECKPOINTS + EVALUATION + 100000 ЭПОХ
# SUPABASE DATABASE
# БЕЗ SUPABASE STORAGE
# ============================================================

APP_NAME = "AI Care v6"

BASE_DIR = Path(__file__).resolve().parent

LOCAL_MODEL = BASE_DIR / "model.npz"
LOCAL_DATASET = BASE_DIR / "dataset.json"
LOCAL_EVALUATION = BASE_DIR / "evaluation.json"

HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "128"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.003"))
MIN_LEARNING_RATE = float(os.getenv("MIN_LEARNING_RATE", "0.0002"))
LEARNING_RATE_DECAY = float(os.getenv("LEARNING_RATE_DECAY", "0.99995"))

GRADIENT_CLIP = float(os.getenv("GRADIENT_CLIP", "5.0"))

MAX_CONTEXT_MESSAGES = int(
    os.getenv("MAX_CONTEXT_MESSAGES", "8")
)

MAX_MESSAGE_LENGTH = 1000
MAX_RESPONSE_LENGTH = 100

DEFAULT_TEMPERATURE = 0.8
RETRIEVAL_THRESHOLD = 0.55

MAX_TRAIN_EPOCHS = 100000


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "",
).strip()

SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "",
).strip()

ADMIN_TOKEN = os.getenv(
    "ADMIN_TOKEN",
    "",
).strip()

supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase = create_client(
            SUPABASE_URL,
            SUPABASE_SERVICE_ROLE_KEY,
        )
    except Exception as exc:
        print("Supabase initialization error:", exc)
        supabase = None


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
dataset_lock = threading.RLock()


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
        elif token == "(":
            result += " " + token
        else:
            result += " " + token

    result = result.replace("( ", "(")

    return result.strip()


# ============================================================
# DATASET
# ============================================================

DEFAULT_DATASET = [

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

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
        "prompt": "добрый день",
        "response": "Добрый день! Чем могу помочь?",
        "category": "greeting",
    },
    {
        "prompt": "доброе утро",
        "response": "Доброе утро! Чем могу помочь сегодня?",
        "category": "greeting",
    },
    {
        "prompt": "добрый вечер",
        "response": "Добрый вечер! Что хочешь узнать?",
        "category": "greeting",
    },
    {
        "prompt": "как дела",
        "response": "У меня всё хорошо. Готов общаться и помогать.",
        "category": "greeting",
    },
    {
        "prompt": "как ты",
        "response": "У меня всё нормально. Готов помочь тебе.",
        "category": "greeting",
    },
    {
        "prompt": "ты здесь",
        "response": "Да, я здесь и готов помочь.",
        "category": "greeting",
    },
    {
        "prompt": "приветик",
        "response": "Привет! Чем займёмся?",
        "category": "greeting",
    },
    {
        "prompt": "хай",
        "response": "Привет! Что хочешь узнать?",
        "category": "greeting",
    },

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    {
        "prompt": "что ты умеешь",
        "response": "Я умею отвечать на вопросы, поддерживать разговор, использовать память, находить похожие примеры и учиться на подготовленных данных.",
        "category": "capabilities",
    },
    {
        "prompt": "кто ты",
        "response": "Я AI Care — экспериментальная нейросеть с собственной RNN, памятью, retrieval и системой обучения.",
        "category": "capabilities",
    },
    {
        "prompt": "что такое ai care",
        "response": "AI Care — экспериментальный помощник, который использует собственную нейросеть, память и обучающий датасет.",
        "category": "capabilities",
    },
    {
        "prompt": "ты нейросеть",
        "response": "Да. В этом проекте используется собственная рекуррентная нейронная сеть.",
        "category": "capabilities",
    },
    {
        "prompt": "ты настоящий искусственный интеллект",
        "response": "Я программная система искусственного интеллекта. Моя модель и логика работают внутри этого приложения.",
        "category": "capabilities",
    },
    {
        "prompt": "ты умеешь учиться",
        "response": "Да. Модель может обучаться на подготовленных примерах из датасета.",
        "category": "capabilities",
    },
    {
        "prompt": "ты умеешь запоминать",
        "response": "Да. При настроенной базе данных я могу сохранять отдельные факты и предпочтения пользователя.",
        "category": "capabilities",
    },
    {
        "prompt": "у тебя есть память",
        "response": "Да. В AI Care предусмотрена отдельная система памяти пользователя.",
        "category": "capabilities",
    },

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    {
        "prompt": "что такое нейросеть",
        "response": "Нейросеть — это математическая модель, которая учится находить закономерности в данных.",
        "category": "education",
    },
    {
        "prompt": "что такое rnn",
        "response": "RNN — рекуррентная нейронная сеть, которая обрабатывает последовательности и передаёт состояние между шагами.",
        "category": "education",
    },
    {
        "prompt": "что такое машинное обучение",
        "response": "Машинное обучение — это подход, при котором модель находит закономерности в данных и использует их для получения результата.",
        "category": "education",
    },
    {
        "prompt": "что такое искусственный интеллект",
        "response": "Искусственный интеллект — это область, в которой создают системы, способные выполнять задачи, требующие обработки информации и принятия решений.",
        "category": "education",
    },
    {
        "prompt": "что такое алгоритм",
        "response": "Алгоритм — это последовательность действий, которая приводит к решению определённой задачи.",
        "category": "education",
    },
    {
        "prompt": "что такое база данных",
        "response": "База данных — это организованное хранилище информации, которым программа может управлять и из которого может получать данные.",
        "category": "education",
    },
    {
        "prompt": "что такое api",
        "response": "API — это интерфейс, через который одна программа может взаимодействовать с другой программой или сервисом.",
        "category": "education",
    },
    {
        "prompt": "что такое json",
        "response": "JSON — текстовый формат представления структурированных данных, часто используемый при обмене информацией между программами.",
        "category": "education",
    },
    {
        "prompt": "что такое python",
        "response": "Python — популярный язык программирования, который используют для сайтов, автоматизации, анализа данных и искусственного интеллекта.",
        "category": "education",
    },
    {
        "prompt": "что такое fastapi",
        "response": "FastAPI — Python-фреймворк для создания быстрых веб-API.",
        "category": "education",
    },
    {
        "prompt": "что такое supabase",
        "response": "Supabase — платформа с базой данных PostgreSQL, API, авторизацией и другими инструментами для приложений.",
        "category": "education",
    },
    {
        "prompt": "что такое epoch",
        "response": "Эпоха — один полный проход обучения модели по всем выбранным обучающим примерам.",
        "category": "education",
    },
    {
        "prompt": "что такое loss",
        "response": "Loss — числовая мера ошибки модели. Во время обучения модель обычно старается уменьшить эту величину.",
        "category": "education",
    },
    {
        "prompt": "что такое adam",
        "response": "Adam — популярный алгоритм оптимизации, который адаптирует шаг обновления параметров модели во время обучения.",
        "category": "education",
    },
    {
        "prompt": "что такое gradient clipping",
        "response": "Gradient clipping ограничивает слишком большие градиенты и помогает сделать обучение нейросети стабильнее.",
        "category": "education",
    },

    # --------------------------------------------------------
    # PROGRAMMING
    # --------------------------------------------------------

    {
        "prompt": "помоги с программированием",
        "response": "Конечно. Покажи код или опиши задачу, и мы разберём её по шагам.",
        "category": "programming",
    },
    {
        "prompt": "я не понимаю код",
        "response": "Давай разберём код по частям: сначала посмотрим функции, затем данные и после этого общую логику программы.",
        "category": "programming",
    },
    {
        "prompt": "как изучать программирование",
        "response": "Лучше регулярно писать небольшие программы и постепенно переходить к более сложным проектам.",
        "category": "programming",
    },
    {
        "prompt": "как создать сайт",
        "response": "Для простого сайта обычно используют HTML, CSS и JavaScript. После этого сайт можно разместить на хостинге.",
        "category": "programming",
    },
    {
        "prompt": "как создать бота",
        "response": "Сначала выбери платформу, затем создай обработчики сообщений и подключи базу данных или другие нужные сервисы.",
        "category": "programming",
    },
    {
        "prompt": "как сделать api",
        "response": "Для Python можно использовать FastAPI. Обычно создают приложение, описывают маршруты и возвращают данные в JSON.",
        "category": "programming",
    },
    {
        "prompt": "как исправить ошибку python",
        "response": "Сначала нужно посмотреть полный текст ошибки и строку, на которой она возникла. После этого можно определить причину и исправление.",
        "category": "programming",
    },
    {
        "prompt": "у меня typeerror",
        "response": "TypeError обычно означает, что операция выполняется с неподходящим типом данных. Нужен полный текст ошибки и место её возникновения.",
        "category": "programming",
    },
    {
        "prompt": "у меня ошибка в коде",
        "response": "Покажи полный текст ошибки и участок кода, где она возникает. Тогда причину будет намного проще найти.",
        "category": "programming",
    },
    {
        "prompt": "как запустить python",
        "response": "Обычно Python-файл запускают командой python имя_файла.py или python3 имя_файла.py в терминале.",
        "category": "programming",
    },
    {
        "prompt": "как установить библиотеку python",
        "response": "Большинство библиотек устанавливают через pip, например командой pip install имя_пакета.",
        "category": "programming",
    },
    {
        "prompt": "как сделать telegram бота",
        "response": "Для Telegram-бота можно использовать Python и библиотеку aiogram. Понадобится токен бота и обработчики сообщений.",
        "category": "programming",
    },
    {
        "prompt": "как подключить supabase к python",
        "response": "Обычно устанавливают клиент Supabase, создают клиент с URL и ключом проекта, а затем работают с таблицами через API.",
        "category": "programming",
    },
    {
        "prompt": "как обучить нейросеть",
        "response": "Нужно подготовить данные, преобразовать их в подходящий формат, определить модель, функцию ошибки и алгоритм оптимизации, а затем выполнять итерации обучения.",
        "category": "programming",
    },
    {
        "prompt": "как увеличить датасет",
        "response": "Можно добавить больше разнообразных вопросов и качественных ответов по каждой теме. Особенно полезны разные формулировки одного и того же запроса.",
        "category": "programming",
    },
    {
        "prompt": "как сделать нейросеть лучше",
        "response": "Помогают качественный и разнообразный датасет, подходящая архитектура, стабильный оптимизатор, корректная проверка качества и аккуратная настройка обучения.",
        "category": "programming",
    },

    # --------------------------------------------------------
    # SKINCARE
    # --------------------------------------------------------

    {
        "prompt": "как ухаживать за кожей",
        "response": "Обычно полезно мягко очищать кожу и использовать подходящий увлажняющий крем. Если проблема сильная или не проходит, лучше обратиться к врачу.",
        "category": "skincare",
    },
    {
        "prompt": "у меня жирная кожа",
        "response": "При жирной коже лучше использовать мягкое очищение и не пытаться постоянно пересушивать кожу. Если жирность сопровождается выраженным воспалением, стоит обратиться к дерматологу.",
        "category": "skincare",
    },
    {
        "prompt": "что делать если жирная кожа",
        "response": "Используй мягкое очищение и лёгкий подходящий увлажняющий крем. Агрессивное пересушивание может раздражать кожу.",
        "category": "skincare",
    },
    {
        "prompt": "у меня сухая кожа",
        "response": "При сухости обычно помогают мягкое очищение и регулярное увлажнение. Если кожа сильно раздражена или трескается, лучше обратиться к специалисту.",
        "category": "skincare",
    },
    {
        "prompt": "у меня комбинированная кожа",
        "response": "Для комбинированной кожи обычно подходит мягкое очищение и увлажнение без чрезмерного пересушивания отдельных участков.",
        "category": "skincare",
    },
    {
        "prompt": "у меня чёрные точки",
        "response": "Чёрные точки часто связаны с закупоркой пор. Не стоит агрессивно выдавливать их или травмировать кожу.",
        "category": "skincare",
    },
    {
        "prompt": "как убрать чёрные точки",
        "response": "Лучше использовать мягкий уход и не выдавливать точки руками. При стойкой проблеме можно обсудить уход с дерматологом.",
        "category": "skincare",
    },
    {
        "prompt": "как убрать черные точки",
        "response": "Не стоит сильно выдавливать их. Мягкое очищение и подходящий уход обычно безопаснее для кожи.",
        "category": "skincare",
    },
    {
        "prompt": "как убрать жирность кожи",
        "response": "Лучше использовать мягкое очищение и не пересушивать кожу слишком агрессивными средствами.",
        "category": "skincare",
    },
    {
        "prompt": "почему кожа жирная",
        "response": "Жирность кожи зависит от работы сальных желёз, генетики, возраста и других факторов. Она не означает, что кожу нужно постоянно пересушивать.",
        "category": "skincare",
    },
    {
        "prompt": "нужно ли увлажнять жирную кожу",
        "response": "Да, жирная кожа тоже нуждается в увлажнении. Обычно выбирают лёгкое средство, которое комфортно ощущается на коже.",
        "category": "skincare",
    },
    {
        "prompt": "как часто умываться",
        "response": "Обычно достаточно мягкого очищения утром и вечером. Слишком частое умывание может раздражать кожу.",
        "category": "skincare",
    },
    {
        "prompt": "можно ли выдавливать прыщи",
        "response": "Лучше не выдавливать воспаления самостоятельно, потому что это может усилить раздражение и повысить риск следов.",
        "category": "skincare",
    },
    {
        "prompt": "что делать с прыщами",
        "response": "Начни с мягкого ухода и не травмируй кожу. Если высыпания выраженные или долго не проходят, стоит обратиться к дерматологу.",
        "category": "skincare",
    },
    {
        "prompt": "у меня прыщи",
        "response": "Акне довольно распространено. Лучше не выдавливать воспаления и использовать мягкий уход, а при выраженных высыпаниях обратиться к дерматологу.",
        "category": "skincare",
    },
    {
        "prompt": "как улучшить кожу лица",
        "response": "Полезно придерживаться простого ухода: мягкое очищение, увлажнение и защита кожи от солнца. Не стоит постоянно менять много средств одновременно.",
        "category": "skincare",
    },
    {
        "prompt": "нужен ли крем для лица",
        "response": "Увлажняющий крем может поддерживать защитный барьер кожи. Выбирай средство, которое подходит твоему типу кожи и не вызывает раздражения.",
        "category": "skincare",
    },
    {
        "prompt": "нужен ли солнцезащитный крем",
        "response": "Солнцезащитное средство помогает уменьшать воздействие ультрафиолета на кожу. Его особенно полезно использовать при длительном нахождении на солнце.",
        "category": "skincare",
    },
    {
        "prompt": "как выбрать уход за кожей",
        "response": "Лучше начинать с базового ухода и добавлять новые средства постепенно, чтобы понимать реакцию кожи.",
        "category": "skincare",
    },
    {
        "prompt": "можно ли использовать много средств",
        "response": "Большое количество средств не обязательно улучшает кожу. Проще начать с нескольких базовых продуктов и постепенно оценивать результат.",
        "category": "skincare",
    },

    # --------------------------------------------------------
    # WELLBEING
    # --------------------------------------------------------

    {
        "prompt": "я устал",
        "response": "Если устал, полезно сделать небольшой перерыв, попить воды и дать себе время восстановиться.",
        "category": "wellbeing",
    },
    {
        "prompt": "мне плохо",
        "response": "Мне жаль, что тебе сейчас плохо. Постарайся немного отдохнуть и поговорить с человеком, которому доверяешь.",
        "category": "wellbeing",
    },
    {
        "prompt": "я нервничаю",
        "response": "Попробуй ненадолго остановиться, спокойно подышать и сосредоточиться на ближайшем небольшом шаге.",
        "category": "wellbeing",
    },
    {
        "prompt": "я переживаю",
        "response": "Переживать перед важными событиями нормально. Попробуй сосредоточиться на том, что можешь сделать прямо сейчас.",
        "category": "wellbeing",
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
        "prompt": "как отдохнуть",
        "response": "Можно сделать небольшой перерыв, выйти на прогулку, почитать или просто спокойно провести немного времени без нагрузки.",
        "category": "wellbeing",
    },
    {
        "prompt": "я не высыпаюсь",
        "response": "Попробуй придерживаться более стабильного режима сна и по возможности уменьшить поздние занятия перед сном.",
        "category": "wellbeing",
    },

    # --------------------------------------------------------
    # CONVERSATION
    # --------------------------------------------------------

    {
        "prompt": "мне скучно",
        "response": "Можно придумать небольшую игру, изучить что-нибудь новое или заняться своим проектом.",
        "category": "conversation",
    },
    {
        "prompt": "что поделать",
        "response": "Можно поработать над проектом, изучить новую тему, посмотреть интересный фильм или придумать небольшую игру.",
        "category": "conversation",
    },
    {
        "prompt": "давай поговорим",
        "response": "Конечно. Я готов поговорить на любую подходящую тему.",
        "category": "conversation",
    },
    {
        "prompt": "мне нечем заняться",
        "response": "Можно заняться своим проектом или попробовать изучить что-нибудь новое.",
        "category": "conversation",
    },
    {
        "prompt": "спасибо",
        "response": "Пожалуйста! Рад помочь.",
        "category": "conversation",
    },
    {
        "prompt": "большое спасибо",
        "response": "Пожалуйста! Обращайся.",
        "category": "conversation",
    },
    {
        "prompt": "понятно",
        "response": "Отлично.",
        "category": "conversation",
    },
    {
        "prompt": "хорошо",
        "response": "Хорошо.",
        "category": "conversation",
    },
    {
        "prompt": "ладно",
        "response": "Договорились.",
        "category": "conversation",
    },

    # --------------------------------------------------------
    # FUN
    # --------------------------------------------------------

    {
        "prompt": "расскажи шутку",
        "response": "Почему программист любит тёмную тему? Потому что свет притягивает баги.",
        "category": "fun",
    },
    {
        "prompt": "пошути",
        "response": "Программист сказал, что исправит одну маленькую ошибку. Через три часа это уже был новый проект.",
        "category": "fun",
    },
    {
        "prompt": "расскажи анекдот",
        "response": "У программиста всё просто: если работает — не трогай, если не работает — ищи баг.",
        "category": "fun",
    },

    # --------------------------------------------------------
    # FACTS
    # --------------------------------------------------------

    {
        "prompt": "расскажи факт",
        "response": "У осьминога три сердца.",
        "category": "facts",
    },
    {
        "prompt": "интересный факт",
        "response": "У осьминога три сердца, а его кровь содержит медьсодержащий белок.",
        "category": "facts",
    },
    {
        "prompt": "скажи факт",
        "response": "Молния может нагревать окружающий воздух до температуры выше, чем поверхность Солнца.",
        "category": "facts",
    },

    # --------------------------------------------------------
    # LEARNING
    # --------------------------------------------------------

    {
        "prompt": "что можно изучить",
        "response": "Можно изучить Python, веб-разработку, базы данных, алгоритмы или основы машинного обучения.",
        "category": "education",
    },
    {
        "prompt": "что изучать после python",
        "response": "После базового Python можно перейти к структурам данных, API, базам данных, веб-разработке или машинному обучению.",
        "category": "education",
    },
    {
        "prompt": "как научиться программировать",
        "response": "Лучше сочетать теорию с практикой: изучать небольшую тему и сразу применять её в маленьком проекте.",
        "category": "education",
    },
    {
        "prompt": "как учить python",
        "response": "Начни с синтаксиса, переменных, условий, циклов и функций, а затем закрепляй всё небольшими практическими программами.",
        "category": "education",
    },

    # --------------------------------------------------------
    # AI CARE
    # --------------------------------------------------------

    {
        "prompt": "как обучается ai care",
        "response": "AI Care обучается на примерах из датасета. Каждый пример содержит вопрос и правильный ответ, а модель постепенно изменяет свои параметры, уменьшая ошибку.",
        "category": "ai_care",
    },
    {
        "prompt": "что такое датасет ai care",
        "response": "Датасет AI Care — это набор обучающих примеров, в которых указаны запрос пользователя и желаемый ответ.",
        "category": "ai_care",
    },
    {
        "prompt": "зачем нужны эпохи",
        "response": "Эпохи определяют, сколько раз модель проходит по всему обучающему датасету.",
        "category": "ai_care",
    },
    {
        "prompt": "что будет если поставить много эпох",
        "response": "Большее число эпох даёт модели больше шагов для обучения, но слишком длительное обучение на маленьком датасете может привести к запоминанию примеров вместо хорошего обобщения.",
        "category": "ai_care",
    },
    {
        "prompt": "зачем нужен retrieval",
        "response": "Retrieval помогает находить похожий вопрос в датасете и использовать уже подготовленный качественный ответ.",
        "category": "ai_care",
    },
    {
        "prompt": "зачем нужна память",
        "response": "Память позволяет сохранять отдельные сведения о пользователе и использовать их в последующих разговорах.",
        "category": "ai_care",
    },
    {
        "prompt": "зачем нужна проверка ответов",
        "response": "Проверка помогает увидеть, насколько хорошо модель отвечает на контрольные вопросы и не ухудшилось ли качество после обучения.",
        "category": "ai_care",
    },
    {
        "prompt": "что такое checkpoint",
        "response": "Checkpoint — сохранённое состояние модели, которое позволяет восстановить её после перезапуска приложения.",
        "category": "ai_care",
    },
    {
        "prompt": "можно ли обучать ai care постепенно",
        "response": "Да. Можно добавлять новые качественные примеры и запускать дополнительные эпохи обучения.",
        "category": "ai_care",
    },
    {
        "prompt": "можно ли добавить новый вопрос",
        "response": "Да. Администратор может добавить новый вопрос и правильный ответ в датасет.",
        "category": "ai_care",
    },

]


dataset = []


def normalize_example(item):
    return {
        "prompt": str(item.get("prompt", "")).strip(),
        "response": str(item.get("response", "")).strip(),
        "category": (
            str(item.get("category", "general")).strip()
            or "general"
        ),
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

    with dataset_lock:
        if supabase is not None:
            try:
                response = (
                    supabase
                    .table("ai_dataset")
                    .select(
                        "id,prompt,response,category"
                    )
                    .order("id")
                    .execute()
                )

                rows = response.data or []

                if rows:
                    dataset = deduplicate_dataset(rows)
                    save_local_dataset()
                    return

            except Exception as exc:
                print(
                    "Supabase dataset load error:",
                    exc,
                )

        if LOCAL_DATASET.exists():
            try:
                loaded = json.loads(
                    LOCAL_DATASET.read_text(
                        encoding="utf-8"
                    )
                )

                dataset = deduplicate_dataset(
                    loaded
                )

                if dataset:
                    return

            except Exception as exc:
                print(
                    "Local dataset load error:",
                    exc,
                )

        dataset = deduplicate_dataset(
            DEFAULT_DATASET
        )

        save_local_dataset()


def save_local_dataset():
    try:
        with dataset_lock:
            LOCAL_DATASET.write_text(
                json.dumps(
                    dataset,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    except Exception as exc:
        print(
            "Local dataset save error:",
            exc,
        )


def save_dataset_to_supabase():
    if supabase is None:
        return False

    try:
        with db_lock:
            rows = []

            for item in dataset:
                rows.append(
                    {
                        "prompt": item["prompt"],
                        "response": item["response"],
                        "category": item.get(
                            "category",
                            "general",
                        ),
                    }
                )

            if rows:
                supabase.table(
                    "ai_dataset"
                ).upsert(
                    rows,
                    on_conflict="prompt,response",
                ).execute()

        return True

    except Exception as exc:
        print(
            "Supabase dataset save error:",
            exc,
        )

        return False


# ============================================================
# VOCABULARY
# ============================================================

vocab = {}
id_to_token = []


def build_vocab(items):
    global vocab
    global id_to_token

    tokens = []

    for special in SPECIAL_TOKENS:
        if special not in tokens:
            tokens.append(special)

    for item in items:
        tokens.extend(
            tokenize(item["prompt"])
        )

        tokens.extend(
            tokenize(item["response"])
        )

    unique = []

    for token in tokens:
        if token not in unique:
            unique.append(token)

    id_to_token = unique

    vocab = {
        token: index
        for index, token in enumerate(
            id_to_token
        )
    }


def token_id(token):
    return vocab.get(
        token,
        vocab.get("<UNK>", 1),
    )


# ============================================================
# MODEL
# ============================================================

class RNNModel:

    def __init__(
        self,
        hidden_size,
        vocab_size,
    ):
        self.hidden_size = int(
            hidden_size
        )

        self.vocab_size = int(
            vocab_size
        )

        scale = 0.05

        self.Wxh = (
            np.random.randn(
                self.hidden_size,
                self.vocab_size,
            )
            * scale
        )

        self.Whh = (
            np.random.randn(
                self.hidden_size,
                self.hidden_size,
            )
            * scale
        )

        self.Why = (
            np.random.randn(
                self.vocab_size,
                self.hidden_size,
            )
            * scale
        )

        self.bh = np.zeros(
            self.hidden_size
        )

        self.by = np.zeros(
            self.vocab_size
        )

        # ----------------------------------------------------
        # ADAM
        # ----------------------------------------------------

        self.adam_step = 0

        self.m_Wxh = np.zeros_like(
            self.Wxh
        )
        self.v_Wxh = np.zeros_like(
            self.Wxh
        )

        self.m_Whh = np.zeros_like(
            self.Whh
        )
        self.v_Whh = np.zeros_like(
            self.Whh
        )

        self.m_Why = np.zeros_like(
            self.Why
        )
        self.v_Why = np.zeros_like(
            self.Why
        )

        self.m_bh = np.zeros_like(
            self.bh
        )
        self.v_bh = np.zeros_like(
            self.bh
        )

        self.m_by = np.zeros_like(
            self.by
        )
        self.v_by = np.zeros_like(
            self.by
        )

    def copy(self):

        new_model = RNNModel(
            self.hidden_size,
            self.vocab_size,
        )

        for name in (
            "Wxh",
            "Whh",
            "Why",
            "bh",
            "by",
            "m_Wxh",
            "v_Wxh",
            "m_Whh",
            "v_Whh",
            "m_Why",
            "v_Why",
            "m_bh",
            "v_bh",
            "m_by",
            "v_by",
        ):
            setattr(
                new_model,
                name,
                getattr(self, name).copy(),
            )

        new_model.adam_step = (
            self.adam_step
        )

        return new_model

    def adam_update(
        self,
        parameter,
        gradient,
        m,
        v,
        learning_rate,
    ):

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        m *= beta1
        m += (1.0 - beta1) * gradient

        v *= beta2
        v += (1.0 - beta2) * (
            gradient * gradient
        )

        step = max(
            1,
            self.adam_step,
        )

        m_hat = (
            m / (
                1.0
                - beta1 ** step
            )
        )

        v_hat = (
            v / (
                1.0
                - beta2 ** step
            )
        )

        parameter -= (
            learning_rate
            * m_hat
            / (
                np.sqrt(v_hat)
                + epsilon
            )
        )

    def forward(
        self,
        inputs,
        targets=None,
        h0=None,
    ):

        if h0 is None:
            h = np.zeros(
                self.hidden_size
            )
        else:
            h = h0.copy()

        hs = [h]
        ps = []
        loss = 0.0

        for i, input_id in enumerate(
            inputs
        ):

            if (
                input_id < 0
                or input_id >= self.vocab_size
            ):
                input_id = token_id(
                    "<UNK>"
                )

            x = np.zeros(
                self.vocab_size
            )

            x[input_id] = 1.0

            h = np.tanh(
                self.Wxh @ x
                + self.Whh @ h
                + self.bh
            )

            logits = (
                self.Why @ h
                + self.by
            )

            logits -= np.max(
                logits
            )

            exp_logits = np.exp(
                np.clip(
                    logits,
                    -50,
                    50,
                )
            )

            probs = exp_logits / (
                np.sum(exp_logits)
                + 1e-12
            )

            ps.append(probs)
            hs.append(h)

            if targets is not None:

                target = int(
                    targets[i]
                )

                if (
                    target < 0
                    or target >= self.vocab_size
                ):
                    target = token_id(
                        "<UNK>"
                    )

                loss -= np.log(
                    probs[target]
                    + 1e-12
                )

        return (
            float(loss),
            hs,
            ps,
            h,
        )

    def train_example(
        self,
        inputs,
        targets,
        learning_rate,
    ):

        if not inputs:
            return 0.0

        if len(inputs) != len(
            targets
        ):
            raise ValueError(
                "Длина inputs и targets должна совпадать."
            )

        loss, hs, ps, _ = (
            self.forward(
                inputs,
                targets,
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

        dh_next = np.zeros(
            self.hidden_size
        )

        for t in reversed(
            range(len(inputs))
        ):

            target = int(
                targets[t]
            )

            if (
                target < 0
                or target >= self.vocab_size
            ):
                target = token_id(
                    "<UNK>"
                )

            dy = ps[t].copy()
            dy[target] -= 1.0

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
                1.0
                - hs[t + 1] ** 2
            ) * dh

            dbh += dh_raw

            dWhh += np.outer(
                dh_raw,
                hs[t],
            )

            input_id = int(
                inputs[t]
            )

            if (
                input_id < 0
                or input_id >= self.vocab_size
            ):
                input_id = token_id(
                    "<UNK>"
                )

            x = np.zeros(
                self.vocab_size
            )

            x[input_id] = 1.0

            dWxh += np.outer(
                dh_raw,
                x,
            )

            dh_next = (
                self.Whh.T
                @ dh_raw
            )

        gradients = [
            dWxh,
            dWhh,
            dWhy,
            dbh,
            dby,
        ]

        for grad in gradients:
            np.nan_to_num(
                grad,
                copy=False,
                nan=0.0,
                posinf=GRADIENT_CLIP,
                neginf=-GRADIENT_CLIP,
            )

            np.clip(
                grad,
                -GRADIENT_CLIP,
                GRADIENT_CLIP,
                out=grad,
            )

        self.adam_step += 1

        self.adam_update(
            self.Wxh,
            dWxh,
            self.m_Wxh,
            self.v_Wxh,
            learning_rate,
        )

        self.adam_update(
            self.Whh,
            dWhh,
            self.m_Whh,
            self.v_Whh,
            learning_rate,
        )

        self.adam_update(
            self.Why,
            dWhy,
            self.m_Why,
            self.v_Why,
            learning_rate,
        )

        self.adam_update(
            self.bh,
            dbh,
            self.m_bh,
            self.v_bh,
            learning_rate,
        )

        self.adam_update(
            self.by,
            dby,
            self.m_by,
            self.v_by,
            learning_rate,
        )

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

    if (
        old_model.hidden_size
        != HIDDEN_SIZE
    ):
        return RNNModel(
            HIDDEN_SIZE,
            new_size,
        )

    if (
        old_model.vocab_size
        == new_size
    ):
        return old_model

    new_model = RNNModel(
        HIDDEN_SIZE,
        new_size,
    )

    for token, old_id in (
        old_vocab.items()
    ):

        if token not in vocab:
            continue

        new_id = vocab[token]

        if (
            old_id
            < old_model.Wxh.shape[1]
            and new_id
            < new_model.Wxh.shape[1]
        ):
            new_model.Wxh[
                :,
                new_id
            ] = old_model.Wxh[
                :,
                old_id
            ]

            new_model.m_Wxh[
                :,
                new_id
            ] = old_model.m_Wxh[
                :,
                old_id
            ]

            new_model.v_Wxh[
                :,
                new_id
            ] = old_model.v_Wxh[
                :,
                old_id
            ]

        if (
            old_id
            < old_model.Why.shape[0]
            and new_id
            < new_model.Why.shape[0]
        ):
            new_model.Why[
                new_id,
                :
            ] = old_model.Why[
                old_id,
                :
            ]

            new_model.m_Why[
                new_id,
                :
            ] = old_model.m_Why[
                old_id,
                :
            ]

            new_model.v_Why[
                new_id,
                :
            ] = old_model.v_Why[
                old_id,
                :
            ]

        if (
            old_id
            < old_model.by.shape[0]
            and new_id
            < new_model.by.shape[0]
        ):
            new_model.by[
                new_id
            ] = old_model.by[
                old_id
            ]

            new_model.m_by[
                new_id
            ] = old_model.m_by[
                old_id
            ]

            new_model.v_by[
                new_id
            ] = old_model.v_by[
                old_id
            ]

    new_model.Whh = (
        old_model.Whh.copy()
    )

    new_model.bh = (
        old_model.bh.copy()
    )

    new_model.m_Whh = (
        old_model.m_Whh.copy()
    )

    new_model.v_Whh = (
        old_model.v_Whh.copy()
    )

    new_model.m_bh = (
        old_model.m_bh.copy()
    )

    new_model.v_bh = (
        old_model.v_bh.copy()
    )

    new_model.adam_step = (
        old_model.adam_step
    )

    return new_model


# ============================================================
# SERIALIZATION
# ============================================================

def serialize_model(
    current_model,
):

    vocab_json = json.dumps(
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

            Wxh=current_model.Wxh,
            Whh=current_model.Whh,
            Why=current_model.Why,

            bh=current_model.bh,
            by=current_model.by,

            m_Wxh=current_model.m_Wxh,
            v_Wxh=current_model.v_Wxh,

            m_Whh=current_model.m_Whh,
            v_Whh=current_model.v_Whh,

            m_Why=current_model.m_Why,
            v_Why=current_model.v_Why,

            m_bh=current_model.m_bh,
            v_bh=current_model.v_bh,

            m_by=current_model.m_by,
            v_by=current_model.v_by,

            adam_step=np.array(
                [
                    current_model.adam_step
                ],
                dtype=np.int64,
            ),

            hidden_size=np.array(
                [
                    current_model.hidden_size
                ],
                dtype=np.int64,
            ),

            vocab=np.array(
                [vocab_json],
                dtype=object,
            ),
        )

        return Path(path).read_bytes()

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

        model_loaded = RNNModel(
            saved_hidden,
            len(saved_vocab),
        )

        model_loaded.Wxh = (
            loaded["Wxh"]
        )

        model_loaded.Whh = (
            loaded["Whh"]
        )

        model_loaded.Why = (
            loaded["Why"]
        )

        model_loaded.bh = (
            loaded["bh"]
        )

        model_loaded.by = (
            loaded["by"]
        )

        if "m_Wxh" in loaded:
            model_loaded.m_Wxh = (
                loaded["m_Wxh"]
            )

        if "v_Wxh" in loaded:
            model_loaded.v_Wxh = (
                loaded["v_Wxh"]
            )

        if "m_Whh" in loaded:
            model_loaded.m_Whh = (
                loaded["m_Whh"]
            )

        if "v_Whh" in loaded:
            model_loaded.v_Whh = (
                loaded["v_Whh"]
            )

        if "m_Why" in loaded:
            model_loaded.m_Why = (
                loaded["m_Why"]
            )

        if "v_Why" in loaded:
            model_loaded.v_Why = (
                loaded["v_Why"]
            )

        if "m_bh" in loaded:
            model_loaded.m_bh = (
                loaded["m_bh"]
            )

        if "v_bh" in loaded:
            model_loaded.v_bh = (
                loaded["v_bh"]
            )

        if "m_by" in loaded:
            model_loaded.m_by = (
                loaded["m_by"]
            )

        if "v_by" in loaded:
            model_loaded.v_by = (
                loaded["v_by"]
            )

        if "adam_step" in loaded:
            model_loaded.adam_step = int(
                loaded["adam_step"][0]
            )

        return (
            model_loaded,
            saved_vocab,
        )

    finally:

        try:
            Path(path).unlink()
        except Exception:
            pass


# ============================================================
# MODEL STATE
# ============================================================

model = None

trained_epochs = 0
last_loss = None

training = {
    "running": False,
    "epoch": 0,
    "target_epoch": 0,
    "loss": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "current_learning_rate": LEARNING_RATE,
    "examples_processed": 0,
    "stop_requested": False,
}


stop_event = threading.Event()


# ============================================================
# LOCAL MODEL
# ============================================================

def load_local_model():

    if not LOCAL_MODEL.exists():
        return None

    try:

        data = LOCAL_MODEL.read_bytes()

        return deserialize_model(
            data
        )

    except Exception as exc:

        print(
            "Local model load error:",
            exc,
        )

        return None


def save_local_model(
    current_model,
):

    try:

        data = serialize_model(
            current_model
        )

        temp_path = (
            LOCAL_MODEL.with_suffix(
                ".tmp.npz"
            )
        )

        temp_path.write_bytes(
            data
        )

        temp_path.replace(
            LOCAL_MODEL
        )

        return True

    except Exception as exc:

        print(
            "Local model save error:",
            exc,
        )

        return False


# ============================================================
# SUPABASE MODEL STATE
# ============================================================

def save_model_to_supabase(
    current_model,
    trained_epoch_count,
    loss,
):

    # Локальный checkpoint сохраняется
    # ВСЕГДА, даже если Supabase недоступен.

    local_ok = save_local_model(
        current_model
    )

    if supabase is None:
        return local_ok

    try:

        data = serialize_model(
            current_model
        )

        encoded = base64.b64encode(
            data
        ).decode("ascii")

        payload = {
            "id": 1,
            "trained_epochs": int(
                trained_epoch_count
            ),
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

        return True

    except Exception as exc:

        print(
            "Supabase model save error:",
            exc,
        )

        return local_ok


def load_model_from_supabase():

    if supabase is None:
        return (
            None,
            {},
            0,
            None,
        )

    try:

        response = (
            supabase
            .table("ai_model_state")
            .select(
                "trained_epochs,"
                "loss,"
                "hidden_size,"
                "vocab,"
                "model_blob"
            )
            .eq("id", 1)
            .limit(1)
            .execute()
        )

        rows = response.data or []

        if not rows:
            return (
                None,
                {},
                0,
                None,
            )

        row = rows[0]

        blob = row.get(
            "model_blob"
        )

        if not blob:
            return (
                None,
                {},
                0,
                None,
            )

        data = base64.b64decode(
            blob
        )

        loaded_model, saved_vocab = (
            deserialize_model(data)
        )

        return (
            loaded_model,
            saved_vocab,
            int(
                row.get(
                    "trained_epochs"
                )
                or 0
            ),
            row.get("loss"),
        )

    except Exception as exc:

        print(
            "Supabase model load error:",
            exc,
        )

        return (
            None,
            {},
            0,
            None,
        )


# ============================================================
# INITIALIZE MODEL
# ============================================================

def initialize_model():

    global model
    global vocab
    global id_to_token
    global trained_epochs
    global last_loss

    build_vocab(
        dataset
    )

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

        old_vocab = (
            saved_vocab.copy()
        )

        loaded_model = (
            expand_model_vocabulary(
                loaded_model,
                old_vocab,
            )
        )

        model = loaded_model

    else:

        model = RNNModel(
            HIDDEN_SIZE,
            len(vocab),
        )

    trained_epochs = (
        saved_epochs
    )

    last_loss = saved_loss

    save_local_model(
        model
    )


# ============================================================
# MEMORY
# ============================================================

def get_memories(
    user_id,
):

    if supabase is None:
        return []

    try:

        response = (
            supabase
            .table("ai_memories")
            .select(
                "memory_key,"
                "memory_value"
            )
            .eq(
                "user_id",
                user_id,
            )
            .order("id")
            .execute()
        )

        return response.data or []

    except Exception as exc:

        print(
            "Memory load error:",
            exc,
        )

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
        on_conflict=(
            "user_id,memory_key"
        ),
    ).execute()


def extract_memory(
    user_id,
    text,
):

    if supabase is None:
        return

    lowered = (
        text.lower().strip()
    )

    match = re.search(
        r"(?:меня зовут|моё имя|мое имя)\s+"
        r"([а-яёa-z0-9_-]{2,30})",
        lowered,
    )

    if match:

        try:
            save_memory(
                user_id,
                "name",
                match.group(1),
            )
        except Exception as exc:
            print(
                "Name memory error:",
                exc,
            )

    match = re.search(
        r"(?:я люблю|мне нравится|"
        r"моя любимая игра|"
        r"мой любимый фильм)\s+"
        r"(.{2,150})",
        lowered,
    )

    if match:

        try:
            save_memory(
                user_id,
                "preference",
                match.group(1).strip(),
            )
        except Exception as exc:
            print(
                "Preference memory error:",
                exc,
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

            try:
                save_memory(
                    user_id,
                    "user_note",
                    value[:500],
                )
            except Exception as exc:
                print(
                    "User note memory error:",
                    exc,
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
                "text": text[
                    :MAX_MESSAGE_LENGTH
                ],
            }
        ).execute()

    except Exception as exc:

        print(
            "Message save error:",
            exc,
        )


def get_history(
    user_id,
):

    if supabase is None:
        return []

    try:

        response = (
            supabase
            .table("ai_messages")
            .select(
                "role,text,created_at"
            )
            .eq(
                "user_id",
                user_id,
            )
            .order(
                "id",
                desc=False,
            )
            .limit(
                MAX_CONTEXT_MESSAGES * 2
            )
            .execute()
        )

        return response.data or []

    except Exception as exc:

        print(
            "History load error:",
            exc,
        )

        return []


# ============================================================
# RETRIEVAL
# ============================================================

def word_set(
    text,
):

    return set(
        token
        for token in tokenize(text)
        if token
        not in SPECIAL_TOKENS
    )


def similarity(
    a,
    b,
):

    a_set = word_set(a)
    b_set = word_set(b)

    if not a_set or not b_set:
        return 0.0

    intersection = len(
        a_set.intersection(
            b_set
        )
    )

    union = len(
        a_set.union(
            b_set
        )
    )

    if union == 0:
        return 0.0

    jaccard = (
        intersection
        / union
    )

    contains_bonus = 0.0

    a_lower = a.lower()
    b_lower = b.lower()

    if (
        a_lower in b_lower
        or b_lower in a_lower
    ):
        contains_bonus = 0.25

    # Бонус за совпадающие последовательности
    a_tokens = tokenize(a)
    b_tokens = tokenize(b)

    sequence_bonus = 0.0

    if (
        len(a_tokens) >= 2
        and len(b_tokens) >= 2
    ):

        for i in range(
            len(a_tokens) - 1
        ):

            pair = (
                a_tokens[i],
                a_tokens[i + 1],
            )

            for j in range(
                len(b_tokens) - 1
            ):

                if pair == (
                    b_tokens[j],
                    b_tokens[j + 1],
                ):
                    sequence_bonus = max(
                        sequence_bonus,
                        0.10,
                    )

    return min(
        1.0,
        jaccard
        + contains_bonus
        + sequence_bonus,
    )


def retrieve_response(
    text,
):

    best_item = None
    best_score = 0.0

    with dataset_lock:

        current_dataset = list(
            dataset
        )

    for item in current_dataset:

        score = similarity(
            text,
            item["prompt"],
        )

        if score > best_score:

            best_score = score
            best_item = item

    return (
        best_item,
        best_score,
    )


# ============================================================
# TRAINING DATA
# ============================================================

def make_training_sequence(
    item,
):

    prompt_tokens = tokenize(
        item["prompt"]
    )

    response_tokens = tokenize(
        item["response"]
    )

    if not response_tokens:
        return [], []

    # --------------------------------------------------------
    # Последовательность:
    #
    # prompt -> BOS -> response -> EOS
    #
    # Модель учится:
    # 1. учитывать prompt
    # 2. после BOS начинать response
    # 3. продолжать response
    # 4. завершать EOS
    # --------------------------------------------------------

    input_tokens = (
        prompt_tokens
        + ["<BOS>"]
        + response_tokens
    )

    target_tokens = []

    if prompt_tokens:

        target_tokens.extend(
            prompt_tokens[1:]
        )

        target_tokens.append(
            "<BOS>"
        )

    else:

        target_tokens.append(
            "<BOS>"
        )

    target_tokens.extend(
        response_tokens[1:]
    )

    target_tokens.append(
        "<EOS>"
    )

    # Выравниваем длины.

    if len(target_tokens) < len(
        input_tokens
    ):

        target_tokens.extend(
            ["<EOS>"]
            * (
                len(input_tokens)
                - len(target_tokens)
            )
        )

    elif len(target_tokens) > len(
        input_tokens
    ):

        input_tokens.extend(
            ["<BOS>"]
            * (
                len(target_tokens)
                - len(input_tokens)
            )
        )

    inputs = [
        token_id(token)
        for token in input_tokens
    ]

    targets = [
        token_id(token)
        for token in target_tokens
    ]

    return (
        inputs,
        targets,
    )


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(
    epoch,
):

    lr = (
        LEARNING_RATE
        * (
            LEARNING_RATE_DECAY
            ** max(
                0,
                epoch,
            )
        )
    )

    return max(
        MIN_LEARNING_RATE,
        lr,
    )


# ============================================================
# TRAINING
# ============================================================

def train_worker(
    additional_epochs,
):

    global model
    global trained_epochs
    global last_loss

    training["running"] = True
    training["error"] = None
    training["started_at"] = time.time()
    training["finished_at"] = None
    training["stop_requested"] = False
    training["epoch"] = trained_epochs
    training["loss"] = last_loss
    training["examples_processed"] = 0

    training["target_epoch"] = min(
        MAX_TRAIN_EPOCHS,
        trained_epochs
        + additional_epochs,
    )

    try:

        for _ in range(
            additional_epochs
        ):

            if stop_event.is_set():
                break

            with model_lock:

                if model is None:
                    raise RuntimeError(
                        "Модель не инициализирована."
                    )

                current_model = model

            total_loss = 0.0
            count = 0

            with dataset_lock:
                shuffled = list(
                    dataset
                )

            np.random.shuffle(
                shuffled
            )

            current_epoch = (
                trained_epochs + 1
            )

            learning_rate = (
                get_learning_rate(
                    current_epoch
                )
            )

            training[
                "current_learning_rate"
            ] = learning_rate

            for item in shuffled:

                if stop_event.is_set():
                    break

                try:

                    inputs, targets = (
                        make_training_sequence(
                            item
                        )
                    )

                    if not inputs:
                        continue

                    if len(inputs) != len(
                        targets
                    ):
                        continue

                    with model_lock:

                        loss = (
                            current_model.train_example(
                                inputs,
                                targets,
                                learning_rate,
                            )
                        )

                    if not np.isfinite(
                        loss
                    ):
                        continue

                    total_loss += float(
                        loss
                    )

                    count += 1

                    training[
                        "examples_processed"
                    ] += 1

                except (
                    TypeError,
                    ValueError,
                    FloatingPointError,
                    OverflowError,
                ) as exc:

                    print(
                        "Training example skipped:",
                        type(exc).__name__,
                        str(exc),
                    )

                    continue

            if stop_event.is_set():
                break

            epoch_loss = (
                total_loss
                / max(
                    count,
                    1,
                )
            )

            trained_epochs += 1
            last_loss = float(
                epoch_loss
            )

            with model_lock:

                # Публикуем обновлённую модель.
                model = current_model

                # Сохраняем checkpoint
                # после КАЖДОЙ эпохи.
                save_model_to_supabase(
                    current_model,
                    trained_epochs,
                    epoch_loss,
                )

            training["epoch"] = (
                trained_epochs
            )

            training["loss"] = (
                epoch_loss
            )

            print(
                f"Epoch {trained_epochs}: "
                f"loss={epoch_loss:.6f} "
                f"lr={learning_rate:.8f}"
            )

        if stop_event.is_set():

            training[
                "stop_requested"
            ] = True

    except Exception as exc:

        training["error"] = (
            f"{type(exc).__name__}: "
            f"{str(exc)}"
        )

        print(
            "Training error:",
            type(exc).__name__,
            str(exc),
        )

    finally:

        training["running"] = False
        training["finished_at"] = (
            time.time()
        )

        stop_event.clear()


def start_training(
    epochs,
):

    if training["running"]:
        raise HTTPException(
            status_code=409,
            detail="Обучение уже запущено.",
        )

    try:
        epochs = int(
            epochs
        )
    except (
        TypeError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="epochs должен быть числом.",
        )

    if (
        epochs < 1
        or epochs > MAX_TRAIN_EPOCHS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Количество эпох должно "
                "быть от 1 до 100000."
            ),
        )

    if not dataset:

        raise HTTPException(
            status_code=400,
            detail="Датасет пуст.",
        )

    if (
        trained_epochs + epochs
        > MAX_TRAIN_EPOCHS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Общее количество обучающих "
                "эпох не может превышать 100000."
            ),
        )

    stop_event.clear()

    thread = threading.Thread(
        target=train_worker,
        args=(epochs,),
        daemon=True,
    )

    thread.start()


def stop_training():

    if not training["running"]:
        return False

    training[
        "stop_requested"
    ] = True

    stop_event.set()

    return True


# ============================================================
# GENERATION
# ============================================================

def generate_response(
    text,
    user_id,
    temperature=DEFAULT_TEMPERATURE,
):

    global model

    retrieved, score = (
        retrieve_response(text)
    )

    # Очень похожий запрос получает
    # заранее проверенный ответ.

    if (
        retrieved is not None
        and score >= RETRIEVAL_THRESHOLD
    ):

        return retrieved[
            "response"
        ]

    memories = get_memories(
        user_id
    )

    history = get_history(
        user_id
    )

    context_parts = []

    for memory in memories:

        context_parts.append(
            f"{memory['memory_key']}: "
            f"{memory['memory_value']}"
        )

    for message in history[
        -MAX_CONTEXT_MESSAGES:
    ]:

        context_parts.append(
            message["text"]
        )

    context_parts.append(
        text
    )

    context = " ".join(
        context_parts
    )

    context_tokens = tokenize(
        context
    )

    if not context_tokens:

        return (
            "Я пока не знаю, что ответить."
        )

    with model_lock:

        if model is None:

            return (
                "Модель пока не готова."
            )

        current_model = (
            model.copy()
        )

    hidden = np.zeros(
        current_model.hidden_size
    )

    # --------------------------------------------------------
    # CONTEXT ENCODING
    # --------------------------------------------------------

    for token in context_tokens:

        input_id = token_id(
            token
        )

        if (
            input_id < 0
            or input_id
            >= current_model.vocab_size
        ):
            input_id = token_id(
                "<UNK>"
            )

        x = np.zeros(
            current_model.vocab_size
        )

        x[input_id] = 1.0

        hidden = np.tanh(
            current_model.Wxh @ x
            + current_model.Whh @ hidden
            + current_model.bh
        )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    current_id = token_id(
        "<BOS>"
    )

    generated = []

    seen = {}

    try:
        temperature = float(
            temperature
        )
    except (
        TypeError,
        ValueError,
    ):
        temperature = (
            DEFAULT_TEMPERATURE
        )

    temperature = max(
        0.2,
        min(
            temperature,
            2.0,
        ),
    )

    for _ in range(
        MAX_RESPONSE_LENGTH
    ):

        x = np.zeros(
            current_model.vocab_size
        )

        if (
            current_id < 0
            or current_id
            >= current_model.vocab_size
        ):
            current_id = token_id(
                "<UNK>"
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

        logits = (
            logits / temperature
        )

        logits -= np.max(
            logits
        )

        probs = np.exp(
            np.clip(
                logits,
                -50,
                50,
            )
        )

        # PAD и BOS не генерируем.

        for special in (
            "<PAD>",
            "<BOS>",
        ):

            idx = vocab.get(
                special
            )

            if idx is not None:
                probs[idx] = 0.0

        # UNK сильно уменьшаем.

        unk_idx = vocab.get(
            "<UNK>"
        )

        if unk_idx is not None:
            probs[unk_idx] *= 0.05

        # EOS разрешён.

        total = probs.sum()

        if (
            not np.isfinite(total)
            or total <= 0
        ):
            break

        probs /= total

        next_id = np.random.choice(
            len(probs),
            p=probs,
        )

        next_token = (
            id_to_token[next_id]
        )

        if next_token == "<EOS>":
            break

        if next_token in (
            "<PAD>",
            "<BOS>",
        ):
            break

        seen[next_token] = (
            seen.get(
                next_token,
                0,
            )
            + 1
        )

        if seen[next_token] >= 4:
            break

        generated.append(
            next_token
        )

        current_id = next_id

    answer = detokenize(
        generated
    )

    if not answer:

        if (
            retrieved is not None
            and score > 0
        ):
            return retrieved[
                "response"
            ]

        return (
            "Я пока не уверен в ответе. "
            "Добавь похожий пример в датасет, "
            "и после обучения я смогу отвечать лучше."
        )

    return answer[
        :MAX_MESSAGE_LENGTH
    ]


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
    temperature: float = (
        DEFAULT_TEMPERATURE
    )


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
def chat(
    request: ChatRequest,
):

    user_id = (
        request.user_id
        .strip()[:100]
    )

    text = (
        request.text
        .strip()
    )

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

    if len(text) > (
        MAX_MESSAGE_LENGTH
    ):

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

    retrieved, score = (
        retrieve_response(
            text
        )
    )

    return {
        "answer": answer,
        "trained_epochs": trained_epochs,
        "retrieval_available": True,
        "retrieval_score": round(
            score,
            4,
        ),
        "retrieval_used": (
            retrieved is not None
            and score
            >= RETRIEVAL_THRESHOLD
        ),
    }


# ============================================================
# HISTORY
# ============================================================

@app.get(
    "/api/history/{user_id}"
)
def history(
    user_id: str,
):

    return {
        "messages": get_history(
            user_id[:100]
        )
    }


# ============================================================
# MEMORY
# ============================================================

@app.get(
    "/api/memory/{user_id}"
)
def memory(
    user_id: str,
):

    return {
        "memories": get_memories(
            user_id[:100]
        )
    }


@app.post("/api/memory")
def create_memory(
    request: MemoryRequest,
):

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


@app.delete(
    "/api/memory/{user_id}/{key}"
)
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

@app.delete(
    "/api/chat/{user_id}"
)
def delete_chat(
    user_id: str,
):

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

@app.get(
    "/api/admin/dataset"
)
def admin_dataset(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):

    check_admin(
        x_admin_token
    )

    return {
        "dataset": dataset,
        "count": len(dataset),
    }


@app.post(
    "/api/admin/dataset"
)
def admin_add_dataset(
    request: DatasetRequest,
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):

    global vocab
    global id_to_token
    global model

    check_admin(
        x_admin_token
    )

    prompt = (
        request.prompt.strip()
    )

    response = (
        request.response.strip()
    )

    category = (
        request.category.strip()
        or "general"
    )

    if not prompt or not response:

        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt и response обязательны."
            ),
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

    with dataset_lock:

        for item in dataset:

            if (
                item["prompt"].lower(),
                item["response"].lower(),
            ) == key:

                return {
                    "ok": True,
                    "message": (
                        "Такой пример уже есть."
                    ),
                    "dataset_count": len(
                        dataset
                    ),
                }

        dataset.append(
            new_item
        )

        dataset[:] = (
            deduplicate_dataset(
                dataset
            )
        )

    save_local_dataset()

    if supabase is not None:

        try:

            supabase.table(
                "ai_dataset"
            ).upsert(
                new_item,
                on_conflict=(
                    "prompt,response"
                ),
            ).execute()

        except Exception as exc:

            print(
                "Dataset Supabase save error:",
                exc,
            )

    old_vocab = (
        vocab.copy()
    )

    build_vocab(
        dataset
    )

    with model_lock:

        model = (
            expand_model_vocabulary(
                model,
                old_vocab,
            )
        )

        save_local_model(
            model
        )

    return {
        "ok": True,
        "dataset_count": len(
            dataset
        ),
        "vocab_size": len(
            vocab
        ),
    }


# ============================================================
# APPROVED TRAINING EXAMPLES
# ============================================================

@app.post(
    "/api/admin/approve"
)
def approve_example(
    request: ApproveRequest,
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):

    global model
    global vocab
    global id_to_token

    check_admin(
        x_admin_token
    )

    item = {
        "prompt": (
            request.prompt.strip()
        ),
        "response": (
            request.response.strip()
        ),
        "category": (
            request.category.strip()
            or "approved"
        ),
    }

    if (
        not item["prompt"]
        or not item["response"]
    ):

        raise HTTPException(
            status_code=400,
            detail="Пустой пример.",
        )

    dataset.append(
        item
    )

    dataset[:] = (
        deduplicate_dataset(
            dataset
        )
    )

    save_local_dataset()

    if supabase is not None:

        try:

            supabase.table(
                "ai_dataset"
            ).upsert(
                item,
                on_conflict=(
                    "prompt,response"
                ),
            ).execute()

        except Exception as exc:

            print(
                "Approve Supabase error:",
                exc,
            )

    old_vocab = (
        vocab.copy()
    )

    build_vocab(
        dataset
    )

    with model_lock:

        model = (
            expand_model_vocabulary(
                model,
                old_vocab,
            )
        )

        save_local_model(
            model
        )

    return {
        "ok": True,
        "message": "Пример добавлен.",
        "dataset_count": len(
            dataset
        ),
    }


# ============================================================
# TRAIN API
# ============================================================

@app.post(
    "/api/admin/train"
)
def admin_train(
    request: TrainRequest,
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):

    check_admin(
        x_admin_token
    )

    try:

        epochs = int(
            request.epochs
        )

    except (
        TypeError,
        ValueError,
    ):

        raise HTTPException(
            status_code=400,
            detail="epochs должен быть числом.",
        )

    start_training(
        epochs
    )

    return {
        "ok": True,
        "message": (
            f"Запущено +{epochs} эпох."
        ),
        "current_epoch": (
            trained_epochs
        ),
        "target_epoch": min(
            MAX_TRAIN_EPOCHS,
            trained_epochs
            + epochs,
        ),
    }


# ============================================================
# STOP TRAINING
# ============================================================

@app.post(
    "/api/admin/train/stop"
)
def admin_stop_training(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):

    check_admin(
        x_admin_token
    )

    stopped = stop_training()

    return {
        "ok": True,
        "stopped": stopped,
        "message": (
            "Остановка обучения запрошена."
            if stopped
            else
            "Обучение сейчас не запущено."
        ),
    }


# ============================================================
# TRAIN STATUS
# ============================================================

@app.get(
    "/api/admin/train/status"
)
def admin_train_status(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):

    check_admin(
        x_admin_token
    )

    return {
        "running": (
            training["running"]
        ),
        "epoch": trained_epochs,
        "target_epoch": (
            training["target_epoch"]
        ),
        "loss": last_loss,
        "error": training["error"],
        "dataset_count": len(
            dataset
        ),
        "vocab_size": len(
            vocab
        ),
        "hidden_size": HIDDEN_SIZE,
        "max_epochs": MAX_TRAIN_EPOCHS,
        "learning_rate": (
            training[
                "current_learning_rate"
            ]
        ),
        "examples_processed": (
            training[
                "examples_processed"
            ]
        ),
        "stop_requested": (
            training[
                "stop_requested"
            ]
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

EVALUATION_TESTS = [

    {
        "question": "привет",
        "keywords": [
            "привет",
            "помочь",
        ],
        "category": "greeting",
    },

    {
        "question": "что ты умеешь",
        "keywords": [
            "умею",
            "помог",
        ],
        "category": "capabilities",
    },

    {
        "question": "что такое нейросеть",
        "keywords": [
            "нейросеть",
            "модель",
        ],
        "category": "education",
    },

    {
        "question": "что такое rnn",
        "keywords": [
            "rnn",
            "рекуррент",
        ],
        "category": "education",
    },

    {
        "question": "что такое python",
        "keywords": [
            "python",
            "язык",
        ],
        "category": "education",
    },

    {
        "question": "помоги с программированием",
        "keywords": [
            "код",
            "програм",
        ],
        "category": "programming",
    },

    {
        "question": "у меня typeerror",
        "keywords": [
            "typeerror",
            "тип",
        ],
        "category": "programming",
    },

    {
        "question": "как создать сайт",
        "keywords": [
            "сайт",
            "html",
        ],
        "category": "programming",
    },

    {
        "question": "у меня жирная кожа",
        "keywords": [
            "жир",
            "кож",
        ],
        "category": "skincare",
    },

    {
        "question": "у меня чёрные точки",
        "keywords": [
            "чёр",
            "точ",
            "пор",
        ],
        "category": "skincare",
    },

    {
        "question": "как ухаживать за кожей",
        "keywords": [
            "кож",
            "очищ",
        ],
        "category": "skincare",
    },

    {
        "question": "я устал",
        "keywords": [
            "устал",
            "отдох",
        ],
        "category": "wellbeing",
    },

    {
        "question": "как улучшить сон",
        "keywords": [
            "сон",
            "режим",
        ],
        "category": "wellbeing",
    },

    {
        "question": "мне скучно",
        "keywords": [
            "игр",
            "изуч",
            "проект",
        ],
        "category": "conversation",
    },

    {
        "question": "расскажи шутку",
        "keywords": [
            "програм",
            "баг",
        ],
        "category": "fun",
    },

    {
        "question": "расскажи факт",
        "keywords": [
            "осьмин",
            "серд",
        ],
        "category": "facts",
    },

    {
        "question": "зачем нужен retrieval",
        "keywords": [
            "retrieval",
            "похож",
            "ответ",
        ],
        "category": "ai_care",
    },

    {
        "question": "зачем нужна память",
        "keywords": [
            "памят",
            "сохраня",
        ],
        "category": "ai_care",
    },

    {
        "question": "что такое checkpoint",
        "keywords": [
            "checkpoint",
            "сохран",
            "модел",
        ],
        "category": "ai_care",
    },

    {
        "question": "что такое adam",
        "keywords": [
            "adam",
            "оптим",
        ],
        "category": "ai_care",
    },

]


def score_evaluation_answer(
    answer,
    keywords,
    expected=None,
):

    answer_lower = (
        answer.lower()
    )

    matched = []

    for keyword in keywords:

        if keyword.lower() in (
            answer_lower
        ):

            matched.append(
                keyword
            )

    keyword_score = (
        len(matched)
        / max(
            len(keywords),
            1,
        )
    )

    length_score = 1.0

    if len(answer.strip()) < 5:
        length_score = 0.0

    elif len(answer.strip()) > 2:
        length_score = 1.0

    final_score = (
        keyword_score * 0.85
        + length_score * 0.15
    )

    if final_score >= 0.80:
        grade = "excellent"

    elif final_score >= 0.60:
        grade = "good"

    elif final_score >= 0.35:
        grade = "weak"

    else:
        grade = "bad"

    return {
        "score": round(
            final_score,
            3,
        ),
        "grade": grade,
        "matched_keywords": matched,
        "keyword_score": round(
            keyword_score,
            3,
        ),
    }


def run_evaluation():

    results = []

    total_score = 0.0

    retrieval_hits = 0

    for test in EVALUATION_TESTS:

        question = test[
            "question"
        ]

        answer = generate_response(
            question,
            "evaluation_user",
            0.7,
        )

        retrieved, retrieval_score = (
            retrieve_response(
                question
            )
        )

        quality = (
            score_evaluation_answer(
                answer,
                test["keywords"],
            )
        )

        if (
            retrieved is not None
            and retrieval_score
            >= RETRIEVAL_THRESHOLD
        ):
            retrieval_hits += 1

        total_score += quality[
            "score"
        ]

        results.append(
            {
                "question": question,
                "category": test[
                    "category"
                ],
                "answer": answer,
                "retrieval_score": round(
                    retrieval_score,
                    3,
                ),
                "retrieval_used": (
                    retrieved is not None
                    and retrieval_score
                    >= RETRIEVAL_THRESHOLD
                ),
                "quality": quality,
            }
        )

    average_score = (
        total_score
        / max(
            len(results),
            1,
        )
    )

    result = {
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "trained_epochs": (
            trained_epochs
        ),
        "dataset_count": len(
            dataset
        ),
        "vocab_size": len(
            vocab
        ),
        "average_score": round(
            average_score,
            3,
        ),
        "retrieval_hits": (
            retrieval_hits
        ),
        "test_count": len(
            results
        ),
        "results": results,
    }

    try:

        LOCAL_EVALUATION.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    except Exception as exc:

        print(
            "Evaluation save error:",
            exc,
        )

    return result


@app.get(
    "/api/admin/evaluate"
)
def evaluate(
    x_admin_token: Optional[str] = Header(
        default=None
    ),
):

    check_admin(
        x_admin_token
    )

    return run_evaluation()


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "app": APP_NAME,
        "supabase": (
            supabase is not None
        ),
        "dataset_count": len(
            dataset
        ),
        "vocab_size": len(
            vocab
        ),
        "hidden_size": HIDDEN_SIZE,
        "trained_epochs": (
            trained_epochs
        ),
        "max_epochs": MAX_TRAIN_EPOCHS,
        "training": (
            training["running"]
        ),
        "last_loss": last_loss,
    }


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
    content="width=device-width,initial-scale=1"
>

<title>AI Care v6 — Admin</title>

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

h1,
h2 {
    margin-top: 0;
}

input,
textarea,
select {
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
    margin-bottom: 8px;
}

button:hover {
    opacity: .9;
}

button.danger {
    background: #b93838;
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

.progress {
    width: 100%;
    height: 12px;
    background: #080c15;
    border-radius: 20px;
    overflow: hidden;
    margin: 15px 0;
}

.progress-bar {
    height: 100%;
    width: 0%;
    background: #4d7cff;
    transition: width .3s;
}

.score {
    font-size: 28px;
    font-weight: bold;
    margin: 10px 0;
}

.good {
    color: #69e69a;
}

.bad {
    color: #ff7777;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>🧠 AI Care v6</h1>

<div class="small">
RNN + Adam + Retrieval + Memory + Evaluation
</div>

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

<label>
Дополнительные эпохи
</label>

<input
    id="epochs"
    type="number"
    value="10"
    min="1"
    max="100000"
/>

<button onclick="train()">
Начать обучение
</button>

<button
    class="danger"
    onclick="stopTraining()"
>
Остановить обучение
</button>

<button onclick="status()">
Обновить статус
</button>

<div class="progress">
    <div
        id="progressBar"
        class="progress-bar"
    ></div>
</div>

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

<div id="evaluationScore"></div>

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
        document.getElementById(
            "token"
        ).value;

    sessionStorage.setItem(
        "ai_admin_token",
        token
    );

    alert(
        "Токен сохранён для этой вкладки."
    );

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
        await fetch(
            url,
            options
        );

    let data;

    try {

        data =
            await response.json();

    } catch (e) {

        throw new Error(
            "Сервер вернул некорректный ответ."
        );

    }

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

        const target =
            Number(
                data.target_epoch || 0
            );

        const epoch =
            Number(
                data.epoch || 0
            );

        let progress = 0;

        if (target > 0) {

            const startEpoch =
                data.running
                    ? Math.max(
                        0,
                        epoch -
                        1
                    )
                    : 0;

            if (
                target >
                startEpoch
            ) {

                progress =
                    Math.min(
                        100,
                        Math.max(
                            0,
                            (
                                (
                                    epoch
                                    -
                                    startEpoch
                                )
                                /
                                (
                                    target
                                    -
                                    startEpoch
                                )
                            )
                            * 100
                        )
                    );

            }

        }

        document.getElementById(
            "progressBar"
        ).style.width =
            data.running
                ? progress + "%"
                : "0%";

        document.getElementById(
            "stats"
        ).innerHTML = `

            <div class="stat">
                Эпох:
                ${data.epoch}
            </div>

            <div class="stat">
                Цель:
                ${data.target_epoch}
            </div>

            <div class="stat">
                Датасет:
                ${data.dataset_count}
            </div>

            <div class="stat">
                Словарь:
                ${data.vocab_size}
            </div>

            <div class="stat">
                Hidden:
                ${data.hidden_size}
            </div>

            <div class="stat">
                Loss:
                ${data.loss ?? "-"}
            </div>

            <div class="stat">
                LR:
                ${data.learning_rate ?? "-"}
            </div>

            <div class="stat">
                ${data.running
                    ? "🟢 Обучение"
                    : "⚪ Остановлено"}
            </div>

        `;

    } catch (e) {

        document.getElementById(
            "trainingStatus"
        ).textContent =
            e.message;

    }

}


async function train() {

    const epochs =
        Number(
            document.getElementById(
                "epochs"
            ).value
        );

    if (
        !Number.isInteger(epochs)
        || epochs < 1
        || epochs > 100000
    ) {

        alert(
            "Укажи от 1 до 100000 эпох."
        );

        return;

    }

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

        alert(
            data.message
            + "\nЦелевая эпоха: "
            + data.target_epoch
        );

        status();

    } catch (e) {

        alert(
            e.message
        );

    }

}


async function stopTraining() {

    try {

        const data =
            await api(
                "/api/admin/train/stop",
                {
                    method: "POST"
                }
            );

        alert(
            data.message
        );

        status();

    } catch (e) {

        alert(
            e.message
        );

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

        alert(
            e.message
        );

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

                    <b>
                    #${index + 1}
                    [${escapeHtml(
                        item.category
                    )}]
                    </b>

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

                root.appendChild(
                    div
                );

            }
        );

    } catch (e) {

        alert(
            e.message
        );

    }

}


async function evaluateAI() {

    try {

        const data =
            await api(
                "/api/admin/evaluate"
            );

        const score =
            Number(
                data.average_score || 0
            );

        const scoreElement =
            document.getElementById(
                "evaluationScore"
            );

        scoreElement.className =
            "score "
            + (
                score >= 0.6
                    ? "good"
                    : "bad"
            );

        scoreElement.textContent =
            "Средняя оценка: "
            + Math.round(
                score * 100
            )
            + "%";

        document.getElementById(
            "evaluation"
        ).textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    } catch (e) {

        alert(
            e.message
        );

    }

}


function escapeHtml(
    text
) {

    return String(text)
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );

}


setInterval(
    () => {

        if (
            getToken()
        ) {

            status();

        }

    },
    5000
);


if (
    getToken()
) {

    document.getElementById(
        "token"
    ).value =
        getToken();

    status();
    loadDataset();

}

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
        "Hidden size:",
        HIDDEN_SIZE,
    )

    print(
        "Trained epochs:",
        trained_epochs,
    )

    print(
        "Supabase:",
        supabase is not None,
    )

    print(
        "Max epochs:",
        MAX_TRAIN_EPOCHS,
    )
