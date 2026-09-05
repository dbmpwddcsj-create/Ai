import os
import re
import json
import math
import hashlib
import secrets
import time

from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs, unquote

import numpy as np
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "ASCEND AI"

ADMIN_PASSWORD = os.getenv(
    "ADMIN_PASSWORD",
    "CHANGE_THIS_PASSWORD"
)

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
).rstrip("/")

SUPABASE_SECRET_KEY = os.getenv(
    "SUPABASE_SECRET_KEY",
    ""
)

# ============================================================
# SEARCH ENGINES
# ============================================================
#
# API НЕ используется.
#
# Порядок:
#
# 1. Yandex
# 2. Startpage
# 3. DuckDuckGo
#
# Если один поисковик отдаёт CAPTCHA / блокировку / ошибку,
# ASCEND автоматически переходит к следующему.
#

YANDEX_SEARCH_URLS = [
    "https://yandex.ru/search/",
    "https://yandex.com/search/"
]

STARTPAGE_SEARCH_URL = (
    "https://www.startpage.com/sp/search"
)

DUCKDUCKGO_SEARCH_URL = (
    "https://html.duckduckgo.com/html/"
)


# ============================================================
# LIMITS
# ============================================================

MAX_MEMORY = 30

MAX_SEARCH_RESULTS = 6

MAX_SOURCE_TEXT = 3500

MAX_MESSAGE_LENGTH = 5000


# ============================================================
# TIMEOUTS
# ============================================================

YANDEX_TIMEOUT = 15

STARTPAGE_TIMEOUT = 15

DUCKDUCKGO_TIMEOUT = 12

PAGE_TIMEOUT = 12


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version="1.3.0"
)


# ============================================================
# SUPABASE REST CLIENT
# ============================================================

def supabase_request(
    method,
    table,
    data=None,
    params=None
):
    """
    Минимальный клиент Supabase REST API.

    SUPABASE_SECRET_KEY используется только на сервере.

    Никогда не передаём этот ключ во frontend.
    """

    if (
        not SUPABASE_URL
        or
        not SUPABASE_SECRET_KEY
    ):
        return []

    url = (
        f"{SUPABASE_URL}"
        f"/rest/v1/{table}"
    )

    if params:

        try:

            url += "?" + urlencode(
                params,
                doseq=True
            )

        except Exception as e:

            print(
                "SUPABASE PARAM ERROR:",
                repr(e)
            )

            return []

    body = None

    if data is not None:

        try:

            body = json.dumps(
                data,
                ensure_ascii=False
            ).encode("utf-8")

        except Exception as e:

            print(
                "SUPABASE JSON ERROR:",
                repr(e)
            )

            return []

    headers = {

        "apikey":
            SUPABASE_SECRET_KEY,

        "Authorization":
            (
                "Bearer "
                + SUPABASE_SECRET_KEY
            ),

        "Content-Type":
            "application/json",

        "Prefer":
            "return=representation"

    }

    try:

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
            timeout=20
        )

    except Exception as e:

        print(
            "SUPABASE REQUEST ERROR:",
            repr(e)
        )

        return []

    if response.status_code >= 400:

        print(
            "SUPABASE ERROR:",
            response.status_code,
            response.text[:1000]
        )

        return []

    if not response.text:

        return []

    try:

        return response.json()

    except Exception:

        return []


# ============================================================
# TEXT NORMALIZATION
# ============================================================

RUSSIAN_STOPWORDS = {

    "и",
    "а",
    "но",
    "или",
    "да",

    "в",
    "во",
    "на",
    "за",
    "из",

    "к",
    "ко",

    "с",
    "со",

    "у",

    "о",
    "об",

    "от",
    "до",

    "по",
    "для",

    "при",

    "над",

    "под",

    "не",
    "ни",

    "же",
    "ли",
    "бы",

    "как",
    "что",
    "это",

    "этот",
    "эта",
    "эти",

    "мне",
    "меня",
    "моя",
    "мой",

    "есть",
    "можно",
    "нужно",
    "надо",

    "ну",
    "вот"
}


def normalize(
    text
):

    if not text:

        return ""

    text = str(
        text
    )

    text = text.lower()

    text = text.replace(
        "ё",
        "е"
    )

    text = re.sub(
        r"[^а-яa-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(
    text
):

    words = normalize(
        text
    ).split()

    return [

        word

        for word in words

        if (
            word
            not in RUSSIAN_STOPWORDS
            and
            len(word) >= 2
        )

    ]


def stable_hash(
    text
):

    return hashlib.sha256(
        normalize(
            text
        ).encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# SYNONYMS
# ============================================================

SYNONYMS = {

    "жирный": [

        "жирный",
        "жирная",
        "жирную",
        "сальная",
        "сальный",
        "себум",
        "жирность"

    ],

    "прыщи": [

        "прыщи",
        "прыщ",
        "акне",
        "угри",
        "угрей",
        "высыпания"

    ],

    "лицо": [

        "лицо",
        "лица",
        "лицу",
        "фейс"

    ],

    "волосы": [

        "волосы",
        "волос",
        "волосяной"

    ],

    "питание": [

        "питание",
        "еда",
        "продукты",
        "рацион",
        "диета"

    ],

    "сон": [

        "сон",
        "спать",
        "засыпать",
        "недосып",
        "бессонница"

    ],

    "тренировки": [

        "тренировка",
        "тренировки",
        "спорт",
        "мышцы",
        "зал",
        "качаться",
        "упражнение",
        "упражнения"

    ],

    "перхоть": [

        "перхоть",
        "себорейный",
        "шелушение",
        "шелушится",
        "шелушение кожи головы",
        "кожа головы"

    ],

    "мешки": [

        "мешки",
        "отеки",
        "отек",
        "под глазами",
        "глазами"

    ],

    "темные круги": [

        "темные круги",
        "темные круги под глазами",
        "синяки под глазами",
        "круги под глазами"

    ],

    "морщины": [

        "морщины",
        "морщина",
        "складки",
        "старение"

    ]

}


def expand_query(
    text
):

    normalized_text = normalize(
        text
    )

    words = tokenize(
        text
    )

    expanded = set(
        words
    )

    for canonical, variants in SYNONYMS.items():

        found = False

        for variant in variants:

            normalized_variant = normalize(
                variant
            )

            if not normalized_variant:

                continue

            if (
                normalized_variant
                in
                normalized_text
            ):

                found = True

                break

        if found:

            expanded.add(
                canonical
            )

            for variant in variants:

                for word in tokenize(
                    variant
                ):

                    expanded.add(
                        word
                    )

    return list(
        expanded
    )


# ============================================================
# NEURAL BRAIN
# ============================================================

class NeuralBrain:

    """
    Простое обучаемое классификационное ядро.

    Оно НЕ является генеративной нейросетью.

    Его задача:
    определить наиболее вероятную категорию
    вопроса пользователя.
    """

    def __init__(
        self
    ):

        self.vocabulary = []

        self.word_index = {}

        self.categories = []

        self.category_index = {}

        self.W1 = None
        self.b1 = None

        self.W2 = None
        self.b2 = None

        self.ready = False

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    def build(
        self,
        knowledge
    ):

        vocabulary = set()

        categories = set()

        for item in knowledge:

            text = (

                item.get(
                    "question",
                    ""
                )

                + " "

                + item.get(
                    "answer",
                    ""
                )

                + " "

                + " ".join(
                    item.get(
                        "tags",
                        []
                    )
                )

            )

            for word in expand_query(
                text
            ):

                vocabulary.add(
                    word
                )

            category = item.get(
                "category"
            )

            if category:

                categories.add(
                    category
                )

        self.vocabulary = sorted(
            vocabulary
        )

        self.word_index = {

            word: index

            for index, word
            in enumerate(
                self.vocabulary
            )

        }

        self.categories = sorted(
            categories
        )

        self.category_index = {

            category: index

            for index, category
            in enumerate(
                self.categories
            )

        }

        if not self.vocabulary:

            self.ready = False

            return

        if not self.categories:

            self.ready = False

            return

        input_size = len(
            self.vocabulary
        )

        hidden_size = min(

            128,

            max(
                16,
                input_size // 2
            )

        )

        output_size = len(
            self.categories
        )

        rng = np.random.default_rng(
            42
        )

        self.W1 = rng.normal(

            0,

            np.sqrt(
                2 / input_size
            ),

            (
                input_size,
                hidden_size
            )

        )

        self.b1 = np.zeros(
            hidden_size
        )

        self.W2 = rng.normal(

            0,

            np.sqrt(
                2 / hidden_size
            ),

            (
                hidden_size,
                output_size
            )

        )

        self.b2 = np.zeros(
            output_size
        )

        self.ready = True

    # --------------------------------------------------------
    # VECTORIZE
    # --------------------------------------------------------

    def vectorize(
        self,
        text
    ):

        vector = np.zeros(
            len(
                self.vocabulary
            )
        )

        for word in expand_query(
            text
        ):

            index = self.word_index.get(
                word
            )

            if index is not None:

                vector[index] += 1

        norm = np.linalg.norm(
            vector
        )

        if norm > 0:

            vector /= norm

        return vector

    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    @staticmethod
    def relu(
        x
    ):

        return np.maximum(
            0,
            x
        )

    # --------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------

    @staticmethod
    def softmax(
        x
    ):

        x = x - np.max(
            x
        )

        exp = np.exp(
            x
        )

        return exp / (
            np.sum(exp)
            + 1e-9
        )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(
        self,
        x
    ):

        z1 = (

            x
            @ self.W1
            + self.b1

        )

        h = self.relu(
            z1
        )

        z2 = (

            h
            @ self.W2
            + self.b2

        )

        output = self.softmax(
            z2
        )

        return (
            z1,
            h,
            output
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    def train(
        self,
        knowledge,
        epochs=180,
        learning_rate=0.035
    ):

        self.build(
            knowledge
        )

        if not self.ready:

            return {

                "success":
                    False,

                "epochs":
                    0

            }

        dataset = []

        for item in knowledge:

            question = item.get(
                "question",
                ""
            )

            tags = item.get(
                "tags",
                []
            )

            text = (

                question
                + " "
                + " ".join(tags)

            )

            vector = self.vectorize(
                text
            )

            category = item.get(
                "category"
            )

            label = \
                self.category_index.get(
                    category
                )

            if label is None:

                continue

            dataset.append(
                (
                    vector,
                    label
                )
            )

        if not dataset:

            return {

                "success":
                    False,

                "epochs":
                    0

            }

        for _ in range(
            epochs
        ):

            for x, label in dataset:

                z1, h, prediction = \
                    self.forward(
                        x
                    )

                target = np.zeros(
                    len(
                        self.categories
                    )
                )

                target[label] = 1

                error = (
                    prediction
                    - target
                )

                dW2 = np.outer(
                    h,
                    error
                )

                db2 = error

                dh = (
                    error
                    @ self.W2.T
                )

                dz1 = (
                    dh
                    * (z1 > 0)
                )

                dW1 = np.outer(
                    x,
                    dz1
                )

                db1 = dz1

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

        return {

            "success":
                True,

            "epochs":
                epochs,

            "samples":
                len(dataset),

            "vocabulary":
                len(
                    self.vocabulary
                ),

            "categories":
                len(
                    self.categories
                )

        }

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        text
    ):

        if not self.ready:

            return None, 0.0

        x = self.vectorize(
            text
        )

        if not np.any(x):

            return None, 0.0

        _, _, output = \
            self.forward(
                x
            )

        index = int(
            np.argmax(
                output
            )
        )

        return (

            self.categories[index],

            float(
                output[index]
            )

        )


# ============================================================
# DEFAULT KNOWLEDGE
# ============================================================

DEFAULT_KNOWLEDGE = [

    {
        "title":
            "Жирная кожа",

        "category":
            "skin",

        "question":
            "Что делать если у меня жирная кожа?",

        "answer":
            """
Если кожа быстро становится жирной, не стоит постоянно и агрессивно
обезжиривать её.

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

        "tags": [

            "жирная кожа",
            "себум",
            "кожа",
            "лицо",
            "акне",
            "прыщи"

        ]
    },


    {
        "title":
            "Прыщи",

        "category":
            "skin",

        "question":
            "Как избавиться от прыщей и акне?",

        "answer":
            """
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

        "tags": [

            "прыщи",
            "акне",
            "угри",
            "кожа",
            "лицо"

        ]
    },


    {
        "title":
            "Улучшение внешности",

        "category":
            "appearance",

        "question":
            "Как улучшить внешность?",

        "answer":
            """
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

        "tags": [

            "внешность",
            "лицо",
            "красота",
            "уход"

        ]
    },


    {
        "title":
            "Питание",

        "category":
            "nutrition",

        "question":
            "Что есть чтобы лучше выглядеть?",

        "answer":
            """
Для внешнего вида обычно важнее сбалансированный рацион,
чем экстремальная диета.

Старайся регулярно получать:

• достаточное количество белка;
• овощи и фрукты;
• цельные продукты;
• полезные жиры;
• достаточное количество жидкости.

Не нужно исключать целые группы продуктов
без конкретной причины.
""",

        "tags": [

            "питание",
            "еда",
            "рацион",
            "диета",
            "внешность"

        ]
    },


    {
        "title":
            "Сон",

        "category":
            "lifestyle",

        "question":
            "Как сон влияет на внешность?",

        "answer":
            """
Стабильный режим сна важен для общего самочувствия.

Полезно:

• ложиться примерно в одно время;
• вставать примерно в одно время;
• уменьшить яркий экран перед сном;
• не употреблять много кофеина поздно вечером;
• обеспечить комфортные условия для сна.

Главное — стабильность режима.
""",

        "tags": [

            "сон",
            "режим",
            "внешность",
            "лицо",
            "недосып"

        ]
    },


    {
        "title":
            "Тренировки",

        "category":
            "fitness",

        "question":
            "Как тренироваться чтобы улучшить тело?",

        "answer":
            """
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

        "tags": [

            "тренировки",
            "спорт",
            "мышцы",
            "тело",
            "зал"

        ]
    }

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

    if (
        not SUPABASE_URL
        or
        not SUPABASE_SECRET_KEY
    ):

        knowledge_cache = \
            DEFAULT_KNOWLEDGE.copy()

        brain.train(
            knowledge_cache
        )

        print(
            "Supabase not configured."
        )

        print(
            "Using default knowledge:",
            len(
                knowledge_cache
            )
        )

        return

    rows = supabase_request(
        "GET",
        "knowledge",
        params={

            "select":
                "*",

            "approved":
                "eq.true",

            "order":
                "created_at.desc"

        }
    )

    if rows:

        knowledge_cache = rows

    else:

        knowledge_cache = \
            DEFAULT_KNOWLEDGE.copy()

    brain.train(
        knowledge_cache
    )

    print(
        "Knowledge:",
        len(
            knowledge_cache
        )
    )

    print(
        "Brain ready:",
        brain.ready
    )


# ============================================================
# KNOWLEDGE SIMILARITY
# ============================================================

def similarity(
    a,
    b
):

    a_words = set(
        expand_query(
            a
        )
    )

    b_words = set(
        expand_query(
            b
        )
    )

    if (
        not a_words
        or
        not b_words
    ):

        return 0.0

    intersection = len(
        a_words
        &
        b_words
    )

    union = len(
        a_words
        |
        b_words
    )

    return (

        intersection
        /
        max(
            1,
            union
        )

    )


# ============================================================
# LOCAL KNOWLEDGE SEARCH
# ============================================================

def search_local_knowledge(
    query
):

    predicted_category, \
        confidence = brain.predict(
            query
        )

    results = []

    query_expanded = set(
        expand_query(
            query
        )
    )

    normalized_query = normalize(
        query
    )

    for item in knowledge_cache:

        q = item.get(
            "question",
            ""
        )

        tags = " ".join(
            item.get(
                "tags",
                []
            )
        )

        title = item.get(
            "title",
            ""
        )

        score_question = similarity(
            query,
            q
        )

        score_tags = similarity(
            query,
            tags
        )

        score_title = similarity(
            query,
            title
        )

        direct_similarity = max(

            score_question,

            score_tags,

            score_title

        )

        # ----------------------------------------------------
        # Категория теперь является только дополнительным
        # сигналом и не может сама создать совпадение.
        # ----------------------------------------------------

        category_bonus = 0.0

        if (
            predicted_category

            and

            item.get(
                "category"
            ) == predicted_category

            and

            direct_similarity >= 0.08
        ):

            category_bonus = (
                confidence
                * 0.15
            )

        score = (

            score_question
            * 0.50

            +

            score_tags
            * 0.25

            +

            score_title
            * 0.15

            +

            category_bonus

        )

        # ----------------------------------------------------
        # Прямые ключевые совпадения.
        # ----------------------------------------------------

        combined_text = normalize(

            q
            + " "
            + title
            + " "
            + tags

        )

        exact_keyword_bonus = 0.0

        for word in query_expanded:

            if (
                len(word) >= 4
                and
                word in combined_text
            ):

                exact_keyword_bonus += 0.03

        exact_keyword_bonus = min(

            exact_keyword_bonus,

            0.15

        )

        score += (
            exact_keyword_bonus
        )

        results.append(
            (
                score,
                item
            )
        )

    results.sort(

        key=lambda x: x[0],

        reverse=True

    )

    # --------------------------------------------------------
    # Минимальный порог.
    # --------------------------------------------------------

    filtered_results = [

        item

        for item in results

        if item[0] >= 0.10

    ]

    print(
        "LOCAL QUERY:",
        query
    )

    print(
        "LOCAL PREDICTED CATEGORY:",
        predicted_category
    )

    print(
        "LOCAL CATEGORY CONFIDENCE:",
        confidence
    )

    if filtered_results:

        print(
            "LOCAL TOP RESULT:",
            filtered_results[0][1].get(
                "title",
                ""
            ),
            "score=",
            filtered_results[0][0]
        )

    else:

        print(
            "LOCAL RESULT: none"
        )

    return filtered_results[:5]


# ============================================================
# COMMON WEB HELPERS
# ============================================================

def clean_text(
    text
):

    text = re.sub(
        r"\s+",
        " ",
        text or ""
    )

    return text.strip()


def valid_http_url(
    url
):

    try:

        parsed = urlparse(
            url
        )

        return parsed.scheme in {

            "http",
            "https"

        }

    except Exception:

        return False


def unwrap_yandex_url(
    url
):

    if not url:

        return ""

    try:

        parsed = urlparse(
            url
        )

        query = parse_qs(
            parsed.query
        )

        for key in (
            "url",
            "u",
            "r"
        ):

            values = query.get(
                key
            )

            if values:

                decoded = unquote(
                    values[0]
                )

                if valid_http_url(
                    decoded
                ):

                    return decoded

    except Exception:

        pass

    return url


# ============================================================
# YANDEX BLOCK DETECTOR
# ============================================================

def looks_like_yandex_block(
    html_text,
    status_code
):

    if status_code in {
        403,
        429
    }:

        return True

    normalized = (
        html_text or ""
    ).lower()

    patterns = [

        "captcha",

        "smartcaptcha",

        "проверка браузера",

        "подтвердите, что вы не робот",

        "showcaptcha",

        "робот"

    ]

    return any(

        pattern in normalized

        for pattern in patterns

    )


# ============================================================
# YANDEX PARSER
# ============================================================

def parse_yandex_results(
    html_text,
    limit=MAX_SEARCH_RESULTS
):

    soup = BeautifulSoup(
        html_text,
        "html.parser"
    )

    results = []

    seen_urls = set()

    blocks = soup.select(
        "li.serp-item"
    )

    if not blocks:

        blocks = soup.select(
            ".serp-item"
        )

    for block in blocks:

        title = ""

        title_node = (

            block.select_one(
                "h2"
            )

            or

            block.select_one(
                ".OrganicTitleContentSpan"
            )

            or

            block.select_one(
                ".OrganicTitle"
            )

        )

        if title_node:

            title = clean_text(
                title_node.get_text(
                    " ",
                    strip=True
                )
            )

        url = ""

        for a in block.find_all(
            "a"
        ):

            href = a.get(
                "href"
            )

            if not href:

                continue

            href = unwrap_yandex_url(
                href
            )

            if not valid_http_url(
                href
            ):

                continue

            host = urlparse(
                href
            ).netloc.lower()

            if "yandex." in host:

                continue

            url = href

            break

        snippet = ""

        selectors = [

            ".TextContainer",

            ".organic__content-wrapper",

            ".OrganicText",

            ".OrganicSnippet",

            ".text-container",

            ".serp-item__text"

        ]

        snippet_node = None

        for selector in selectors:

            snippet_node = \
                block.select_one(
                    selector
                )

            if snippet_node:

                break

        if snippet_node:

            snippet = clean_text(
                snippet_node.get_text(
                    " ",
                    strip=True
                )
            )

        if not snippet:

            snippet = clean_text(
                block.get_text(
                    " ",
                    strip=True
                )
            )

        if len(snippet) > 1500:

            snippet = snippet[
                :1500
            ]

        if not url:

            continue

        if url in seen_urls:

            continue

        seen_urls.add(
            url
        )

        results.append({

            "title":
                title,

            "url":
                url,

            "snippet":
                snippet,

            "source":
                "yandex"

        })

        if len(results) >= limit:

            break

    return results


# ============================================================
# YANDEX SEARCH
# ============================================================

def yandex_search(
    query,
    limit=MAX_SEARCH_RESULTS
):

    """
    Обычная HTML-страница Яндекса.

    Никакого Yandex API.
    """

    query = query.strip()

    if not query:

        return []

    headers = {

        "User-Agent":
            (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140.0.0.0 "
                "Safari/537.36"
            ),

        "Accept":
            (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "image/avif,"
                "image/webp,"
                "*/*;q=0.8"
            ),

        "Accept-Language":
            (
                "ru-RU,ru;q=0.9,"
                "en-US;q=0.8,en;q=0.7"
            ),

        "Cache-Control":
            "no-cache",

        "Pragma":
            "no-cache",

        "Upgrade-Insecure-Requests":
            "1"

    }

    for base_url in YANDEX_SEARCH_URLS:

        print("")
        print(
            "------------------------------------------"
        )

        print(
            "YANDEX QUERY:",
            query
        )

        print(
            "YANDEX URL:",
            base_url
        )

        try:

            response = requests.get(

                base_url,

                params={

                    "text":
                        query,

                    "lr":
                        "10393"

                },

                headers=headers,

                timeout=YANDEX_TIMEOUT,

                allow_redirects=True

            )

            print(
                "YANDEX HTTP:",
                response.status_code
            )

            print(
                "YANDEX FINAL URL:",
                response.url
            )

            print(
                "YANDEX CONTENT TYPE:",
                response.headers.get(
                    "content-type",
                    ""
                )
            )

        except Exception as e:

            print(
                "YANDEX REQUEST ERROR:",
                repr(e)
            )

            continue

        html_text = (
            response.text
            or
            ""
        )

        if looks_like_yandex_block(

            html_text,

            response.status_code

        ):

            print(
                "YANDEX BLOCK/CAPTCHA DETECTED"
            )

            continue

        if response.status_code >= 400:

            print(
                "YANDEX BAD STATUS"
            )

            continue

        results = parse_yandex_results(

            html_text,

            limit=limit

        )

        print(
            "YANDEX RESULTS:",
            len(results)
        )

        for index, item in enumerate(

            results,

            start=1

        ):

            print(

                f"YANDEX RESULT {index}:",

                item.get(
                    "title",
                    ""
                )[:150],

                item.get(
                    "url",
                    ""
                )

            )

        if results:

            print(
                "YANDEX SEARCH SUCCESS"
            )

            print(
                "------------------------------------------"
            )

            return results

    print(
        "YANDEX SEARCH FAILED: 0 RESULTS"
    )

    print(
        "------------------------------------------"
    )

    return []


# ============================================================
# STARTPAGE PARSER
# ============================================================

def parse_startpage_results(
    html_text,
    limit=MAX_SEARCH_RESULTS
):

    soup = BeautifulSoup(
        html_text,
        "html.parser"
    )

    results = []

    seen_urls = set()

    # Startpage uses div.result for web results.
    blocks = soup.select(
        "#results > div.result"
    )

    if not blocks:

        blocks = soup.select(
            "div.result"
        )

    for block in blocks:

        title_node = (
            block.select_one(
                "a.result-title"
            )
            or
            block.select_one(
                "h3 a"
            )
            or
            block.select_one(
                "a"
            )
        )

        if not title_node:

            continue

        href = title_node.get(
            "href",
            ""
        )

        title = clean_text(
            title_node.get_text(
                " ",
                strip=True
            )
        )

        if not href:

            continue

        if not valid_http_url(
            href
        ):

            continue

        host = urlparse(
            href
        ).netloc.lower()

        if "startpage." in host:

            continue

        if href in seen_urls:

            continue

        snippet_node = (
            block.select_one(
                ".w-gl__description"
            )
            or
            block.select_one(
                ".result-desc"
            )
            or
            block.select_one(
                ".description"
            )
            or
            block.select_one(
                "p"
            )
        )

        snippet = ""

        if snippet_node:

            snippet = clean_text(
                snippet_node.get_text(
                    " ",
                    strip=True
                )
            )

        if len(snippet) > 1500:

            snippet = snippet[
                :1500
            ]

        seen_urls.add(
            href
        )

        results.append({

            "title":
                title[:250],

            "url":
                href,

            "snippet":
                snippet,

            "source":
                "startpage"

        })

        if len(results) >= limit:

            break

    return results


# ============================================================
# STARTPAGE SEARCH
# ============================================================

def startpage_search(
    query,
    limit=MAX_SEARCH_RESULTS
):

    """
    Обычный Startpage без API.
    Используем GET-форму поиска.
    """

    query = query.strip()

    if not query:

        return []

    headers = {

        "User-Agent":
            (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140.0.0.0 "
                "Safari/537.36"
            ),

        "Accept":
            (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "*/*;q=0.8"
            ),

        "Accept-Language":
            (
                "ru-RU,ru;q=0.9,"
                "en-US;q=0.8,en;q=0.7"
            ),

        "Referer":
            "https://www.startpage.com/"

    }

    print(
        "------------------------------------------"
    )

    print(
        "STARTPAGE QUERY:",
        query
    )

    try:

        response = requests.get(

            STARTPAGE_SEARCH_URL,

            params={

                "query":
                    query

            },

            headers=headers,

            timeout=STARTPAGE_TIMEOUT,

            allow_redirects=True

        )

        print(
            "STARTPAGE HTTP:",
            response.status_code
        )

        print(
            "STARTPAGE FINAL URL:",
            response.url
        )

        print(
            "STARTPAGE CONTENT TYPE:",
            response.headers.get(
                "content-type",
                ""
            )
        )

    except Exception as e:

        print(
            "STARTPAGE ERROR:",
            repr(e)
        )

        print(
            "------------------------------------------"
        )

        return []

    if response.status_code >= 400:

        print(
            "STARTPAGE BAD STATUS"
        )

        print(
            "------------------------------------------"
        )

        return []

    html_text = (
        response.text
        or
        ""
    )

    lower_html = html_text.lower()

    block_patterns = [

        "captcha",

        "verify you are human",

        "are you a robot",

        "unusual traffic",

        "challenge"

    ]

    if any(

        pattern in lower_html

        for pattern in block_patterns

    ):

        print(
            "STARTPAGE BLOCK/CAPTCHA DETECTED"
        )

        print(
            "------------------------------------------"
        )

        return []

    results = parse_startpage_results(

        html_text,

        limit=limit

    )

    print(
        "STARTPAGE RESULTS:",
        len(results)
    )

    for index, item in enumerate(

        results,

        start=1

    ):

        print(

            f"STARTPAGE RESULT {index}:",

            item.get(
                "title",
                ""
            )[:150],

            item.get(
                "url",
                ""
            )

        )

    if results:

        print(
            "STARTPAGE SEARCH SUCCESS"
        )

    else:

        print(
            "STARTPAGE SEARCH FAILED: 0 RESULTS"
        )

    print(
        "------------------------------------------"
    )

    return results


# ============================================================
# DUCKDUCKGO SEARCH
# ============================================================

def duckduckgo_search(
    query,
    limit=MAX_SEARCH_RESULTS
):

    """
    Резервный обычный DuckDuckGo HTML-поиск.
    """

    query = query.strip()

    if not query:

        return []

    headers = {

        "User-Agent":
            (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140.0.0.0 "
                "Safari/537.36"
            ),

        "Accept":
            (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "*/*;q=0.8"
            ),

        "Accept-Language":
            (
                "ru-RU,ru;q=0.9,"
                "en-US;q=0.8,en;q=0.7"
            ),

        "Referer":
            "https://html.duckduckgo.com/"

    }

    print(
        "------------------------------------------"
    )

    print(
        "DUCKDUCKGO QUERY:",
        query
    )

    try:

        response = requests.get(

            DUCKDUCKGO_SEARCH_URL,

            params={

                "q":
                    query,

                "kl":
                    "ru-ru",

                "kp":
                    "1"

            },

            headers=headers,

            timeout=DUCKDUCKGO_TIMEOUT,

            allow_redirects=True

        )

        print(
            "DUCKDUCKGO HTTP:",
            response.status_code
        )

        print(
            "DUCKDUCKGO FINAL URL:",
            response.url
        )

    except Exception as e:

        print(
            "DUCKDUCKGO ERROR:",
            repr(e)
        )

        print(
            "------------------------------------------"
        )

        return []

    if response.status_code >= 400:

        print(
            "DUCKDUCKGO BAD STATUS"
        )

        print(
            "------------------------------------------"
        )

        return []

    html_text = (
        response.text
        or
        ""
    )

    lower_html = html_text.lower()

    if (
        "captcha"
        in
        lower_html
        or
        "challenge"
        in
        lower_html
        or
        "verify you are human"
        in
        lower_html
    ):

        print(
            "DUCKDUCKGO BLOCK/CAPTCHA DETECTED"
        )

        print(
            "------------------------------------------"
        )

        return []

    soup = BeautifulSoup(
        html_text,
        "html.parser"
    )

    results = []

    seen_urls = set()

    blocks = soup.select(
        ".result"
    )

    for block in blocks:

        title_node = block.select_one(
            ".result__a"
        )

        if not title_node:

            continue

        href = title_node.get(
            "href",
            ""
        )

        title = clean_text(
            title_node.get_text(
                " ",
                strip=True
            )
        )

        snippet_node = \
            block.select_one(
                ".result__snippet"
            )

        snippet = ""

        if snippet_node:

            snippet = clean_text(
                snippet_node.get_text(
                    " ",
                    strip=True
                )
            )

        if not href:

            continue

        if not valid_http_url(
            href
        ):

            continue

        if href in seen_urls:

            continue

        if not title:

            continue

        seen_urls.add(
            href
        )

        results.append({

            "title":
                title[:250],

            "url":
                href,

            "snippet":
                snippet[:1500],

            "source":
                "duckduckgo"

        })

        if len(results) >= limit:

            break

    print(
        "DUCKDUCKGO RESULTS:",
        len(results)
    )

    for index, item in enumerate(

        results,

        start=1

    ):

        print(

            f"DUCKDUCKGO RESULT {index}:",

            item.get(
                "title",
                ""
            )[:150],

            item.get(
                "url",
                ""
            )

        )

    if results:

        print(
            "DUCKDUCKGO SEARCH SUCCESS"
        )

    else:

        print(
            "DUCKDUCKGO SEARCH FAILED: 0 RESULTS"
        )

    print(
        "------------------------------------------"
    )

    return results


# ============================================================
# PAGE FETCH
# ============================================================

def fetch_page_text(
    url
):

    if not valid_http_url(
        url
    ):

        return ""

    headers = {

        "User-Agent":
            (
                "Mozilla/5.0 "
                "(compatible; ASCEND-AI/1.3)"
            ),

        "Accept-Language":
            (
                "ru-RU,ru;q=0.9,"
                "en;q=0.8"
            ),

        "Accept":
            (
                "text/html,"
                "application/xhtml+xml"
            )

    }

    print(
        "FETCH SOURCE:",
        url
    )

    try:

        response = requests.get(

            url,

            headers=headers,

            timeout=PAGE_TIMEOUT,

            allow_redirects=True

        )

        print(
            "SOURCE HTTP:",
            response.status_code
        )

        print(
            "SOURCE FINAL URL:",
            response.url
        )

        if response.status_code >= 400:

            print(
                "SOURCE ERROR STATUS"
            )

            return ""

        content_type = (

            response.headers.get(
                "content-type",
                ""
            )

            .lower()

        )

        if (
            "text/html"
            not in content_type
        ):

            print(
                "SOURCE NOT HTML:",
                content_type
            )

            return ""

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([

            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
            "form"

        ]):

            tag.decompose()

        text = clean_text(

            soup.get_text(
                " ",
                strip=True
            )

        )

        text = text[
            :MAX_SOURCE_TEXT
        ]

        print(
            "SOURCE TEXT LENGTH:",
            len(text)
        )

        return text

    except Exception as e:

        print(
            "SOURCE FETCH ERROR:",
            repr(e)
        )

        return ""


# ============================================================
# COLLECT WEB INFORMATION
# ============================================================

def collect_web_information(
    query
):

    print("")
    print(
        "=========================================="
    )

    print(
        "WEB SEARCH START"
    )

    print(
        "QUERY:",
        query
    )

    # ========================================================
    # 1. YANDEX
    # ========================================================

    print(
        "WEB SEARCH: trying Yandex..."
    )

    search_results = yandex_search(
        query
    )

    search_engine = "yandex"

    # ========================================================
    # 2. STARTPAGE
    # ========================================================

    if not search_results:

        print(
            "WEB SEARCH: Yandex failed."
        )

        print(
            "WEB SEARCH: switching to Startpage..."
        )

        search_results = \
            startpage_search(
                query
            )

        search_engine = "startpage"

    # ========================================================
    # 3. DUCKDUCKGO
    # ========================================================

    if not search_results:

        print(
            "WEB SEARCH: Startpage failed."
        )

        print(
            "WEB SEARCH: switching to DuckDuckGo..."
        )

        search_results = \
            duckduckgo_search(
                query
            )

        search_engine = "duckduckgo"

    # ========================================================
    # 4. ALL FAILED
    # ========================================================

    if not search_results:

        print(
            "WEB SEARCH: all search engines failed."
        )

        print(
            "=========================================="
        )

        return []

    print(
        "WEB SEARCH ENGINE:",
        search_engine
    )

    print(
        "WEB SEARCH RESULTS:",
        len(
            search_results
        )
    )

    # ========================================================
    # FETCH SOURCE PAGES
    # ========================================================

    enriched = []

    for index, result in enumerate(

        search_results,

        start=1

    ):

        page_text = fetch_page_text(
            result["url"]
        )

        enriched.append({

            **result,

            "page_text":
                page_text

        })

        print(

            f"SOURCE {index}: "
            f"title={result.get('title', '')[:100]} "
            f"text={len(page_text)} "
            f"snippet={len(result.get('snippet', ''))}"

        )

    print(
        "WEB SEARCH COMPLETE"
    )

    print(
        "=========================================="
    )

    return enriched


# ============================================================
# SAVE WEB SOURCES
# ============================================================

def save_web_sources(
    session_id,
    query,
    results
):

    if (
        not SUPABASE_URL
        or
        not SUPABASE_SECRET_KEY
    ):

        return

    for result in results:

        payload = {

            "session_id":
                session_id,

            "query":
                query,

            "title":
                result.get(
                    "title",
                    ""
                ),

            "url":
                result.get(
                    "url",
                    ""
                ),

            "snippet":
                result.get(
                    "snippet",
                    ""
                ),

            "page_text":
                result.get(
                    "page_text",
                    ""
                ),

            "source":
                result.get(
                    "source",
                    "web"
                )

        }

        supabase_request(

            "POST",

            "web_sources",

            payload

        )


# ============================================================
# WEB CONTEXT
# ============================================================

def build_web_context(
    results
):

    pieces = []

    for index, item in enumerate(

        results,

        start=1

    ):

        title = item.get(
            "title",
            ""
        )

        url = item.get(
            "url",
            ""
        )

        snippet = item.get(
            "snippet",
            ""
        )

        page_text = item.get(
            "page_text",
            ""
        )

        text = (

            page_text
            or
            snippet

        )

        if not text:

            continue

        pieces.append(

            f"""
ИСТОЧНИК {index}
Название: {title}
URL: {url}

Информация:
{text}
"""

        )

    return "\n".join(
        pieces
    )


# ============================================================
# SENTENCE SPLIT
# ============================================================

def split_sentences(
    text
):

    text = text.replace(
        "\n",
        " "
    )

    parts = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [

        clean_text(
            x
        )

        for x in parts

        if len(
            clean_text(
                x
            )
        ) > 20

    ]


# ============================================================
# SENTENCE RANKING
# ============================================================

def rank_sentences(
    query,
    text,
    limit=8
):

    sentences = split_sentences(
        text
    )

    qwords = set(
        expand_query(
            query
        )
    )

    scored = []

    for sentence in sentences:

        swords = set(
            expand_query(
                sentence
            )
        )

        overlap = len(
            qwords
            &
            swords
        )

        if overlap:

            score = (

                overlap

                /

                math.sqrt(
                    max(
                        1,
                        len(swords)
                    )
                )

            )

            scored.append(

                (
                    score,
                    sentence
                )

            )

    scored.sort(

        key=lambda x: x[0],

        reverse=True

    )

    return [

        sentence

        for _, sentence
        in scored[:limit]

    ]


# ============================================================
# WEB FALLBACK RESPONSE
# ============================================================

def fallback_web_answer(
    query,
    web_results
):

    web_text = build_web_context(
        web_results
    )

    if not web_text:

        return ""

    sentences = rank_sentences(

        query,

        web_text,

        limit=8

    )

    if not sentences:

        sentences = split_sentences(
            web_text
        )[:5]

    if not sentences:

        return ""

    answer = (

        "🌐 Я нашёл информацию "
        "по твоему вопросу в интернете.\n\n"

    )

    for sentence in sentences:

        answer += (

            "• "
            + sentence
            + "\n"

        )

    answer += (

        "\n"
        "⚠️ Информация взята из найденных "
        "в интернете источников. Для важных "
        "вопросов проверяй первоисточники."

    )

    return answer


# ============================================================
# RESPONSE GENERATOR
# ============================================================

def generate_response(

    query,

    memory,

    local_results,

    web_results

):

    best_score = 0.0

    best_item = None

    if local_results:

        best_score, best_item = \
            local_results[0]

    print(
        "LOCAL BEST SCORE:",
        best_score
    )

    if best_item:

        print(
            "LOCAL BEST:",
            best_item.get(
                "title",
                ""
            )
        )

    print(
        "WEB RESULTS:",
        len(
            web_results
        )
    )

    # ========================================================
    # LOCAL KNOWLEDGE
    # ========================================================

    if (
        best_item
        and
        best_score >= 0.18
    ):

        answer = (

            best_item
            .get(
                "answer",
                ""
            )
            .strip()

        )

        # Дополнение веб-данными,
        # только если есть реальные
        # релевантные предложения.

        if web_results:

            web_text = \
                build_web_context(
                    web_results
                )

            sentences = rank_sentences(

                query,

                web_text,

                limit=4

            )

            if sentences:

                answer += (

                    "\n\n"
                    "🌐 Дополнение из "
                    "актуального поиска:\n"

                )

                for sentence in sentences:

                    answer += (

                        "\n• "
                        + sentence

                    )

        return answer

    # ========================================================
    # WEB ANSWER
    # ========================================================

    web_answer = fallback_web_answer(

        query,

        web_results

    )

    if web_answer:

        return web_answer

    # ========================================================
    # FINAL FALLBACK
    # ========================================================

    return (

        "Веб-поиск сейчас не смог получить "
        "результаты от доступных поисковиков.\n\n"
        "Попробуй повторить запрос немного позже "
        "или сформулировать его подробнее."

    )


# ============================================================
# MEMORY
# ============================================================

def save_message(

    session_id,

    role,

    content

):

    if (
        not SUPABASE_URL
        or
        not SUPABASE_SECRET_KEY
    ):

        return None

    rows = supabase_request(

        "POST",

        "chat_messages",

        {

            "session_id":
                session_id,

            "role":
                role,

            "content":
                content

        }

    )

    if rows:

        return rows[0]

    return None


def get_memory(
    session_id
):

    if (
        not SUPABASE_URL
        or
        not SUPABASE_SECRET_KEY
    ):

        return []

    rows = supabase_request(

        "GET",

        "chat_messages",

        params={

            "select":
                "role,content,created_at",

            "session_id":
                f"eq.{session_id}",

            "order":
                "created_at.desc",

            "limit":
                str(
                    MAX_MEMORY
                )

        }

    )

    rows.reverse()

    return rows


# ============================================================
# TRAINING LOG
# ============================================================

def save_training_log(

    question,

    answer,

    category,

    source

):

    if (
        not SUPABASE_URL
        or
        not SUPABASE_SECRET_KEY
    ):

        return

    supabase_request(

        "POST",

        "training_log",

        {

            "question":
                question,

            "answer":
                answer,

            "category":
                category,

            "source":
                source,

            "approved":
                True

        }

    )


# ============================================================
# ADMIN AUTH
# ============================================================

def create_admin_token():

    timestamp = str(
        int(
            time.time()
        )
    )

    raw = (

        ADMIN_PASSWORD
        + ":"
        + timestamp

    )

    signature = hashlib.sha256(

        raw.encode()

    ).hexdigest()

    return (

        timestamp
        + "."
        + signature

    )


def verify_admin_token(
    token
):

    if not token:

        return False

    parts = token.split(
        "."
    )

    if len(parts) != 2:

        return False

    timestamp, signature = parts

    try:

        timestamp_int = int(
            timestamp
        )

    except Exception:

        return False

    if (

        abs(

            int(
                time.time()
            )
            -
            timestamp_int

        )

        >

        43200

    ):

        return False

    expected = hashlib.sha256(

        (

            ADMIN_PASSWORD
            + ":"
            + timestamp

        ).encode()

    ).hexdigest()

    return secrets.compare_digest(

        signature,

        expected

    )


def check_admin(
    request: Request
):

    token = request.headers.get(

        "X-Admin-Token",

        ""

    )

    if not verify_admin_token(
        token
    ):

        raise HTTPException(

            status_code=401,

            detail="Нет доступа."

        )


# ============================================================
# MODELS
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


class FeedbackRequest(BaseModel):

    session_id: str

    message_id: Optional[str] = None

    rating: int

    comment: Optional[str] = ""


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="ru">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>ASCEND AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    background: #09090b;
    color: white;
    font-family:
        Inter,
        Arial,
        sans-serif;
}

button,
textarea,
input {
    font: inherit;
}

button {
    cursor: pointer;
}

.container {
    width: min(1100px, 94%);
    margin: 0 auto;
}

.header {
    height: 75px;
    border-bottom: 1px solid #252529;
    display: flex;
    align-items: center;
}

.header-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 22px;
    font-weight: 900;
    letter-spacing: .5px;
}

.logo span {
    opacity: .4;
}

.header-actions {
    display: flex;
    gap: 8px;
}

.header-button {
    background: #151518;
    color: white;
    border: 1px solid #2b2b30;
    border-radius: 12px;
    padding: 10px 15px;
}

.chat {
    height: calc(100vh - 75px);
    display: flex;
    flex-direction: column;
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 35px 0;
}

.message {
    display: flex;
    margin-bottom: 22px;
}

.message.user {
    justify-content: flex-end;
}

.message.ai {
    justify-content: flex-start;
}

.bubble {
    max-width: min(760px, 85%);
    padding: 16px 19px;
    border-radius: 19px;
    line-height: 1.6;
    white-space: pre-wrap;
}

.message.ai .bubble {
    background: #141416;
    border: 1px solid #28282c;
}

.message.user .bubble {
    background: #f7d45b;
    color: #111;
}

.sources {
    max-width: 760px;
    margin-top: -10px;
    margin-bottom: 25px;
}

.source-card {
    background: #111113;
    border: 1px solid #252529;
    border-radius: 12px;
    padding: 12px;
    margin-top: 7px;
}

.source-card a {
    color: #f7d45b;
    text-decoration: none;
    word-break: break-word;
}

.composer {
    padding: 15px 0 25px;
}

.composer-box {
    display: flex;
    gap: 10px;
    background: #111113;
    border: 1px solid #29292e;
    padding: 9px;
    border-radius: 17px;
}

.composer textarea {
    flex: 1;
    border: 0;
    outline: 0;
    background: transparent;
    color: white;
    resize: none;
    padding: 13px;
    min-height: 50px;
    max-height: 150px;
}

.send {
    min-width: 110px;
    border: 0;
    border-radius: 12px;
    background: #f7d45b;
    color: #111;
    font-weight: 800;
}

.send:disabled {
    opacity: .55;
    cursor: not-allowed;
}

.admin {
    display: none;
    padding: 35px 0 60px;
}

.card {
    background: #111113;
    border: 1px solid #29292e;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.card h2 {
    margin-top: 0;
}

.field {
    margin-bottom: 15px;
}

.field label {
    display: block;
    opacity: .65;
    margin-bottom: 7px;
    font-size: 13px;
}

.field input,
.field textarea {
    width: 100%;
    background: #09090b;
    color: white;
    border: 1px solid #29292e;
    border-radius: 11px;
    padding: 13px;
    outline: none;
}

.field textarea {
    min-height: 150px;
    resize: vertical;
}

.primary {
    background: #f7d45b;
    color: #111;
    border: 0;
    border-radius: 11px;
    padding: 12px 17px;
    font-weight: 800;
}

.primary:disabled {
    opacity: .55;
    cursor: not-allowed;
}

.knowledge-item {
    border-top: 1px solid #29292e;
    padding: 17px 0;
}

.knowledge-item:first-child {
    border-top: 0;
}

.badge {
    display: inline-block;
    background: #242428;
    padding: 5px 8px;
    border-radius: 7px;
    font-size: 12px;
    opacity: .8;
}

.status {
    margin-top: 12px;
    opacity: .7;
    font-size: 13px;
}

.hidden {
    display: none !important;
}

@media(max-width:700px) {

    .bubble {
        max-width: 94%;
    }

    .send {
        min-width: 80px;
    }

    .header-button {
        padding: 9px 11px;
    }

}

</style>

</head>

<body>

<header class="header">

<div class="container header-inner">

<div class="logo">
ASCEND <span>AI</span>
</div>

<div class="header-actions">

<button
class="header-button"
onclick="showChat()">
💬 Чат
</button>

<button
class="header-button"
onclick="showAdmin()">
⚙ Админка
</button>

</div>

</div>

</header>


<main class="container">


<section id="chatSection">

<div class="chat">

<div
id="messages"
class="messages">

<div class="message ai">

<div class="bubble">
Привет! Я ASCEND AI 🧠

Я могу помочь с вопросами
об уходе за кожей, лице,
внешности, питании,
волосах и тренировках.

Я также могу искать информацию
в обычном интернете.

Сначала проверяю собственную базу,
а при необходимости ищу информацию
в интернете.

Что тебя интересует?
</div>

</div>

</div>


<div class="composer">

<div class="composer-box">

<textarea
id="messageInput"
placeholder="Напиши свой вопрос..."
></textarea>

<button
id="sendButton"
class="send"
onclick="sendMessage()">

Отправить

</button>

</div>

</div>

</div>

</section>


<section
id="adminSection"
class="admin">


<div
id="adminLoginCard"
class="card">

<h2>⚙️ Админка</h2>

<p>
Вход в панель управления
нейросетью.
</p>

<div class="field">

<label>
Пароль
</label>

<input
id="adminPassword"
type="password"
placeholder="Пароль администратора">

</div>

<button
class="primary"
onclick="loginAdmin()">

Войти

</button>

<div
id="loginStatus"
class="status">
</div>

</div>


<div
id="adminPanel"
class="hidden">


<div class="card">

<h2>🧠 Состояние нейросети</h2>

<div id="brainStats">
Загрузка...
</div>

</div>


<div class="card">

<h2>📚 Добавить знание</h2>

<div class="field">

<label>
Название
</label>

<input
id="title"
placeholder="Например: Жирная кожа">

</div>


<div class="field">

<label>
Категория
</label>

<input
id="category"
placeholder="skin">

</div>


<div class="field">

<label>
Пример вопроса пользователя
</label>

<input
id="question"
placeholder="Что делать если у меня жирная кожа?">

</div>


<div class="field">

<label>
Ответ нейросети
</label>

<textarea
id="answer"
placeholder="Напиши правильный ответ...">
</textarea>

</div>


<div class="field">

<label>
Теги через запятую
</label>

<input
id="tags"
placeholder="кожа, жирная кожа, себум, лицо">

</div>


<button
class="primary"
onclick="addKnowledge()">

🧠 Обучить нейросеть

</button>

<div
id="trainStatus"
class="status">
</div>

</div>


<div class="card">

<h2>📖 База знаний</h2>

<div id="knowledgeList">
Загрузка...
</div>

</div>


<div class="card">

<h2>🌐 Последние поиски</h2>

<div id="webSearchList">
Загрузка...
</div>

</div>


</div>

</section>


</main>


<script>

const SESSION_KEY =
    "ascend_session_id";


let sessionId =
    localStorage.getItem(
        SESSION_KEY
    );


if (!sessionId) {

    sessionId =
        crypto.randomUUID();

    localStorage.setItem(
        SESSION_KEY,
        sessionId
    );
}


let adminToken =
    localStorage.getItem(
        "ascend_admin_token"
    );


function addMessage(
    role,
    text
) {

    const messages =
        document.getElementById(
            "messages"
        );

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        "message " +
        (
            role === "user"
            ? "user"
            : "ai"
        );


    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        "bubble";

    bubble.textContent =
        text;


    wrapper.appendChild(
        bubble
    );

    messages.appendChild(
        wrapper
    );

    messages.scrollTop =
        messages.scrollHeight;

}


function addSources(
    sources
) {

    if (
        !sources
        ||
        sources.length === 0
    ) {

        return;
    }


    const messages =
        document.getElementById(
            "messages"
        );


    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        "sources";


    const heading =
        document.createElement(
            "div"
        );

    heading.textContent =
        "🌐 Источники:";

    heading.style.opacity =
        "0.65";

    heading.style.marginBottom =
        "8px";

    wrapper.appendChild(
        heading
    );


    sources.forEach(
        source => {

            const card =
                document.createElement(
                    "div"
                );

            card.className =
                "source-card";


            const title =
                document.createElement(
                    "div"
                );

            title.textContent =
                source.title ||
                source.url;


            const link =
                document.createElement(
                    "a"
                );

            link.href =
                source.url;

            link.target =
                "_blank";

            link.rel =
                "noopener noreferrer";

            link.textContent =
                source.url;


            card.appendChild(
                title
            );

            card.appendChild(
                link
            );

            wrapper.appendChild(
                card
            );

        }
    );


    messages.appendChild(
        wrapper
    );

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


    if (!message) {

        return;
    }


    if (
        message.length > 5000
    ) {

        alert(
            "Сообщение слишком длинное."
        );

        return;
    }


    addMessage(
        "user",
        message
    );


    input.value = "";

    button.disabled = true;

    button.textContent =
        "Ищу...";


    addMessage(
        "ai",
        "🌐 Проверяю собственную базу и ищу информацию в интернете..."
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

                    body: JSON.stringify({

                        session_id:
                            sessionId,

                        message:
                            message

                    })

                }
            );


        let data;

        try {

            data =
                await response.json();

        } catch {

            data = {

                detail:
                    "Сервер вернул некорректный ответ."

            };

        }


        const messages =
            document.getElementById(
                "messages"
            );


        if (
            messages.lastElementChild
        ) {

            messages.lastElementChild.remove();

        }


        if (!response.ok) {

            addMessage(

                "ai",

                data.detail
                ||
                "Ошибка сервера."

            );

        } else {

            addMessage(

                "ai",

                data.answer
                ||
                "Сервер не вернул ответ."

            );

            addSources(
                data.sources
            );

        }

    } catch (error) {

        console.error(
            error
        );

        const messages =
            document.getElementById(
                "messages"
            );


        if (
            messages.lastElementChild
        ) {

            messages.lastElementChild.remove();

        }


        addMessage(

            "ai",

            "Ошибка соединения с сервером."

        );

    }


    button.disabled = false;

    button.textContent =
        "Отправить";

}


document
.getElementById(
    "messageInput"
)
.addEventListener(
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


function showChat() {

    document.getElementById(
        "chatSection"
    ).style.display =
        "block";


    document.getElementById(
        "adminSection"
    ).style.display =
        "none";

}


function showAdmin() {

    document.getElementById(
        "chatSection"
    ).style.display =
        "none";


    document.getElementById(
        "adminSection"
    ).style.display =
        "block";

}


async function loginAdmin() {

    const password =
        document.getElementById(
            "adminPassword"
        ).value;


    const status =
        document.getElementById(
            "loginStatus"
        );


    status.textContent =
        "Проверка...";


    try {

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


        let data;

        try {

            data =
                await response.json();

        } catch {

            data = {

                detail:
                    "Некорректный ответ сервера."

            };

        }


        if (!response.ok) {

            status.textContent =
                data.detail
                ||
                "Неверный пароль.";

            return;
        }


        adminToken =
            data.token;


        localStorage.setItem(
            "ascend_admin_token",
            adminToken
        );


        document.getElementById(
            "adminPanel"
        ).classList.remove(
            "hidden"
        );


        status.textContent =
            "Авторизация успешна.";

        loadAdminData();

    } catch {

        status.textContent =
            "Ошибка соединения.";

    }

}


function adminHeaders() {

    return {

        "Content-Type":
            "application/json",

        "X-Admin-Token":
            adminToken

    };

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
        .map(
            x => x.trim()
        )
        .filter(Boolean);


    const status =
        document.getElementById(
            "trainStatus"
        );


    status.textContent =
        "Обучаю нейросеть...";


    try {

        const response =
            await fetch(
                "/api/admin/knowledge",
                {

                    method: "POST",

                    headers:
                        adminHeaders(),

                    body: JSON.stringify({

                        title,
                        category,
                        question,
                        answer,
                        tags

                    })

                }
            );


        let data;

        try {

            data =
                await response.json();

        } catch {

            data = {

                detail:
                    "Некорректный ответ сервера."

            };

        }


        if (!response.ok) {

            status.textContent =
                data.detail
                ||
                "Ошибка.";

            return;
        }


        status.textContent =
            "✅ Знание добавлено. Нейросеть переобучена.";


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


        loadAdminData();

    } catch (error) {

        console.error(
            error
        );

        status.textContent =
            "Ошибка соединения с сервером.";

    }

}


async function loadAdminData() {

    if (!adminToken) {

        return;
    }


    try {

        const statsResponse =
            await fetch(

                "/api/admin/stats",

                {

                    headers:
                        adminHeaders()

                }

            );


        if (statsResponse.ok) {

            const stats =
                await statsResponse.json();


            document.getElementById(
                "brainStats"
            ).innerHTML = `

                <p>
                    🧠 Модель:
                    <strong>
                        ${
                            stats.brain_ready
                            ? "готова"
                            : "не готова"
                        }
                    </strong>
                </p>

                <p>
                    📚 Знаний:
                    <strong>
                        ${stats.knowledge}
                    </strong>
                </p>

                <p>
                    🔤 Словарь:
                    <strong>
                        ${stats.vocabulary}
                    </strong>
                </p>

                <p>
                    🏷️ Категорий:
                    <strong>
                        ${stats.categories}
                    </strong>
                </p>

            `;

        }


        const knowledgeResponse =
            await fetch(

                "/api/admin/knowledge",

                {

                    headers:
                        adminHeaders()

                }

            );


        if (
            !knowledgeResponse.ok
        ) {

            return;
        }


        const knowledge =
            await knowledgeResponse.json();


        const list =
            document.getElementById(
                "knowledgeList"
            );


        list.innerHTML = "";


        knowledge.forEach(
            item => {

                const element =
                    document.createElement(
                        "div"
                    );

                element.className =
                    "knowledge-item";


                element.innerHTML = `

                    <span class="badge">

                        ${
                            escapeHtml(
                                item.category
                                ||
                                ""
                            )
                        }

                    </span>

                    <h3>

                        ${
                            escapeHtml(
                                item.title
                                ||
                                ""
                            )
                        }

                    </h3>

                    <p>

                        <strong>
                            Вопрос:
                        </strong>

                        <br>

                        ${
                            escapeHtml(
                                item.question
                                ||
                                ""
                            )
                        }

                    </p>

                    <p>

                        ${
                            escapeHtml(
                                item.answer
                                ||
                                ""
                            )
                        }

                    </p>

                `;


                list.appendChild(
                    element
                );

            }
        );


        const webResponse =
            await fetch(

                "/api/admin/web-sources",

                {

                    headers:
                        adminHeaders()

                }

            );


        if (webResponse.ok) {

            const webData =
                await webResponse.json();


            const webList =
                document.getElementById(
                    "webSearchList"
                );


            webList.innerHTML = "";


            webData.forEach(
                item => {

                    const element =
                        document.createElement(
                            "div"
                        );

                    element.className =
                        "knowledge-item";


                    element.innerHTML = `

                        <span class="badge">

                            🌐

                            ${
                                escapeHtml(
                                    item.source
                                    ||
                                    "web"
                                )
                            }

                        </span>

                        <h3>

                            ${
                                escapeHtml(
                                    item.title
                                    ||
                                    ""
                                )
                            }

                        </h3>

                        <p>

                            <strong>
                                Запрос:
                            </strong>

                            ${
                                escapeHtml(
                                    item.query
                                    ||
                                    ""
                                )
                            }

                        </p>

                        <a

                            href="${
                                escapeHtml(
                                    item.url
                                    ||
                                    "#"
                                )
                            }"

                            target="_blank"

                            rel="noopener noreferrer">

                            Открыть источник

                        </a>

                    `;


                    webList.appendChild(
                        element
                    );

                }
            );

        }

    } catch (error) {

        console.error(
            error
        );

    }

}


function escapeHtml(
    value
) {

    return String(
        value ?? ""
    )

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

</script>

</body>

</html>
"""


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def index():

    return HTML


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
async def health():

    return {

        "status":
            "ok",

        "brain_ready":
            brain.ready,

        "knowledge":
            len(
                knowledge_cache
            ),

        "yandex_search":
            True,

        "yandex_api":
            False,

        "startpage_fallback":
            True,

        "duckduckgo_fallback":
            True

    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/api/chat"
)
async def chat(
    data: ChatRequest
):

    message = data.message.strip()

    if not message:

        raise HTTPException(

            400,

            "Пустой запрос."

        )


    if (
        len(message)
        >
        MAX_MESSAGE_LENGTH
    ):

        raise HTTPException(

            400,

            "Сообщение слишком длинное."

        )


    print("")
    print(
        "=" * 60
    )

    print(
        "NEW CHAT REQUEST:",
        message
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    memory = get_memory(
        data.session_id
    )


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    save_message(

        data.session_id,

        "user",

        message

    )


    # --------------------------------------------------------
    # LOCAL KNOWLEDGE
    # --------------------------------------------------------

    local_results = \
        search_local_knowledge(
            message
        )


    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    web_results = \
        collect_web_information(
            message
        )


    # --------------------------------------------------------
    # SAVE SOURCES
    # --------------------------------------------------------

    save_web_sources(

        data.session_id,

        message,

        web_results

    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    answer = generate_response(

        message,

        memory,

        local_results,

        web_results

    )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    assistant_message = \
        save_message(

            data.session_id,

            "assistant",

            answer

        )


    # --------------------------------------------------------
    # TRAINING LOG
    # --------------------------------------------------------

    category = None

    if local_results:

        category = \
            local_results[0][1].get(
                "category"
            )


    save_training_log(

        message,

        answer,

        category or "web",

        "web-search"

    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = []

    for result in web_results:

        sources.append({

            "title":
                result.get(
                    "title",
                    ""
                ),

            "url":
                result.get(
                    "url",
                    ""
                )

        })


    print(
        "FINAL WEB SOURCES:",
        len(sources)
    )

    print(
        "=" * 60
    )


    return {

        "answer":
            answer,

        "sources":
            sources,

        "knowledge_found":
            bool(
                local_results
            ),

        "web_found":
            bool(
                web_results
            ),

        "memory_used":
            len(
                memory
            ),

        "message_id":
            (

                assistant_message.get(
                    "id"
                )

                if assistant_message

                else None

            )

    }


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.post(
    "/api/admin/login"
)
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


    token = create_admin_token()


    return {

        "success":
            True,

        "token":
            token

    }


# ============================================================
# ADMIN STATS
# ============================================================

@app.get(
    "/api/admin/stats"
)
async def admin_stats(
    request: Request
):

    check_admin(
        request
    )


    return {

        "brain_ready":
            brain.ready,

        "knowledge":
            len(
                knowledge_cache
            ),

        "vocabulary":
            len(
                brain.vocabulary
            ),

        "categories":
            len(
                brain.categories
            )

    }


# ============================================================
# ADMIN KNOWLEDGE GET
# ============================================================

@app.get(
    "/api/admin/knowledge"
)
async def admin_knowledge(
    request: Request
):

    check_admin(
        request
    )

    return knowledge_cache


# ============================================================
# ADMIN KNOWLEDGE CREATE
# ============================================================

@app.post(
    "/api/admin/knowledge"
)
async def admin_add_knowledge(

    request: Request,

    data: KnowledgeCreate

):

    check_admin(
        request
    )


    title = data.title.strip()


    category = normalize(
        data.category
    )


    question = data.question.strip()


    answer = data.answer.strip()


    tags = [

        x.strip()

        for x in data.tags

        if x.strip()

    ]


    if not title:

        raise HTTPException(

            400,

            "Название обязательно."

        )


    if not category:

        raise HTTPException(

            400,

            "Категория обязательна."

        )


    if not question:

        raise HTTPException(

            400,

            "Вопрос обязателен."

        )


    if not answer:

        raise HTTPException(

            400,

            "Ответ обязателен."

        )


    item = {

        "title":
            title,

        "category":
            category,

        "question":
            question,

        "answer":
            answer,

        "tags":
            tags,

        "approved":
            True

    }


    saved = []


    if (
        SUPABASE_URL
        and
        SUPABASE_SECRET_KEY
    ):

        saved = supabase_request(

            "POST",

            "knowledge",

            item

        )


    if saved:

        knowledge_cache.append(
            saved[0]
        )

    else:

        item["id"] = \
            stable_hash(

                title
                + question
                + answer

            )

        knowledge_cache.append(
            item
        )


    result = brain.train(
        knowledge_cache
    )


    save_training_log(

        question,

        answer,

        category,

        "admin"

    )


    return {

        "success":
            True,

        "training":
            result,

        "knowledge":
            len(
                knowledge_cache
            )

    }


# ============================================================
# ADMIN DELETE KNOWLEDGE
# ============================================================

@app.delete(
    "/api/admin/knowledge/{knowledge_id}"
)
async def admin_delete_knowledge(

    knowledge_id: str,

    request: Request

):

    check_admin(
        request
    )


    global knowledge_cache


    knowledge_cache = [

        item

        for item in knowledge_cache

        if str(

            item.get(
                "id"
            )

        )

        != str(

            knowledge_id

        )

    ]


    if (

        SUPABASE_URL

        and

        SUPABASE_SECRET_KEY

    ):

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

        "success":
            True

    }


# ============================================================
# ADMIN WEB SOURCES
# ============================================================

@app.get(
    "/api/admin/web-sources"
)
async def admin_web_sources(
    request: Request
):

    check_admin(
        request
    )


    if (

        not SUPABASE_URL

        or

        not SUPABASE_SECRET_KEY

    ):

        return []


    rows = supabase_request(

        "GET",

        "web_sources",

        params={

            "select":
                (
                    "id,query,title,url,"
                    "snippet,source,created_at"
                ),

            "order":
                "created_at.desc",

            "limit":
                "50"

        }

    )


    return rows


# ============================================================
# FEEDBACK
# ============================================================

@app.post(
    "/api/feedback"
)
async def feedback(
    data: FeedbackRequest
):

    if (

        data.rating < 1

        or

        data.rating > 5

    ):

        raise HTTPException(

            400,

            "Оценка должна быть от 1 до 5."

        )


    if (

        SUPABASE_URL

        and

        SUPABASE_SECRET_KEY

    ):

        supabase_request(

            "POST",

            "ai_feedback",

            {

                "session_id":
                    data.session_id,

                "message_id":
                    data.message_id,

                "rating":
                    data.rating,

                "comment":
                    data.comment or ""

            }

        )


    return {

        "success":
            True

    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event(
    "startup"
)
async def startup():

    load_knowledge()


    print("")

    print(
        "=" * 60
    )

    print(
        "                  ASCEND AI"
    )

    print(
        "=" * 60
    )

    print(
        "Knowledge:",
        len(
            knowledge_cache
        )
    )

    print(
        "Neural brain:",
        brain.ready
    )

    print(
        "Yandex HTML search:",
        True
    )

    print(
        "Yandex API:",
        False
    )

    print(
        "Startpage fallback:",
        True
    )

    print(
        "DuckDuckGo fallback:",
        True
    )

    print(
        "=" * 60
    )

    print("")
