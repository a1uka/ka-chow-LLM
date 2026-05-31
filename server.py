import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI, APIConnectionError, APIError, APITimeoutError

# cfg
FAISS_INDEX_PATH = "faiss_roag_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

LLM_BASE_URL = "http://localhost:1234/v1"   # llm loc
LLM_MODEL = "local-model"
LLM_API_KEY = "not-needed"
# prompt
SYSTEM_PROMPT = """Ты — ИИ-ассистент для врачей акушеров-гинекологов, работающий строго на основе клинических рекомендаций РОАГ.
- Отвечай только на основании предоставленных фрагментов (контекста).
- Если в контексте нет ответа, честно скажи: "В предоставленных клинических рекомендациях ответ не найден".
- Не добавляй информацию из своих общих знаний.
- Отвечай на русском языке, максимально точно и структурированно.
- Указывай источники (названия файлов), которые упоминаются в метаданных.
"""

EMERGENCY_TRIGGERS = [
    "сильное кровотечение", "кровь идёт", "судороги", "потеря сознания",
    "резкая слабость", "острая боль в животе", "давление 160",
    "не чувствую ребёнка", "не шевелится", "отслойка плаценты",
    "преэклампсия", "эклампсия", "внезапная одышка",
    "кровь из влагалища", "слишком сильная боль", "высокая температура 40"
]

EMERGENCY_RESPONSE = (
    "❗ Описанные вами симптомы могут указывать на неотложное состояние. "
    "Пожалуйста, немедленно обратитесь в скорую помощь (тел. 112 или 103) "
    "или к вашему лечащему врачу. "
    "Данный ассистент не предназначен для диагностики и консультаций в экстренных ситуациях."
)

# global objects
vectorstore = None
embeddings = None

def load_resources():
    global vectorstore, embeddings
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError(f"Индекс не найден: {FAISS_INDEX_PATH}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(lifespan=lifespan)

# models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    emergency: bool = False
    status: str = "ok"    # "ok", "emergency", "error"

# emergency check
def is_emergency(text: str) -> bool:
    text_lower = text.lower()
    for trigger in EMERGENCY_TRIGGERS:
        if trigger in text_lower:
            return True
    return False

# llm call
def call_llm(messages, max_retries=2, timeout=15):
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=timeout)
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except APITimeoutError:
            last_exception = "Превышено время ожидания"
        except APIConnectionError:
            last_exception = "Сервер языковой модели недоступен"
        except APIError as e:
            last_exception = f"Ошибка API: {e}"
        except Exception as e:
            last_exception = f"Неизвестная ошибка: {e}"

        if attempt < max_retries:
            time.sleep(1)
    return f"❌ {last_exception}. Пожалуйста, попробуйте позже."

# rag query
def ask_assistant(user_query: str, vectorstore) -> str:
    docs = vectorstore.similarity_search(user_query, k=4)
    context = "\n\n".join(
        [f"Источник: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {user_query}\n\nОтвет:"}
    ]
    return call_llm(messages)

# endpoints
@app.get("/")
async def root():
    return {"status": "ok", "message": "Ассистент работает"}

@app.get("/health/llm")
async def health_llm():
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=5)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        return {"llm_status": "ok", "response": "pong"}
    except APIConnectionError:
        return {"llm_status": "unreachable"}
    except APITimeoutError:
        return {"llm_status": "timeout"}
    except Exception as e:
        return {"llm_status": "error", "detail": str(e)}

@app.post("/ask", response_model=QueryResponse)
async def ask_endpoint(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Пустой запрос")

    if is_emergency(query):
        return {"answer": EMERGENCY_RESPONSE, "status": "emergency", "emergency": True}

    answer = ask_assistant(query, vectorstore)
    if answer.startswith("❌"):
        return {"answer": answer, "status": "error", "emergency": False}
    return {"answer": answer, "status": "ok", "emergency": False}

# start
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="debug")