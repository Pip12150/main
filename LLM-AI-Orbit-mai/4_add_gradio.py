'''
그라디오 추가하고 오른쪽에서 로그 볼 수 있게 수정
'''
import json
import os
import math
from pathlib import Path
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# 환경 설정
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDED_JSON = "guide_step5_embedded_chunk.json"
QA_CACHE_JSON = "qa_cache.json"

TOP_K = 5
QA_SIM_THRESHOLD = 0.88


# =========================
# 유틸
# =========================
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def embed_query(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return res.data[0].embedding


# =========================
# RAG
# =========================
def search_chunks(query: str):
    chunks = json.loads(Path(EMBEDDED_JSON).read_text(encoding="utf-8"))
    query_emb = embed_query(query)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_emb, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:TOP_K]


def build_context(chunks):
    context = []
    for i, (_, chunk) in enumerate(chunks, 1):
        context.append(
            f"{i}.\n"
            f"제목: {chunk['title']}\n"
            f"유형: {chunk['type']}\n"
            f"내용:\n{chunk['content']}\n"
        )
    return "\n".join(context)


def generate_answer(query: str, context: str):
    messages = [
        {
            "role": "system",
            "content": (
                "너는 Gradio 기반 웹 UI 사용법을 안내하는 도우미다. "
                "반드시 제공된 문서 내용만 사용해서 답변해라. "
                "문서에 없는 내용은 추측하지 말고 '문서에 없는 내용입니다'라고 답해라."
            )
        },
        {
            "role": "user",
            "content": f"[문서 컨텍스트]\n{context}\n\n[질문]\n{query}"
        }
    ]

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0
    )

    return res.choices[0].message.content


# =========================
# QA 캐시
# =========================
def load_qa_cache():
    if not os.path.exists(QA_CACHE_JSON):
        return []
    return json.loads(Path(QA_CACHE_JSON).read_text(encoding="utf-8"))


def save_qa_cache(cache):
    Path(QA_CACHE_JSON).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def search_qa_cache(query: str):
    cache = load_qa_cache()
    if not cache:
        return None, 0.0

    query_emb = embed_query(query)

    best_score = 0.0
    best_item = None

    for item in cache:
        score = cosine_similarity(query_emb, item["question_embedding"])
        if score > best_score:
            best_score = score
            best_item = item

    if best_score >= QA_SIM_THRESHOLD:
        return best_item, best_score

    return None, best_score


def add_qa_cache(query: str, answer: str):
    cache = load_qa_cache()
    cache.append({
        "question": query,
        "question_embedding": embed_query(query),
        "answer": answer
    })
    save_qa_cache(cache)


# =========================
# Trace 생성 (핵심)
# =========================
def make_trace(
    question,
    qa_hit,
    qa_score,
    rag_titles,
    llm_called
):
    return {
        "question": question,
        "qa_cache": {
            "hit": qa_hit,
            "score": round(qa_score, 4)
        },
        "rag": {
            "used": not qa_hit,
            "top_chunks": rag_titles
        },
        "llm": {
            "called": llm_called
        }
    }


# =========================
# Gradio 처리 함수
# =========================
def chat(query, chat_history, trace_history):
    # 1️⃣ QA 캐시 확인
    cached, score = search_qa_cache(query)
    if cached:
        chat_history.append((query, cached["answer"]))

        trace_history.append(
            make_trace(
                question=query,
                qa_hit=True,
                qa_score=score,
                rag_titles=[],
                llm_called=False
            )
        )

        return chat_history, trace_history, ""

    # 2️⃣ RAG + LLM
    chunks = search_chunks(query)
    context = build_context(chunks)
    answer = generate_answer(query, context)

    add_qa_cache(query, answer)

    chat_history.append((query, answer))

    trace_history.append(
        make_trace(
            question=query,
            qa_hit=False,
            qa_score=score,
            rag_titles=[c[1]["title"] for c in chunks],
            llm_called=True
        )
    )

    return chat_history, trace_history, ""


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="RAG + QA Cache Test Bot") as demo:
    gr.Markdown("## 🧪 RAG + QA Cache 테스트용 챗봇")

    chat_state = gr.State([])
    trace_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Chat")

            query_box = gr.Textbox(
                label="질문 입력",
                placeholder="질문을 입력하세요",
            )

            send_btn = gr.Button("전송")

        with gr.Column(scale=2):
            trace_view = gr.JSON(
                label="🔍 Debug Trace (모든 질문 누적)"
            )

    send_btn.click(
        fn=chat,
        inputs=[query_box, chat_state, trace_state],
        outputs=[chatbot, trace_view, query_box]
    )

    query_box.submit(
        fn=chat,
        inputs=[query_box, chat_state, trace_state],
        outputs=[chatbot, trace_view, query_box]
    )

demo.launch()
