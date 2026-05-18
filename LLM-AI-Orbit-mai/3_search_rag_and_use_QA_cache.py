'''
반복된 질문에 대해서 비용 절감을 위해 QA 캐시를 사용하는 방식
RAG 캐시는 검색 결과에 대해서 캐싱하기 때문에 재사용성이 낮음
반면, QA 캐시는 최종답변에 대해서 캐싱하기 때문에 재사용성이 높고 비용절감에도 효과적.
'''
import json
import os
import math
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDED_JSON = "guide_step5_embedded_chunk.json"
TOP_K = 5

QA_CACHE_JSON = "qa_cache.json"
QA_SIM_THRESHOLD = 0.88


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def embed_query(query: str):
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return res.data[0].embedding


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

'''
    캐시 부분 추가
'''

def load_qa_cache():
    # 캐시 로드
    if not os.path.exists(QA_CACHE_JSON):
        return []
    return json.loads(Path(QA_CACHE_JSON).read_text(encoding="utf-8"))

def save_qa_cache(cache):
    #  캐시 저장
    Path(QA_CACHE_JSON).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def search_qa_cache(query: str):
    # QA 캐시 검색
    cache = load_qa_cache()
    if not cache:
        return None

    query_emb = embed_query(query)

    best_score = 0
    best_item = None

    for item in cache:
        score = cosine_similarity(query_emb, item["question_embedding"])
        if score > best_score:
            best_score = score
            best_item = item

    print("best score : ", best_score)
    if best_score >= QA_SIM_THRESHOLD:
        return best_item

    return None

def add_qa_cache(query: str, answer: str):
    cache = load_qa_cache()
    cache.append({
        "question": query,
        "question_embedding": embed_query(query),
        "answer": answer
    })
    save_qa_cache(cache)


def main():
    while True:
        query = input("\n💬 질문 입력 (exit 종료): ").strip()
        if query.lower() == "exit":
            break

        # 1. QA 캐시 먼저 확인
        cached = search_qa_cache(query)
        if cached:
            print("\n🤖 답변 (QA 캐시 HIT)")
            print("-" * 40)
            print(cached["answer"])
            continue

        # 2. 캐시 MISS → 기존 RAG + LLM
        chunks = search_chunks(query)
        context = build_context(chunks)
        answer = generate_answer(query, context)

        # 3. QA 캐시에 저장
        add_qa_cache(query, answer)

        print("\n🤖 답변 (LLM 생성)")
        print("-" * 40)
        print(answer)


if __name__ == "__main__":
    main()
