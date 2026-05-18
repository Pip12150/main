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


def main():
    while True:
        query = input("\n💬 질문 입력 (exit 종료): ").strip()
        if query.lower() == "exit":
            break

        chunks = search_chunks(query)
        context = build_context(chunks)
        answer = generate_answer(query, context)

        print("\n🤖 답변\n" + "-" * 40)
        print(answer)


if __name__ == "__main__":
    main()
