#!/usr/bin/env python3
import json
import os
import time
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

MODEL = "gpt-4o-mini"

client = OpenAI()

FACT_SHEET_PATH = "fact_sheet_ko.json"


# ---------------------------
# Load facts
# ---------------------------
def load_facts(path=FACT_SHEET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Fact sheet not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


FACTS = load_facts()

# ---------------------------
# Normalize user question
# ---------------------------
NORMALIZE_PROMPT = """
너는 사용자의 질문을 documentation fact key에 매칭 가능한 짧은 명사형 질의로 변환하는 검색 정규화기다.

규칙:
- 단 하나의 문구만 출력
- 설명하지 말고 JSON만 출력
- 구조:
{
  "normalized_query": ""
}
"""


def normalize_query(q: str):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": NORMALIZE_PROMPT},
            {"role": "user", "content": q}
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data["normalized_query"].strip()
    except:
        return q.strip()


# ---------------------------
# Simple similarity search (문자열 기반)
# ---------------------------
def find_best_fact(norm_q: str, facts):
    norm_q_low = norm_q.lower()

    best = None
    best_score = 0.0

    for f in facts:
        key = f.get("key", "")
        val = f.get("value", "")

        text = key.lower() + " " + val.lower()

        # 간단한 scoring
        score = (
                (1.0 if norm_q_low in key.lower() else 0.0) +
                (0.5 if norm_q_low in val.lower() else 0.0)
        )

        if score > best_score:
            best_score = score
            best = f

    return best, best_score


# ---------------------------
# Main query function
# ---------------------------
def query_fact_sheet(question: str):
    print("\n============================")
    print(f"❓ Question: {question}")

    # 1) Normalize
    t1 = time.perf_counter()
    norm = normalize_query(question)
    t2 = time.perf_counter()

    print(f"🧠 Normalized → '{norm}' ({(t2 - t1) * 1000:.2f}ms)")

    # 2) Search
    best, score = find_best_fact(norm, FACTS)

    if not best:
        return {
            "key": norm,
            "value": "UNKNOWN",
            "confidence": "unknown"
        }

    return {
        "key": best["key"],
        "value": best["value"],
        "scope": best.get("scope", ""),
        "confidence": "high" if score >= 1 else "low"
    }


# ---------------------------
# CLI execution
# ---------------------------
if __name__ == "__main__":
    while True:
        q = input("\n🔎 Ask a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break

        result = query_fact_sheet(q)
        print("\n📌 Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))