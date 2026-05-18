#!/usr/bin/env python3
import json
import os
import re
from openai import OpenAI
from PyPDF2 import PdfReader

from dotenv import load_dotenv
load_dotenv()

MODEL = "gpt-4o-mini"

client = OpenAI()


###########################################
# 1) PDF → 텍스트 추출
###########################################
def load_text(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() for page in reader.pages)
        return text
    else:
        return open(path, "r", encoding="utf-8").read()


###########################################
# 2) 문서를 청크로 나누기
###########################################
def chunk_text(text, max_chars=3000):
    chunks = []
    buf = []

    for line in text.split("\n"):
        buf.append(line)
        if sum(len(x) for x in buf) > max_chars:
            chunks.append("\n".join(buf))
            buf = []
    if buf:
        chunks.append("\n".join(buf))

    return chunks


###########################################
# 3) LLM으로 Fact Sheet 생성
###########################################
FACT_PROMPT = """
You are an AI that converts documentation into a structured Fact Sheet.

IMPORTANT:
- The document is written in Korean.
- ALL values MUST be written in Korean.
- Do NOT translate technical terms unless they are commonly used in Korean.
- keys must remain in English (machine-friendly identifiers).

RULES:
- Produce ONLY JSON (list of objects).
- Each fact object must contain:
  {
    "key": "unique_name",
    "value": "meaning of the fact (Korean)",
    "scope": ""
  }
- keys MUST be short, machine-friendly identifiers like:
  "image_viewer.purpose", "dataset.setup", "train.monitor.outputs"
- value MUST be a concise but complete explanation IN KOREAN.
- NEVER add hallucinated content.
- If unsure, skip that content.

Return JSON only.
"""


def llm_extract_facts(text_chunk):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FACT_PROMPT},
            {"role": "user", "content": text_chunk},
        ],
    )
    content = resp.choices[0].message.content.strip()

    # JSON 복구 (깨진 JSON 방지)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except:
        # JSON이 아니면 다시 시도
        content = re.search(r"\[.*\]", content, re.S)
        if content:
            try:
                return json.loads(content.group(0))
            except:
                return []
    return []


###########################################
# 4) 전체 문서를 Fact Sheet로 변환
###########################################
def convert_to_fact_sheet(path, out_path="fact_sheet.json"):
    print(f"📄 Loading document: {path}")
    text = load_text(path)
    chunks = chunk_text(text)

    all_facts = []
    print(f"✂️ Total chunks: {len(chunks)}")

    for i, ch in enumerate(chunks):
        print(f"➡️ Processing chunk {i + 1}/{len(chunks)}...")
        facts = llm_extract_facts(ch)
        all_facts.extend(facts)

    # 중복 key 제거
    unique = {}
    for f in all_facts:
        unique[f["key"]] = f

    all_facts = list(unique.values())

    print(f"✅ Total extracted facts: {len(all_facts)}")

    # 저장
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_facts, f, indent=2, ensure_ascii=False)

    print(f"📦 Saved Fact Sheet → {out_path}")
    return all_facts


###########################################
# main
###########################################
if __name__ == "__main__":
    input = "./Gradio MLOps Guide_260101.pdf"
    output = "fact_sheet_ko.json"

    convert_to_fact_sheet(input, output)