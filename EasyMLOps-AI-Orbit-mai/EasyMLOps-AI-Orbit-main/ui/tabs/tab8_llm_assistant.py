from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import requests


APP_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("LLM_DATA_DIR", APP_ROOT / "llm_data"))

CHUNKS_JSON = DATA_DIR / "guide_step5_embedded_chunk.json"
QA_CACHE_JSON = DATA_DIR / "qa_cache.json"
OVERLAY_KB_JSON = DATA_DIR / "overlay_kb.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")

TOP_K = 5
CACHE_SCORE_THRESHOLD = 0.72
OVERLAY_SCORE_THRESHOLD = 0.72

EDIT_MODE = "Edit selected answer"
CREATE_MODE = "Create new answer"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "with",
    "you",
}


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_json_file(path: Path) -> dict[str, str]:
    _save_json(path, [])
    return {"file": str(path), "cleared_at": datetime.now().isoformat()}


def _tokens(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9가-힣_]+", (text or "").lower())
    return [token for token in raw if len(token) > 1 and token not in STOPWORDS]


def _normalize_question(question: str) -> str:
    return " ".join(_tokens(question))


def _text_score(query: str, text: str) -> float:
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0

    text_terms = _tokens(text)
    if not text_terms:
        return 0.0

    text_set = set(text_terms)
    overlap = [term for term in set(query_terms) if term in text_set]
    if not overlap:
        return 0.0

    coverage = len(overlap) / max(len(set(query_terms)), 1)
    frequency = sum(min(text_terms.count(term), 3) for term in overlap) / (3 * len(overlap))
    return (coverage * 0.75) + (frequency * 0.25)


def _strip_runtime_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key not in {"question_embedding", "embedding"}
    }


def _search_chunks(query: str) -> list[tuple[float, dict[str, Any]]]:
    chunks = _load_json(CHUNKS_JSON, [])
    scored = []

    for chunk in chunks:
        title = str(chunk.get("title", ""))
        content = str(chunk.get("content", ""))
        score = (_text_score(query, title) * 2.0) + _text_score(query, content)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:TOP_K]


def _build_context(chunks: list[tuple[float, dict[str, Any]]]) -> str:
    if not chunks:
        return "No matching documentation chunks were found."

    return "\n".join(
        f"{idx}.\nTitle: {chunk.get('title', '')}\n"
        f"Type: {chunk.get('type', '')}\nContent:\n{chunk.get('content', '')}\n"
        for idx, (_, chunk) in enumerate(chunks, 1)
    )


def _ollama_chat(messages: list[dict[str, str]]) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )

    if response.status_code >= 400:
        raise RuntimeError(response.text)

    payload = response.json()
    return payload.get("message", {}).get("content", "").strip()


def _generate_answer(query: str, context: str) -> str:
    return _ollama_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for a Gradio-based MLOps UI. "
                    "Use only the provided documentation context when possible. "
                    "Answer in the same language as the user's question. "
                    "If the context does not contain the answer, say that it is not documented."
                ),
            },
            {"role": "user", "content": f"[Documentation]\n{context}\n\n[Question]\n{query}"},
        ]
    )


def _search_qa_cache(query: str) -> tuple[dict[str, Any] | None, float]:
    cache = _load_json(QA_CACHE_JSON, [])
    if not cache:
        return None, 0.0

    normalized = _normalize_question(query)
    best, best_score = None, 0.0
    for item in cache:
        cached_question = str(item.get("question", ""))
        if _normalize_question(cached_question) == normalized:
            return item, 1.0

        score = _text_score(query, cached_question)
        if score > best_score:
            best, best_score = item, score

    if best_score >= CACHE_SCORE_THRESHOLD:
        return best, best_score
    return None, best_score


def _add_qa_cache(query: str, answer: str) -> None:
    cache = _load_json(QA_CACHE_JSON, [])
    cache.append(
        {
            "created_at": datetime.now().isoformat(),
            "question": query,
            "answer": answer,
            "provider": "ollama",
            "model": OLLAMA_CHAT_MODEL,
        }
    )
    _save_json(QA_CACHE_JSON, cache)


def _purge_qa_cache(question: str) -> list[str]:
    cache = _load_json(QA_CACHE_JSON, [])
    if not cache:
        return []

    normalized = _normalize_question(question)
    new_cache, removed = [], []
    for item in cache:
        cached_question = str(item.get("question", ""))
        if _normalize_question(cached_question) == normalized or _text_score(question, cached_question) >= CACHE_SCORE_THRESHOLD:
            removed.append(cached_question)
        else:
            new_cache.append(item)

    _save_json(QA_CACHE_JSON, new_cache)
    return removed


def _search_overlay_kb(query: str) -> tuple[dict[str, Any] | None, float]:
    overlay = _load_json(OVERLAY_KB_JSON, [])
    if not overlay:
        return None, 0.0

    normalized = _normalize_question(query)
    best, best_score = None, 0.0
    for item in overlay:
        question = str(item.get("question", ""))
        if _normalize_question(question) == normalized:
            return item, 1.0

        score = _text_score(query, question)
        if score > best_score:
            best, best_score = item, score

    if best_score >= OVERLAY_SCORE_THRESHOLD:
        return best, best_score
    return None, best_score


def _make_trace(
    question: str,
    source: str,
    score: float,
    rag_titles: list[str],
) -> dict[str, Any]:
    return {
        "question": question,
        "provider": "ollama",
        "model": OLLAMA_CHAT_MODEL,
        "source": source,
        "score": round(score, 4),
        "rag_chunks": rag_titles,
    }


def _safe_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if "connection" in lowered or "refused" in lowered:
        return "Ollama is not reachable. Start Ollama and check http://127.0.0.1:11434."
    if "not found" in lowered or "pull model" in lowered or "does not exist" in lowered:
        return f"Ollama model {OLLAMA_CHAT_MODEL} is not installed. Run: ollama pull {OLLAMA_CHAT_MODEL}"
    if "timed out" in lowered or "timeout" in lowered:
        return f"Ollama model {OLLAMA_CHAT_MODEL} took too long to respond. Try again or use a smaller model."
    return message


def _chat(
    query: str,
    chat_history: list[dict[str, str]] | None,
    trace_history: list[dict[str, Any]] | None,
):
    chat_history = list(chat_history or [])
    trace_history = list(trace_history or [])
    query = (query or "").strip()

    if not query:
        return chat_history, trace_history, ""

    try:
        overlay_item, overlay_score = _search_overlay_kb(query)
        if overlay_item:
            answer = overlay_item.get("answer", "")
            trace = _make_trace(query, "overlay", overlay_score, [])
        else:
            cache_item, cache_score = _search_qa_cache(query)
            if cache_item:
                answer = cache_item.get("answer", "")
                trace = _make_trace(query, "qa_cache", cache_score, [])
            else:
                chunks = _search_chunks(query)
                answer = _generate_answer(query, _build_context(chunks))
                _add_qa_cache(query, answer)
                trace = _make_trace(query, "ollama_rag", 0.0, [chunk.get("title", "") for _, chunk in chunks])

        chat_history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ]
        )
        trace_history.append(trace)
    except Exception as exc:
        chat_history.extend(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"LLM error: {_safe_error_message(exc)}"},
            ]
        )
        trace_history.append(_make_trace(query, "error", 0.0, []))

    return chat_history, trace_history, ""


def _select_chat(evt: gr.SelectData, chat_history: list[dict[str, str]] | None):
    chat_history = list(chat_history or [])
    index = evt.index[0] if isinstance(evt.index, tuple) else evt.index

    if not isinstance(index, int) or index < 0 or index >= len(chat_history):
        return "", "", "", gr.update()

    item = chat_history[index]
    if item.get("role") == "assistant":
        question = chat_history[index - 1].get("content", "") if index > 0 else ""
        answer = item.get("content", "")
    elif item.get("role") == "user":
        question = item.get("content", "")
        if index + 1 >= len(chat_history) or chat_history[index + 1].get("role") != "assistant":
            return "", "", "", gr.update()
        answer = chat_history[index + 1].get("content", "")
    else:
        return "", "", "", gr.update()

    return question, answer, answer, gr.update(value=EDIT_MODE)


def _save_overlay(user: str, mode: str, question: str, edited_answer: str, original_answer: str):
    user = (user or "local-user").strip()
    question = (question or "").strip()
    edited_answer = (edited_answer or "").strip()

    if not question or not edited_answer:
        return {"status": "error", "message": "Question and answer are required."}

    try:
        overlay = _load_json(OVERLAY_KB_JSON, [])
        item = {
            "user": user,
            "created_at": datetime.now().isoformat(),
            "mode": mode,
            "question": question,
            "original_answer": None if mode == CREATE_MODE else original_answer,
            "answer": edited_answer,
            "provider": "ollama",
            "model": OLLAMA_CHAT_MODEL,
        }
        overlay.append(item)
        _save_json(OVERLAY_KB_JSON, overlay)
        removed = _purge_qa_cache(question)
        result = _strip_runtime_fields(item)
        result["purged_cache_questions"] = removed
        return result
    except Exception as exc:
        return {"status": "error", "message": _safe_error_message(exc)}


def _on_overlay_mode_change(mode: str):
    if mode == EDIT_MODE:
        return gr.update(label="Question"), gr.update(label="Edited answer")
    return gr.update(label="New question", value=""), gr.update(label="Answer", value="")


def build_llm_assistant_tab() -> None:
    with gr.Tab("8. LLM Assistant"):
        gr.Markdown(f"### MLOps Assistant - Ollama `{OLLAMA_CHAT_MODEL}`")

        chat_state = gr.State([])
        trace_state = gr.State([])
        original_answer_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=420)
                query_box = gr.Textbox(label="Question", placeholder="Ask about the MLOps guide...")
                send_btn = gr.Button("Send", variant="primary")

            with gr.Column(scale=2):
                trace_view = gr.JSON(label="Trace")

        gr.Markdown("### Overlay KB")
        with gr.Row():
            user_box = gr.Textbox(label="Editor", value="local-user")
            overlay_mode = gr.Radio([EDIT_MODE, CREATE_MODE], value=EDIT_MODE, label="Mode")

        edit_q = gr.Textbox(label="Question")
        edit_a = gr.Textbox(label="Edited answer", lines=6)

        with gr.Row():
            save_btn = gr.Button("Save overlay answer")
            clear_overlay_btn = gr.Button("Clear overlay KB")
            clear_cache_btn = gr.Button("Clear QA cache")

        overlay_view = gr.JSON(label="Overlay result")

        send_btn.click(_chat, [query_box, chat_state, trace_state], [chatbot, trace_view, query_box])
        query_box.submit(_chat, [query_box, chat_state, trace_state], [chatbot, trace_view, query_box])

        chatbot.select(
            _select_chat,
            [chat_state],
            [edit_q, edit_a, original_answer_state, overlay_mode],
        )

        save_btn.click(
            _save_overlay,
            [user_box, overlay_mode, edit_q, edit_a, original_answer_state],
            overlay_view,
        )
        clear_overlay_btn.click(lambda: _clear_json_file(OVERLAY_KB_JSON), outputs=overlay_view)
        clear_cache_btn.click(lambda: _clear_json_file(QA_CACHE_JSON), outputs=trace_view)
        overlay_mode.change(_on_overlay_mode_change, inputs=overlay_mode, outputs=[edit_q, edit_a])
