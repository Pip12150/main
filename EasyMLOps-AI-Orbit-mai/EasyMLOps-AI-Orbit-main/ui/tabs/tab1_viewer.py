# ui/tabs/tab1_viewer.py
from __future__ import annotations

import os
import gradio as gr
import cv2

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from core.config import EXPLORER_ROOT
from core.utilities import (
    _is_image, _stem, _safe_dir_from_selection,
    _list_images_in_dir, _draw_outlines_only
)

# -----------------------------
# State model
# -----------------------------
@dataclass
class SourceState:
    """
        1번 탭(Viewer)의 전체 상태를 담는 단일 State 모델.

        이 State 하나로:
        - 서버/로컬 모드 구분
        - 이미지/라벨(txt) 소스 위치
        - 현재 탐색 인덱스
        - 현재 선택된 이미지
        를 모두 관리한다.

        Gradio의 gr.State에는 dict 형태로 저장되며,
        실제 로직에서는 SourceState <-> dict 변환을 통해 사용한다.
    """
    mode: str  # "server" | "local"
    # 현재 소스 모드
    # "server": 서버 내 디렉토리 기반 탐색
    # "local" : 로컬 업로드 파일 기반 탐색

    # -----------------
    # Server mode fields
    # -----------------
    server_img_dir: str = ""        # 서버 원본 이미지 디렉토리
    server_txt_dir: str = ""        # 서버 추론 결과 txt 디렉토리
    server_images: List[str] = None # server_img_dir 내 이미지 전체 목록(정렬된 절대경로)

    # -----------------
    # Local mode fields
    # -----------------
    local_images: List[str] = None  # 업로드된 로컬 이미지 파일 경로들(Gradio cache)
    local_txts: List[str] = None    # 업로드된 로컬 txt 파일 경로들(Gradio cache)

    # -----------------
    # Navigation state
    # -----------------
    idx: int = 0                    # 현재 이미지 index (이전/다음 이동용)
    current_image_path: str = ""    # 현재 선택된 이미지의 실제 경로


def _state_to_dict(s: SourceState) -> Dict[str, Any]:
    """
        SourceState(dataclass)를
        Gradio gr.State에 저장 가능한 dict 형태로 변환한다.

        - gr.State는 dataclass를 직접 안전하게 저장하지 못하는 환경이 있어
          항상 dict로 직렬화해서 사용한다.
    """
    return {
        "mode": s.mode,
        "server_img_dir": s.server_img_dir,
        "server_txt_dir": s.server_txt_dir,
        "server_images": s.server_images or [],
        "local_images": s.local_images or [],
        "local_txts": s.local_txts or [],
        "idx": int(s.idx),
        "current_image_path": s.current_image_path or "",
    }

def _dict_to_state(d: Dict[str, Any]) -> SourceState:
    """
        gr.State에 저장된 dict를
        SourceState(dataclass)로 복원한다.

        모든 key에 대해 기본값을 안전하게 보장하여
        State 깨짐(None, 타입 오류)을 방지한다.
    """
    d = d or {}
    return SourceState(
        mode=d.get("mode", "server"),
        server_img_dir=d.get("server_img_dir", ""),
        server_txt_dir=d.get("server_txt_dir", ""),
        server_images=list(d.get("server_images", []) or []),
        local_images=list(d.get("local_images", []) or []),
        local_txts=list(d.get("local_txts", []) or []),
        idx=int(d.get("idx", 0) or 0),
        current_image_path=d.get("current_image_path", "") or "",
    )


# -----------------------------
# Core behaviors (탐색·매칭·렌더링 로직)
# -----------------------------
def _resolve_current_image(st: SourceState) -> Tuple[Optional[str], SourceState]:
    """
        현재 SourceState를 기준으로
        '지금 보여줘야 할 이미지'를 결정한다.

        역할:
        - mode(server/local)에 따라 이미지 목록 선택
        - idx를 목록 길이에 맞게 보정(modulo)
        - current_image_path를 갱신

        반환:
        - (현재 이미지 경로 또는 None, 갱신된 State)
    """
    # server mode
    if st.mode == "server":
        imgs = st.server_images or []
        if not imgs:
            st.current_image_path = ""
            return None, st
        st.idx = int(st.idx) % len(imgs)
        st.current_image_path = imgs[st.idx]
        return st.current_image_path, st

    # local mode
    imgs = st.local_images or []
    if not imgs:
        st.current_image_path = ""
        return None, st
    st.idx = int(st.idx) % len(imgs)
    st.current_image_path = imgs[st.idx]
    return st.current_image_path, st


def _find_txt_for_image(st: SourceState, image_path: str) -> str:
    """
        현재 이미지(image_path)에 대응하는 추론 결과 txt 경로를 찾는다.

        규칙:
        - server 모드:
            server_txt_dir/{image_stem}.txt
        - local 모드:
            업로드된 local_txts 중 basename이 동일한 파일

        반환:
        - 매칭된 txt 경로 (없으면 빈 문자열)
    """
    if not image_path:
        return ""

    stem = _stem(image_path)

    if st.mode == "server":
        if not st.server_txt_dir:
            return ""
        candidate = os.path.join(st.server_txt_dir, f"{stem}.txt")
        return candidate if os.path.exists(candidate) else ""

    # local: 업로드된 txt 파일 중 stem.txt 매칭
    txts = st.local_txts or []
    target_name = f"{stem}.txt"
    for p in txts:
        if os.path.basename(p) == target_name:
            return p
    return ""


def on_set_server(server_img_sel: str, server_txt_sel: str, prev_state: Dict[str, Any]):
    """
        [서버로 설정] 버튼 클릭 시 호출됨.

        역할:
        1) FileExplorer 선택 결과를 안전하게 '디렉토리 경로'로 정규화
        2) 서버 이미지 디렉토리 스캔 → 이미지 목록 생성
        3) mode를 'server'로 전환
        4) 첫 번째 이미지 로드 및 미리보기 출력

        반환:
        - 갱신된 State
        - 원본 이미지(numpy RGB)
        - 추론 이미지 초기화(None)
        - 현재 파일명
        - 상태 요약 문자열
    """
    st = _dict_to_state(prev_state)

    img_dir = _safe_dir_from_selection(server_img_sel)
    txt_dir = _safe_dir_from_selection(server_txt_sel)

    # ✅ 사용자 선택이 "파일"이었는지 감지해서 안내 추가
    img_note = ""
    if server_img_sel and os.path.isfile(server_img_sel):
        img_note = f"(파일 선택 감지 → 상위 폴더 사용: {os.path.dirname(server_img_sel)})"

    txt_note = ""
    if server_txt_sel and os.path.isfile(server_txt_sel):
        txt_note = f"(파일 선택 감지 → 상위 폴더 사용: {os.path.dirname(server_txt_sel)})"

    imgs = _list_images_in_dir(img_dir)

    st.mode = "server"
    st.server_img_dir = img_dir
    st.server_txt_dir = txt_dir
    st.server_images = imgs
    st.idx = 0

    cur, st = _resolve_current_image(st)
    preview = None
    if cur:
        # 원본 미리보기 (RGB numpy)
        bgr = cv2.imread(cur)
        preview = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    info = f"[SERVER] images={len(imgs)} | img_dir={img_dir} {img_note} | txt_dir={txt_dir} {txt_note}"
    filename = os.path.basename(cur) if cur else ""

    return _state_to_dict(st), preview, None, filename, info


def on_set_local(local_img_files, local_txt_files, prev_state: Dict[str, Any]):
    """
        [로컬로 설정] 버튼 클릭 시 호출됨.

        역할:
        1) 업로드된 로컬 이미지/텍스트 파일 목록을 서버 캐시 경로로 수집
        2) 이미지/텍스트 목록을 basename 기준으로 정렬
        3) mode를 'local'로 전환
        4) 첫 이미지 미리보기 출력

        주의:
        - 브라우저 로컬 경로가 아니라 Gradio cache 경로(.name)를 사용
    """
    st = _dict_to_state(prev_state)

    img_paths = []
    if local_img_files:
        for f in local_img_files:
            p = getattr(f, "name", None)
            if p and _is_image(p):
                img_paths.append(p)
    img_paths = sorted(img_paths, key=lambda p: os.path.basename(p))

    txt_paths = []
    if local_txt_files:
        for f in local_txt_files:
            p = getattr(f, "name", None)
            if p and p.lower().endswith(".txt"):
                txt_paths.append(p)
    txt_paths = sorted(txt_paths, key=lambda p: os.path.basename(p))

    st.mode = "local"
    st.local_images = img_paths
    st.local_txts = txt_paths
    st.idx = 0

    cur, st = _resolve_current_image(st)
    preview = None
    if cur:
        bgr = cv2.imread(cur)
        preview = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    info = f"[LOCAL] images={len(img_paths)} | txts={len(txt_paths)}"
    filename = os.path.basename(cur) if cur else ""

    return _state_to_dict(st), preview, None, filename, info


def on_prev(prev_state: Dict[str, Any]):
    """
        [이전] 버튼 클릭 시 호출됨.

        역할:
        - 현재 idx를 -1 이동
        - mode(server/local)에 따라 이미지 재선택
        - 원본 이미지 미리보기 갱신
    """
    st = _dict_to_state(prev_state)
    st.idx = int(st.idx) - 1
    cur, st = _resolve_current_image(st)

    preview = None
    if cur:
        bgr = cv2.imread(cur)
        preview = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    filename = os.path.basename(cur) if cur else ""
    return _state_to_dict(st), preview, None, filename


def on_next(prev_state: Dict[str, Any]):
    """
        [다음] 버튼 클릭 시 호출됨.

        역할:
        - 현재 idx를 +1 이동
        - mode(server/local)에 따라 이미지 재선택
        - 원본 이미지 미리보기 갱신
    """
    st = _dict_to_state(prev_state)
    st.idx = int(st.idx) + 1
    cur, st = _resolve_current_image(st)

    preview = None
    if cur:
        bgr = cv2.imread(cur)
        preview = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    filename = os.path.basename(cur) if cur else ""
    return _state_to_dict(st), preview, None, filename


def on_load_infer(prev_state: Dict[str, Any]):
    """
        [추론결과 불러오기] 버튼 클릭 시 호출됨.

        역할:
        1) 현재 이미지 기준으로 매칭되는 txt 탐색
        2) txt가 없으면 → 원본 이미지만 반환
        3) txt가 있으면 → 원본 위에 '윤곽선만' 그려서 반환

        반환:
        - 추론 결과 이미지(numpy RGB)
        - 사용된 txt 파일명(또는 없음 표시)
    """
    st = _dict_to_state(prev_state)
    cur, st = _resolve_current_image(st)
    if not cur:
        return None, ""

    txt_path = _find_txt_for_image(st, cur)
    if not txt_path:
        # txt 없으면 원본만
        bgr = cv2.imread(cur)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        return rgb, "(txt 없음)"

    out = _draw_outlines_only(cur, txt_path, color_bgr=(0, 255, 0), thickness=2)
    return out, os.path.basename(txt_path)

def _is_dir(p: str) -> bool:
    return bool(p) and os.path.isdir(p)

def _validate_server_selections(img_sel: str, txt_sel: str):
    """
    server_img_sel, server_txt_sel 둘 다 폴더인지 검사해서:
    - 안내문구 2개(이미지용, txt용)
    - 버튼 활성화 여부(interactive)
    를 결정한다.
    """
    # 이미지 선택 안내
    if not img_sel:
        img_msg = "⚠️ 원본 이미지 **폴더**를 선택해주세요."
    elif os.path.isdir(img_sel):
        img_msg = f"✅ 원본 이미지 폴더 선택됨: `{img_sel}`"
    else:
        img_msg = f"❌ 파일이 선택되었습니다. **폴더만** 선택하세요: `{img_sel}`"

    # txt 선택 안내
    if not txt_sel:
        txt_msg = "⚠️ 추론결과 txt **폴더**를 선택해주세요."
    elif os.path.isdir(txt_sel):
        txt_msg = f"✅ txt 폴더 선택됨: `{txt_sel}`"
    else:
        txt_msg = f"❌ 파일이 선택되었습니다. **폴더만** 선택하세요: `{txt_sel}`"

    # 둘 다 폴더면 버튼 활성화
    ok = _is_dir(img_sel) and _is_dir(txt_sel)

    return (
        gr.Button(interactive=ok),  # btn_set_server 업데이트
        img_msg,                    # server_img_hint
        txt_msg,                    # server_txt_hint
    )


# -----------------------------
# UI
# -----------------------------
def build_tab1_viewer():
    with gr.Tab("1. 이미지 뷰어"):
        # 상태
        state = gr.State(
            value=_state_to_dict(
                SourceState(mode="server", server_images=[], local_images=[], local_txts=[])
            )
        )

        with gr.Accordion("서버 설정", open=True):
            # ✅ 서버 선택 상태 안내(폴더/파일 선택 여부 표시)
            server_img_hint = gr.Markdown()
            server_txt_hint = gr.Markdown()

            with gr.Row():
                with gr.Column(scale=1):
                    server_img_sel = gr.FileExplorer(
                        label="1) 서버 원본 이미지 경로 탐색\n(폴더만 선택 가능 / 파일 선택 시 버튼이 비활성화됩니다)",
                        root_dir=EXPLORER_ROOT,
                        file_count="single",
                    )
                with gr.Column(scale=1):
                    server_txt_sel = gr.FileExplorer(
                        label="2) 서버 추론결과 txt 경로 탐색\n(폴더만 선택 가능 / 파일 선택 시 버튼이 비활성화됩니다)",
                        root_dir=EXPLORER_ROOT,
                        file_count="single",
                    )
                with gr.Column(scale=0):
                    # ✅ 처음엔 비활성화(둘 다 폴더 선택되면 활성화)
                    btn_set_server = gr.Button(
                        "3) 서버로 설정", variant="primary", interactive=False
                    )

        with gr.Accordion("로컬 설정", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    local_img_files = gr.File(
                        label="4) 로컬 원본 이미지 선택(여러개)",
                        file_count="multiple",
                        file_types=["image"],
                    )
                with gr.Column(scale=1):
                    local_txt_files = gr.File(
                        label="5) 로컬 추론결과 txt 선택(여러개)",
                        file_count="multiple",
                        file_types=[".txt"],
                    )
                with gr.Column(scale=0):
                    btn_set_local = gr.Button("6) 로컬로 설정", variant="primary")

        info_box = gr.Textbox(label="설정 상태", lines=8, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 7) 원본 이미지")
                orig_img = gr.Image(type="numpy", label="원본이미지")
                cur_name = gr.Textbox(label="현재 파일명", interactive=False)

                with gr.Row():
                    btn_prev = gr.Button("7-1) 이전")
                    btn_next = gr.Button("7-2) 다음")

            with gr.Column(scale=1):
                gr.Markdown("### 8) 추론 결과(윤곽선만)")
                infer_img = gr.Image(type="numpy", label="추론결과(윤곽선)")
                txt_name = gr.Textbox(label="사용한 txt", interactive=False)

                btn_load = gr.Button("8-1) 추론결과 불러오기", variant="primary")

        # -----------------
        # events
        # -----------------

        # ✅ 서버 폴더 선택 검증 → 버튼 활성/비활성 + 안내문구
        server_img_sel.change(
            fn=_validate_server_selections,
            inputs=[server_img_sel, server_txt_sel],
            outputs=[btn_set_server, server_img_hint, server_txt_hint],
        )
        server_txt_sel.change(
            fn=_validate_server_selections,
            inputs=[server_img_sel, server_txt_sel],
            outputs=[btn_set_server, server_img_hint, server_txt_hint],
        )

        # ✅ 서버로 설정
        btn_set_server.click(
            fn=on_set_server,
            inputs=[server_img_sel, server_txt_sel, state],
            outputs=[state, orig_img, infer_img, cur_name, info_box],
        )

        # ✅ 로컬로 설정
        btn_set_local.click(
            fn=on_set_local,
            inputs=[local_img_files, local_txt_files, state],
            outputs=[state, orig_img, infer_img, cur_name, info_box],
        )

        # ✅ 이전/다음
        btn_prev.click(
            fn=on_prev,
            inputs=[state],
            outputs=[state, orig_img, infer_img, cur_name],
        )
        btn_next.click(
            fn=on_next,
            inputs=[state],
            outputs=[state, orig_img, infer_img, cur_name],
        )

        # ✅ 추론결과 불러오기
        btn_load.click(
            fn=on_load_infer,
            inputs=[state],
            outputs=[infer_img, txt_name],
        )

