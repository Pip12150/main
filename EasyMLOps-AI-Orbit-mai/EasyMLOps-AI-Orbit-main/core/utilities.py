from typing import List, Optional, Tuple
import os
import re
import cv2
import shutil
import numpy as np
import gradio as gr

# -----------------------------
# Utilities for Tab 1
# -----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def _is_image(p: str) -> bool:
    """
       [유틸] 주어진 경로가 이미지 파일인지 확장자로 판별한다.

       Args:
           p: 파일 경로

       Returns:
           True면 IMG_EXTS에 속하는 이미지 확장자, 아니면 False
    """
    return bool(p) and p.lower().endswith(IMG_EXTS)

def _stem(path: str) -> str:
    """
       [유틸] 주어진 경로가 이미지 파일인지 확장자로 판별한다.

       Args:
           p: 파일 경로

       Returns:
           True면 IMG_EXTS에 속하는 이미지 확장자, 아니면 False
   """
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def _safe_dir_from_selection(selection: Optional[str]) -> str:
    """
        [유틸] 파일 경로에서 확장자를 제거한 stem(파일명)을 반환한다.

        Args:
            path: 파일 경로

        Returns:
            확장자 제외 파일명 (예: "/a/b/c.jpg" -> "c")
    """
    if not selection:
        return ""
    selection = selection.strip()
    if os.path.isdir(selection):
        return selection
    if os.path.isfile(selection):
        return os.path.dirname(selection)
    # 존재하지 않으면 그대로 반환(사용자가 직접 입력한 경우 대비)
    return selection

def _list_images_in_dir(dir_path: str) -> List[str]:
    """
        [유틸] 특정 디렉토리(1-depth) 내 이미지 파일 목록을 수집한다.

        - 재귀 탐색이 아니라 현재 폴더에 있는 파일만 수집
        - IMG_EXTS 기준 확장자 필터 적용
        - full path 리스트를 반환

        Args:
            dir_path: 탐색할 디렉토리

        Returns:
            이미지 파일 전체 경로 리스트(sorted)
    """
    if not dir_path or not os.path.isdir(dir_path):
        return []
    files = []
    for f in os.listdir(dir_path):
        p = os.path.join(dir_path, f)
        if os.path.isfile(p) and _is_image(p):
            files.append(p)
    return sorted(files)

def _read_txt_lines(txt_path: str) -> List[str]:
    """
        [유틸] txt 파일을 줄 단위로 읽어서 공백/빈줄을 제거한 리스트로 반환한다.

        Args:
            txt_path: 라벨(txt) 경로

        Returns:
            비어있지 않은 라인들(list[str])
            - 파일이 없으면 []
    """
    if not txt_path or not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _draw_outlines_only(
    image_path: str,
    txt_path: str,
    color_bgr: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> Optional[np.ndarray]:
    """
        [시각화] YOLO seg txt를 읽어 원본 이미지 위에 '윤곽선(polylines)만' 그려 반환한다.

        기대 포맷(주석 기준):
          - YOLO seg 포맷: class x1 y1 x2 y2 ... (좌표는 0~1 정규화)
          - bbox 포맷처럼 좌표 4개만 있는 라인은 현재 로직에서 스킵한다
            (원하면 bbox outline 추가 가능)

        반환 형태:
          - gr.Image(type="numpy")에 넣기 쉬운 RGB ndarray 반환

        처리 흐름:
          1) image_path를 cv2.imread로 읽어 BGR로 로드
          2) txt를 읽어 라인별로 파싱
          3) polygon 좌표들을 (w,h)로 scale하여 정수 픽셀 좌표로 변환
          4) cv2.polylines로 외곽선만 그리기
          5) 최종 결과를 RGB로 변환해서 반환

        Args:
            image_path: 원본 이미지 경로
            txt_path: YOLO seg 라벨(txt) 경로
            color_bgr: 선 색상(BGR)
            thickness: 선 두께

        Returns:
            RGB ndarray (성공)
            None (이미지 로드 실패)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    lines = _read_txt_lines(txt_path)
    if not lines:
        # txt가 없으면 원본만 반환
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    for line in lines:
        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue

        # 첫 값은 class로 가정, 나머지는 좌표
        try:
            coords = list(map(float, parts[2:]))
        except ValueError:
            continue

        # polygon은 (x,y)쌍이므로 짝수개 필요, 최소 3점(6개) 이상
        if len(coords) < 6 or (len(coords) % 2) != 0:
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = int(round(coords[i] * w))
            y = int(round(coords[i + 1] * h))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)

        # 윤곽선만
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color_bgr, thickness=thickness)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# Utilities for Tab 3
# -----------------------------
class FolderNavigator:
    """
        [안전한 폴더 탐색기 로직 + UI 캡슐화]

        목적:
          - root_dir 바깥으로 빠져나가지 못하도록 path 안전장치를 둔 상태로
            폴더 이동(enter/parent/refresh) 로직을 제공한다.
          - Gradio UI(현재 경로 표시 + 하위 폴더 dropdown + 버튼)를 함께 생성한다.

        핵심 제약:
          - 항상 root_dir 하위 경로만 허용
          - 현재 경로/이동 경로가 root_dir 밖이면 root_dir로 강제
    """

    def __init__(self, root_dir: str, default_path: str = ""):
        """
            Args:
                root_dir: 탐색 가능한 최상위 루트 디렉토리(이 밖은 접근 불가)
                default_path: 초기 진입 경로(없으면 root_dir)
        """
        self.root_dir = os.path.abspath(root_dir)

        init = os.path.abspath(default_path) if default_path else self.root_dir
        self.current_dir = self._normalize_path(init)

    # ------------------------------------------------------------------
    # Path safety / normalization
    # ------------------------------------------------------------------
    def _is_safe_subpath(self, path: str) -> bool:
        """
            [보안] path가 root_dir 하위인지 확인한다.

            Returns:
                True: root_dir 하위
                False: root_dir 바깥(또는 commonpath 계산 실패)
        """
        try:
            return os.path.commonpath([self.root_dir, path]) == self.root_dir
        except ValueError:
            return False

    def _normalize_path(self, path: str) -> str:
        """
            [경로 정규화]
            - path를 절대경로로 변환
            - root_dir 하위가 아니면 root_dir로 강제
            - 디렉토리가 아니면 root_dir로 강제

            Args:
                path: 후보 경로

            Returns:
                안전하게 정규화된 디렉토리 경로
        """
        path = os.path.abspath(path)
        if not self._is_safe_subpath(path):
            return self.root_dir
        if not os.path.isdir(path):
            return self.root_dir
        return path

    # ------------------------------------------------------------------
    # Core navigation logic
    # ------------------------------------------------------------------
    def list_subdirs(self, path: str = "") -> List[str]:
        """
            [하위 폴더 목록]
            주어진 path(없으면 현재 경로) 하위의 '폴더 이름' 목록을 반환한다.

            Args:
                path: 조회할 경로(옵션)

            Returns:
                하위 폴더명 리스트(sorted)
        """
        cur = self._normalize_path(path or self.current_dir)

        names = []
        for n in os.listdir(cur):
            p = os.path.join(cur, n)
            if os.path.isdir(p):
                names.append(n)
        return sorted(names)

    def go_parent(self) -> Tuple[str, List[str]]:
        """
            [상위 폴더로 이동]
            현재 경로의 부모로 이동하되,
            root_dir 밖으로 나가려 하면 root_dir로 강제된다.

            Returns:
                (현재경로, 하위폴더목록)
        """
        parent = os.path.dirname(self.current_dir)
        self.current_dir = self._normalize_path(parent)
        return self.current_dir, self.list_subdirs(self.current_dir)

    def refresh(self) -> Tuple[str, List[str]]:
        """
            [새로고침]
            현재 경로는 유지한 채로,
            하위 폴더 목록을 다시 읽어온다.

            Returns:
                (현재경로, 하위폴더목록)
        """
        self.current_dir = self._normalize_path(self.current_dir)
        return self.current_dir, self.list_subdirs(self.current_dir)

    def enter_subdir(self, subdir_name: str) -> Tuple[str, List[str]]:
        """
            [하위 폴더로 진입]
            현재 경로 하위의 subdir_name으로 이동한다.

            Args:
                subdir_name: 이동할 하위 폴더명

            Returns:
                (현재경로, 하위폴더목록)
        """
        if not subdir_name:
            return self.current_dir, self.list_subdirs(self.current_dir)

        nxt = os.path.join(self.current_dir, subdir_name)
        self.current_dir = self._normalize_path(nxt)
        return self.current_dir, self.list_subdirs(self.current_dir)

    # ------------------------------------------------------------------
    # UI builder
    # ------------------------------------------------------------------
    def build_ui(self, label: str):
        """
            [Gradio UI 생성]
            FolderNavigator에 대응하는 UI를 만든다.

            UI 구성:
              - 현재 경로 표시 Textbox(interactive=False)
              - "상위 폴더", "새로고침" 버튼
              - 하위 폴더 Dropdown

            이벤트:
              - 상위 폴더 클릭 -> _ui_go_parent()
              - 새로고침 클릭 -> _ui_refresh()
              - 드롭다운 변경 -> _ui_enter()

            Returns:
              (current_path_tb, subdir_dd, btn_up, btn_refresh)
        """

        cur_path = gr.Textbox(
            label=f"{label} - 현재 경로",
            value=self.current_dir,
            interactive=False,
        )

        with gr.Row():
            btn_up = gr.Button("상위 폴더")
            btn_refresh = gr.Button("새로고침")

        subdir_dd = gr.Dropdown(
            label=f"{label} - 하위 폴더",
            choices=self.list_subdirs(self.current_dir),
            value=None,
            interactive=True,
        )

        # ----------------------
        # event bindings
        # ----------------------
        btn_up.click(
            fn=lambda _: self._ui_go_parent(),
            inputs=[cur_path],
            outputs=[cur_path, subdir_dd],
        )

        btn_refresh.click(
            fn=lambda _: self._ui_refresh(),
            inputs=[cur_path],
            outputs=[cur_path, subdir_dd],
        )

        subdir_dd.change(
            fn=lambda _, name: self._ui_enter(name),
            inputs=[cur_path, subdir_dd],
            outputs=[cur_path, subdir_dd],
        )

        return cur_path, subdir_dd, btn_up, btn_refresh

    # ------------------------------------------------------------------
    # UI adapters (Gradio update 형태로 변환)
    # ------------------------------------------------------------------
    def _ui_go_parent(self):
        """
            [UI 어댑터] go_parent() 실행 결과를
            (현재경로 문자열, Dropdown 업데이트) 형태로 변환해 반환한다.
        """
        cur, choices = self.go_parent()
        return cur, gr.update(choices=choices, value=None)

    def _ui_refresh(self):
        """
            [UI 어댑터] refresh() 실행 결과를
            (현재경로 문자열, Dropdown 업데이트) 형태로 변환해 반환한다.
        """
        cur, choices = self.refresh()
        return cur, gr.update(choices=choices, value=None)

    def _ui_enter(self, name: str):
        """
            [UI 어댑터] enter_subdir(name) 실행 결과를
            (현재경로 문자열, Dropdown 업데이트) 형태로 변환해 반환한다.
        """
        cur, choices = self.enter_subdir(name)
        return cur, gr.update(choices=choices, value=None)

def build_folder_picker(label: str, root_dir: str, default_path: str = ""):
    """
        [팩토리 함수]
        FolderNavigator를 생성하고 UI를 구성해 반환한다.
        app.py에서 '한 줄로 폴더 선택 UI'를 붙이기 쉽게 만든 래퍼.

        Args:
            label: UI 라벨 prefix
            root_dir: 탐색 루트
            default_path: 초기 경로

        Returns:
            FolderNavigator.build_ui() 반환값
    """
    navigator = FolderNavigator(root_dir, default_path)
    return navigator.build_ui(label)

def save_uploaded_file(file_obj, target_dir: str) -> str:
    """
        [업로드 파일 서버 저장]
        gr.File로 업로드된 파일(임시 경로)을 target_dir로 복사하여 서버에 저장한다.

        정책:
          - 업로드된 원본 파일명 그대로 사용
          - 동일 파일명이 이미 존재하면 덮어쓰지 않고 기존 파일 경로 반환

        입력:
            file_obj: gr.File에서 넘어오는 값
                      - dict 형태일 수도 있고(tempfile info)
                      - file-like 객체일 수도 있음(.name 속성)
            target_dir: 저장할 서버 디렉토리

        출력:
            dst_path: 서버에 저장된(또는 이미 존재하는) 파일의 절대 경로
                     - 실패 시 "" 반환

        주의:
          - gradio 버전에 따라 file_obj 구조가 달라질 수 있어 dict/.name 모두 대응
          - target_dir는 자동 생성(os.makedirs)
    """
    if file_obj is None:
        return ""

    os.makedirs(target_dir, exist_ok=True)

    # gradio File: dict or tempfile-like object
    if isinstance(file_obj, dict):
        src_path = file_obj.get("name", "")
    else:
        src_path = getattr(file_obj, "name", "")

    if not src_path or not os.path.exists(src_path):
        return ""

    # ✅ 업로드된 파일명 그대로 사용
    filename = os.path.basename(src_path)
    dst_path = os.path.join(target_dir, filename)

    # ✅ 이미 존재하면 복사하지 않음
    if os.path.exists(dst_path):
        return dst_path

    # ✅ 존재하지 않을 때만 저장
    shutil.copy2(src_path, dst_path)
    return dst_path
