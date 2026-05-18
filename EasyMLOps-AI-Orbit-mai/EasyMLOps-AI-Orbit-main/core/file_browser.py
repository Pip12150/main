# core/file_browser.py
import os
from typing import List, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MODEL_EXTS = (".pt",)

def list_dir(path: str) -> Tuple[str, List[str], List[str]]:
    """
        [디렉토리 탐색]
        주어진 경로의 하위 항목을 조회하여
        '디렉토리 목록'과 '파일 목록'을 분리해서 반환한다.

        입력:
            path: 탐색할 디렉토리 경로
                  - None / 빈 문자열이면 "/"로 처리
                  - 존재하지 않거나 디렉토리가 아니면 "/"로 fallback

        처리:
            - os.listdir로 1-depth 항목만 조회
            - 디렉토리 / 파일을 분리
            - 이름 기준 정렬(sorted)

        출력:
            current_path: 실제로 사용된 절대 경로
            dirs: 하위 디렉토리 이름 리스트
            files: 하위 파일 이름 리스트

        활용 예:
            - Gradio 파일 브라우저 UI
            - 좌측 디렉토리 트리 / 우측 파일 리스트 구성
    """
    path = (path or "").strip()
    if not path:
        path = "/"
    path = os.path.abspath(path)

    if not os.path.isdir(path):
        # fallback
        path = "/"

    dirs, files = [], []
    try:
        for name in sorted(os.listdir(path)):
            p = os.path.join(path, name)
            if os.path.isdir(p):
                dirs.append(name)
            elif os.path.isfile(p):
                files.append(name)
    except Exception:
        pass

    return path, dirs, files

def join_path(cur: str, name: str) -> str:
    """
        [경로 이동]
        현재 경로(cur)에 하위 항목(name)을 결합하여
        절대 경로로 반환한다.

        입력:
            cur: 현재 디렉토리 경로
            name: 이동할 하위 디렉토리 또는 파일명

        출력:
            결합된 절대 경로

        활용 예:
            - 디렉토리 클릭 시 하위로 이동
            - 파일 선택 시 전체 경로 생성
    """
    return os.path.abspath(os.path.join(cur, name))

def parent_dir(cur: str) -> str:
    """
        [상위 디렉토리 이동]
        현재 경로에서 한 단계 상위 디렉토리의
        절대 경로를 반환한다.

        입력:
            cur: 현재 디렉토리 경로

        출력:
            상위 디렉토리 절대 경로

        활용 예:
            - '..' 버튼
            - 파일 브라우저에서 뒤로 가기
    """
    return os.path.abspath(os.path.join(cur, ".."))

def filter_files(files: List[str], exts: Tuple[str, ...]) -> List[str]:
    """
        [파일 확장자 필터링]
        파일명 리스트에서 특정 확장자만 필터링한다.

        입력:
            files: 파일명 리스트 (basename)
            exts: 허용 확장자 튜플 (예: IMAGE_EXTS, MODEL_EXTS)

        출력:
            확장자 조건을 만족하는 파일명 리스트

        활용 예:
            - 이미지 파일만 선택 가능하도록 제한
            - 모델(.pt) 파일만 표시
    """
    return [f for f in files if f.lower().endswith(exts)]
