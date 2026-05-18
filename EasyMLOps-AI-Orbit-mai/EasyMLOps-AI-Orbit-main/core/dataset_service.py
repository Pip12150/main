import os
import shutil
import glob
from typing import Any, Dict, List, Tuple, Set
import pandas as pd
from core.config import UPLOAD_NEWDATASET_ROOT

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def _stem(path: str) -> str:
    """
        [유틸] 파일 경로에서 확장자를 제외한 basename(stem)만 반환한다.
        예) /a/b/photo_1.jpg -> photo_1
    """
    return os.path.splitext(os.path.basename(path))[0]

def _is_img(path: str) -> bool:
    """
        [유틸] 주어진 경로가 이미지 확장자(IMG_EXTS)에 해당하는지 판별한다.
    """
    return path.lower().endswith(IMG_EXTS)

def build_upload_cache(image_files, txt_files) -> Tuple[List[str], Dict]:
    """
        [업로드 캐시 구성]
        gr.File(file_count="multiple")로 받은 이미지/라벨 업로드 목록을
        stem(파일명-확장자) 기준으로 빠르게 매칭할 수 있도록 캐시(dict)를 만든다.

        입력:
          - image_files: 업로드된 이미지 파일 리스트 (gradio 파일 객체 or 경로 문자열)
          - txt_files  : 업로드된 txt 파일 리스트 (gradio 파일 객체 or 경로 문자열)

        처리:
          - imgs: stem -> image_path
          - txts: stem -> txt_path
          - fn2stem: checkbox에 표시되는 filename -> stem 매핑

        출력:
          - choices: 체크박스에 보여줄 "이미지 파일명" 리스트(정렬됨)
          - cache  : {"imgs":..., "txts":..., "fn2stem":...}
          - log    : 업로드 수/매칭 쌍 수 요약 로그 문자열

        주의:
          - stem 매칭은 "basename 기준"이므로, 같은 stem이 중복되면 마지막 값으로 덮어쓴다.
    """
    imgs = {}
    txts = {}

    if image_files:
        for f in image_files:
            p = f.name if hasattr(f, "name") else str(f)
            if _is_img(p):
                imgs[_stem(p)] = p

    if txt_files:
        for f in txt_files:
            p = f.name if hasattr(f, "name") else str(f)
            if p.lower().endswith(".txt"):
                txts[_stem(p)] = p

    # checkbox는 파일명으로 보여주고, 내부는 filename->stem 매핑
    choices = sorted([os.path.basename(p) for p in imgs.values()])
    fn2stem = {os.path.basename(p): s for s, p in imgs.items()}

    cache = {"imgs": imgs, "txts": txts, "fn2stem": fn2stem}

    n_pair = sum(1 for s in imgs.keys() if s in txts)
    log = f"[LOCAL] images={len(imgs)}, txt={len(txts)}, pairs={n_pair} (basename 기준)"
    return choices, cache, log

def copy_existing_dataset_into_final(
    existing_root: str,
    final_root: str,
    overwrite: bool = True,
    exclude_names: Set[str] | None = None,
) -> str:
    """
        [기존 데이터셋 -> 최종 데이터셋 루트로 복사]
        existing_root 하위의 파일/폴더 구조를 그대로 final_root로 복사한다.

        입력:
          - existing_root: 기존 데이터셋 루트 경로
          - final_root   : 최종 저장 루트(미리 생성되어 있어야 함)
          - overwrite    : True면 동일 파일이 있어도 덮어씀
          - exclude_names: 제외할 폴더명/파일명 set (os.walk에서 내려가지 않거나 파일 복사 스킵)

        출력:
          - 처리 결과 로그 문자열(복사 파일 수/스킵 수, from/to)

        주의:
          - exclude_names는 "이름 단위" 필터이므로, 경로가 아니라 dir/file 이름으로만 제외한다.
          - final_root는 사전에 존재해야 한다(없으면 에러 메시지 반환).
    """
    existing_root = (existing_root or "").strip()
    final_root = (final_root or "").strip()
    exclude_names = exclude_names or set()

    if not os.path.isdir(existing_root):
        return f"[ERROR] 기존 데이터셋 경로가 유효하지 않음: {existing_root}"
    if not os.path.isdir(final_root):
        return f"[ERROR] final_root가 유효하지 않음(먼저 생성 필요): {final_root}"

    copied_files = 0
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(existing_root):
        # 1) 제외 디렉토리 제거 (walk가 안 내려가게)
        dirnames[:] = [d for d in dirnames if d not in exclude_names]

        rel = os.path.relpath(dirpath, existing_root)
        dst_dir = os.path.join(final_root, rel) if rel != "." else final_root
        os.makedirs(dst_dir, exist_ok=True)

        # 2) 파일 복사
        for fn in filenames:
            if fn in exclude_names:
                skipped += 1
                continue

            src = os.path.join(dirpath, fn)
            dst = os.path.join(dst_dir, fn)

            if (not overwrite) and os.path.exists(dst):
                skipped += 1
                continue

            shutil.copy2(src, dst)
            copied_files += 1

    return f"[OK] 기존 데이터셋 복사 완료: {copied_files} files (skipped {skipped}) | from={existing_root} -> to={final_root}"

def move_selected_pairs(
    selected_image_filenames: List[str],
    cache: Dict,
    existing_root: str,
    new_root: str,
    new_label_dirname: str = "txt",  # new_root 하위 라벨 폴더명(txt or labels)
    overwrite: bool = True,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
        [업로드 선택 이미지/라벨 쌍을 2곳에 복사]
        체크된 이미지 파일명 리스트를 기준으로,
        cache(stem 매칭)에서 이미지/라벨 쌍을 찾아서

        1) 기존 데이터셋 existing_root/images/train, existing_root/labels/train 에 복사하고
        2) 신규 데이터셋 new_root/images, new_root/{new_label_dirname} 에도 복사한다.

        입력:
          - selected_image_filenames: 체크된 이미지 "파일명" 리스트(basename)
          - cache: build_upload_cache()가 만든 {"imgs","txts","fn2stem"} 포함 dict
          - existing_root: 기존 데이터셋 루트
          - new_root: 이번 데이터셋 루트
          - new_label_dirname: new_root 하위 라벨 폴더명(예: "labels" 또는 "txt")
          - overwrite: False면 대상 경로에 파일이 존재할 경우 스킵

        출력:
          - log: 성공/실패(스킵) 요약 + 경로 정보 + 실패 상세(최대 20개)
          - moved: (img_basename, txt_basename) 성공 목록

        주의:
          - stem 매핑 실패/파일 실존 여부/라벨 누락 등의 케이스를 missing으로 수집한다.
          - "move"라는 이름이지만 실제 동작은 copy2(복사)이다.
    """
    if not selected_image_filenames:
        return "[ABORT] 선택된 이미지가 없습니다.", []

    if not existing_root or not os.path.isdir(existing_root):
        return f"[ABORT] 기존 데이터셋 경로가 유효하지 않습니다: {existing_root}", []

    if not new_root or not os.path.isdir(new_root):
        return f"[ABORT] 이번 데이터셋 경로가 유효하지 않습니다: {new_root}", []

    # destination
    ex_img_train = os.path.join(existing_root, "images", "train")
    ex_lbl_train = os.path.join(existing_root, "labels", "train")
    os.makedirs(ex_img_train, exist_ok=True)
    os.makedirs(ex_lbl_train, exist_ok=True)

    new_img_dir = os.path.join(new_root, "images")
    new_lbl_dir = os.path.join(new_root, new_label_dirname)
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_lbl_dir, exist_ok=True)

    moved = []
    missing = []

    for fn in selected_image_filenames:
        stem = cache.get("fn2stem", {}).get(fn)
        if not stem:
            missing.append((fn, "stem 매핑 실패"))
            continue

        img_src = cache["imgs"].get(stem)
        txt_src = cache["txts"].get(stem)

        if not img_src or not os.path.exists(img_src):
            missing.append((fn, "이미지 없음"))
            continue
        if not txt_src or not os.path.exists(txt_src):
            missing.append((fn, "대응 txt 없음"))
            continue

        # 1) 기존 train
        img_dst1 = os.path.join(ex_img_train, os.path.basename(img_src))
        txt_dst1 = os.path.join(ex_lbl_train, os.path.basename(txt_src))

        # 2) 이번 데이터셋
        img_dst2 = os.path.join(new_img_dir, os.path.basename(img_src))
        txt_dst2 = os.path.join(new_lbl_dir, os.path.basename(txt_src))

        # overwrite 정책
        if (not overwrite) and (os.path.exists(img_dst1) or os.path.exists(txt_dst1) or os.path.exists(img_dst2) or os.path.exists(txt_dst2)):
            missing.append((fn, "이미 존재(스킵)"))
            continue

        shutil.copy2(img_src, img_dst1)
        shutil.copy2(txt_src, txt_dst1)
        shutil.copy2(img_src, img_dst2)
        shutil.copy2(txt_src, txt_dst2)

        moved.append((os.path.basename(img_src), os.path.basename(txt_src)))

    log = []
    log.append(f"[OK] 요청={len(selected_image_filenames)}개, 성공={len(moved)}개, 실패/스킵={len(missing)}개")
    log.append(f"[DST] existing train: {ex_img_train}, {ex_lbl_train}")
    log.append(f"[DST] new dataset: {new_img_dir}, {new_lbl_dir}")

    if missing:
        log.append("[DETAIL] 실패/스킵(최대 20개):")
        for a, b in missing[:20]:
            log.append(f" - {a}: {b}")

    return "\n".join(log), moved


# -----------------------------
#             NEW
# -----------------------------
def _norm(p: str) -> str:
    """
        [유틸] None/공백 입력을 안전하게 처리하기 위해 문자열 strip을 적용한다.
    """
    return (p or "").strip()

def _is_dir(p: str) -> bool:
    """
        [유틸] 경로가 비어있지 않고 실제 디렉토리인지 확인한다.
    """
    return bool(p) and os.path.isdir(p)

def _mkdir(p: str):
    """
        [유틸] 디렉토리를 생성한다(이미 있으면 무시).
    """
    os.makedirs(p, exist_ok=True)

def _list_images_onelevel(dir_path: str) -> List[str]:
    """
        [유틸] 특정 폴더의 "1 depth"에 있는 이미지 파일 목록만 가져온다.
        - recursive 검색이 아니라 dir_path/*.{ext} 수준만 스캔
        - IMG_EXTS 확장자를 순회하며 glob로 수집

        입력:
          - dir_path: 이미지가 들어있는 디렉토리

        출력:
          - 이미지 파일 전체 경로 리스트(정렬됨)
    """
    if not _is_dir(dir_path):
        return []
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort()
    return paths

# -----------------------------
# 1) 기존 데이터셋 통계 (고정 구조)
# existing_root/images/train|val 기준
# -----------------------------
def build_existing_dataset_stats_df(existing_root: str) -> Tuple[pd.DataFrame, str]:
    """
        [기존 데이터셋 통계 DataFrame 생성]
        existing_root/images/train 과 existing_root/images/val 의 이미지 개수를 세어
        split별 count 테이블(DataFrame)로 만든다.

        입력:
          - existing_root: 기존 데이터셋 루트(고정 구조 가정)

        출력:
          - df: columns=["split","count"] (train, val 2행)
          - msg: 로드 성공/경고 메시지

        주의:
          - 폴더 구조가 없거나 경로가 invalid면 count=0 형태의 df를 반환한다.
    """
    existing_root = _norm(existing_root)
    if not _is_dir(existing_root):
        df = pd.DataFrame([{"split": "train", "count": 0}, {"split": "val", "count": 0}])
        return df, f"[WARN] 기존 데이터셋 경로가 유효하지 않음: {existing_root}"

    train_dir = os.path.join(existing_root, "images", "train")
    val_dir   = os.path.join(existing_root, "images", "val")

    train_cnt = len(_list_images_onelevel(train_dir))
    val_cnt = len(_list_images_onelevel(val_dir))

    df = pd.DataFrame([
        {"split": "train", "count": train_cnt},
        {"split": "val", "count": val_cnt},
    ])
    return df, f"[OK] 기존 데이터셋 통계 로드 (train={train_cnt}, val={val_cnt})"

# -----------------------------
# 2) 신규 데이터셋 이미지 목록 (one-level)
# new_root/images/* 만
# -----------------------------
def list_new_images_for_checkbox_onelevel(new_root: str) -> Tuple[List[str], str]:
    """
        [신규 데이터셋 이미지 목록 -> 체크박스용]
        new_root/images 폴더(1 depth)에서 이미지 파일을 모아
        checkbox에 넣기 좋은 basename 리스트로 반환한다.

        입력:
          - new_root: 신규 데이터셋 루트

        출력:
          - filenames: 이미지 파일명 리스트(basename)
          - msg: 로드 결과 메시지

        주의:
          - new_root/images 폴더가 없으면 에러 메시지를 반환한다.
    """
    new_root = _norm(new_root)
    if not _is_dir(new_root):
        return [], f"[WARN] 신규 데이터셋 경로가 유효하지 않음: {new_root}"

    img_dir = os.path.join(new_root, "images")
    if not _is_dir(img_dir):
        return [], f"[ERROR] 신규 데이터셋에 images 폴더가 없음: {img_dir}"

    img_paths = _list_images_onelevel(img_dir)
    filenames = [os.path.basename(p) for p in img_paths]
    return filenames, f"[OK] 신규 이미지 {len(filenames)}개 로드"

# -----------------------------
# 3) 최종 저장 루트 생성 (사용자 입력 폴더명으로 생성)
# final_root = out_root / dataset_name
# 그리고 images/train|val + labels/train|val 생성
# -----------------------------
def ensure_out_dataset_root(out_root: str, dataset_name: str) -> Tuple[str, str]:
    """
        [최종 저장 루트(표준 YOLO 구조) 생성]
        사용자가 입력한 dataset_name을 out_root 하위 폴더로 만들고,
        YOLO 표준 구조(images/labels + train/val)를 미리 생성한다.

        최종 경로:
          final_root = out_root / dataset_name
          final_root/images/train
          final_root/images/val
          final_root/labels/train
          final_root/labels/val

        입력:
          - out_root: 상위 저장 루트(존재해야 함)
          - dataset_name: 사용자 입력 폴더명(경로 탈출 방지 최소 검증)

        출력:
          - msg: 생성 결과 메시지
          - final_root: 생성된 최종 루트 경로(실패 시 "")

        주의:
          - dataset_name에 /, \\, .. 포함 시 차단(경로 탈출 방지)
    """
    out_root = _norm(out_root)
    dataset_name = _norm(dataset_name)

    if not _is_dir(out_root):
        return f"[ERROR] out_root가 유효하지 않음: {out_root}", ""

    if not dataset_name:
        return "[ERROR] 신규 데이터셋 폴더명을 입력해야 함", ""

    # 폴더명 안전성(최소 방어)
    if any(x in dataset_name for x in ["/", "\\", ".."]):
        return "[ERROR] 폴더명에 '/', '\\', '..' 는 사용할 수 없음", ""

    final_root = os.path.join(out_root, dataset_name)
    _mkdir(final_root)

    # 고정 구조 생성
    for d in [
        os.path.join(final_root, "images", "train"),
        os.path.join(final_root, "images", "val"),
        os.path.join(final_root, "labels", "train"),
        os.path.join(final_root, "labels", "val"),
    ]:
        _mkdir(d)

    return f"[OK] 최종 저장 루트 준비 완료: {final_root}", final_root

# -----------------------------
# 4) 신규(one-level) -> 최종(train/val) 복사
# 신규: new_root/images/* , new_root/labels/* (한 레벨)
# 체크한 파일=Train, 미체크=Val
# -----------------------------
def split_new_dataset_by_selection_onelevel(
    new_root: str,
    final_out_root: str,
    selected_train_filenames: List[str],
    overwrite: bool = True,
) -> Tuple[str, pd.DataFrame]:
    """
        [신규 데이터셋(one-level)을 최종(train/val)로 분할 복사]
        신규 데이터셋 구조(가정):
          new_root/images/*.(img)
          new_root/labels/*.txt

        분할 규칙:
          - 체크된 이미지 파일명(selected_train_filenames) => train
          - 미체크 => val
        그리고 최종 저장소(final_out_root)의 표준 구조로 복사:
          final_out_root/images/train|val
          final_out_root/labels/train|val

        입력:
          - new_root: 신규 데이터셋 루트
          - final_out_root: ensure_out_dataset_root()로 만든 최종 루트
          - selected_train_filenames: train으로 보낼 이미지 filename(basename) 리스트
          - overwrite: 덮어쓰기 여부

        출력:
          - msg: 복사 결과 요약(이미지/라벨/누락/저장경로/체크 수)
          - df : split별로 어떤 파일이 어디로 갔는지 기록한 DataFrame
                 columns=["split","image","label_found"]

        주의:
          - 라벨은 stem 매칭(이미지 파일명 기준)으로 찾음
          - 라벨 파일이 없으면 label_found=False로 기록하고 missing_lbl 카운트 증가
    """

    new_root = _norm(new_root)
    final_out_root = _norm(final_out_root)

    if not _is_dir(new_root):
        return f"[ERROR] new_root가 유효하지 않음: {new_root}", pd.DataFrame()
    if not _is_dir(final_out_root):
        return f"[ERROR] final_out_root가 유효하지 않음(먼저 생성 버튼 실행 필요): {final_out_root}", pd.DataFrame()

    src_img_dir = os.path.join(new_root, "images")
    src_lbl_dir = os.path.join(new_root, "labels")

    if not _is_dir(src_img_dir):
        return f"[ERROR] 신규 images 폴더가 없음: {src_img_dir}", pd.DataFrame()
    if not _is_dir(src_lbl_dir):
        return f"[ERROR] 신규 labels 폴더가 없음: {src_lbl_dir}", pd.DataFrame()

    img_paths = _list_images_onelevel(src_img_dir)
    img_map = {os.path.basename(p): p for p in img_paths}

    lbl_paths = glob.glob(os.path.join(src_lbl_dir, "*.txt"))
    lbl_map = {os.path.splitext(os.path.basename(p))[0]: p for p in lbl_paths}

    selected_train = set(selected_train_filenames or [])

    dst_img_train = os.path.join(final_out_root, "images", "train")
    dst_img_val   = os.path.join(final_out_root, "images", "val")
    dst_lbl_train = os.path.join(final_out_root, "labels", "train")
    dst_lbl_val   = os.path.join(final_out_root, "labels", "val")

    rows = []
    copied_img = copied_lbl = missing_lbl = 0

    for img_fn, src_img in img_map.items():
        stem, _ = os.path.splitext(img_fn)
        split = "train" if img_fn in selected_train else "val"

        dst_img = os.path.join(dst_img_train if split == "train" else dst_img_val, img_fn)
        dst_lbl = os.path.join(dst_lbl_train if split == "train" else dst_lbl_val, f"{stem}.txt")

        # image copy
        if overwrite or (not os.path.exists(dst_img)):
            shutil.copy2(src_img, dst_img)
        copied_img += 1

        # label copy
        src_lbl = lbl_map.get(stem)
        if src_lbl and os.path.isfile(src_lbl):
            if overwrite or (not os.path.exists(dst_lbl)):
                shutil.copy2(src_lbl, dst_lbl)
            copied_lbl += 1
            label_found = True
        else:
            missing_lbl += 1
            label_found = False

        rows.append({"split": split, "image": img_fn, "label_found": label_found})

    df = pd.DataFrame(rows)

    msg = (
        f"[OK] 분할/복사 완료\n"
        f" - 이미지: {copied_img}개\n"
        f" - 라벨: {copied_lbl}개\n"
        f" - 라벨 누락: {missing_lbl}개\n"
        f" - 저장 루트: {final_out_root}\n"
        f" - train 체크 수: {len(selected_train)}"
    )
    return msg, df

# -----------------------------
# 신규 데이터셋 로컬로부터 업로드
# -----------------------------
def _safe_name(name: str) -> str:
    """
        [유틸] 사용자 입력 폴더명에 대해 최소한의 안전성 검사를 수행한다.
        - 공백/None이면 ""
        - '/', '\\', '..' 포함 시 "" 반환 (경로 탈출 방지)
    """
    name = (name or "").strip()
    if not name:
        return ""
    if any(x in name for x in ["/", "\\", ".."]):
        return ""
    return name

def _get_path(f: Any) -> str:
    """
        [유틸] gradio 업로드 객체/문자열/딕셔너리에서 파일 경로를 꺼낸다.
        지원 케이스:
          - f가 str인 경우: 그대로 반환
          - f가 {"name": "..."} 형태 dict인 경우: dict["name"]
          - 그 외: f.name 속성이 있으면 사용
    """
    if isinstance(f, str):
        return f
    if isinstance(f, dict) and "name" in f:
        return f["name"]
    return getattr(f, "name", "") or ""

def upload_files_to_labeling_dataset(
    dataset_name: str,
    img_files: List[Any],
    txt_files: List[Any],
    overwrite: bool = True,
) -> Tuple[str, Dict[str, str]]:
    """
        [로컬 업로드 파일을 라벨링용 신규 데이터셋 폴더로 저장]
        업로드된 이미지/라벨 파일을
          UPLOAD_NEWDATASET_ROOT/<dataset_name>/images
          UPLOAD_NEWDATASET_ROOT/<dataset_name>/labels
        로 복사 저장한다.

        저장 정책:
          - 이미지 확장자(IMG_EXTS)만 images에 저장
          - txt는 "이미지와 stem이 매칭되는 것만" labels에 저장
            (이미지 없이 올라온 txt는 저장하지 않고 extra_txt로만 카운트)

        입력:
          - dataset_name: 생성할 폴더명(금지문자 검증)
          - img_files: 업로드 이미지 리스트(gradio 객체/dict/str)
          - txt_files: 업로드 txt 리스트(gradio 객체/dict/str)
          - overwrite: 덮어쓰기 여부

        출력:
          - log: 저장 결과 요약(저장 경로/이미지 수/매칭 라벨 수/누락 라벨 수/미매칭 txt 수)
          - paths: {"dataset_root":..., "images_dir":..., "labels_dir":...}

        주의:
          - 요구사항에 따라 "미매칭 txt도 저장" 정책으로 바꾸고 싶으면 옵션(extra 저장)을 추가하면 된다.
    """
    dataset_name = _safe_name(dataset_name)
    if not dataset_name:
        return "[ERROR] 폴더명(dataset_name)이 비어있거나 금지문자(/,\\,..)가 포함됨", {}

    # 루트 생성
    dataset_root = os.path.join(UPLOAD_NEWDATASET_ROOT, dataset_name)
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    img_files = img_files or []
    txt_files = txt_files or []

    img_paths = [_get_path(x) for x in img_files if _get_path(x)]
    txt_paths = [_get_path(x) for x in txt_files if _get_path(x)]

    # txt 매핑(stem->path)
    txt_map = {}
    for p in txt_paths:
        fn = os.path.basename(p)
        stem, ext = os.path.splitext(fn)
        if ext.lower() == ".txt":
            txt_map[stem] = p

    copied_img = 0
    copied_txt = 0
    missing_txt = 0
    extra_txt = 0

    used_txt_stems = set()

    # 이미지 저장 + 대응 라벨 저장
    for img_p in img_paths:
        img_fn = os.path.basename(img_p)
        stem, ext = os.path.splitext(img_fn)
        if ext.lower() not in IMG_EXTS:
            continue

        dst_img = os.path.join(images_dir, img_fn)
        if overwrite or (not os.path.exists(dst_img)):
            shutil.copy2(img_p, dst_img)
        copied_img += 1

        src_txt = txt_map.get(stem)
        if src_txt and os.path.isfile(src_txt):
            dst_txt = os.path.join(labels_dir, f"{stem}.txt")
            if overwrite or (not os.path.exists(dst_txt)):
                shutil.copy2(src_txt, dst_txt)
            copied_txt += 1
            used_txt_stems.add(stem)
        else:
            missing_txt += 1

    # 이미지 없이 txt만 올라온(불필요) 라벨은 필요하면 같이 저장할 수도 있는데,
    # 요구사항이 “업로드한 파일 저장”이라면 남는 txt도 저장하도록 옵션화 가능.
    # 여기서는 "이미지와 매칭되는 txt만 저장" 기준으로 extra 카운트만 제공.
    for stem in txt_map.keys():
        if stem not in used_txt_stems:
            extra_txt += 1

    log = (
        f"[OK] 업로드 저장 완료\n"
        f" - dataset_root: {dataset_root}\n"
        f" - images 저장: {copied_img}개\n"
        f" - labels 저장(매칭): {copied_txt}개\n"
        f" - labels 누락(이미지 기준): {missing_txt}개\n"
        f" - 매칭되지 않은 txt(미저장): {extra_txt}개"
    )

    return log, {"dataset_root": dataset_root, "images_dir": images_dir, "labels_dir": labels_dir}