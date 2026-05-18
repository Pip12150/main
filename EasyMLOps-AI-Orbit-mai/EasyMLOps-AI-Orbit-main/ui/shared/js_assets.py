# ui/shared/js_assets.py
import os
import json
import cv2

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_all_js(base_dir: str = ".") -> str:
    util_js = _read(os.path.join(base_dir, "6_util.js"))
    load_json_js = _read(os.path.join(base_dir, "6_load_json.js"))
    editor_js = _read(os.path.join(base_dir, "6_je.editor.js"))
    return "\n\n".join([util_js, load_json_js, editor_js])

def save_polygons_for_editor_from_seg_txt(
    image_path: str,
    txt_path: str,
    classes_txt_path: str | None = None,
    json_path: str = "seg_output.json",
    assume_normalized: str = "auto",   # "auto" | True | False
    conf_threshold: float = 0.0,
    min_points: int = 3
):
    """
    Seg txt format:
      class conf x1 y1 x2 y2 x3 y3 ...  (polygon)
    -> Gradio polygon editor JSON format:
      {"annotations":[{"id", "class_id", "class_name", "conf", "segmentation":[flat]}]}
    - supports normalized or pixel coords
    """

    # image size (needed if normalized)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지 로드 실패: {image_path}")
    h, w = img.shape[:2]

    # class names (optional)
    class_names = None
    if classes_txt_path and os.path.exists(classes_txt_path):
        with open(classes_txt_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    def class_name(cid: int) -> str:
        if class_names and 0 <= cid < len(class_names):
            return class_names[cid]
        return ""

    annotations = []
    ann_id = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2 + min_points * 2:
                # cls + conf + 최소 (x,y)*3
                continue

            cid = int(float(parts[0]))
            conf = float(parts[1])
            if conf < conf_threshold:
                continue

            coords = list(map(float, parts[2:]))

            # 좌표 짝수개인지 확인
            if len(coords) % 2 != 0:
                coords = coords[:-1]  # 마지막 하나 버림(방어)

            # 점 개수 확인
            num_pts = len(coords) // 2
            if num_pts < min_points:
                continue

            # normalized 여부 판단
            if assume_normalized == "auto":
                # 모든 좌표가 0~1 근처면 normalized로 간주(여유 포함)
                is_norm = True
                for v in coords[: min(len(coords), 40)]:  # 너무 길면 일부만 검사
                    if not (0.0 <= v <= 1.01):
                        is_norm = False
                        break
            else:
                is_norm = bool(assume_normalized)

            # norm -> pixel
            flat = []
            if is_norm:
                for i in range(0, len(coords), 2):
                    x = coords[i] * w
                    y = coords[i+1] * h
                    # clamp
                    x = max(0.0, min(float(w - 1), x))
                    y = max(0.0, min(float(h - 1), y))
                    flat.extend([float(x), float(y)])
            else:
                for i in range(0, len(coords), 2):
                    x = max(0.0, min(float(w - 1), coords[i]))
                    y = max(0.0, min(float(h - 1), coords[i+1]))
                    flat.extend([float(x), float(y)])

            annotations.append({
                "id": ann_id,
                "class_id": cid,
                "class_name": class_name(cid),
                "conf": conf,
                "segmentation": [flat]
            })
            ann_id += 1

    out = {"annotations": annotations}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Polygon JSON saved -> {json_path}")
    return json_path, out
