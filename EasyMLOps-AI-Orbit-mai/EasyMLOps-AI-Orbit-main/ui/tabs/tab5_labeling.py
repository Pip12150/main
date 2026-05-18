# ui/tabs/tab5_labeling.py
import time, json, base64
import os
import cv2
import tempfile
from pathlib import Path
from io import BytesIO
from PIL import Image
import gradio as gr

from ui.shared.js_assets import save_polygons_for_editor_from_seg_txt

COCO_CLASSES = ["oil", "water", "bubble"]
COCO_COLORS = ["#ff0000", "#00aaff", "#ffaa00"]

def pil_to_b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_polygon_for_edit(selected_file, image_file):
    """
    업로드된 JSON + 이미지로 JS editor에 폴리곤 로드하기
    """

    if selected_file is None:
        return {"status": "error", "message": "⚠ JSON 파일이 선택되지 않았습니다."}
    if image_file is None:
        return {"status": "error", "message": "⚠ 이미지 파일도 함께 업로드해야 합니다."}

    file_path = selected_file.name
    poly_raw = Path(file_path).read_text(encoding="utf-8")

    img_path = image_file.name
    img = Image.open(img_path)
    b64img = pil_to_b64(img)

    return {
        "polygon_json": poly_raw,
        "image_b64": b64img,
        "meta": {"colors": COCO_COLORS, "names": COCO_CLASSES},
        "_nonce": time.time(),
        "loaded_file_path": file_path
    }



def gen_json_return_file_and_json(img_f, txt_f, cls_f):
    """
    (이미지 + 추론 txt + classes.txt) → 임시 JSON 생성 + JS editor 로딩
    """
    json_path, out = save_polygons_for_editor_from_seg_txt(
        image_path=img_f.name,
        txt_path=txt_f.name,
        classes_txt_path=cls_f.name if cls_f else None,
        json_path="temp_generated.json",
        conf_threshold=0.25,
        assume_normalized="auto"
    )
    return json_path, out


def json_file_to_yolo_seg_txt(json_f, image_f, out_name="edited.txt"):
    """
    ✅ 핵심: JSON 파일 + 이미지 파일로 YOLO-seg 학습용 txt 생성해서 다운로드
    txt 형식:  class_id x1 y1 x2 y2 ... (0~1 정규화)
    """
    if json_f is None:
        raise gr.Error("Export할 JSON 파일을 선택해주세요. (Select polygon JSON file)")
    if image_f is None:
        raise gr.Error("이미지 파일이 필요합니다.")

    img = cv2.imread(image_f.name)
    if img is None:
        raise gr.Error(f"이미지 로드 실패: {image_f.name}")
    h, w = img.shape[:2]

    json_text = Path(json_f.name).read_text(encoding="utf-8")
    try:
        data = json.loads(json_text)
    except Exception as e:
        raise gr.Error(f"JSON 파싱 실패: {e}")

    anns = data.get("annotations", [])
    lines = []
    for ann in anns:
        class_id = ann.get("class_id", 0)
        try:
            class_id = int(class_id)
        except Exception:
            # 혹시 (0,) 같은 이상한 형태면 문자 정리
            s = str(class_id).replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").strip()
            class_id = int(float(s))

        seg = ann.get("segmentation", None)
        if not seg or not seg[0]:
            continue

        poly = seg[0]  # [x1,y1,x2,y2,...] (pixel coords)
        coords = []
        for i in range(0, len(poly), 2):
            x = poly[i] / w
            y = poly[i + 1] / h
            # 0~1 클램프
            x = 0.0 if x < 0 else (1.0 if x > 1 else x)
            y = 0.0 if y < 0 else (1.0 if y > 1 else y)
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")

        if coords:
            lines.append(f"{class_id} " + " ".join(coords))

    # 안전한 임시 파일 생성
    if not out_name or out_name.strip() == "":
        out_name = "edited.txt"
    if not out_name.endswith(".txt"):
        out_name += ".txt"

        # 시스템의 임시 디렉토리 경로를 가져와서 사용자가 원하는 파일명과 합칩니다.
    temp_dir = tempfile.gettempdir()
    out_path = os.path.join(temp_dir, out_name)

    # 파일 쓰기
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    # --- 수정된 부분 끝 ---

    return out_path


def build_tab5_labeling():
    from ui.shared.js_assets import load_all_js
    ALL_JS = load_all_js("./json")

    with gr.Tab("5. Labeling"):
        gr.HTML(f"<script>{ALL_JS}</script>")

        with gr.Row():
            load_json_file = gr.File(label="Select polygon JSON file(select only)", file_types=[".json"])
            image_file = gr.File(label="Select image file (.jpg or .png)", file_types=["image"])
            load_btn = gr.Button("Load Polygon + Image + rendering")

            infer_txt_file = gr.File(label="Select inference txt (YOLO seg)", file_types=[".txt"])
            classes_txt_file = gr.File(label="Select classes.txt", file_types=[".txt"])
            gen_btn = gr.Button("Generate JSON from Image+TXT")
            gen_out = gr.JSON(visible=False)
            gen_file = gr.File(label="Download JSON", visible=True)

            # 이미지+txt에서 JSON 생성 후 JS editor 호출
            gen_btn.click(
                fn=gen_json_return_file_and_json,
                inputs=[image_file, infer_txt_file, classes_txt_file],
                outputs=[gen_file, gen_out]
            ).then(
                fn=None,
                inputs=[gen_out],
                outputs=None,
                js=r"""
                (p) => { window.js_editor(p); return null; }
                """
            )

        with gr.Row():
            js_log_box = gr.Textbox(label="JS Log", elem_id="js_log_box", lines=8, max_lines=12, interactive=False)

        gr.HTML("""
        <div style="display:flex; justify-content:center; width:100%; margin-top:10px;">
            <canvas id="edit_canvas" width="800" height="600" style="border:1px solid white;"></canvas>
        </div>
        """)

        # Load JSON + Attach JS Editor
        load_out = gr.JSON(visible=False)
        ev = load_btn.click(fn=load_polygon_for_edit, inputs=[load_json_file, image_file], outputs=load_out)
        ev.then(fn=None, inputs=load_out, outputs=None, js="(p) => window.js_editor(p)")

        # save json (기존 기능 유지: 파일로 저장)
        with gr.Row():
            save_name = gr.Textbox(label="Save as (filename.json)", value="edited.json")
            save_btn = gr.Button("Save(in json)")

        save_btn.click(
            fn=None, inputs=[save_name], outputs=None,
            js="""
            (filename) => {
                const cur = window.get_current_json();
                const jsonText = (typeof cur === "string") ? cur : JSON.stringify(cur, null, 2);
                return window.save_json_via_filepicker(filename || "edited.json", jsonText);
            }
            """
        )

        # ✅ NEW: Export YOLO TXT (파일 기반 변환)
        with gr.Row():
            export_txt_name = gr.Textbox(label="Export as (filename.txt)", value="edited.txt")
            export_txt_btn = gr.Button("Export YOLO TXT")
            export_txt_file = gr.File(label="Download YOLO TXT", visible=True)

        export_txt_btn.click(
            fn=json_file_to_yolo_seg_txt,
            inputs=[load_json_file, image_file, export_txt_name],
            outputs=[export_txt_file]
        )

        # NEW polygon controls (기존 유지)
        with gr.Row():
            drop_choices = [(f"{i:02d} — {COCO_CLASSES[i]}  [{COCO_COLORS[i]}]", i) for i in range(len(COCO_CLASSES))]
            new_class = gr.Dropdown(choices=drop_choices, label="Class ID for New Polygon", value=0)
            add_mode_btn = gr.Button("➕ New Polygon", scale=1)
            finish_poly_btn = gr.Button("✔ Finish Polygon", scale=1)

            add_mode_btn.click(fn=None, inputs=None, outputs=None, js="() => add_mode_on()")
            finish_poly_btn.click(fn=None, inputs=new_class, outputs=None, js="(cls_value) => finish_poly(cls_value)")
