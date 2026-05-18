# ui/tabs/tab7_inference.py
import gradio as gr
import os
import csv
import zipfile
from xml.sax.saxutils import escape
from core.file_browser import IMAGE_EXTS, parent_dir, filter_files, MODEL_EXTS, list_dir, join_path
from core.utilities import build_folder_picker
from core.config import PROJECT_ROOT
from core.inf_conf import _predict_one
import cv2
import shutil
import pandas as pd
from pathlib import Path
from ui.shared.js_assets import save_polygons_for_editor_from_seg_txt #json 만들기 위함

from datetime import datetime
def get_today_ymd():
    return datetime.now().strftime("%y%m%d")

def build_inference_tab(
    default_img_dir: str,
    default_model_dir: str,
):
    """
    7번째 탭 UI 구성
    """
    with gr.Row():
        # ======================
        # Left: 원본 이미지 선택 + 실행 버튼
        # ======================
        with gr.Column(scale=2):

            with gr.Column():
                gr.Markdown("### 이미지 폴더 선택하기")
                eval_img_path_tb, _, _, _ = build_folder_picker(
                    label="평가 이미지 폴더",
                    root_dir=PROJECT_ROOT,
                    default_path=PROJECT_ROOT,
                )

            with gr.Column():
                gr.Markdown("### 모델 경로 선택하기")

                model_cur = gr.Textbox(label="현재 경로", value=default_model_dir)
                with gr.Row():
                    model_btn_up = gr.Button("상위 폴더", key="model_up")
                    model_btn_refresh = gr.Button("새로고침", key="model_refresh")

                model_dirs = gr.Dropdown(label="폴더", choices=[], interactive=True)
                model_files = gr.Dropdown(label="모델 파일(.pt)", choices=[], interactive=True)

                weights_dir_tb = gr.Textbox(label="선택된 모델 경로", interactive=False)


            with gr.Accordion(label="추론 파라미터 선택", open=False):
                eval_imgsz_slider = gr.Slider(256, 2048, step=64, value=640, label="imgsz")
                eval_conf_tb = gr.Number(value=0.25, label="conf")
                eval_iou_tb = gr.Number(value=0.5, label="iou")
                eval_device_tb = gr.Textbox(value="0", label="device")

            btn_infer = gr.Button("추론 시작", variant="primary")
            progress_md = gr.Markdown("⏳ inference 대기중...")
            infer_log_tb = gr.Textbox(label="추론 상태 로그", lines=15, interactive=False)
            report_summary_md = gr.Markdown("Inference report: -")
            report_csv_file = gr.File(label="CSV report")
            report_xlsx_file = gr.File(label="Excel report")

        with gr.Column(scale=3):
            gr.Markdown("### inference 결과")
            viewer_orig_name = gr.Textbox(label="현재 파일명", lines=1, interactive=False)
            report_table = gr.Dataframe(
                headers=["cls", "count", "avg_confidence", "min_confidence", "max_confidence"],
                label="Inference statistics by class",
                interactive=False,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 원본 이미지")
                    viewer_orig_img = gr.Image(type="numpy", label="원본 이미지")

                with gr.Column(scale=1):
                    gr.Markdown("### 추론 결과")
                    viewer_infer_img = gr.Image(type="numpy", label="추론 결과")

            with gr.Row():
                btn_prev_img = gr.Button("이전")
                btn_next_img = gr.Button("다음")

            gr.Markdown("### 복사할 이미지 체크 - image, txt, json이 저장됩니다.")
            with gr.Row():
                btn_mark_bad = gr.Button("복사할 이미지로 체크")
                btn_unmark_bad = gr.Button("체크 해제")
                btn_save_bad = gr.Button("선택한 이미지 저장")
            with gr.Row():
                bad_list_md = gr.Textbox(label="선택된 이미지 리스트", lines=10, interactive=False)

    ''' state '''
    server_img_dir_state = gr.State()  # 원본 이미지 폴더
    server_infer_dir_state = gr.State()  # inference 결과 폴더
    viewer_state = gr.State()  # tab1의 SourceState

    ''' event '''
    def _empty_report_outputs():
        empty_df = pd.DataFrame(
            columns=["cls", "count", "avg_confidence", "min_confidence", "max_confidence"]
        )
        return "Inference report: -", empty_df, None, None

    def _update_class_acc(acc: dict, cls: int, count: int, mean_conf: float, min_conf: float, max_conf: float):
        if count <= 0:
            return
        if cls not in acc:
            acc[cls] = {
                "count": 0,
                "conf_sum": 0.0,
                "min_confidence": min_conf,
                "max_confidence": max_conf,
            }
        acc[cls]["count"] += count
        acc[cls]["conf_sum"] += mean_conf * count
        acc[cls]["min_confidence"] = min(acc[cls]["min_confidence"], min_conf)
        acc[cls]["max_confidence"] = max(acc[cls]["max_confidence"], max_conf)

    def _build_report_table(class_acc: dict) -> pd.DataFrame:
        rows = []
        total_count = 0
        total_conf_sum = 0.0
        total_min = None
        total_max = None

        for cls in sorted(class_acc):
            data = class_acc[cls]
            count = int(data["count"])
            avg_conf = data["conf_sum"] / count if count else 0.0
            min_conf = float(data["min_confidence"])
            max_conf = float(data["max_confidence"])
            rows.append({
                "cls": str(cls),
                "count": count,
                "avg_confidence": round(avg_conf, 4),
                "min_confidence": round(min_conf, 4),
                "max_confidence": round(max_conf, 4),
            })
            total_count += count
            total_conf_sum += data["conf_sum"]
            total_min = min_conf if total_min is None else min(total_min, min_conf)
            total_max = max_conf if total_max is None else max(total_max, max_conf)

        if total_count:
            rows.append({
                "cls": "ALL",
                "count": total_count,
                "avg_confidence": round(total_conf_sum / total_count, 4),
                "min_confidence": round(float(total_min), 4),
                "max_confidence": round(float(total_max), 4),
            })

        return pd.DataFrame(
            rows,
            columns=["cls", "count", "avg_confidence", "min_confidence", "max_confidence"],
        )

    def _write_simple_xlsx(path: str, sheets: dict[str, list[list]]):
        def col_name(idx: int) -> str:
            name = ""
            while idx:
                idx, rem = divmod(idx - 1, 26)
                name = chr(65 + rem) + name
            return name

        def cell_xml(row_idx: int, col_idx: int, value):
            ref = f"{col_name(col_idx)}{row_idx}"
            if value is None or value == "":
                return f'<c r="{ref}"/>'
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f'<c r="{ref}"><v>{value}</v></c>'
            return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

        workbook_sheets = []
        workbook_rels = []
        content_overrides = []

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for sheet_idx, (sheet_name, rows) in enumerate(sheets.items(), 1):
                safe_name = str(sheet_name)[:31] or f"Sheet{sheet_idx}"
                workbook_sheets.append(
                    f'<sheet name="{escape(safe_name)}" sheetId="{sheet_idx}" r:id="rId{sheet_idx}"/>'
                )
                workbook_rels.append(
                    f'<Relationship Id="rId{sheet_idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{sheet_idx}.xml"/>'
                )
                content_overrides.append(
                    f'<Override PartName="/xl/worksheets/sheet{sheet_idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                )

                row_xml = []
                for row_idx, row in enumerate(rows, 1):
                    cells = "".join(
                        cell_xml(row_idx, col_idx, val) for col_idx, val in enumerate(row, 1)
                    )
                    row_xml.append(f'<row r="{row_idx}">{cells}</row>')
                zf.writestr(
                    f"xl/worksheets/sheet{sheet_idx}.xml",
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                    f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>',
                )

            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                + "".join(content_overrides)
                + "</Types>",
            )
            zf.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                "</Relationships>",
            )
            zf.writestr(
                "xl/workbook.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>',
            )
            zf.writestr(
                "xl/_rels/workbook.xml.rels",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(workbook_rels)
                + '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                + "</Relationships>",
            )
            zf.writestr(
                "xl/styles.xml",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
                '<fills count="2"><fill><patternFill patternType="none"/></fill>'
                '<fill><patternFill patternType="gray125"/></fill></fills>'
                '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
                '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                "</styleSheet>",
            )

    def _write_inference_reports(report_dir: str, summary_rows: list[list], class_df: pd.DataFrame, image_rows: list[dict]):
        csv_path = os.path.join(report_dir, "inference_report.csv")
        xlsx_path = os.path.join(report_dir, "inference_report.xlsx")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Summary"])
            writer.writerows(summary_rows)
            writer.writerow([])
            writer.writerow(["Per Class"])
            writer.writerow(list(class_df.columns))
            writer.writerows(class_df.values.tolist())
            writer.writerow([])
            writer.writerow(["Per Image"])
            image_headers = [
                "image",
                "status",
                "detections",
                "avg_confidence",
                "min_confidence",
                "max_confidence",
                "error",
            ]
            writer.writerow(image_headers)
            for row in image_rows:
                writer.writerow([row.get(h, "") for h in image_headers])

        _write_simple_xlsx(
            xlsx_path,
            {
                "Summary": [["Metric", "Value"], *summary_rows],
                "Per Class": [list(class_df.columns), *class_df.values.tolist()],
                "Per Image": [
                    [
                        "image",
                        "status",
                        "detections",
                        "avg_confidence",
                        "min_confidence",
                        "max_confidence",
                        "error",
                    ],
                    *[
                        [
                            row.get("image", ""),
                            row.get("status", ""),
                            row.get("detections", 0),
                            row.get("avg_confidence", 0.0),
                            row.get("min_confidence", ""),
                            row.get("max_confidence", ""),
                            row.get("error", ""),
                        ]
                        for row in image_rows
                    ],
                ],
            },
        )
        return csv_path, xlsx_path

    def infer_folder(
            img_dir: str,
            model_path: str,
            pt_name: str,
            imgsz: int,
            conf: float,
            iou: float,
            device: str,
    ):
        logs = []
        report_summary, report_df, report_csv, report_xlsx = _empty_report_outputs()

        if not os.path.isdir(img_dir):
            yield "❌ 이미지 폴더가 유효하지 않습니다.", "", "", "❌", report_summary, report_df, report_csv, report_xlsx
            return

        print("model_path: ", model_path)
        if not os.path.exists(model_path):
            yield "❌ best.pt가 존재하지 않습니다.", "", "", "❌", report_summary, report_df, report_csv, report_xlsx
            return

        img_files = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])

        total = len(img_files)
        if total == 0:
            yield "❌ 이미지 없음", "", "", "❌", report_summary, report_df, report_csv, report_xlsx
            return

        img_save_dir, txt_save_dir = build_inf_save_dir(
            PROJECT_ROOT,
            img_folder=img_dir,
            model_path=model_path,
        )
        report_dir = str(Path(img_save_dir).parent)
        class_acc = {}
        image_rows = []
        failed_images = 0

        logs.append(f"📁 이미지 저장 위치: {img_save_dir}")
        yield "\n".join(logs), img_dir, img_save_dir, f"inference 시작 - 0 / {total}", report_summary, report_df, report_csv, report_xlsx

        for i, fname in enumerate(img_files, 1):
            img_path = os.path.join(img_dir, fname)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                failed_images += 1
                image_rows.append({
                    "image": fname,
                    "status": "load_failed",
                    "detections": 0,
                    "avg_confidence": 0.0,
                    "min_confidence": "",
                    "max_confidence": "",
                    "error": "OpenCV could not read the image.",
                })
                logs.append(f"{fname}: ❌ 로드 실패")
                yield "\n".join(logs), img_dir, img_save_dir, f"inference 에러 - {i} / {total}", report_summary, report_df, report_csv, report_xlsx
                continue

            try:
                vis, summary, res = _predict_one(
                    model_path=model_path,
                    img_bgr=img_bgr,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                )
            except Exception as exc:
                failed_images += 1
                image_rows.append({
                    "image": fname,
                    "status": "predict_failed",
                    "detections": 0,
                    "avg_confidence": 0.0,
                    "min_confidence": "",
                    "max_confidence": "",
                    "error": str(exc),
                })
                logs.append(f"{fname}: ❌ inference 실패 ({exc})")
                yield "\n".join(logs), img_dir, img_save_dir, f"inference 에러 - {i} / {total}", report_summary, report_df, report_csv, report_xlsx
                continue

            # 1. overlay 이미지 저장
            img_out_path = os.path.join(img_save_dir, fname)
            if not cv2.imwrite(img_out_path, vis[:, :, ::-1]):
                failed_images += 1
                image_rows.append({
                    "image": fname,
                    "status": "save_failed",
                    "detections": 0,
                    "avg_confidence": 0.0,
                    "min_confidence": "",
                    "max_confidence": "",
                    "error": f"Could not write result image: {img_out_path}",
                })
                logs.append(f"{fname}: ❌ 결과 이미지 저장 실패")
                yield "\n".join(logs), img_dir, img_save_dir, f"inference 에러 - {i} / {total}", report_summary, report_df, report_csv, report_xlsx
                continue

            # 2. txt 저장
            txt_name = os.path.splitext(fname)[0] + ".txt"
            txt_out_path = os.path.join(txt_save_dir, txt_name)
            save_yolo_txt_from_res(res, txt_out_path)

            image_count = 0
            image_conf_sum = 0.0
            image_min = None
            image_max = None
            if summary is not None and not summary.empty:
                for _, row in summary.iterrows():
                    cls = int(row["cls"])
                    count = int(row["count"])
                    mean_conf = float(row["conf_mean"])
                    min_conf = float(row["conf_min"])
                    max_conf = float(row["conf_max"])
                    _update_class_acc(class_acc, cls, count, mean_conf, min_conf, max_conf)
                    image_count += count
                    image_conf_sum += mean_conf * count
                    image_min = min_conf if image_min is None else min(image_min, min_conf)
                    image_max = max_conf if image_max is None else max(image_max, max_conf)

            image_rows.append({
                "image": fname,
                "status": "ok",
                "detections": image_count,
                "avg_confidence": round(image_conf_sum / image_count, 4) if image_count else 0.0,
                "min_confidence": round(float(image_min), 4) if image_min is not None else "",
                "max_confidence": round(float(image_max), 4) if image_max is not None else "",
                "error": "",
            })

            logs.append(f"{fname}: ✅ inference 완료")
            report_df = _build_report_table(class_acc)
            yield "\n".join(logs), img_dir, img_save_dir, f"inference 진행중 - {i} / {total}", report_summary, report_df, report_csv, report_xlsx

        logs.append("🎉 inference 완료")
        report_df = _build_report_table(class_acc)
        total_detections = (
            int(report_df[report_df["cls"] == "ALL"]["count"].iloc[0])
            if not report_df.empty and (report_df["cls"] == "ALL").any()
            else 0
        )
        avg_conf = (
            float(report_df[report_df["cls"] == "ALL"]["avg_confidence"].iloc[0])
            if total_detections
            else 0.0
        )
        images_with_detections = sum(1 for row in image_rows if int(row["detections"]) > 0)
        summary_rows = [
            ["Images processed", total],
            ["Images with detections", images_with_detections],
            ["Failed images", failed_images],
            ["Total detections", total_detections],
            ["Avg confidence", f"{avg_conf * 100:.2f}%"],
            ["Result folder", report_dir],
        ]
        report_summary = (
            f"**Inference report**  \n"
            f"Total detections: **{total_detections}**  \n"
            f"Avg Confidence: **{avg_conf * 100:.2f}%**  \n"
            f"Images with detections: **{images_with_detections}/{total}**  \n"
            f"Failed images: **{failed_images}**"
        )
        report_csv, report_xlsx = _write_inference_reports(report_dir, summary_rows, report_df, image_rows)
        yield "\n".join(logs), img_dir, img_save_dir, f"inference 완료 - {total} / {total}", report_summary, report_df, report_csv, report_xlsx

    def build_inf_save_dir(project_root: str, img_folder: str, model_path: str):
        # dataset name
        dataset_name = os.path.basename(os.path.normpath(img_folder))

        # model info
        # .../demo_exp20_epoch200/weights/best.pt
        model_name = os.path.splitext(os.path.basename(model_path))[0]  # best
        train_name = os.path.basename(
            os.path.dirname(os.path.dirname(model_path))
        )  # demo_exp20_epoch200

        dir_name = f"{dataset_name}__{train_name}__{model_name}"

        save_root = os.path.join(project_root, "tab7_inference", dir_name)
        img_save_dir = os.path.join(save_root, "result_images")
        txt_save_dir = os.path.join(save_root, "labels")

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(txt_save_dir, exist_ok=True)

        return img_save_dir, txt_save_dir

    def list_pt_files(training_dir: str):
        weights_dir = training_dir
        print("weights_dir : ", weights_dir)
        if not os.path.isdir(weights_dir):
            return gr.update(choices=[], value=None)

        pts = sorted([
            f for f in os.listdir(weights_dir)
            if f.endswith(".pt")
        ])

        default = "best.pt" if "best.pt" in pts else (pts[0] if pts else None)
        return gr.update(choices=pts, value=default)

    def init_infer_view(orig_dir: str, infer_dir: str):
        imgs = sorted([
            f for f in os.listdir(infer_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])

        if not imgs:
            return None, None, "", "❌ 결과 이미지 없음", {
                "orig_dir": orig_dir,
                "infer_dir": infer_dir,
                "images": [],
                "idx": 0,
                "bad_images": [],
            }

        first = imgs[0]

        orig_img = cv2.imread(os.path.join(orig_dir, first))
        infer_img = cv2.imread(os.path.join(infer_dir, first))

        return (
            orig_img[:, :, ::-1] if orig_img is not None else None,
            infer_img[:, :, ::-1] if infer_img is not None else None,
            first,
            {
                "orig_dir": orig_dir,
                "infer_dir": infer_dir,
                "images": imgs,
                "idx": 0,
                "bad_images": [],
            }
        )

    # yolo txt 저장
    def save_yolo_txt_from_res(res, txt_path: str):
        """
        Ultralytics YOLO segmentation 결과(res) → YOLO seg txt 저장
        포맷: class conf x1 y1 x2 y2 ...
        """
        if res.masks is None or res.boxes is None:
            open(txt_path, "w").close()
            return

        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        polygons = res.masks.xyn  # normalized polygon

        lines = []
        for cls, conf, poly in zip(cls_ids, confs, polygons):
            if poly is None or len(poly) < 3:
                continue

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lines.append(f"{cls} {conf:.6f} {coords}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

    # json 생성
    def generate_json_from_img_txt(
            img_path: str,
            txt_path: str,
            json_out_path: str,
            conf_threshold: float = 0.25,
    ):
        """
        inference img + txt → polygon json 생성
        """
        if not os.path.exists(img_path):
            return False, f"[JSON] 이미지 없음: {os.path.basename(img_path)}"

        if not os.path.exists(txt_path):
            return False, f"[JSON] txt 없음: {os.path.basename(txt_path)}"

        json_path, _ = save_polygons_for_editor_from_seg_txt(
            image_path=img_path,
            txt_path=txt_path,
            classes_txt_path=None,  # 필요하면 나중에 추가
            json_path=json_out_path,
            conf_threshold=conf_threshold,
            assume_normalized="auto",
        )

        return True, json_path

    # 이미지 이전 / 다음
    def viewer_move(step: int, state: dict):
        if not state or not state.get("images"):
            return state, None, None, "", ""

        images = state["images"]
        idx = state["idx"]

        new_idx = max(0, min(idx + step, len(images) - 1))
        fname = images[new_idx]

        orig_path = os.path.join(state["orig_dir"], fname)
        infer_path = os.path.join(state["infer_dir"], fname)

        orig_img = cv2.imread(orig_path)
        infer_img = cv2.imread(infer_path)

        state["idx"] = new_idx

        return (
            state,
            orig_img[:, :, ::-1] if orig_img is not None else None,
            infer_img[:, :, ::-1] if infer_img is not None else None,
            fname,  # orig name
        )

    def on_prev(state: dict):
        return viewer_move(-1, state)

    def on_next(state: dict):
        return viewer_move(1, state)

    # bad img 선택
    def mark_bad(state: dict):
        if not state or not state.get("images"):
            return state, "선택된 이미지 없음"

        state.setdefault("bad_images", [])

        fname = state["images"][state["idx"]]
        if fname not in state["bad_images"]:
            state["bad_images"].append(fname)

        return state, render_bad_list(state)

    def unmark_bad(state: dict):
        if not state or not state.get("images"):
            return state, "선택된 이미지 없음"

        state.setdefault("bad_images", [])

        fname = state["images"][state["idx"]]
        if fname in state["bad_images"]:
            state["bad_images"].remove(fname)

        return state, render_bad_list(state)

    def save_bad_images(state: dict):
        if not state or not state.get("bad_images"):
            return "⚠️ 저장할 이미지가 없습니다."

        orig_dir = state["orig_dir"]
        infer_dir = Path(state["infer_dir"])
        labels_dir = infer_dir.parent / "labels"
        labels_dir = str(labels_dir)

        dataset_name = os.path.basename(os.path.normpath(orig_dir))
        date_tag = datetime.now().strftime("%y%m%d")
        dataset_name_with_date = f"{dataset_name}_{date_tag}"

        save_root = os.path.join(PROJECT_ROOT, "tab7_inference", "bad_cases")
        save_img_dir = os.path.join(save_root, dataset_name_with_date, "images")
        save_txt_dir = os.path.join(save_root, dataset_name_with_date, "labels")
        save_json_dir = os.path.join(save_root, dataset_name_with_date, "json")
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_txt_dir, exist_ok=True)
        os.makedirs(save_json_dir, exist_ok=True)

        logs = []
        count = 0

        for fname in state["bad_images"]:
            src_img = os.path.join(orig_dir, fname)
            dst_img = os.path.join(save_img_dir, fname)

            txt_fname = fname.replace(".jpg", ".txt")
            src_txt = os.path.join(labels_dir, txt_fname)
            dst_txt = os.path.join(save_txt_dir, txt_fname)

            json_fname = fname.replace(".jpg", ".json")
            dst_json = os.path.join(save_json_dir, json_fname)

            if not os.path.exists(src_img):
                logs.append(f"[MISSING IMAGE] {fname}")
                continue

            if not os.path.exists(src_txt):
                logs.append(f"[MISSING TXT] {txt_fname}")
                continue

            # 1. copy image
            shutil.copy2(src_img, dst_img)

            # 2. copy txt
            shutil.copy2(src_txt, dst_txt)

            # 3. generate json
            ok, msg = generate_json_from_img_txt(
                img_path=dst_img,
                txt_path=dst_txt,
                json_out_path=dst_json,
                conf_threshold=0.25,
            )

            if not ok:
                logs.append(msg)
                continue

            count += 1

        summary = f"✅ 이미지 {count}개 복사 완료\n📁 {save_img_dir}"

        if logs:
            summary += "\n\n⚠️ 로그:\n" + "\n".join(logs)

        return summary

    def render_bad_list(state: dict):
        if not state or not state.get("bad_images"):
            return "선택된 이미지 없음"

        lines = "\n".join([f"- {f}" for f in state["bad_images"]])
        return f"**총 {len(state['bad_images'])}개 선택됨**\n\n{lines}"

    def _refresh_model(cur_path: str):
        cur, dirs, files = list_dir(cur_path)
        files = filter_files(files, MODEL_EXTS)
        return cur, gr.update(choices=dirs, value=None), gr.update(choices=files, value=None)

    def _enter_model_dir(cur_path: str, dir_name: str):
        if not dir_name:
            return _refresh_model(cur_path)
        nxt = join_path(cur_path, dir_name)
        return _refresh_model(nxt)

    def _pick_file(cur_path: str, file_name: str):
        if not cur_path or not file_name:
            return ""
        return os.path.join(cur_path, file_name)

    ''' event 등록 '''
    model_btn_refresh.click(
        fn=_refresh_model,
        inputs=[model_cur],
        outputs=[model_cur, model_dirs, model_files],
    )

    model_btn_up.click(
        fn=lambda p: _refresh_model(parent_dir(p)),
        inputs=[model_cur],
        outputs=[model_cur, model_dirs, model_files],
    )

    model_dirs.change(
        fn=_enter_model_dir,
        inputs=[model_cur, model_dirs],
        outputs=[model_cur, model_dirs, model_files],
    )

    model_files.change(
        fn=_pick_file,
        inputs=[model_cur, model_files],
        outputs=[weights_dir_tb],
    )

    btn_infer.click(
        fn=infer_folder,
        inputs=[
            eval_img_path_tb,
            weights_dir_tb,
            model_files,
            eval_imgsz_slider,
            eval_conf_tb,
            eval_iou_tb,
            eval_device_tb,
        ],
        outputs=[
            infer_log_tb,
            server_img_dir_state,
            server_infer_dir_state,
            progress_md,
            report_summary_md,
            report_table,
            report_csv_file,
            report_xlsx_file,
        ],
    ).then(
        fn=init_infer_view,
        inputs=[
            server_img_dir_state,
            server_infer_dir_state,
        ],
        outputs=[
            viewer_orig_img,
            viewer_infer_img,
            viewer_orig_name,
            viewer_state,
        ]
    )

    # 이미지 이전 / 다음
    btn_prev_img.click(
        fn=on_prev,
        inputs=[viewer_state],
        outputs=[
            viewer_state,
            viewer_orig_img,
            viewer_infer_img,
            viewer_orig_name,
        ],
    )

    btn_next_img.click(
        fn=on_next,
        inputs=[viewer_state],
        outputs=[
            viewer_state,
            viewer_orig_img,
            viewer_infer_img,
            viewer_orig_name,
        ],
    )

    # bad img 선택
    btn_mark_bad.click(
        fn=mark_bad,
        inputs=[viewer_state],
        outputs=[viewer_state, bad_list_md],
    )

    btn_unmark_bad.click(
        fn=unmark_bad,
        inputs=[viewer_state],
        outputs=[viewer_state, bad_list_md],
    )

    btn_save_bad.click(
        fn=save_bad_images,
        inputs=[viewer_state],
        outputs=[infer_log_tb],  # 로그창 재활용
    )


def build_tab7_inference(PROJECT_ROOT, RUNS_DIR):
    with gr.Tab("7. 모델 inference"):
        build_inference_tab(
            # default_img_dir="/home/gpuadmin/seongje_maixcam/yolo11_seg_dataset/images/val",
            # default_model_dir="/home/gpuadmin/seongje_gradio2/test_yolo_project/runs/segment",
            default_img_dir=PROJECT_ROOT,
            default_model_dir=RUNS_DIR,
        )
