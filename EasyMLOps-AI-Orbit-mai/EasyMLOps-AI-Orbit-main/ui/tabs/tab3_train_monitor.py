# ui/tabs/tab3_train_monitor.py
import os
import shutil

import gradio as gr
import yaml

from core.config import (
    LOSS_COLUMNS,
    METRIC_COLUMNS,
    PROJECT_ROOT,
    RUNS_DIR,
    UPLOAD_TRAINING_INFO_DIR,
)
from core.train_monitor_service import (
    build_epoch_conf_monitor_ui,
    next_page,
    prev_page,
    refresh_6plots_compare,
)
from core.utilities import build_folder_picker, save_uploaded_file
from core.utils_csv import _build_runs_map, _file_to_path, _on_run_change
from core.yolo_train import run_epoch_eval_manual


def build_tab3_train_monitor(trainer):
    with gr.Tab("3. Train Monitor"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Training")
                task = gr.Radio(["detect", "segment"], value="segment", label="YOLO Task")

                data_yaml_file = gr.File(
                    label="Upload data.yaml",
                    file_types=[".yaml", ".yml"],
                    file_count="single",
                )
                data_yaml_path = gr.Textbox(label="Saved data.yaml path", interactive=False)
                btn_check_conf = gr.Button("Check Label Conf")
                preprocess_status = gr.Textbox(
                    label="Pre-processing status",
                    lines=6,
                    interactive=False,
                )

                model_pt_file = gr.File(
                    label="Upload model (.pt)",
                    file_types=[".pt"],
                    file_count="single",
                )
                model_pt_path = gr.Textbox(label="Saved model path", interactive=False)

                data_yaml_file.change(
                    fn=lambda f: save_uploaded_file(f, UPLOAD_TRAINING_INFO_DIR),
                    inputs=[data_yaml_file],
                    outputs=[data_yaml_path],
                )
                model_pt_file.change(
                    fn=lambda f: save_uploaded_file(f, UPLOAD_TRAINING_INFO_DIR),
                    inputs=[model_pt_file],
                    outputs=[model_pt_path],
                )

                with gr.Row():
                    monitor_imgsz = gr.Slider(
                        label="imgsz",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=640,
                    )
                    monitor_epochs = gr.Slider(
                        label="epochs",
                        minimum=1,
                        maximum=500,
                        step=1,
                        value=100,
                    )
                with gr.Row():
                    monitor_batch = gr.Slider(
                        label="batch",
                        minimum=1,
                        maximum=128,
                        step=1,
                        value=16,
                    )
                    monitor_lr0 = gr.Number(label="lr0", value=0.001)

                with gr.Row():
                    btn_start_train = gr.Button("Start Training (CLI)", variant="primary")
                    btn_stop_train = gr.Button("Force Stop Training", variant="stop")
                train_status = gr.Textbox(label="Status", lines=8, interactive=False)

                btn_stop_train.click(
                    fn=lambda: trainer.stop_train(),
                    inputs=[],
                    outputs=[train_status],
                )

                gr.Markdown("### Epoch Model Evaluation (manual)")
                with gr.Row():
                    with gr.Column(scale=1):
                        weights_path_tb, _, _, _ = build_folder_picker(
                            label="weights folder",
                            root_dir=PROJECT_ROOT,
                            default_path=os.path.join(PROJECT_ROOT, "runs"),
                        )
                    with gr.Column(scale=1):
                        eval_img_path_tb, _, _, _ = build_folder_picker(
                            label="evaluation image folder",
                            root_dir=PROJECT_ROOT,
                            default_path=os.path.join(PROJECT_ROOT, "datasets"),
                        )

                with gr.Row():
                    eval_imgsz = gr.Slider(
                        label="imgsz",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=640,
                    )
                    eval_conf = gr.Number(label="conf_thres", value=0.25)
                    eval_iou = gr.Number(label="iou_thres", value=0.5)
                    eval_device = gr.Textbox(label="device", value="0")

                btn_eval = gr.Button("Run Epoch Evaluation", variant="primary")
                eval_log = gr.Textbox(label="Evaluation log", lines=12, interactive=False)
                btn_eval.click(
                    fn=run_epoch_eval_manual,
                    inputs=[
                        weights_path_tb,
                        eval_img_path_tb,
                        eval_imgsz,
                        eval_conf,
                        eval_iou,
                        eval_device,
                    ],
                    outputs=[eval_log],
                )

                gr.Markdown("### Monitoring Source (Primary)")
                results_csv_path = gr.Textbox(
                    label="results.csv path (empty = latest auto)",
                    value="",
                )
                results_csv_file = gr.File(
                    label="Select results.csv",
                    file_types=[".csv"],
                    file_count="single",
                )
                refresh_sec = gr.Slider(
                    label="refresh seconds",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=2,
                )

                gr.Markdown("### Previous Runs (Compare)")
                compare_enabled = gr.Checkbox(value=True, label="Compare with previous run")
                btn_refresh_runs = gr.Button("Refresh runs")
                runs_dropdown = gr.Dropdown(label="Previous run", choices=[], value=None)
                runs_map_state = gr.State(value={})
                prev_results_csv_path = gr.Textbox(
                    label="Selected previous results.csv",
                    interactive=False,
                    value="",
                )

            with gr.Column(scale=3):
                gr.Markdown("### Live Training Curves")
                view_mode = gr.Radio(["metrics", "loss"], value="metrics", label="View")

                with gr.Row():
                    btn_prev = gr.Button("Prev")
                    page_state = gr.State(1)
                    page_view = gr.Markdown("Page: 1")
                    btn_next = gr.Button("Next")

                plot6 = []
                for _ in range(2):
                    with gr.Row():
                        for _ in range(3):
                            plot6.append(gr.Plot(label=""))

                last_update = gr.Markdown("Last update: -")
                with gr.Accordion("Epoch/Best/Last Conf trend (scan)", open=False):
                    build_epoch_conf_monitor_ui(default_weights_dir="")

        timer = gr.Timer(value=2.0)

        results_csv_file.change(
            fn=lambda f: _file_to_path(f),
            inputs=[results_csv_file],
            outputs=[results_csv_path],
        )

        btn_refresh_runs.click(
            fn=_build_runs_map,
            inputs=[task],
            outputs=[runs_dropdown, runs_map_state, prev_results_csv_path],
        )
        runs_dropdown.change(
            fn=_on_run_change,
            inputs=[runs_dropdown, runs_map_state],
            outputs=[prev_results_csv_path],
        )

        btn_prev.click(fn=prev_page, inputs=[page_state], outputs=[page_state])
        btn_next.click(
            fn=lambda p, m: next_page(p, METRIC_COLUMNS if m == "metrics" else LOSS_COLUMNS),
            inputs=[page_state, view_mode],
            outputs=[page_state],
        )
        page_state.change(
            fn=lambda p: f"Page: {int(p)}",
            inputs=[page_state],
            outputs=[page_view],
        )

        timer.tick(
            fn=lambda primary_csv, rs, p, m, comp_csv, comp_on: refresh_6plots_compare(
                primary_csv,
                rs,
                int(p),
                m,
                RUNS_DIR,
                METRIC_COLUMNS,
                LOSS_COLUMNS,
                compare_csv_path=comp_csv,
                compare_enabled=comp_on,
            ),
            inputs=[
                results_csv_path,
                refresh_sec,
                page_state,
                view_mode,
                prev_results_csv_path,
                compare_enabled,
            ],
            outputs=[*plot6, last_update, timer, page_state],
        )

        btn_check_conf.click(
            fn=sanitize_segmentation_labels,
            inputs=[data_yaml_path],
            outputs=[preprocess_status],
        )

        btn_start_train.click(
            fn=lambda t, dy, mp, isz, ep, ba, lr0: start_train_with_preprocessing(
                trainer, t, dy, mp, isz, ep, ba, lr0
            ),
            inputs=[
                task,
                data_yaml_path,
                model_pt_path,
                monitor_imgsz,
                monitor_epochs,
                monitor_batch,
                monitor_lr0,
            ],
            outputs=[preprocess_status, train_status],
        )


def has_conf(parts):
    if len(parts) < 6:
        return False

    # YOLO labels without confidence have an odd token count:
    # detect: class x y w h = 5, segment: class x1 y1 ... = 1 + 2n.
    # Labels exported from inference with confidence have an even token count:
    # class conf x y w h = 6, class conf x1 y1 ... = 2 + 2n.
    if len(parts) % 2 != 0:
        return False

    try:
        conf = float(parts[1])
        return 0.0 <= conf <= 1.0
    except Exception:
        return False


def sanitize_segmentation_labels(data_yaml_path):
    data_yaml_path = str(data_yaml_path or "").strip()
    if not data_yaml_path:
        return "[ERROR] data.yaml path is empty. Upload/select data.yaml first."

    if not os.path.exists(data_yaml_path):
        return f"[ERROR] data.yaml does not exist: {data_yaml_path}"

    log = ["[PREPROCESS] Checking labels for confidence columns before training..."]

    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        return f"[ERROR] Failed to read data.yaml: {exc}"

    if "train" not in data:
        return "[ERROR] data.yaml must contain 'train'."

    base_path = str(data.get("path") or os.path.dirname(data_yaml_path))
    if not os.path.isabs(base_path):
        base_path = os.path.normpath(os.path.join(os.path.dirname(data_yaml_path), base_path))

    def as_path_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if str(v).strip()]
        return [str(value)]

    def get_label_dir(img_rel):
        img_rel = str(img_rel)
        label_rel = img_rel.replace("images", "labels", 1)
        if os.path.isabs(label_rel):
            return os.path.normpath(label_rel)
        return os.path.normpath(os.path.join(base_path, label_rel))

    label_dirs = []
    for split_name in ("train", "val"):
        for image_path in as_path_list(data.get(split_name)):
            label_dir = get_label_dir(image_path)
            if label_dir not in label_dirs:
                label_dirs.append(label_dir)

    total_checked = 0
    total_changed = 0
    existing_label_dirs = 0

    for label_dir in label_dirs:
        if not os.path.exists(label_dir):
            log.append(f"[WARN] Label directory not found, skipped: {label_dir}")
            continue
        if not os.path.isdir(label_dir):
            log.append(f"[WARN] Label path is not a directory, skipped: {label_dir}")
            continue

        existing_label_dirs += 1

        backup_dir = label_dir + "_with_conf"
        if not os.path.exists(backup_dir):
            shutil.copytree(label_dir, backup_dir)
            log.append(f"[BACKUP] {label_dir} -> {backup_dir}")
        else:
            log.append(f"[BACKUP] already exists: {backup_dir}")

        changed_files = []

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(label_dir, fname)
            total_checked += 1

            new_lines = []
            file_changed = False

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue

                        parts = stripped.split()
                        if has_conf(parts):
                            parts = [parts[0]] + parts[2:]
                            file_changed = True

                        new_lines.append(" ".join(parts))
            except Exception as exc:
                log.append(f"[WARN] Failed to read label file, skipped: {fpath} ({exc})")
                continue

            if file_changed:
                try:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write("\n".join(new_lines))
                        if new_lines:
                            f.write("\n")
                except Exception as exc:
                    log.append(f"[WARN] Failed to write label file, skipped: {fpath} ({exc})")
                    continue

                changed_files.append(fname)
                total_changed += 1

        if changed_files:
            log.append(f"[MODIFIED] {label_dir}: {len(changed_files)} file(s)")
            for changed in changed_files[:20]:
                log.append(f"  - {changed}")
            if len(changed_files) > 20:
                log.append(f"  ... and {len(changed_files) - 20} more")
        else:
            log.append(f"[CLEAN] {label_dir}")

    if existing_label_dirs == 0:
        log.append("[ERROR] No valid label directories were found. Training cannot start safely.")
        return "\n".join(log)

    if total_checked == 0:
        log.append("[ERROR] No label .txt files were found. Training cannot start safely.")
        return "\n".join(log)

    log.append(f"[DONE] Checked {total_checked} label file(s), modified {total_changed}.")
    return "\n".join(log)


def start_train_with_preprocessing(trainer, task, data_yaml, model_pt, imgsz, epochs, batch, lr0):
    preprocess_log = sanitize_segmentation_labels(data_yaml)
    if preprocess_log.startswith("[ERROR]") or "\n[ERROR]" in preprocess_log:
        status = preprocess_log + "\n[ABORTED] Training was not started."
        return preprocess_log, status

    train_msg = trainer.start_train(task, data_yaml, model_pt, imgsz, epochs, batch, lr0)
    return preprocess_log, preprocess_log + "\n\n" + train_msg
