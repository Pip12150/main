# core/config.py
import os

# =============================
# Project root (repo root)
# =============================
PROJECT_ROOT = os.path.abspath("./test_yolo_project")

# =============================
# Workspace / Examples (repo-relative)
# =============================
WORKSPACE_ROOT = os.path.join(PROJECT_ROOT, "workspace")
EXAMPLES_ROOT = os.path.join(PROJECT_ROOT, "examples")

# FileExplorer root (optional)
# - default: repo root
# - can override by env var (useful on server)
EXPLORER_ROOT = os.environ.get("EXPLORER_ROOT", PROJECT_ROOT)

# Inference results (runtime output; should be gitignored)
INF_RESULTS_ROOT = os.environ.get(
    "INF_RESULTS_ROOT",
    os.path.join(PROJECT_ROOT, "inf_results"),
)

#tab2
UPLOAD_NEWDATASET_ROOT = os.path.join(PROJECT_ROOT, "tab2_datasets_for_labeling")

# tab3
UPLOAD_TRAINING_INFO_DIR = os.path.join(PROJECT_ROOT, "tab3_training_info")
YOLO_CLI = "/home/qisens/anaconda3/envs/gradio/bin/yolo"    # YOLO 가상환경 경로로 바꿔주기
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
METRIC_COLUMNS = [
    "metrics/mAP50-95(B)", "metrics/mAP50(B)",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50-95(M)", "metrics/mAP50(M)",
    "metrics/precision(M)", "metrics/recall(M)",
]
LOSS_COLUMNS = [
    "train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss",
]

# tab4
INF_RESULTS_ROOT = os.path.join(PROJECT_ROOT, "tab4_inf_results")
LABELING_DEST_ROOT = os.path.join(PROJECT_ROOT, "tab4_datasets_for_labeling")


