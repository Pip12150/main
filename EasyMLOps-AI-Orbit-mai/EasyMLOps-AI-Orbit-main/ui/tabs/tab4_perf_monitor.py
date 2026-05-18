# ui/tabs/tab4_perf_monitor.py
import gradio as gr
from core.config import INF_RESULTS_ROOT

# features/perf_monitor.py
import matplotlib.pyplot as plt
import gradio as gr
import os
import shutil
from typing import List, Tuple
import pandas as pd
import zipfile
import tempfile
import time
from pathlib import Path
from core.inf_conf import collect_daily_confidence
from core.config import LABELING_DEST_ROOT

def plot_daily_trend(summary_df: pd.DataFrame):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Daily Confidence Mean")

    if summary_df is None or summary_df.empty:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        plt.close(fig)
        return fig

    x = summary_df["date"].tolist()
    ax.plot(x, summary_df["mean"].tolist(), label="mean")
    ax.set_xlabel("date(YYYYMMDD)")
    ax.grid(True)
    ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_hist(raw_df: pd.DataFrame, date_str: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Confidence distribution (date={date_str})")

    if raw_df is None or raw_df.empty or not date_str:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        plt.close(fig)
        return fig

    sub = raw_df[raw_df["date"] == date_str]["confidence"]
    if sub.empty:
        ax.text(0.5, 0.5, "There is no data for the selected date", ha="center", va="center")
        ax.axis("off")
        plt.close(fig)
        return fig

    ax.hist(sub.tolist(), bins=30)
    ax.set_xlabel("confidence")
    ax.grid(True)
    plt.tight_layout()
    plt.close(fig)
    return fig


def _low_mean_dates_from_daily_rows(daily_rows: list[dict], mean_threshold: float) -> list[str]:
    out = []
    for r in daily_rows or []:
        try:
            d = str(r.get("date", ""))
            m = float(r.get("mean", 0.0))
            if d and m <= float(mean_threshold):
                out.append(d)
        except:
            continue
    return out


def copy_labeling_sources_for_dates(
    inf_root: str,
    dates: List[str],
    dest_root: str,
    prefix_with_date: bool = True,
    skip_existing: bool = True,
) -> Tuple[int, int, str]:
    """
    선택된 날짜들의 org_img / txt 파일을 dest_root/images, dest_root/txt 로 '복사(copy2)'한다.
    원본은 유지됨.
    """
    inf_root = (inf_root or "").strip()
    dest_root = (dest_root or "").strip()

    if not inf_root or not os.path.isdir(inf_root):
        return 0, 0, f"[ERROR] inf_root 경로가 유효하지 않음: {inf_root}"
    if not dest_root:
        return 0, 0, "[ERROR] dest_root가 비어있음"

    dst_img_dir = os.path.join(dest_root, "images")
    dst_txt_dir = os.path.join(dest_root, "txt")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_txt_dir, exist_ok=True)

    copied_img = 0
    copied_txt = 0
    logs = []

    def _copy_dir_files(src_dir: str, dst_dir: str, date_str: str, kind: str) -> int:
        if not os.path.isdir(src_dir):
            logs.append(f"[WARN] {date_str} / {kind} 폴더 없음: {src_dir}")
            return 0

        cnt = 0
        for fn in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fn)
            if not os.path.isfile(src_path):
                continue

            dst_name = f"{date_str}__{fn}" if prefix_with_date else fn
            dst_path = os.path.join(dst_dir, dst_name)

            if skip_existing and os.path.exists(dst_path):
                logs.append(f"[SKIP] 이미 존재: {dst_path}")
                continue

            try:
                shutil.copy2(src_path, dst_path)
                cnt += 1
            except Exception as e:
                logs.append(f"[ERROR] 복사 실패: {src_path} -> {dst_path} ({e})")
        return cnt

    for d in dates or []:
        date_dir = os.path.join(inf_root, str(d))
        org_dir = os.path.join(date_dir, "org_img")
        txt_dir = os.path.join(date_dir, "txt")

        copied_img += _copy_dir_files(org_dir, dst_img_dir, str(d), "org_img")
        copied_txt += _copy_dir_files(txt_dir, dst_txt_dir, str(d), "txt")

    logs.insert(0, f"[OK] 복사 완료: images={copied_img}, txt={copied_txt}")
    logs.append(f"[DEST] {dst_img_dir}")
    logs.append(f"[DEST] {dst_txt_dir}")

    return copied_img, copied_txt, "\n".join(logs)


# ✅ 핵심 수정 1: raw_state를 dict로 반환 (raw_df + daily_rows 포함)
def refresh_conf_monitor(inf_root: str, class_filter: str, conf_min_value):
    class_filter = (class_filter or "").strip()
    conf_min = float(conf_min_value) if conf_min_value is not None else None

    raw_df, summary_df = collect_daily_confidence(
        inf_root=inf_root,
        class_filter=class_filter,
        conf_min=conf_min,
    )

    # ✅ 추가: 최근 10일만 유지
    if summary_df is not None and not summary_df.empty:
        summary_df = summary_df.sort_values("date").tail(10)

    summary_df = summary_df if summary_df is not None else pd.DataFrame(columns=["date", "count", "mean"])
    raw_df = raw_df if raw_df is not None else pd.DataFrame(columns=["date", "confidence"])

    table_data = summary_df.values.tolist() if not summary_df.empty else []
    trend_fig = plot_daily_trend(summary_df)

    dates = summary_df["date"].astype(str).tolist() if not summary_df.empty else []

    # ✅ daily_rows(list[dict]) 생성
    daily_rows = summary_df.to_dict(orient="records") if not summary_df.empty else []

    # ✅ raw_state를 dict로 고정
    state = {
        "raw_df": raw_df,
        "daily_rows": daily_rows,
        "dates": dates,
    }

    return (
        table_data,
        trend_fig,
        state,
    )


# ✅ 핵심 수정 2: raw_state(dict)에서 raw_df를 꺼내서 hist 그리기
def change_date_hist(date_str: str, state: dict):
    raw_df = None
    if isinstance(state, dict):
        raw_df = state.get("raw_df", None)
    return plot_hist(raw_df, date_str)


def build_perf_monitor_tab(default_root: str):
    """
    Tab4 UI + (mean<=threshold 날짜 선택) -> org_img/txt 복사 기능 포함
    """
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 일자별 추론 confidence 모니터링")

            inf_root = gr.Textbox(
                label="inf_results 루트 경로",
                value=default_root,
                placeholder="/home/gpuadmin/seongje_gradio2/inf_results",
            )
            class_filter = gr.Textbox(label="class 필터(비우면 전체)", value="")
            conf_min = gr.Number(label="confidence 최소값(필터)", value=0.0)

            btn_refresh = gr.Button("집계/갱신")

            gr.Markdown("### 저성능 날짜 선별 및 라벨링 데이터셋 복사")

            mean_threshold = gr.Number(label="저성능 기준 (일자 mean conf ≤)", value=0.8)

            low_dates = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="복사할 날짜 선택 (mean ≤ 기준)",
                interactive=True,
            )

            dest_root = gr.Textbox(
                label="라벨링 목적지 (고정)",
                value=LABELING_DEST_ROOT,
                interactive=False,
            )

            with gr.Row():
                btn_select_low = gr.Button("mean≤기준 날짜 불러오기")
                btn_copy = gr.Button("선택 날짜 복사", variant="primary")
                btn_zip = gr.Button("ZIP 다운로드", variant="secondary")

            download_file = gr.File(label="ZIP 파일", interactive=False)
            copy_log = gr.Textbox(label="복사 로그", lines=10, interactive=False)

        with gr.Column(scale=3):
            daily_table = gr.Dataframe(
                headers=["date", "count", "mean"],
                datatype=["str", "number", "number"],
                interactive=False,
                label="일자별 통계",
            )
            trend_plot = gr.Plot(label="일자별 추세")
            # date_pick = gr.Dropdown(choices=[], label="날짜 선택", interactive=True)
            # hist_plot = gr.Plot(label="선택 날짜 분포")

    raw_state = gr.State(value=None)

    # events: 집계/갱신
    btn_refresh.click(
        fn=refresh_conf_monitor,
        inputs=[inf_root, class_filter, conf_min],
        outputs=[daily_table, trend_plot, raw_state],
    )

    # ✅ mean≤threshold 날짜 후보 갱신 (state["daily_rows"] 기반)
    def _update_low_dates(state_val, th):
        if not isinstance(state_val, dict):
            return gr.update(choices=[], value=[])

        daily_rows = state_val.get("daily_rows", [])
        cands = _low_mean_dates_from_daily_rows(daily_rows, float(th))
        return gr.update(choices=cands, value=[])

    btn_select_low.click(
        fn=_update_low_dates,
        inputs=[raw_state, mean_threshold],
        outputs=[low_dates],
    )

    # ✅ 선택 날짜 복사
    def _copy_selected_dates(inf_root_path, selected_dates, dest_root_path):
        if not selected_dates:
            return "선택된 날짜가 없습니다."

        _, _, log = copy_labeling_sources_for_dates(
            inf_root=inf_root_path,
            dates=selected_dates,
            dest_root=dest_root_path,
            prefix_with_date=True,
            skip_existing=True,
        )
        return log

    btn_copy.click(
        fn=_copy_selected_dates,
        inputs=[inf_root, low_dates, dest_root],
        outputs=[copy_log],
    )

    def _make_zip(inf_root_path, selected_dates):
        if not selected_dates:
            return None

        inf_root_path = (inf_root_path or "").strip()
        if not inf_root_path or not os.path.isdir(inf_root_path):
            return None

        # ---- 1) 파일명 만들기 (선택 날짜 포함) ----
        dates = sorted([str(d) for d in selected_dates if d])
        if not dates:
            return None

        if len(dates) <= 5:
            date_part = "_".join(dates)
        else:
            date_part = f"{dates[0]}_{dates[-1]}_plus{len(dates) - 2}"

        zip_name = f"inf_{date_part}.zip"

        # ---- 2) temp 폴더에 '항상 같은 이름'으로 저장 => 덮어쓰기 ----
        zip_dir = os.path.join(tempfile.gettempdir(), "qisens_inf_zips")
        os.makedirs(zip_dir, exist_ok=True)
        zip_path = os.path.join(zip_dir, zip_name)

        # 기존 파일이 있으면 덮어쓰기 위해 삭제
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except:
                pass

        # ---- 3) zip 만들기 ----
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for d in dates:
                date_dir = os.path.join(inf_root_path, d)
                org_dir = os.path.join(date_dir, "org_img")
                txt_dir = os.path.join(date_dir, "txt")

                # org_img
                if os.path.isdir(org_dir):
                    for fn in os.listdir(org_dir):
                        src = os.path.join(org_dir, fn)
                        if os.path.isfile(src):
                            arc = os.path.join(d, "org_img", fn)
                            zf.write(src, arcname=arc)

                # txt
                if os.path.isdir(txt_dir):
                    for fn in os.listdir(txt_dir):
                        src = os.path.join(txt_dir, fn)
                        if os.path.isfile(src):
                            arc = os.path.join(d, "txt", fn)
                            zf.write(src, arcname=arc)

        # ---- 4) (선택) temp zip 폴더 정리: 1일 이상 된 zip 삭제 ----
        _cleanup_old_zips(zip_dir, keep_days=1)

        return zip_path

    def _cleanup_old_zips(zip_dir: str, keep_days: int = 1):
        """zip_dir 내 *.zip 중 keep_days(일)보다 오래된 것 삭제"""
        try:
            now = time.time()
            ttl = keep_days * 24 * 3600
            for p in Path(zip_dir).glob("*.zip"):
                try:
                    if now - p.stat().st_mtime > ttl:
                        p.unlink(missing_ok=True)
                except:
                    continue
        except:
            return

    btn_zip.click(
        fn=_make_zip,
        inputs=[inf_root, low_dates],
        outputs=[download_file],
    )

    return {
        "inf_root": inf_root,
        "class_filter": class_filter,
        "conf_min": conf_min,
        "btn_refresh": btn_refresh,
        "daily_table": daily_table,
        "trend_plot": trend_plot,
        "raw_state": raw_state,
        "mean_threshold": mean_threshold,
        "low_dates": low_dates,
        "dest_root": dest_root,
        "btn_select_low": btn_select_low,
        "btn_copy": btn_copy,
        "copy_log": copy_log,
    }

def build_tab4_perf_monitor():
    with gr.Tab("4. 모델 성능 모니터링"):
        build_perf_monitor_tab(default_root=INF_RESULTS_ROOT)
