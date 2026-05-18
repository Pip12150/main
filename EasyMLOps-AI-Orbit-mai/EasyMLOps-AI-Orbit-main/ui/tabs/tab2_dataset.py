import gradio as gr
import pandas as pd

from core.config import PROJECT_ROOT
from core.dataset_service import (
    build_existing_dataset_stats_df,
    list_new_images_for_checkbox_onelevel,
    ensure_out_dataset_root,
    split_new_dataset_by_selection_onelevel,
    copy_existing_dataset_into_final,
    upload_files_to_labeling_dataset
)


def build_tab2_dataset():
    with gr.Tab("2. Dataset 설정"):
        final_out_root_state = gr.State("")
        all_selected_state = gr.State(False)

        with gr.Row():
            # =========================
            # LEFT
            # =========================
            with gr.Column(scale=2):
                use_existing = gr.Radio(
                    ["Yes", "No"], value="No", label="기존 데이터셋 활용 여부 체크"
                )

                existing_dataset_dir = gr.FileExplorer(
                    label="기존 데이터셋 경로 선택 (폴더)",
                    root_dir=PROJECT_ROOT,
                    file_count="single",
                )

                new_dataset_dir = gr.FileExplorer(
                    label="이번 데이터셋 경로 선택 (images/, labels/ 한 레벨)",
                    root_dir=PROJECT_ROOT,
                    file_count="single",
                )

                out_dataset_dir = gr.FileExplorer(
                    label="신규 데이터셋 저장 상위 경로",
                    root_dir=PROJECT_ROOT,
                    file_count="single",
                )

                dataset_name = gr.Textbox(
                    label="신규 데이터셋 폴더명",
                    placeholder="예) dataset_20251231_v1",
                )

                create_out_btn = gr.Button("📁 신규 데이터셋 저장 폴더 생성/확인")

            # =========================
            # RIGHT
            # =========================
            with gr.Column(scale=3):
                log_box = gr.Textbox(label="검증/실행 로그", lines=8)

                existing_stats_df = gr.Dataframe(
                    label="기존 데이터셋 통계", interactive=False
                )

                gr.Markdown("⭐ 신규 이미지 목록 ⭐")

                select_all_btn = gr.Button("✅ 전체 선택")

                new_list_box = gr.CheckboxGroup(
                    label="(체크=Train / 미체크=Val)",
                    choices=[],
                    value=[],
                )

                selection_stats = gr.Markdown(
                    "**Train:** 0 | **Val:** 0 | **Total:** 0"
                )

                out_root_view = gr.Textbox(
                    label="최종 저장 루트", interactive=False
                )

                split_btn = gr.Button("✅ train/val 분할 복사 실행")

                split_result_df = gr.Dataframe(
                    label="분할/복사 결과", interactive=False
                )

        # ============================================================
        # UTIL
        # ============================================================
        def _make_stats_text(train_cnt, total_cnt):
            val_cnt = total_cnt - train_cnt
            if total_cnt == 0:
                return "**Train:** 0 | **Val:** 0 | **Total:** 0"
            return (
                f"**Train:** {train_cnt} ({train_cnt/total_cnt*100:.1f}%) | "
                f"**Val:** {val_cnt} ({val_cnt/total_cnt*100:.1f}%) | "
                f"**Total:** {total_cnt}"
            )

        # ============================================================
        # 기존 데이터셋 toggle
        # ============================================================
        def _toggle_existing(v):
            return gr.update(interactive=(v == "Yes")), f"[INFO] use_existing={v}"

        use_existing.change(
            _toggle_existing,
            inputs=[use_existing],
            outputs=[existing_dataset_dir, log_box]
        )

        # ============================================================
        # 기존 데이터셋 통계
        # ============================================================
        def _load_existing_stats(use, ex_root):
            if use != "Yes":
                return (
                    pd.DataFrame(
                        [{"split": "train", "count": 0},
                         {"split": "val", "count": 0}]
                    ),
                    "[INFO] 기존 데이터셋 미사용"
                )
            df, msg = build_existing_dataset_stats_df(ex_root)
            return df, msg

        existing_dataset_dir.change(
            _load_existing_stats,
            inputs=[use_existing, existing_dataset_dir],
            outputs=[existing_stats_df, log_box]
        )

        # ============================================================
        # 신규 이미지 목록 로드
        # ============================================================
        def _load_new_list(new_root):
            choices, msg = list_new_images_for_checkbox_onelevel(new_root)
            return (
                gr.update(choices=choices, value=[]),
                "✅ 전체 선택",
                False,
                _make_stats_text(0, len(choices)),
                msg
            )

        new_dataset_dir.change(
            _load_new_list,
            inputs=[new_dataset_dir],
            outputs=[
                new_list_box,
                select_all_btn,
                all_selected_state,
                selection_stats,
                log_box
            ]
        )

        # ============================================================
        # 전체 선택 / 해제 버튼
        # ============================================================
        def _toggle_all(new_root, all_selected):
            choices, _ = list_new_images_for_checkbox_onelevel(new_root)
            total = len(choices)

            if not all_selected:
                return (
                    gr.update(value=choices),
                    "🧹 전체 해제",
                    True,
                    _make_stats_text(total, total)
                )
            else:
                return (
                    gr.update(value=[]),
                    "✅ 전체 선택",
                    False,
                    _make_stats_text(0, total)
                )

        select_all_btn.click(
            _toggle_all,
            inputs=[new_dataset_dir, all_selected_state],
            outputs=[
                new_list_box,
                select_all_btn,
                all_selected_state,
                selection_stats
            ]
        )

        # ============================================================
        # 개별 체크 변경 → 통계만 갱신
        # ============================================================
        def _update_stats(selected, new_root):
            choices, _ = list_new_images_for_checkbox_onelevel(new_root)
            total = len(choices)
            train_cnt = len(selected)
            is_all = (total > 0 and train_cnt == total)

            return (
                "🧹 전체 해제" if is_all else "✅ 전체 선택",
                is_all,
                _make_stats_text(train_cnt, total)
            )

        new_list_box.change(
            _update_stats,
            inputs=[new_list_box, new_dataset_dir],
            outputs=[select_all_btn, all_selected_state, selection_stats]
        )

        # ============================================================
        # out dataset 생성
        # ============================================================
        def _create_out(use, ex_root, out_root, name):
            print("out_root", out_root, "// name", name)
            msg, final_root = ensure_out_dataset_root(out_root, name)
            if not final_root:
                return msg, "", ""

            if use == "Yes":
                copy_msg = copy_existing_dataset_into_final(
                    existing_root=ex_root,
                    final_root=final_root,
                    overwrite=True,
                    exclude_names={"cache", "__cache__", ".cache", "raw"}
                )
                msg += "\n" + copy_msg

            return msg, final_root, final_root

        create_out_btn.click(
            _create_out,
            inputs=[use_existing, existing_dataset_dir, out_dataset_dir, dataset_name],
            outputs=[log_box, final_out_root_state, out_root_view]
        )

        # ============================================================
        # train / val 분할 실행
        # ============================================================
        def _split(selected_train, new_root, final_out_root):
            msg, df = split_new_dataset_by_selection_onelevel(
                new_root=new_root,
                final_out_root=final_out_root,
                selected_train_filenames=selected_train,
                overwrite=True,
            )
            return msg, df

        split_btn.click(
            _split,
            inputs=[new_list_box, new_dataset_dir, final_out_root_state],
            outputs=[log_box, split_result_df]
        )

        # ============================================================
        # 로컬 업로드 → labeling dataset
        # ============================================================
        with gr.Group():
            gr.Markdown("### ⬆️ 로컬 업로드 → datasets_for_labeling/<폴더명>/images,labels")

            labeling_dataset_name = gr.Textbox(
                label="업로드 저장 폴더명",
                placeholder="예) labeling_20251231_v1",
            )

            local_images = gr.File(
                label="로컬 이미지 업로드(여러개)",
                file_count="multiple",
                file_types=["image"],
            )

            local_txts = gr.File(
                label="로컬 txt 업로드(여러개)",
                file_count="multiple",
                file_types=[".txt"],
            )

            upload_btn = gr.Button("⬆️ 서버로 업로드 저장")
            upload_log = gr.Textbox(label="업로드 로그", lines=8)
            upload_path_view = gr.Textbox(
                label="생성된 dataset_root", interactive=False
            )

            def _upload_to_labeling_root(name, imgs, txts):
                log, info = upload_files_to_labeling_dataset(
                    dataset_name=name,
                    img_files=imgs,
                    txt_files=txts,
                    overwrite=True,
                )
                return log, (info.get("dataset_root", "") if info else "")

            upload_btn.click(
                fn=_upload_to_labeling_root,
                inputs=[labeling_dataset_name, local_images, local_txts],
                outputs=[upload_log, upload_path_view],
            )
