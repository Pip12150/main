# app.py
import os

import gradio as gr

from core.config import PROJECT_ROOT, YOLO_CLI, RUNS_DIR  #필요한 파일 경로 가져옴
from core.yolo_train import YoloTrainer         #학습/추론 담당 클래스

from ui.shared.js_assets import load_all_js     #커스텀 JS파일을 읽어 Gradio에 주입
from ui.tabs.tab1_viewer import build_tab1_viewer
from ui.tabs.tab2_dataset import build_tab2_dataset
from ui.tabs.tab3_train_monitor import build_tab3_train_monitor
from ui.tabs.tab4_perf_monitor import build_tab4_perf_monitor
from ui.tabs.tab5_labeling import build_tab5_labeling
from ui.tabs.tab6_compare import build_tab6_compare
from ui.tabs.tab7_inference import build_tab7_inference
from ui.tabs.tab8_llm_assistant import build_llm_assistant_tab

SEARCH_BTN_CSS = """
#search_btn_row {
  display: flex !important;
  flex-direction: row !important;
  gap: 8px;
}

#search_btn_row button {
  flex: 1;
  min-width: 0;
  height: 44px;
  padding: 10px 14px;
  font-size: 15px;
  font-weight: 500;
  border-radius: 10px;
}
"""

#탭 상단 고정 css
STICKY_TABS_CSS = """
/* 1) 탭바 후보들을 전부 sticky 처리 */
.gradio-container [role="tablist"],
.gradio-container .tab-nav,
.gradio-container .tabs,
.gradio-container .tabs > div:first-child {
    position: sticky !important;
    top: 0 !important;
    z-index: 99999 !important;
    background: white !important;
}

/* 2) sticky가 깨지는 주요 원인 제거: 부모 overflow/transform */
.gradio-container,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .app,
.gradio-container .main,
.gradio-container .prose {
    overflow: visible !important;
    transform: none !important;
}

/* 3) 탭이 고정되면서 아래 내용이 가려지면 여백 확보(필요시 숫자 조절) */
.gradio-container .tabitem,
.gradio-container [role="tabpanel"] {
    scroll-margin-top: 64px !important;
}
"""

CUSTOM_CSS = SEARCH_BTN_CSS + "\n" + STICKY_TABS_CSS

def create_demo():
    #모든 js파일들을 all_js에 담음
    all_js = load_all_js("./json")

    #yolo 실행파일 경로 및 모델폴더 경로 지정
    trainer = YoloTrainer(yolo_cli=YOLO_CLI, project_root=PROJECT_ROOT)

    with gr.Blocks(title="Easy MLOps") as demo:
        with gr.Tabs():
            #1.이미지 뷰어
            build_tab1_viewer()
            #2.Dataset 설정
            build_tab2_dataset()
            #3.Train Monitor 탭
            build_tab3_train_monitor(trainer=trainer)
            #4.모델 성능 모니터링
            build_tab4_perf_monitor()
            #5.Labeling
            build_tab5_labeling()
            #6.모델추론결과 비교
            build_tab6_compare(PROJECT_ROOT, RUNS_DIR)
            #7.inference
            build_tab7_inference(PROJECT_ROOT, RUNS_DIR)
            #8.LLM Assistant
            build_llm_assistant_tab()

    return demo, all_js
    #return demo, js(UI 및 JavaScript 리턴)

if __name__ == "__main__":
    demo, ALL_JS = create_demo()

    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    port = int(os.getenv("GRADIO_SERVER_PORT", os.getenv("GRADIO_PORT", "7860")))

    demo.launch(
        server_name=host,
        server_port=port,
        css=CUSTOM_CSS,
        js=ALL_JS,
        share=False,
    )
