import os
import threading    #동시 실행 방지용 (학습 프로세스 one-by-one)
import subprocess   #YOLO CLI를 "외부 프로세스"로 실행하기 위해 사용 / Python API 사용시 프로세스가 꼬일 수 있음 / Popen은 해당 라이브러리의 클래스
from core.epoch_eval import run_epoch_reports_in_weights_dir

def normalize_path_from_explorer(val) -> str:
    """
    -> str : 해당 함수가 sting return 하는것을 의미함
             코드 읽을 시 편리함
    """

    """
    [경로 정규화 유틸]
    gr.FileExplorer(또는 유사 컴포넌트)에서 반환하는 값이
    gr.FileExplorer: 유저가 호스트 컴퓨터의 파일 browse 가능하도록 함.
                     input-유저가 사용할 파일 경로 지정
                     output-선택된 파일 경로
    - list[str]
    - str
    - None
    형태로 섞여 들어올 수 있어서(해당 형태로 반환됨), 이를 "단일 경로 문자열"로 정규화한다.
    즉 안전한 문자열 경로로 바꿔주는 것!

    규칙:
      - None -> ""
      - list -> 첫 번째 요소만 사용(단일 선택 UI 기준)
      - str  -> 그대로 반환
      - 그 외 -> ""

    Args:
        val: FileExplorer 입력값(list | str | None)

    Returns:
        단일 경로 문자열(없으면 "")
    """
    if val is None:
        return ""
    if isinstance(val, list):
        if not val:
            return ""
        return str(val[0])   # 첫 번째 선택만 사용 -> 첫 번째 경로: 기준 경로
    if isinstance(val, str):
        return val
    return ""

class YoloTrainer:
    def __init__(self, yolo_cli: str, project_root: str):
        """
            Args:
                yolo_cli: yolo 실행 파일 경로 또는 커맨드(ex: "yolo")
                project_root: 프로젝트 루트(상대경로 기준점, runs 폴더 생성 기준)
        """
        self.yolo_cli = yolo_cli
        self.project_root = project_root
        self._proc = None                   #현재 실행 중인 학습 프로세스/subprocess.Popen의 객체(None -> 아무것도 아닌 상태)
        self._lock = threading.Lock()       #버튼 연타 방지(동시에 프로세스 돌아가지 못하도록 해줌)
        self._epoch_eval_done_for = set()   #집합

    def start_train(self, task, data_yaml, model_pt, imgsz, epochs, batch, lr0, device_str="0"):
        """
            [학습 시작 함수]
            현재 실행 중인 학습 프로세스가 없다면, yolo CLI로 train을 시작한다.

            처리 흐름:
              1) lock으로 동시 실행 방지
              2) self._proc가 실행 중이면 중복 실행을 막고 메시지 반환
              3) data_yaml을 절대경로로 정규화(상대경로면 project_root 기준)
              4) 모델 경로/이름 결정(model_pt 비어있으면 기본 모델 사용)
              5) ultralytics CLI 인자 구성(save_period=10 등)
              6) subprocess.Popen으로 실행(백그라운드 프로세스)
              7) 실행 명령 문자열을 로그로 반환

            Args:
                task: "detect" / "segment" 등 (CLI에서 task train 형태로 사용)
                data_yaml: dataset yaml 경로(상대/절대 가능)
                model_pt: 초기 가중치(pt) 경로 또는 모델명(비어있으면 기본값)
                imgsz: 이미지 크기
                epochs: 학습 epoch 수
                batch: batch size
                lr0: 초기 learning rate(ultralytics 인자)
                device_str: GPU 지정 문자열(ex: "0,1,2,3")

            Returns:
                사용자에게 표시할 상태 메시지(실행 명령 포함)

            주의:
              - name=demo_exp는 고정이라 여러 번 실행하면 exp명이 충돌/증가할 수 있음
                (원하면 timestamp나 사용자 입력 run_name으로 바꾸는 게 좋다)
              - Popen 실행 후 stdout/stderr를 캡처하지 않으므로 로그는 별도 처리 필요
        """
        with self._lock:
            #이미 실행한 프로세스가 있고 해당 프로세스가 아직 작동중일 때
            if self._proc is not None and self._proc.poll() is None:    #poll(): 실행중일 떄 > None
                return "이미 학습이 실행 중입니다."

            def norm(p: str) -> str:
                #isabs() 통해  p가 절대경로/상대경로인지 확인 > 상대령로라면 project_root/p로 반환
                return p if os.path.isabs(p) else os.path.join(self.project_root, p)

            data_abs = norm(data_yaml)
            #strip()으로 경로 앞/뒤 공백 제거
            model_arg = model_pt.strip() if model_pt else "yolov8n-seg.pt"

            #YOLO CLI 명령어 생성
            cmd = [
                self.yolo_cli,
                task, "train",
                f"data={data_abs}",
                f"model={model_arg}",
                f"imgsz={int(imgsz)}",
                f"epochs={int(epochs)}",
                f"batch={int(batch)}",
                "workers=0", #RAM 절약, workers는 데이터 읽는 프로세스 수
                f"lr0={lr0}",
                f"device={device_str}",
                #epoch 10마다 저장
                f"save_period=10",
                f"project={os.path.join(self.project_root, 'runs', task)}",
                f"name=demo_exp",
            ]
            #백그라운드 실행, 로그는 runs 폴더로만 확인
            #cwd(current working directory): subprocess 실행 기준 디렉토리
            self._proc = subprocess.Popen(cmd, cwd=self.project_root)

        return f"학습 시작: {' '.join(cmd)}"

    def stop_train(self):
        """
            [학습 중지 버튼]
            실행 중인 YOLO 학습 subprocess를 종료한다.

            처리 흐름:
              1) lock으로 상태 보호
              2) _proc가 없으면 종료할 프로세스 없음
              3) poll()로 이미 종료되었으면 _proc 정리
              4) terminate(SIGTERM) 시도 후 wait(timeout=5)
              5) terminate로 종료되지 않으면 kill(SIGKILL)
              6) _proc = None으로 정리

            Returns:
                사용자에게 표시할 상태 메시지
        """
        #with: 자원을 획득/사용/소멸시킬 때 사용
        with self._lock:    #acquire
            if self._proc is None:
                return "실행 중인 학습 프로세스가 없습니다."

            if self._proc.poll() is not None:
                self._proc = None
                return "학습 프로세스는 이미 종료된 상태입니다."

            try:            #execute
                self._proc.terminate()  # SIGTERM, 종료신호를 보내 정상 종료를 유도
                self._proc.wait(timeout=5)
            except Exception:
                # terminate로 안 죽으면 kill, 강제종료신호 보냄.즉시 중단
                self._proc.kill()
            finally:        #release
                self._proc = None

            return "학습이 강제 종료되었습니다."

    def is_running(self) -> bool:
        """
            [실행 상태 확인]
            현재 subprocess가 존재하고, poll()이 None이면 실행 중으로 판단한다.

            Returns:
                True면 실행 중, False면 미실행/종료 상태
        """
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def get_latest_run_dir(self, task: str) -> str:
        """
            [최신 run 디렉토리 탐색]
            runs/<task>/ 하위의 디렉토리들 중 수정시간(mtime)이 가장 최신인 폴더를 반환한다.

            Args:
                task: "detect" / "segment" 등

            Returns:
                최신 run 디렉토리 절대경로
                - 없으면 "" 반환

            주의:
              - "최신" 기준은 폴더 자체의 mtime이며,
                실제로는 weights/results.csv mtime 기준이 더 정확할 수 있다.
        """
        base = os.path.join(self.project_root, "runs", task)
        if not os.path.isdir(base):
            return ""
        # base 아래 run 폴더들 중 mtime 최신
        cands = []
        for name in os.listdir(base):       #base 폴더 안 모든 파일/폴더 이름 가져옴
            d = os.path.join(base, name)    #base/name 주소를 절대경로로 바꿈
            if os.path.isdir(d):
                cands.append(d)             #cands는 실험결과 경로 리스트만 남음
        if not cands:
            return ""
        #os.path.getmtime(p): 마지막 수정시간, sort 통해 내림차순으로 정렬
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        #제일 최신파일 return
        return cands[0]

def run_epoch_eval_manual(
        weights_dir,
        infer_img_dir,
        imgsz,
        conf_thres,
        iou_thres,
        device="0",
        out_dir=None,
):
    """
        [수동 epoch 평가 실행]
        weights_dir 내 epoch*.pt 파일들을 대상으로,
        infer_img_dir 이미지 폴더에서 추론을 수행해 epoch별 conf 리포트(txt)를 생성한다.

        사용 의도:
          - 학습이 끝난 후(또는 학습 중간) weights 폴더를 지정하고
            특정 검증 이미지 폴더로 epoch별 성능(conf 평균)을 빠르게 비교

        처리 흐름:
          1) FileExplorer 입력값들을 normalize_path_from_explorer로 문자열 경로화
          2) weights_dir / infer_img_dir 유효성 검사
          3) run_epoch_reports_in_weights_dir 호출:
               - epoch*.pt 스캔
               - 각 pt에 대해 evaluate_one_weight 수행
               - epochXX_conf.txt 형태의 리포트 생성
          4) 생성된 txt 목록을 로그로 합쳐 반환

        Args:
            weights_dir: weights 폴더(Gradio FileExplorer 값일 수 있음)
            infer_img_dir: 평가용 이미지 폴더(Gradio FileExplorer 값일 수 있음)
            imgsz: 추론 이미지 사이즈
            conf_thres: confidence threshold
            iou_thres: iou threshold
            device: "0" 등 (YOLO predict device 인자)
            out_dir: 리포트 txt 저장 폴더(없으면 weights_dir)

        Returns:
            상태 로그 문자열
            - 성공: "[OK] ..." + 생성된 txt 경로들
            - 실패: "[ERROR]/[WARN] ..." 메시지
    """
    weights_dir = normalize_path_from_explorer(weights_dir)
    infer_img_dir = normalize_path_from_explorer(infer_img_dir)
    out_dir = normalize_path_from_explorer(out_dir)

    #빈 문자열이거나 선택한 경로가 폴더가 아닐 때
    if not weights_dir or not os.path.isdir(weights_dir):
        return f"[ERROR] weights_dir가 유효하지 않습니다: {weights_dir}"

    if not infer_img_dir or not os.path.isdir(infer_img_dir):
        return f"[ERROR] 평가 이미지 폴더가 유효하지 않습니다: {infer_img_dir}"

    #epoch.eval.py에서 import한 함수
    txts = run_epoch_reports_in_weights_dir(
        weights_dir=weights_dir,
        img_dir=infer_img_dir,
        out_dir=out_dir if out_dir else weights_dir,
        imgsz=int(imgsz),
        conf_thres=float(conf_thres),
        iou_thres=float(iou_thres),
        device=str(device),
    )

    if not txts:
        return f"[WARN] epoch*.pt 파일을 찾지 못했습니다: {weights_dir}"

    return "[OK] epoch별 평가 완료\n" + "\n".join(txts)
