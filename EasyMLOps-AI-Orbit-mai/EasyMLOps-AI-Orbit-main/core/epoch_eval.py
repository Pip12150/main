# core/epoch_eval.py
import os
import glob #특정 패턴에 맞는 파일 목록 찾을 때(.pt파일 찾아오기)
import cv2  #이미지 읽기 (실식나 CV 목적으로 한 라이브러리)
import re   #정규표현식 사용(epoch 파일 이름 패턴 매칭)
import numpy as np
from ultralytics import YOLO

"""
코드 전체 flow:
1.run_epoch_reports_in_weights_dir()통하여 weights 폴더 안에 epoch.pt들을 한번에 평가 >> 결과로 epoch.pt 파일 경로 리스트 반환
    -_list_epoch_weights(weights_dir): epoch.pt 파일들 찾아서 파일이름 리스트 반환
    2.write_epoch_conf_report() 통해 epoch(10단위) pt 하나당 txt 하나 생성 >> 결과로 해당 txt 파일 경로 return
        3.evaluate_one_weight() 통해 실제 추론 루프 돌림 >> 결과로 global_mean, per_image return
            -_list_image(img_dir): 이미지 경로 리스트 반환
            -_extract_confs(result): 추론 돌린 이미지의 conf값 추출 후 numpy배열로 반환
            -_sage_mean(arr): 이미지별 conf mean 값 계산(만약 numpy배열 객체가 없거나 길이가 0이면 0.0으로 반환) 
"""

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _list_images(img_dir: str) -> list[str]:
    """
        [유틸] img_dir 하위(재귀)에서 이미지 파일 목록을 수집한다.

        - os.walk로 모든 서브폴더를 탐색
        - IMG_EXTS에 해당하는 확장자만 필터링
        - 정렬된 전체 경로 리스트 반환

        Args:
            img_dir: 이미지 루트 디렉토리

        Returns:
            img_paths: 이미지 파일 전체 경로 리스트(sorted)
    """
    paths = []
    #roots: 현재 탐색중인 디렉토리 경로, dirs(_): 해당 디렉토리의 하위 디렉토리 리스트, files: 현재 디렉토리의 파일 이름 리스트
    for root, _, files in os.walk(img_dir):
        for fn in files:
            #파일 모든 이름 소문자로 변경 후(확장자 통일시키기 위함), IMG_EXTS 튜플에 포함된 확장자명 검색
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, fn)) #path 리스트에 추가
    return sorted(paths) #오름차순으로 경로 반환

def _safe_mean(arr: np.ndarray) -> float:
    #ndarray: n차원 배열 객체, 한자의 데이터 타입만 저장 가능/ 파이썬 list(linked list)와 달리 연속적인 메모리 공간에 저장되어 데이터 접근이 빠름
    #linked_list: 불연속적인 메모리 공간에 저장, 각 노드는 다음 노드의 포인터 가리킴, 데이터 수정이 빠름
    #여기서 arr은 1차원 배열임
    """
        [유틸] 빈 배열/None에 안전한 평균 계산.

        - arr가 None이거나 길이가 0이면 0.0 반환
        - 그 외는 np.mean 결과를 float로 반환

        Args:
            arr: 1D numpy array (예: confidence 목록)

        Returns:
            mean_value: 평균값 (없으면 0.0)
    """
    #arr이 None 이거나 길이가 0일 때
    if arr is None or len(arr) == 0:
        return 0.0
    return float(np.mean(arr))

def _extract_confs(result) -> np.ndarray:
    """
        [유틸] Ultralytics YOLO 예측 결과에서 confidence 배열을 추출한다.

        - detect/segment 공통으로 result.boxes.conf가 존재하면 사용
          (detect/segment 모두 탐지 결과는 result.boxes, conf 값은 results.boxes.conf에 들어있음)
        - 없으면 빈 배열 반환

        Args:
            result: model.predict(...)[0] 형태의 Ultralytics Result 객체

        Returns:
            confs: shape=(N,) float32 numpy array (없으면 빈 배열)
    """
    # detect/segment 공통: boxes.conf가 있으면 그걸 사용
    if result is None or result.boxes is None or result.boxes.conf is None:
        return np.array([], dtype=np.float32)
    return result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    #boxes(bbox 좌표/class id, confidence), result, conf(Pytorch tensor) 는 ultralytics.YOLO의 객체
    #deatach(): Pytorch Tensor 메서드(추론결과 다루기 위함)
    #cpu(): Pytorch Tensor 메서드(Tensor를 GPU>CPU 메모리로 이동,numpy는 GPU tensor를 직접 못 다루기 때문)
    #numpy(): ndarray로 변환
    #>>>정리하자면 result의 conf값을 detach()사용하여 그래프에서 값만 떼어내고 CPU로 옮긴 뒤 numpy 배열로 변환(type은 float32)

def evaluate_one_weight(
    weight_path: str,
    img_dir: str,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5, #intersecntion over Union: 교집합 영역 넓이 / 합집합 영역 넓이
    device: str = "0", #분석 실행할 gpu 지정(첫번째 그래픽 카드 지정)
) -> tuple[float, list[tuple[str, float]]]:
    """
        [단일 weight 평가]
        주어진 weight(.pt)로 img_dir의 모든 이미지를 추론하고,
        각 이미지별 confidence 평균(mean_conf)을 구한 뒤,
        '이미지별 평균(conf)'들의 평균(global_mean)을 계산한다.

        계산 방식:
          1) 이미지별로 탐지된 객체들의 conf 배열을 추출
          2) 해당 이미지의 mean_conf = mean(confs) (객체 없으면 0.0)
          3) global_mean = mean(모든 이미지의 mean_conf)

        Args:
            weight_path: 평가할 YOLO weight 경로(.pt)
            img_dir: 평가용 이미지 루트 디렉토리(재귀 탐색)
            imgsz: 추론 입력 이미지 사이즈
            conf_thres: YOLO predict conf threshold
            iou_thres: YOLO predict iou threshold
            device: 추론 device 문자열(예: "0", "0,1", "cpu")

        Returns:
            global_mean:
                전체 이미지에 대해 "이미지별 mean_conf"를 평균낸 값
            per_image:
                [(이미지파일명(basename), mean_conf), ...]
                이미지별 평균 conf 기록
    """

    #모델 weight 불러오기, 해당 가중치 파일을 기반으로 모델 객체 새롭게 생성
    model = YOLO(weight_path)
    #평가할 이미지 리스트 수집
    img_paths = _list_images(img_dir)

    #파일명과 평균conf값 저장
    per_image = []  # [(filename, mean_conf)]
    #평균conf값만 저장
    per_image_means = []  # [mean_conf, mean_conf, ...]

    #img_paths의 이미지 로드
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        #추론 run, [0]은 결과 리스트의 첫번째 값만 사용하기 위함
        #(이미지는 1장씩 불러오므로 어차피 res는 하나의 결과값만을 가진 리스트임. 그러나 리스트를 사용하기 위해 [0]가 필요한 것)
        res = model.predict(
            source=img,
            imgsz=int(imgsz),
            conf=float(conf_thres),
            iou=float(iou_thres),
            device=str(device),
            verbose=False,
        )[0]
        #불러온 이미지들에서 잡힌 객체들의 conf값
        confs = _extract_confs(res)

        # 이미지별 conf mean (객체 없으면 0.0)
        mean_conf = _safe_mean(confs)
        per_image.append((os.path.basename(p), mean_conf))
        per_image_means.append(mean_conf)

    #“기록된 이미지별 평균(conf)”들의 평균
    global_mean = float(np.mean(per_image_means)) if len(per_image_means) > 0 else 0.0

    return global_mean, per_image


def write_epoch_conf_report(
    weight_path: str,
    img_dir: str,
    out_dir: str,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    device: str,
) -> str:
    """
        [리포트 txt 생성]
        weight 하나에 대해 evaluate_one_weight()를 수행하고,
        (epoch명 + global_mean)을 파일명에 포함한 txt 리포트를 생성한다.

        파일명 규칙 예:
          epoch10.pt -> epoch10_0.7677.txt

        파일 내용:
          각 줄에 "이미지파일명 - mean_conf" 기록

        Args:
            weight_path: 평가할 YOLO weight 경로
            img_dir: 평가 이미지 디렉토리
            out_dir: 리포트 txt 저장 디렉토리
            imgsz/conf_thres/iou_thres/device: evaluate_one_weight에 전달할 추론 파라미터

        Returns:
            out_path: 생성된 txt 파일의 전체 경로
    """
    #evaluate_one_weight() 함수 불러와서 global_mean, per_image 변수에 각각 저장
    #(함수의 리턴값: global_mean, per_image)
    global_mean, per_image = evaluate_one_weight(
        weight_path=weight_path,
        img_dir=img_dir,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device,
    )
    #basename 통해 경로전체에서 가장 마지막 파일명만 추출
    #splitext 통해 파일명과 확장자명을 분리하여 튜플로 만들고 첫번째 값만 가져옴
    base = os.path.splitext(os.path.basename(weight_path))[0]
    #"{base파일이름}_{모든이미지 conf값(global_mean)}.txt"
    fname = f"{base}_{global_mean:.4f}.txt"
    #txt저장 디렉토리 생성
    os.makedirs(out_dir, exist_ok=True)  #해당 폴더가 이미 존재하면 그냥 넘어가고, 없으면 생성
    # False이면 해당 폴더 존재 시 에러발생
    out_path = os.path.join(out_dir, fname)
    #epoch별 conf값 파일안에 들어갈 내용작성(이미지 별 평균conf값)
    lines = [f"{img_name} - {m:.4f}" for img_name, m in per_image]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path

def _list_epoch_weights(weights_dir: str) -> list[str]:
    """
        [유틸] weights_dir에서 epoch weight 파일만 골라 리스트로 반환한다.

        대상:
          - epoch10.pt, epoch_10.pt, epoch010.pt 등 'epoch[_]?숫자.pt' 패턴

        제외:
          - best.pt, last.pt (원하면 옵션으로 포함하도록 확장 가능)

        정렬:
          - 현재 코드는 out.sort()로 경로 문자열 정렬만 수행 중
          - 숫자 기반 정렬을 원하면 _epoch_num을 key로 사용해야 함(아래 참고)

        Args:
            weights_dir: weights(.pt)들이 있는 디렉토리

        Returns:
            epoch_weight_paths: epoch 패턴에 해당하는 weight 경로 리스트
    """
    # epoch*.pt 검색 + best.pt/last.pt는 제외(원하면 포함 옵션도 가능)
    #glob.glob: 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환 >> .pt파일 모두 가져옴
    cands = glob.glob(os.path.join(weights_dir, "*.pt"))
    out = []
    for p in cands:
        bn = os.path.basename(p).lower()
        #best.pt, last.pt인 경우 넘어감
        if bn in ("best.pt", "last.pt"):
            continue
        # epoch10.pt, epoch_10.pt, epoch010.pt 등 허용
        if re.search(r"epoch[_]?\d+\.pt$", bn):
            out.append(p)
    # 숫자 기준 정렬(오름차순)
    def _epoch_num(path: str) -> int:
        #(\d+) 통해서 해당 부분만 따로 사용할 것임을 표시
        m = re.search(r"epoch[_]?(\d+)\.pt$", os.path.basename(path).lower())
        #group(1) 통해 m에서 첫번쨰 괄호에 든 내용(숫자 부분)만 가져와서 return/숫자가 없다면 10**9같은 큰 숫자 return
        return int(m.group(1)) if m else 10**9
    #현재는 문자열 기준 정렬인 상태 / 숫자 기준 정렬을 원하면 out.sort(key = _epoch_num)처럼 작성
    out.sort()
    return out

def run_epoch_reports_in_weights_dir(
    weights_dir: str,
    img_dir: str,
    out_dir: str | None = None,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    device: str = "0",
) -> list[str]:
    """
        [배치 실행] weights_dir 내 모든 epoch*.pt에 대해 리포트를 생성한다.

        동작:
          1) out_dir가 비어있으면 weights_dir를 기본 out_dir로 사용
          2) _list_epoch_weights로 epoch weight 목록 수집
          3) 각 weight마다 write_epoch_conf_report 실행
          4) 생성된 txt 경로 리스트 반환

        Args:
            weights_dir: epoch weight들이 있는 디렉토리
            img_dir: 평가 이미지 디렉토리
            out_dir: 결과 txt 저장 폴더(None이면 weights_dir)
            imgsz/conf_thres/iou_thres/device: 추론 파라미터

        Returns:
            outputs: 생성된 report txt 경로 리스트
    """
    if out_dir is None or not str(out_dir).strip():
        out_dir = weights_dir

    #epoch_파일명 리스트 수집
    weight_paths = _list_epoch_weights(weights_dir)

    outputs = []
    for wp in weight_paths:
        out_txt = write_epoch_conf_report(
            weight_path=wp,
            img_dir=img_dir,
            out_dir=out_dir,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            device=device,
        )
        outputs.append(out_txt)

    return outputs

