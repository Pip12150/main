import os
import csv
import gradio as gr
from core.config import PROJECT_ROOT

"""
1._build_runs_map(): results.csv가 있는 demo_exp목록 골라냄 / Gradio 드롭다운에서 가장 최신의 csv파일이 화면에 기본값으로 뜨도록 함
2._on_run_change(): 사용자가 드롭다운 변경 시 선택한 것에 해당되는 results.csv 파일 경로를 찾아서 반환
3._file_to_path(): Gradio gr.File 입력값이 여러 형태로 들어오는 것을 통일(파일 경로 문자열)
4.find_latest_results_csv(): 디렉토리를 재귀 탐색하여 results.csv 파일 중 수정시간이 가장 최신인 파일 반환
5.read_results_csv(): csv파일을 읽어서 딕셔너리 형태를 감싼 리스트 형태로 반환 
6.ensure_epoch(): read_results_csv()로 읽은 데이터에 epoch 칼럼이 없다면 row 순서를 epoch로 추가
"""

def _build_runs_map(task_name: str):
    """
        [runs 목록 스캔 + 드롭다운 구성]
        PROJECT_ROOT/runs/<task_name>/ 하위의 실행(run) 폴더들을 훑어서,
        각 run_dir 안에 results.csv가 존재하는 경우만 수집한다.

        목적:
          - UI 드롭다운에 run_name 목록을 "최근 수정된 results.csv 순"으로 보여주기
          - 선택된 run_name -> results.csv 경로로 매핑되는 runs_map을 함께 반환
          - 기본값은 가장 최근 results.csv를 가진 run으로 지정

        입력:
          task_name: Ultralytics task 폴더명 (예: "detect", "segment", "pose")

        처리:
          1) base = PROJECT_ROOT/runs/task_name
          2) base 하위의 run_name 디렉토리 순회
          3) run_dir/results.csv 존재 시 runs_map[run_name] = csv_path 등록
          4) choices를 results.csv 수정시간(getmtime) 기준 내림차순 정렬
          5) 가장 최신 run을 default로 지정

        반환:
          - dropdown_update: gr.update(choices=[...], value=default_run)
          - runs_map: {run_name: results.csv_path}
          - default_csv: default_run에 해당하는 results.csv 경로(없으면 "")

        주의:
          - base 경로가 없거나 results.csv가 하나도 없으면
            choices는 빈 리스트, default_run은 None, default_csv는 ""가 된다.
          - getmtime은 파일 수정시간 기준이라 "복사/수정"에 따라 순서가 바뀔 수 있다.
    """
    base = os.path.join(PROJECT_ROOT, "runs", task_name) #task_name: segmentation / detection
    runs_map = {}

    #base가 폴더일 때:
    if os.path.isdir(base):
        for run_name in os.listdir(base): #base 폴더 내용을 리스트 형태로 반환(파일명/폴더명만 가져옴!!)
            run_dir = os.path.join(base, run_name) #demo_exp 경로
            csv_path = os.path.join(run_dir, "results.csv")  #results.csv파일 경로
            # run_dir이 폴더이고 csv_path가 존재한다면 runs_map 딕셔너리에 추가
            if os.path.isdir(run_dir) and os.path.exists(csv_path):
                runs_map[run_name] = csv_path
    #key는 demo_exp파일>value는 csv경로 / 내림차순으로 정렬
    #key=lambda n:정렬의 기준, 별도의 계산 기준으로 정렬 >> run_map[n]에 해당하는 value 경로의 마지막 수정일 받음,
    choices = sorted(
        runs_map.keys(), #키 값들을 받아옴
        key=lambda n: os.path.getmtime(runs_map[n]), #key값:정렬 기준값/lambda n...함수를 key에 전달(lambda는 일회용 함수)
        reverse=True
    )
    #choices에는 runs_map의 키값들만 있음

    #choices 리스트에서 가장 최근 demo_exp를 기본값으로 설정
    default_run = choices[0] if choices else None
    #default_csv = runs_map 딕셔너리에서 default_run 값을 가진 딕셔너리의 value값(csv파일 경로)
    default_csv = runs_map[default_run] if default_run else ""

    #gr.update: Gradio Dropdown메뉴/선택창을 새로고침하는 명령어...
    #choices=choices: 드롭다운 메뉴에 나타날 리스트를 최신순으로 정렬된 폴더명들로 채움
    #value=default_run: 메뉴가 처음 뜰 때 기본적으로 가장 최신 결과인 choice[0]이 선택되어 있도록 함(자동으로 최신폴더 로드되어 있도록 함)
    return gr.update(choices=choices, value=default_run), runs_map, default_csv


def _on_run_change(selected_run, runs_map):
    """
        [드롭다운 변경 핸들러]
        사용자가 run_name 드롭다운에서 값을 바꿨을 때,
        runs_map에서 해당 run_name의 results.csv 경로를 찾아 반환한다.

        입력:
          selected_run: 드롭다운에서 선택된 run_name
          runs_map: {run_name: results.csv_path}

        반환:
          - 매칭되면 results.csv 경로 문자열
          - 선택이 없거나 매칭 실패면 "" (빈 문자열)

        주의:
          - runs_map이 dict가 아닐 수도 있어 isinstance 체크를 한다.
            (Gradio state 전달 과정에서 None/다른 타입이 들어오는 상황 방어)
    """
    #매칭이 안되거나 선택을 안했다면...
    if not selected_run:
        return ""
    #runs_map가 딕셔너리 타입이고 selected_run이 runs_map 딕셔너리(key값 중)에 selected_run(드롭다운 메뉴에서 고른 값)이 있을 때
    if isinstance(runs_map, dict) and selected_run in runs_map:
        return runs_map[selected_run] #runs_map에서 selected_run key값에 대한 value 반환
    return "" #실패 시 ""반환

def _file_to_path(f):
    """
        [gr.File -> path 변환 유틸]
        Gradio 버전에 따라 gr.File 입력값이 다음 형태로 들어올 수 있다:
          - dict: {"name": "/tmp/....", ...}
          - tempfile-like object: .name 속성에 경로가 들어있음
          - None

        입력:
          f: gr.File 컴포넌트에서 전달된 객체

        반환:
          - 파일 경로 문자열(가능하면)
          - 실패 시 "" 반환
    """
    if f is None:
        return ""
    # dict 형태: {"name": "...", ...}
    if isinstance(f, dict):
        return f.get("name", "") or "" #name키에서 경로(value)를 꺼냄
    #tempfile-like object: 진짜 파일은 아니지만, 파이썬이 마치 파일인 것처럼 취급해 주는 객체(RAM위에 임시로 존재)
    return getattr(f, "name", "") or "" #f에 .name이라는 속성이 있는지 확인,없으면 빈 문자열("")반환

def find_latest_results_csv(runs_dir: str) -> str | None:
    """
        [runs_dir 하위에서 최신 results.csv 탐색]
        runs_dir 하위 전체를 재귀적으로 탐색하여 results.csv 후보를 모은 뒤,
        수정시간(getmtime)이 가장 최신인 results.csv 경로를 반환한다.

        입력:
          runs_dir: 탐색 루트 디렉토리 (예: PROJECT_ROOT/runs/segment)

        반환:
          - 최신 results.csv의 전체 경로(str)
          - 후보가 없으면 None

        주의:
          - os.walk 기반이라 runs_dir가 매우 크면 탐색 비용이 커질 수 있다.
          - "최신" 기준은 파일 수정시간이며, 실제 학습 종료 시점과 다를 수 있다.
    """
    candidates = []
    #runs_dir 밑 폴더 및 파일 모두 확인
    for root, _, files in os.walk(runs_dir):
        #results.csv 파일이 있다면 candidates 리스트에 해당 경로 추가
        if "results.csv" in files:
            candidates.append(os.path.join(root, "results.csv"))
    if not candidates:
        return None
    #파일 수정시간순으로 내림차순 정렬
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0] #첫번쨰 인자값만 반환

def read_results_csv(csv_path: str) -> list[dict]:
    """
        [results.csv 로더]
        Ultralytics results.csv를 csv.DictReader로 읽어
        각 row를 dict로 리스트에 담아 반환한다.

        입력:
          csv_path: results.csv 파일 경로

        반환:
          rows: list[dict]
            예) [{"epoch":"0","train/box_loss":"1.23", ...}, {...}, ...]

        주의:
          - csv.DictReader는 모든 값을 기본적으로 "문자열"로 읽는다.
            숫자 계산/plot 전에 float 변환이 필요하다.
          - 파일이 없거나 접근 불가하면 예외가 발생할 수 있으니,
            상위에서 존재 여부 체크를 하는 패턴이 안전하다.
    """
    #rows라는 빈 리스트 생성
    rows: list[dict] = []
    with open(csv_path, "r", newline="") as f: #newline: OS별 다른 줄바꿈 방식 때문에 데이터가 꼬이는 것을 방지
        reader = csv.DictReader(f) #DictReader: 첫줄(헤더)을 읽어서 컬럼명으로 기억, key로 지정
        for r in reader:           #csv파일의 두번째 줄부터 한줄씩 반복해서 읽고, 이때 r은 딕셔너리 형태로 저장됨
            rows.append(r)         #해당 딕셔너리들은 리스트 형태로 저장
    return rows

def ensure_epoch(rows: list[dict]) -> list[dict]:
    """
        [epoch 컬럼 보정]
        results.csv에 'epoch' 컬럼이 없는 경우가 있을 수 있으므로
        첫 row에 epoch 키가 없으면 index를 epoch로 추가해준다.

        입력:
          rows: read_results_csv로 읽은 list[dict]

        반환:
          epoch 키가 보장된 rows(list[dict])

        주의:
          - rows가 비어있으면 그대로 반환
          - epoch가 문자열로 들어오는 경우도 있으므로, 이후 plot에서 int/float 변환 권장
    """
    #rows에 값이 있고, 첫번쨰 딕셔너리 안에 "epoch"값이 없다면...
    if rows and ("epoch" not in rows[0]):
        for i, r in enumerate(rows):
            r["epoch"] = i #for문 돌고 있는 현재 row(dict)에 "epoch"라는 키 값을 새로 추가,값은 현재 인덱스"i"
    return rows
