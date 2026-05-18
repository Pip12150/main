
## 1. 이미지 뷰어 - 서버
[FACT]
- 용도: 기존 배포한 모델이 추론한 결과를 확인하기 위함

[STEP]
- 서버설정: Gradio MLOps가 동작하는 서버에서 원본 이미지, 추론결과 txt가 존재하는 경로 지정
- 사용법
- 1) 우측상단 드롭다운 버튼
- 2) 원본 이미지 저장된 경로 지정
- 3) 추론결과 txt 파일 저장된 경로 지정
- 4) 서버로 설정 버튼 클릭
- 지정된 경로의 원본 이미지 자동 출력
- 5) 추론결과 불러오기
- 지정된 경로의 원본 이미지와 동일한 txt 파일을 바탕으로 추론결과를 그림

[NOTE]
- # 이미지 뷰어


## 1. 이미지 뷰어 - 서버

## 1. 이미지 뷰어 - 로컬
[FACT]
- 용도: 기존 배포한 모델이 추론한 결과를 확인하기 위함

[STEP]
- 로컬설정: 사용자 PC에서 원본 이미지, 추론결과 txt가 존재하는 경로 지정
- 사용법
- 1) 우측상단 드롭다운 버튼
- 2) 원본 이미지 업로드
- 3) 추론결과 txt 파일 업로드
- 4) 로컬로 설정 버튼 클릭
- 지정된 경로의 원본 이미지 자동 출력
- 5) 추론결과 불러오기
- 지정된 경로의 원본 이미지와 동일한 txt 파일을 바탕으로 추론결과를 그림


## 1. 이미지 뷰어 - 로컬
[FACT]
- # Dataset 설정


## 1. 이미지 뷰어 - 로컬

## 1. 이미지 뷰어 - 로컬

## 2. Dataset 설정 – 로컬 데이터 업로드
[FACT]
- core 패키지 config.py 설정파일로 수정 가능

[STEP]
- 용도: 로컬에서 작업한 데이터셋을 서버 내 지정된 경로로 업로드
- 사용법
- 1) 저장 폴더명 지정
- 2) 이미지 업로드
- 3) txt 업로드
- 4) 서버로 업로드 저장
- Default 저장 경로: seonge_gradio/test_yolo_project/datasets_for_labeling/<폴더명>/images,labels 저장

[NOTE]
- UPLOAD_ROOT = "/home/gpuadmin/seongje_gradio2/test_yolo_project"
- UPLOAD_NEWDATASET_ROOT = f"{UPLOAD_ROOT}/datasets_for_labeling"


## 2. Dataset 설정 – 로컬 데이터 업로드

## 2. Dataset 설정
[FACT]
- 용도: 모델 train을 위한 학습 데이터셋 구축

[STEP]
- 동작방식: 로컬에서 작업한 데이터셋을 서버 내 지정된 경로로 업로드
- 사용자의 설정에 따라, 기존 데이터셋과 업로드한 신규 데이터셋을 결합 또는 신규 데이터셋 만 활용
- 사용법
- 1) 기존 데이터셋 활용 여부 체크 (Yes or No)
- 2) (Yes일 경우) 기존 데이터셋 경로 설정


## 2. Dataset 설정

## 2. Dataset 설정
[STEP]
- 가정: 로컬에서 작업한 학습데이터를 업로드 완료함
- 사용법
- 3) 이번 데이터셋 경로 선택

[NOTE]
- 신규 이미지 목록 확인


## 신규 데이터셋 목록

## 신규 데이터셋 목록

## 2. Dataset 설정
[STEP]
- 가정: 로컬에서 작업한 학습데이터를 업로드 완료함
- 사용법
- 4) 새로운 데이터셋 저장 경로 설정
- 5) 새로운 데이터셋 폴더명 지정
- 6) 신규 데이터셋 저장 폴더 생성/확인


## 2. Dataset 설정

## 2. Dataset 설정
[STEP]
- 가정: 로컬에서 작업한 학습데이터를 업로드 완료함
- 사용법
- 7) Train 데이터셋으로 포함 시킬 데이터 체크
- 8) 체크 기준으로 train/val 분할 복사 실행

[NOTE]
- 체크: train / 미체크: val 로 복사됨


## 로그 확인

## 최종 저장 경로 확인

## 최종 저장 경로 확인
[NOTE]
- # Train Monitor


## 최종 저장 경로 확인

## 최종 저장 경로 확인

## 3. Train Monitor
[FACT]
- 용도: 모델 training & 진행상황 확인

[STEP]
- 사용법
- 1) data.yaml 업로드
- 서버내 지정된 경로에 저장됨
- 2) 모델 업로드
- 서버내 지정된 경로에 저장됨
- 3) 학습 파라미터 설정
- 4) 학습 시작
- data.yaml 파일 저장 경로: /home/gpuadmin/seongje_gradio2/test_yolo_project/configs
- 모델 저장 경로: /home/gpuadmin/seongje_gradio2/test_yolo_project/base_model


## 3. Train Monitor

## 3. Train Monitor – Epoch 별 평가
[STEP]
- 용도: Train 시 저장된 Epoch 별 모델의 conf 평가
- 현재 save period를 통해 10 epoch 별로 모델이 저장됨
- 사용법
- 1) 이번 train 시 생성된 모델이 있는 경로 설정
- 2) conf 평가에 사용할 이미지가 있는 경로 설정
- 3) 추론 파라미터 설정
- 4) 선택 경로로 Epoch 평가 실행 버튼

[NOTE]
- 가급적 수정하지 말것


## 3. Train Monitor – Epoch 별 평가

## 3. Train Monitor – Epoch 별 평가
[STEP]
- 오른쪽 그림의 좌측 weights 경로 복사  우측 붙여넣기 후 ‘스캔 & 그래프 업데이트’ 버튼 클릭

[RESULT]
- 왼쪽 그림과 같이 평가 로그가 출력됨


## 3. Train Monitor – Epoch 별 평가

## 3. Train Monitor – Metric
[FACT]
- 용도: 모델 train 평가지표 확인

[RESULT]
- 가장 최근 train한 결과를 자동으로

[NOTE]
- 표출함
- metric / loss로 구성되어 있음


## 3. Train Monitor – Metric
[NOTE]
- # 모델 성능 모니터링


## 3. Train Monitor – Metric

## 3. Train Monitor – Metric

## 4. 모델 성능 모니터링
[FACT]
- 용도: 현재 배포되어 사용중인 모델의 추론 성능을 confidence 기반으로 모니터링

[STEP]
- 사용법
- 1) 집계/갱신 버튼 클릭
- 2) 우측 결과 확인

[NOTE]
- 가정: 모델 추론 시, txt 파일에 conf를 기록함 (구조: class, conf, x1, y1, x2, y2, x3, y3, ...)


## 추론결과 저장 경로: /home/gpuadmin/seongje_gradio2/inf_results

## 일자별 conf 확인
[NOTE]
- 통계의 Conf 산출 방식
- 가정
- ∙ 일자별로 복수개의 이미지를 추론한다
- 방식
- ∙ 각 이미지에 대해 클래스별 추론 Confidence의 평균값을 산출
- ∙ 동일 일자에 대해 이미지별 평균 Confidence를 다시 평균하여 일자 단위 Confidence 지표로 계산


## 일자별 conf 확인

## 4. 모델 성능 모니터링 – 저성능 날짜 선별 및 데이터셋 복사
[FACT]
- 용도: Conf 기준으로 현재 배포된 모델의 성능이 저하된 일자의 추론 결과를 다시 학습데이터로 활용하기 위해 사용

[STEP]
- 사용법
- 1) Conf 기준값 설정
- 2) mean ≤ 기준 날짜 불러오기
- 3) 날짜 선택
- 4) 선택 날짜 복사


## 경로 확인

## 복사할 경로: /home/gpuadmin/seongje_gradio2/test_yolo_project/datasets_for_labeling

## 복사할 경로: /home/gpuadmin/seongje_gradio2/test_yolo_project/datasets_for_labeling

## 4. 모델 성능 모니터링 – 저성능 날짜 선별 및 데이터셋 복사
[FACT]
- 용도: 재학습 데이터로 사용할 데이터를 로컬로 다운로드 하기 위함

[STEP]
- 사용법
- 5) ZIP 다운로드
- 6) 파일 클릭

[NOTE]
- .zip 파일 생성됨
- 다운로드


## 4. 모델 성능 모니터링 – 저성능 날짜 선별 및 데이터셋 복사
[NOTE]
- # Labeling


## 4. 모델 성능 모니터링 – 저성능 날짜 선별 및 데이터셋 복사

## 4. 모델 성능 모니터링 – 저성능 날짜 선별 및 데이터셋 복사

## 5. Labeling
[FACT]
- 용도: 학습데이터셋 생성, 수정을 비롯한 레이블 편집
- 구현된 기능
- 구현필요 기능

[STEP]
- (생성된 json 파일은 /home/gpuadmin/seongje_gradio2 경로에 저장됨)
- edited.json으로만 저장되는데, 이미지와 같은 파일명으로 저장하기

[NOTE]
- 현재는 별도의 레이블링 Tool을 사용하는 것을 권장함
- 다음과 같은 경우에 사용할 수 있음
- 이미지 & json(레이블 파일)이 있는 경우
- 이미지 & txt(추론결과)가 있는 경우
- 사용불가
- 이미지만 있는 경우
- 추론결과 txt 파일로 부터 json 파일 생성
- 현재 읽을 수 있는 json은 coco json 형식
- (X-Anylabeling에서 생성한 json은 읽을 수 없음)
- 신규 레이블 생성
- json을 다시 txt로 변환


## 5. Labeling

## 5. Labeling
[FACT]
- 용도: 학습데이터셋 생성, 수정을 비롯한 레이블 편집

[STEP]
- 사용법
- 1) 이미지 업로드
- 2) 추론결과 txt 업로드
- 3) classes.txt 업로드
- 4) Generate JSON 버튼
- 5) JSON 업로드
- 6) Load Polygon 버튼


## 5. Labeling

## 5. Labeling

## 5. Labeling

## 6. 모델 추론결과 비교
[FACT]
- 용도: 기존모델과 신규모델(재학습한)의 추론 성능을 실제 이미지 추론결과를 바탕으로 육안 비교

[STEP]
- 사용법
- (처음에 ‘새로고침’ 클릭 필수)
- 1) 비교실험 할 원본 이미지 선택
- 2) 기존 모델 경로 선택
- 3) 신규 모델 경로 선택
- 4) 추론 결과 비교 버튼

