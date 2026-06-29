# GP + MLP Ensemble Active Sampling v2

이 패키지는 v1의 GP 기반 active sampling 코드에 MLP ensemble 전환/학습/평가 로직을 추가한 버전입니다.

## 포함 기능

- GP → MLP 전환 조건 판단
- GP와 MLP 성능 비교 후 자동 선택
- MLP shared trunk + multi-head 구조
- MLP ensemble 학습
- ensemble disagreement 기반 `clf_uncertainty`
- 모델 종류에 따른 `BOUNDARY_WEIGHTS` 자동 변경
- MLP 기반 acquisition score 계산

## 설치

기본 실행:

```bash
pip install numpy pandas scipy scikit-learn
```

MLP ensemble까지 사용하려면:

```bash
pip install torch
```

PyTorch가 없으면 `MODEL_MODE = "auto"`에서도 자동으로 GP로 fallback합니다.

## 실행 방법

1. `config.py`에서 실제 CSV 컬럼명, 이산형 변수명, 이산 수준을 수정합니다.

```python
INPUT_CSV = "initial_dataset.csv"
CONTINUOUS_COLS = ["x1", "x2", "x3", "x4"]
DISCRETE_COLS = ["disc1", "disc2", "disc3"]
PASSFAIL_COL = "pass_fail"  # PASS=0, FAIL=1
TMAX_COL = "tmax"
```

2. 실행합니다.

```bash
python main_active_sampling.py
```

3. 결과 파일을 확인합니다.

```text
outputs/next_sampling_candidates.csv
outputs/scored_candidate_pool_preview.csv
outputs/combo_diagnostics.csv
outputs/model_selection_report.json
```

## 모델 선택 설정

`config.py`의 `MODEL_MODE`를 사용합니다.

```python
MODEL_MODE = "auto"
```

- `"gp"`: GP 강제 사용
- `"mlp"`: MLP 조건 만족 시 MLP 강제 사용, 불만족 시 GP fallback
- `"auto"`: MLP 조건 만족 시 GP와 MLP CV 성능 비교 후 선택

## MLP 전환 조건

기본값은 다음과 같습니다.

```python
MLP_MIN_TOTAL_SAMPLES = 350
MLP_MIN_PASS_SAMPLES = 40
MLP_MIN_FAIL_SAMPLES = 40
MLP_MIN_SAMPLES_PER_COMBO = 8
```

즉, 초기 168~224개 수준에서는 대부분 GP가 선택되고, 데이터가 충분히 쌓인 뒤 MLP가 후보로 평가됩니다.

## GP vs MLP 비교 기준

기본적으로 FAIL recall을 가장 중요하게 봅니다.

```python
MODEL_SELECTION_WEIGHTS = {
    "fail_recall": 0.70,
    "fail_f1": 0.30,
}
```

MLP가 GP보다 `MLP_SELECTION_MARGIN` 이상 좋아야 자동 선택됩니다.

## BOUNDARY_WEIGHTS 차이

GP 사용 시:

```python
BOUNDARY_WEIGHTS_GP = {
    "boundary": 0.70,
    "clf_uncertainty": 0.00,
    "local_sparsity": 0.15,
    "combo_priority": 0.15,
}
```

MLP ensemble 사용 시:

```python
BOUNDARY_WEIGHTS_MLP = {
    "boundary": 0.55,
    "clf_uncertainty": 0.15,
    "local_sparsity": 0.15,
    "combo_priority": 0.15,
}
```

MLP의 `clf_uncertainty`는 ensemble 모델들의 `p_fail` 예측 표준편차입니다.

## Optional multi-head outputs

기타 회귀 출력이나 시계열 압축 feature는 sampling bucket에는 넣지 않지만, MLP 학습에는 같이 넣을 수 있습니다.

```python
OTHER_REGRESSION_COLS = ["energy", "loc_x", "loc_y", "loc_z"]
TIME_FEATURE_COLS = ["temp_feature_1", "temp_feature_2"]
```

이 컬럼들은 MLP의 optional extra regression head에서 학습됩니다.
