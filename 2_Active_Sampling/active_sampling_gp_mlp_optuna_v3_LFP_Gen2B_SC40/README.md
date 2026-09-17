# GP + MLP Ensemble + Optuna/CV Stabilization Active Sampling v3

이 버전은 GP/MLP active sampling v2에 **필요 시점 자동 Optuna/CV 안정화**를 추가한 버전입니다.

## 추가된 핵심 기능

- Optuna가 설치되어 있고 데이터 조건이 충족되면 GP hyperparameter tuning 자동 수행
- MLP 전환 조건이 충족되고 PyTorch가 설치되어 있으면 MLP CV/ensemble 후보 평가
- MLP Optuna 조건이 충족되면 MLP hyperparameter tuning 자동 수행
- Tmax GPR Optuna 조건이 충족되면 NoTP 구간 RMSE 최소화 기준으로 Tmax hyperparameter tuning 수행
- CV 성능은 평균만 보지 않고 `stable_score = weighted_mean - penalty × weighted_std`로 평가
- CV 결과에 분류 지표와 함께 Tmax 회귀 지표(MAE/RMSE/R2) 저장
- fold별 metric을 `outputs/cv_fold_metrics.csv`로 저장
- Optuna 수행/스킵 이유를 `outputs/optuna_report.json`으로 저장
- GP/MLP 선택 결과를 `outputs/model_selection_report.json`으로 저장

## 설치

기본 실행:

```bash
pip install numpy pandas scipy scikit-learn
```

Optuna 자동튜닝까지 사용:

```bash
pip install optuna
```

MLP ensemble까지 사용:

```bash
pip install torch
```

Optuna 또는 PyTorch가 없어도 코드는 멈추지 않고 자동으로 해당 기능을 건너뜁니다.

## 실행

```bash
python 0_main_active_sampling.py
```

## 모델 선택 흐름

1. 데이터 로드
2. valid 이산 조합 생성
3. Optuna 조건 확인
4. 조건 충족 시 GP/MLP hyperparameter tuning
5. GP CV 평가
6. MLP 조건 충족 시 MLP CV 평가
7. stable_score 기준으로 GP/MLP 자동 선택
8. 선택된 모델로 candidate pool scoring
9. next_sampling_candidates.csv 저장

## Optuna 조건

`config.py`에서 조정합니다.

```python
ENABLE_OPTUNA_AUTO = True
GP_OPTUNA_MIN_TOTAL_SAMPLES = 224
MLP_OPTUNA_MIN_TOTAL_SAMPLES = 350
TMAX_OPTUNA_MIN_PASS_SAMPLES = 80
```

즉, 초기 168개 수준에서는 Optuna가 자동으로 스킵되고, 데이터가 쌓이면 고려됩니다.

## CV 안정화

기본 metric 우선순위는 다음입니다.

```python
MODEL_SELECTION_WEIGHTS = {
    "tp_recall": 0.70,
    "tp_f1": 0.30,
}
```

MLP Optuna는 하드 게이트 + 분류 우선 최적화를 사용합니다.

```python
MLP_OPTUNA_GATE_MIN_TP_RECALL = 0.20
MLP_OPTUNA_GATE_MIN_TP_F1 = 0.15
MLP_OPTUNA_GATE_MAX_TMAX_RMSE = None
MLP_OPTUNA_GATE_MIN_TMAX_R2 = None
MLP_OPTUNA_TMAX_TIEBREAK_WEIGHT = 0.01
```

MLP ensemble 다양성을 높이기 위해 stratified bootstrap을 옵션으로 켤 수 있습니다.

```python
MLP_ENSEMBLE_BOOTSTRAP = True
MLP_BOOTSTRAP_STRATIFIED = True
MLP_BOOTSTRAP_SAMPLE_RATIO = 1.0
```

- `MLP_ENSEMBLE_BOOTSTRAP=True`: 각 ensemble 멤버를 복원추출 샘플로 학습
- `MLP_BOOTSTRAP_STRATIFIED=True`: NoTP/TP 비율을 유지하며 bootstrap 샘플링
- `MLP_BOOTSTRAP_SAMPLE_RATIO`: 원본 대비 bootstrap 샘플 크기 비율

그리고 안정화 점수는 다음 형태입니다.

```text
stable_score = weighted_mean_score - CV_STD_PENALTY × weighted_std_score
```

기본값:

```python
CV_STD_PENALTY = 0.50
```

fold별 성능이 들쭉날쭉한 모델은 평균이 높아도 덜 선택됩니다.

## Sampling bucket

기타 출력 전용 bucket은 넣지 않았습니다. 기본은 아래입니다.

```python
BUCKET_RATIO = {
    "boundary": 0.60,
    "notp_high_tmax": 0.30,
    "uncertainty_sparse": 0.07,
    "random_check": 0.03,
}
```

기타 출력은 sampling 기준에는 쓰지 않지만, `OTHER_REGRESSION_COLS`, `TIME_FEATURE_COLS`에 넣으면 MLP multi-head에서 함께 학습 가능합니다.

## 출력 파일

```text
outputs/next_sampling_candidates.csv
outputs/scored_candidate_pool_preview.csv
outputs/combo_diagnostics.csv
outputs/model_selection_report.json
outputs/optuna_report.json
outputs/cv_fold_metrics.csv
```

## 주의

- Optuna + MLP CV는 시간이 걸릴 수 있습니다.
- 초기 데이터가 적을 때는 GP가 자동 선택될 가능성이 높습니다.
- MLP는 기본적으로 350개 이상이고, 클래스 최소치(기본 30%) 등 자격 조건을 만족할 때만 후보로 평가됩니다.
- `MaxT_Adj`는 NoTP 데이터 기준으로 학습/평가됩니다.
