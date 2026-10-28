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

## 입력 및 DOE 설정

| 입력 | 설계 범위 | 신규 DOE 제외 구간 |
| --- | --- | --- |
| `A_Cell_D` | 8~26 | 16.0 +/- 0.01 |
| `C_Barrier_Thx` | 1~4 | 1.5 +/- 0.01 |
| `E_Barrier_Outer_Thx` | 1~3 | 1.0 +/- 0.01 |
| `F_ThermalResin_Thx` | 0.5~2.5 | 1.0 +/- 0.01 |
| `G_PreForce` | 0~1000 N | 정확히 100 N (half_width=0) |

- 재료 조합은 `B_Barrier_Type=A1`, `D_Barrier_Outer_Type=PU`입니다.
- 학습 DOE 계획은 총 400개, 초기 50개, 추가 350개입니다. 이 값은 계획 설정이며 400개 도달 시 실행을 자동 중단하거나 마지막 배치를 자동 축소하는 기능은 아닙니다.
- 별도 최종 테스트 DOE의 기본 개수는 기존 100개(경계 70개 + 분포 확인 30개)를 유지합니다.
- Interaction 범위는 원변수 범위의 곱으로 자동 계산됩니다. 현재 `CellD_x_Resin`은 4~65, `Barrier_x_Outer`는 1~12입니다.
- Cell_D 샘플링/최적화 구간은 8~12.5, 12.5~17, 17~21.5, 21.5~26이며, 마지막 구간은 상한 26을 포함합니다.

- `G_PreForce`는 셀과 barrier를 적층한 모듈/팩의 체결 하중이며, 단위는 N입니다.
- 설계 범위는 `0~1000 N`이고, 별도 간격 제한은 없습니다. 신규 후보 생성 시 정확히 `100 N`인 값만 제외합니다.
- 예측 앱 및 명령행 직접 예측의 기본 연속형 입력값은 [config.py](config.py)의 `EXCLUDED_REFERENCE_RANGES`에 지정된 각 변수의 `center`를 사용합니다. 현재 A/C/E/F/G 기본값은 각각 `16 / 1.5 / 1 / 1 / 100`이며, 새 surrogate bundle에 함께 저장됩니다. 별도 `CONTINUOUS_DEFAULTS` 설정은 사용하지 않습니다.
- 기본 연속형 입력은 5개이며, 기존 interaction 2개를 포함한 연속형 학습 feature는 7개입니다. `G_PreForce` 관련 interaction은 추가하지 않습니다.
- 학습 및 최종 테스트 CSV에 `G_PreForce` 열과 실제 값이 필요합니다. 누락된 열이나 값을 기본값 `100`으로 자동 대체하지 않습니다.
- 학습 및 최종 테스트 CSV를 읽을 때 열 이름과 문자열 값의 모든 공백을 제거합니다. A/C/E/F 치수는 차이값으로 입력하며, 로드 시 각각 center `16 / 1.5 / 1 / 1`을 더해 절대값으로 복원합니다. G_PreForce와 출력 값, Macro 열은 center를 더하지 않습니다. 원본 CSV 파일은 변경하지 않습니다.
- 이미 절대값으로 변환한 CSV를 학습/최종 테스트 입력으로 넣으면 center가 다시 더해지므로 사용하지 마세요. Interaction은 복원된 절대값으로 계산합니다.
- CSV 배치 후 위 학습 명령을 다시 실행해야 새 입력이 반영된 모델이 생성됩니다. 기존에 저장된 모델은 설정 변경만으로 새 입력을 사용하지 않습니다.
- `G_PreForce`는 표준화된 전체 입력 거리와 후보 중복 검사에 포함됩니다. 버킷별 추가 치수 거리 제약은 기존 4개 치수 변수에 유지합니다.

### 그래프 및 DOE 출력

- Active sampling 산점도는 기본 변수의 모든 쌍을 표시합니다. 현재 출력은 `sampling_all_continuous_pairs_10panel.png`이며, 이 중 4개 패널에 `G_PreForce`가 포함됩니다.
- [make_holdout_distribution_plots.py](make_holdout_distribution_plots.py)는 기본 변수 5개의 분포와 모든 변수 쌍의 산점도 10개를 저장합니다. 기존 첫 산점도 파일명은 유지합니다.
- 두 테스트 DOE 생성기는 기본 변수만 생성한 뒤 interaction을 계산하고, 설정된 범위와 제외 구간을 적용합니다.

### 후보 CSV 출력 형식

`next_sampling_candidates.csv`는 다음 형식으로 저장합니다.

| Excel 열 | CSV 열 이름 | 저장 값 |
| --- | --- | --- |
| A | `A_Cell_D` | 원래 값 - 16.0 |
| B | `B_Barrier_Type` | 원래 값 |
| C | `C_Barrier_Thx` | 원래 값 - 1.5 |
| D | `D_Barrier_Outer_Type` | 원래 값 |
| E | `E_Barrier_Outer_Thx` | 원래 값 - 1.0 |
| F | `F_ThermalResin_Thx` | 원래 값 - 1.0 |
| G | `G_PreForce` | 원래 값 |
| H | 공란 | 헤더와 모든 행을 비움 |
| I 이후 | 나머지 출력 열 | 순위, bucket, 모델, 예측, 점수, interaction 등 기존 값 유지 |

- 차감할 값은 [config.py](config.py)의 `EXCLUDED_REFERENCE_RANGES`에 지정된 각 변수의 `center`를 사용합니다. `G_PreForce`는 차감하지 않습니다.
- 이 변환은 후보 CSV 저장에만 적용합니다. 학습, 예측, 산점도, 후보 pool과 interaction 계산은 계속 절대값을 사용합니다.
- 후보 CSV 형식의 A/C/E/F 차이값은 학습/최종 테스트 로더가 center를 더해 복원합니다. 예측 앱이나 명령행 예측 CSV 입력에는 절대값이 필요합니다. H열은 pandas로 읽으면 `Unnamed: 7`, 공백 제거 로더에서는 `Unnamed:7`로 표시될 수 있습니다.
- [data_loader.py](data_loader.py)의 `load_candidate_data`는 새 출력 형식을 절대값으로 복원하며, 기존 형식은 그대로 읽습니다. 보조 분석 스크립트도 이 함수를 사용합니다. 복원 시에는 내보내기에 사용한 것과 동일한 center 설정이 필요합니다.

### 변경 검증

실제 CFD 데이터나 재학습 없이 구성 데이터로 설정 전달, 후보 생성, 예측 입력, DOE 출력, 후보 CSV 열 순서와 차이값/절대값 왕복 변환, 산점도 구성, 최적화 상한을 검사합니다.

```bash
python -m unittest -v test_design_space
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
