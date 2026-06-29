# GP Active Sampling MVP

이 코드는 CFD 기반 PASS/FAIL + Tmax 중심 active sampling을 위한 1차 MVP입니다.

## 핵심 전략

- 현재 데이터: 28개 valid 이산 조합 × 각 조합별 연속형 4D LHS 6개 = 168개
- 목표: 현재 수준 기준 총 636개
- PASS/FAIL label:
  - PASS = 0
  - FAIL = 1
- 모델:
  - GPC_PASSFAIL: 전체 데이터 사용
  - GPR_Tmax_given_PASS: PASS 데이터만 사용
- Candidate pool:
  - 출력값 없는 입력 후보만 생성
  - 이산 constraint 만족
  - 기준값 ±0.01 구간 제외
- Sampling:
  - boundary
  - PASS-high-Tmax
  - uncertainty/sparse
  - random check
- 이산 조합별 완전 균등분배는 강제하지 않음
  - minimum floor와 upper cap만 적용

## 설치

```bash
pip install numpy pandas scipy scikit-learn
```

## 사용 순서

1. `config.py` 수정
   - `INPUT_CSV`
   - `CONTINUOUS_COLS`
   - `DISCRETE_COLS`
   - `PASSFAIL_COL`
   - `TMAX_COL`
   - `DISCRETE_LEVELS`

2. CSV 파일 준비

필수 컬럼 예시:

```text
x1,x2,x3,x4,disc1,disc2,disc3,pass_fail,tmax
```

3. 실행

```bash
python main_active_sampling.py
```

4. 결과 확인

```text
outputs/next_sampling_candidates.csv
outputs/scored_candidate_pool_preview.csv
outputs/combo_diagnostics.csv
```

## 주의

- 기존 labeled CFD 데이터는 기준값 ±0.01 제외 필터를 적용하지 않습니다.
- 제외 필터는 신규 candidate pool에만 적용됩니다.
- `GPR_Tmax_given_PASS`는 PASS 데이터만 사용합니다.
- FAIL 케이스의 Tmax는 학습하지 않습니다.
- 처음에는 Optuna를 넣지 않았습니다. 1차 MVP가 정상 작동한 뒤 Optuna를 붙이는 것을 추천합니다.
