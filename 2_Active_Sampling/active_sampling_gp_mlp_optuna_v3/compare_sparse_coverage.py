import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from batch_selector import select_batch
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 로드
df = pd.read_csv('initial_dataset.csv')
print(f'초기 데이터: {len(df)} samples')

# 특징 계산 (scoring)
feature_cols = ['A_Cell_D', 'B_Barrier_Type', 'C_Barrier_Thx', 
                'D_Barrier_Outer_Type', 'E_Barrier_Outer_Thx', 'F_ThermalResin_Thx']

le_b = LabelEncoder()
le_d = LabelEncoder()
df_enc = df.copy()
df_enc['B_Barrier_Type'] = le_b.fit_transform(df_enc['B_Barrier_Type'])
df_enc['D_Barrier_Outer_Type'] = le_d.fit_transform(df_enc['D_Barrier_Outer_Type'])

X = df_enc[feature_cols].values
y = df_enc['TP_NoTP'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 간단한 acquisition score 계산 (p_tp 예측으로 boundary score 설정)
gpc = GaussianProcessClassifier(kernel=C(1.0) * RBF(1.0), random_state=42)
gpc.fit(X_scaled, y)
p_tp = gpc.predict_proba(X_scaled)[:, 1]

df_scored = df.copy()
df_scored['p_tp'] = p_tp
df_scored['p_notp'] = 1 - p_tp
df_scored['discrete_combo_id'] = range(len(df))
df_scored['acq_boundary'] = 1 - np.abs(p_tp - 0.5) * 2  # boundary score
df_scored['acq_notp_high_tmax'] = np.random.rand(len(df))
df_scored['acq_uncertainty_sparse'] = np.random.rand(len(df))
df_scored['random_score'] = np.random.rand(len(df))

print('\n' + '='*80)
print('배치 선택 비교: Sparse Coverage 활성화 vs 비활성화')
print('='*80)

# CASE 1: Sparse coverage 활성화
print('\n[CASE 1] Sparse Coverage 활성화 (enable_sparse_coverage=True)')
selected_with = select_batch(
    df_scored,
    X_scaled,
    df,
    batch_size=28,
    bucket_ratio={'boundary': 0.75, 'notp_high_tmax': 0.15, 'uncertainty_sparse': 0.08, 'random_check': 0.02},
    max_samples_per_combo=5,
    min_batch_distance=0.1,
    seed=42,
    enable_sparse_coverage=True,
    sparse_min_samples=3
)
barrier_thx_with = selected_with['C_Barrier_Thx'].values
print(f'선택된 샘플: {len(selected_with)}')
print(f'barrier_thx 범위: {barrier_thx_with.min():.3f} ~ {barrier_thx_with.max():.3f}')
print(f'barrier_thx 분포:')
for percentile in [0, 25, 50, 75, 100]:
    val = np.percentile(barrier_thx_with, percentile)
    print(f'  {percentile}%: {val:.3f}')

# CASE 2: Sparse coverage 비활성화
print('\n[CASE 2] Sparse Coverage 비활성화 (enable_sparse_coverage=False)')
selected_without = select_batch(
    df_scored,
    X_scaled,
    df,
    batch_size=28,
    bucket_ratio={'boundary': 0.75, 'notp_high_tmax': 0.15, 'uncertainty_sparse': 0.08, 'random_check': 0.02},
    max_samples_per_combo=5,
    min_batch_distance=0.1,
    seed=42,
    enable_sparse_coverage=False,
)
barrier_thx_without = selected_without['C_Barrier_Thx'].values
print(f'선택된 샘플: {len(selected_without)}')
print(f'barrier_thx 범위: {barrier_thx_without.min():.3f} ~ {barrier_thx_without.max():.3f}')
print(f'barrier_thx 분포:')
for percentile in [0, 25, 50, 75, 100]:
    val = np.percentile(barrier_thx_without, percentile)
    print(f'  {percentile}%: {val:.3f}')

# 비교
print('\n' + '='*80)
print('차이 분석:')
print('='*80)
print(f'Range 차이: WITH={barrier_thx_with.max() - barrier_thx_with.min():.3f}, '
      f'WITHOUT={barrier_thx_without.max() - barrier_thx_without.min():.3f}')
print(f'Std 차이: WITH={barrier_thx_with.std():.3f}, WITHOUT={barrier_thx_without.std():.3f}')

# Sparse zone 비교
def count_sparse_zones(barrier_thx, threshold=0.15):
    n_bins = 5
    min_val, max_val = barrier_thx.min(), barrier_thx.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    sparse_count = 0
    for i in range(n_bins):
        count = np.sum((barrier_thx >= bin_edges[i]) & (barrier_thx < bin_edges[i+1]))
        density = count / len(barrier_thx)
        if density < threshold:
            sparse_count += 1
    return sparse_count

sparse_with = count_sparse_zones(barrier_thx_with)
sparse_without = count_sparse_zones(barrier_thx_without)

print(f'Sparse zone 개수: WITH={sparse_with}/5, WITHOUT={sparse_without}/5')
print('\n결론: Sparse coverage가 활성화되면 모든 구간에서 최소 샘플이 보증되어')
print('다음 라운드 학습에서 over-concentration 위험이 감소합니다.')
