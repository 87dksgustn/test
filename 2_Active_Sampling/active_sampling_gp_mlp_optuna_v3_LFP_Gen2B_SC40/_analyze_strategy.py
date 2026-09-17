import pandas as pd
import numpy as np

# Itr_11 데이터 (훈련) + Itr_12 outputs 사용
df = pd.read_csv('Itr_11_dataset.csv')
pool = pd.read_csv('outputs/Itr_12/scored_candidate_pool_preview.csv')
current = pd.read_csv('outputs/Itr_12/next_sampling_candidates.csv')
adjusted = pd.read_csv('outputs/Itr_12/next_sampling_candidates_adjusted.csv')

print('=== Itr_11 훈련 데이터 분포 ===')
print(f'총 샘플: {len(df)}')
tp_count = (df['TP_NoTP']=='TP').sum()
notp_count = (df['TP_NoTP']=='NoTP').sum()
print(f'TP: {tp_count}, NoTP: {notp_count}')

def get_zone(x):
    if x <= 1.0: return 'TP_zone'
    elif x <= 1.5: return 'Boundary'
    else: return 'NoTP_zone'

df['barrier_zone'] = df['C_Barrier_Thx'].apply(get_zone)
print('\n영역별 TP_NoTP 분포:')
cross = df.groupby('barrier_zone')['TP_NoTP'].value_counts().unstack(fill_value=0)
print(cross)

print('\n' + '='*50)
print('=== 두 샘플링 전략 비교 ===')
print('='*50)
print(f'기존 전략: {len(current)}개')
print(f'취약 집중: {len(adjusted)}개')

for name, samples in [('기존', current), ('취약집중', adjusted)]:
    samples = samples.copy()
    samples['barrier_zone'] = samples['C_Barrier_Thx'].apply(get_zone)
    print(f'\n{name} - 영역별:')
    print(samples['barrier_zone'].value_counts().sort_index())

print('\n' + '='*50)
print('=== Combo 다양성 비교 ===')
print('='*50)
for name, samples in [('기존', current), ('취약집중', adjusted)]:
    n_combos = samples['discrete_combo_id'].nunique()
    probs = samples['discrete_combo_id'].value_counts(normalize=True)
    combo_entropy = -sum(probs * np.log2(probs))
    print(f'{name}: {n_combos}개 combo, 엔트로피={combo_entropy:.2f}')

# 권장 하이브리드 전략 시뮬레이션
print('\n' + '='*50)
print('=== 권장: 하이브리드 전략 ===')
print('='*50)
print("""
[현재 전략의 문제점]
- 기존: combo 다양성 좋지만 취약 combo(013, 008, 010, 002) 누락
- 취약집중: 취약 combo 보강되지만 기존 combo 희석 가능

[권장 하이브리드 전략]
1. 기본 버킷 유지 (boundary, uncertainty_sparse, notp_high_tmax)
2. 오분류/취약 combo에 추가 샘플링 (combo_reinforce)
3. random_check로 전체 탐색 유지

[바운더리 정확도 손실 위험 분석]
- 취약 combo 집중 → 해당 combo의 경계면 학습 강화
- 다른 combo 샘플 감소 → 일반화 성능 저하 가능
- 권장: 취약 combo 추가는 전체의 30% 이하로 제한
""")

# 비율 분석
print('\n[현재 비율 분석]')
for name, samples in [('기존', current), ('취약집중', adjusted)]:
    total = len(samples)
    boundary_n = (samples['selected_bucket'] == 'boundary').sum()
    reinforce_n = samples['selected_bucket'].isin(['combo_reinforce', 'misclass_reinforce']).sum()
    print(f'{name}:')
    print(f'  - boundary: {boundary_n}/{total} ({100*boundary_n/total:.0f}%)')
    print(f'  - reinforce: {reinforce_n}/{total} ({100*reinforce_n/total:.0f}%)')
