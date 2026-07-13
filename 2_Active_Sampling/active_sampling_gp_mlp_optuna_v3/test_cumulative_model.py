import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

# 1. 초기 데이터
df_initial = pd.read_csv('initial_dataset.csv')
print(f"Initial dataset: {len(df_initial)} samples")

# 2. 초기 데이터의 barrier_thx 분포
print(f"Initial barrier_thx range: {df_initial['C_Barrier_Thx'].min():.2f} ~ {df_initial['C_Barrier_Thx'].max():.2f}")

# 3. 편향된 데이터셋 생성: barrier_thx 1-2 구간에 집중 (Try_1처럼)
# 전체 샘플 중 70%를 1-2 구간에서, 나머지 30%를 다른 구간에서
np.random.seed(42)
center_samples = df_initial[(df_initial['C_Barrier_Thx'] >= 1.0) & (df_initial['C_Barrier_Thx'] <= 2.0)]
outer_samples = df_initial[(df_initial['C_Barrier_Thx'] < 1.0) | (df_initial['C_Barrier_Thx'] > 2.0)]

# 리샘플링: 1-2 구간은 많이, 그 외는 적게
n_center_target = int(len(df_initial) * 0.7)
n_outer_target = len(df_initial) - n_center_target

center_resampled = center_samples.sample(n=n_center_target, replace=True, random_state=42)
outer_resampled = outer_samples.sample(n=n_outer_target, replace=True, random_state=42)

df_biased = pd.concat([center_resampled, outer_resampled], ignore_index=True)
print(f"\nBiased dataset (70% from 1-2 range): {len(df_biased)} samples")
print(f"Biased barrier_thx distribution:")
print(df_biased['C_Barrier_Thx'].describe().round(3))

# 4. Holdout 데이터
df_holdout = pd.read_csv('outputs/tp_notp_holdout_predictions.csv')

# feature/target 분리 및 인코딩
from sklearn.preprocessing import LabelEncoder

feature_cols = ['A_Cell_D', 'B_Barrier_Type', 'C_Barrier_Thx', 
                'D_Barrier_Outer_Type', 'E_Barrier_Outer_Thx', 'F_ThermalResin_Thx']
le_b = LabelEncoder()
le_d = LabelEncoder()

df_initial_enc = df_initial.copy()
df_initial_enc['B_Barrier_Type'] = le_b.fit_transform(df_initial_enc['B_Barrier_Type'])
df_initial_enc['D_Barrier_Outer_Type'] = le_d.fit_transform(df_initial_enc['D_Barrier_Outer_Type'])

df_biased_enc = df_biased.copy()
df_biased_enc['B_Barrier_Type'] = le_b.transform(df_biased_enc['B_Barrier_Type'])
df_biased_enc['D_Barrier_Outer_Type'] = le_d.transform(df_biased_enc['D_Barrier_Outer_Type'])

df_holdout_enc = df_holdout.copy()
df_holdout_enc['B_Barrier_Type'] = le_b.transform(df_holdout_enc['B_Barrier_Type'])
df_holdout_enc['D_Barrier_Outer_Type'] = le_d.transform(df_holdout_enc['D_Barrier_Outer_Type'])

X_initial = df_initial_enc[feature_cols].values
y_initial = df_initial_enc['TP_NoTP'].values

X_biased = df_biased_enc[feature_cols].values
y_biased = df_biased_enc['TP_NoTP'].values

X_holdout = df_holdout_enc[feature_cols].values
y_holdout = df_holdout_enc['TP_NoTP'].values

# 스케일링
scaler_init = StandardScaler()
X_initial_scaled = scaler_init.fit_transform(X_initial)
X_holdout_scaled_init = scaler_init.transform(X_holdout)

scaler_bias = StandardScaler()
X_biased_scaled = scaler_bias.fit_transform(X_biased)
X_holdout_scaled_bias = scaler_bias.transform(X_holdout)

# 모델 학습 (초기 데이터로)
gpc_initial = GaussianProcessClassifier(kernel=C(1.0) * RBF(1.0), n_restarts_optimizer=10, random_state=42)
gpc_initial.fit(X_initial_scaled, y_initial)

# 모델 학습 (편향된 데이터로)
gpc_biased = GaussianProcessClassifier(kernel=C(1.0) * RBF(1.0), n_restarts_optimizer=10, random_state=42)
gpc_biased.fit(X_biased_scaled, y_biased)

# Holdout 평가
y_pred_initial = gpc_initial.predict(X_holdout_scaled_init)
p_tp_initial = gpc_initial.predict_proba(X_holdout_scaled_init)[:, 1]

y_pred_biased = gpc_biased.predict(X_holdout_scaled_bias)
p_tp_biased = gpc_biased.predict_proba(X_holdout_scaled_bias)[:, 1]

# barrier_thx 구간별 평가
df_holdout_init = df_holdout.copy()
df_holdout_init['barrier_bin'] = pd.cut(df_holdout_init['C_Barrier_Thx'], 
                                         bins=[0,1,2,3], 
                                         labels=['0-1','1-2','2-3'])
df_holdout_init['pred_tp'] = y_pred_initial
df_holdout_init['p_tp'] = p_tp_initial

df_holdout_bias = df_holdout.copy()
df_holdout_bias['barrier_bin'] = pd.cut(df_holdout_bias['C_Barrier_Thx'], 
                                        bins=[0,1,2,3], 
                                        labels=['0-1','1-2','2-3'])
df_holdout_bias['pred_tp'] = y_pred_biased
df_holdout_bias['p_tp'] = p_tp_biased

print("\n" + "="*70)
print("COMPARISON: Initial Model vs Biased (Concentrated) Model")
print("="*70)

print("\n{:<10} {:<15} {:<15} {:<15}".format("Barrier_Thx", "Initial Model", "Biased Model", "Change"))
print("-" * 55)

for b in ['0-1','1-2','2-3']:
    sub_init = df_holdout_init[df_holdout_init['barrier_bin'] == b]
    sub_bias = df_holdout_bias[df_holdout_bias['barrier_bin'] == b]
    
    if sub_init.empty or sub_bias.empty: 
        continue
    
    y_true = sub_init['TP_NoTP'].values
    acc_init = accuracy_score(y_true, sub_init['pred_tp'].values)
    acc_bias = accuracy_score(y_true, sub_bias['pred_tp'].values)
    
    change = acc_bias - acc_init
    sign = '+' if change >= 0 else ''
    print(f"{b:<10} {acc_init:<15.3f} {acc_bias:<15.3f} {sign}{change:.3f}")

print("\n" + "="*70)
print("Biased 모델의 구간별 상세 성능:")
print("="*70)

for b in ['0-1','1-2','2-3']:
    sub = df_holdout_bias[df_holdout_bias['barrier_bin'] == b]
    if sub.empty: 
        continue
    
    y_true = sub['TP_NoTP'].values
    y_pred = sub['pred_tp'].values
    
    tp_count = (y_true == 1).sum()
    notp_count = (y_true == 0).sum()
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{b} (n={len(sub)}, TP={tp_count}, NoTP={notp_count})")
    print(f"  accuracy={acc:.3f}, recall={rec:.3f}, precision={prec:.3f}")
