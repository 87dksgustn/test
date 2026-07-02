import numpy as np
import pandas as pd
import os
from itertools import product
from pathlib import Path
from scipy.stats import qmc
from sklearn.metrics import pairwise_distances
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =========================================================
# 0. Matplotlib 한글 폰트 설정
# =========================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 1. 사용자가 직접 수정하는 영역
# =========================================================

continuous_vars = {
    # "Cell_H": (300, 600),
    # "Cell_W": (90, 120),
    "Cell_D": (8, 16),
    "Barrier_Thx": (0.5, 2.5),
    "Barrier_Outer_Thx": (1.1, 3.0),
    # "Cooling_LPM": (0.0, 35.0),
    # "Venting_Gap": (1.0, 10.0),
    "ThermalResin_Thx": (0.5, 2.5),
    # "Housing_Btm_Thx": (1, 12),
    # "SideBeam_Thx": (14, 30),
}

# 기준값(center) 주변 +/- min_delta 구간은 샘플링에서 제외
continuous_exclusion_windows = {
    "Cell_D": {"center": 12.4, "min_delta": 0.01},
    "Barrier_Thx": {"center": 0.85, "min_delta": 0.01},
    "Barrier_Outer_Thx": {"center": 2.0, "min_delta": 0.01},
    "ThermalResin_Thx": {"center": 1.0, "min_delta": 0.01},
}

discrete_vars = {
    "Barrier_Type": ["Si1", "Si2", "Si3","Aerogel1", "Aerogel2"],
    "Barrier_Outer_Type": ["PU", "Si1", "Si2", "Si3"],
    # "Cooling_Loc": ["Top", "Bottom"],
    # "Heater_Type" : ["Small", "Medium"],
    # "Heater_Loc" : ["Center", "DSF", "Lead"],
    # "Cell/Barrier": [1, 2],
}

samples_per_discrete_combination = 10

n_trials = 300
seed_min = 0
seed_max = 100000

weight_A = 0.6   # min_over_groups_min_distance
weight_B = 0.4   # group_min_distance_q10 (하위 10% 분위수)

# seed 탐색 조기종료 설정 (patience 기반)
early_stop_warmup_trials = 80
early_stop_patience = 120
early_stop_min_delta = 1e-6

# greedy 배정 후 swap 국소개선 반복 횟수
local_swap_iterations = 60

# Optuna 병렬 trial 실행 수 (-1: 사용 가능한 모든 코어)
optuna_n_jobs = -1

# ---------------------------------------------------------
# Knowledge-driven biased initial sampling (경험지식 기반)
# ---------------------------------------------------------
# True면 변수별 편향 규칙을 적용한 LHS를 우선 사용
bias_sampling_enabled = True

# direction
# - "high"  : 큰 값 쪽으로 치우침
# - "low"   : 작은 값 쪽으로 치우침
# - "center": 중앙값 근처로 치우침
# - "edge"  : 양끝(극단) 값 쪽으로 치우침
# strength
# - 1.0: 편향 없음
# - 1.5~3.0: 보통 권장 범위
bias_sampling_rules = {
    "Cell_D": {"direction": "high", "strength": 1.0},
    "Barrier_Thx": {"direction": "low", "strength": 2.0},
    "Barrier_Outer_Thx": {"direction": "high", "strength": 1.5},
    "ThermalResin_Thx": {"direction": "high", "strength": 1.5},
}

# ---------------------------------------------------------
# Boundary-focused initial sampling (pass/fail 경계 집중)
# ---------------------------------------------------------
# False면 기존 균등 LHS + seed 최적화 흐름과 동일하게 동작
boundary_focus_enabled = False

# 과거 해석 결과(입력 + pass/fail)가 담긴 CSV 경로
historical_labeled_data_csv = "historical_pass_fail_results.csv"

# CSV 내 pass/fail 컬럼명 및 pass로 해석할 값들
pass_fail_column = "pass_fail"
pass_labels = [1, "1", True, "true", "pass", "PASS", "ok", "OK"]

# 모델 학습/샘플링 제어 파라미터
boundary_min_training_samples = 60
boundary_candidate_multiplier = 20
boundary_top_pool_multiplier = 5
boundary_exploration_ratio = 0.20


# lazy initialization cache
_boundary_focus_model = None
_boundary_focus_ready = False
_boundary_focus_checked = False



output_csv = "initial_DOE_best_seed_greedy_maximin.csv"
optuna_result_csv = "optuna_seed_optimization_results.csv"
group_distance_csv = "best_seed_group_min_distances.csv"
trials_dir = Path("Trials")


# =========================================================
# 2. DOE 생성 함수
# =========================================================

def apply_bias_to_unit_column(unit_col, direction, strength):
    direction_norm = str(direction).strip().lower()
    strength_value = max(float(strength), 1.0)

    if strength_value == 1.0:
        return unit_col

    if direction_norm == "high":
        return 1.0 - np.power(1.0 - unit_col, strength_value)

    if direction_norm == "low":
        return np.power(unit_col, strength_value)

    if direction_norm == "center":
        left_mask = unit_col < 0.5
        right_mask = ~left_mask

        transformed = np.empty_like(unit_col)

        transformed[left_mask] = 0.5 * np.power(
            2.0 * unit_col[left_mask],
            1.0 / strength_value
        )
        transformed[right_mask] = 1.0 - 0.5 * np.power(
            2.0 * (1.0 - unit_col[right_mask]),
            1.0 / strength_value
        )

        return transformed

    if direction_norm == "edge":
        left_mask = unit_col < 0.5
        right_mask = ~left_mask

        transformed = np.empty_like(unit_col)

        transformed[left_mask] = 0.5 * np.power(
            2.0 * unit_col[left_mask],
            strength_value
        )
        transformed[right_mask] = 1.0 - 0.5 * np.power(
            2.0 * (1.0 - unit_col[right_mask]),
            strength_value
        )

        return transformed

    raise ValueError(
        f"지원하지 않는 bias direction입니다: {direction}. "
        "사용 가능: high, low, center, edge"
    )


def generate_lhs_samples(continuous_vars, n_samples, seed, use_bias=False):
    var_names = list(continuous_vars.keys())
    bounds = np.array(list(continuous_vars.values()), dtype=float)

    sampler = qmc.LatinHypercube(
        d=len(var_names),
        seed=seed
    )

    X_unit = sampler.random(n=n_samples)

    X_scaled = np.zeros_like(X_unit)

    for col_idx, var_name in enumerate(var_names):
        lower, upper = bounds[col_idx]
        unit_col = X_unit[:, col_idx]

        if use_bias:
            rule = bias_sampling_rules.get(var_name)
            if rule is not None:
                direction = rule.get("direction", "high")
                strength = rule.get("strength", 1.0)
                unit_col = apply_bias_to_unit_column(unit_col, direction, strength)

        exclusion = continuous_exclusion_windows.get(var_name)
        if exclusion is None:
            X_scaled[:, col_idx] = qmc.scale(
                unit_col,
                lower,
                upper
            )
            continue

        center = float(exclusion["center"])
        min_delta = float(exclusion["min_delta"])
        left_end = center - min_delta
        right_start = center + min_delta

        if not (lower <= left_end and right_start <= upper):
            raise ValueError(
                f"{var_name} exclusion window가 범위를 벗어났습니다: "
                f"range=({lower}, {upper}), exclusion=({left_end}, {right_start})"
            )

        left_len = left_end - lower
        right_len = upper - right_start
        allowed_total = left_len + right_len

        if allowed_total <= 0:
            raise ValueError(
                f"{var_name}에서 exclusion window 적용 후 샘플링 가능한 구간이 없습니다."
            )

        if left_len <= 0:
            X_scaled[:, col_idx] = right_start + unit_col * right_len
            continue

        if right_len <= 0:
            X_scaled[:, col_idx] = lower + unit_col * left_len
            continue

        left_ratio = left_len / allowed_total
        left_mask = unit_col < left_ratio
        right_mask = ~left_mask

        if np.any(left_mask):
            X_scaled[left_mask, col_idx] = lower + (
                unit_col[left_mask] / left_ratio
            ) * left_len

        if np.any(right_mask):
            X_scaled[right_mask, col_idx] = right_start + (
                (unit_col[right_mask] - left_ratio) / (1.0 - left_ratio)
            ) * right_len

    df_cont = pd.DataFrame(X_scaled, columns=var_names)

    return df_cont, X_unit


def generate_biased_lhs_samples(continuous_vars, n_samples, seed):
    return generate_lhs_samples(
        continuous_vars=continuous_vars,
        n_samples=n_samples,
        seed=seed,
        use_bias=True
    )


def generate_discrete_combinations(discrete_vars):
    var_names = list(discrete_vars.keys())
    levels = list(discrete_vars.values())

    combinations = list(product(*levels))

    si_types = {"Si1", "Si2", "Si3"}
    barrier_key = "Barrier_Type"
    outer_key = "Barrier_Outer_Type"

    if barrier_key in var_names and outer_key in var_names:
        barrier_idx = var_names.index(barrier_key)
        outer_idx = var_names.index(outer_key)

        filtered_combinations = []
        for combo in combinations:
            barrier_type = combo[barrier_idx]
            outer_type = combo[outer_idx]

            # 둘 다 Si 계열일 때만 동일 타입을 강제
            if (barrier_type in si_types and outer_type in si_types) and (barrier_type != outer_type):
                continue

            filtered_combinations.append(combo)

        combinations = filtered_combinations

    return pd.DataFrame(combinations, columns=var_names)


def _normalize_label_to_binary(value, pass_label_set_raw, pass_label_set_str):
    if pd.isna(value):
        return np.nan

    if value in pass_label_set_raw:
        return 1

    value_str = str(value).strip().lower()
    if value_str in pass_label_set_str:
        return 1

    return 0


def initialize_boundary_focus_model(continuous_vars, discrete_vars):
    global _boundary_focus_model
    global _boundary_focus_ready
    global _boundary_focus_checked

    if _boundary_focus_checked:
        return _boundary_focus_model

    _boundary_focus_checked = True

    if not boundary_focus_enabled:
        print("Boundary focus 비활성화: 기존 균등 LHS를 사용합니다.")
        return None

    data_path = Path(historical_labeled_data_csv)
    if not data_path.exists():
        print(
            f"Boundary focus 비활성화: 학습 데이터 파일이 없습니다 ({data_path}). "
            "기존 균등 LHS를 사용합니다."
        )
        return None

    cont_cols = list(continuous_vars.keys())
    disc_cols = list(discrete_vars.keys())
    required_cols = cont_cols + disc_cols + [pass_fail_column]

    df_hist = pd.read_csv(data_path)

    missing_cols = [col for col in required_cols if col not in df_hist.columns]
    if missing_cols:
        print(
            "Boundary focus 비활성화: 학습 데이터 컬럼 누락 "
            f"{missing_cols}. 기존 균등 LHS를 사용합니다."
        )
        return None

    df_hist = df_hist[required_cols].dropna().copy()

    pass_label_set_raw = set(pass_labels)
    pass_label_set_str = {str(v).strip().lower() for v in pass_labels}

    y = df_hist[pass_fail_column].apply(
        lambda v: _normalize_label_to_binary(v, pass_label_set_raw, pass_label_set_str)
    )

    valid_mask = ~y.isna()
    X = df_hist.loc[valid_mask, cont_cols + disc_cols].copy()
    y = y.loc[valid_mask].astype(int)

    if len(X) < boundary_min_training_samples:
        print(
            "Boundary focus 비활성화: 학습 데이터 수가 부족합니다 "
            f"({len(X)} < {boundary_min_training_samples}). 기존 균등 LHS를 사용합니다."
        )
        return None

    if y.nunique() < 2:
        print(
            "Boundary focus 비활성화: pass/fail이 한 클래스만 존재합니다. "
            "기존 균등 LHS를 사용합니다."
        )
        return None

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", cont_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), disc_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=2026,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X, y)

    _boundary_focus_model = model
    _boundary_focus_ready = True

    pass_ratio = float(np.mean(y))
    print(
        "Boundary focus 활성화: surrogate 학습 완료 "
        f"(samples={len(X)}, pass_ratio={pass_ratio:.3f})."
    )

    return _boundary_focus_model


def select_diverse_subset(X_unit, n_select, seed):
    n_candidates = X_unit.shape[0]

    if n_select >= n_candidates:
        return np.arange(n_candidates)

    rng = np.random.default_rng(seed + 991)
    distance_matrix = pairwise_distances(X_unit, X_unit, metric="euclidean")

    first_idx = int(rng.integers(0, n_candidates))
    selected = [first_idx]

    min_dist_to_selected = distance_matrix[first_idx, :].copy()
    min_dist_to_selected[first_idx] = -np.inf

    while len(selected) < n_select:
        next_idx = int(np.argmax(min_dist_to_selected))
        selected.append(next_idx)

        min_dist_to_selected = np.minimum(
            min_dist_to_selected,
            distance_matrix[next_idx, :]
        )
        min_dist_to_selected[selected] = -np.inf

    return np.array(selected, dtype=int)


def compute_boundary_scores(df_cont_candidates, df_disc, model):
    n_candidates = len(df_cont_candidates)
    boundary_distance_sum = np.zeros(n_candidates, dtype=float)
    pass_prob_sum = np.zeros(n_candidates, dtype=float)

    disc_cols = list(df_disc.columns)

    for _, disc_row in df_disc.iterrows():
        X_eval = df_cont_candidates.copy()
        for col in disc_cols:
            X_eval[col] = disc_row[col]

        pass_prob = model.predict_proba(X_eval)[:, 1]

        boundary_distance_sum += np.abs(pass_prob - 0.5)
        pass_prob_sum += pass_prob

    boundary_distance_mean = boundary_distance_sum / len(df_disc)
    pass_prob_mean = pass_prob_sum / len(df_disc)

    # 값이 클수록 경계(0.5)에 가까움
    boundary_closeness = 0.5 - boundary_distance_mean

    return boundary_closeness, pass_prob_mean


def generate_boundary_focused_samples(
    continuous_vars,
    discrete_vars,
    n_samples,
    seed,
    model
):
    candidate_count = max(n_samples * boundary_candidate_multiplier, n_samples)

    df_cont_candidates, X_unit_candidates = generate_lhs_samples(
        continuous_vars=continuous_vars,
        n_samples=candidate_count,
        seed=seed
    )

    df_disc = generate_discrete_combinations(discrete_vars)

    boundary_closeness, _ = compute_boundary_scores(
        df_cont_candidates=df_cont_candidates,
        df_disc=df_disc,
        model=model
    )

    sorted_indices = np.argsort(-boundary_closeness)

    top_pool_size = min(
        candidate_count,
        max(n_samples, n_samples * boundary_top_pool_multiplier)
    )

    top_pool_indices = sorted_indices[:top_pool_size]

    n_explore = int(round(n_samples * boundary_exploration_ratio))
    n_explore = max(0, n_explore)

    rng = np.random.default_rng(seed + 1729)
    all_indices = np.arange(candidate_count)
    remaining_indices = np.setdiff1d(all_indices, top_pool_indices, assume_unique=False)

    if n_explore > 0 and len(remaining_indices) > 0:
        n_explore = min(n_explore, len(remaining_indices))
        explore_indices = rng.choice(remaining_indices, size=n_explore, replace=False)
        pool_indices = np.unique(np.concatenate([top_pool_indices, explore_indices]))
    else:
        pool_indices = top_pool_indices

    if len(pool_indices) < n_samples:
        pool_indices = all_indices

    selected_in_pool = select_diverse_subset(
        X_unit=X_unit_candidates[pool_indices],
        n_select=n_samples,
        seed=seed
    )

    selected_indices = pool_indices[selected_in_pool]

    df_cont_selected = df_cont_candidates.iloc[selected_indices].reset_index(drop=True)
    X_unit_selected = X_unit_candidates[selected_indices]

    return df_cont_selected, X_unit_selected


def generate_continuous_samples(continuous_vars, discrete_vars, n_samples, seed):
    if bias_sampling_enabled:
        print("Bias sampling 활성화: 경험지식 기반 비균등 LHS를 사용합니다.")
        return generate_biased_lhs_samples(
            continuous_vars=continuous_vars,
            n_samples=n_samples,
            seed=seed
        )

    model = initialize_boundary_focus_model(continuous_vars, discrete_vars)

    if model is None:
        print("Uniform sampling 사용: 균등 LHS를 사용합니다.")
        return generate_lhs_samples(
            continuous_vars=continuous_vars,
            n_samples=n_samples,
            seed=seed
        )

    return generate_boundary_focused_samples(
        continuous_vars=continuous_vars,
        discrete_vars=discrete_vars,
        n_samples=n_samples,
        seed=seed,
        model=model
    )


def compute_distance_matrix(X_unit):
    distances = pairwise_distances(X_unit, X_unit, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    return distances


def compute_group_min_distance(distance_matrix, group):
    if len(group) < 2:
        return np.inf

    group_distances = distance_matrix[np.ix_(group, group)]
    return np.min(group_distances)


def greedy_maximin_assignment(distance_matrix, n_groups, group_size, seed):
    rng = np.random.default_rng(seed)

    n_samples = distance_matrix.shape[0]

    if n_samples != n_groups * group_size:
        raise ValueError(
            f"샘플 수 불일치: n_samples={n_samples}, "
            f"n_groups * group_size={n_groups * group_size}"
        )

    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    groups = [[] for _ in range(n_groups)]

    # 각 이산 조합에 1개씩 먼저 배정
    for g in range(n_groups):
        groups[g].append(all_indices[g])

    remaining_indices = list(all_indices[n_groups:])

    # 남은 샘플은 greedy maximin 방식으로 배정
    for idx in remaining_indices:

        best_group = None
        best_score = -np.inf

        for g in range(n_groups):

            if len(groups[g]) >= group_size:
                continue

            min_distance_to_group = np.min(distance_matrix[idx, groups[g]])

            if min_distance_to_group > best_score:
                best_score = min_distance_to_group
                best_group = g

        groups[best_group].append(idx)

    return groups


def clone_groups(groups):
    return [group.copy() for group in groups]


def local_swap_refinement(distance_matrix, groups, seed, n_iterations):
    rng = np.random.default_rng(seed + 2026)

    best_groups = clone_groups(groups)
    group_min_distances = np.array([
        compute_group_min_distance(distance_matrix, group)
        for group in best_groups
    ], dtype=float)
    best_A = np.min(group_min_distances)

    for _ in range(n_iterations):
        g1, g2 = rng.choice(len(best_groups), size=2, replace=False)

        i1 = int(rng.integers(0, len(best_groups[g1])))
        i2 = int(rng.integers(0, len(best_groups[g2])))

        best_groups[g1][i1], best_groups[g2][i2] = (
            best_groups[g2][i2],
            best_groups[g1][i1]
        )

        updated_g1_min = compute_group_min_distance(distance_matrix, best_groups[g1])
        updated_g2_min = compute_group_min_distance(distance_matrix, best_groups[g2])

        if len(group_min_distances) <= 2:
            unaffected_min = np.inf
        else:
            unaffected_min = np.min(np.delete(group_min_distances, [g1, g2]))

        candidate_A = min(unaffected_min, updated_g1_min, updated_g2_min)

        if candidate_A > best_A:
            best_A = candidate_A
            group_min_distances[g1] = updated_g1_min
            group_min_distances[g2] = updated_g2_min
        else:
            best_groups[g1][i1], best_groups[g2][i2] = (
                best_groups[g2][i2],
                best_groups[g1][i1]
            )

    return best_groups


# =========================================================
# 3. DOE 품질 지표 계산 함수
# =========================================================

def calculate_group_min_distances(X_unit, groups, distance_matrix=None):
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(X_unit)

    group_min_distances = []

    for group_id, group in enumerate(groups, start=1):
        min_distance = compute_group_min_distance(distance_matrix, group)

        group_min_distances.append({
            "discrete_combination_id": group_id,
            "group_min_distance": min_distance
        })

    return pd.DataFrame(group_min_distances)


def calculate_min_over_groups_min_distance(X_unit, groups, distance_matrix=None):
    df_group_dist = calculate_group_min_distances(
        X_unit,
        groups,
        distance_matrix=distance_matrix
    )

    min_over_groups = df_group_dist["group_min_distance"].min()
    mean_group_min = df_group_dist["group_min_distance"].mean()
    q10_group_min = df_group_dist["group_min_distance"].quantile(0.1)

    return min_over_groups, mean_group_min, q10_group_min


def calculate_global_mean_nn_distance(X_unit, distance_matrix=None):
    if distance_matrix is None:
        distance_matrix = compute_distance_matrix(X_unit)

    nearest_distances = np.min(distance_matrix, axis=1)

    return np.mean(nearest_distances)


def evaluate_seed(seed):
    df_disc = generate_discrete_combinations(discrete_vars)

    n_groups = len(df_disc)
    total_samples = n_groups * samples_per_discrete_combination

    _, X_unit = generate_continuous_samples(
        continuous_vars=continuous_vars,
        discrete_vars=discrete_vars,
        n_samples=total_samples,
        seed=seed
    )

    distance_matrix = compute_distance_matrix(X_unit)

    groups = greedy_maximin_assignment(
        distance_matrix=distance_matrix,
        n_groups=n_groups,
        group_size=samples_per_discrete_combination,
        seed=seed
    )

    groups = local_swap_refinement(
        distance_matrix=distance_matrix,
        groups=groups,
        seed=seed,
        n_iterations=local_swap_iterations
    )

    min_over_groups, mean_group_min, q10_group_min = calculate_min_over_groups_min_distance(
        X_unit,
        groups,
        distance_matrix=distance_matrix
    )

    global_mean_nn = calculate_global_mean_nn_distance(
        X_unit,
        distance_matrix=distance_matrix
    )

    return {
        "seed": seed,
        "min_over_groups_min_distance": min_over_groups,
        "mean_group_min_distance": mean_group_min,
        "group_min_distance_q10": q10_group_min,
        "global_mean_nn_distance": global_mean_nn,
    }


# =========================================================
# 4. Optuna 목적 함수
# =========================================================

def objective(trial):
    seed = trial.suggest_int("seed", seed_min, seed_max)

    result = evaluate_seed(seed)

    A = result["min_over_groups_min_distance"]
    q10 = result["group_min_distance_q10"]

    trial.set_user_attr("seed", seed)
    trial.set_user_attr("min_over_groups_min_distance", A)
    trial.set_user_attr("group_min_distance_q10", q10)
    trial.set_user_attr("mean_group_min_distance", result["mean_group_min_distance"])
    trial.set_user_attr("global_mean_nn_distance", result["global_mean_nn_distance"])

    # Optuna 탐색 및 최종 best seed 선정에 사용하는 통합 score
    # A: 최악 조합 방지, q10: 하위권 조합 전반 품질 확인
    score = weight_A * A + weight_B * q10
    trial.set_user_attr("score", score)

    return score


def stop_when_target_reached(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    score = trial.user_attrs.get("score", trial.value)

    if score is None:
        return

    state = study.user_attrs.get("early_stop_state")
    if state is None:
        state = {
            "best_score": -np.inf,
            "best_trial_number": -1,
            "last_improvement_trial": -1,
        }

    if score > state["best_score"] + early_stop_min_delta:
        state["best_score"] = score
        state["best_trial_number"] = trial.number
        state["last_improvement_trial"] = trial.number

    study.set_user_attr("early_stop_state", state)

    best_trial = study.best_trial
    best_A = best_trial.user_attrs.get("min_over_groups_min_distance")
    best_q10 = best_trial.user_attrs.get("group_min_distance_q10")
    no_improve_trials = trial.number - state["last_improvement_trial"]

    print(
        f"Trial {trial.number} finished, "
        f"현재 Best trial {best_trial.number}: "
        f"A(min)={best_A:.6f}, q10={best_q10:.6f}, Score={best_trial.value:.6f}, "
        f"NoImprove={no_improve_trials}/{early_stop_patience}"
    )

    if (
        trial.number + 1 >= early_stop_warmup_trials
        and no_improve_trials >= early_stop_patience
    ):
        print(
            f"조기 종료: 최근 {no_improve_trials}회 동안 score 개선이 없어 탐색을 종료합니다."
        )
        study.stop()


# =========================================================
# 5. 최종 DOE 생성 함수
# =========================================================

def build_initial_doe(
    continuous_vars,
    discrete_vars,
    samples_per_discrete_combination,
    seed
):
    df_disc = generate_discrete_combinations(discrete_vars)

    n_discrete_combinations = len(df_disc)
    total_samples = n_discrete_combinations * samples_per_discrete_combination

    print(f"이산 조합 수: {n_discrete_combinations}")
    print(f"조합별 샘플 수: {samples_per_discrete_combination}")
    print(f"총 DOE 샘플 수: {total_samples}")
    print(f"사용 seed: {seed}")

    df_cont, X_unit = generate_continuous_samples(
        continuous_vars=continuous_vars,
        discrete_vars=discrete_vars,
        n_samples=total_samples,
        seed=seed
    )

    distance_matrix = compute_distance_matrix(X_unit)

    groups = greedy_maximin_assignment(
        distance_matrix=distance_matrix,
        n_groups=n_discrete_combinations,
        group_size=samples_per_discrete_combination,
        seed=seed
    )

    groups = local_swap_refinement(
        distance_matrix=distance_matrix,
        groups=groups,
        seed=seed,
        n_iterations=local_swap_iterations
    )

    rows = []

    for disc_idx, sample_indices in enumerate(groups):

        disc_row = df_disc.iloc[disc_idx].to_dict()

        for sample_idx in sample_indices:

            cont_row = df_cont.iloc[sample_idx].to_dict()

            row = {}
            row.update(cont_row)
            row.update(disc_row)

            row["discrete_combination_id"] = disc_idx + 1
            row["lhs_sample_id"] = sample_idx + 1
            row["seed"] = seed

            rows.append(row)

    df_doe = pd.DataFrame(rows)

    df_doe = df_doe.sort_values(
        ["discrete_combination_id", "lhs_sample_id"]
    ).reset_index(drop=True)

    return df_doe, X_unit, groups


# =========================================================
# 6. 시각화 함수
# =========================================================

def plot_score_contour(df_trials, save_path=None):
    A_col = "min_over_groups_min_distance"
    q10_col = "group_min_distance_q10"

    A = df_trials[A_col].values
    q10 = df_trials[q10_col].values

    A_min, A_max = A.min(), A.max()
    q10_min, q10_max = q10.min(), q10.max()

    A_norm = (
        (A - A_min) / (A_max - A_min)
        if A_max > A_min else np.ones_like(A)
    )
    q10_norm = (
        (q10 - q10_min) / (q10_max - q10_min)
        if q10_max > q10_min else np.ones_like(q10)
    )
    normalized_score = weight_A * A_norm + weight_B * q10_norm

    x_grid = np.linspace(0, 1, 100)
    y_grid = np.linspace(0, 1, 100)

    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    Z_grid = weight_A * Y_grid + weight_B * X_grid

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(
        X_grid,
        Y_grid,
        Z_grid,
        levels=20,
        alpha=0.8
    )

    plt.colorbar(contour, label="Normalized Score")

    plt.scatter(
        q10_norm,
        A_norm,
        c=normalized_score,
        edgecolors="black",
        s=60
    )

    best_pos = 0

    plt.scatter(
        q10_norm[best_pos],
        A_norm[best_pos],
        marker="*",
        s=250,
        edgecolors="black",
        label=f"Best Seed = {int(df_trials.iloc[best_pos]['seed'])}"
    )

    plt.xlabel("q10_norm: group_min_distance_q10")
    plt.ylabel("A_norm: min_over_groups_min_distance")
    plt.title("DOE Seed 최적화 Contour Plot (0.6A + 0.4q10)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # plt.show()


def plot_group_min_distance_bar(X_unit_best, groups_best, save_path=None):
    df_group_dist = calculate_group_min_distances(
        X_unit_best,
        groups_best
    )

    plt.figure(figsize=(12, 5))

    plt.bar(
        df_group_dist["discrete_combination_id"],
        df_group_dist["group_min_distance"]
    )

    min_value = df_group_dist["group_min_distance"].min()
    mean_value = df_group_dist["group_min_distance"].mean()

    plt.axhline(
        min_value,
        linestyle="--",
        linewidth=2,
        label=f"Min = {min_value:.4f}"
    )

    plt.axhline(
        mean_value,
        linestyle=":",
        linewidth=2,
        label=f"Mean = {mean_value:.4f}"
    )

    plt.xlabel("이산 조합 ID")
    plt.ylabel("조합 내 Minimum Distance")
    plt.title("Best Seed 기준 이산 조합별 Minimum Distance")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # plt.show()

    return df_group_dist


def summarize_sampling_distribution(df_sample, label):
    summary_rows = []

    for column in df_sample.columns:
        values = df_sample[column].to_numpy(dtype=float)
        summary_rows.append({
            "sampling_type": label,
            "variable": column,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "q10": float(np.quantile(values, 0.1)),
            "q50": float(np.quantile(values, 0.5)),
            "q90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
        })

    return pd.DataFrame(summary_rows)


def normalize_continuous_df_to_unit(df_sample, continuous_vars):
    unit_df = pd.DataFrame(index=df_sample.index)

    for var_name, (lower, upper) in continuous_vars.items():
        span = float(upper) - float(lower)
        if span <= 0:
            raise ValueError(f"{var_name} 범위가 유효하지 않습니다: ({lower}, {upper})")

        unit_df[var_name] = (df_sample[var_name].astype(float) - float(lower)) / span

    return unit_df


def compute_bias_adherence_score(df_unit, bias_rules):
    if not bias_rules:
        return 0.5

    per_var_scores = []

    for var_name, rule in bias_rules.items():
        if var_name not in df_unit.columns:
            continue

        direction = str(rule.get("direction", "high")).strip().lower()
        values = df_unit[var_name].to_numpy(dtype=float)

        if direction == "high":
            score = float(np.mean(values))
        elif direction == "low":
            score = float(1.0 - np.mean(values))
        elif direction == "center":
            mean_abs_center_dist = float(np.mean(np.abs(values - 0.5)))
            score = float(np.clip(1.0 - (mean_abs_center_dist / 0.25), 0.0, 1.0))
        elif direction == "edge":
            mean_abs_center_dist = float(np.mean(np.abs(values - 0.5)))
            score = float(np.clip((mean_abs_center_dist - 0.25) / 0.25, 0.0, 1.0))
        else:
            score = 0.5

        per_var_scores.append(score)

    if len(per_var_scores) == 0:
        return 0.5

    return float(np.mean(per_var_scores))


def compute_pre_cfd_sampling_scores(df_sample, continuous_vars, bias_rules):
    df_unit = normalize_continuous_df_to_unit(df_sample, continuous_vars)
    X_unit = df_unit.to_numpy(dtype=float)

    discrepancy = float(qmc.discrepancy(X_unit, method="CD"))
    coverage_score = float(1.0 / (1.0 + 20.0 * discrepancy))

    distance_matrix = pairwise_distances(X_unit, X_unit, metric="euclidean")
    np.fill_diagonal(distance_matrix, np.inf)
    nearest_distances = np.min(distance_matrix, axis=1)

    d = X_unit.shape[1]
    max_unit_distance = float(np.sqrt(d))
    mean_nn = float(np.mean(nearest_distances))
    q10_nn = float(np.quantile(nearest_distances, 0.1))

    mean_nn_norm = float(np.clip(mean_nn / max_unit_distance, 0.0, 1.0))
    q10_nn_norm = float(np.clip(q10_nn / max_unit_distance, 0.0, 1.0))
    diversity_score = float(0.5 * mean_nn_norm + 0.5 * q10_nn_norm)

    bias_adherence_score = compute_bias_adherence_score(df_unit, bias_rules)

    pre_cfd_score = float(
        0.35 * coverage_score
        + 0.15 * diversity_score
        + 0.50 * bias_adherence_score
    )

    return {
        "coverage_discrepancy": discrepancy,
        "coverage_score": coverage_score,
        "mean_nn_distance_unit": mean_nn,
        "q10_nn_distance_unit": q10_nn,
        "diversity_score": diversity_score,
        "bias_adherence_score": bias_adherence_score,
        "pre_cfd_score": pre_cfd_score,
    }


def save_coverage_cdf_plot(df_uniform, df_biased, continuous_vars, save_dir):
    var_names = list(continuous_vars.keys())
    n_vars = len(var_names)
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 3.8 * n_rows),
        squeeze=False
    )

    for idx, var_name in enumerate(var_names):
        ax = axes[idx // n_cols][idx % n_cols]
        lower, upper = continuous_vars[var_name]

        uniform_sorted = np.sort(df_uniform[var_name].values)
        biased_sorted = np.sort(df_biased[var_name].values)
        n = len(uniform_sorted)
        ecdf_y = np.arange(1, n + 1) / n

        ideal_x = np.linspace(lower, upper, 100)
        ideal_cdf = (ideal_x - lower) / (upper - lower)

        ax.plot(ideal_x, ideal_cdf, "k--", linewidth=1.5, label="Ideal Uniform CDF")
        ax.step(uniform_sorted, ecdf_y, where="post", label="Uniform", color="#4C78A8", linewidth=1.2)
        ax.step(biased_sorted, ecdf_y, where="post", label="Biased", color="#F58518", linewidth=1.2)

        ax.set_xlabel(var_name)
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"CDF: {var_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_xlim(lower, upper)
        ax.set_ylim(0, 1)

    for idx in range(n_vars, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Coverage CDF Comparison (Uniform vs Biased)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    cdf_path = save_dir / "2_coverage_CDF.png"
    fig.savefig(cdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return cdf_path


def save_coverage_heatmap_plot(df_uniform, df_biased, continuous_vars, save_dir):
    var_names = list(continuous_vars.keys())

    if len(var_names) < 2:
        return None

    x_var = var_names[0]
    y_var = var_names[1]
    x_lower, x_upper = continuous_vars[x_var]
    y_lower, y_upper = continuous_vars[y_var]

    n_bins = 12
    x_edges = np.linspace(x_lower, x_upper, n_bins + 1)
    y_edges = np.linspace(y_lower, y_upper, n_bins + 1)

    uniform_hist, _, _ = np.histogram2d(
        df_uniform[x_var], df_uniform[y_var],
        bins=[x_edges, y_edges]
    )
    biased_hist, _, _ = np.histogram2d(
        df_biased[x_var], df_biased[y_var],
        bins=[x_edges, y_edges]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(
        uniform_hist.T,
        origin="lower",
        extent=[x_lower, x_upper, y_lower, y_upper],
        aspect="auto",
        cmap="Blues"
    )
    axes[0].set_xlabel(x_var)
    axes[0].set_ylabel(y_var)
    axes[0].set_title("Uniform Sampling Density")
    fig.colorbar(im0, ax=axes[0], label="Count")

    im1 = axes[1].imshow(
        biased_hist.T,
        origin="lower",
        extent=[x_lower, x_upper, y_lower, y_upper],
        aspect="auto",
        cmap="Oranges"
    )
    axes[1].set_xlabel(x_var)
    axes[1].set_ylabel(y_var)
    axes[1].set_title("Biased Sampling Density")
    fig.colorbar(im1, ax=axes[1], label="Count")

    fig.suptitle(f"Coverage Heatmap: {x_var} vs {y_var}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    heatmap_path = save_dir / "2_coverage_heatmap.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return heatmap_path


def save_coverage_grid_plot(df_uniform, df_biased, continuous_vars, save_dir):
    var_names = list(continuous_vars.keys())

    if len(var_names) < 2:
        return None

    x_var = var_names[0]
    y_var = var_names[1]
    x_lower, x_upper = continuous_vars[x_var]
    y_lower, y_upper = continuous_vars[y_var]

    n_bins = 8
    x_edges = np.linspace(x_lower, x_upper, n_bins + 1)
    y_edges = np.linspace(y_lower, y_upper, n_bins + 1)

    def compute_occupancy(df):
        hist, _, _ = np.histogram2d(
            df[x_var], df[y_var],
            bins=[x_edges, y_edges]
        )
        occupied = (hist > 0).astype(int)
        return occupied, hist

    uniform_occ, uniform_count = compute_occupancy(df_uniform)
    biased_occ, biased_count = compute_occupancy(df_biased)

    total_cells = n_bins * n_bins
    uniform_occupancy_rate = float(np.sum(uniform_occ)) / total_cells
    biased_occupancy_rate = float(np.sum(biased_occ)) / total_cells

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(n_bins):
        for j in range(n_bins):
            x_center = (x_edges[i] + x_edges[i + 1]) / 2
            y_center = (y_edges[j] + y_edges[j + 1]) / 2

            if uniform_occ[i, j] > 0:
                axes[0].add_patch(plt.Rectangle(
                    (x_edges[i], y_edges[j]),
                    x_edges[i + 1] - x_edges[i],
                    y_edges[j + 1] - y_edges[j],
                    facecolor="#4C78A8",
                    alpha=0.4,
                    edgecolor="gray",
                    linewidth=0.5
                ))
                axes[0].text(
                    x_center, y_center,
                    str(int(uniform_count[i, j])),
                    ha="center", va="center", fontsize=7
                )
            else:
                axes[0].add_patch(plt.Rectangle(
                    (x_edges[i], y_edges[j]),
                    x_edges[i + 1] - x_edges[i],
                    y_edges[j + 1] - y_edges[j],
                    facecolor="white",
                    edgecolor="gray",
                    linewidth=0.5
                ))

            if biased_occ[i, j] > 0:
                axes[1].add_patch(plt.Rectangle(
                    (x_edges[i], y_edges[j]),
                    x_edges[i + 1] - x_edges[i],
                    y_edges[j + 1] - y_edges[j],
                    facecolor="#F58518",
                    alpha=0.4,
                    edgecolor="gray",
                    linewidth=0.5
                ))
                axes[1].text(
                    x_center, y_center,
                    str(int(biased_count[i, j])),
                    ha="center", va="center", fontsize=7
                )
            else:
                axes[1].add_patch(plt.Rectangle(
                    (x_edges[i], y_edges[j]),
                    x_edges[i + 1] - x_edges[i],
                    y_edges[j + 1] - y_edges[j],
                    facecolor="white",
                    edgecolor="gray",
                    linewidth=0.5
                ))

    axes[0].set_xlim(x_lower, x_upper)
    axes[0].set_ylim(y_lower, y_upper)
    axes[0].set_xlabel(x_var)
    axes[0].set_ylabel(y_var)
    axes[0].set_title(f"Uniform Grid Occupancy: {uniform_occupancy_rate * 100:.1f}%")
    axes[0].set_aspect("auto")

    axes[1].set_xlim(x_lower, x_upper)
    axes[1].set_ylim(y_lower, y_upper)
    axes[1].set_xlabel(x_var)
    axes[1].set_ylabel(y_var)
    axes[1].set_title(f"Biased Grid Occupancy: {biased_occupancy_rate * 100:.1f}%")
    axes[1].set_aspect("auto")

    fig.suptitle(f"Grid Occupancy: {x_var} vs {y_var}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    grid_path = save_dir / "2_coverage_grid.png"
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return grid_path, uniform_occupancy_rate, biased_occupancy_rate


def save_sampling_comparison_plots(
    continuous_vars,
    n_samples,
    seed,
    save_dir
):
    var_names = list(continuous_vars.keys())

    df_uniform, _ = generate_lhs_samples(
        continuous_vars=continuous_vars,
        n_samples=n_samples,
        seed=seed,
        use_bias=False
    )
    df_biased, _ = generate_lhs_samples(
        continuous_vars=continuous_vars,
        n_samples=n_samples,
        seed=seed,
        use_bias=True
    )

    summary_uniform = summarize_sampling_distribution(df_uniform, "uniform")
    summary_biased = summarize_sampling_distribution(df_biased, "biased")

    metrics_uniform = compute_pre_cfd_sampling_scores(
        df_sample=df_uniform,
        continuous_vars=continuous_vars,
        bias_rules=bias_sampling_rules
    )
    metrics_biased = compute_pre_cfd_sampling_scores(
        df_sample=df_biased,
        continuous_vars=continuous_vars,
        bias_rules=bias_sampling_rules
    )

    metrics_rows = [
        {
            "sampling_type": "uniform",
            "variable": "__overall__",
            **metrics_uniform,
        },
        {
            "sampling_type": "biased",
            "variable": "__overall__",
            **metrics_biased,
        },
    ]
    df_metrics = pd.DataFrame(metrics_rows)

    df_summary = pd.concat(
        [summary_uniform, summary_biased, df_metrics],
        ignore_index=True,
        sort=False
    )

    summary_path = save_dir / "sampling_comparison_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    n_vars = len(var_names)
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 3.8 * n_rows),
        squeeze=False
    )

    for idx, var_name in enumerate(var_names):
        ax = axes[idx // n_cols][idx % n_cols]

        ax.hist(
            df_uniform[var_name],
            bins=24,
            alpha=0.45,
            density=True,
            label="Uniform",
            color="#4C78A8"
        )
        ax.hist(
            df_biased[var_name],
            bins=24,
            alpha=0.45,
            density=True,
            label="Biased",
            color="#F58518"
        )

        ax.set_title(var_name)
        ax.grid(True, alpha=0.25)
        ax.legend()

    for idx in range(n_vars, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Uniform vs Biased Initial Sampling Distribution", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    hist_path = save_dir / "1_bias_histogram.png"
    fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    scatter_path = None
    if len(var_names) >= 2:
        x_var = var_names[0]
        y_var = var_names[1]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            df_uniform[x_var],
            df_uniform[y_var],
            s=28,
            alpha=0.45,
            label="Uniform",
            color="#4C78A8",
            edgecolors="none"
        )
        ax.scatter(
            df_biased[x_var],
            df_biased[y_var],
            s=28,
            alpha=0.45,
            label="Biased",
            color="#F58518",
            edgecolors="none"
        )

        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f"Uniform vs Biased Scatter: {x_var} vs {y_var}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        scatter_path = save_dir / "1_bias_scatter.png"
        fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    cdf_path = save_coverage_cdf_plot(df_uniform, df_biased, continuous_vars, save_dir)

    heatmap_path = None
    grid_path = None
    uniform_occ_rate = None
    biased_occ_rate = None

    return {
        "summary_path": summary_path,
        "hist_path": hist_path,
        "scatter_path": scatter_path,
        "cdf_path": cdf_path,
        "heatmap_path": heatmap_path,
        "grid_path": grid_path,
        "uniform_occupancy_rate": uniform_occ_rate,
        "biased_occupancy_rate": biased_occ_rate,
        "uniform_pre_cfd_score": metrics_uniform["pre_cfd_score"],
        "biased_pre_cfd_score": metrics_biased["pre_cfd_score"],
        "uniform_detail": metrics_uniform,
        "biased_detail": metrics_biased,
    }


def create_trial_result_dir(trials_dir, bias_score, coverage_score, A_score):
    trials_dir.mkdir(exist_ok=True)

    result_dir = trials_dir / f"bias{bias_score:.4f}_cvrg{coverage_score:.4f}_A{A_score:.4f}"
    result_dir.mkdir(exist_ok=True)

    return result_dir


# =========================================================
# 7. 실행
# =========================================================

study = optuna.create_study(direction="maximize")
study.optimize(
    objective,
    n_trials=n_trials,
    n_jobs=optuna_n_jobs,
    callbacks=[stop_when_target_reached]
)


# Trial 결과 정리
trial_results = []

for trial in study.trials:
    trial_results.append({
        "trial_number": trial.number,
        "seed": trial.params.get("seed"),
        "score": trial.value,
        "min_over_groups_min_distance": trial.user_attrs.get("min_over_groups_min_distance"),
        "group_min_distance_q10": trial.user_attrs.get("group_min_distance_q10"),
        "mean_group_min_distance": trial.user_attrs.get("mean_group_min_distance"),
        "global_mean_nn_distance": trial.user_attrs.get("global_mean_nn_distance"),
    })

df_trials = pd.DataFrame(trial_results)


# Score 기준 best seed 선정
A_col = "min_over_groups_min_distance"
q10_col = "group_min_distance_q10"
score_col = "score"

df_trials = df_trials.sort_values(
    score_col,
    ascending=False
).reset_index(drop=True)

best_seed = int(df_trials.loc[0, "seed"])
best_A = df_trials.loc[0, A_col]
best_q10 = df_trials.loc[0, q10_col]
best_score = df_trials.loc[0, score_col]


print("\nOptuna seed 최적화 완료")
print(f"Best seed by Score: {best_seed}")
print(f"A (min_over_groups): {best_A:.6f}")
print(f"q10 (하위 10%): {best_q10:.6f}")
print(f"Score: {best_score:.6f}")

print("\nBest seed DOE 품질 지표")
print(f"min_over_groups_min_distance: {best_A:.6f}")
print(f"group_min_distance_q10: {best_q10:.6f}")
print(f"mean_group_min_distance: {df_trials.loc[0, 'mean_group_min_distance']:.6f}")
print(f"global_mean_nn_distance: {df_trials.loc[0, 'global_mean_nn_distance']:.6f}")


# 최적 seed로 최종 DOE 생성 (폴더명 결정을 위해 먼저 생성)
df_doe, X_unit_best, groups_best = build_initial_doe(
    continuous_vars=continuous_vars,
    discrete_vars=discrete_vars,
    samples_per_discrete_combination=samples_per_discrete_combination,
    seed=best_seed
)

# 폴더명 결정을 위해 biased 샘플의 pre-CFD 점수 먼저 계산
df_biased_for_score, _ = generate_lhs_samples(
    continuous_vars=continuous_vars,
    n_samples=len(df_doe),
    seed=best_seed,
    use_bias=True
)
biased_metrics_for_folder = compute_pre_cfd_sampling_scores(
    df_sample=df_biased_for_score,
    continuous_vars=continuous_vars,
    bias_rules=bias_sampling_rules
)

bias_score_for_folder = biased_metrics_for_folder["bias_adherence_score"]
coverage_score_for_folder = biased_metrics_for_folder["coverage_score"]

result_dir = create_trial_result_dir(
    trials_dir,
    bias_score=bias_score_for_folder,
    coverage_score=coverage_score_for_folder,
    A_score=best_A
)
output_csv_path = result_dir / output_csv
optuna_result_csv_path = result_dir / optuna_result_csv
group_distance_csv_path = result_dir / group_distance_csv
contour_plot_path = result_dir / "3_A_seed_optimization_contour.png"

print(f"\n결과 저장 폴더: {result_dir}")


# Optuna 결과 저장
df_trials.to_csv(
    optuna_result_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nOptuna 결과 저장 완료: {optuna_result_csv_path}")


df_doe.to_csv(
    output_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n최종 DOE 생성 완료")
print(f"저장 파일: {output_csv_path}")
print(df_doe.head())


# 편향 샘플링 vs 균등 샘플링 비교 저장
comparison_paths = save_sampling_comparison_plots(
    continuous_vars=continuous_vars,
    n_samples=len(df_doe),
    seed=best_seed,
    save_dir=result_dir
)

print(f"\n샘플링 비교 요약 저장 완료: {comparison_paths['summary_path']}")
print(f"1_bias_histogram 저장 완료: {comparison_paths['hist_path']}")
if comparison_paths["scatter_path"] is not None:
    print(f"1_bias_scatter 저장 완료: {comparison_paths['scatter_path']}")
print(f"2_coverage_CDF 저장 완료: {comparison_paths['cdf_path']}")

print("\nPre-CFD Sampling Score (출력값 없이 입력분포만으로 평가)")
print(f"Uniform pre_cfd_score: {comparison_paths['uniform_pre_cfd_score']:.6f}")
print(f"Biased  pre_cfd_score: {comparison_paths['biased_pre_cfd_score']:.6f}")

uniform_detail = comparison_paths["uniform_detail"]
biased_detail = comparison_paths["biased_detail"]

print(
    "Uniform detail: "
    f"coverage={uniform_detail['coverage_score']:.6f}, "
    f"diversity={uniform_detail['diversity_score']:.6f}, "
    f"bias_adherence={uniform_detail['bias_adherence_score']:.6f}"
)
print(
    "Biased detail:  "
    f"coverage={biased_detail['coverage_score']:.6f}, "
    f"diversity={biased_detail['diversity_score']:.6f}, "
    f"bias_adherence={biased_detail['bias_adherence_score']:.6f}"
)


# 시각화 실행
plot_score_contour(df_trials, save_path=contour_plot_path)

df_group_dist = plot_group_min_distance_bar(
    X_unit_best,
    groups_best,
    save_path=None
)

df_group_dist.to_csv(
    group_distance_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n조합별 minimum distance 저장 완료: {group_distance_csv_path}")
print(f"3_A_seed_optimization_contour 저장 완료: {contour_plot_path}")
print(df_group_dist.head())