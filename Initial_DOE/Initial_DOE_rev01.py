import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
from scipy.stats import qmc
from sklearn.metrics import pairwise_distances
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
    "Barrier_Thx": (0.5, 3.0),
    "Barrier_Outer_Thx": (0.5, 3.0),
    # "Cooling_LPM": (0.0, 35.0),
    # "Venting_Gap": (1.0, 10.0),
    "ThermalResin_Thx": (0.5, 3.0),
    "Housing_Btm_Thx": (1, 12),
    "SideBeam_Thx": (14, 30),
}

discrete_vars = {
    "Barrier_Type": ["Si", "Aerogel"],
    "Barrier_Outer_Type": ["Si", "Aerogel"],
    # "Cooling_Loc": ["Top", "Bottom"],
    # "Heater_Type" : ["Small", "Medium"],
    # "Heater_Loc" : ["Center", "DSF", "Lead"],
    "Cell/Barrier": [1, 2],
}

samples_per_discrete_combination = 7

n_trials = 300
seed_min = 0
seed_max = 100000

weight_A = 0.7   # min_over_groups_min_distance
weight_B = 0.3   # global_mean_nn_distance

# seed 탐색 조기종료 설정 (patience 기반)
early_stop_warmup_trials = 80
early_stop_patience = 120
early_stop_min_delta = 1e-6

# greedy 배정 후 swap 국소개선 반복 횟수
local_swap_iterations = 60



output_csv = "initial_DOE_best_seed_greedy_maximin.csv"
optuna_result_csv = "optuna_seed_optimization_results.csv"
group_distance_csv = "best_seed_group_min_distances.csv"
trials_dir = Path("Trials")


# =========================================================
# 2. DOE 생성 함수
# =========================================================

def generate_lhs_samples(continuous_vars, n_samples, seed):
    var_names = list(continuous_vars.keys())
    bounds = np.array(list(continuous_vars.values()), dtype=float)

    sampler = qmc.LatinHypercube(
        d=len(var_names),
        seed=seed
    )

    X_unit = sampler.random(n=n_samples)

    X_scaled = qmc.scale(
        X_unit,
        bounds[:, 0],
        bounds[:, 1]
    )

    df_cont = pd.DataFrame(X_scaled, columns=var_names)

    return df_cont, X_unit


def generate_discrete_combinations(discrete_vars):
    var_names = list(discrete_vars.keys())
    levels = list(discrete_vars.values())

    combinations = list(product(*levels))

    return pd.DataFrame(combinations, columns=var_names)


def greedy_maximin_assignment(X_unit, n_groups, group_size, seed):
    rng = np.random.default_rng(seed)

    n_samples = X_unit.shape[0]

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

        x_candidate = X_unit[idx].reshape(1, -1)

        for g in range(n_groups):

            if len(groups[g]) >= group_size:
                continue

            group_points = X_unit[groups[g]]

            distances = pairwise_distances(
                x_candidate,
                group_points,
                metric="euclidean"
            )

            min_distance_to_group = np.min(distances)

            if min_distance_to_group > best_score:
                best_score = min_distance_to_group
                best_group = g

        groups[best_group].append(idx)

    return groups


def clone_groups(groups):
    return [group.copy() for group in groups]


def local_swap_refinement(X_unit, groups, seed, n_iterations):
    rng = np.random.default_rng(seed + 2026)

    best_groups = clone_groups(groups)
    best_A, _ = calculate_min_over_groups_min_distance(X_unit, best_groups)

    for _ in range(n_iterations):
        g1, g2 = rng.choice(len(best_groups), size=2, replace=False)

        i1 = int(rng.integers(0, len(best_groups[g1])))
        i2 = int(rng.integers(0, len(best_groups[g2])))

        best_groups[g1][i1], best_groups[g2][i2] = (
            best_groups[g2][i2],
            best_groups[g1][i1]
        )

        candidate_A, _ = calculate_min_over_groups_min_distance(X_unit, best_groups)

        if candidate_A > best_A:
            best_A = candidate_A
        else:
            best_groups[g1][i1], best_groups[g2][i2] = (
                best_groups[g2][i2],
                best_groups[g1][i1]
            )

    return best_groups


# =========================================================
# 3. DOE 품질 지표 계산 함수
# =========================================================

def calculate_group_min_distances(X_unit, groups):
    group_min_distances = []

    for group_id, group in enumerate(groups, start=1):
        X_group = X_unit[group]

        distances = pairwise_distances(X_group, X_group)
        distances[distances == 0] = np.inf

        min_distance = np.min(distances)

        group_min_distances.append({
            "discrete_combination_id": group_id,
            "group_min_distance": min_distance
        })

    return pd.DataFrame(group_min_distances)


def calculate_min_over_groups_min_distance(X_unit, groups):
    df_group_dist = calculate_group_min_distances(X_unit, groups)

    min_over_groups = df_group_dist["group_min_distance"].min()
    mean_group_min = df_group_dist["group_min_distance"].mean()

    return min_over_groups, mean_group_min


def calculate_global_mean_nn_distance(X_unit):
    distances = pairwise_distances(X_unit, X_unit)
    distances[distances == 0] = np.inf

    nearest_distances = np.min(distances, axis=1)

    return np.mean(nearest_distances)


def evaluate_seed(seed):
    df_disc = generate_discrete_combinations(discrete_vars)

    n_groups = len(df_disc)
    total_samples = n_groups * samples_per_discrete_combination

    _, X_unit = generate_lhs_samples(
        continuous_vars=continuous_vars,
        n_samples=total_samples,
        seed=seed
    )

    groups = greedy_maximin_assignment(
        X_unit=X_unit,
        n_groups=n_groups,
        group_size=samples_per_discrete_combination,
        seed=seed
    )

    groups = local_swap_refinement(
        X_unit=X_unit,
        groups=groups,
        seed=seed,
        n_iterations=local_swap_iterations
    )

    min_over_groups, mean_group_min = calculate_min_over_groups_min_distance(
        X_unit,
        groups
    )

    global_mean_nn = calculate_global_mean_nn_distance(X_unit)

    return {
        "seed": seed,
        "min_over_groups_min_distance": min_over_groups,
        "mean_group_min_distance": mean_group_min,
        "global_mean_nn_distance": global_mean_nn,
    }


# =========================================================
# 4. Optuna 목적 함수
# =========================================================

def objective(trial):
    seed = trial.suggest_int("seed", seed_min, seed_max)

    result = evaluate_seed(seed)

    A = result["min_over_groups_min_distance"]
    B = result["global_mean_nn_distance"]

    trial.set_user_attr("seed", seed)
    trial.set_user_attr("min_over_groups_min_distance", A)
    trial.set_user_attr("mean_group_min_distance", result["mean_group_min_distance"])
    trial.set_user_attr("global_mean_nn_distance", B)

    # Optuna 탐색 및 최종 best seed 선정에 사용하는 통합 score
    score = weight_A * A + weight_B * B
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
    best_B = best_trial.user_attrs.get("global_mean_nn_distance")
    no_improve_trials = trial.number - state["last_improvement_trial"]

    print(
        f"Trial {trial.number} finished, "
        f"현재 Best trial {best_trial.number}: "
        f"A={best_A:.6f}, B={best_B:.6f}, Score={best_trial.value:.6f}, "
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

    df_cont, X_unit = generate_lhs_samples(
        continuous_vars=continuous_vars,
        n_samples=total_samples,
        seed=seed
    )

    groups = greedy_maximin_assignment(
        X_unit=X_unit,
        n_groups=n_discrete_combinations,
        group_size=samples_per_discrete_combination,
        seed=seed
    )

    groups = local_swap_refinement(
        X_unit=X_unit,
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
    B_col = "global_mean_nn_distance"

    A = df_trials[A_col].values
    B = df_trials[B_col].values

    A_min, A_max = A.min(), A.max()
    B_min, B_max = B.min(), B.max()

    A_norm = (
        (A - A_min) / (A_max - A_min)
        if A_max > A_min else np.ones_like(A)
    )
    B_norm = (
        (B - B_min) / (B_max - B_min)
        if B_max > B_min else np.ones_like(B)
    )
    normalized_score = weight_A * A_norm + weight_B * B_norm

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
        B_norm,
        A_norm,
        c=normalized_score,
        edgecolors="black",
        s=60
    )

    best_pos = 0

    plt.scatter(
        B_norm[best_pos],
        A_norm[best_pos],
        marker="*",
        s=250,
        edgecolors="black",
        label=f"Best Seed = {int(df_trials.iloc[best_pos]['seed'])}"
    )

    plt.xlabel("B_norm: global_mean_nn_distance")
    plt.ylabel("A_norm: min_over_groups_min_distance")
    plt.title("DOE Seed 최적화 Contour Plot")
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


def create_trial_result_dir(trials_dir, A, score):
    trials_dir.mkdir(exist_ok=True)

    existing_numbers = []

    for path in trials_dir.iterdir():
        if not path.is_dir():
            continue

        number_part = path.name.split("_", 1)[0]

        if number_part.isdigit():
            existing_numbers.append(int(number_part))

    next_number = max(existing_numbers, default=0) + 1
    result_dir = trials_dir / f"{next_number}_{A:.6f}_{score:.6f}"
    result_dir.mkdir()

    return result_dir


# =========================================================
# 7. 실행
# =========================================================

study = optuna.create_study(direction="maximize")
study.optimize(
    objective,
    n_trials=n_trials,
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
        "global_mean_nn_distance": trial.user_attrs.get("global_mean_nn_distance"),
        "mean_group_min_distance": trial.user_attrs.get("mean_group_min_distance"),
    })

df_trials = pd.DataFrame(trial_results)


# Score 기준 best seed 선정
A_col = "min_over_groups_min_distance"
B_col = "global_mean_nn_distance"
score_col = "score"

df_trials = df_trials.sort_values(
    score_col,
    ascending=False
).reset_index(drop=True)

best_seed = int(df_trials.loc[0, "seed"])
best_A = df_trials.loc[0, A_col]
best_score = df_trials.loc[0, score_col]


print("\nOptuna seed 최적화 완료")
print(f"Best seed by Score: {best_seed}")
print(f"A: {best_A:.6f}")
print(f"B: {df_trials.loc[0, B_col]:.6f}")
print(f"Score: {best_score:.6f}")

print("\nBest seed DOE 품질 지표")
print(f"min_over_groups_min_distance: {best_A:.6f}")
print(f"global_mean_nn_distance: {df_trials.loc[0, B_col]:.6f}")
print(f"mean_group_min_distance: {df_trials.loc[0, 'mean_group_min_distance']:.6f}")


result_dir = create_trial_result_dir(trials_dir, best_A, best_score)
output_csv_path = result_dir / output_csv
optuna_result_csv_path = result_dir / optuna_result_csv
group_distance_csv_path = result_dir / group_distance_csv
contour_plot_path = result_dir / "seed_optimization_contour.png"
group_bar_plot_path = result_dir / "group_min_distance_bar.png"

print(f"\n결과 저장 폴더: {result_dir}")


# Optuna 결과 저장
df_trials.to_csv(
    optuna_result_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nOptuna 결과 저장 완료: {optuna_result_csv_path}")


# 최적 seed로 최종 DOE 생성
df_doe, X_unit_best, groups_best = build_initial_doe(
    continuous_vars=continuous_vars,
    discrete_vars=discrete_vars,
    samples_per_discrete_combination=samples_per_discrete_combination,
    seed=best_seed
)

df_doe.to_csv(
    output_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n최종 DOE 생성 완료")
print(f"저장 파일: {output_csv_path}")
print(df_doe.head())


# 시각화 실행
plot_score_contour(df_trials, save_path=contour_plot_path)

df_group_dist = plot_group_min_distance_bar(
    X_unit_best,
    groups_best,
    save_path=group_bar_plot_path
)

df_group_dist.to_csv(
    group_distance_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n조합별 minimum distance 저장 완료: {group_distance_csv_path}")
print(f"Contour plot 저장 완료: {contour_plot_path}")
print(f"Group bar plot 저장 완료: {group_bar_plot_path}")
print(df_group_dist.head())