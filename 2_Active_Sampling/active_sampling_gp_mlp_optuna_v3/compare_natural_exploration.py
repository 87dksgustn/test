"""
Compare barrier_thx distributions across iterations.
Tests hypothesis: without hard sparse coverage constraint, does natural 
uncertainty-driven exploration provide balanced coverage?
"""
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path("outputs")

def analyze_distribution(try_num, csv_path):
    """Analyze barrier_thx distribution for a Try"""
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    thx = df["C_Barrier_Thx"].values
    
    # Define 5 zones
    min_val, max_val = 0.25, 2.5
    n_bins = 5
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    print(f"\n{'='*60}")
    print(f"Try_{try_num} - barrier_thx distribution (batch_size={len(df)})")
    print(f"{'='*60}")
    
    total = len(thx)
    for zone_idx in range(n_bins):
        bin_start, bin_end = bin_edges[zone_idx], bin_edges[zone_idx + 1]
        count = np.sum((thx >= bin_start) & (thx < bin_end))
        pct = 100.0 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"[{bin_start:5.2f}-{bin_end:5.2f}]: {count:3d} ({pct:5.1f}%) {bar}")
    
    print()
    return thx

# Check all Try_X directories
tries = []
for try_dir in sorted(output_dir.glob("Try_*")):
    try_num = int(try_dir.name.split("_")[1])
    csv_path = try_dir / "next_sampling_candidates.csv"
    
    if csv_path.exists():
        thx = analyze_distribution(try_num, csv_path)
        if thx is not None:
            tries.append((try_num, thx))

print("\n" + "="*60)
print("SUMMARY: Are distributions different across iterations?")
print("="*60)

if len(tries) > 1:
    # Compare first and last
    t1, t1_thx = tries[0]
    tn, tn_thx = tries[-1]
    
    # Binning
    bin_edges = np.linspace(0.25, 2.5, 6)
    t1_binned = np.digitize(t1_thx, bin_edges[:-1]) - 1
    tn_binned = np.digitize(tn_thx, bin_edges[:-1]) - 1
    
    # KL divergence to measure shift
    from scipy.stats import entropy
    
    t1_counts = np.bincount(t1_binned, minlength=5)
    tn_counts = np.bincount(tn_binned, minlength=5)
    
    t1_probs = t1_counts / t1_counts.sum()
    tn_probs = tn_counts / tn_counts.sum()
    
    kl_div = entropy(tn_probs, t1_probs)
    print(f"\nKL divergence Try_{t1} → Try_{tn}: {kl_div:.4f}")
    if kl_div > 0.1:
        print("✓ Distributions ARE shifting (exploration working)")
    else:
        print("✗ Distributions are stable/similar (exploration limited)")
else:
    print(f"Only {len(tries)} try(ies) found; need at least 2 for comparison")
