"""
PSI (Population Stability Index): FY2010+ (reference) vs. FY2000-2009
(comparison), on the FROZEN PD model's predicted-score distribution.

Formal, numeric follow-up to the qualitative population-instability story (easy credit and looser underwriting pre-2008 vs. near-zero rates, scarce capital and tighter underwriting post-2008) -- does
the SCORE DISTRIBUTION actually shift as much as that economic story
implies?

PSI, per bucket: (%comparison - %reference) x ln(%comparison / %reference).
Summed across buckets. Rule-of-thumb reading: <0.1 no real shift,
0.1-0.25 moderate, >0.25 major population shift. Also saves a chart of the
per-bucket shift (12_psi_bucket_shift.png).

Kept COMPLETELY SEPARATE from every other script, including
backtest_2000_2009.py -- only imports and calls existing functions (from
target_construction.py and backtest_2000_2009.py itself), edits nothing.
Deletable with zero effect elsewhere.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from target_construction import load_model_base, remove_young_loans, remove_cancelled_and_committed, assign_target
from backtest_2000_2009 import build_backtest_population, score_backtest_population

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_BUCKETS = 10
EPS = 1e-6  # avoids log(0) / division by zero in near-empty buckets
OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")


def score_reference_population():
    """FY2010+ (the population the PD model was actually trained/tested
    on), scored through the SAME frozen-model function backtest_2000_2009.py
    already uses -- that function takes any df, so it's reused as-is here,
    not duplicated."""
    raw = load_model_base()
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df)
    df = assign_target(df, n_months=60)
    predicted_pd = score_backtest_population(df)
    return predicted_pd


def compute_psi(reference_scores, comparison_scores, n_buckets=N_BUCKETS):
    """Bucket edges are defined on the REFERENCE population's score
    distribution (standard PSI convention), then the SAME edges are
    applied to the comparison population."""
    edges = np.quantile(reference_scores, np.linspace(0, 1, n_buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # so every score falls in some bucket

    ref_counts, _ = np.histogram(reference_scores, bins=edges)
    comp_counts, _ = np.histogram(comparison_scores, bins=edges)

    ref_pct = ref_counts / ref_counts.sum() + EPS
    comp_pct = comp_counts / comp_counts.sum() + EPS

    psi_per_bucket = (comp_pct - ref_pct) * np.log(comp_pct / ref_pct)

    table = pd.DataFrame({
        "bucket_upper_edge": edges[1:],
        "reference_pct": ref_pct - EPS,
        "comparison_pct": comp_pct - EPS,
        "psi_contribution": psi_per_bucket,
    })
    return table, psi_per_bucket.sum()


def plot_psi(table, psi_total, out_path):
    """Left: share of loans in each predicted-PD bucket, reference vs.
    comparison (the reference is ~10% per bucket by construction).
    Right: how much each bucket adds to the total PSI."""
    x = np.arange(1, len(table) + 1)
    width = 0.4
    fig, (ax_share, ax_contrib) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_share.bar(x - width / 2, table["reference_pct"] * 100, width,
                 label="FY2010+ (reference)", color="#4C72B0")
    ax_share.bar(x + width / 2, table["comparison_pct"] * 100, width,
                 label="FY2000-2009 (comparison)", color="#DD8452")
    ax_share.set_xticks(x)
    ax_share.set_xlabel("predicted-PD bucket (1 = safest, 10 = riskiest)")
    ax_share.set_ylabel("% of loans in bucket")
    ax_share.set_title("Share of loans per bucket")
    ax_share.legend()

    ax_contrib.bar(x, table["psi_contribution"], color="#7F7F7F")
    ax_contrib.set_xticks(x)
    ax_contrib.set_xlabel("predicted-PD bucket (1 = safest, 10 = riskiest)")
    ax_contrib.set_ylabel("contribution to PSI")
    ax_contrib.set_title(f"Contribution to total PSI ({psi_total:.3f})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("scoring FY2010+ reference population...")
    reference_scores = score_reference_population()
    print(f"reference population: {len(reference_scores)} loans, mean predicted PD "
          f"{reference_scores.mean()*100:.2f}%")

    print("\nscoring FY2000-2009 comparison population...")
    comparison_df = build_backtest_population()
    comparison_scores = score_backtest_population(comparison_df)
    print(f"comparison population: {len(comparison_scores)} loans, mean predicted PD "
          f"{comparison_scores.mean()*100:.2f}%")

    table, psi_total = compute_psi(reference_scores, comparison_scores)

    print(f"\n=== PSI table, by predicted-PD bucket (edges from reference population deciles) ===")
    print(table.round(4).to_string())

    print(f"\n=== TOTAL PSI: {psi_total:.4f} ===")
    if psi_total < 0.1:
        reading = "no real shift"
    elif psi_total < 0.25:
        reading = "moderate shift, worth flagging"
    else:
        reading = "MAJOR population shift"
    print(f"rule-of-thumb reading: {reading}")

    Path(OUT).mkdir(parents=True, exist_ok=True)
    plot_psi(table, psi_total, f"{OUT}/12_psi_bucket_shift.png")
    print(f"\nsaved {OUT}/12_psi_bucket_shift.png")

    return {"psi_total": psi_total, "table": table}


if __name__ == "__main__":
    main()
