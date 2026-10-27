"""
Reconstruction of the "pre-NAICS diagnostic pass" (originally committed
2026-08-11 as outputs/figures/01-09, PNGs only -- no generating .py file was
ever committed, and none was found anywhere in git history). This is NEW
code written to reproduce the same 9 diagnostics described in that commit
message, not recovered original code -- treat any specific chart styling as
this rewrite's choice, not a match to whatever the original looked like.

Covers, across all three raw FOIA files, BEFORE NAICS sector assignment
(i.e. on load_sba.load_raw() + normalize_categoricals() output, not
sector_eda's canonical loader):
  01. missingness by file
  02. duplicate-row position gaps
  03. TermInMonths==0 rate by LoanStatus
  04. BusinessAge distribution
  05. InitialInterestRate distribution
  06. TermInMonths==0 rate by RevolverStatus
  07. LoanStatus distribution
  08. GrossApproval distribution
  09. loan volume by ApprovalFY, across the FY2000-2009/2010-2019/2020+
      file boundaries

Run: python pre_naics_diagnostics_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "data_prep"))
import load_sba

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/figures"
ORDER = ["2000_2009", "2010_2019", "2020_Curr"]
COLORS = {"2000_2009": "#4C72B0", "2010_2019": "#DD8452", "2020_Curr": "#55A868"}


def load_all():
    frames = {}
    for name in ORDER:
        df = load_sba.load_raw(load_sba.FILES[name])
        load_sba.normalize_categoricals(df)
        frames[name] = df
    return frames


def chart01_missingness_by_file(frames):
    miss = pd.DataFrame({name: df.isna().mean() * 100 for name, df in frames.items()})
    miss = miss[(miss > 0).any(axis=1)]
    miss = miss.loc[miss.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(miss.values, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(len(miss.columns)))
    ax.set_xticklabels(miss.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(miss.index)))
    ax.set_yticklabels(miss.index, fontsize=7)
    ax.set_title("Missing % by column x file (raw, pre-NAICS)")
    for i in range(len(miss.index)):
        for j in range(len(miss.columns)):
            val = miss.values[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 50 else "black")
    fig.colorbar(im, ax=ax, label="missing %")
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_missingness_by_file.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/01_missingness_by_file.png")


def chart02_duplicate_row_gaps(frames):
    """Row-position gaps between consecutive duplicate rows -- clusters at
    small gaps would suggest duplicates come from adjacent chunks in the
    FOIA export rather than scattered coincidental repeats. Gaps are computed
    WITHIN each duplicate-content group only (rows that are copies of each
    other), not across unrelated duplicate groups that happen to sit near
    each other in the file -- mixing those in would inflate the range with
    gaps that don't represent a real repeat."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, name in zip(axes, ORDER):
        df = frames[name]
        dup_mask = df.duplicated(keep=False)
        dup_positions = np.flatnonzero(dup_mask.values)
        # .indices below returns positions LOCAL to the iloc'd slice (0..k-1,
        # since iloc resets the index) -- map back through dup_positions to
        # get each row's real position in the original file.
        groups = df.iloc[dup_positions].groupby(list(df.columns), sort=False, dropna=False).indices

        gaps = []
        for local_positions in groups.values():
            positions = np.sort(dup_positions[np.asarray(local_positions)])
            if len(positions) > 1:
                gaps.extend(np.diff(positions))
        gaps = np.array(gaps)

        if len(gaps):
            ax.hist(gaps, bins=40, color=COLORS[name])
        else:
            ax.text(0.5, 0.5, "no duplicate rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{name}\n(n={len(gaps)} gaps, {len(groups)} groups)")
        ax.set_xlabel("row-position gap between duplicate copies")
    axes[0].set_ylabel("count")
    fig.suptitle("Row-position gaps within exact-duplicate row groups (raw, pre-NAICS)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_duplicate_row_gaps.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/02_duplicate_row_gaps.png")


def _grouped_bar_by_file(frames, col, ax, title, ylabel, normalize_within=None):
    """Shared helper: grouped bar of `col`'s category mix (or a rate over
    `col`) across the 3 files. If normalize_within is given, plots the %
    of TermInMonths==0 within each category of `col` instead of the
    category's share of the file."""
    cats = sorted(set().union(*[set(df[col].dropna().unique()) for df in frames.values()]))
    x = np.arange(len(cats))
    width = 0.25
    for i, name in enumerate(ORDER):
        df = frames[name]
        if normalize_within == "term_zero_rate":
            vals = df.groupby(col)["TermInMonths"].apply(lambda s: (s == 0).mean() * 100)
        else:
            vals = df[col].value_counts(normalize=True) * 100
        vals = vals.reindex(cats).fillna(0)
        ax.bar(x + i * width, vals.values, width, label=name, color=COLORS[name])
    ax.set_xticks(x + width)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return cats


def chart03_termzero_by_loanstatus(frames):
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar_by_file(frames, "LoanStatus", ax,
                          "TermInMonths==0 rate by LoanStatus (raw, pre-NAICS)",
                          "% of loans with TermInMonths==0", normalize_within="term_zero_rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_term_zero_by_loanstatus.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/03_term_zero_by_loanstatus.png")


def chart04_businessage_distribution(frames):
    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bar_by_file(frames, "BusinessAge", ax,
                          "BusinessAge distribution by file (raw, pre-reconciliation)",
                          "% of loans in file")
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_businessage_distribution.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/04_businessage_distribution.png")


def chart05_interest_rate_distribution(frames):
    fig, ax = plt.subplots(figsize=(9, 5))
    all_vals = pd.concat([frames[n]["InitialInterestRate"].dropna() for n in ORDER])
    bins = np.linspace(all_vals.min(), all_vals.quantile(0.995), 61)
    for name in ORDER:
        col = frames[name]["InitialInterestRate"]
        vals = col.dropna()
        pct_missing = col.isna().mean() * 100
        ax.hist(vals, bins=bins, alpha=0.5, color=COLORS[name], edgecolor="none",
                label=f"{name} (n={len(vals):,}, {pct_missing:.1f}% missing)")
    ax.set_xlabel("InitialInterestRate (%)")
    ax.set_ylabel("count")
    ax.set_title("InitialInterestRate distribution by file (raw, overlapping)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/05_interest_rate_distribution.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/05_interest_rate_distribution.png")


def chart06_termzero_vs_revolverstatus(frames):
    fig, ax = plt.subplots(figsize=(7, 5))
    _grouped_bar_by_file(frames, "RevolverStatus", ax,
                          "TermInMonths==0 rate by RevolverStatus (raw, pre-NAICS)",
                          "% of loans with TermInMonths==0", normalize_within="term_zero_rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_termzero_vs_revolverstatus.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/06_termzero_vs_revolverstatus.png")


def chart07_loanstatus_distribution(frames):
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar_by_file(frames, "LoanStatus", ax,
                          "LoanStatus distribution by file (raw)",
                          "% of loans in file")
    fig.tight_layout()
    fig.savefig(f"{OUT}/07_loanstatus_distribution.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/07_loanstatus_distribution.png")


def chart08_grossapproval_distribution(frames):
    """log10 scale -- GrossApproval is heavily right-skewed (a handful of
    loans in the millions next to a mass under $100K), so a linear axis
    crushes the whole shape into one tall bar near zero."""
    fig, ax = plt.subplots(figsize=(9, 5))
    log_vals_all = pd.concat([np.log10(frames[n]["GrossApproval"].dropna()) for n in ORDER])
    bins = np.linspace(log_vals_all.min(), log_vals_all.max(), 61)
    for name in ORDER:
        vals = frames[name]["GrossApproval"].dropna()
        log_vals = np.log10(vals)
        ax.hist(log_vals, bins=bins, alpha=0.5, color=COLORS[name], edgecolor="none",
                label=f"{name} (n={len(vals):,}, median=${vals.median():,.0f})")
    ax.set_xlabel("log10(GrossApproval)")
    ax.set_ylabel("count")
    ax.set_title("GrossApproval distribution by file (raw, log10 scale, overlapping)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/08_grossapproval_distribution.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/08_grossapproval_distribution.png")


def chart09_volume_by_approvalfy(frames):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in ORDER:
        df = frames[name]
        counts = df["ApprovalFY"].value_counts().sort_index()
        ax.plot(counts.index, counts.values, marker="o", label=name, color=COLORS[name])
    ax.set_xlabel("ApprovalFY")
    ax.set_ylabel("loan count")
    ax.set_title("Loan volume by ApprovalFY, across file boundaries (raw)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/09_volume_by_approvalfy.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/09_volume_by_approvalfy.png")


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    frames = load_all()
    for name, df in frames.items():
        print(f"{name}: {len(df):,} rows loaded")
    print()

    chart01_missingness_by_file(frames)
    chart02_duplicate_row_gaps(frames)
    chart03_termzero_by_loanstatus(frames)
    chart04_businessage_distribution(frames)
    chart05_interest_rate_distribution(frames)
    chart06_termzero_vs_revolverstatus(frames)
    chart07_loanstatus_distribution(frames)
    chart08_grossapproval_distribution(frames)
    chart09_volume_by_approvalfy(frames)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
