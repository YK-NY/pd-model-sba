"""
Checks all three raw FOIA files for column misalignment -- rows where a
value has landed in the wrong column. Can happen when something goes
wrong while a file is exported (e.g. a stray character in a free-text
field confuses the process writing the file).

Not part of the pipeline. Doesn't touch or depend on anything else in
the project. Safe to delete, or keep as a standing data-quality check.

Two checks, run per file:

1. Did every row keep its shape? A row should have exactly as many
   values as the header does. Any row with too many or too few is
   direct, unambiguous proof something broke -- no guessing needed.

2. Does each column actually hold what it's supposed to? Some columns
   can only ever hold a few valid answers (open/closed, yes/no, a short
   list of loan types). Anything outside that list gets flagged. One
   flag alone is often just an honest oddity in the data. Several flags
   on the SAME row is the real warning sign -- that's what it looks
   like when everything after some point has shifted sideways, so every
   column downstream is holding the wrong kind of value.

RESULT (2026-09-25): all three files checked, all three clean. No
field-count mismatches, and no row fails more than one value check at
once, in any file. The only flags raised were older/retired SBA loan
program names not in this script's initial vocabulary list -- confirmed
legitimate by checking what values actually exist in each file, not
assumed from documentation. This had apparently been checked before, but the result was
never written down, so it resurfaced as an open question. Written down
here now so that doesn't happen again.

Run: python check_column_alignment.py
"""
import csv
from pathlib import Path

import pandas as pd

DATA_RAW = Path(__file__).resolve().parent / "data" / "raw"

FILES = {
    "2000_2009": DATA_RAW / "FOIA_7a_FY2000_FY2009_asof_260630.csv",
    "2010_2019": DATA_RAW / "FOIA_7a_FY2010_FY2019_asof_260630.csv",
    "2020_Curr": DATA_RAW / "FOIA_7a_FY2020_Present_asof_260630.csv",
}

# columns that can only hold a small, known set of values.
# NOTE: these raw files use full descriptive strings, not short codes, for
# ProcessingMethod (confirmed by inspecting actual unique values first --
# don't assume the data dictionary's short codes are what's on disk).
FIXED_VOCAB = {
    "LoanStatus": {"PIF", "CHGOFF", "CANCLD", "EXEMPT", "COMMIT"},
    "Program": {"7A"},
    "BusinessType": {"INDIVIDUAL", "PARTNERSHIP", "CORPORATION"},
    "CollateralInd": {"Y", "N"},
    "SoldSecMrktInd": {"Y", "N"},
    "FixedorVariableInterestInd": {"F", "V"},
    "ProcessingMethod": {
        "PREFERREDLENDERSPROGRAM", "SBAEXPRESSPROGRAM", "7AGENERAL",
        "COMMUNITYADVANTAGEINITIATIVE", "WORKINGCAPITALCAPLINE",
        "INTERNATIONALTRADELOANS", "PREFERREDLENDERSWITHEWCP",
        "CONTRACTCAPLINE", "7AWITHEWCP", "EXPORTEXPRESS", "7(A)WCP",
        "BUILDERSCAPLINE", "SEASONALCAPLINE",
        "COMMUNITYADVANTAGERECOVERYLOAN", "PREFERREDLENDERSWITHWCP",
        "PLPMARC", "COMMUNITYADVANTAGEINTERNATIONALTRADE", "MARC7AGENERAL",
        "COMMUNITYADVANTAGERLOC",
    },
}
# RevolverStatus: data dictionary says 0/1, but a known documented mismatch
# says the file actually uses Y/N -- accept both rather than assume.
FIXED_VOCAB["RevolverStatus"] = {"0", "1", "Y", "N"}

# columns that must parse as a plain number
NUMERIC_COLS = ["GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate",
                 "TermInMonths", "JobsSupported", "ApprovalFY", "GrossChargeOffAmount"]

# columns that must parse as a date (or be genuinely blank)
DATE_COLS = ["AsOfDate", "ApprovalDate", "FirstDisbursementDate",
             "PaidInFullDate", "ChargeOffDate"]


def check_field_counts(path, sample_rows=5):
    """Every data row should have the same number of fields as the header,
    once parsed with real CSV quoting rules (not a naive .split(','))."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_expected = len(header)
        bad = []
        for i, row in enumerate(reader, start=2):  # line 1 is the header
            if len(row) != n_expected:
                bad.append((i, len(row)))
                if len(bad) >= sample_rows:
                    break
        return n_expected, bad


def check_value_sanity(df):
    """Per-row failure count across fixed-vocab, numeric, and date columns.
    Returns a per-column failure count and the rows with 2+ failures."""
    fail_flags = pd.DataFrame(index=df.index)

    for col, valid in FIXED_VOCAB.items():
        if col not in df.columns:
            continue
        # strip ALL whitespace (not just leading/trailing), same as
        # normalize_categoricals does -- "P I F" and "PIF" are the same
        # value, not evidence of anything.
        cleaned = df[col].astype("string").str.replace(r"\s+", "", regex=True).str.upper()
        fail_flags[col] = cleaned.notna() & ~cleaned.isin(valid)

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        as_num = pd.to_numeric(df[col], errors="coerce")
        fail_flags[col] = df[col].notna() & as_num.isna()

    for col in DATE_COLS:
        if col not in df.columns:
            continue
        as_date = pd.to_datetime(df[col], errors="coerce")
        fail_flags[col] = df[col].astype("string").str.strip().notna() & \
                           (df[col].astype("string").str.strip() != "") & as_date.isna()

    per_col_fail_count = fail_flags.sum().sort_values(ascending=False)
    per_row_fail_count = fail_flags.sum(axis=1)
    multi_fail_rows = df.index[per_row_fail_count >= 2]

    return per_col_fail_count, multi_fail_rows, fail_flags


def run_one(name, path):
    print(f"\n{'=' * 70}\n{name}  ({path.name})\n{'=' * 70}")

    print("\n--- 1. field-count check (raw CSV parsing) ---")
    n_expected, bad = check_field_counts(path)
    print(f"expected fields per row (from header): {n_expected}")
    if bad:
        print(f"rows with wrong field count (first {len(bad)} found): {bad}")
    else:
        print("no field-count mismatches found in the rows scanned")

    print("\n--- 2. value-sanity check (all columns loaded as plain text) ---")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    per_col_fail_count, multi_fail_rows, fail_flags = check_value_sanity(df)

    print("failures per column (value doesn't fit what that column should hold):")
    print(per_col_fail_count[per_col_fail_count > 0].to_string())

    print(f"\nrows failing 2+ checks at once (strongest shift suspects): {len(multi_fail_rows)}")
    if len(multi_fail_rows) > 0:
        show = multi_fail_rows[:5]
        cols_to_show = ["LocationID", "LoanStatus", "RevolverStatus", "ProcessingMethod",
                         "CollateralInd", "GrossApproval", "ApprovalDate"]
        cols_to_show = [c for c in cols_to_show if c in df.columns]
        print(f"first {len(show)} such rows:")
        print(df.loc[show, cols_to_show].to_string())

    return {
        "file": name,
        "field_count_mismatches": len(bad),
        "columns_with_failures": int((per_col_fail_count > 0).sum()),
        "rows_with_2plus_failures": len(multi_fail_rows),
        "total_rows": len(df),
    }


if __name__ == "__main__":
    summary = []
    # FY2020+ first -- trying to rediscover the already-confirmed issue.
    # Then FY2010-2019 -- the one that was never row-level checked.
    # Then FY2000-2009 -- for completeness, all three files this project uses.
    for name in ["2020_Curr", "2010_2019", "2000_2009"]:
        summary.append(run_one(name, FILES[name]))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(pd.DataFrame(summary).to_string(index=False))
