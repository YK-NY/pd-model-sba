"""
Consolidated load + preliminary EDA for the SBA 7(a) FOIA files.
The same sequence of checks runs against the two raw files (FY2010-2019 and FY2020-current).
This module holds that shared logic so it runs once, against both files
"""

import re
import pandas as pd
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Config shared by both files
# ---------------------------------------------------------------------------

# resolved relative to this file, not the caller's cwd, so this runs the
# same whether invoked as `python src/data_prep/load_sba.py` from repo root
# or imported from a notebook under notebooks/
_DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

FILES = {
    "2000_2009": str(_DATA_RAW / "FOIA_7a_FY2000_FY2009_asof_260630.csv"),
    "2010_2019": str(_DATA_RAW / "FOIA_7a_FY2010_FY2019_asof_260630.csv"),
    "2020_Curr": str(_DATA_RAW / "FOIA_7a_FY2020_Present_asof_260630.csv"),
}

# identifier like fields that may look numeric, but must be strings (zipcode, naicscode etc.. ) are cast to the correct dtype on load
DTYPE_MAP = {
    "LocationID": "string",
    "BorrZip": "string",
    "BankFDICNumber": "string",
    "BankNCUANumber": "string",
    "BankZip": "string",
    "NaicsCode": "string",
    "FranchiseCode": "string",
    "SBADistrictOffice": "string",
    "CongressionalDistrict": "string",
    "Program": "string",
    "ProcessingMethod": "string",
    "LoanStatus": "string",
    "FixedorVariableInterestInd": "string",
    "RevolverStatus": "string",
    "CollateralInd": "string",
    "SoldSecMrktInd": "string",
}

DATE_COLS = ["AsOfDate", "ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"]

CANDIDATE_ID_COLS = ["LocationID", "BorrZip", "BankFDICNumber", "BankNCUANumber", "BankZip"]

VALID_LOAN_STATUS = {"PIF", "CHGOFF", "CANCLD", "EXEMPT", "COMMIT"}

# any whitespace character (space, tab, non-breaking space \xa0, etc.),
# not just plain " " — plain " ".replace(" ", "") misses these
_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path):
    return pd.read_csv(path, dtype=DTYPE_MAP, parse_dates=DATE_COLS, low_memory=False)


def check_bom(path):
    """
    BOM - Byte Order Mark character
    Detect a UTF-8 BOM at the start of the file. If present and not handled,
    pandas silently prepends '\\ufeff' to the FIRST column name only — every
    later reference to that column (e.g. df["Program"]) then raises a
    KeyError that looks like a typo, not an encoding issue. Easy to miss
    since the column shows up fine in df.head() / most viewers.
    """
    with open(path, "rb") as f:
        head = f.read(3)
    has_bom = head == b"\xef\xbb\xbf"
    print(f"BOM at file start: {has_bom}")
    return has_bom


def clean_column_names(df):
    """
    Strip a stray BOM prefix and any whitespace (leading/trailing/internal)
    from column names, so df["SomeColumn"] never silently fails because the
    header actually reads "SomeColumn " or "\\ufeffSomeColumn".
    """
    before = list(df.columns)
    cleaned = (
        df.columns
        .str.replace("﻿", "", regex=False)
        .map(lambda c: _WHITESPACE_RE.sub("", c))
    )
    df.columns = cleaned
    changed = [(b, a) for b, a in zip(before, cleaned) if b != a]   # keep only column names that had changes
    if changed:
        print(f"column names changed by cleanup: {changed}")
    else:
        print("column names: no cleanup needed")
    return df


# ---------------------------------------------------------------------------
# Structural validation on load
# ---------------------------------------------------------------------------

def check_row_count(path, df):
    # csv.reader (not len(open().readlines())) so quoted newlines inside
    # fields don't inflate the raw count
    with open(path, newline="", encoding="utf-8") as f:
        raw_row_count = sum(1 for _ in csv.reader(f)) - 1  # minus header ; counting rows in the raw file
    match = raw_row_count == len(df)
    print(f"raw rows: {raw_row_count}  |  df rows: {len(df)}  |  match: {match}")
    return match


def check_dtypes(df):
    print(df.dtypes[list(DTYPE_MAP.keys()) + DATE_COLS])


def check_id_cardinality(df):
    """
    nunique() against row count for every plausible 'ID-like' column, so
    none of them get assumed to be a unique key (or assumed not to be)
    without actually checking.
    """
    for col in CANDIDATE_ID_COLS:
        if col in df.columns:
            print(f"{col}: nunique={df[col].nunique()}  rows={len(df)}  ratio={df[col].nunique() / len(df):.4f}")


def check_full_duplicate_rows(df):
    """A fully duplicated row is easy to miss — it doesn't fail any single-column check."""
    n_dupes = df.duplicated().sum()
    print(f"fully duplicated rows: {n_dupes}")
    return n_dupes


# ---------------------------------------------------------------------------
# Categorical normalization
# ---------------------------------------------------------------------------

def normalize_categoricals(df):
    """
    Every column in DTYPE_MAP is a fixed-vocabulary code field — none of
    them should legitimately contain whitespace. Strip ALL whitespace
    (leading, trailing, AND in between, e.g. "P I F" -> "PIF", not just
    " PIF " -> "PIF") and uppercase, before anything else touches them.

    Also converts any resulting empty string to NaN — a whitespace-only
    cell (" ") looks non-null to .isna() before stripping, then silently
    becomes "" (still non-null) after a naive strip unless handled here,
    quietly deflating missingness stats.
    """
    for col in DTYPE_MAP:
        before_nunique = df[col].nunique(dropna=False)
        df[col] = df[col].str.replace(_WHITESPACE_RE, "", regex=True).str.upper()
        df[col] = df[col].replace("", pd.NA)
        after_nunique = df[col].nunique(dropna=False)
        if before_nunique != after_nunique:
            print(f"{col}: nunique changed {before_nunique} -> {after_nunique} after whitespace/case normalization "
                  f"(hidden duplicate categories collapsed)")
    return df


def check_categorical_values(df):
    print("LoanStatus:\n", df["LoanStatus"].value_counts(dropna=False), "\n")
    unexpected_status = set(df["LoanStatus"].dropna().unique()) - VALID_LOAN_STATUS
    print("unexpected LoanStatus values:", unexpected_status)

    print("\nRevolverStatus:\n", df["RevolverStatus"].value_counts(dropna=False))
    # documented mismatch: data dictionary says 0/1, actual values are Y/N — confirm here each time
    print("\nProgram:\n", df["Program"].value_counts(dropna=False))
    print("\nProcessingMethod:\n", df["ProcessingMethod"].value_counts(dropna=False))
    print("\nFixedorVariableInterestInd:\n", df["FixedorVariableInterestInd"].value_counts(dropna=False))
    print("\nCollateralInd:\n", df["CollateralInd"].value_counts(dropna=False))
    print("\nSoldSecMrktInd:\n", df["SoldSecMrktInd"].value_counts(dropna=False))


# all check_* functions check for a particular feature or issue and ONLY actual issues found are addressed, 
def check_sentinel_strings(df):
    """
    Values like the literal text "NA", "NULL", "N/A", "NONE" read as valid
    strings, not as missing — .isna() won't catch them. Flag any of these
    surviving inside the normalized categorical columns.
    """
    sentinels = {"NA", "N/A", "NULL", "NONE", "NAN", "-", "."}
    for col in DTYPE_MAP:
        hits = df[col].dropna().unique()
        found = sorted(set(hits) & sentinels)
        if found:
            print(f"{col}: sentinel-like string values present (not recognized as missing): {found}")


# ---------------------
# to address any found sentinels in future datasets
# df[col] = df[col].replace(sentinels, pd.NA)
# ---------------------

# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

def check_missing_data(df):
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(missing_pct)

    # extra attention on the fields with a known history of issues
    for col in ["ChargeOffDate", "PaidInFullDate", "GrossChargeOffAmount"]:
        if col in df.columns:
            print(f"{col}: {df[col].isna().mean() * 100:.2f}% missing")


# ---------------------------------------------------------------------------
# Known data-quality checks carried over from the notebooks
# ---------------------------------------------------------------------------

def check_chargeoff_date_by_status(df):
    result = df.groupby("LoanStatus")["ChargeOffDate"].agg(
        total_loans="size",  # size, not count, so it includes NaN rows too
        populated_dates=lambda s: s.notna().sum(),
    )
    print(result)
    return result


def check_pif_date_by_status(df):
    result = df.groupby("LoanStatus")["PaidInFullDate"].agg(
        total_loans="size",
        populated_dates=lambda s: s.notna().sum(),
    )
    print(result)
    return result


def check_chgoff_missing_date(df):
    n = len(df[(df["LoanStatus"] == "CHGOFF") & (df["ChargeOffDate"].isna())])
    print(f"CHGOFF rows missing ChargeOffDate: {n}")
    return n


def check_chgoff_amt_on_nonchgoff(df):
    n = len(df[(df["LoanStatus"] != "CHGOFF") & (df["GrossChargeOffAmount"].notna()) & (df["GrossChargeOffAmount"] != 0)])
    print(f"non-CHGOFF rows with a nonzero GrossChargeOffAmount: {n}")
    return n


def check_chgoff_missing_amt(df):
    n = len(df[(df["LoanStatus"] == "CHGOFF") & (df["GrossChargeOffAmount"].isna())])
    print(f"CHGOFF rows missing GrossChargeOffAmount: {n}")
    return n


def check_chgoff_before_approval(df):
    n = len(df[df["ChargeOffDate"] < df["ApprovalDate"]])
    print(f"ChargeOffDate before ApprovalDate: {n}")
    return n


def check_pif_before_approval(df):
    n = len(df[df["PaidInFullDate"] < df["ApprovalDate"]])
    print(f"PaidInFullDate before ApprovalDate: {n}")
    return n


def check_term_zero(df):
    term_zero_count = (df["TermInMonths"] == 0).sum()
    print("Loans with term=0:", term_zero_count)
    print(df.loc[df["TermInMonths"] == 0, "ProcessingMethod"].value_counts())
    return term_zero_count


# ---------------------------------------------------------------------------
# Fresh checks: the non-obvious checks
# ---------------------------------------------------------------------------

def check_date_parse_failures(df):
    """
    parse_dates doesn't error on an unparseable value — it just leaves NaT.
    A date column with unexpectedly high NaT vs the raw file's non-blank
    cell count means the parse silently failed for some rows.
    """
    for col in DATE_COLS:
        if col in df.columns:
            n_nat = df[col].isna().sum()
            print(f"{col}: {n_nat} NaT out of {len(df)} rows ({n_nat / len(df) * 100:.2f}%)")


def check_numeric_columns_for_stray_text(df):
    """
    Columns pandas inferred as object/string that "look" numeric (mostly
    digits) can hide a handful of rows with stray text (e.g. a footnote
    character) that silently blocks a later pd.to_numeric() cast. Flag any
    string-dtype, non-DTYPE_MAP column where the vast majority of values are
    numeric-looking but a small minority aren't.
    """
    for col in df.columns:
        if col in DTYPE_MAP or col in DATE_COLS:
            continue
        if df[col].dtype in ("object", "string"):
            sample = df[col].dropna().astype(str)
            if len(sample) == 0:
                continue
            looks_numeric = sample.str.match(r"^-?\d+(\.\d+)?$")
            frac_numeric = looks_numeric.mean()  # mean on a boolean series gives true over a threshold. 
            if 0.9 < frac_numeric < 1.0:
                bad_examples = sample[~looks_numeric].unique()[:5]
                print(f"{col}: {frac_numeric*100:.1f}% look numeric, non-numeric outliers e.g. {list(bad_examples)}")


def check_zip_code_length(df):
    """
    BorrZip/BankZip are loaded as `string` specifically to preserve leading
    zeros (e.g. "02110"). Flag any entries that don't look like a valid
    5 or 9-digit zip — a sign leading zeros were lost upstream, before this
    load, or that non-US formats snuck in.
    """
    for col in ["BorrZip", "BankZip"]:
        if col in df.columns:
            vals = df[col].dropna()
            bad = vals[~vals.str.match(r"^\d{5}(\d{4})?$")]
            print(f"{col}: {len(bad)} values not matching 5 or 9-digit zip format, e.g. {list(bad.unique()[:5])}")


# after the check, the confirmed issues are addressed and duplciate rows dropped.
def clean_confirmed_issues(df, verbose=True):
    """
    Applies the cleaning decisions that have actually been confirmed and
    settled.
    Assumes normalize_categoricals(df) has already been run.

    Each rule is applied and counted separately so the effect of any single
    rule is auditable, not folded into one opaque row-count delta.
    """
    start = len(df)

    # impossible date ordering
    df = df[~(df["PaidInFullDate"] < df["ApprovalDate"])]
    df = df[~(df["ChargeOffDate"] < df["ApprovalDate"])]

    # EXEMPT should mean "disbursed, not yet resolved" -- a populated
    # PaidInFullDate or ChargeOffDate on an EXEMPT row contradicts that
    df = df[~((df["LoanStatus"] == "EXEMPT") & (df["PaidInFullDate"].notna() | df["ChargeOffDate"].notna()))]

    # fully duplicated rows (FOIA export artifact)
    df = df[~df.duplicated(keep="first")]

    # non-US BankZip entries (small count, real data but out of scope)
    if "BankZip" in df.columns:
        us_zip = df["BankZip"].str.match(r"^\d{5}(\d{4})?$")
        df = df[us_zip.fillna(True)]  # keep rows with missing BankZip, drop only confirmed non-US format

    # implausible InitialInterestRate (>20% has no plausible Prime+spread
    # explanation across the whole modeling window; found only in 2000-09,
    # both rows CANCLD)
    if "InitialInterestRate" in df.columns:
        df = df[~(df["InitialInterestRate"] > 20)]

    # implausible GrossApproval (<$1,000 is below any plausible SBA 7(a)
    # minimum; found only in 2000-09, clustered in SBAEXPRESSPROGRAM with
    # round-number values -- looks like a units/scaling error, not
    # corrected because the true multiplier isn't confirmed)
    df = df[~(df["GrossApproval"] < 1000)]

    if verbose:
        print(f"clean_confirmed_issues: {start} -> {len(df)} rows ({start - len(df)} dropped)")

    return df.copy()


# place holder target column, for completion; NOT FINAL.
def assign_target(df, status_col="LoanStatus"):
    """
    Binary target: 1 = CHGOFF, 0 = PIF, NA for every other status
    (CANCLD/EXEMPT/COMMIT). This does NOT filter rows, just labels
    the two statuses that are unambiguous today.
    """
    df["target"] = df[status_col].map({"CHGOFF": 1, "PIF": 0})
    return df


def run_data_quality_checks(name, df, path):
    print(f"\n\n{'='*30} {name}: LOADING / STRUCTURAL {'='*30}")
    check_bom(path)
    clean_column_names(df)
    check_row_count(path, df)
    check_dtypes(df)
    check_id_cardinality(df)
    check_full_duplicate_rows(df)
    check_date_parse_failures(df)
    check_numeric_columns_for_stray_text(df)
    check_zip_code_length(df)

    print(f"\n\n{'='*30} {name}: CATEGORICAL NORMALIZATION {'='*30}")
    normalize_categoricals(df)
    check_categorical_values(df)
    check_sentinel_strings(df)

    print(f"\n\n{'='*30} {name}: MISSING DATA {'='*30}")
    check_missing_data(df)

    print(f"\n\n{'='*30} {name}: KNOWN ISSUE CHECKS {'='*30}")
    check_chargeoff_date_by_status(df)
    check_pif_date_by_status(df)
    check_chgoff_missing_date(df)
    check_chgoff_amt_on_nonchgoff(df)
    check_chgoff_missing_amt(df)
    check_chgoff_before_approval(df)
    check_pif_before_approval(df)
    check_term_zero(df)

    return df


def load_and_check(name, path):
    df = load_raw(path)
    print(f"{name}: loaded shape {df.shape}")
    run_data_quality_checks(name, df, path)
    return df


def main():
    dfs = {}
    for name, path in FILES.items():
        dfs[name] = load_and_check(name, path)
    return dfs


if __name__ == "__main__":
    main()
