"""
BusinessAge bucket reconciliation across the fine-grained (pre-2018) and
coarse (2018+) SBA coding schemes.

Collapses the fine-grained buckets DOWN into the coarse scheme (the only
direction possible -- coarse buckets can't be un-split). "Change of
Ownership" has no pre-2018 equivalent and is kept as its own category
rather than folded into an age bucket (agreed: it's too large a group to
drop, and conceptually orthogonal (different info) to business age) -- pre-2018 loans that
were genuinely ownership changes are invisibly absorbed into whichever age
bucket they were recorded under; that's a stated, unrecoverable limitation,
not something this mapping can fix.
"""
import pandas as pd

# source value (as it appears in the raw data, after normalize_categoricals'
# whitespace/case handling would NOT apply -- BusinessAge is free text, not
# in DTYPE_MAP) -> target bucket
_BUSINESS_AGE_MAP = {
    # fine-grained (pre-2018) -> coarse target
    "New, Less than 1 Year old": "New Business or 2 years or less",
    "Less than 3 years old but at least 2": "New Business or 2 years or less",
    "Less than 4 years old but at least 3": "Existing or more than 2 years old",
    "Less than 5 years old but at least 4": "Existing or more than 2 years old",
    "Existing, 5 or more years": "Existing or more than 2 years old",
    # coarse (2018+) -> identity
    "New Business or 2 years or less": "New Business or 2 years or less",
    "Existing or more than 2 years old": "Existing or more than 2 years old",
    "Startup, Loan Funds will Open Business": "Startup, Loan Funds will Open Business",
    "Change of Ownership": "Change of Ownership",
    "Unanswered": "Unanswered",
}


def reclassify_business_age(df, col="BusinessAge"):
    """
    Adds BusinessAgeBucket -- the reconciled category -- alongside the
    original BusinessAge column (kept for traceability, not overwritten).
    Unmapped/unexpected values become "Unclassified" rather than raising,
    so a new/unseen value surfaces as a visible bucket instead of a crash.
    """
    mapped = df[col].map(_BUSINESS_AGE_MAP)
    df["BusinessAgeBucket"] = mapped.where(df[col].isna() | mapped.notna(), "Unclassified")
    df.loc[df[col].isna(), "BusinessAgeBucket"] = pd.NA
    return df


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import load_sba

    for name in ["2000_2009", "2010_2019", "2020_Curr"]:
        df = load_sba.load_raw(load_sba.FILES[name])
        reclassify_business_age(df)
        print(f"=== {name} ===")
        print("before (BusinessAge):")
        print(df["BusinessAge"].value_counts(dropna=False))
        print("\nafter (BusinessAgeBucket):")
        print(df["BusinessAgeBucket"].value_counts(dropna=False))
        print()


if __name__ == "__main__":
    main()
