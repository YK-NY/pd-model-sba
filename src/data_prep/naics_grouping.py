"""
NAICS 2-digit sector assignment.

Maps NaicsCode -> the 20 official Census NAICS sectors, with the standard
range-consolidation (31-33 Manufacturing, 44-45 Retail Trade,
48-49 Transportation and Warehousing).

assign_naics_sector() assigns the standard 20-sector grouping only.
merge_sparse_sectors() is a separate, explicit step that folds the
confirmed-sparse sectors (Management of Companies and Enterprises, Public
Administration) into "Other" -- decided based on pooled CHGOFF event counts
across all 3 files (36 and 40 events respectively, both unstable per-era).
Utilities (124 pooled events) and Mining (241 pooled events) were
considered and kept standalone -- comfortably past a reasonable
event-count threshold once pooled.
"""
import pandas as pd

# 2-digit NAICS prefix -> (sector code label, sector name).
# Codes sharing one sector code label are the standard Census consolidations.
_SECTOR_MAP = {
    "11": ("11", "Agriculture, Forestry, Fishing and Hunting"),
    "21": ("21", "Mining, Quarrying, and Oil and Gas Extraction"),
    "22": ("22", "Utilities"),
    "23": ("23", "Construction"),
    "31": ("31-33", "Manufacturing"),
    "32": ("31-33", "Manufacturing"),
    "33": ("31-33", "Manufacturing"),
    "42": ("42", "Wholesale Trade"),
    "44": ("44-45", "Retail Trade"),
    "45": ("44-45", "Retail Trade"),
    "48": ("48-49", "Transportation and Warehousing"),
    "49": ("48-49", "Transportation and Warehousing"),
    "51": ("51", "Information"),
    "52": ("52", "Finance and Insurance"),
    "53": ("53", "Real Estate and Rental and Leasing"),
    "54": ("54", "Professional, Scientific, and Technical Services"),
    "55": ("55", "Management of Companies and Enterprises"),
    "56": ("56", "Administrative and Support and Waste Management and Remediation Services"),
    "61": ("61", "Educational Services"),
    "62": ("62", "Health Care and Social Assistance"),
    "71": ("71", "Arts, Entertainment, and Recreation"),
    "72": ("72", "Accommodation and Food Services"),
    "81": ("81", "Other Services (except Public Administration)"),
    "92": ("92", "Public Administration"),
}


def naics_2digit(code):
    """First 2 characters of a NaicsCode string, or NA if not usable."""
    if pd.isna(code):
        return pd.NA
    code = str(code).strip()
    if len(code) < 2:
        return pd.NA
    return code[:2]


def assign_naics_sector(df, code_col="NaicsCode"):
    """
    Adds two columns to df:
      - NaicsSectorCode: the sector-level code label (e.g. "31-33")
      - NaicsSectorName: the sector name (e.g. "Manufacturing")
    Codes that don't match a known 2-digit sector prefix (including missing
    NaicsCode) get "Unclassified" / "Unclassified or Missing" rather than
    raising -- these should be sanity-checked (see main() below), not
    silently trusted.
    """
    two_digit = df[code_col].map(naics_2digit)
    mapped = two_digit.map(_SECTOR_MAP)
    df["NaicsSectorCode"] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else "Unclassified")
    df["NaicsSectorName"] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else "Unclassified or Missing")
    return df


# Sectors confirmed sparse via pooled CHGOFF event-count analysis
# -- merged into "Other", not dropped.
_SPARSE_SECTORS = {"Management of Companies and Enterprises", "Public Administration"}


def merge_sparse_sectors(df, sector_name_col="NaicsSectorName", sector_code_col="NaicsSectorCode"):
    """
    Folds the confirmed-sparse sectors into "Other". Run AFTER
    assign_naics_sector(). Utilities and Mining are deliberately NOT merged
    -- both have enough pooled CHGOFF events to be estimated on their own.
    """
    is_sparse = df[sector_name_col].isin(_SPARSE_SECTORS)
    df.loc[is_sparse, sector_name_col] = "Other"
    df.loc[is_sparse, sector_code_col] = "Other"
    return df


def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import load_sba

    ORDER = ["2000_2009", "2010_2019", "2020_Curr"]

    for name in ORDER:
        df = load_sba.load_raw(load_sba.FILES[name])
        assign_naics_sector(df)
        merge_sparse_sectors(df)
        print(f"=== {name} ===")
        counts = df["NaicsSectorName"].value_counts(dropna=False)
        print(counts)
        n_unclassified = (df["NaicsSectorCode"] == "Unclassified").sum()
        print(f"Unclassified/missing: {n_unclassified} ({n_unclassified/len(df)*100:.2f}%)")
        print()


if __name__ == "__main__":
    main()
