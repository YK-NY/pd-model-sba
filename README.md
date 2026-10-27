# SBA 7(a) Credit Risk Model

For a prospective US small-business loan, this model estimates at origination the probability of default within five years (PD), the share of the loan lost if it defaults (LGD) and the exposure at that point (EAD), and combines them into an expected loss.
Real SME credit models are proprietary and rarely published, so this is a transparent, reproducible reference implementation built on public data. It is a reference model for study and benchmarking, not a tool for real lending decisions.

## Data

- **Source:** SBA 7(a) loan-level FOIA data from [data.sba.gov](https://data.sba.gov/dataset/7-a-504-foia) (three files: FY2000–2009, FY2010–2019, FY2020–present, plus the data dictionary), and the monthly Prime rate from [FRED, series MPRIME](https://fred.stlouisfed.org/series/MPRIME), used only to fill in missing interest rates in the FY2000–2009 backtest.
- **Size:** about 1.6 million loans. After the filters below, 543,123 loans form the modeling population.
- **Time window:** loans approved FY2000 to the present. The model is trained on FY2010–2018 approvals and tested on FY2019–2021. FY2000–2009, which includes the 2008 recession, is held out as a backtest.
- **How default is defined:** a loan is a default if it was charged off within 60 months of approval. Only loans with at least 60 months of history are used, so every loan is judged over the same period. Cancelled loans and loans not yet disbursed are excluded.

## Method and results

- **PD:** logistic regression on 9 custom-binned (weight-of-evidence) features, compared against XGBoost. Test AUC **0.803**, KS **0.448**.
- **Recession backtest:** the frozen PD model applied to 607,914 FY2000–2009 loans keeps its ranking ability (AUC 0.779, KS 0.428) but understates the level of defaults (9.29% predicted against 14.73% actual). The population shift is large (PSI 0.570).
- **LGD:** XGBoost regression, test R² 0.137, ahead of a segment-average lookup (0.072) and a linear regression (0.079).
- **EAD:** term loans use the approved amount; revolving loans use an XGBoost model.

The full writeup, with methodology, findings and limitations, is in [`whitepaper/whitepaper.md`](whitepaper/whitepaper.md).

## How to run

**1. Set up** (Python 3.12):

```bash
git clone <repository-url>
cd pd-model-sba
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Add the data.** The raw files are not in this repository. Download the three 7(a) files and the data dictionary from the [SBA page](https://data.sba.gov/dataset/7-a-504-foia) and `MPRIME.csv` from [FRED](https://fred.stlouisfed.org/series/MPRIME) (columns `observation_date` and `MPRIME`). Put them in `data/raw/` with these names, or change the names in `FILES` in `src/data_prep/load_sba.py` (SBA refreshes its files, so current downloads carry a newer "as of" date and the numbers will differ slightly):

```
data/raw/FOIA_7a_FY2000_FY2009_asof_260630.csv
data/raw/FOIA_7a_FY2010_FY2019_asof_260630.csv
data/raw/FOIA_7a_FY2020_Present_asof_260630.csv
data/raw/MPRIME.csv
```

**3. Train the models.** Run from the repository root, in this order. Each step saves its model to `outputs/model_artifacts/`.

```bash
python src/model/train_logistic.py
python src/model/train_lgd_xgboost.py
python src/model/train_ead.py
```

**4. Score loans.** This scores a sample of real loans that were approved but not yet disbursed and prints a portfolio summary:

```bash
python src/model/score_loan.py
```

**5. Validate.** These load the trained PD model and write charts to `outputs/figures/`:

```bash
python src/model/validate.py
python src/model/extra_validation_charts.py
python src/model/backtest_2000_2009.py
python src/model/psi_2000_2009.py
```

**Optional comparison runs** (not needed for the steps above): `src/features/woe_binning.py` (information-value table), `src/model/run_ablation.py`, `src/model/cross_validate.py`, `src/model/train_xgboost.py`, `src/model/xgboost_pdp_check.py` and `src/model/train_lgd.py`.

**macOS, XGBoost note:** if you see `XGBoost Library (libxgboost.dylib) could not be loaded`, XGBoost needs the OpenMP runtime (`libomp`). `scikit-learn`'s own installed wheel already includes a copy, so you can copy it into your Python installation's `lib/` directory instead of installing Homebrew:

```bash
cp .venv/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib \
   "$(python -c 'import sys; print(sys.base_prefix)')/lib/libomp.dylib"
```

**Layout:**

```
data/raw/                 raw files (not in this repo)
src/data_prep/            loading, cleaning, sector and business-age mapping
src/features/             target construction, binning
src/model/                PD, LGD, EAD training, scoring, validation, backtesting
notebooks/                EDA and LGD notebooks (use a kernel pointing at .venv)
outputs/figures/          generated charts
outputs/model_artifacts/  trained models (not in this repo)
whitepaper/               the writeup
```

[`docs/FILE_INDEX.md`](docs/FILE_INDEX.md) has a one-line description of every script.

## Known limitations and roadmap

**Limitations:** the PD covers 60 months, not a lifetime or 12 months. LGD is gross of recoveries, since the data has none. The data has no borrower financials, credit scores or payment history. The model is calibrated to FY2010–2018 and understates defaults in a downturn.

**Phase 2 (planned):** confidence intervals for individual scores, a formal calibration test, an LGD and EAD backtest, a downturn mode, a lender-risk feature and portfolio concentration analysis.

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You may share and adapt it for non-commercial purposes, provided you credit **YK** and link back to this repository. Commercial use requires written permission. The SBA loan data it uses is public and is not covered by this license. The full text is in [`LICENSE.md`](LICENSE.md).

Questions or feedback: open an issue on this repository.
