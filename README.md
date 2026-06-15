# WellCo Uplift Modeling — Causal Outreach Targeting

**Goal**: Identify which members will benefit *most* from outreach — not who is most likely to churn.

## The Key Distinction

| Churn prediction | Uplift modeling |
|---|---|
| Ranks by P(churn) | Ranks by P(churn\|no outreach) − P(churn\|outreach) |
| Targets highest-risk members | Targets members where outreach makes the most difference |
| Treats outreach as a feature | Treats outreach as a causal treatment variable |

Outreach was **not randomly assigned** — the unknown policy targeted predicted high-risk members, creating confounding. We correct for this using propensity-score weighting.

## Approach

### Models
- **TabPFN** (v2) and **TabICL** (v2) — state-of-the-art in-context learners for tabular data; no gradient-descent retraining
- Both used in **T-Learner** and **X-Learner** setups (4 variants total), then **ensembled**

### Causal Pipeline

```
Raw data
  │
  ▼
Feature engineering (14 features)
  │  tenure, web engagement, app usage, ICD codes
  ▼
Propensity model  →  P(outreach=1 | features)
  │  TabPFN, 5-fold CV, trim ps ∈ (0.05, 0.95)
  ▼
T-Learner: μ₀(x) − μ₁(x)         X-Learner: cross-imputed pseudo-outcomes
  │  TabPFN + TabICL                  │  TabPFN + TabICL
  └──────────────┬─────────────────────┘
                 ▼
           Ensemble ITE = mean of 4 variants
                 │
                 ▼
       Rank by ITE descending
       Select members with ITE > 0
```

### ITE Definition
`ITE = P(churn | no outreach, X=x) − P(churn | outreach, X=x)`

Positive ITE = outreach reduces this member's churn probability. We target only members with ITE > 0.

## Results

- **2,038 members** recommended out of 10,000 test members (20.4%)
- Top member estimated ITE: **14.5 pp** churn reduction from outreach
- Persuadable segment is distinct from high-risk-but-unresponsive members

## Repository Structure

```
├── README.md
├── requirements.txt
├── feature_engineering.py        # Shared feature pipeline (train + test)
├── 01_eda_causal.ipynb           # Treatment/control balance, heterogeneity
├── 02_features.ipynb             # Feature engineering, covariate shift check
├── 03_propensity.ipynb           # Propensity model, overlap diagnostics, trimming
├── 04_uplift_model.ipynb         # T-Learner + X-Learner with TabPFN & TabICL
├── 05_evaluation.ipynb           # Qini curve, AUUC, n-selection, predictions
└── predictions.csv               # Top n=2,038 test members: member_id, ite_score, rank
```

Data files (`train/`, `test/`) are **not** included in the repository.

## Setup

### Prerequisites
- Python 3.9
- TabPFN API token from [ux.priorlabs.ai](https://ux.priorlabs.ai) (free, one-time license)

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Layout

Place data files so the relative paths resolve correctly:

```
<repo-root>/          ← this README
├── train/
│   ├── churn_labels.csv
│   ├── web_visits.csv
│   ├── app_usage.csv
│   └── claims.csv
├── test/
│   ├── test_members.csv
│   ├── test_web_visits.csv
│   ├── test_app_usage.csv
│   └── test_claims.csv
└── Shay_Assignment_Uplift/   ← notebooks live here
```

### TabPFN Authentication

```bash
export TABPFN_TOKEN="your_api_key_here"
```

Or inside a notebook cell before calling `.fit()`:

```python
import os
os.environ["TABPFN_TOKEN"] = "your_api_key_here"
```

### Running the Pipeline

Execute notebooks in order:

```bash
cd Shay_Assignment_Uplift/

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 01_eda_causal.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 02_features.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 03_propensity.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 04_uplift_model.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 05_evaluation.ipynb
```

`predictions.csv` is written by notebook 05.

## Causal Assumptions

1. **Conditional ignorability**: given the 14 observed features, treatment assignment is as good as random — i.e., no unobserved confounders correlated with both outreach and churn beyond what features capture.
2. **Overlap**: every member has a non-zero probability of receiving (and not receiving) outreach. We enforce this by trimming members with propensity outside (0.05, 0.95).
3. **SUTVA**: one member's outreach does not affect another's churn outcome.

Assumption 1 is untestable. The propensity AUC of ~0.6–0.65 suggests modest, not extreme, confounding — the policy was imperfect, leaving useful overlap.

## Selecting n

We select all members with ensemble ITE > 0. The ITE-vs-rank curve crosses zero at rank ≈ 2,038. Members below this threshold have estimated zero or negative benefit; targeting them wastes outreach resources and may harm retention (over-contact).

Since outreach cost is unknown, we present n as a function of the benefit threshold. A cost-benefit analysis could raise or lower this threshold.
