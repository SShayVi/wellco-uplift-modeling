"""
Shared feature engineering for WellCo uplift modeling.
Used by notebooks 02-05.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"
OUT_DIR = Path(__file__).parent

OUTREACH_DATE = pd.Timestamp("2025-07-15")
OBS_START = pd.Timestamp("2025-07-01")

# WellCo-domain web visits only
WELLCO_DOMAINS = ["health.wellco", "care.portal", "living.better", "guide.wellness"]
WELLCO_PATTERN = "|".join(WELLCO_DOMAINS)

# Clinically expected ICD codes (WellCo focus)
EXPECTED_ICD = {"E11.9", "I10", "Z71.3"}


def build_web_features(web: pd.DataFrame, member_ids: pd.Series) -> pd.DataFrame:
    web = web.copy()
    web["is_wellco"] = web["url"].str.contains(WELLCO_PATTERN, case=False)
    web["domain"] = web["url"].str.extract(r"https?://([^/]+)")[0]

    wellco_web = web[web["is_wellco"]]

    g_all = web.groupby("member_id")
    g_wco = wellco_web.groupby("member_id")

    feats = pd.DataFrame({"member_id": member_ids})

    # Visit counts
    feats = feats.merge(
        g_all.size().rename("total_visits").reset_index(), on="member_id", how="left"
    )
    feats = feats.merge(
        g_wco.size().rename("wellco_visits").reset_index(), on="member_id", how="left"
    )
    feats[["total_visits", "wellco_visits"]] = feats[
        ["total_visits", "wellco_visits"]
    ].fillna(0)

    feats["health_content_ratio"] = feats["wellco_visits"] / (
        feats["total_visits"] + 1
    )

    # Recency: days since last WellCo visit before outreach date
    last_wco = (
        wellco_web.groupby("member_id")["timestamp"].max().rename("last_wellco_visit")
    )
    feats = feats.merge(last_wco.reset_index(), on="member_id", how="left")
    feats["days_since_wellco"] = (
        OUTREACH_DATE - feats["last_wellco_visit"]
    ).dt.days.clip(lower=0)
    feats["days_since_wellco"] = feats["days_since_wellco"].fillna(99)  # never visited

    # Unique WellCo domains visited
    wco_domain_counts = (
        wellco_web.groupby("member_id")["domain"]
        .nunique()
        .rename("unique_wellco_domains")
    )
    feats = feats.merge(
        wco_domain_counts.reset_index(), on="member_id", how="left"
    )
    feats["unique_wellco_domains"] = feats["unique_wellco_domains"].fillna(0)

    # Drop raw timestamp column
    feats = feats.drop(columns=["last_wellco_visit"])
    return feats


def build_app_features(app: pd.DataFrame, member_ids: pd.Series) -> pd.DataFrame:
    app = app.copy()
    g = app.groupby("member_id")

    feats = pd.DataFrame({"member_id": member_ids})
    feats = feats.merge(
        g.size().rename("app_sessions").reset_index(), on="member_id", how="left"
    )
    feats["app_sessions"] = feats["app_sessions"].fillna(0)

    # Recency
    last_app = g["timestamp"].max().rename("last_app_session")
    feats = feats.merge(last_app.reset_index(), on="member_id", how="left")
    feats["days_since_app"] = (
        OUTREACH_DATE - feats["last_app_session"]
    ).dt.days.clip(lower=0)
    feats["days_since_app"] = feats["days_since_app"].fillna(99)
    feats = feats.drop(columns=["last_app_session"])

    # Sessions in final 7 days (high recency engagement)
    recent = app[app["timestamp"] >= (OUTREACH_DATE - pd.Timedelta(days=7))]
    recent_counts = (
        recent.groupby("member_id").size().rename("app_sessions_last7d")
    )
    feats = feats.merge(recent_counts.reset_index(), on="member_id", how="left")
    feats["app_sessions_last7d"] = feats["app_sessions_last7d"].fillna(0)

    return feats


def build_claims_features(claims: pd.DataFrame, member_ids: pd.Series) -> pd.DataFrame:
    # Deduplicate exact duplicate records
    claims = claims.drop_duplicates()

    feats = pd.DataFrame({"member_id": member_ids})

    # Expected ICD flags
    for icd in sorted(EXPECTED_ICD):
        col = "icd_" + icd.replace(".", "_")
        has_icd = (
            claims[claims["icd_code"] == icd]
            .groupby("member_id")
            .size()
            .gt(0)
            .rename(col)
        )
        feats = feats.merge(has_icd.reset_index(), on="member_id", how="left")
        feats[col] = feats[col].fillna(False).astype(int)

    # Count of unexpected ICD codes (noise signal)
    unexpected = claims[~claims["icd_code"].isin(EXPECTED_ICD)]
    unexpected_counts = (
        unexpected.groupby("member_id")["icd_code"]
        .nunique()
        .rename("unexpected_icd_count")
    )
    feats = feats.merge(unexpected_counts.reset_index(), on="member_id", how="left")
    feats["unexpected_icd_count"] = feats["unexpected_icd_count"].fillna(0)

    # Total distinct expected ICD codes (comorbidity count)
    expected_claims = claims[claims["icd_code"].isin(EXPECTED_ICD)]
    comorbidity = (
        expected_claims.groupby("member_id")["icd_code"]
        .nunique()
        .rename("expected_icd_count")
    )
    feats = feats.merge(comorbidity.reset_index(), on="member_id", how="left")
    feats["expected_icd_count"] = feats["expected_icd_count"].fillna(0)

    return feats


def build_features(split: str = "train") -> pd.DataFrame:
    """Build full feature matrix for 'train' or 'test'."""
    if split == "train":
        labels = pd.read_csv(TRAIN_DIR / "churn_labels.csv", parse_dates=["signup_date"])
        web = pd.read_csv(TRAIN_DIR / "web_visits.csv", parse_dates=["timestamp"])
        app = pd.read_csv(TRAIN_DIR / "app_usage.csv", parse_dates=["timestamp"])
        claims = pd.read_csv(TRAIN_DIR / "claims.csv", parse_dates=["diagnosis_date"])
        member_ids = labels["member_id"]
    else:
        labels = pd.read_csv(TEST_DIR / "test_members.csv")
        web = pd.read_csv(TEST_DIR / "test_web_visits.csv", parse_dates=["timestamp"])
        app = pd.read_csv(TEST_DIR / "test_app_usage.csv", parse_dates=["timestamp"])
        claims = pd.read_csv(TEST_DIR / "test_claims.csv", parse_dates=["diagnosis_date"])
        member_ids = labels["member_id"]

    # Tenure
    labels = labels.copy()
    if "signup_date" in labels.columns:
        labels["tenure_days"] = (
            OUTREACH_DATE - pd.to_datetime(labels["signup_date"])
        ).dt.days
    else:
        labels["tenure_days"] = np.nan

    base = labels[
        ["member_id", "tenure_days"]
        + (["churn", "outreach"] if split == "train" else [])
    ].copy()

    web_feats = build_web_features(web, member_ids)
    app_feats = build_app_features(app, member_ids)
    claim_feats = build_claims_features(claims, member_ids)

    df = (
        base.merge(web_feats, on="member_id", how="left")
        .merge(app_feats, on="member_id", how="left")
        .merge(claim_feats, on="member_id", how="left")
    )
    return df


FEATURE_COLS = [
    "tenure_days",
    "total_visits",
    "wellco_visits",
    "health_content_ratio",
    "days_since_wellco",
    "unique_wellco_domains",
    "app_sessions",
    "days_since_app",
    "app_sessions_last7d",
    "icd_E11_9",
    "icd_I10",
    "icd_Z71_3",
    "unexpected_icd_count",
    "expected_icd_count",
]
