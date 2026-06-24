"""
V4 Uplift Model — X-Learner with IPW Correction

Architecture:
  Stage 1: Fit m0 (control) and m1 (treated) with IPW resampling [same as v2]
  Stage 2: Cross-predict pseudo-outcomes:
    D1_i = mu0(x_i) - Y_i  for treated members  (positive = outreach helped)
    D0_i = Y_i - mu1(x_i)  for control members  (positive = would benefit)
  Stage 3: Fit second-stage classifiers on binarized (median-split) pseudo-outcomes
    tau1: predict P(D1 > median | X), fit on treated members
    tau0: predict P(D0 > median | X), fit on control members
  Stage 4: Propensity-weighted combination
    ITE(x) = e(x) * tau0(x) + (1 - e(x)) * tau1(x)

Why X-Learner > T-Learner here:
  - Cross-prediction directly estimates ITE for each observed member,
    rather than computing mu0-mu1 for a held-out test point.
  - Propensity weighting in Stage 4 further adjusts for confounding.
  - More robust when treatment groups are imbalanced (40:60) or
    when the propensity model doesn't fully balance covariates
    (app_sessions SMD overcorrected in v2: -0.504 -> +0.587).
"""

import os
import sys
import warnings

# Run from the notebook directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from feature_engineering import FEATURE_COLS

sns.set_style('whitegrid')
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet('train_features.parquet')
trim  = pd.read_parquet('trim_mask.parquet')
train = train.merge(trim, on='member_id')
df    = train[train['in_support']].copy().reset_index(drop=True)

X  = df[FEATURE_COLS].values.astype(float)
T  = df['outreach'].values
Y  = df['churn'].values
ps = df['propensity'].values  # OOF propensity from notebook 03

print(f"In-support: {len(df):,}  |  Treated: {T.sum():,}  Control: {(1-T).sum():,}")
print(f"Propensity range: [{ps.min():.3f}, {ps.max():.3f}]")

# ── Utility functions ─────────────────────────────────────────────────────────
def ipw_resample(X_grp, Y_grp, ps_grp, weight_fn, rng, n=None, eps=1e-6):
    w = weight_fn(ps_grp)
    w = np.clip(w, 0, np.percentile(w, 99))
    w = w / w.sum()
    n = n or len(X_grp)
    idx = rng.choice(len(X_grp), size=n, replace=True, p=w)
    return X_grp[idx], Y_grp[idx]


def qini_curve(y, t, ite_score, n_bins=100):
    df_ = pd.DataFrame({'y': y, 't': t, 's': ite_score}).sort_values('s', ascending=False).reset_index(drop=True)
    n = len(df_)
    total_t, total_c = t.sum(), (1 - t).sum()
    fracs, qinis, randoms = [], [], []
    for k in np.linspace(1, n, n_bins, dtype=int):
        top = df_.iloc[:k]
        nt, nc = top['t'].sum(), (1 - top['t']).sum()
        if nt == 0 or nc == 0:
            continue
        qini_k = top[top['t']==1]['y'].sum() - top[top['t']==0]['y'].sum() * (nt / (nc + 1e-9))
        random_k = (k / n) * (df_[df_['t']==1]['y'].sum() - df_[df_['t']==0]['y'].sum() * (total_t / (total_c + 1e-9)))
        fracs.append(k / n); qinis.append(qini_k); randoms.append(random_k)
    return np.array(fracs), np.array(qinis), np.array(randoms)


def auuc(fracs, qinis, randoms):
    return (np.trapz(qinis, fracs) - np.trapz(randoms, fracs)) / (abs(np.trapz(randoms, fracs)) + 1e-9)


def make_tabpfn(): return TabPFNClassifier(n_estimators=8, random_state=42)
def make_tabicl(): return TabICLClassifier(random_state=42)

# ── X-Learner core ────────────────────────────────────────────────────────────
def x_learner_stage1(make_model, X, T, Y, ps, seed=42, eps=1e-6):
    """IPW-corrected Stage 1: fit m0 (control) and m1 (treated)."""
    rng = np.random.default_rng(seed)
    X1_, Y1_, ps1_ = X[T==1], Y[T==1], ps[T==1]
    X0_, Y0_, ps0_ = X[T==0], Y[T==0], ps[T==0]

    X0_r, Y0_r = ipw_resample(X0_, Y0_, ps0_, lambda p: p / (1 - p + eps), rng)
    X1_r, Y1_r = ipw_resample(X1_, Y1_, ps1_, lambda p: (1 - p) / (p + eps), rng)

    m1 = make_model(); m1.fit(X1_r, Y1_r)
    m0 = make_model(); m0.fit(X0_r, Y0_r)
    return m0, m1


def x_learner_stage2(make_model, m0, m1, X, T, Y):
    """Stage 2: cross-predict pseudo-outcomes, binarize, fit second-stage models."""
    X1_, Y1_ = X[T==1], Y[T==1]
    X0_, Y0_ = X[T==0], Y[T==0]

    mu0_on_X1 = m0.predict_proba(X1_)[:, 1]  # P(churn | no outreach) for treated members
    mu1_on_X0 = m1.predict_proba(X0_)[:, 1]  # P(churn | outreach)    for control members

    # Pseudo-outcomes: positive = outreach reduces churn for this member
    D1 = mu0_on_X1 - Y1_   # treated: counterfactual-no-outreach minus actual treated outcome
    D0 = Y0_ - mu1_on_X0   # control: actual control outcome minus counterfactual-with-outreach

    # Binarize at MEDIAN: rank-preserving, balanced 50/50 class labels
    D1_bin = (D1 > np.median(D1)).astype(int)
    D0_bin = (D0 > np.median(D0)).astype(int)

    tau1_model = make_model(); tau1_model.fit(X1_, D1_bin)  # P(D1 > median | X)
    tau0_model = make_model(); tau0_model.fit(X0_, D0_bin)  # P(D0 > median | X)

    return tau0_model, tau1_model, D0, D1


def x_learner_predict(tau0_model, tau1_model, X_pred, ps_pred):
    """Stage 4: propensity-weighted combination."""
    tau1 = tau1_model.predict_proba(X_pred)[:, 1]
    tau0 = tau0_model.predict_proba(X_pred)[:, 1]
    return ps_pred * tau0 + (1 - ps_pred) * tau1


def ipw_x_learner(make_model, X, T, Y, ps, X_test, ps_test, seed=42):
    """Full pipeline: train on (X, T, Y, ps), predict on (X_test, ps_test)."""
    m0, m1 = x_learner_stage1(make_model, X, T, Y, ps, seed)
    tau0_model, tau1_model, D0, D1 = x_learner_stage2(make_model, m0, m1, X, T, Y)
    return x_learner_predict(tau0_model, tau1_model, X_test, ps_test)


# ── Fit propensity model for test scoring ─────────────────────────────────────
print("\nFitting propensity model on full train (for test set scoring)...")
ps_full_model = TabPFNClassifier(n_estimators=8, random_state=42)
ps_full_model.fit(X, T)
auc_ps = roc_auc_score(T, ps_full_model.predict_proba(X)[:, 1])
print(f"  Propensity model AUC on train: {auc_ps:.4f}")

test  = pd.read_parquet('test_features.parquet')
X_test = test[FEATURE_COLS].values.astype(float)
ps_test = ps_full_model.predict_proba(X_test)[:, 1]
print(f"  Test propensity: mean={ps_test.mean():.3f}, range=[{ps_test.min():.3f}, {ps_test.max():.3f}]")


# ── 5-fold CV for AUUC ────────────────────────────────────────────────────────
print("\n=== 5-fold CV — X-Learner ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_key = (df['churn'].astype(str) + '_' + df['outreach'].astype(str)).values

ite_oof_tabpfn = np.zeros(len(df))
ite_oof_tabicl = np.zeros(len(df))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, strat_key)):
    Xtr, Ttr, Ytr, pstr = X[tr_idx], T[tr_idx], Y[tr_idx], ps[tr_idx]
    Xval, psval = X[val_idx], ps[val_idx]

    # TabPFN X-Learner
    m0_pf, m1_pf = x_learner_stage1(make_tabpfn, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_pf, tau1_pf, _, _ = x_learner_stage2(make_tabpfn, m0_pf, m1_pf, Xtr, Ttr, Ytr)
    ite_oof_tabpfn[val_idx] = x_learner_predict(tau0_pf, tau1_pf, Xval, psval)

    # TabICL X-Learner
    m0_ic, m1_ic = x_learner_stage1(make_tabicl, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_ic, tau1_ic, _, _ = x_learner_stage2(make_tabicl, m0_ic, m1_ic, Xtr, Ttr, Ytr)
    ite_oof_tabicl[val_idx] = x_learner_predict(tau0_ic, tau1_ic, Xval, psval)

    print(f"  Fold {fold+1}: TabPFN ITE={ite_oof_tabpfn[val_idx].mean():.4f}, "
          f"TabICL ITE={ite_oof_tabicl[val_idx].mean():.4f}")

ite_oof_ensemble = (ite_oof_tabpfn + ite_oof_tabicl) / 2

print(f"\nCV ITE stats:")
print(f"  TabPFN:   mean={ite_oof_tabpfn.mean():.4f}, std={ite_oof_tabpfn.std():.4f}, "
      f"%pos={( ite_oof_tabpfn>0.5).mean():.2%}")
print(f"  TabICL:   mean={ite_oof_tabicl.mean():.4f}, std={ite_oof_tabicl.std():.4f}, "
      f"%pos={(ite_oof_tabicl>0.5).mean():.2%}")
print(f"  Ensemble: mean={ite_oof_ensemble.mean():.4f}, std={ite_oof_ensemble.std():.4f}")

# ── AUUC ──────────────────────────────────────────────────────────────────────
print("\n=== AUUC ===")
fracs_pf, qinis_pf, rands_pf = qini_curve(Y, T, ite_oof_tabpfn)
fracs_ic, qinis_ic, rands_ic = qini_curve(Y, T, ite_oof_tabicl)
fracs_en, qinis_en, rands_en = qini_curve(Y, T, ite_oof_ensemble)

auuc_pf = auuc(fracs_pf, qinis_pf, rands_pf)
auuc_ic = auuc(fracs_ic, qinis_ic, rands_ic)
auuc_en = auuc(fracs_en, qinis_en, rands_en)

print(f"  TabPFN X-Learner  : {auuc_pf:.4f}")
print(f"  TabICL X-Learner  : {auuc_ic:.4f}")
print(f"  Ensemble X-Learner: {auuc_en:.4f}")
print(f"  [V2 IPW T-Learner was: -0.4016]  (more negative = better)")

# ── Mega-ensemble: v2 + v4 ────────────────────────────────────────────────────
# Load v2 CV ITE for comparison (recompute from saved parquet)
test_preds_v2 = pd.read_parquet('test_predictions_v2.parquet')

# v2 OOF ITE must be recomputed; use rank-normalization to combine v2 train ITE + v4 train ITE
# v2 train ITE was stored in df during notebook 04 execution — not saved to parquet.
# Instead compute rank-normalized combination on test set only (for final predictions).
print("\nBuilding rank-normalized mega-ensemble for test set...")

# ── Fit on full train, predict test ──────────────────────────────────────────
print("\n=== Full train fit → test predictions ===")

print("TabPFN X-Learner...")
ite_test_tabpfn = ipw_x_learner(make_tabpfn, X, T, Y, ps, X_test, ps_test, seed=42)

print("TabICL X-Learner...")
ite_test_tabicl = ipw_x_learner(make_tabicl, X, T, Y, ps, X_test, ps_test, seed=42)

ite_test_ensemble = (ite_test_tabpfn + ite_test_tabicl) / 2

print(f"\nTest ITE v4:")
print(f"  TabPFN:   mean={ite_test_tabpfn.mean():.4f}, range=[{ite_test_tabpfn.min():.3f}, {ite_test_tabpfn.max():.3f}]")
print(f"  TabICL:   mean={ite_test_tabicl.mean():.4f}, range=[{ite_test_tabicl.min():.3f}, {ite_test_tabicl.max():.3f}]")
print(f"  Ensemble: mean={ite_test_ensemble.mean():.4f}, range=[{ite_test_ensemble.min():.3f}, {ite_test_ensemble.max():.3f}]")

# Mega-ensemble: rank-normalize v2 and v4 ensemble, then average ranks
v2_ite_test = test_preds_v2.set_index('member_id')['ite_v2_ensemble'].reindex(test['member_id']).values
v4_ite_test = ite_test_ensemble

# Rank normalization to [0, 1]: (rank - 1) / (n - 1), higher ITE = higher rank score
n = len(v2_ite_test)
v2_rank_norm = (pd.Series(v2_ite_test).rank(ascending=True) - 1) / (n - 1)
v4_rank_norm = (pd.Series(v4_ite_test).rank(ascending=True) - 1) / (n - 1)
ite_test_mega = (v2_rank_norm.values + v4_rank_norm.values) / 2

print(f"  Mega-ensemble (v2+v4 rank-avg): mean={ite_test_mega.mean():.4f}")

# ── Save prediction CSVs ───────────────────────────────────────────────────────
print("\n=== Saving prediction files ===")

for suffix, ite_scores in [
    ('tabpfn',    ite_test_tabpfn),
    ('tabicl',    ite_test_tabicl),
    ('ensemble',  ite_test_ensemble),
    ('mega',      ite_test_mega),
]:
    pred_df = pd.DataFrame({
        'member_id': test['member_id'].values,
        'ite_score': ite_scores,
    })
    pred_df['rank'] = pred_df['ite_score'].rank(ascending=False).astype(int)
    pred_df = pred_df.sort_values('rank').reset_index(drop=True)
    fname = f'predictions_v4_{suffix}.csv'
    pred_df.to_csv(fname, index=False)
    n_pos = (ite_scores > ite_scores.mean()).sum()  # "above average" since ITE is [0,1]
    print(f"  Saved {fname}: ITE range=[{ite_scores.min():.4f}, {ite_scores.max():.4f}], "
          f"mean={ite_scores.mean():.4f}")

# ── Qini comparison plot ───────────────────────────────────────────────────────
print("\nGenerating Qini comparison plot...")

# Also compute v2 T-Learner OOF Qini for comparison from 5-fold CV
# (We re-run the v2 CV here for apples-to-apples comparison)
print("Re-running 5-fold CV for v2 (comparison baseline)...")
ite_oof_v2_pf = np.zeros(len(df))
ite_oof_v2_ic = np.zeros(len(df))

def ipw_t_learner(make_model, X, T, Y, ps, X_test, seed=42, eps=1e-6):
    rng = np.random.default_rng(seed)
    X1_, Y1_, ps1_ = X[T==1], Y[T==1], ps[T==1]
    X0_, Y0_, ps0_ = X[T==0], Y[T==0], ps[T==0]
    X0_r, Y0_r = ipw_resample(X0_, Y0_, ps0_, lambda p: p / (1 - p + eps), rng)
    X1_r, Y1_r = ipw_resample(X1_, Y1_, ps1_, lambda p: (1 - p) / (p + eps), rng)
    m1 = make_model(); m1.fit(X1_r, Y1_r)
    m0 = make_model(); m0.fit(X0_r, Y0_r)
    mu1 = m1.predict_proba(X_test)[:, 1]
    mu0 = m0.predict_proba(X_test)[:, 1]
    return mu0 - mu1

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, strat_key)):
    Xtr, Ttr, Ytr, pstr = X[tr_idx], T[tr_idx], Y[tr_idx], ps[tr_idx]
    Xval = X[val_idx]
    ite_oof_v2_pf[val_idx] = ipw_t_learner(make_tabpfn, Xtr, Ttr, Ytr, pstr, Xval, seed=fold)
    ite_oof_v2_ic[val_idx] = ipw_t_learner(make_tabicl, Xtr, Ttr, Ytr, pstr, Xval, seed=fold)
    print(f"  V2 Fold {fold+1} done")

ite_oof_v2_ensemble = (ite_oof_v2_pf + ite_oof_v2_ic) / 2
fracs_v2, qinis_v2, rands_v2 = qini_curve(Y, T, ite_oof_v2_ensemble)
auuc_v2 = auuc(fracs_v2, qinis_v2, rands_v2)
print(f"V2 re-computed AUUC: {auuc_v2:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(fracs_v2, qinis_v2, lw=2, color='teal',   label=f'V2 IPW T-Learner (AUUC={auuc_v2:.3f})')
ax.plot(fracs_en, qinis_en, lw=2, color='darkorange', label=f'V4 X-Learner (AUUC={auuc_en:.3f})')
ax.plot(fracs_v2, rands_v2, lw=1.5, ls='--', color='gray', label='Random')
ax.fill_between(fracs_v2, rands_v2, qinis_v2, alpha=0.10, color='teal')
ax.fill_between(fracs_en, rands_en, qinis_en, alpha=0.10, color='darkorange')
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('Fraction of population targeted')
ax.set_ylabel('Qini value')
ax.set_title('Qini Curve — V2 IPW T-Learner vs V4 X-Learner (5-fold CV)')
ax.legend()
plt.tight_layout()
plt.savefig('qini_v2_v4_comparison.png', dpi=100, bbox_inches='tight')
print("Saved qini_v2_v4_comparison.png")

print("\n=== FINAL SUMMARY ===")
print(f"  V2 AUUC (IPW T-Learner):   {auuc_v2:.4f}")
print(f"  V4 AUUC (X-Learner TabPFN): {auuc_pf:.4f}")
print(f"  V4 AUUC (X-Learner TabICL): {auuc_ic:.4f}")
print(f"  V4 AUUC (X-Learner Ens):    {auuc_en:.4f}")
print(f"\n  More negative = better.")
print(f"  V4 {'IMPROVES' if auuc_en < auuc_v2 else 'does NOT improve'} upon V2.")
