"""
V5 Uplift Model — X-Learner with Do-PFN ITE as augmented feature

Do-PFN produced a continuous causal ITE score (v3) but with two problems:
  1. Ranking is inverted (AUUC = +0.24, worse than random)
  2. Magnitude is wrong (mean ITE = -0.16 vs true ~+0.02)

Fix: Instead of using Do-PFN for end-to-end ITE estimation, use it as a
FEATURE inside the X-Learner Stage 3 classifiers. The second-stage TabPFN/
TabICL models learn the correct direction and effective threshold from data.

Two variants:
  v5_binary : binary Do-PFN signal (thresholded at median, inverted so 1=persuadable)
  v5_cont   : continuous Do-PFN ITE as feature (let classifier learn the threshold)
  v5_best   : whichever has lower AUUC on CV

The "right threshold" for Do-PFN is the median of training ITE (splits 50/50),
with direction flipped (ITE < median → persuadable, because v3 ranking is inverted).
The classifiers then learn how much to weight this signal.
"""

import os, sys, warnings, pickle
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
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from feature_engineering import FEATURE_COLS

sns.set_style('whitegrid')
np.random.seed(42)

# ── Load train data ───────────────────────────────────────────────────────────
print("Loading train/test data...")
train = pd.read_parquet('train_features.parquet')
trim  = pd.read_parquet('trim_mask.parquet')
train = train.merge(trim, on='member_id')
df    = train[train['in_support']].copy().reset_index(drop=True)

X  = df[FEATURE_COLS].values.astype(float)
T  = df['outreach'].values
Y  = df['churn'].values
ps = df['propensity'].values

test   = pd.read_parquet('test_features.parquet')
X_test = test[FEATURE_COLS].values.astype(float)

print(f"In-support: {len(df):,}  |  Treated: {T.sum():,}  Control: {(1-T).sum():,}")

# ── Load Do-PFN predictions (already computed in v3) ─────────────────────────
print("\nLoading Do-PFN v3 predictions...")
with open('ite_cv_v3.pkl', 'rb') as f:
    cv_v3 = pickle.load(f)

id_to_ite_v3 = dict(zip(cv_v3['member_ids'], cv_v3['ite_cv']))
ite_v3_train = np.array([id_to_ite_v3[mid] for mid in df['member_id'].values])

v3_test_df = pd.read_csv('predictions_v3_dopfn.csv').set_index('member_id')['ite_score']
ite_v3_test = test['member_id'].map(v3_test_df).values

print(f"Train Do-PFN ITE: mean={ite_v3_train.mean():.4f}, "
      f"range=[{ite_v3_train.min():.3f}, {ite_v3_train.max():.3f}]")
print(f"Test  Do-PFN ITE: mean={ite_v3_test.mean():.4f}, "
      f"range=[{ite_v3_test.min():.3f}, {ite_v3_test.max():.3f}]")

# ── The "right threshold" ─────────────────────────────────────────────────────
# Do-PFN ranking is INVERTED: low ITE in v3 = persuadable (confirmed by AUUC=+0.24)
# Threshold at median → 50/50 balanced binary signal, direction correct after flip
THRESHOLD = np.median(ite_v3_train)
print(f"\nDo-PFN threshold (median of train ITE): {THRESHOLD:.4f}")
print(f"  Members below threshold (=persuadable per inverted v3): "
      f"{(ite_v3_train < THRESHOLD).mean():.1%}")

# Binary signal: 1 = persuadable (ITE_v3 < threshold, i.e., low/negative → actually beneficial)
dopfn_bin_train = (ite_v3_train < THRESHOLD).astype(float).reshape(-1, 1)
dopfn_bin_test  = (ite_v3_test  < THRESHOLD).astype(float).reshape(-1, 1)

# Continuous signal: use -ite_v3 so sign is corrected (positive = persuadable)
dopfn_cont_train = (-ite_v3_train).reshape(-1, 1)  # flip sign
dopfn_cont_test  = (-ite_v3_test).reshape(-1, 1)

# Augmented feature matrices
X_aug_bin  = np.hstack([X, dopfn_bin_train])   # 15 features
X_aug_cont = np.hstack([X, dopfn_cont_train])  # 15 features

X_test_aug_bin  = np.hstack([X_test, dopfn_bin_test])
X_test_aug_cont = np.hstack([X_test, dopfn_cont_test])

print(f"Feature matrices: original={X.shape[1]}, augmented={X_aug_bin.shape[1]}")

# ── Load propensity for test set (from run_xlearner.py full-train ps model) ──
# Refit propensity on full train data for test scoring
from sklearn.metrics import roc_auc_score
ps_full_model = TabPFNClassifier(n_estimators=8, random_state=42)
ps_full_model.fit(X, T)
ps_test = ps_full_model.predict_proba(X_test)[:, 1]
print(f"\nTest propensity: mean={ps_test.mean():.3f}, "
      f"range=[{ps_test.min():.3f}, {ps_test.max():.3f}]")

# ── Shared utilities ─────────────────────────────────────────────────────────
def ipw_resample(X_grp, Y_grp, ps_grp, weight_fn, rng, n=None, eps=1e-6):
    w = weight_fn(ps_grp)
    w = np.clip(w, 0, np.percentile(w, 99))
    w = w / w.sum()
    n = n or len(X_grp)
    idx = rng.choice(len(X_grp), size=n, replace=True, p=w)
    return X_grp[idx], Y_grp[idx]


def qini_curve(y, t, ite_score, n_bins=100):
    dff = pd.DataFrame({'y': y, 't': t, 's': ite_score}).sort_values('s', ascending=False).reset_index(drop=True)
    n = len(dff)
    total_t, total_c = t.sum(), (1 - t).sum()
    fracs, qinis, randoms = [], [], []
    for k in np.linspace(1, n, n_bins, dtype=int):
        top = dff.iloc[:k]
        nt, nc = top['t'].sum(), (1 - top['t']).sum()
        if nt == 0 or nc == 0:
            continue
        qini_k = top[top['t']==1]['y'].sum() - top[top['t']==0]['y'].sum() * (nt / (nc + 1e-9))
        random_k = (k/n) * (dff[dff['t']==1]['y'].sum() - dff[dff['t']==0]['y'].sum() * (total_t/(total_c+1e-9)))
        fracs.append(k/n); qinis.append(qini_k); randoms.append(random_k)
    return np.array(fracs), np.array(qinis), np.array(randoms)


def auuc(fracs, qinis, randoms):
    return (np.trapz(qinis, fracs) - np.trapz(randoms, fracs)) / (abs(np.trapz(randoms, fracs)) + 1e-9)


def make_tabpfn(): return TabPFNClassifier(n_estimators=8, random_state=42)
def make_tabicl(): return TabICLClassifier(random_state=42)


# ── X-Learner core (same Stage 1/2 as v4, Stage 3 uses augmented X) ─────────
def x_learner_stage1(make_model, X, T, Y, ps, seed=42, eps=1e-6):
    """IPW-corrected Stage 1 on ORIGINAL features (no Do-PFN signal)."""
    rng = np.random.default_rng(seed)
    X1_, Y1_, ps1_ = X[T==1], Y[T==1], ps[T==1]
    X0_, Y0_, ps0_ = X[T==0], Y[T==0], ps[T==0]
    X0_r, Y0_r = ipw_resample(X0_, Y0_, ps0_, lambda p: p / (1-p+eps), rng)
    X1_r, Y1_r = ipw_resample(X1_, Y1_, ps1_, lambda p: (1-p) / (p+eps), rng)
    m1 = make_model(); m1.fit(X1_r, Y1_r)
    m0 = make_model(); m0.fit(X0_r, Y0_r)
    return m0, m1


def x_learner_stage2_and_3(make_model, m0, m1, X, T, Y, X_aug):
    """
    Stage 2: cross-predict pseudo-outcomes D1, D0.
    Stage 3: fit second-stage classifiers on augmented X (with Do-PFN signal).
    """
    X1_, Y1_ = X[T==1], Y[T==1]
    X0_, Y0_ = X[T==0], Y[T==0]
    X1_aug   = X_aug[T==1]
    X0_aug   = X_aug[T==0]

    mu0_on_X1 = m0.predict_proba(X1_)[:, 1]
    mu1_on_X0 = m1.predict_proba(X0_)[:, 1]

    D1 = mu0_on_X1 - Y1_   # positive = outreach helped
    D0 = Y0_ - mu1_on_X0   # positive = would benefit

    D1_bin = (D1 > np.median(D1)).astype(int)
    D0_bin = (D0 > np.median(D0)).astype(int)

    # Stage 3: fit on AUGMENTED features (includes Do-PFN ITE signal)
    tau1_model = make_model(); tau1_model.fit(X1_aug, D1_bin)
    tau0_model = make_model(); tau0_model.fit(X0_aug, D0_bin)

    return tau0_model, tau1_model


def x_learner_predict(tau0_model, tau1_model, X_test_aug, ps_test):
    tau1 = tau1_model.predict_proba(X_test_aug)[:, 1]
    tau0 = tau0_model.predict_proba(X_test_aug)[:, 1]
    return ps_test * tau0 + (1 - ps_test) * tau1


def ipw_x_learner_v5(make_model, X, T, Y, ps, X_aug, X_test_aug, ps_test, seed=42):
    """Full V5 pipeline: Stage 1 on X, Stage 3 on X_aug."""
    m0, m1 = x_learner_stage1(make_model, X, T, Y, ps, seed)
    tau0_model, tau1_model = x_learner_stage2_and_3(make_model, m0, m1, X, T, Y, X_aug)
    return x_learner_predict(tau0_model, tau1_model, X_test_aug, ps_test)


# ── 5-fold CV: binary Do-PFN feature ─────────────────────────────────────────
print("\n=== 5-fold CV: V5a (Do-PFN BINARY threshold feature in Stage 3) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_key = (df['churn'].astype(str) + '_' + df['outreach'].astype(str)).values

ite_oof_v5a_pf = np.zeros(len(df))
ite_oof_v5a_ic = np.zeros(len(df))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, strat_key)):
    Xtr, Ttr, Ytr, pstr = X[tr_idx], T[tr_idx], Y[tr_idx], ps[tr_idx]
    X_aug_tr = X_aug_bin[tr_idx]
    X_aug_val = X_aug_bin[val_idx]
    psval = ps[val_idx]

    m0_pf, m1_pf = x_learner_stage1(make_tabpfn, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_pf, tau1_pf = x_learner_stage2_and_3(make_tabpfn, m0_pf, m1_pf, Xtr, Ttr, Ytr, X_aug_tr)
    ite_oof_v5a_pf[val_idx] = x_learner_predict(tau0_pf, tau1_pf, X_aug_val, psval)

    m0_ic, m1_ic = x_learner_stage1(make_tabicl, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_ic, tau1_ic = x_learner_stage2_and_3(make_tabicl, m0_ic, m1_ic, Xtr, Ttr, Ytr, X_aug_tr)
    ite_oof_v5a_ic[val_idx] = x_learner_predict(tau0_ic, tau1_ic, X_aug_val, psval)

    print(f"  Fold {fold+1}: TabPFN={ite_oof_v5a_pf[val_idx].mean():.4f}, "
          f"TabICL={ite_oof_v5a_ic[val_idx].mean():.4f}")

ite_oof_v5a = (ite_oof_v5a_pf + ite_oof_v5a_ic) / 2

# ── 5-fold CV: continuous Do-PFN feature ─────────────────────────────────────
print("\n=== 5-fold CV: V5b (Do-PFN CONTINUOUS ITE feature in Stage 3) ===")

ite_oof_v5b_pf = np.zeros(len(df))
ite_oof_v5b_ic = np.zeros(len(df))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, strat_key)):
    Xtr, Ttr, Ytr, pstr = X[tr_idx], T[tr_idx], Y[tr_idx], ps[tr_idx]
    X_aug_tr = X_aug_cont[tr_idx]
    X_aug_val = X_aug_cont[val_idx]
    psval = ps[val_idx]

    m0_pf, m1_pf = x_learner_stage1(make_tabpfn, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_pf, tau1_pf = x_learner_stage2_and_3(make_tabpfn, m0_pf, m1_pf, Xtr, Ttr, Ytr, X_aug_tr)
    ite_oof_v5b_pf[val_idx] = x_learner_predict(tau0_pf, tau1_pf, X_aug_val, psval)

    m0_ic, m1_ic = x_learner_stage1(make_tabicl, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_ic, tau1_ic = x_learner_stage2_and_3(make_tabicl, m0_ic, m1_ic, Xtr, Ttr, Ytr, X_aug_tr)
    ite_oof_v5b_ic[val_idx] = x_learner_predict(tau0_ic, tau1_ic, X_aug_val, psval)

    print(f"  Fold {fold+1}: TabPFN={ite_oof_v5b_pf[val_idx].mean():.4f}, "
          f"TabICL={ite_oof_v5b_ic[val_idx].mean():.4f}")

ite_oof_v5b = (ite_oof_v5b_pf + ite_oof_v5b_ic) / 2

# ── AUUC comparison ───────────────────────────────────────────────────────────
print("\n=== AUUC Comparison ===")
# V4 baseline (from run_xlearner.py)
ite_oof_v4_pf = np.zeros(len(df))
ite_oof_v4_ic = np.zeros(len(df))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, strat_key)):
    Xtr, Ttr, Ytr, pstr = X[tr_idx], T[tr_idx], Y[tr_idx], ps[tr_idx]
    Xval, psval = X[val_idx], ps[val_idx]
    m0_pf, m1_pf = x_learner_stage1(make_tabpfn, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_pf, tau1_pf = x_learner_stage2_and_3(make_tabpfn, m0_pf, m1_pf, Xtr, Ttr, Ytr, X[tr_idx])
    ite_oof_v4_pf[val_idx] = x_learner_predict(tau0_pf, tau1_pf, Xval, psval)
    m0_ic, m1_ic = x_learner_stage1(make_tabicl, Xtr, Ttr, Ytr, pstr, seed=fold)
    tau0_ic, tau1_ic = x_learner_stage2_and_3(make_tabicl, m0_ic, m1_ic, Xtr, Ttr, Ytr, X[tr_idx])
    ite_oof_v4_ic[val_idx] = x_learner_predict(tau0_ic, tau1_ic, Xval, psval)
    print(f"  V4 Fold {fold+1} done")

ite_oof_v4 = (ite_oof_v4_pf + ite_oof_v4_ic) / 2

fracs_v4, qinis_v4, rands_v4 = qini_curve(Y, T, ite_oof_v4)
fracs_v5a, qinis_v5a, rands_v5a = qini_curve(Y, T, ite_oof_v5a)
fracs_v5b, qinis_v5b, rands_v5b = qini_curve(Y, T, ite_oof_v5b)

auuc_v4  = auuc(fracs_v4,  qinis_v4,  rands_v4)
auuc_v5a = auuc(fracs_v5a, qinis_v5a, rands_v5a)
auuc_v5b = auuc(fracs_v5b, qinis_v5b, rands_v5b)

print(f"\n  V4 X-Learner (no Do-PFN):          AUUC = {auuc_v4:.4f}")
print(f"  V5a X-Learner + Do-PFN BINARY:     AUUC = {auuc_v5a:.4f}")
print(f"  V5b X-Learner + Do-PFN CONTINUOUS: AUUC = {auuc_v5b:.4f}")
print(f"  (more negative = better)")

best_v5 = 'v5a' if auuc_v5a < auuc_v5b else 'v5b'
best_auuc_v5 = min(auuc_v5a, auuc_v5b)
print(f"\n  Best V5 variant: {best_v5} (AUUC = {best_auuc_v5:.4f})")
best_ite_oof_v5 = ite_oof_v5a if best_v5 == 'v5a' else ite_oof_v5b
best_X_aug      = X_aug_bin    if best_v5 == 'v5a' else X_aug_cont
best_X_test_aug = X_test_aug_bin if best_v5 == 'v5a' else X_test_aug_cont

# ── Fit on full train, predict test ──────────────────────────────────────────
print(f"\n=== Full train fit → test predictions (best: {best_v5}) ===")

print("TabPFN V5...")
ite_test_v5_pf = ipw_x_learner_v5(make_tabpfn, X, T, Y, ps, best_X_aug, best_X_test_aug, ps_test, seed=42)

print("TabICL V5...")
ite_test_v5_ic = ipw_x_learner_v5(make_tabicl, X, T, Y, ps, best_X_aug, best_X_test_aug, ps_test, seed=42)

ite_test_v5_ensemble = (ite_test_v5_pf + ite_test_v5_ic) / 2

print(f"\nTest ITE V5 ({best_v5}):")
print(f"  TabPFN:   mean={ite_test_v5_pf.mean():.4f}, "
      f"range=[{ite_test_v5_pf.min():.3f}, {ite_test_v5_pf.max():.3f}]")
print(f"  TabICL:   mean={ite_test_v5_ic.mean():.4f}, "
      f"range=[{ite_test_v5_ic.min():.3f}, {ite_test_v5_ic.max():.3f}]")
print(f"  Ensemble: mean={ite_test_v5_ensemble.mean():.4f}, "
      f"range=[{ite_test_v5_ensemble.min():.3f}, {ite_test_v5_ensemble.max():.3f}]")

# Also save both v5a and v5b full-train predictions
print("\nFitting both v5a and v5b full-train for complete prediction files...")
ite_test_v5a_pf = ipw_x_learner_v5(make_tabpfn, X, T, Y, ps, X_aug_bin, X_test_aug_bin, ps_test, seed=42)
ite_test_v5a_ic = ipw_x_learner_v5(make_tabicl, X, T, Y, ps, X_aug_bin, X_test_aug_bin, ps_test, seed=42)
ite_test_v5a    = (ite_test_v5a_pf + ite_test_v5a_ic) / 2

ite_test_v5b_pf = ipw_x_learner_v5(make_tabpfn, X, T, Y, ps, X_aug_cont, X_test_aug_cont, ps_test, seed=42)
ite_test_v5b_ic = ipw_x_learner_v5(make_tabicl, X, T, Y, ps, X_aug_cont, X_test_aug_cont, ps_test, seed=42)
ite_test_v5b    = (ite_test_v5b_pf + ite_test_v5b_ic) / 2

# ── Save prediction CSVs ──────────────────────────────────────────────────────
print("\n=== Saving prediction files ===")
for suffix, ite_scores in [
    ('v5a_tabpfn',  ite_test_v5a_pf),
    ('v5a_tabicl',  ite_test_v5a_ic),
    ('v5a_ensemble',ite_test_v5a),
    ('v5b_tabpfn',  ite_test_v5b_pf),
    ('v5b_tabicl',  ite_test_v5b_ic),
    ('v5b_ensemble',ite_test_v5b),
]:
    pred_df = pd.DataFrame({
        'member_id': test['member_id'].values,
        'ite_score': ite_scores,
    })
    pred_df['rank'] = pred_df['ite_score'].rank(ascending=False).astype(int)
    pred_df = pred_df.sort_values('rank').reset_index(drop=True)
    fname = f'predictions_{suffix}.csv'
    pred_df.to_csv(fname, index=False)
    print(f"  Saved {fname}: ITE range=[{ite_scores.min():.4f}, {ite_scores.max():.4f}]")

# ── Qini comparison plot ──────────────────────────────────────────────────────
print("\nGenerating Qini comparison plot...")
# V3 Do-PFN baseline
fracs_v3, qinis_v3, rands_v3 = qini_curve(Y, T, ite_v3_train)
auuc_v3 = auuc(fracs_v3, qinis_v3, rands_v3)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fracs_v4,  qinis_v4,  lw=2, color='teal',       label=f'V4 X-Learner (AUUC={auuc_v4:.3f})')
ax.plot(fracs_v5a, qinis_v5a, lw=2, color='darkorange',  label=f'V5a +DoPFN binary (AUUC={auuc_v5a:.3f})')
ax.plot(fracs_v5b, qinis_v5b, lw=2, color='purple',      label=f'V5b +DoPFN continuous (AUUC={auuc_v5b:.3f})')
ax.plot(fracs_v3,  qinis_v3,  lw=1.5, color='crimson', ls=':', label=f'V3 Do-PFN direct (AUUC={auuc_v3:.3f})')
ax.plot(fracs_v4,  rands_v4,  lw=1.5, color='gray', ls='--', label='Random')
ax.axhline(0, color='black', lw=0.7)
ax.set_xlabel('Fraction of population targeted')
ax.set_ylabel('Qini value')
ax.set_title('Qini Curve — V4 vs V5 (X-Learner with/without Do-PFN feature)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('qini_v4_v5_comparison.png', dpi=100, bbox_inches='tight')
print("Saved qini_v4_v5_comparison.png")

print("\n=== FINAL SUMMARY ===")
print(f"  V3 Do-PFN (direct):                 AUUC = {auuc_v3:.4f}")
print(f"  V4 X-Learner (no Do-PFN):           AUUC = {auuc_v4:.4f}")
print(f"  V5a X-Learner + Do-PFN binary:      AUUC = {auuc_v5a:.4f}")
print(f"  V5b X-Learner + Do-PFN continuous:  AUUC = {auuc_v5b:.4f}")
print(f"\n  Do-PFN contribution vs V4: "
      f"{'IMPROVED' if best_auuc_v5 < auuc_v4 else 'NO IMPROVEMENT'} "
      f"(delta={best_auuc_v5 - auuc_v4:+.4f})")
print(f"\n  Threshold used: {THRESHOLD:.4f} (median of train Do-PFN ITE)")
print(f"  Interpretation: ITE_v3 < {THRESHOLD:.4f} → persuadable (inverted because v3 ranking is inverted)")
print("\nDone.")
