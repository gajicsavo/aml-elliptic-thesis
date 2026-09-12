import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier)
from pathlib import Path
from task2_prepare_data import prepare_data

RESULTS = Path(__file__).parent.parent / "results"

X_train, y_train, X_test, y_test, _, test_df = prepare_data()

print("Treniram modele za vizualizacije...")
rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et  = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
bag = BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=10, random_state=42),
        n_estimators=10, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)
et.fit(X_train, y_train)
bag.fit(X_train, y_train)

prob_rf  = rf.predict_proba(X_test)[:, 1]
prob_et  = et.predict_proba(X_test)[:, 1]
prob_bag = bag.predict_proba(X_test)[:, 1]
prob_ens = (prob_rf + prob_et + prob_bag) / 3

# ── Grafik 1: ROC krive ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for name, prob in [("Random Forest", prob_rf),
                   ("Extra Trees",   prob_et),
                   ("Bagging (RF)",  prob_bag),
                   ("Ensemble",      prob_ens)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f"{name}  (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC krive — detekcija illicit transakcija\n(Elliptic dataset, test set: time_step 35–49)", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = RESULTS / "roc_curves.png"
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"ROC krive sačuvane: {roc_path}")

# ── Grafik 2: F1 po time_step za ensemble ────────────────────────────────────
time_steps = sorted(test_df["time_step"].unique())
f1_per_step = []

for ts in time_steps:
    mask = (test_df["time_step"] == ts).values
    if mask.sum() == 0:
        f1_per_step.append(np.nan)
        continue
    y_true_ts = y_test[mask]
    y_pred_ts = (prob_ens[mask] >= 0.5).astype(int)
    if y_true_ts.sum() == 0:
        f1_per_step.append(np.nan)
    else:
        f1_per_step.append(f1_score(y_true_ts, y_pred_ts, zero_division=0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_steps, f1_per_step, marker="o", linewidth=2, color="steelblue", markersize=5)
ax.axvline(x=43, color="red", linestyle="--", linewidth=1.5, label="Time_step 43\n(dark market shutdown)")
ax.set_xlabel("Time step", fontsize=12)
ax.set_ylabel("F1-score (illicit klasa)", fontsize=12)
ax.set_title("F1-score ensembla po vremenskom koraku\n(test set: time_step 35–49)", fontsize=13)
ax.set_ylim([0, 1.05])
ax.set_xticks(time_steps)
ax.tick_params(axis="x", labelsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
f1_path = RESULTS / "f1_per_timestep.png"
plt.savefig(f1_path, dpi=150)
plt.close()
print(f"F1 po time_step sačuvan: {f1_path}")

print("\n--- F1 po time_step (ensemble) ---")
for ts, f1 in zip(time_steps, f1_per_step):
    bar = "█" * int((f1 or 0) * 30)
    print(f"  ts {ts:>2}: {f1:.3f}  {bar}")
