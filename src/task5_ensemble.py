import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from task2_prepare_data import prepare_data

X_train, y_train, X_test, y_test, _, _ = prepare_data()

print("Treniram modele za ensemble...")
rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
et  = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
bag = BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=10, random_state=42),
        n_estimators=10, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)
et.fit(X_train, y_train)
bag.fit(X_train, y_train)

# Average probability ensemble
prob_rf  = rf.predict_proba(X_test)[:, 1]
prob_et  = et.predict_proba(X_test)[:, 1]
prob_bag = bag.predict_proba(X_test)[:, 1]

prob_ensemble = (prob_rf + prob_et + prob_bag) / 3
y_pred_ensemble = (prob_ensemble >= 0.5).astype(int)

def evaluate(y_true, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model":     name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1":        f1_score(y_true, y_pred),
        "FP":        fp,
        "FN":        fn,
    }

results = [
    evaluate(y_test, rf.predict(X_test),  "Random Forest"),
    evaluate(y_test, et.predict(X_test),  "Extra Trees"),
    evaluate(y_test, bag.predict(X_test), "Bagging (RF)"),
    evaluate(y_test, y_pred_ensemble,     "Ensemble (RF+ET+Bag)"),
]

df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("POREĐENJE MODELA + ENSEMBLE — test set (time_step 35–49)")
print("=" * 75)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
print("=" * 75)

best = df.loc[df["F1"].idxmax(), "Model"]
print(f"\nNajbolji F1 (illicit): {best}  —  F1={df['F1'].max():.4f}")

df.to_csv("../results/model_comparison.csv", index=False)
print("Ažurirana tabela sačuvana u results/model_comparison.csv")

# Sačuvaj verovatnoće ensemblea za Task 6 (ROC krive)
np.save("../results/prob_rf.npy",       prob_rf)
np.save("../results/prob_et.npy",       prob_et)
np.save("../results/prob_bag.npy",      prob_bag)
np.save("../results/prob_ensemble.npy", prob_ensemble)
np.save("../results/y_test.npy",        y_test)
print("Verovatnoće sačuvane u results/ (za ROC krive)")
