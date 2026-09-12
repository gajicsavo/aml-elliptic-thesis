import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from task2_prepare_data import prepare_data

X_train, y_train, X_test, y_test, _, _ = prepare_data()

print("Treniram Random Forest (n_estimators=100)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 55)
print("TASK 3 — Random Forest (baseline)")
print("=" * 55)
print(f"\n  Accuracy:           {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision (illicit): {prec:.4f}")
print(f"  Recall    (illicit): {rec:.4f}")
print(f"  F1        (illicit): {f1:.4f}")

print("\n--- Confusion Matrix ---")
print(f"             Predicted")
print(f"             licit   illicit")
print(f"  Actual licit    {tn:>5}   {fp:>5}    (FP: {fp})")
print(f"  Actual illicit  {fn:>5}   {tp:>5}    (FN: {fn})")

print("\n--- Detaljan izveštaj ---")
print(classification_report(y_test, y_pred, target_names=["licit", "illicit"]))
print("=" * 55)
