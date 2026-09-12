import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier, AdaBoostClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from task2_prepare_data import prepare_data

X_train, y_train, X_test, y_test, _, _ = prepare_data()

models = [
    ("Random Forest",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("Extra Trees",    ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("Bagging (RF)",   BaggingClassifier(
                           estimator=RandomForestClassifier(n_estimators=10, random_state=42),
                           n_estimators=10, random_state=42, n_jobs=-1)),
    ("AdaBoost",       AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME")),
]

results = []
trained = {}

for name, model in models:
    print(f"Treniram {name}...")
    model.fit(X_train, y_train)
    trained[name] = model

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "FP":        fp,
        "FN":        fn,
    })

df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("POREĐENJE MODELA — test set (time_step 35–49)")
print("=" * 75)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
print("=" * 75)

# Sačuvaj za kasniju upotrebu (Task 5 dodaje ensemble)
df.to_csv("../results/model_comparison.csv", index=False)
print("\nRezultati sačuvani u results/model_comparison.csv")
