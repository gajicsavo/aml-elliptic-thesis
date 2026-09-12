import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
edgelist = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")
classes  = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")

# Prve dve kolone su txId i time_step, ostale su features (166 ukupno)
features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, features.shape[1] - 1)]

print("=" * 55)
print("ELLIPTIC DATASET — osnovne informacije")
print("=" * 55)

print(f"\n[features] redovi: {len(features):,}  kolone: {features.shape[1]}")
print(f"[edgelist] redovi: {len(edgelist):,}  kolone: {edgelist.shape[1]}")
print(f"[classes]  redovi: {len(classes):,}  kolone: {classes.shape[1]}")

print("\n--- Raspodela klasa ---")
class_counts = classes["class"].value_counts()
total = len(classes)
for label, count in class_counts.items():
    pct = count / total * 100
    name = {"1": "illicit", "2": "licit"}.get(str(label), "unknown")
    print(f"  {label:>10}  ({name:<8})  {count:>7,}  ({pct:.1f}%)")

print(f"\n  UKUPNO                    {total:>7,}")

labeled = classes[classes["class"] != "unknown"]
print(f"\n  Označenih (licit+illicit): {len(labeled):,}")
print(f"  Neoznačenih (unknown):     {total - len(labeled):,}")

print("\n--- Time step raspon ---")
print(f"  min={features['time_step'].min()}  max={features['time_step'].max()}")
print(f"  Broj jedinstvenih time step-ova: {features['time_step'].nunique()}")
print("=" * 55)
