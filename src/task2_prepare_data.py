import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_dataset():
    features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    classes  = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, features.shape[1] - 1)]
    return features, classes


def prepare_data():
    features, classes = load_dataset()

    # Spoji features sa labelama
    df = features.merge(classes, on="txId")

    # Ukloni unknown — zadržavamo samo licit (2) i illicit (1)
    df = df[df["class"] != "unknown"].copy()
    df["class"] = df["class"].astype(int)

    # Binarna labela: 1 = illicit, 0 = licit
    df["label"] = (df["class"] == 1).astype(int)

    feature_cols = [c for c in df.columns if c.startswith("f")]

    # Temporalni split: 1-34 trening, 35-49 test
    train = df[df["time_step"] <= 34]
    test  = df[df["time_step"] >= 35]

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test  = test[feature_cols].values
    y_test  = test["label"].values

    return X_train, y_train, X_test, y_test, train, test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, train, test = prepare_data()

    print("=" * 55)
    print("TASK 2 — Priprema podataka")
    print("=" * 55)

    print(f"\nBroj feature-a: {X_train.shape[1]}")

    print("\n--- TRENING SET (time_step 1–34) ---")
    illicit_tr = y_train.sum()
    licit_tr   = (y_train == 0).sum()
    print(f"  Ukupno:   {len(y_train):>7,}")
    print(f"  illicit:  {illicit_tr:>7,}  ({illicit_tr/len(y_train)*100:.1f}%)")
    print(f"  licit:    {licit_tr:>7,}  ({licit_tr/len(y_train)*100:.1f}%)")

    print("\n--- TEST SET (time_step 35–49) ---")
    illicit_te = y_test.sum()
    licit_te   = (y_test == 0).sum()
    print(f"  Ukupno:   {len(y_test):>7,}")
    print(f"  illicit:  {illicit_te:>7,}  ({illicit_te/len(y_test)*100:.1f}%)")
    print(f"  licit:    {licit_te:>7,}  ({licit_te/len(y_test)*100:.1f}%)")

    print("\n--- Provera time step granica ---")
    print(f"  Trening time_step raspon: {train['time_step'].min()}–{train['time_step'].max()}")
    print(f"  Test    time_step raspon: {test['time_step'].min()}–{test['time_step'].max()}")
    print("=" * 55)
