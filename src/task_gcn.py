import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from pathlib import Path

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── 1. Učitaj podatke ────────────────────────────────────────────────────────
print("Učitavam podatke...")
features_raw = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
classes      = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
edgelist     = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")

features_raw.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, features_raw.shape[1] - 1)]

df = features_raw.merge(classes, on="txId")
df = df[df["class"] != "unknown"].copy()
df["class"] = df["class"].astype(int)
df["label"] = (df["class"] == 1).astype(int)

# Mapiranje txId → indeks čvora (reindeksiramo da bude 0..N-1)
node_ids  = df["txId"].values
id_to_idx = {txid: idx for idx, txid in enumerate(node_ids)}
N = len(df)

print(f"Čvorovi (labeled): {N:,}")

# ── 2. Edge index (samo grane između labeled čvorova) ────────────────────────
src_ids = edgelist["txId1"].map(id_to_idx)
dst_ids = edgelist["txId2"].map(id_to_idx)
mask    = src_ids.notna() & dst_ids.notna()
src     = torch.tensor(src_ids[mask].astype(int).values, dtype=torch.long)
dst     = torch.tensor(dst_ids[mask].astype(int).values, dtype=torch.long)
# Neusmereni graf — dodaj obe strane
edge_index = torch.stack([torch.cat([src, dst]),
                          torch.cat([dst, src])], dim=0)
print(f"Grane (u labeled podgrafu): {edge_index.shape[1]//2:,}")

# ── 3. Node features i labele ────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c.startswith("f")]
X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
y = torch.tensor(df["label"].values,      dtype=torch.float32)

# Temporalni split maske
time_steps   = df["time_step"].values
train_mask   = torch.tensor(time_steps <= 34, dtype=torch.bool)
test_mask    = torch.tensor(time_steps >= 35, dtype=torch.bool)

data = Data(x=X, edge_index=edge_index, y=y,
            train_mask=train_mask, test_mask=test_mask)

print(f"Trening čvorova: {train_mask.sum().item():,}  |  Test čvorova: {test_mask.sum().item():,}")

# ── 4. GCN arhitektura ───────────────────────────────────────────────────────
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x).squeeze()

# ── 5. Class weight za imbalans ──────────────────────────────────────────────
n_illicit = train_mask.float() @ y
n_licit   = train_mask.sum() - n_illicit
pos_weight = torch.tensor([n_licit / n_illicit])
print(f"\nClass weight (pos_weight): {pos_weight.item():.2f}")

model     = GCN(in_channels=X.shape[1], hidden=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ── 6. Trening ───────────────────────────────────────────────────────────────
print("\nTreniram GCN (100 epoha)...")

# Za BCEWithLogitsLoss radimo logits (pre sigmoid)
class GCN_logits(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index).squeeze()

model     = GCN_logits(in_channels=X.shape[1], hidden=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

prev_loss = None
no_improve = 0

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss   = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            logits_eval = model(data.x, data.edge_index)
            prob_train  = torch.sigmoid(logits_eval[data.train_mask])
            pred_train  = (prob_train >= 0.5).float()
            tr_f1 = f1_score(data.y[data.train_mask].numpy(),
                             pred_train.numpy(), zero_division=0)
        print(f"  Epoha {epoch:>3}  loss={loss.item():.4f}  train_F1={tr_f1:.4f}")

        # Provjera konvergencije — ako loss ne pada 3 puta zaredom, stani
        if prev_loss is not None and loss.item() >= prev_loss - 0.0005:
            no_improve += 1
            if no_improve >= 3:
                print(f"\nLoss nije konvergirao nakon {epoch} epoha — zaustavljam.")
                break
        else:
            no_improve = 0
        prev_loss = loss.item()

# ── 7. Evaluacija ────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits_eval  = model(data.x, data.edge_index)
    prob_test    = torch.sigmoid(logits_eval[data.test_mask]).numpy()
    y_test_np    = data.y[data.test_mask].numpy()

y_pred = (prob_test >= 0.5).astype(int)

# Provjera da model nije kolabsirao na jednu klasu
unique_preds = np.unique(y_pred)
if len(unique_preds) == 1:
    print(f"\nUPOZORENJE: Model predviđa samo klasu {unique_preds[0]} — GCN nije konvergirao.")
    print("Ovo je znak da treba više podešavanja (više epoha, drugačiji lr, arhitektura).")
    print("Preporučujem da se GCN ostavi za kasniji rad — rezultati nisu smisleni.")
else:
    tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred).ravel()
    acc  = accuracy_score(y_test_np, y_pred)
    prec = precision_score(y_test_np, y_pred, zero_division=0)
    rec  = recall_score(y_test_np, y_pred, zero_division=0)
    f1   = f1_score(y_test_np, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print("GCN — rezultati na test setu (time_step 35–49)")
    print("=" * 55)
    print(f"  Accuracy:            {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (illicit): {prec:.4f}")
    print(f"  Recall    (illicit): {rec:.4f}")
    print(f"  F1        (illicit): {f1:.4f}")
    print(f"  FP: {fp}   FN: {fn}")
    print("=" * 55)

    # Sačuvaj u CSV
    gcn_row = pd.DataFrame([{
        "Model": "GCN (2-layer)", "Accuracy": acc,
        "Precision": prec, "Recall": rec, "F1": f1, "FP": fp, "FN": fn
    }])
    csv_path = RESULTS_DIR / "model_comparison.csv"
    existing  = pd.read_csv(csv_path)
    # Ukloni stari GCN red ako postoji
    existing = existing[existing["Model"] != "GCN (2-layer)"]
    updated  = pd.concat([existing, gcn_row], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    print(f"\nTabela ažurirana: {csv_path}")
