"""
churn_nn_visualisations.py
==========================
Standalone script that reproduces all four Neural Network comparative
visualisations from the churn_pytorch_standout.ipynb notebook:

    1. Training Loss & AUC Curves      (nn_training_curves.png)
    2. Grouped Metric Bar Chart         (nn_metrics_comparison.png)
    3. ROC Curve Overlay — 3 models    (nn_roc_comparison.png)
    4. Confusion Matrices — 3 models   (nn_confusion_matrices.png)

Usage
-----
    python churn_nn_visualisations.py

All four plots are saved to ./images/results/ and displayed inline
when run inside a Jupyter notebook.

Dependencies
------------
    pip install torch scikit-learn numpy pandas matplotlib seaborn joblib
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./images/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Design tokens — dark GitHub-style theme ───────────────────────────────────
BG       = "#0D1117"   # canvas background
FG       = "#E6EDF3"   # primary text / labels
GRID_COL = "#30363D"   # gridlines and spines
ANNO     = "#8B949E"   # secondary annotation text

COLORS = {
    "Random Forest":       "#2E75B6",   # steel blue
    "Logistic Regression": "#ED7D31",   # orange
    "Neural Network":      "#70AD47",   # green
}

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA & MODEL PREPARATION
# Run this block once; Sections 2-5 only generate plots.
# ═════════════════════════════════════════════════════════════════════════════

def prepare_data():
    """
    Load the feature-engineered splits via churn_library, apply StandardScaler
    for the NN and LR, and return numpy arrays + PyTorch tensors.
    Returns a dict of all arrays needed by the training and plotting functions.
    """
    sys.path.insert(0, ".")
    import churn_library as cls

    df = cls.import_data("./data/bank_data.csv")
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df, response="Churn"
    )

    X_tr = X_train.values.astype(np.float32)
    X_te = X_test.values.astype(np.float32)
    y_tr = y_train.values.astype(np.float32)
    y_te = y_test.values.astype(np.float32)

    # Scale for NN and LR (RF is scale-invariant)
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)   # fit on train only — no leakage
    X_te_s   = scaler.transform(X_te)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t   = torch.tensor(X_tr_s, dtype=torch.float32).to(device)
    y_tr_t   = torch.tensor(y_tr,   dtype=torch.float32).unsqueeze(1).to(device)
    X_te_t   = torch.tensor(X_te_s, dtype=torch.float32).to(device)
    y_te_t   = torch.tensor(y_te,   dtype=torch.float32).unsqueeze(1).to(device)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    return dict(
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        X_tr_s=X_tr_s, X_te_s=X_te_s,
        X_tr_t=X_tr_t, y_tr_t=y_tr_t,
        X_te_t=X_te_t, y_te_t=y_te_t,
        train_dl=train_dl, device=device,
        feature_names=list(X_train.columns),
    )


# ── Neural Network definition ─────────────────────────────────────────────────

class ChurnNet(nn.Module):
    """
    4-layer feedforward MLP for binary churn classification.

    Architecture : 19 → 128 → 64 → 32 → 1 (logit)
    Regularisation: BatchNorm1d + Dropout at every hidden layer.
    Output        : raw logit — sigmoid applied at inference time only.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # ── Layer 1: 19 → 128 ────────────────────────────────────────
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # ── Layer 2: 128 → 64 ────────────────────────────────────────
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            # ── Layer 3: 64 → 32 ─────────────────────────────────────────
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # ── Output: 32 → 1 (logit) ───────────────────────────────────
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Return probability of churn (class 1) as a numpy array."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).cpu().numpy().squeeze()


def train_neural_network(data: dict, epochs: int = 60):
    """
    Train ChurnNet and record per-epoch loss and AUC for both
    train and validation sets.

    Parameters
    ----------
    data   : dict returned by prepare_data()
    epochs : number of training epochs (default 60)

    Returns
    -------
    model        : trained ChurnNet
    history      : dict with keys train_losses, val_losses,
                   train_aucs, val_aucs, epochs
    """
    device   = data["device"]
    X_tr_t   = data["X_tr_t"]
    y_tr_t   = data["y_tr_t"]
    X_te_t   = data["X_te_t"]
    y_te_t   = data["y_te_t"]
    y_tr     = data["y_tr"]
    y_te     = data["y_te"]
    train_dl = data["train_dl"]

    # Class-imbalance-aware loss
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    pos_weight   = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model        = ChurnNet(input_dim=X_tr_t.shape[1]).to(device)
    optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []

    print(f"Training ChurnNet — {epochs} epochs …")
    for epoch in range(epochs):
        # ── Training pass ────────────────────────────────────────────────
        model.train()
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        # ── Evaluation pass ──────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            tr_logits = model(X_tr_t)
            te_logits = model(X_te_t)
            tr_loss   = criterion(tr_logits, y_tr_t).item()
            te_loss   = criterion(te_logits, y_te_t).item()
            tr_prob   = torch.sigmoid(tr_logits).cpu().numpy().squeeze()
            te_prob   = torch.sigmoid(te_logits).cpu().numpy().squeeze()
            tr_auc    = roc_auc_score(y_tr, tr_prob)
            te_auc    = roc_auc_score(y_te, te_prob)

        train_losses.append(tr_loss);  val_losses.append(te_loss)
        train_aucs.append(tr_auc);     val_aucs.append(te_auc)

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss {tr_loss:.4f} | Val Loss {te_loss:.4f} | "
                f"Train AUC {tr_auc:.4f} | Val AUC {te_auc:.4f}"
            )

    print("Training complete.\n")
    history = dict(
        train_losses=train_losses, val_losses=val_losses,
        train_aucs=train_aucs,     val_aucs=val_aucs,
        epochs=list(range(1, epochs + 1)),
    )
    return model, history


def train_sklearn_models(data: dict):
    """
    Train Random Forest (GridSearchCV, 5-fold CV) and Logistic Regression
    (StandardScaler pipeline) on the same splits used by the NN.

    Returns
    -------
    best_rfc  : fitted RandomForestClassifier (best estimator)
    lrc       : fitted Pipeline (StandardScaler + LogisticRegression)
    """
    X_tr, X_te = data["X_tr"], data["X_te"]
    y_tr = data["y_tr"]

    PARAM_GRID = {
        "n_estimators": [200, 500],
        "max_features": ["sqrt"],
        "max_depth":    [4, 5, 100],
        "criterion":    ["gini", "entropy"],
    }

    print("Training Random Forest (GridSearchCV, cv=5) …")
    cv_rfc   = GridSearchCV(
        RandomForestClassifier(random_state=SEED),
        PARAM_GRID, cv=5, n_jobs=-1,
    )
    cv_rfc.fit(X_tr, y_tr)
    best_rfc = cv_rfc.best_estimator_
    print(f"  Best params: {cv_rfc.best_params_}")

    print("Training Logistic Regression (StandardScaler pipeline) …")
    lrc = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(solver="lbfgs", max_iter=3000)),
    ])
    lrc.fit(X_tr, y_tr)
    print("  Done.\n")

    return best_rfc, lrc


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PLOT 1: TRAINING LOSS & AUC CURVES
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, save: bool = True) -> plt.Figure:
    """
    Plot two side-by-side panels:
      Left  — BCEWithLogitsLoss (train vs validation)
      Right — ROC-AUC           (train vs validation)

    Parameters
    ----------
    history : dict with keys train_losses, val_losses, train_aucs,
              val_aucs, epochs  (returned by train_neural_network)
    save    : if True, save PNG to OUTPUT_DIR

    Returns
    -------
    matplotlib Figure
    """
    epochs      = history["epochs"]
    tr_losses   = history["train_losses"]
    va_losses   = history["val_losses"]
    tr_aucs     = history["train_aucs"]
    va_aucs     = history["val_aucs"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), facecolor=BG
    )
    fig.suptitle(
        "PyTorch Neural Network — Training History (60 Epochs)",
        color=FG, fontsize=14, fontweight="bold", y=1.02,
    )

    # ── Shared axis styling ───────────────────────────────────────────────
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=10)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.grid(True, color=GRID_COL, linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    # ── Left panel: Loss ──────────────────────────────────────────────────
    ax1.plot(epochs, tr_losses, color="#58A6FF", lw=2.2,
             label="Train Loss", zorder=3)
    ax1.plot(epochs, va_losses, color="#F78166", lw=2.2,
             label="Validation Loss", zorder=3)
    ax1.fill_between(epochs, tr_losses, va_losses,
                     alpha=0.08, color="#58A6FF")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("BCEWithLogitsLoss", fontsize=11)
    ax1.set_title("Loss Curve", color=FG, fontsize=13, fontweight="bold")
    ax1.legend(
        facecolor="#161B22", edgecolor=GRID_COL,
        labelcolor=FG, fontsize=10,
    )

    # Annotate final values
    ax1.annotate(
        f"Final: {tr_losses[-1]:.4f}",
        xy=(epochs[-1], tr_losses[-1]),
        xytext=(-60, 8), textcoords="offset points",
        color="#58A6FF", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#58A6FF", lw=1),
    )
    ax1.annotate(
        f"Final: {va_losses[-1]:.4f}",
        xy=(epochs[-1], va_losses[-1]),
        xytext=(-60, -16), textcoords="offset points",
        color="#F78166", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#F78166", lw=1),
    )

    # ── Right panel: AUC ─────────────────────────────────────────────────
    ax2.plot(epochs, tr_aucs, color="#58A6FF", lw=2.2,
             label="Train AUC", zorder=3)
    ax2.plot(epochs, va_aucs, color="#F78166", lw=2.2,
             label="Validation AUC", zorder=3)
    ax2.fill_between(epochs, tr_aucs, va_aucs,
                     alpha=0.08, color="#58A6FF")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("ROC-AUC", fontsize=11)
    ax2.set_title("AUC Curve", color=FG, fontsize=13, fontweight="bold")
    ax2.set_ylim(0.85, 1.0)
    ax2.legend(
        facecolor="#161B22", edgecolor=GRID_COL,
        labelcolor=FG, fontsize=10,
    )

    # Annotate convergence plateau
    peak_epoch = int(np.argmax(va_aucs)) + 1
    peak_auc   = max(va_aucs)
    ax2.axvline(x=peak_epoch, color="#F78166", lw=1, linestyle=":",
                alpha=0.7)
    ax2.text(
        peak_epoch + 1, 0.87,
        f"Peak: {peak_auc:.4f}\n@ epoch {peak_epoch}",
        color="#F78166", fontsize=9,
    )

    # LR reduction markers (StepLR fires at epochs 20, 40)
    for step_ep in (20, 40):
        ax2.axvline(x=step_ep, color=ANNO, lw=1, linestyle="--", alpha=0.5)
        ax1.axvline(x=step_ep, color=ANNO, lw=1, linestyle="--", alpha=0.5)
        ax2.text(step_ep + 0.5, 0.855, f"LR×0.5\n@{step_ep}",
                 color=ANNO, fontsize=7.5)

    plt.tight_layout()

    if save:
        out = os.path.join(OUTPUT_DIR, "nn_training_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {out}")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PLOT 2: GROUPED METRIC BAR CHART (5 metrics × 3 models)
# ═════════════════════════════════════════════════════════════════════════════

def plot_metric_bars(results: list[dict], save: bool = True) -> plt.Figure:
    """
    Grouped bar chart comparing AUC, Accuracy, F1, Precision, and Recall
    across all three models.

    Parameters
    ----------
    results : list of dicts, each with keys:
              Model, AUC, Accuracy, F1, Precision, Recall
    save    : if True, save PNG to OUTPUT_DIR

    Returns
    -------
    matplotlib Figure
    """
    METRIC_KEYS = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
    n_metrics   = len(METRIC_KEYS)
    n_models    = len(results)
    x           = np.arange(n_metrics)
    bar_w       = 0.24
    offsets     = np.linspace(
        -(n_models - 1) * bar_w / 2,
         (n_models - 1) * bar_w / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
    ax.set_facecolor(BG)

    for i, result in enumerate(results):
        model_name = result["Model"]
        values     = [result[k] for k in METRIC_KEYS]
        bars = ax.bar(
            x + offsets[i], values, bar_w,
            label=model_name,
            color=COLORS[model_name],
            alpha=0.88,
            edgecolor="black", linewidth=0.5,
            zorder=3,
        )

        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.009,
                f"{val:.3f}",
                ha="center", va="bottom",
                color=FG, fontsize=8.5, fontweight="bold",
            )

    # ── Highlight best value per metric with a gold star ─────────────────
    for j, metric in enumerate(METRIC_KEYS):
        best_val = max(r[metric] for r in results)
        ax.annotate(
            "★",
            xy=(x[j] + offsets[[r[metric] for r in results].index(best_val)],
                best_val + 0.035),
            ha="center", va="bottom",
            color="#FFD700", fontsize=11,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_KEYS, color=FG, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Score", color=FG, fontsize=12)
    ax.set_title(
        "Model Performance Metrics — 5 Metrics × 3 Models\n"
        "★ = Best score per metric",
        color=FG, fontsize=14, fontweight="bold", pad=14,
    )
    ax.tick_params(colors=FG)
    ax.grid(True, axis="y", color=GRID_COL, alpha=0.5, linestyle="--", zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.legend(
        facecolor="#161B22", edgecolor=GRID_COL,
        labelcolor=FG, fontsize=11, loc="upper right",
    )

    plt.tight_layout()

    if save:
        out = os.path.join(OUTPUT_DIR, "nn_metrics_comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {out}")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PLOT 3: ROC CURVE OVERLAY (3 models)
# ═════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(roc_data: dict, results: list[dict],
                    save: bool = True) -> plt.Figure:
    """
    Overlay ROC curves for Random Forest, Logistic Regression, and Neural
    Network on a single axis with AUC scores in the legend.

    Parameters
    ----------
    roc_data : dict mapping model name → (fpr, tpr, thresholds)
    results  : list of metric dicts (for AUC lookup)
    save     : if True, save PNG to OUTPUT_DIR

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(BG)

    model_order = ["Random Forest", "Neural Network", "Logistic Regression"]

    for name in model_order:
        fpr, tpr, _ = roc_data[name]
        auc = next(r["AUC"] for r in results if r["Model"] == name)
        ax.plot(
            fpr, tpr,
            color=COLORS[name], lw=2.5,
            label=f"{name}  (AUC = {auc:.4f})",
            zorder=3,
        )
        # Shade under curve
        ax.fill_between(fpr, tpr, alpha=0.04, color=COLORS[name])

    # Random baseline
    ax.plot(
        [0, 1], [0, 1], "--",
        color="#586069", lw=1.3,
        label="Random Baseline  (AUC = 0.5000)",
        zorder=2,
    )
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="white")

    # Optimal point markers (closest to top-left corner)
    for name in model_order:
        fpr, tpr, _ = roc_data[name]
        dist   = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
        opt_ix = np.argmin(dist)
        ax.scatter(
            fpr[opt_ix], tpr[opt_ix],
            color=COLORS[name], s=80, zorder=5,
            marker="o", edgecolors="white", linewidths=0.8,
        )

    ax.set_xlabel("False Positive Rate", color=FG, fontsize=12)
    ax.set_ylabel("True Positive Rate",  color=FG, fontsize=12)
    ax.set_title(
        "ROC Curve Comparison — Random Forest · Logistic Regression · Neural Network",
        color=FG, fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.tick_params(colors=FG, labelsize=10)
    ax.grid(True, color=GRID_COL, alpha=0.5, linestyle="--", zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.legend(
        facecolor="#161B22", edgecolor=GRID_COL,
        labelcolor=FG, fontsize=11, loc="lower right",
    )

    # Annotation: AUC gap between RF and NN
    rf_auc = next(r["AUC"] for r in results if r["Model"] == "Random Forest")
    nn_auc = next(r["AUC"] for r in results if r["Model"] == "Neural Network")
    ax.text(
        0.38, 0.07,
        f"RF vs NN AUC gap: {rf_auc - nn_auc:.4f}",
        color=ANNO, fontsize=9, style="italic",
        transform=ax.transAxes,
    )

    plt.tight_layout()

    if save:
        out = os.path.join(OUTPUT_DIR, "nn_roc_comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {out}")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PLOT 4: CONFUSION MATRICES (3 models, side-by-side)
# ═════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(cm_dict: dict, save: bool = True) -> plt.Figure:
    """
    Side-by-side confusion matrices for all three models on the test set.

    Parameters
    ----------
    cm_dict : dict mapping model name → 2×2 numpy array
    save    : if True, save PNG to OUTPUT_DIR

    Returns
    -------
    matplotlib Figure
    """
    model_order = ["Random Forest", "Logistic Regression", "Neural Network"]
    cmaps       = ["Blues", "Oranges", "Greens"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG)
    fig.suptitle(
        "Confusion Matrices — Test Set Comparison  (n = 3,039)",
        color=FG, fontsize=14, fontweight="bold", y=1.02,
    )

    labels = ["Retained\n(0)", "Churned\n(1)"]

    for ax, name, cmap in zip(axes, model_order, cmaps):
        cm    = cm_dict[name]
        total = cm.sum()
        acc   = (cm[0, 0] + cm[1, 1]) / total
        prec  = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0,1]+cm[1,1]) > 0 else 0
        rec   = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1,0]+cm[1,1]) > 0 else 0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        ax.set_facecolor(BG)
        im = ax.imshow(cm, cmap=cmap, aspect="auto", vmin=0, vmax=cm.max())

        # Cell annotations — count + percentage
        for i in range(2):
            for j in range(2):
                count   = cm[i, j]
                pct     = count / total * 100
                cell_bg = im.cmap(im.norm(count))
                lum     = 0.299*cell_bg[0] + 0.587*cell_bg[1] + 0.114*cell_bg[2]
                txt_col = "white" if lum < 0.5 else "black"
                ax.text(
                    j, i,
                    f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color=txt_col,
                    fontsize=14, fontweight="bold",
                )

        # Labels for TP / TN / FP / FN quadrants
        quadrant_labels = {
            (0, 0): "TN", (0, 1): "FP",
            (1, 0): "FN", (1, 1): "TP",
        }
        for (i, j), ql in quadrant_labels.items():
            ax.text(
                j + 0.42, i - 0.42, ql,
                ha="right", va="top",
                color=ANNO, fontsize=8, style="italic",
            )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nRetained", "Predicted\nChurned"],
                           color=FG, fontsize=10)
        ax.set_yticklabels(labels, color=FG, fontsize=10, rotation=90, va="center")
        ax.set_xlabel("Predicted Label", color=FG, fontsize=10)
        ax.set_ylabel("Actual Label",    color=FG, fontsize=10)
        ax.set_title(
            f"{name}\n"
            f"Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}",
            color=FG, fontsize=10.5, fontweight="bold", pad=10,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    plt.tight_layout()

    if save:
        out = os.path.join(OUTPUT_DIR, "nn_confusion_matrices.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {out}")

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """
    End-to-end pipeline:
      1. Prepare data
      2. Train Neural Network (60 epochs) → collect training history
      3. Train sklearn models (RF + LR)
      4. Generate all predictions and metrics
      5. Produce and save all four plots
    """
    print("=" * 60)
    print("  Churn NN Visualisation Pipeline")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("\n[1/5] Preparing data …")
    data = prepare_data()
    y_te = data["y_te"]
    print(f"      Train {data['X_tr'].shape} | Test {data['X_te'].shape}")

    # ── 2. Train Neural Network ───────────────────────────────────────────
    print("\n[2/5] Training Neural Network …")
    nn_model, history = train_neural_network(data, epochs=60)

    # ── 3. Train sklearn models ───────────────────────────────────────────
    print("[3/5] Training sklearn models …")
    best_rfc, lrc = train_sklearn_models(data)

    # ── 4. Predictions & metrics ──────────────────────────────────────────
    print("[4/5] Computing predictions and metrics …")
    X_tr, X_te = data["X_tr"], data["X_te"]

    rf_proba = best_rfc.predict_proba(X_te)[:, 1]
    lr_proba = lrc.predict_proba(X_te)[:, 1]
    nn_proba = nn_model.predict_proba(data["X_te_t"])

    rf_pred  = best_rfc.predict(X_te)
    lr_pred  = lrc.predict(X_te)
    nn_pred  = (nn_proba >= 0.5).astype(int)

    def metrics(name, y_true, y_pred, y_proba):
        return {
            "Model":     name,
            "AUC":       round(roc_auc_score(y_true, y_proba), 4),
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "F1":        round(f1_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall":    round(recall_score(y_true, y_pred), 4),
        }

    results = [
        metrics("Random Forest",       y_te, rf_pred, rf_proba),
        metrics("Logistic Regression", y_te, lr_pred, lr_proba),
        metrics("Neural Network",      y_te, nn_pred, nn_proba),
    ]

    print("\n  ── Model Comparison ──────────────────────────────────")
    for r in results:
        print(
            f"  {r['Model']:<22} | AUC {r['AUC']:.4f} | "
            f"F1 {r['F1']:.4f} | Prec {r['Precision']:.4f} | "
            f"Rec {r['Recall']:.4f}"
        )
    print()

    roc_data = {
        "Random Forest":       roc_curve(y_te, rf_proba),
        "Logistic Regression": roc_curve(y_te, lr_proba),
        "Neural Network":      roc_curve(y_te, nn_proba),
    }

    cm_dict = {
        "Random Forest":       confusion_matrix(y_te, rf_pred),
        "Logistic Regression": confusion_matrix(y_te, lr_pred),
        "Neural Network":      confusion_matrix(y_te, nn_pred),
    }

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("[5/5] Generating plots …\n")

    plot_training_curves(history)
    plot_metric_bars(results)
    plot_roc_curves(roc_data, results)
    plot_confusion_matrices(cm_dict)

    print(f"\nAll four plots saved to: {OUTPUT_DIR}/")
    print("=" * 60)

    return results, history, roc_data, cm_dict


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, history, roc_data, cm_dict = main()
    plt.show()
