"""
STEP 4 — Evaluation: accuracy report + visualisations
======================================================
Splits gestures.csv into train/test, measures accuracy,
plots confusion matrix and eigenvector visualisation.

Usage:
    python step4_evaluate.py --csv gestures.csv --model pca_model.npz
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")


# ── load model ─────────────────────────────────────────────────────────────────
def load_model(path):
    data        = np.load(path, allow_pickle=True)
    class_names = data['class_names'].tolist()
    model = {}
    for label in class_names:
        safe = label.replace(' ', '_')
        model[label] = {
            'mean'     : data[f"{safe}__mean"],
            'eigenvecs': data[f"{safe}__eigenvecs"],
        }
    return model, class_names


def classify(vec, model, class_names):
    dists = {}
    for label in class_names:
        mu   = model[label]['mean']
        U    = model[label]['eigenvecs']
        vc   = vec - mu
        proj = U @ (U.T @ vc)
        dists[label] = float(np.linalg.norm(vc - proj))
    best = min(dists, key=dists.get)
    return best, dists[best]


# ── train/test split evaluation ────────────────────────────────────────────────
def evaluate(csv_path, model, class_names, test_size=0.2):
    df      = pd.read_csv(csv_path)
    X       = df.drop(columns=['label']).values.astype(np.float32)
    y       = df['label'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    print(f"Train samples: {len(X_tr)}   Test samples: {len(X_te)}\n")

    y_pred = []
    for vec in X_te:
        pred, _ = classify(vec, model, class_names)
        y_pred.append(pred)
    y_pred = np.array(y_pred)

    acc = (y_pred == y_te).mean() * 100
    print(f"Test accuracy: {acc:.1f}%\n")
    print(classification_report(y_te, y_pred, target_names=class_names))
    return y_te, y_pred, acc


# ── plot confusion matrix ──────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, class_names, out_file):
    cm   = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title("Confusion matrix — test set", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    print(f"Confusion matrix → {out_file}")
    plt.close()


# ── plot eigenvectors as ghost-hand shapes ─────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def vec_to_hand(vec):
    """Convert a 42-float vector back to (21, 2) landmark array."""
    return vec.reshape(21, 2)

def draw_hand(ax, pts, color='steelblue', alpha=1.0, lw=1.5):
    """Draw skeleton on matplotlib axis."""
    for a, b in CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                color=color, lw=lw, alpha=alpha)
    ax.scatter(pts[:, 0], pts[:, 1], s=20, color=color,
               alpha=alpha, zorder=5)

def plot_eigengestures(model, class_names, out_file, n_eigen=3):
    """
    For each gesture, plot:
    - mean hand shape
    - top-3 eigen-gestures (mean ± eigen-vector)
    """
    n_classes = len(class_names)
    cols = n_eigen + 1   # mean + n_eigen
    fig = plt.figure(figsize=(cols * 2.2, n_classes * 2.2))
    gs  = gridspec.GridSpec(n_classes, cols, hspace=0.4, wspace=0.3)

    col_titles = ["Mean shape"] + [f"Eigen {i+1}" for i in range(n_eigen)]

    for row, label in enumerate(class_names):
        mu   = model[label]['mean']            # (42,)
        vecs = model[label]['eigenvecs']        # (42, k)

        # ── mean hand ──────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        pts = vec_to_hand(mu)
        pts[:, 1] = -pts[:, 1]   # flip Y for natural orientation
        draw_hand(ax, pts, color='#2ec99e', lw=2)
        ax.set_xlim(-2, 2); ax.set_ylim(-2.5, 1)
        ax.set_aspect('equal'); ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[0], fontsize=9, color='#888')
        ax.text(0, -2.3, label, ha='center', va='bottom',
                fontsize=9, fontweight='bold')

        # ── top eigen-gestures ─────────────────────────────────────────
        for col in range(min(n_eigen, vecs.shape[1])):
            ax = fig.add_subplot(gs[row, col + 1])
            ev = vecs[:, col]
            scale = np.linalg.norm(mu) * 0.4

            pts_pos = vec_to_hand(mu + scale * ev)
            pts_neg = vec_to_hand(mu - scale * ev)
            pts_pos[:, 1] = -pts_pos[:, 1]
            pts_neg[:, 1] = -pts_neg[:, 1]

            draw_hand(ax, pts_neg, color='#f5a623', alpha=0.4, lw=1)
            draw_hand(ax, pts_pos, color='#7c6af7', alpha=0.8, lw=1.5)

            ax.set_xlim(-2, 2); ax.set_ylim(-2.5, 1)
            ax.set_aspect('equal'); ax.axis('off')
            if row == 0:
                ax.set_title(col_titles[col + 1], fontsize=9, color='#888')

    fig.suptitle("Eigen-gestures per class  (purple = +direction, orange = −direction)",
                 fontsize=11, y=1.01)
    plt.savefig(out_file, dpi=120, bbox_inches='tight')
    print(f"Eigen-gestures plot → {out_file}")
    plt.close()


# ── PCA scatter (global 2-component) ──────────────────────────────────────────
def plot_scatter(csv_path, class_names, out_file):
    df      = pd.read_csv(csv_path)
    X       = df.drop(columns=['label']).values.astype(np.float32)
    labels  = df['label'].values

    mu  = X.mean(axis=0)
    Xc  = X - mu
    C   = (Xc.T @ Xc) / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)
    idx  = np.argsort(vals)[::-1]
    PC2  = vecs[:, idx][:, :2]
    proj = Xc @ PC2

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(class_names):
        mask = labels == label
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   label=label, alpha=0.4, s=12, color=colors[i])

    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("Gesture clusters in 2D PCA space")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=120)
    print(f"PCA scatter → {out_file}")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   default="gestures.csv")
    ap.add_argument("--model", default="pca_model.npz")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    model, class_names = load_model(args.model)

    # 1. accuracy
    y_true, y_pred, acc = evaluate(args.csv, model, class_names, args.test_size)

    # 2. confusion matrix
    plot_confusion(y_true, y_pred, class_names, "confusion_matrix.png")

    # 3. eigen-gesture visualisation
    plot_eigengestures(model, class_names, "eigengestures.png")

    # 4. PCA scatter
    plot_scatter(args.csv, class_names, "pca_scatter.png")

    print(f"\nAll done. Final test accuracy: {acc:.1f}%")
