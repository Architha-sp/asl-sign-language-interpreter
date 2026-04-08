"""
STEP 2 — Build PCA subspace model from gestures.csv → pca_model.npz
=====================================================================
This is the core Linear Algebra step.

For each gesture label it:
  1. Stacks all 42-dim vectors into a matrix  X  (n_samples × 42)
  2. Computes the mean vector  μ
  3. Computes the covariance matrix  C = (X - μ)ᵀ(X - μ) / (n-1)
  4. Runs eigendecomposition on C  →  eigenvectors = "Eigen-gestures"
  5. Keeps the top-k eigenvectors as the gesture's subspace basis

Recognition later works by:
  - Projecting a live vector onto the subspace
  - Measuring reconstruction error (Euclidean distance)
  - Closest subspace = predicted gesture

Usage:
    python step2_train_pca.py --csv gestures.csv --out pca_model.npz --k 10
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def build_pca_model(csv_path, out_path, k=10, plot=True):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    labels     = df['label'].values
    features   = df.drop(columns=['label']).values.astype(np.float32)  # (N, 42)
    class_names = sorted(df['label'].unique())

    print(f"  Total samples : {len(df)}")
    print(f"  Gestures      : {class_names}")
    print(f"  Feature dims  : {features.shape[1]}")
    print(f"  PCA components: {k}\n")

    model = {}   # label → {mean, eigenvecs, eigenvals}

    for label in class_names:
        mask = labels == label
        X    = features[mask]          # (n, 42)
        n    = X.shape[0]

        if n < k:
            print(f"  WARNING [{label}]: only {n} samples, reducing k to {n}")
            k_use = n
        else:
            k_use = k

        # ── Linear Algebra core ────────────────────────────────────────────
        mu   = X.mean(axis=0)          # (42,)  mean vector
        Xc   = X - mu                  # centre the data
        C    = (Xc.T @ Xc) / (n - 1)  # (42×42) covariance matrix

        # eigendecomposition — eigh is for symmetric matrices (faster, stable)
        eigenvalues, eigenvectors = np.linalg.eigh(C)   # ascending order

        # flip to descending, keep top-k
        idx   = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]             # (42, 42)

        top_vecs = eigenvectors[:, :k_use]              # (42, k) — subspace basis
        top_vals = eigenvalues[:k_use]

        variance_explained = top_vals.sum() / (eigenvalues.sum() + 1e-9) * 100
        # ──────────────────────────────────────────────────────────────────

        model[label] = {
            'mean'      : mu,
            'eigenvecs' : top_vecs,
            'eigenvals' : top_vals,
        }
        print(f"  [{label}]  n={n:4d}  variance explained by top-{k_use}: {variance_explained:.1f}%")

    # save model
    save_dict = {}
    for label, data in model.items():
        safe = label.replace(' ', '_')
        save_dict[f"{safe}__mean"]     = data['mean']
        save_dict[f"{safe}__eigenvecs"] = data['eigenvecs']
        save_dict[f"{safe}__eigenvals"] = data['eigenvals']
    save_dict['class_names'] = np.array(class_names)

    np.savez(out_path, **save_dict)
    print(f"\nModel saved → {out_path}")

    # ── optional: visualise PCA scatter (first 2 components) ──────────────
    if plot:
        print("\nGenerating PCA scatter plot...")
        _plot_pca_scatter(features, labels, class_names, model, out_path)

    return model, class_names


def _plot_pca_scatter(features, labels, class_names, model, out_path):
    """Project all samples onto global top-2 PCs and scatter plot."""
    mu  = features.mean(axis=0)
    Xc  = features - mu
    C   = (Xc.T @ Xc) / (len(features) - 1)
    vals, vecs = np.linalg.eigh(C)
    idx  = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    PC2  = vecs[:, :2]                       # (42, 2)
    proj = Xc @ PC2                          # (N, 2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, label in enumerate(class_names):
        mask = labels == label
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   label=label, alpha=0.5, s=15, color=colors[i])

    ax.set_xlabel("PC 1  (most variation)")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA scatter — gesture clusters in 2D subspace")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = out_path.replace('.npz', '_pca_scatter.png')
    plt.savefig(plot_file, dpi=120)
    print(f"  Scatter plot saved → {plot_file}")
    plt.close()


def evaluate_model(csv_path, model, class_names, k=10):
    """Quick leave-some-out accuracy check on the CSV itself."""
    print("\n── Quick accuracy check (reconstruction distance) ──")
    df     = pd.read_csv(csv_path)
    labels = df['label'].values
    X      = df.drop(columns=['label']).values.astype(np.float32)

    correct = total = 0
    confusion = {c: {c2: 0 for c2 in class_names} for c in class_names}

    for vec, true_label in zip(X, labels):
        pred = _classify(vec, model, class_names)
        confusion[true_label][pred] += 1
        if pred == true_label:
            correct += 1
        total += 1

    acc = correct / total * 100
    print(f"  Overall accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"\n  Confusion matrix:")
    header = f"{'':12s}" + "".join(f"{c:10s}" for c in class_names)
    print("  " + header)
    for true in class_names:
        row = f"  {true:12s}" + "".join(
            f"{confusion[true][pred]:10d}" for pred in class_names)
        print(row)
    return acc


def _classify(vec, model, class_names):
    """Return the gesture label with smallest reconstruction distance."""
    best_label = None
    best_dist  = float('inf')
    for label in class_names:
        mu   = model[label]['mean']
        U    = model[label]['eigenvecs']      # (42, k)
        vc   = vec - mu
        proj = U @ (U.T @ vc)                # reconstruct via subspace
        dist = np.linalg.norm(vc - proj)
        if dist < best_dist:
            best_dist  = dist
            best_label = label
    return best_label


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="gestures.csv")
    ap.add_argument("--out",  default="pca_model.npz")
    ap.add_argument("--k",    type=int, default=10,
                    help="Number of principal components per gesture (default 10)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--eval",    action="store_true",
                    help="Run accuracy check after building model")
    args = ap.parse_args()

    model, class_names = build_pca_model(
        args.csv, args.out, k=args.k, plot=not args.no_plot)

    if args.eval:
        evaluate_model(args.csv, model, class_names, k=args.k)
