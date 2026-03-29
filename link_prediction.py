"""
Link Prediction on Facebook Social Graph
=========================================
Predicts whether a friendship (edge) exists between two people
using graph-structural features + Random Forest classifier.

Dataset: Stanford SNAP - facebook_combined.txt
"""

import random
import warnings
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────
# 1. LOAD GRAPH
# ─────────────────────────────────────────────

def load_graph(path: str) -> nx.Graph:
    G = nx.read_edgelist(path, nodetype=int)
    print(f"Graph loaded:")
    print(f"  Nodes : {G.number_of_nodes():,}")
    print(f"  Edges : {G.number_of_edges():,}")
    print(f"  Density     : {nx.density(G):.4f}")
    print(f"  Avg degree  : {sum(d for _, d in G.degree()) / G.number_of_nodes():.2f}")
    return G


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_features(G: nx.Graph, u: int, v: int, degree: dict) -> dict:
    """
    Compute rich structural features for a node pair (u, v).
    All features are standard link-prediction heuristics.
    """
    common = list(nx.common_neighbors(G, u, v))
    n_common = len(common)

    du, dv = degree[u], degree[v]

    # Jaccard coefficient: common / union of neighbors
    union = du + dv - n_common
    jaccard = n_common / union if union > 0 else 0.0

    # Adamic-Adar: weight common neighbors by inverse log-degree
    adamic_adar = sum(
        1.0 / np.log(degree[w]) for w in common if degree[w] > 1
    )

    # Resource allocation index
    resource_alloc = sum(
        1.0 / degree[w] for w in common if degree[w] > 0
    )

    # Preferential attachment
    pref_attach = du * dv

    return {
        "deg_u": du,
        "deg_v": dv,
        "common_neighbors": n_common,
        "jaccard": jaccard,
        "adamic_adar": adamic_adar,
        "resource_alloc": resource_alloc,
        "pref_attach": pref_attach,
        "deg_diff": abs(du - dv),
        "deg_sum": du + dv,
    }


def build_dataset(G: nx.Graph, sample_size: int = 50_000) -> pd.DataFrame:
    print(f"\nBuilding dataset ({sample_size:,} positive + {sample_size:,} negative)...")

    edges = list(G.edges())
    non_edges = list(nx.non_edges(G))

    edges_sample = random.sample(edges, min(sample_size, len(edges)))
    non_edges_sample = random.sample(non_edges, min(sample_size, len(non_edges)))

    degree = dict(G.degree())
    records = []

    for label, pairs in [(1, edges_sample), (0, non_edges_sample)]:
        for u, v in pairs:
            feats = compute_features(G, u, v, degree)
            feats["label"] = label
            records.append(feats)

    df = pd.DataFrame(records)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Class balance: {df['label'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "deg_u", "deg_v", "common_neighbors",
    "jaccard", "adamic_adar", "resource_alloc",
    "pref_attach", "deg_diff", "deg_sum",
]

def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest"        : RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression"  : LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}

    print("\nTraining models...\n")
    for name, model in models.items():
        scaler = StandardScaler() if name == "Logistic Regression" else None
        X_tr = scaler.fit_transform(X_train) if scaler else X_train
        X_te = scaler.transform(X_test) if scaler else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        cm    = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model": model,
            "scaler": scaler,
            "accuracy": acc,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "report": report,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "X_test": X_te,
        }

        print(f"  {name}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    ROC-AUC  : {auc:.4f}")
        print()

    return results, X_train, y_train


# ─────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────

def plot_results(results: dict, X_train: pd.DataFrame, y_train: pd.Series):
    palette = {
        "Random Forest"      : "#1D9E75",
        "Gradient Boosting"  : "#378ADD",
        "Logistic Regression": "#7F77DD",
    }

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    title_kw  = dict(color="#e8e8e8", fontsize=11, fontweight="bold", pad=10)
    label_kw  = dict(color="#a0a0a0", fontsize=9)
    tick_kw   = dict(colors="#707070", labelsize=8)

    # ── Panel 1: Model comparison bar chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1d27")
    names   = list(results.keys())
    accs    = [results[n]["accuracy"] * 100 for n in names]
    aucs    = [results[n]["roc_auc"] * 100  for n in names]
    x       = np.arange(len(names))
    w       = 0.35
    colors  = [palette[n] for n in names]

    bars1 = ax1.bar(x - w/2, accs, w, color=colors, alpha=0.9, label="Accuracy")
    bars2 = ax1.bar(x + w/2, aucs, w, color=colors, alpha=0.5, label="ROC-AUC")

    for bar in list(bars1) + list(bars2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{bar.get_height():.1f}", ha="center", va="bottom",
                 color="#e8e8e8", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(" ", "\n") for n in names], **label_kw)
    ax1.set_ylim(85, 101)
    ax1.set_ylabel("Score (%)", **label_kw)
    ax1.set_title("Model Comparison", **title_kw)
    ax1.tick_params(axis="both", **tick_kw)
    ax1.spines[:].set_visible(False)
    ax1.legend(fontsize=7, labelcolor="#a0a0a0", facecolor="#1a1d27", framealpha=0.5)

    # ── Panel 2: Confusion matrix (best model = RF) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1d27")
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    cm = results[best_name]["confusion_matrix"]
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="YlGn",
        linewidths=0.5, linecolor="#0f1117",
        xticklabels=["Pred: No edge", "Pred: Edge"],
        yticklabels=["True: No edge", "True: Edge"],
        ax=ax2, cbar=False,
        annot_kws={"size": 10, "color": "#1a1d27", "weight": "bold"},
    )
    ax2.set_title(f"Confusion Matrix\n({best_name})", **title_kw)
    ax2.tick_params(axis="both", **tick_kw)

    # ── Panel 3: Feature importance (RF) ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1a1d27")
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    colors_fi = ["#1D9E75" if v > 0.1 else "#378ADD" if v > 0.04 else "#7F77DD"
                 for v in importances]
    ax3.barh(importances.index, importances.values, color=colors_fi, alpha=0.85)
    ax3.set_title("Feature Importance\n(Random Forest)", **title_kw)
    ax3.tick_params(axis="both", **tick_kw)
    ax3.spines[:].set_visible(False)
    ax3.set_xlabel("Importance", **label_kw)

    # ── Panel 4: ROC curves ──
    from sklearn.metrics import roc_curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#1a1d27")
    ax4.plot([0, 1], [0, 1], "--", color="#555", linewidth=0.8, label="Random")
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
        ax4.plot(fpr, tpr, color=palette[name], linewidth=1.8,
                 label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax4.set_xlabel("False Positive Rate", **label_kw)
    ax4.set_ylabel("True Positive Rate", **label_kw)
    ax4.set_title("ROC Curves", **title_kw)
    ax4.tick_params(axis="both", **tick_kw)
    ax4.legend(fontsize=7, labelcolor="#a0a0a0", facecolor="#1a1d27", framealpha=0.5)
    ax4.spines[:].set_visible(False)

    # ── Panel 5: Precision-Recall by class ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#1a1d27")
    metrics   = ["precision", "recall", "f1-score"]
    classes   = ["0", "1"]
    class_labels = ["No edge", "Edge"]
    bar_width = 0.25
    x5 = np.arange(len(metrics))
    for i, (cls, lbl) in enumerate(zip(classes, class_labels)):
        vals = [results[best_name]["report"][cls][m] for m in metrics]
        color = "#1D9E75" if cls == "1" else "#378ADD"
        ax5.bar(x5 + i * bar_width, vals, bar_width, label=lbl, color=color, alpha=0.85)
    ax5.set_xticks(x5 + bar_width / 2)
    ax5.set_xticklabels(["Precision", "Recall", "F1"], **label_kw)
    ax5.set_ylim(0.8, 1.02)
    ax5.set_title(f"Per-class Metrics\n({best_name})", **title_kw)
    ax5.tick_params(axis="both", **tick_kw)
    ax5.spines[:].set_visible(False)
    ax5.legend(fontsize=7, labelcolor="#a0a0a0", facecolor="#1a1d27", framealpha=0.5)

    # ── Panel 6: Summary table ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#1a1d27")
    ax6.axis("off")
    table_data = []
    for name in results:
        r = results[name]
        rep = r["report"]
        table_data.append([
            name.replace(" ", "\n"),
            f"{r['accuracy']:.4f}",
            f"{r['roc_auc']:.4f}",
            f"{rep['1']['precision']:.4f}",
            f"{rep['1']['recall']:.4f}",
        ])
    tbl = ax6.table(
        cellText=table_data,
        colLabels=["Model", "Acc", "AUC", "Prec", "Recall"],
        cellLoc="center", loc="center",
        bbox=[0, 0.1, 1, 0.8],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#252836" if row % 2 == 0 else "#1a1d27")
        cell.set_text_props(color="#e8e8e8")
        cell.set_edgecolor("#0f1117")
    ax6.set_title("Summary Table", **title_kw)

    fig.suptitle("Link Prediction · Facebook Social Graph",
                 color="#e8e8e8", fontsize=14, fontweight="bold", y=0.98)

    plt.savefig("/mnt/user-data/outputs/link_prediction_results.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Plot saved → link_prediction_results.png")
    plt.show()


# ─────────────────────────────────────────────
# 5. PREDICT NEW PAIRS
# ─────────────────────────────────────────────

def predict_pair(G: nx.Graph, model, scaler, u: int, v: int) -> dict:
    """Predict whether an edge is likely between nodes u and v."""
    degree = dict(G.degree())
    feats  = compute_features(G, u, v, degree)
    X      = pd.DataFrame([feats])[FEATURE_COLS]
    if scaler:
        X = scaler.transform(X)
    prob = model.predict_proba(X)[0][1]
    return {
        "nodes": (u, v),
        "link_probability": round(prob, 4),
        "prediction": "EDGE" if prob >= 0.5 else "NO EDGE",
        "features": feats,
    }


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load
    G = load_graph("data/facebook_combined.txt")

    # Build dataset
    df = build_dataset(G, sample_size=50_000)

    # Train
    results, X_train, y_train = train_and_evaluate(df)

    # Plot
    plot_results(results, X_train, y_train)

    # Example: predict a specific pair
    best_model  = results["Random Forest"]["model"]
    best_scaler = results["Random Forest"]["scaler"]

    sample_nodes = list(G.nodes())[:5]
    print("\nExample predictions:")
    for u, v in zip(sample_nodes, sample_nodes[1:]):
        if not G.has_edge(u, v):
            result = predict_pair(G, best_model, best_scaler, u, v)
            print(f"  Nodes {result['nodes']}: {result['prediction']} "
                  f"(prob={result['link_probability']:.3f})")
