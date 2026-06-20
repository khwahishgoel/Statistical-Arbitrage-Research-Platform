"""
Statistical Arbitrage Research Platform
Phase 8b: K-Means Clustering — Automated Pair Discovery (Python)

Instead of hand-picking sectors, uses K-Means to automatically
group stocks by return profile, then finds cointegration candidates
within each cluster. Compares ML-discovered pairs vs manual pairs.
"""

from pathlib import Path
from datetime import datetime, timezone
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from statsmodels.tsa.stattools import coint
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from db import load_prices, get_db

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_CLUSTERS    = 6      # number of stock clusters
CORR_THRESH   = 0.75   # minimum correlation within cluster
COINT_PVALUE  = 0.10   # cointegration significance level
ROLL_WINDOW   = 60     # days for feature rolling stats
RANDOM_STATE  = 42


#Build return features for clustering 
def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Features per stock for K-Means:
      - annualised return
      - annualised volatility
      - skewness of returns
      - max drawdown
      - rolling correlation with market (beta proxy)
      - momentum (12-month return)
    """
    log_ret  = np.log(prices / prices.shift(1)).dropna()
    market   = log_ret.mean(axis=1)   # equal-weight market proxy

    features = {}
    for ticker in log_ret.columns:
        r = log_ret[ticker].dropna()
        if len(r) < 252:
            continue

        ann_ret  = r.mean() * 252
        ann_vol  = r.std()  * np.sqrt(252)
        skew     = float(r.skew())

        # Max drawdown
        cum      = (1 + r).cumprod()
        roll_max = cum.cummax()
        dd       = (cum - roll_max) / roll_max
        max_dd   = float(dd.min())

        # Beta to market
        cov      = np.cov(r.values, market.loc[r.index].values)
        beta     = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

        # 12-month momentum
        momentum = float(r.tail(252).sum())

        features[ticker] = {
            "ann_return":  ann_ret,
            "ann_vol":     ann_vol,
            "skewness":    skew,
            "max_drawdown": max_dd,
            "beta":        beta,
            "momentum":    momentum,
        }

    return pd.DataFrame(features).T


# Find optimal k via silhouette score
def find_optimal_k(X_scaled: np.ndarray,
                   k_range: range = range(3, 9)) -> int:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                    n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(scores, key=scores.get)
    print("  Silhouette scores by k:")
    for k, s in sorted(scores.items()):
        marker = " ←" if k == best_k else ""
        print(f"    k={k}: {s:.4f}{marker}")
    return best_k


# Run K-Means
def cluster_stocks(features: pd.DataFrame) -> tuple:
    scaler  = StandardScaler()
    X       = scaler.fit_transform(features.values)

    print("Finding optimal number of clusters...")
    best_k  = find_optimal_k(X)
    print(f"\nFitting K-Means with k={best_k}...")

    km      = KMeans(n_clusters=best_k, random_state=RANDOM_STATE,
                     n_init=20)
    labels  = km.fit_predict(X)

    clusters = pd.Series(labels, index=features.index, name="cluster")

    # Print cluster composition
    print("\nCluster composition:")
    for c in sorted(clusters.unique()):
        members = clusters[clusters == c].index.tolist()
        print(f"  Cluster {c}: {members}")

    return clusters, km, scaler, X, best_k


# Scan pairs within each cluster 
def scan_cluster_pairs(prices: pd.DataFrame,
                       clusters: pd.Series) -> pd.DataFrame:
    """
    For each cluster, test all pairs for correlation + cointegration.
    Only considers pairs within the same cluster.
    """
    log_ret  = np.log(prices / prices.shift(1)).dropna()
    results  = []
    total_tested = 0

    for c in sorted(clusters.unique()):
        members = clusters[clusters == c].index.tolist()
        # Only keep members that exist in prices
        members = [m for m in members if m in prices.columns]
        if len(members) < 2:
            continue

        cluster_pairs = list(itertools.combinations(members, 2))
        total_tested += len(cluster_pairs)

        for a, b in cluster_pairs:
            price_a = prices[a].dropna()
            price_b = prices[b].dropna()

            # Align series
            common  = price_a.index.intersection(price_b.index)
            if len(common) < 252:
                continue
            price_a = price_a[common]
            price_b = price_b[common]

            # Correlation check
            ret_a   = log_ret[a].reindex(common).dropna()
            ret_b   = log_ret[b].reindex(common).dropna()
            corr    = float(ret_a.corr(ret_b))
            if corr < CORR_THRESH:
                continue

            # Cointegration test
            score, pvalue, _ = coint(price_a.values, price_b.values)
            if pvalue > COINT_PVALUE:
                continue

            # Hedge ratio via OLS
            from numpy.linalg import lstsq
            A    = np.column_stack([price_b.values,
                                    np.ones(len(price_b))])
            beta, alpha = lstsq(A, price_a.values, rcond=None)[0]

            # Half-life
            spread = price_a.values - beta * price_b.values
            delta  = np.diff(spread)
            lagged = spread[:-1]
            lam    = np.dot(delta, lagged) / np.dot(lagged, lagged)
            hl     = -np.log(2) / np.log(1 + lam) if lam < 0 else np.nan

            # Already in manual pairs?
            manual_pairs = {"CVX/XOM", "XLE/XOM", "GS/MS",
                            "XOM/CVX", "XOM/XLE", "MS/GS"}
            is_new = f"{a}/{b}" not in manual_pairs and \
                     f"{b}/{a}" not in manual_pairs

            results.append({
                "pair":        f"{a}/{b}",
                "ticker_a":    a,
                "ticker_b":    b,
                "cluster":     int(c),
                "correlation": round(corr, 4),
                "coint_pvalue": round(pvalue, 4),
                "beta":        round(beta, 6),
                "halflife":    round(hl, 1) if not np.isnan(hl) else None,
                "is_new_pair": is_new,
                "tradeable":   (not np.isnan(hl) and
                                hl is not None and
                                5 <= hl <= 126),
            })

    df = pd.DataFrame(results)
    print(f"\nTested {total_tested} within-cluster pairs")
    print(f"Passed correlation (>{CORR_THRESH}): {len(results)}")
    if not df.empty:
        print(f"Cointegrated (p<{COINT_PVALUE}): {len(df)}")
        print(f"Tradeable (hl 5-126d): "
              f"{df['tradeable'].sum()}")
        new = df[df['is_new_pair'] & df['tradeable']]
        print(f"NEW pairs not in manual list: {len(new)}")
    return df


#PCA visualisation
def plot_clusters(X_scaled: np.ndarray, clusters: pd.Series,
                  features: pd.DataFrame, pairs_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.35)

    colors = ["#185FA5", "#3B6D11", "#E24B4A", "#854F0B",
              "#534AB7", "#0F6E56", "#A32D2D", "#633806"]

    # Panel 1: PCA scatter
    ax1 = fig.add_subplot(gs[0, :])
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2  = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    for c in sorted(clusters.unique()):
        mask   = clusters.values == c
        tickers = clusters[mask].index.tolist()
        ax1.scatter(X2[mask, 0], X2[mask, 1],
                    color=colors[c % len(colors)],
                    s=80, alpha=0.8, label=f"Cluster {c}",
                    zorder=3)
        for i, t in enumerate(tickers):
            idx = list(clusters.index).index(t)
            ax1.annotate(t, (X2[idx, 0], X2[idx, 1]),
                         fontsize=7.5, ha="center", va="bottom",
                         xytext=(0, 5), textcoords="offset points",
                         color=colors[c % len(colors)])

    ax1.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=10)
    ax1.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=10)
    ax1.set_title("K-Means Stock Clusters (PCA projection)",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.legend(fontsize=9, loc="best", framealpha=0.8)
    ax1.grid(lw=0.3, color="#E8E6DF")

    # Panel 2: Cluster feature heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    feat_norm = (features - features.mean()) / features.std()
    cluster_means = feat_norm.copy()
    cluster_means["cluster"] = clusters
    cm = cluster_means.groupby("cluster").mean()

    im = ax2.imshow(cm.values, aspect="auto", cmap="RdYlGn",
                    vmin=-1.5, vmax=1.5)
    ax2.set_xticks(range(len(cm.columns)))
    ax2.set_xticklabels(cm.columns, rotation=35, ha="right",
                         fontsize=8)
    ax2.set_yticks(range(len(cm.index)))
    ax2.set_yticklabels([f"Cluster {c}" for c in cm.index],
                         fontsize=9)
    ax2.set_title("Cluster feature profiles\n(normalised)",
                  fontsize=10, pad=6)
    plt.colorbar(im, ax=ax2, shrink=0.8)
    for i in range(len(cm.index)):
        for j in range(len(cm.columns)):
            ax2.text(j, i, f"{cm.values[i,j]:.2f}",
                     ha="center", va="center",
                     fontsize=7,
                     color="black" if abs(cm.values[i,j]) < 1 else "white")

    # Panel 3: Discovered pairs table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    if not pairs_df.empty:
        tradeable = pairs_df[pairs_df["tradeable"]].copy()
        if not tradeable.empty:
            rows = []
            for _, row in tradeable.head(10).iterrows():
                new_tag = "★ NEW" if row["is_new_pair"] else "known"
                rows.append([
                    row["pair"],
                    f"{row['correlation']:.2f}",
                    f"{row['coint_pvalue']:.3f}",
                    f"{row['halflife']:.0f}d" if row["halflife"] else "—",
                    new_tag,
                ])
            table = ax3.table(
                cellText=rows,
                colLabels=["Pair", "ρ", "p-val", "HL", "Status"],
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            for (r, c), cell in table.get_celld().items():
                cell.set_edgecolor("#E8E6DF")
                if r == 0:
                    cell.set_facecolor("#E6F1FB")
                    cell.set_text_props(fontweight="bold",
                                        color="#0C447C")
                elif rows[r-1][4] == "★ NEW":
                    cell.set_facecolor("#EAF3DE")
                elif r % 2 == 0:
                    cell.set_facecolor("#F8F7F4")
            ax3.set_title("Tradeable pairs discovered",
                          fontsize=10, pad=6, y=1.02)
        else:
            ax3.text(0.5, 0.5, "No tradeable pairs found",
                     ha="center", va="center",
                     transform=ax3.transAxes,
                     fontsize=10, color="#888780")

    path = OUTPUT_DIR / "kmeans_clusters.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved → {path}")

if __name__ == "__main__":
    print("── Phase 8b: K-Means Pair Discovery ──\n")

    prices   = load_prices()
    print(f"Loaded prices: {prices.shape[0]} days × "
          f"{prices.shape[1]} tickers\n")

    # Build features
    print("Building stock features...")
    features = build_features(prices)
    print(f"  Features built for {len(features)} stocks\n")

    # Cluster
    clusters, km, scaler, X_scaled, best_k = cluster_stocks(features)

    # Scan pairs within clusters
    print(f"\nScanning pairs within {best_k} clusters...")
    pairs_df = scan_cluster_pairs(prices, clusters)

    # Print results
    if not pairs_df.empty:
        tradeable = pairs_df[pairs_df["tradeable"]]
        new_pairs = tradeable[tradeable["is_new_pair"]]

        print("\n── Tradeable pairs found ──")
        if not tradeable.empty:
            print(tradeable[["pair", "cluster", "correlation",
                              "coint_pvalue", "halflife",
                              "is_new_pair"]].to_string(index=False))
        else:
            print("  None found — try lowering CORR_THRESH or COINT_PVALUE")

        if not new_pairs.empty:
            print(f"\n── {len(new_pairs)} NEW pairs not in manual list ──")
            print(new_pairs[["pair", "correlation",
                              "coint_pvalue",
                              "halflife"]].to_string(index=False))

    # Plot
    print("\nGenerating cluster chart...")
    plot_clusters(X_scaled, clusters, features, pairs_df)

    # Save to MongoDB
    col = get_db()["kmeans_pairs"]
    col.drop()
    if not pairs_df.empty:
        docs = []
        for _, row in pairs_df.iterrows():
            doc = row.to_dict()
            doc["discovered_at"] = datetime.now(timezone.utc)
            # Convert numpy types
            for k, v in doc.items():
                if isinstance(v, np.integer):
                    doc[k] = int(v)
                elif isinstance(v, np.floating):
                    doc[k] = float(v) if not np.isnan(v) else None
                elif isinstance(v, np.bool_):
                    doc[k] = bool(v)
            docs.append(doc)
        col.insert_many(docs)
        print(f"Saved {len(docs)} pairs to MongoDB → "
              f"kmeans_pairs collection")

    print(f"""
── Summary ──
  Stocks clustered:   {len(features)}
  Clusters found:     {best_k}
  Pairs tested:       within-cluster only
  Tradeable found:    {pairs_df['tradeable'].sum() if not pairs_df.empty else 0}
  New discoveries:    {len(new_pairs) if not pairs_df.empty else 0}

Phase 8b complete.
""")