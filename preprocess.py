"""
preprocess.py
=============
Run once. Reads raw CMAPSS .txt files, produces 4 client .npz files.

Usage:
    python preprocess.py
    python preprocess.py --data_dir ./data --output_dir ./client_data

Outputs (written to output_dir):
    FD001.npz, FD002.npz, FD003.npz, FD004.npz
    weibull_params.json
    kl_divergence_matrix.csv
    figures/fig_noniid_distributions.pdf
    figures/fig_health_index.pdf
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import entropy, ks_2samp, weibull_min
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ── Config ───────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Loading ──────────────────────────────────────────────────────────────────
def load_cmapss(path: str) -> pd.DataFrame:
    cols = (
        ["unit_id", "cycle"]
        + [f"op_{i}" for i in range(1, 4)]
        + [f"s_{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    return df


# ── RUL labelling ────────────────────────────────────────────────────────────
def label_train_rul(df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    max_cycle = (
        df.groupby("unit_id")["cycle"].max().reset_index(name="max_cycle")
    )
    df = df.merge(max_cycle, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=max_rul)
    return df.drop(columns=["max_cycle"])


def label_test_rul(
    test_df: pd.DataFrame, rul_df: pd.DataFrame, max_rul: int
) -> pd.DataFrame:
    max_cycle = (
        test_df.groupby("unit_id")["cycle"].max().reset_index(name="max_cycle")
    )
    rul_df = rul_df.copy()
    rul_df["unit_id"] = rul_df.index + 1
    test_df = test_df.merge(max_cycle, on="unit_id").merge(rul_df, on="unit_id")
    test_df["RUL"] = (
        test_df["max_cycle"] + test_df["RUL"] - test_df["cycle"]
    ).clip(upper=max_rul)
    return test_df.drop(columns=["max_cycle"])


# ── Health Index ─────────────────────────────────────────────────────────────
def compute_health_index(
    df: pd.DataFrame, hi_sensors: list[str]
) -> pd.DataFrame:
    """
    Monotonically-increasing HI in [0, 1] via isotonic regression.
    Used as the target sequence for the physics consistency loss.
    """
    df = df.copy()
    df["HI"] = np.nan
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    for unit in df["unit_id"].unique():
        mask = df["unit_id"] == unit
        composite = df.loc[mask, hi_sensors].values.mean(axis=1)
        c_min, c_max = composite.min(), composite.max()
        if c_max - c_min < 1e-8:
            hi = np.zeros(len(composite))
        else:
            hi = (composite - c_min) / (c_max - c_min)
        t = np.arange(len(hi))
        df.loc[mask, "HI"] = iso.fit_transform(t, hi)

    return df


# ── Sliding window ───────────────────────────────────────────────────────────
def create_sequences(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences, labels = [], []
    for unit in df["unit_id"].unique():
        u = df[df["unit_id"] == unit].reset_index(drop=True)
        data = u[sensor_cols].values
        rul = u["RUL"].values
        for i in range(0, len(u) - window_size + 1, stride):
            sequences.append(data[i : i + window_size])
            labels.append(rul[i + window_size - 1])
    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.float32),
    )


# ── KL divergence ────────────────────────────────────────────────────────────
def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
    eps = 1e-10
    edges = np.linspace(
        min(p.min(), q.min()), max(p.max(), q.max()), bins + 1
    )
    ph, _ = np.histogram(p, bins=edges, density=True)
    qh, _ = np.histogram(q, bins=edges, density=True)
    ph = ph + eps; ph /= ph.sum()
    qh = qh + eps; qh /= qh.sum()
    return float(entropy(ph, qh))


# ── Weibull fitting ──────────────────────────────────────────────────────────
def fit_weibull(train_df: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    lifetimes = train_df.groupby("unit_id")["cycle"].max().values.astype(float)
    shape, _, scale = weibull_min.fit(lifetimes, floc=0)
    return float(shape), float(scale), lifetimes


# ── Figures ──────────────────────────────────────────────────────────────────
def plot_noniid(clients: dict, fig_dir: str):
    fd_keys = list(clients.keys())
    colors = ["#2E86C1", "#1E8449", "#E67E22", "#8E44AD"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, fd, c in zip(axes.flatten(), fd_keys, colors):
        y = clients[fd]["y_train"]
        ax.hist(y, bins=40, color=c, alpha=0.85, edgecolor="white", lw=0.4)
        ax.axvline(np.median(y), color="red", ls="--", lw=1.5,
                   label=f"Median={np.median(y):.0f}")
        ax.set_title(f"{fd}  (n={len(y):,})", fontweight="bold")
        ax.set_xlabel("RUL (cycles)"); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Non-IID RUL Label Distributions Across Federated Clients",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig_noniid_distributions.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


def plot_hi(clients: dict, fig_dir: str):
    fd_keys = list(clients.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, fd in zip(axes.flatten(), fd_keys):
        df = clients[fd]["train_df"]
        for uid in df["unit_id"].unique()[:4]:
            u = df[df["unit_id"] == uid]
            ax.plot(u["cycle"].values, u["HI"].values, alpha=0.8, lw=1.4)
        ax.set_title(f"{fd} — Health Index", fontweight="bold")
        ax.set_xlabel("Cycle"); ax.set_ylabel("HI")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Monotonic Health Index Trajectories (Isotonic Regression)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(fig_dir, "fig_health_index.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir   = args.data_dir   or cfg["data"]["data_dir"]
    output_dir = args.output_dir or cfg["data"]["output_dir"]
    set_seeds(cfg["reproducibility"]["seed"])

    sensor_cols  = [f"s_{i}" for i in cfg["data"]["selected_sensors"]]
    hi_sensors   = cfg["data"]["hi_sensors"]
    max_rul      = cfg["data"]["max_rul"]
    window_size  = cfg["data"]["window_size"]
    stride       = cfg["data"]["stride"]
    fd_keys      = cfg["data"]["clients"]

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    clients = {}

    # ── Load + preprocess each client ────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing all 4 clients...")
    for fd in fd_keys:
        train_df = load_cmapss(os.path.join(data_dir, f"train_{fd}.txt"))
        test_df  = load_cmapss(os.path.join(data_dir, f"test_{fd}.txt"))
        rul_df   = pd.read_csv(
            os.path.join(data_dir, f"RUL_{fd}.txt"), header=None, names=["RUL"]
        )

        keep = ["unit_id", "cycle"] + sensor_cols
        train_df = train_df[keep].copy()
        test_df  = test_df[keep].copy()

        train_df = label_train_rul(train_df, max_rul)
        test_df  = label_test_rul(test_df, rul_df, max_rul)

        # Per-client normalisation — never fit on test
        scaler = MinMaxScaler()
        train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
        test_df[sensor_cols]  = scaler.transform(test_df[sensor_cols])

        # Health index
        train_df = compute_health_index(train_df, hi_sensors)
        test_df  = compute_health_index(test_df, hi_sensors)

        X_train, y_train = create_sequences(train_df, sensor_cols, window_size, stride)
        X_test,  y_test  = create_sequences(test_df,  sensor_cols, window_size, stride)

        clients[fd] = {
            "X_train": X_train, "y_train": y_train,
            "X_test":  X_test,  "y_test":  y_test,
            "train_df": train_df, "test_df": test_df,
        }
        print(f"  {fd}: X_train={X_train.shape}  X_test={X_test.shape}")

    # ── Non-IID quantification ────────────────────────────────────────────────
    print("\n[2/5] Computing KL-divergence matrix (Table 1 in paper)...")
    kl_matrix = pd.DataFrame(index=fd_keys, columns=fd_keys, dtype=float)
    for i in fd_keys:
        for j in fd_keys:
            kl_matrix.loc[i, j] = (
                0.0 if i == j
                else round(kl_divergence(clients[i]["y_train"], clients[j]["y_train"]), 4)
            )
    print(kl_matrix.to_string())
    kl_matrix.to_csv(os.path.join(output_dir, "kl_divergence_matrix.csv"))

    # ── Weibull fitting ───────────────────────────────────────────────────────
    print("\n[3/5] Fitting Weibull parameters per client...")
    weibull_params = {}
    print(f"  {'Client':<8} {'k':>8} {'λ':>10} {'KS-stat':>10} {'p-value':>10} {'Fit'}")
    print(f"  {'-'*58}")
    for fd in fd_keys:
        k, lam, lifetimes = fit_weibull(clients[fd]["train_df"])
        sim_samples = weibull_min.rvs(k, scale=lam, size=len(lifetimes) * 20,
                                      random_state=cfg["reproducibility"]["seed"])
        ks_stat, p_val = ks_2samp(lifetimes, sim_samples)
        fit_ok = "PASS" if p_val > 0.05 else "FAIL"
        print(f"  {fd:<8} {k:>8.4f} {lam:>10.2f} {ks_stat:>10.4f} {p_val:>10.4f} {fit_ok}")
        weibull_params[fd] = {"k": k, "lambda": lam}

    with open(os.path.join(output_dir, "weibull_params.json"), "w") as f:
        json.dump(weibull_params, f, indent=2)

    # ── Save .npz files ───────────────────────────────────────────────────────
    print("\n[4/5] Saving client .npz files...")
    for fd in fd_keys:
        path = os.path.join(output_dir, f"{fd}.npz")
        np.savez(
            path,
            X_train=clients[fd]["X_train"],
            y_train=clients[fd]["y_train"],
            X_test =clients[fd]["X_test"],
            y_test =clients[fd]["y_test"],
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {path}  ({size_mb:.1f} MB)")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[5/5] Generating figures...")
    plot_noniid(clients, fig_dir)
    plot_hi(clients, fig_dir)

    print("\nPreprocessing complete. All outputs written to:", output_dir)


if __name__ == "__main__":
    main()