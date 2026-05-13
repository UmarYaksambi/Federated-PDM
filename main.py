# %% [markdown]
# ## 0 — Install Dependencies & Download Data

# %%
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "flwr[simulation]",
    "torch",
    "numpy",
    "pandas==2.2.2",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pyyaml",
    "huggingface_hub",
    "protobuf<6",
    "cryptography<44"
])

print("✅ Dependencies installed (compatible)")

# %%
import os

for dirname, _, filenames in os.walk('/kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

BASE_PATH = "/kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps"

train_path = f"{BASE_PATH}/train_FD001.txt"
test_path  = f"{BASE_PATH}/test_FD001.txt"
rul_path   = f"{BASE_PATH}/RUL_FD001.txt"

# %% [markdown]
# ## 1 — Configuration

# %%
import yaml

CONFIG = {
    "data": {
        "data_dir": "./data",
        "output_dir": "./client_data",
        "clients": ["FD001", "FD002", "FD003", "FD004"],
        "selected_sensors": [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        "hi_sensors": ["s_2", "s_3", "s_4", "s_11", "s_15"],
        "max_rul": 125,
        "window_size": 30,
        "stride": 2,               # 2x fewer samples — faster rounds
    },
    "simulation": {
        "n_trajectories": 300,
        "late_rul_threshold": 50,
        "noise_std": 0.02,
        "max_lifetime": 400,
        "seed": 42,
    },
    "model": {
        "input_size": 14,
        "hidden_channels": 64,
        "num_levels": 4,
        "kernel_size": 3,
        "dropout": 0.15,
        "mc_samples": 20,          # 50→20: faster eval
    },
    "loss": {"lambda_physics": 0.1},
    "training": {
        "epochs_local": 2,         # 5→2: far fewer local steps — less drift
        "batch_size": 256,         # 64→256: 4x fewer gradient steps, better GPU util
        "learning_rate": 6e-4,     # scaled with batch (sqrt rule: 1e-3 × sqrt(256/64))
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "noise_std": 0.02,         # Gaussian augmentation per round
    },
    "federation": {
        "num_rounds": 400,         # 50→400 for full convergence
        "min_clients": 4,
        "strategy": "similarity_weighted",
        "fedprox_mu": 0.01,
        "checkpoint_every": 50,
    },
    "evaluation": {
        "seeds": [42],
        "results_dir": "./results",
    },
    "reproducibility": {"seed": 42},
}

with open("config.yaml", "w") as f:
    yaml.dump(CONFIG, f, default_flow_style=False)

cfg = CONFIG
print("✅ Config written")

# %% [markdown]
# ## 2 — Core Modules (Model, Loss, Simulator, Evaluation)
# All module code is defined inline so the notebook is self-contained.

# %%
import random, json, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import entropy, kstest, weibull_min
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"Device: {DEVICE}  AMP: {USE_AMP}")

# %% [markdown]
# ### 2.1 — TCN Model

# %%
class _ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self._pad = pad
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _chomp(self, x):
        return x[:, :, :-self._pad] if self._pad > 0 else x

    def forward(self, x):
        out = self.relu(self._chomp(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._chomp(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return x.mean(dim=1)


class TCN(nn.Module):
    def __init__(self, input_size=14, hidden_channels=64, num_levels=4, kernel_size=3, dropout=0.2):
        super().__init__()
        blocks = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else hidden_channels
            blocks.append(_ResidualBlock(in_ch, hidden_channels, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.attention = _TemporalAttention(hidden_channels, num_heads=4)
        self.head = nn.Sequential(nn.Linear(hidden_channels, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                w = getattr(m, 'weight_v', None) or getattr(m, 'weight', None)
                if w is not None: nn.init.kaiming_normal_(w, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.attention(x)
        return self.head(x).squeeze(-1)

    def forward_with_hi(self, x):
        x_perm = x.permute(0, 2, 1)
        feat = self.tcn(x_perm)
        hi_seq = torch.sigmoid(feat.mean(dim=1))
        attended = self.attention(feat)
        rul_pred = self.head(attended).squeeze(-1)
        return rul_pred, hi_seq


def build_model(cfg):
    m = cfg["model"]
    return TCN(m["input_size"], m["hidden_channels"], m["num_levels"], m["kernel_size"], m["dropout"])

print("✅ TCN defined")

# %% [markdown]
# ### 2.2 — Loss Functions

# %%
class HybridRULLoss(nn.Module):
    def __init__(self, lambda_physics=0.1):
        super().__init__()
        self.lambda_physics = lambda_physics

    def forward(self, rul_pred, rul_true, hi_seq):
        mse = F.mse_loss(rul_pred, rul_true)
        hi_diff = hi_seq[:, 1:] - hi_seq[:, :-1]
        violation = torch.relu(-hi_diff).mean()
        total = mse + self.lambda_physics * violation
        return total, {"mse": mse.item(), "physics": violation.item(), "total": total.item()}

class AdaptiveHybridRULLoss(nn.Module):
    def __init__(self, lambda_max=0.5, warmup_epochs=50):
        super().__init__()
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def step(self): 
        self.current_epoch += 1

    def forward(self, rul_pred, rul_true, hi_seq):
        lam = self.lambda_max * min(1.0, self.current_epoch / self.warmup_epochs)
        mse = F.mse_loss(rul_pred, rul_true)
        hi_diff = hi_seq[:, 1:] - hi_seq[:, :-1]
        violation = torch.relu(-hi_diff).mean()
        total = mse + lam * violation
        return total, {"mse": mse.item(), "physics": violation.item(), "lambda": lam, "total": total.item()}

class FedProxLoss(nn.Module):
    def __init__(self, mu=0.01):
        super().__init__()
        self.mu = mu

    def proximal_term(self, local_model, global_params):
        device = next(local_model.parameters()).device
        prox = torch.zeros(1, device=device)
        for lp, gp in zip(local_model.parameters(), global_params):
            prox = prox + torch.linalg.norm(lp - gp.detach().to(device)) ** 2
        return (self.mu / 2.0) * prox

    def forward(self, base_loss, local_model, global_params):
        return base_loss + self.proximal_term(local_model, global_params)


class _MSEOnlyLoss:
    def __call__(self, rul_pred, rul_true, hi_seq):
        loss = nn.MSELoss()(rul_pred, rul_true)
        return loss, {"mse": loss.item(), "physics": 0.0, "total": loss.item()}

print("✅ Losses defined")

# %% [markdown]
# ### 2.3 — Evaluation Utilities

# %%
class CMAPSSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loader(X, y, batch_size, shuffle=False, **kw):
    return DataLoader(CMAPSSDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available(), **kw)

def compute_metrics(rul_pred, rul_true):
    err = rul_pred - rul_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    nz = rul_true != 0
    mape = float(np.mean(np.abs(err[nz] / rul_true[nz])) * 100) if nz.any() else float("nan")
    phm = float(np.where(err < 0, np.exp(-err/13.0)-1, np.exp(err/10.0)-1).sum())
    return {"rmse": rmse, "mae": mae, "mape": mape, "phm_score": phm}

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu().numpy())
        trues.append(y.numpy())
    return compute_metrics(np.concatenate(preds), np.concatenate(trues))

def mc_predict(model, loader, device, n_samples=50):
    model.train()
    all_X, all_y = [], []
    with torch.no_grad():
        for X, y in loader: all_X.append(X); all_y.append(y)
    all_X = torch.cat(all_X).to(device)
    true_rul = torch.cat(all_y).numpy()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(all_X).cpu().numpy())
    preds = np.stack(preds)
    model.eval()
    return preds.mean(0), preds.std(0), true_rul

def log_results(results, csv_path, mode="a"):
    hdr = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=results.keys())
        if hdr: w.writeheader()
        w.writerow(results)

print("✅ Evaluation utilities defined")

# %% [markdown]
# ### 2.4 — Weibull Simulator

# %%
class WeibullSimulator:
    def __init__(self, k, lam, cfg, seed=42):
        self.k, self.lam, self.seed = k, lam, seed
        sim = cfg["simulation"]
        self.n_sensors = len(cfg["data"]["selected_sensors"])
        self.window_size = cfg["data"]["window_size"]
        self.max_rul = cfg["data"]["max_rul"]
        self.noise_std = sim["noise_std"]
        self.max_lifetime = sim["max_lifetime"]
        self.threshold = sim["late_rul_threshold"]

    def _single_trajectory(self, offset):
        rng = np.random.default_rng(self.seed + offset)
        T = int(weibull_min.rvs(self.k, scale=self.lam, random_state=self.seed + offset))
        T = max(T, self.window_size + 5); T = min(T, self.max_lifetime)
        cycles = np.linspace(0, T, T)
        deg = weibull_min.cdf(cycles, c=self.k, scale=self.lam)
        weights = rng.uniform(0.3, 1.0, self.n_sensors)
        signals = np.outer(deg, weights)
        signals += rng.normal(0, self.noise_std, signals.shape)
        signals = np.clip(signals, 0, 1).astype(np.float32)
        rul = np.maximum(0, np.minimum(self.max_rul, T - cycles)).astype(np.float32)
        return signals, rul

    def generate(self, n_trajectories):
        seqs, labs = [], []
        for i in range(n_trajectories):
            sig, rul = self._single_trajectory(i)
            for s in range(0, len(sig) - self.window_size + 1):
                wr = rul[s + self.window_size - 1]
                if wr <= self.threshold:
                    seqs.append(sig[s:s+self.window_size]); labs.append(wr)
        if not seqs:
            return np.empty((0, self.window_size, self.n_sensors), dtype=np.float32), np.empty(0, dtype=np.float32)
        X, y = np.array(seqs, dtype=np.float32), np.array(labs, dtype=np.float32)
        idx = np.random.default_rng(self.seed).permutation(len(X))
        return X[idx], y[idx]

print("✅ WeibullSimulator defined")

# %% [markdown]
# ## 3 — Preprocessing

# %%

# def load_cmapss(path):
#     cols = ["unit_id", "cycle"] + [f"op_{i}" for i in range(1,4)] + [f"s_{i}" for i in range(1,22)]
#     return pd.read_csv(path, sep=r"\s+", header=None, names=cols)

def load_cmapss(path):
    col_names = (
        ["unit_id", "cycle"] +
        [f"op_{i}" for i in range(1, 4)] +
        [f"s_{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(
        path,
        sep=r"\s+",        # handles variable whitespace
        header=None,
        names=col_names,
        engine="python"
    )
    # Drop trailing NaN columns if any (some files have trailing spaces)
    df = df.dropna(axis=1, how="all")
    return df
    
def label_train_rul(df, max_rul):
    mc = df.groupby("unit_id")["cycle"].max().reset_index(name="max_cycle")
    df = df.merge(mc, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=max_rul)
    return df.drop(columns=["max_cycle"])

def label_test_rul(test_df, rul_df, max_rul):
    mc = test_df.groupby("unit_id")["cycle"].max().reset_index(name="max_cycle")
    rul_df = rul_df.copy(); rul_df["unit_id"] = rul_df.index + 1
    test_df = test_df.merge(mc, on="unit_id").merge(rul_df, on="unit_id")
    test_df["RUL"] = (test_df["max_cycle"] + test_df["RUL"] - test_df["cycle"]).clip(upper=max_rul)
    return test_df.drop(columns=["max_cycle"])

def compute_health_index(df, hi_sensors):
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    df["HI"] = 0.0

    if len(hi_sensors) == 0:
        return df  # or raise, depending on your design

    for uid in df["unit_id"].unique():
        mask = df["unit_id"] == uid

        sub = df.loc[mask, hi_sensors].values

        if sub.size == 0:
            continue

        composite = np.nanmean(sub, axis=1)

        # Remove NaNs BEFORE isotonic
        valid_idx = ~np.isnan(composite)
        if valid_idx.sum() < 2:
            continue

        composite = composite[valid_idx]

        c_min, c_max = composite.min(), composite.max()
        if c_max - c_min < 1e-8:
            hi = np.zeros_like(composite)
        else:
            hi = (composite - c_min) / (c_max - c_min)

        x = np.arange(len(hi))

        iso = IsotonicRegression(out_of_bounds="clip")
        hi_iso = iso.fit_transform(x, hi)

        idx = df.index[mask][valid_idx]
        df.loc[idx, "HI"] = hi_iso

    return df

def create_sequences(df, sensor_cols, window_size, stride):
    seqs, labs = [], []
    for unit in df["unit_id"].unique():
        u = df[df["unit_id"] == unit].reset_index(drop=True)
        data, rul = u[sensor_cols].values, u["RUL"].values
        for i in range(0, len(u) - window_size + 1, stride):
            seqs.append(data[i:i+window_size]); labs.append(rul[i+window_size-1])
    return np.array(seqs, dtype=np.float32), np.array(labs, dtype=np.float32)

def kl_divergence(p, q, bins=30):
    eps = 1e-10
    edges = np.linspace(min(p.min(), q.min()), max(p.max(), q.max()), bins+1)
    ph, _ = np.histogram(p, bins=edges, density=True)
    qh, _ = np.histogram(q, bins=edges, density=True)
    ph = ph + eps; ph /= ph.sum()
    qh = qh + eps; qh /= qh.sum()
    return float(entropy(ph, qh))
print("dome")

# %%
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Force correct path
data_dir = "/kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps"
output_dir = cfg["data"]["output_dir"]

# --- FIX 1: robust sensor name handling ---
def normalize_sensor_names(sensor_list):
    return [
        s if str(s).startswith("s_") else f"s_{s}"
        for s in sensor_list
    ]

sensor_cols = normalize_sensor_names(cfg["data"]["selected_sensors"])
hi_sensors  = normalize_sensor_names(cfg["data"]["hi_sensors"])

max_rul = cfg["data"]["max_rul"]
window_size = cfg["data"]["window_size"]
stride = cfg["data"]["stride"]
fd_keys = cfg["data"]["clients"]

fig_dir = os.path.join(output_dir, "figures")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

set_seeds(cfg["reproducibility"]["seed"])

clients = {}

print("\n[1/5] Loading and preprocessing all 4 clients...")
print("Sensor cols:", sensor_cols)
print("HI sensors (raw):", hi_sensors)



# %%
for fd in fd_keys:
    train_path = os.path.join(data_dir, f"train_{fd}.txt")
    test_path  = os.path.join(data_dir, f"test_{fd}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{fd}.txt")

    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(test_path),  f"Missing {test_path}"
    assert os.path.exists(rul_path),   f"Missing {rul_path}"

    train_df = load_cmapss(train_path)
    test_df  = load_cmapss(test_path)

    rul_df = pd.read_csv(rul_path, header=None, names=["RUL"])

    # --- FIX 2: auto-detect available sensors ---
    available_sensors = [c for c in train_df.columns if c.startswith("s_")]

    # Keep only sensors that actually exist
    sensor_cols_valid = [s for s in sensor_cols if s in available_sensors]

    if len(sensor_cols_valid) == 0:
        raise ValueError(f"No valid sensor columns found. Available: {available_sensors[:10]}")

    # Use valid ones only
    keep = ["unit_id", "cycle"] + sensor_cols_valid
    train_df = train_df[keep].copy()
    test_df  = test_df[keep].copy()

    # Label RUL
    train_df = label_train_rul(train_df, max_rul)
    test_df  = label_test_rul(test_df, rul_df, max_rul)

    # Normalize
    scaler = MinMaxScaler()
    train_df[sensor_cols_valid] = scaler.fit_transform(train_df[sensor_cols_valid])
    test_df[sensor_cols_valid]  = scaler.transform(test_df[sensor_cols_valid])

    # --- FIX 3: safe HI sensor selection ---
    valid_hi = [s for s in hi_sensors if s in sensor_cols_valid]

    if len(valid_hi) == 0:
        print(f"⚠️ {fd}: No valid HI sensors. Falling back to all sensors.")
        valid_hi = sensor_cols_valid  # fallback instead of crash

    # --- FIX 4: NaN-safe HI computation ---
    train_df = compute_health_index(train_df, valid_hi)
    test_df  = compute_health_index(test_df, valid_hi)

    # Sequence generation
    X_train, y_train = create_sequences(train_df, sensor_cols_valid, window_size, stride)
    X_test, y_test   = create_sequences(test_df, sensor_cols_valid, window_size, stride)

    lifetimes = train_df.groupby("unit_id")["cycle"].max().values.astype(float)

    clients[fd] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "lifetimes": lifetimes
    }

    print(f"{fd}: X_train={X_train.shape} | X_test={X_test.shape}")

# %%
print("\n[2/5] KL-divergence matrix...")
kl_matrix = pd.DataFrame(index=fd_keys, columns=fd_keys, dtype=float)
for i in fd_keys:
    for j in fd_keys:
        kl_matrix.loc[i, j] = 0.0 if i == j else round(kl_divergence(clients[i]["lifetimes"], clients[j]["lifetimes"]), 4)
print(kl_matrix.to_string())
kl_matrix.to_csv(os.path.join(output_dir, "kl_divergence_matrix.csv"))

# %%
print("\n[3/5] Fitting Weibull parameters...")
weibull_params = {}
for fd in fd_keys:
    lt = clients[fd]["lifetimes"]
    shape, _, scale = weibull_min.fit(lt, floc=0)
    ks_stat, p_val = kstest(lt, "weibull_min", args=(shape, 0, scale))
    print(f"  {fd}: k={shape:.4f}  λ={scale:.2f}  KS={ks_stat:.4f}  p={p_val:.4f}")
    weibull_params[fd] = {"k": float(shape), "lambda": float(scale)}
with open(os.path.join(output_dir, "weibull_params.json"), "w") as f:
    json.dump(weibull_params, f, indent=2)

# %%
print("\n[4/5] Saving .npz files...")
for fd in fd_keys:
    path = os.path.join(output_dir, f"{fd}.npz")
    np.savez(path, X_train=clients[fd]["X_train"], y_train=clients[fd]["y_train"],
             X_test=clients[fd]["X_test"], y_test=clients[fd]["y_test"])
    print(f"  {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

# %%
print("\n[5/5] Generating preprocessing figures...")
# Fig 1: Non-IID distributions
colors_4 = ["#2E86C1", "#1E8449", "#E67E22", "#8E44AD"]
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, fd, c in zip(axes.flatten(), fd_keys, colors_4):
    y = clients[fd]["y_train"]
    ax.hist(y, bins=40, color=c, alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(np.median(y), color="red", ls="--", lw=1.5, label=f"Median={np.median(y):.0f}")
    ax.set_title(f"{fd}  (n={len(y):,})", fontweight="bold")
    ax.set_xlabel("RUL"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Non-IID RUL Label Distributions Across Federated Clients", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "fig1_noniid.pdf"), bbox_inches="tight", dpi=300)
plt.show(); print("  Saved fig1")

# Fig 2: Health Index
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, fd in zip(axes.flatten(), fd_keys):
    df = clients[fd]["train_df"]
    for uid in df["unit_id"].unique()[:4]:
        u = df[df["unit_id"] == uid]
        ax.plot(u["cycle"].values, u["HI"].values, alpha=0.8, lw=1.4)
    ax.set_title(f"{fd} — Health Index", fontweight="bold")
    ax.set_xlabel("Cycle"); ax.set_ylabel("HI"); ax.set_ylim(-0.05, 1.05)
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Monotonic Health Index Trajectories", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "fig2_health_index.pdf"), bbox_inches="tight", dpi=300)
plt.show(); print("  Saved fig2")

print("\n✅ Preprocessing complete")

# %%
print(3/0)

# %% [markdown]
# ## 4 — Centralised Training (E1 — Upper Bound)

# %%
def train_centralised(cfg, seed, epochs):
    set_seeds(seed)
    device = DEVICE
    print(f"\n[Centralised] seed={seed} epochs={epochs} device={device}")

    # Pool all client data
    train_X, train_y = [], []
    test_data = {}
    for fd in cfg["data"]["clients"]:
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        train_X.append(npz["X_train"]); train_y.append(npz["y_train"])
        test_data[fd] = (npz["X_test"], npz["y_test"])
    X_train, y_train = np.concatenate(train_X), np.concatenate(train_y)
    train_loader = make_loader(X_train, y_train, cfg["training"]["batch_size"], shuffle=True)
    print(f"  Train: {X_train.shape}")

    model = build_model(cfg).to(device)
    criterion = HybridRULLoss(cfg["loss"]["lambda_physics"])
    opt = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=USE_AMP):
                pred, hi = model.forward_with_hi(X_b)
                loss, _ = criterion(pred, y_b, hi)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += loss.item()
        sched.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:>4}/{epochs}  loss={total_loss/max(len(train_loader),1):.4f}")

    # Evaluate
    print("\n  Per-client results:")
    results = {"experiment": "centralised", "seed": seed}
    all_p, all_t = [], []
    for fd, (Xt, yt) in test_data.items():
        loader = make_loader(Xt, yt, cfg["training"]["batch_size"])
        mp, sp, tr = mc_predict(model, loader, device, cfg["model"]["mc_samples"])
        m = compute_metrics(mp, tr); all_p.append(mp); all_t.append(tr)
        print(f"    {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  PHM={m['phm_score']:.1f}  Unc={sp.mean():.3f}")
        results[f"{fd}_rmse"] = round(m["rmse"], 4); results[f"{fd}_mae"] = round(m["mae"], 4)
        results[f"{fd}_phm"] = round(m["phm_score"], 2); results[f"{fd}_unc"] = round(float(sp.mean()), 4)

    ov = compute_metrics(np.concatenate(all_p), np.concatenate(all_t))
    results["overall_rmse"] = round(ov["rmse"], 4); results["overall_mae"] = round(ov["mae"], 4)
    results["overall_phm"] = round(ov["phm_score"], 2)
    print(f"\n  OVERALL: RMSE={ov['rmse']:.2f}  MAE={ov['mae']:.2f}  PHM={ov['phm_score']:.1f}")

    os.makedirs(cfg["evaluation"]["results_dir"], exist_ok=True)
    ckpt = os.path.join(cfg["evaluation"]["results_dir"], f"centralised_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt); print(f"  Checkpoint: {ckpt}")
    return results, model

# %%
# Run centralised training (E1)
epochs_central = cfg["training"]["epochs_local"] * cfg["federation"]["num_rounds"]  # fair compute budget
csv_central = os.path.join(cfg["evaluation"]["results_dir"], "centralised.csv")
os.makedirs(cfg["evaluation"]["results_dir"], exist_ok=True)

centralised_model = None
# for seed in cfg["evaluation"]["seeds"]:
#     row, centralised_model = train_centralised(cfg, seed, epochs_central)
#     log_results(row, csv_central)
print(f"\n✅ Centralised results saved → {csv_central}")

# %%
from huggingface_hub import HfApi, login

HF_REPO = "Unmeshraj/federated-pdm-cmapss"  # your repo

try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except:
    hf_token = os.environ.get("HF_TOKEN", None)

login(token=hf_token)
api = HfApi()
api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")

# Upload everything saved so far
import glob
files = glob.glob("./results/*.pt") + glob.glob("./results/*.csv")
files.append("config.yaml")

for fpath in files:
    if os.path.exists(fpath):
        api.upload_file(path_or_fileobj=fpath, path_in_repo=os.path.basename(fpath), repo_id=HF_REPO)
        print(f"✅ {os.path.basename(fpath)}")

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# %% [markdown]
# ## 5 — Federated Training (E2–E4

# %%
import time, math
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# --- Server strategies ---

def _weighted_average(results, weights):
    n_layers = len(results[0][0])
    return [sum(w * np.array(params[i]) for w, (params, _) in zip(weights, results)) for i in range(n_layers)]

def _agg_metrics(metrics):
    """Aggregate metrics, skipping non-numeric keys like 'fd'."""
    if not metrics: return {}
    total = sum(n for n, _ in metrics)
    keys = [k for k in metrics[0][1].keys() if isinstance(metrics[0][1][k], (int, float))]
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}

def build_fedavg_strategy(cfg):
    # FIX: pass round + num_rounds to clients for inter-round LR decay
    num_rounds = cfg["federation"]["num_rounds"]
    def fit_config(server_round: int) -> dict:
        return {"round": server_round, "num_rounds": num_rounds}
    return FedAvg(min_fit_clients=cfg["federation"]["min_clients"],
                  min_evaluate_clients=cfg["federation"]["min_clients"],
                  min_available_clients=cfg["federation"]["min_clients"],
                  on_fit_config_fn=fit_config)

class _BaseCustomStrategy(fl.server.strategy.Strategy):
    def __init__(self, cfg, initial_params):
        self.cfg = cfg; self.min_clients = cfg["federation"]["min_clients"]
        self._initial = ndarrays_to_parameters(initial_params)
    def initialize_parameters(self, client_manager=None, **kwargs):
        return self._initial
    def configure_fit(self, server_round=None, parameters=None, client_manager=None, **kwargs):
        # FIX: pass num_rounds so client can compute inter-round LR decay
        config = {"round": server_round, "num_rounds": self.cfg["federation"]["num_rounds"]}
        return [(c, fl.common.FitIns(parameters, config)) for c in client_manager.sample(self.min_clients)]
    def configure_evaluate(self, server_round=None, parameters=None, client_manager=None, **kwargs):
        return [(c, fl.common.EvaluateIns(parameters, {"round": server_round})) for c in client_manager.sample(self.min_clients)]
    def aggregate_fit(self, server_round=None, results=None, failures=None, **kwargs): 
        raise NotImplementedError
    def aggregate_evaluate(self, sr, results, failures, **kwargs):
        if not results: return None, {}
        total = sum(r.num_examples for _, r in results)
        wl = sum(r.loss * r.num_examples for _, r in results) / total
        agg = _agg_metrics([(r.num_examples, r.metrics) for _, r in results])
        return wl, agg
    def evaluate(self, sr, parameters=None, **kwargs): return None


class FedProxStrategy(_BaseCustomStrategy):
    def aggregate_fit(self, sr, results, failures):
        if not results: return None, {}
        pl = [(parameters_to_ndarrays(fr.parameters), fr.num_examples) for _, fr in results]
        total = sum(n for _, n in pl)
        ws = np.array([n/total for _, n in pl])
        agg = _weighted_average(pl, ws)
        return ndarrays_to_parameters(agg), _agg_metrics([(fr.num_examples, fr.metrics) for _, fr in results if fr.metrics is not None])

class SimilarityWeightedStrategy(_BaseCustomStrategy):
    def __init__(self, cfg, initial_params, kl_matrix_path):
        super().__init__(cfg, initial_params)
        kl = pd.read_csv(kl_matrix_path, index_col=0)
        self.fd_keys = cfg["data"]["clients"]
        # weights_dict keyed by fd string (e.g. "FD001") — NOT by index
        mean_kl = {fd: kl.loc[fd, [f for f in self.fd_keys if f != fd]].mean() for fd in self.fd_keys}
        # sims = {fd: np.exp(-mean_kl[fd]) for fd in self.fd_keys}
        sims = {fd: np.exp(-0.3 * mean_kl[fd]) for fd in self.fd_keys}  # softer decay
        total = sum(sims.values())
        self.weights_dict = {fd: s / total for fd, s in sims.items()}
        print("\n[SimilarityWeighted] Aggregation weights:")
        for fd in self.fd_keys:
            print(f"  {fd}: weight={self.weights_dict[fd]:.4f}  (mean_KL={mean_kl[fd]:.4f})")

    def aggregate_fit(self, sr, results, failures):
        if not results: return None, {}
        fallback_w = 1.0 / len(self.fd_keys)
        cd = []
        for proxy, fr in results:
            params = parameters_to_ndarrays(fr.parameters)
            fd = (fr.metrics or {}).get("fd", "")
            orig_w = self.weights_dict.get(fd, fallback_w)
            cd.append((params, fr.num_examples, orig_w))
        rw = np.array([w for _, _, w in cd]); nw = rw / rw.sum()
        nl = len(cd[0][0])
        agg = [sum(nw_i * np.array(params[i]) for (params, _, _), nw_i in zip(cd, nw)) for i in range(nl)]
        return ndarrays_to_parameters(agg), _agg_metrics([(fr.num_examples, fr.metrics) for _, fr in results if fr.metrics is not None])

def build_strategy(name, cfg, init_params):
    if name == "fedavg": return build_fedavg_strategy(cfg)
    elif name == "fedprox": return FedProxStrategy(cfg, init_params)
    elif name == "similarity_weighted":
        return SimilarityWeightedStrategy(cfg, init_params, os.path.join(cfg["data"]["output_dir"], "kl_divergence_matrix.csv"))
    else: raise ValueError(f"Unknown strategy: {name}")

print("✅ FL strategies defined")


# %% [markdown]
# ### 5.1 — FL Client & Tracking + Run Function

# %%
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# --- FL Client ---
class PDMClient(fl.client.NumPyClient):
    def __init__(self, fd, cfg, use_simulation=True, use_fedprox=False, device=DEVICE, use_amp=True):
        self.fd, self.cfg, self.use_simulation, self.use_fedprox, self.device = fd, cfg, use_simulation, use_fedprox, device
        # AMP only on CUDA
        self.use_amp = use_amp and (device.type == "cuda")
        self._base_seed = cfg["reproducibility"]["seed"]
        tc = cfg["training"]
        self.epochs, self.batch_size, self.lr, self.wd = tc["epochs_local"], tc["batch_size"], tc["learning_rate"], tc["weight_decay"]
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        self.X_train, self.y_train, self.X_test, self.y_test = npz["X_train"], npz["y_train"], npz["X_test"], npz["y_test"]
        if use_simulation:
            with open(os.path.join(cfg["data"]["output_dir"], "weibull_params.json")) as f: params = json.load(f)
            self.simulator = WeibullSimulator(k=params[fd]["k"], lam=params[fd]["lambda"], cfg=cfg, seed=self._base_seed)
        self.criterion = HybridRULLoss(cfg["loss"]["lambda_physics"])
        self.fedprox_loss = FedProxLoss(mu=cfg["federation"]["fedprox_mu"])
        self.model = build_model(cfg).to(device)
        print(f"  Client {fd} | train={len(self.X_train):,} test={len(self.X_test):,} sim={'on' if use_simulation else 'off'} amp={'on' if self.use_amp else 'off'}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        ns = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(ns, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Inter-round cosine LR decay
        round_num    = int(config.get("round", 1))
        total_rounds = int(config.get("num_rounds", self.cfg["federation"].get("num_rounds", 400)))
        cos_factor   = 0.5 * (1.0 + math.cos(math.pi * (round_num - 1) / max(total_rounds - 1, 1)))
        effective_lr = self.lr * (0.1 + 0.9 * cos_factor)

        gp = [p.clone().detach() for p in self.model.parameters()]

        # Round-specific shuffle + Gaussian noise augmentation
        rng = np.random.default_rng(self._base_seed + round_num * 7_919)
        noise_std = self.cfg["training"].get("noise_std", 0.0)
        if self.use_simulation:
            Xs, ys = self.simulator.generate(self.cfg["simulation"]["n_trajectories"])
            Xtr = np.concatenate([self.X_train, Xs]); ytr = np.concatenate([self.y_train, ys])
            idx = rng.permutation(len(Xtr))
            Xtr, ytr = Xtr[idx], ytr[idx]
        else:
            idx = rng.permutation(len(self.X_train))
            Xtr = self.X_train[idx].copy(); ytr = self.y_train[idx]

        if noise_std > 0:
            Xtr = Xtr + rng.normal(0, noise_std, Xtr.shape).astype(np.float32)
            Xtr = np.clip(Xtr, 0.0, 1.0)

        loader = make_loader(Xtr, ytr, self.batch_size, shuffle=False)
        opt = AdamW(self.model.parameters(), lr=effective_lr, weight_decay=self.wd)
        sched = CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = GradScaler(enabled=self.use_amp)
        self.model.train(); tl = 0.0
        for _ in range(self.epochs):
            el = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred, hi = self.model.forward_with_hi(Xb)
                    loss, _ = self.criterion(pred, yb, hi)
                    if self.use_fedprox: loss = self.fedprox_loss(loss, self.model, gp)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); el += loss.item()
            sched.step(); tl = el / max(len(loader), 1)
        return self.get_parameters({}), len(Xtr), {"train_loss": tl, "n_samples": len(Xtr), "fd": self.fd, "lr": effective_lr}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters); self.model.eval()
        loader = make_loader(self.X_test, self.y_test, self.batch_size)
        p, t = [], []
        with torch.no_grad():
            for X, y in loader:
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    p.append(self.model(X.to(self.device)).cpu().float().numpy())
                t.append(y.numpy())
        m = compute_metrics(np.concatenate(p), np.concatenate(t))
        return float(m["rmse"]), len(self.X_test), m


# --- Personalized FL Client (shared backbone, private head) ---
class PersonalizedPDMClient(PDMClient):
    """Shared TCN backbone, personalized regression head per client."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.personal_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        ).to(self.device)
        self.personal_opt = AdamW(self.personal_head.parameters(), lr=self.lr)

    def get_parameters(self, config):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items()
                if "head" not in name]

    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        backbone_keys = [k for k in sd.keys() if "head" not in k]
        ns = {k: torch.tensor(v, dtype=torch.float32).to(self.device)
              for k, v in zip(backbone_keys, parameters)}
        sd.update(ns)
        self.model.load_state_dict(sd, strict=True)

    def forward_personal(self, x):
        x = x.permute(0, 2, 1)
        feat = self.model.tcn(x)
        attended = self.model.attention(feat)
        return self.personal_head(attended).squeeze(-1)

    def compute_sensor_importance(self, X_batch):
        self.model.eval()
        X = torch.tensor(X_batch[:32], requires_grad=True).to(self.device)
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            pred = self.model(X)
        pred.sum().backward()
        importance = X.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        self.model.train()
        return importance


def make_client_fn(cfg, use_sim, use_fp, device, use_amp=True):
    """Flower-compatible client_fn using Context API (Flower >= 1.x)."""
    fdk = cfg["data"]["clients"]
    def fn(context):
        partition_id = context.node_config.get("partition-id", int(context.node_id))
        fd = fdk[int(partition_id) % len(fdk)]
        return PDMClient(fd, cfg, use_sim, use_fp, device, use_amp).to_client()
    return fn

# --- Tracking wrapper with best_params + checkpoint upload ---
class _TrackingStrategy(fl.server.strategy.Strategy):
    def __init__(self, w, eval_every=10, checkpoint_every=50, results_dir='./results', seed=42):
        self._w = w; self.eval_every = max(1, eval_every)
        self.checkpoint_every = checkpoint_every
        self.results_dir = results_dir; self.seed = seed
        self.round_metrics = []; self.last_params = None
        self.best_params = None; self.best_rmse = float('inf')
    def initialize_parameters(self, client_manager=None, **kwargs):
        return self._w.initialize_parameters(client_manager)
    def configure_fit(self, server_round=None, parameters=None, client_manager=None, **kwargs):
        return self._w.configure_fit(server_round, parameters, client_manager)
    def configure_evaluate(self, server_round=None, parameters=None, client_manager=None, **kwargs):
        if server_round % self.eval_every != 0: return []
        return self._w.configure_evaluate(server_round, parameters, client_manager)
    def evaluate(self, sr, parameters=None, **kwargs):
        return self._w.evaluate(sr, parameters)
    def aggregate_fit(self, server_round=None, results=None, failures=None, **kwargs):
        params, met = self._w.aggregate_fit(server_round, results, failures)
        if params is not None:
            self.last_params = parameters_to_ndarrays(params)
            # Checkpoint every N rounds
            if server_round % self.checkpoint_every == 0:
                ckpt = os.path.join(self.results_dir,
                    f'proposed_optimized_round{server_round}_seed{self.seed}.pt')
                os.makedirs(self.results_dir, exist_ok=True)
                m = build_model(cfg).to('cpu')
                m.load_state_dict({k: torch.tensor(v) for k, v in
                                   zip(m.state_dict().keys(), self.last_params)})
                torch.save(m.state_dict(), ckpt)
                try: safe_upload(ckpt)
                except Exception: pass
        return params, met

    def aggregate_evaluate(self, server_round=None, results=None, failures=None, **kwargs):
        loss, met = self._w.aggregate_evaluate(server_round, results, failures)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if met:
            rec = dict(met); rec['round'] = server_round; self.round_metrics.append(rec)
            total_r = self._w.cfg['federation']['num_rounds']
            if 'rmse' in rec and rec['rmse'] < self.best_rmse:
                self.best_rmse = rec['rmse']; self.best_params = self.last_params
            kv = '  '.join(f'{k}={v:.4f}' for k, v in rec.items()
                           if k != 'round' and isinstance(v, float))
            print(f'  Round {server_round:>3}/{total_r}  {kv}', flush=True)
        return loss, met

            kv = '  '.join(f'{k}={v:.4f}' for k, v in rec.items()
                           if k != 'round' and isinstance(v, float))
            print(f'  Round {server_round:>3}/{total_r}  {kv}', flush=True)

        return loss, met
print('ye')

# %%
MODE_CONFIG = {
    "fedavg":   (False, False, "fedavg"),
    "fedprox":  (False, True,  "fedprox"),
    "proposed": (False,  False, "similarity_weighted"),
}

def run_federated(cfg, mode, seed, eval_every=5):
    set_seeds(seed)
    device = DEVICE
    use_sim, use_fp, sname = MODE_CONFIG[mode]
    nc = len(cfg["data"]["clients"]); nr = cfg["federation"]["num_rounds"]
    rd = cfg["evaluation"]["results_dir"]; os.makedirs(rd, exist_ok=True)
    use_amp = device.type == "cuda"
    print(f"\n[Federated | {mode.upper()}] seed={seed} rounds={nr} device={device} AMP={use_amp}")

    init_model = build_model(cfg)
    init_params = [val.detach().cpu().numpy() for val in init_model.state_dict().values()]
    base = build_strategy(sname, cfg, init_params)
    strategy = _TrackingStrategy(base, eval_every=10,
                                  checkpoint_every=cfg['federation'].get('checkpoint_every', 50),
                                  results_dir=rd, seed=seed)
    client_fn = make_client_fn(cfg, use_sim, use_fp, device, use_amp=use_amp)

    import ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_gpus=2, num_cpus=4)

    gpu_frac = 0.25

    t0 = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=nc,
        config=fl.server.ServerConfig(num_rounds=nr),
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.5},
    )
    print(f'\n  Simulation finished in {(time.time()-t0)/60:.1f} min')

    use_params = strategy.best_params if strategy.best_params is not None else strategy.last_params
    if use_params is None: raise RuntimeError('No params captured!')
    print(f'  Best RMSE seen: {strategy.best_rmse:.2f}')

    # Load final model
    model = build_model(cfg).to(device)
    sd = model.state_dict()
    ns = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in zip(sd.keys(), use_params)}
    model.load_state_dict(ns, strict=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Evaluate
    print("\n  Final per-client evaluation:")
    row = {"experiment": mode, "seed": seed}; ap, at = [], []
    for fd in cfg["data"]["clients"]:
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        loader = make_loader(npz["X_test"], npz["y_test"], cfg["training"]["batch_size"])
        mp, sp, tr = mc_predict(model, loader, device, cfg["model"]["mc_samples"])
        m = compute_metrics(mp, tr); ap.append(mp); at.append(tr)
        print(f"    {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  PHM={m['phm_score']:.1f}  Unc={sp.mean():.3f}")
        row[f"{fd}_rmse"] = round(m["rmse"], 4); row[f"{fd}_mae"] = round(m["mae"], 4)
        row[f"{fd}_phm"] = round(m["phm_score"], 2); row[f"{fd}_unc"] = round(float(sp.mean()), 4)

    ov = compute_metrics(np.concatenate(ap), np.concatenate(at))
    row["overall_rmse"] = round(ov["rmse"], 4); row["overall_mae"] = round(ov["mae"], 4)
    row["overall_phm"] = round(ov["phm_score"], 2)
    print(f"\n  OVERALL: RMSE={ov['rmse']:.2f}  MAE={ov['mae']:.2f}  PHM={ov['phm_score']:.1f}")

    ckpt = os.path.join(rd, f"{mode}_seed{seed}.pt"); torch.save(model.state_dict(), ckpt)
    if strategy.round_metrics:
        cp = os.path.join(rd, f"{mode}_convergence_seed{seed}.csv")
        with open(cp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=strategy.round_metrics[0].keys()); w.writeheader(); w.writerows(strategy.round_metrics)
    return row, model
print("model configed")

# %%
# from huggingface_hub import HfApi, login
# import glob, traceback

# HF_REPO = "Unmeshraj/federated-pdm-cmapss"
# try:
#     from kaggle_secrets import UserSecretsClient
#     hf_token = UserSecretsClient().get_secret("HF_TOKEN")
# except:
#     hf_token = os.environ.get("HF_TOKEN", None)

# login(token=hf_token)
# api = HfApi()
# api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")

# def safe_upload(fpath, repo=HF_REPO):
#     """Upload with retry — never crashes the main loop."""
#     for attempt in range(3):
#         try:
#             if os.path.exists(fpath):
#                 api.upload_file(path_or_fileobj=fpath, path_in_repo=os.path.basename(fpath), repo_id=repo)
#                 print(f"  ☁️  Uploaded {os.path.basename(fpath)}")
#                 return True
#         except Exception as e:
#             print(f"  ⚠️  Upload attempt {attempt+1}/3 failed for {os.path.basename(fpath)}: {e}")
#     print(f"  ❌ All upload attempts failed for {os.path.basename(fpath)} — file kept locally")
#     return False

# completed = []
# failed = []

# for mode in ["fedavg", "fedprox", "proposed"]:
#     csv_path = os.path.join(cfg["evaluation"]["results_dir"], f"{mode}.csv")
#     for seed in [123,456]
#         print(f"\n{'='*60}")
#         print(f"  Starting: {mode.upper()} | seed={seed}")
#         print(f"{'='*60}")
#         try:
#             row, _ = run_federated(cfg, mode, seed, eval_every=5)
#             log_results(row, csv_path)

#             # Upload checkpoint + csv immediately
#             ckpt = os.path.join(cfg["evaluation"]["results_dir"], f"{mode}_seed{seed}.pt")
#             safe_upload(ckpt)
#             safe_upload(csv_path)

#             completed.append(f"{mode}_seed{seed}")
#             print(f"  ✅ Done: {mode}_seed{seed}")

#         except Exception as e:
#             print(f"  ❌ FAILED: {mode}_seed{seed}")
#             print(traceback.format_exc())
#             failed.append(f"{mode}_seed{seed}")
#             # Still try to upload whatever csv exists
#             safe_upload(csv_path)
#             continue  # don't stop — move to next seed

#     print(f"✅ {mode.upper()} results saved → {csv_path}")

# # Final summary upload — everything in results/
# print("\n📦 Final sweep upload...")
# for fpath in glob.glob(os.path.join(cfg["evaluation"]["results_dir"], "*.pt")) + \
#              glob.glob(os.path.join(cfg["evaluation"]["results_dir"], "*.csv")):
#     safe_upload(fpath)

# print(f"\n{'='*60}")
# print(f"  COMPLETED: {completed}")
# print(f"  FAILED:    {failed}")
# print(f"{'='*60}")
# print("\n✅ All federated experiments complete")

# %%
# # fedprox modes with seed 42
# import ray
# if ray.is_initialized():
#     ray.shutdown()
# ray.init(ignore_reinit_error=True, num_gpus=2, num_cpus=4)

# completed = []
# failed = []

# for mode in ["fedprox"]:
#     csv_path = os.path.join(cfg["evaluation"]["results_dir"], f"{mode}.csv")
#     for seed in [42]:
#         print(f"\n{'='*60}")
#         print(f"  Starting: {mode.upper()} | seed={seed}")
#         print(f"{'='*60}")
#         try:
#             row, _ = run_federated(cfg, mode, seed, eval_every=5)
#             log_results(row, csv_path)

#             ckpt = os.path.join(cfg["evaluation"]["results_dir"], f"{mode}_seed{seed}.pt")
#             safe_upload(ckpt)
#             safe_upload(csv_path)

#             completed.append(f"{mode}_seed{seed}")
#             print(f"  ✅ Done: {mode}_seed{seed}")

#         except Exception as e:
#             print(f"  ❌ FAILED: {mode}_seed{seed}")
#             print(traceback.format_exc())
#             failed.append(f"{mode}_seed{seed}")
#             safe_upload(csv_path)
#             continue  

#     print(f"✅ {mode.upper()} results saved → {csv_path}")

# print("\n📦 Final sweep upload...")
# for fpath in glob.glob(os.path.join(cfg["evaluation"]["results_dir"], "*.pt")) + \
#              glob.glob(os.path.join(cfg["evaluation"]["results_dir"], "*.csv")):
#     safe_upload(fpath)

# print(f"\n{'='*60}")
# print(f"  COMPLETED: {completed}")
# print(f"  FAILED:    {failed}")
# print(f"{'='*60}")
# print("\n✅ All federated experiments complete")

# %%
# %%
if ray.is_initialized(): ray.shutdown()
ray.init(ignore_reinit_error=True, num_gpus=2, num_cpus=4)

completed = []
failed = []

for mode in ['proposed']:
    csv_path = os.path.join(cfg['evaluation']['results_dir'], f'{mode}.csv')
    for seed in [42]:
        print(f"\n{'='*60}")
        print(f'  Starting: {mode.upper()} | seed={seed} | rounds=400 | epochs=2')
        print(f"{'='*60}")
        try:
            row, _ = run_federated(cfg, mode, seed, eval_every=10)
            log_results(row, csv_path)
            ckpt = os.path.join(cfg['evaluation']['results_dir'], f'{mode}_seed{seed}.pt')
            safe_upload(ckpt); safe_upload(csv_path)
            completed.append(f'{mode}_seed{seed}')
            print(f'  ✅ Done: {mode}_seed{seed}  RMSE={row["overall_rmse"]:.2f}')
        except Exception as e:
            print(f'  ❌ FAILED: {mode}_seed{seed}')
            print(traceback.format_exc())
            failed.append(f'{mode}_seed{seed}')
            safe_upload(csv_path)
            continue

print('\n📦 Final sweep upload...')
for fpath in glob.glob(os.path.join(cfg['evaluation']['results_dir'], '*.pt')) + \
             glob.glob(os.path.join(cfg['evaluation']['results_dir'], '*.csv')):
    safe_upload(fpath)

print(f"\n{'='*60}")
print(f'  COMPLETED: {completed}')
print(f'  FAILED:    {failed}')
print(f"{'='*60}")
        try:
            row, _ = run_federated(cfg, mode, seed, eval_every=5)
            log_results(row, csv_path)

            ckpt = os.path.join(cfg["evaluation"]["results_dir"], f"{mode}_seed{seed}.pt")
            safe_upload(ckpt)
            safe_upload(csv_path)

            completed.append(f"{mode}_seed{seed}")
            print(f"  ✅ Done: {mode}_seed{seed}")

        except Exception as e:
            print(f"  ❌ FAILED: {mode}_seed{seed}")
            print(traceback.format_exc())
            failed.append(f"{mode}_seed{seed}")
            safe_upload(csv_path)
            continue  

    print(f"✅ {mode.upper()} results saved → {csv_path}")

print("\n📦 Final sweep upload...")
for fpath in glob.glob(os.path.join(cfg["evaluation"]["results_dir"], "*.pt")) + \
             glob.glob(os.path.join(cfg["evaluation"]["results_dir"], "*.csv")):
    safe_upload(fpath)

print(f"\n{'='*60}")
print(f"  COMPLETED: {completed}")
print(f"  FAILED:    {failed}")
print(f"{'='*60}")
print("\n✅ All federated experiments complete")

# %%
from huggingface_hub import HfApi, login
import glob, traceback, os

HF_REPO = "Unmeshraj/federated-pdm-cmapss"
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except:
    hf_token = os.environ.get("HF_TOKEN", None)

login(token=hf_token)
api = HfApi()
api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")

def safe_upload(fpath, repo=HF_REPO):
    for attempt in range(3):
        try:
            if os.path.exists(fpath):
                api.upload_file(path_or_fileobj=fpath, path_in_repo=os.path.basename(fpath), repo_id=repo)
                print(f"  ☁️  Uploaded {os.path.basename(fpath)}")
                return True
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt+1}/3 failed: {e}")
    return False

# Upload everything that exists
for fpath in glob.glob("./results/*.pt") + glob.glob("./results/*.csv"):
    safe_upload(fpath)

# %% [markdown]
# ## 6 — Ablation Study (E5)

# %%
class _TCNNoAttention(nn.Module):
    def __init__(self, base):
        super().__init__(); self.tcn = base.tcn; self.head = base.head
    def forward(self, x):
        x = x.permute(0, 2, 1); x = self.tcn(x); x = x.mean(dim=-1); return self.head(x).squeeze(-1)
    def forward_with_hi(self, x):
        xp = x.permute(0, 2, 1); feat = self.tcn(xp)
        hi = torch.sigmoid(feat.mean(dim=1)); out = self.head(feat.mean(dim=-1)).squeeze(-1)
        return out, hi

class _AblationClient(PDMClient):
    def __init__(self, use_phys_loss, use_attention, **kw):
        super().__init__(**kw)
        if not use_phys_loss: self.criterion = _MSEOnlyLoss()
        if not use_attention: self.model = _TCNNoAttention(self.model).to(self.device)
    def set_parameters(self, params):
        sd = self.model.state_dict()
        if len(params) != len(sd):
            raise ValueError(f"Param mismatch for {self.fd}: model={len(sd)}, received={len(params)}")
        ns = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in zip(sd.keys(), params)}
        self.model.load_state_dict(ns, strict=True)
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

VARIANTS = {
    "full":          {"use_simulation": True,  "strategy": "similarity_weighted", "use_phys_loss": True,  "use_attention": True,  "description": "Full proposed system"},
    "no_sim":        {"use_simulation": False, "strategy": "similarity_weighted", "use_phys_loss": True,  "use_attention": True,  "description": "− Weibull simulation"},
    "no_sim_weight": {"use_simulation": True,  "strategy": "fedavg",              "use_phys_loss": True,  "use_attention": True,  "description": "− Similarity weighting"},
    "no_phys_loss":  {"use_simulation": True,  "strategy": "similarity_weighted", "use_phys_loss": False, "use_attention": True,  "description": "− Physics loss"},
    "no_attention":  {"use_simulation": True,  "strategy": "similarity_weighted", "use_phys_loss": True,  "use_attention": False, "description": "− Temporal attention"},
}

def run_ablation_variant(cfg, vname, seed):
    v = VARIANTS[vname]; set_seeds(seed); device = DEVICE
    fdk = cfg["data"]["clients"]
    print(f"\n[Ablation | {vname}] seed={seed} → {v['description']}")

    base_init = build_model(cfg)
    init_m = base_init if v["use_attention"] else _TCNNoAttention(base_init)
    # Use state_dict().values() to match get_parameters
    ip = [val.detach().cpu().numpy() for val in init_m.state_dict().values()]
    base_s = build_strategy(v["strategy"], cfg, ip)
    strategy = _TrackingStrategy(base_s, eval_every=999)  # skip mid-round eval in ablation

    def cfn(context):
        partition_id = context.node_config.get("partition-id", int(context.node_id))
        fd = fdk[int(partition_id) % len(fdk)]
        return _AblationClient(fd=fd, cfg=cfg, use_simulation=v["use_simulation"],
                               use_fedprox=False, device=device, use_phys_loss=v["use_phys_loss"],
                               use_attention=v["use_attention"]).to_client()

    fl.simulation.start_simulation(client_fn=cfn, num_clients=len(fdk),
        config=fl.server.ServerConfig(num_rounds=cfg["federation"]["num_rounds"]),
        strategy=strategy, client_resources={"num_cpus": 1, "num_gpus": 0.5})

    if strategy.last_params is None: raise RuntimeError(f"No params for {vname}")

    base = build_model(cfg).to(device)
    model = base if v["use_attention"] else _TCNNoAttention(base).to(device)
    sd = model.state_dict()
    ns = {k: torch.tensor(val, dtype=torch.float32).to(device) for k, val in zip(sd.keys(), strategy.last_params)}
    model.load_state_dict(ns, strict=True)

    res = {"experiment": "ablation", "variant": vname, "description": v["description"], "seed": seed}
    ap, at = [], []
    for fd in fdk:
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        loader = make_loader(npz["X_test"], npz["y_test"], cfg["training"]["batch_size"])
        model.eval(); p, t = [], []
        with torch.no_grad():
            for X, y in loader: p.append(model(X.to(device)).cpu().numpy()); t.append(y.numpy())
        p, t = np.concatenate(p), np.concatenate(t); m = compute_metrics(p, t)
        ap.append(p); at.append(t)
        print(f"  {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}")
        res[f"{fd}_rmse"] = round(m["rmse"], 4); res[f"{fd}_mae"] = round(m["mae"], 4)
        res[f"{fd}_phm"] = round(m["phm_score"], 2)
    ov = compute_metrics(np.concatenate(ap), np.concatenate(at))
    res["overall_rmse"] = round(ov["rmse"], 4); res["overall_mae"] = round(ov["mae"], 4)
    res["overall_phm"] = round(ov["phm_score"], 2)
    print(f"  OVERALL: RMSE={ov['rmse']:.2f}  MAE={ov['mae']:.2f}")
    return res

# %%
# Run ablation study
from huggingface_hub import HfApi, login

HF_REPO = "Unmeshraj/federated-pdm-cmapss"
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except:
    hf_token = os.environ.get("HF_TOKEN", None)

login(token=hf_token)
api = HfApi()
api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")

abl_csv = os.path.join(cfg["evaluation"]["results_dir"], "ablation.csv")
for vname in VARIANTS:
    for seed in [cfg["reproducibility"]["seed"]]:
        row = run_ablation_variant(cfg, vname, seed)
        log_results(row, abl_csv)
        # Upload csv after each variant
        if os.path.exists(abl_csv):
            api.upload_file(path_or_fileobj=abl_csv, path_in_repo="ablation.csv", repo_id=HF_REPO)
            print(f"  ☁️  Uploaded ablation.csv")

print(f"\n✅ Ablation results saved → {abl_csv}")

# %%
import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files

HF_REPO = "Unmeshraj/federated-pdm-cmapss"
RESULTS = "./results"

os.makedirs(RESULTS, exist_ok=True)

files = list_repo_files(repo_id=HF_REPO, repo_type="model")

for f in files:
    if not (f.endswith(".pt") or f.endswith(".csv")):
        continue

    local_name = os.path.basename(f)
    local_path = os.path.join(RESULTS, local_name)

    if os.path.exists(local_path):
        print(f"⏭️ Already exists: {local_name}")
        continue

    try:
        dl = hf_hub_download(
            repo_id=HF_REPO,
            filename=f,
            repo_type="model",
            local_dir="/tmp/hf_dl"
        )

        shutil.copy2(dl, local_path)

        print(f"✅ Downloaded: {local_name}")

    except Exception as e:
        print(f"❌ {f}: {e}")

# %% [markdown]
# ## 7 — All Paper Figures

# %%
RESULTS = cfg['evaluation']['results_dir']
FIG_DIR = os.path.join(RESULTS, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
FD_KEYS = cfg['data']['clients']
PALETTE = {'centralised': '#555555', 'fedavg': '#E74C3C', 'fedprox': '#E67E22', 'proposed': '#2E86C1'}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})

def load_exp(name):
    p = os.path.join(RESULTS, f'{name}.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    print(f'  ⚠️  {name}.csv not found, skipping')
    return None

# Show what we have
print('Available CSVs:')
for f in sorted(os.listdir(RESULTS)):
    if f.endswith('.csv'): print(f'  {f}')
print('\nAvailable checkpoints:')
for f in sorted(os.listdir(RESULTS)):
    if f.endswith('.pt'): print(f'  {f}')

# %%
import os
import matplotlib.pyplot as plt
from huggingface_hub import upload_file

HF_REPO = "Unmeshraj/federated-pdm-cmapss"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name, fig=None, ext="pdf", dpi=300, upload=True):
    if fig is None:
        fig = plt.gcf()

    local_path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")

    fig.savefig(
        local_path,
        bbox_inches="tight",
        dpi=dpi
    )

    print(f"✅ Saved locally: {local_path}")

    if upload:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"outputs/{name}.{ext}",
            repo_id=HF_REPO,
            repo_type="model"
        )

        print(f"🚀 Uploaded to HF: outputs/{name}.{ext}")

# %%
# Fig 3: Real vs Simulated signals
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

npz = np.load(os.path.join(cfg["data"]["output_dir"], "FD001.npz"))
real = npz["X_train"][0]

with open(os.path.join(cfg["data"]["output_dir"], "weibull_params.json")) as f:
    wp = json.load(f)

sim = WeibullSimulator(wp["FD001"]["k"], wp["FD001"]["lambda"], cfg)
Xsim, _ = sim.generate(5)

synth = Xsim[0] if len(Xsim) > 0 else np.zeros_like(real)

for ax, data, title in zip(
    axes,
    [real, synth],
    ["Real CMAPSS Window", "Synthetic Weibull Window"]
):
    for s in range(min(5, data.shape[1])):
        ax.plot(data[:, s], alpha=0.8, lw=1.2)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sensor value")

fig.suptitle(
    "Real vs. Physics-Simulated Sensor Signals",
    fontweight="bold",
    fontsize=13
)

plt.tight_layout()

save_fig("fig3_real_vs_sim")

plt.show()

# %%
exps = {name: load_exp(name) for name in ['centralised', 'fedavg', 'fedprox', 'proposed']}
# Filter out missing
exps = {k: v for k, v in exps.items() if v is not None}
summary = {}
for name, df in exps.items():
    fd_cols = [fd for fd in FD_KEYS if f'{fd}_rmse' in df.columns]
    summary[name] = {fd: (df[f'{fd}_rmse'].mean(), df[f'{fd}_rmse'].std()) for fd in fd_cols}

if summary:
    x = np.arange(len(FD_KEYS)); ne = len(summary); w = 0.18
    offsets = np.linspace(-(ne-1)*w/2, (ne-1)*w/2, ne)
    fig, ax = plt.subplots(figsize=(12, 5))
    for (name, data), off in zip(summary.items(), offsets):
        means = [data.get(fd, (0, 0))[0] for fd in FD_KEYS]
        stds  = [data.get(fd, (0, 0))[1] for fd in FD_KEYS]
        # Replace NaN std (single seed) with 0
        stds = [0 if np.isnan(s) else s for s in stds]
        ax.bar(x+off, means, w, yerr=stds, color=PALETTE.get(name, '#888'),
               label=name.capitalize(), capsize=3, alpha=0.9, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(FD_KEYS, fontsize=12)
    ax.set_ylabel('RMSE (cycles)')
    ax.set_title('RUL Prediction RMSE by Method and Client', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig('fig4_main_results.pdf')
    plt.show()
    print('✅ Fig 4 saved')
else:
    print('No data available for Fig 4')

plt.show()

# %%
abl_df = load_exp('ablation')
if abl_df is not None:
    abl_s = abl_df.groupby('variant').agg(
        rmse_mean=('overall_rmse', 'mean'),
        rmse_std=('overall_rmse', 'std'),
        desc=('description', 'first')
    ).reset_index()
    abl_s['rmse_std'] = abl_s['rmse_std'].fillna(0)
    order = ['full', 'no_sim', 'no_sim_weight', 'no_phys_loss', 'no_attention']
    abl_s['variant'] = pd.Categorical(abl_s['variant'], order)
    abl_s = abl_s.sort_values('variant')
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cs = ['#2E86C1', '#E74C3C', '#E67E22', '#8E44AD', '#1E8449']
    bars = ax.barh(abl_s['desc'], abl_s['rmse_mean'], xerr=abl_s['rmse_std'],
                   color=cs, alpha=0.9, capsize=4, edgecolor='white')
    for b, v in zip(bars, abl_s['rmse_mean']):
        ax.text(v+0.3, b.get_y()+b.get_height()/2, f'{v:.2f}', va='center', fontsize=9)
    ax.set_xlabel('Overall RMSE')
    ax.set_title('Ablation Study', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig('fig5_ablation.pdf')
    plt.show()
    print('✅ Fig 5 saved')
else:
    print('⚠️  Ablation CSV not found — run ablation study first')

# %%
# Try proposed first, fall back to fedprox, then fedavg
ckpt_path = None
ckpt_label = None
for name in ['proposed', 'fedprox', 'fedavg', 'centralised']:
    p = os.path.join(RESULTS, f'{name}_seed42.pt')
    if os.path.exists(p):
        ckpt_path = p
        ckpt_label = name
        break

if ckpt_path:
    print(f'Using checkpoint: {ckpt_label}_seed42.pt')
    model = build_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    
    output_dir = cfg['data']['output_dir']
    fd_plot = [fd for fd in ['FD001', 'FD004'] 
               if os.path.exists(os.path.join(output_dir, f'{fd}.npz'))]
    
    if fd_plot:
        fig, axes = plt.subplots(1, len(fd_plot), figsize=(7*len(fd_plot), 5))
        if len(fd_plot) == 1: axes = [axes]
        print("Hello")
        for ax, fd in zip(axes, fd_plot):
            npz = np.load(os.path.join(output_dir, f'{fd}.npz'))
            loader = make_loader(npz['X_test'], npz['y_test'], cfg['training']['batch_size'])
            mp, sp, tr = mc_predict(model, loader, DEVICE, cfg['model']['mc_samples'])
            idx = np.argsort(tr); t, m, s = tr[idx], mp[idx], sp[idx]
            ax.plot(t, t, 'k--', lw=1.2, alpha=0.5, label='Perfect')
            ax.plot(t, m, color='#2E86C1', lw=1.5, label='Predicted (mean)')
            ax.fill_between(t, m-2*s, m+2*s, alpha=0.25, color='#2E86C1', label='95% CI')
            ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
            ax.set_title(f'{fd}', fontweight='bold'); ax.legend(fontsize=9)
            print("Hello")
        fig.suptitle(f'MC-Dropout Uncertainty-Aware RUL Prediction ({ckpt_label})',
                     fontweight='bold', fontsize=13)
        plt.tight_layout()
        save_fig('fig6_rul_uncertainty.pdf')
        plt.show()
        print('✅ Fig 6 saved')
    else:
        print('⚠️  NPZ files not found locally — need to run preprocessing first')
else:
    print('⚠️  No checkpoint found for Fig 6')

# %%
fig, ax = plt.subplots(figsize=(10, 5))
plotted = False
for name, color in PALETTE.items():
    if name == 'centralised': continue
    p = os.path.join(RESULTS, f'{name}_convergence_seed42.csv')
    if not os.path.exists(p):
        print(f'  ⚠️  No convergence CSV for {name}')
        continue
    conv = pd.read_csv(p)
    # Try rmse column, fall back to first numeric column
    if 'rmse' in conv.columns:
        ycol = 'rmse'
    else:
        numeric_cols = conv.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'round']
        if not numeric_cols: continue
        ycol = numeric_cols[0]
        print(f'  Using column "{ycol}" for {name}')
    xcol = 'round' if 'round' in conv.columns else conv.columns[0]
    ax.plot(conv[xcol], conv[ycol], color=color, lw=2, label=name.capitalize())
    plotted = True

if plotted:
    ax.set_xlabel('Communication Round'); ax.set_ylabel('Global RMSE')
    ax.set_title('Federated Training Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig('fig7_convergence.pdf')
    plt.show()
    print('✅ Fig 7 saved')
else:
    print('⚠️  No convergence data found — skipping Fig 7')
    plt.close()

# %%
proposed_df = load_exp('proposed')
if proposed_df is not None and 'overall_rmse' in proposed_df.columns:
    proposed_rmse = proposed_df['overall_rmse'].values
    print('=== Wilcoxon Signed-Rank Test (proposed vs. baselines) ===')
    for name in ['centralised', 'fedavg', 'fedprox']:
        df = load_exp(name)
        if df is None or 'overall_rmse' not in df.columns: continue
        br = df['overall_rmse'].values
        n = min(len(proposed_rmse), len(br))
        if n < 2:
            print(f'  vs {name:<15} only {n} sample(s) — need ≥2 for Wilcoxon, skipping')
            print(f'             proposed={proposed_rmse[:n].mean():.4f}  baseline={br[:n].mean():.4f}')
            continue
        try:
            stat, p = wilcoxon(proposed_rmse[:n], br[:n])
            sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
            print(f'  vs {name:<15} proposed={proposed_rmse[:n].mean():.4f}  baseline={br[:n].mean():.4f}  p={p:.4f}  {sig}')
        except Exception as e:
            print(f'  vs {name}: {e}')
else:
    print('⚠️  proposed.csv not found or missing overall_rmse — skipping Wilcoxon')

print('\n✅ All figures generated')

# %% [markdown]
# ## 9 — Summary Table

# %%
print('\n' + '='*80)
print('  FINAL RESULTS SUMMARY')
print('='*80)

for exp_name in ['centralised', 'fedavg', 'fedprox', 'proposed']:
    df = load_exp(exp_name)
    if df is None or 'overall_rmse' not in df.columns: continue
    rmse_m = df['overall_rmse'].mean()
    rmse_s = df['overall_rmse'].std()
    mae_m  = df['overall_mae'].mean() if 'overall_mae' in df.columns else float('nan')
    n_seeds = len(df)
    rmse_s_str = f'{rmse_s:.2f}' if not np.isnan(rmse_s) else 'n/a'
    print(f'  {exp_name:<15}  RMSE = {rmse_m:.2f} ± {rmse_s_str}   MAE = {mae_m:.2f}   ({n_seeds} seed(s))')

print()
abl = load_exp('ablation')
if abl is not None:
    print('  ABLATION:')
    for _, row in abl.iterrows():
        print(f"    {row['variant']:<20} RMSE={row['overall_rmse']:.2f}  — {row['description']}")

print('\n' + '='*80)
print('✅ Summary complete')

# %% [markdown]
# ## 8 — Upload Models to Hugging Face Hub

# %%
import glob
from huggingface_hub import HfApi, login

HF_REPO = 'Unmeshraj/federated-pdm-cmapss'
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret('HF_TOKEN')
except:
    hf_token = os.environ.get('HF_TOKEN', None)

login(token=hf_token)
api = HfApi()

for fpath in glob.glob(os.path.join(FIG_DIR, '*.pdf')):
    try:
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f'figures/{os.path.basename(fpath)}',
            repo_id=HF_REPO
        )
        print(f'  ☁️  Uploaded {os.path.basename(fpath)}')
    except Exception as e:
        print(f'  ❌ {os.path.basename(fpath)}: {e}')

print('✅ All figures uploaded')


