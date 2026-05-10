# %% [markdown]
# ## 3 — Preprocessing

# %%
def load_cmapss(path):
    cols = ["unit_id", "cycle"] + [f"op_{i}" for i in range(1,4)] + [f"s_{i}" for i in range(1,22)]
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)

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
    df = df.copy(); df["HI"] = np.nan
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for unit in df["unit_id"].unique():
        mask = df["unit_id"] == unit
        composite = df.loc[mask, hi_sensors].values.mean(axis=1)
        c_min, c_max = composite.min(), composite.max()
        hi = np.zeros(len(composite)) if c_max - c_min < 1e-8 else (composite - c_min) / (c_max - c_min)
        df.loc[mask, "HI"] = iso.fit_transform(np.arange(len(hi)), hi)
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

# %%
# Run preprocessing
set_seeds(cfg["reproducibility"]["seed"])

data_dir = cfg["data"]["data_dir"]
output_dir = cfg["data"]["output_dir"]
sensor_cols = [f"s_{i}" for i in cfg["data"]["selected_sensors"]]
hi_sensors = cfg["data"]["hi_sensors"]
max_rul = cfg["data"]["max_rul"]
window_size = cfg["data"]["window_size"]
stride = cfg["data"]["stride"]
fd_keys = cfg["data"]["clients"]

fig_dir = os.path.join(output_dir, "figures")
os.makedirs(output_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)

clients = {}
print("\n[1/5] Loading and preprocessing all 4 clients...")
for fd in fd_keys:
    train_df = load_cmapss(os.path.join(data_dir, f"train_{fd}.txt"))
    test_df = load_cmapss(os.path.join(data_dir, f"test_{fd}.txt"))
    rul_df = pd.read_csv(os.path.join(data_dir, f"RUL_{fd}.txt"), header=None, names=["RUL"])
    keep = ["unit_id", "cycle"] + sensor_cols
    train_df, test_df = train_df[keep].copy(), test_df[keep].copy()
    train_df = label_train_rul(train_df, max_rul)
    test_df = label_test_rul(test_df, rul_df, max_rul)
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    train_df = compute_health_index(train_df, hi_sensors)
    test_df = compute_health_index(test_df, hi_sensors)
    X_train, y_train = create_sequences(train_df, sensor_cols, window_size, stride)
    X_test, y_test = create_sequences(test_df, sensor_cols, window_size, stride)
    lifetimes = train_df.groupby("unit_id")["cycle"].max().values.astype(float)
    clients[fd] = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
                   "train_df": train_df, "test_df": test_df, "lifetimes": lifetimes}
    print(f"  {fd}: X_train={X_train.shape}  X_test={X_test.shape}")

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
