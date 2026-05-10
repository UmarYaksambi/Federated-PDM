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
        ns = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in zip(sd.keys(), params)}
        self.model.load_state_dict(ns, strict=True)
    def get_parameters(self, config): return [p.cpu().numpy() for p in self.model.parameters()]

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
    ip = [p.detach().numpy() for p in init_m.parameters()]
    base_s = build_strategy(v["strategy"], cfg, ip)
    strategy = _TrackingStrategy(base_s)

    def cfn(cid):
        return _AblationClient(fd=fdk[int(cid)], cfg=cfg, use_simulation=v["use_simulation"],
                               use_fedprox=False, device=device, use_phys_loss=v["use_phys_loss"],
                               use_attention=v["use_attention"])

    fl.simulation.start_simulation(client_fn=cfn, num_clients=len(fdk),
        config=fl.server.ServerConfig(num_rounds=cfg["federation"]["num_rounds"]),
        strategy=strategy, client_resources={"num_cpus": 1, "num_gpus": 0.0})

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
        print(f"  {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}"); 
        res[f"{fd}_rmse"] = round(m["rmse"], 4); res[f"{fd}_mae"] = round(m["mae"], 4)
        res[f"{fd}_phm"] = round(m["phm_score"], 2)
    ov = compute_metrics(np.concatenate(ap), np.concatenate(at))
    res["overall_rmse"] = round(ov["rmse"], 4); res["overall_mae"] = round(ov["mae"], 4)
    res["overall_phm"] = round(ov["phm_score"], 2)
    print(f"  OVERALL: RMSE={ov['rmse']:.2f}  MAE={ov['mae']:.2f}")
    return res

# %%
# Run ablation study
abl_csv = os.path.join(cfg["evaluation"]["results_dir"], "ablation.csv")
for vname in VARIANTS:
    for seed in [cfg["reproducibility"]["seed"]]:   # single seed for ablation to save time
        row = run_ablation_variant(cfg, vname, seed)
        log_results(row, abl_csv)
print(f"\n✅ Ablation results saved → {abl_csv}")

# %% [markdown]
# ## 7 — All Paper Figures

# %%
RESULTS = cfg["evaluation"]["results_dir"]
FIG_DIR = os.path.join(RESULTS, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
FD_KEYS = cfg["data"]["clients"]
PALETTE = {"centralised": "#555555", "fedavg": "#E74C3C", "fedprox": "#E67E22", "proposed": "#2E86C1"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

def load_exp(name):
    p = os.path.join(RESULTS, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

# %%
# Fig 3: Real vs Simulated signals
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
npz = np.load(os.path.join(cfg["data"]["output_dir"], "FD001.npz"))
real = npz["X_train"][0]  # (30, 14) — first real window
with open(os.path.join(cfg["data"]["output_dir"], "weibull_params.json")) as f: wp = json.load(f)
sim = WeibullSimulator(wp["FD001"]["k"], wp["FD001"]["lambda"], cfg)
Xsim, _ = sim.generate(5)
synth = Xsim[0] if len(Xsim) > 0 else np.zeros_like(real)

for i, (ax, data, title) in enumerate(zip(axes, [real, synth], ["Real CMAPSS Window", "Synthetic Weibull Window"])):
    for s in range(min(5, data.shape[1])):
        ax.plot(data[:, s], alpha=0.8, lw=1.2)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Timestep"); ax.set_ylabel("Sensor value")
fig.suptitle("Real vs. Physics-Simulated Sensor Signals", fontweight="bold", fontsize=13)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig3_real_vs_sim.pdf"), bbox_inches="tight", dpi=300)
plt.show()

# %%
# Fig 4: Main results comparison
exps = {name: load_exp(name) for name in ["centralised", "fedavg", "fedprox", "proposed"]}
summary = {}
for name, df in exps.items():
    if df is None: continue
    summary[name] = {fd: (df[f"{fd}_rmse"].mean(), df[f"{fd}_rmse"].std()) for fd in FD_KEYS}

if summary:
    x = np.arange(len(FD_KEYS)); ne = len(summary); w = 0.18
    offsets = np.linspace(-(ne-1)*w/2, (ne-1)*w/2, ne)
    fig, ax = plt.subplots(figsize=(12, 5))
    for (name, data), off in zip(summary.items(), offsets):
        means = [data[fd][0] for fd in FD_KEYS]; stds = [data[fd][1] for fd in FD_KEYS]
        ax.bar(x+off, means, w, yerr=stds, color=PALETTE.get(name,"#888"), label=name.capitalize(), capsize=3, alpha=0.9, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(FD_KEYS, fontsize=12)
    ax.set_ylabel("RMSE (cycles)"); ax.set_title("RUL Prediction RMSE by Method and Client", fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig4_main_results.pdf"), bbox_inches="tight", dpi=300)
    plt.show()

# %%
# Fig 5: Ablation study
abl_df = load_exp("ablation")
if abl_df is not None:
    abl_s = abl_df.groupby("variant").agg(rmse_mean=("overall_rmse","mean"), rmse_std=("overall_rmse","std"), desc=("description","first")).reset_index()
    order = ["full","no_sim","no_sim_weight","no_phys_loss","no_attention"]
    abl_s["variant"] = pd.Categorical(abl_s["variant"], order); abl_s = abl_s.sort_values("variant")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cs = ["#2E86C1","#E74C3C","#E67E22","#8E44AD","#1E8449"]
    bars = ax.barh(abl_s["desc"], abl_s["rmse_mean"], xerr=abl_s["rmse_std"], color=cs, alpha=0.9, capsize=4, edgecolor="white")
    for b, v in zip(bars, abl_s["rmse_mean"]): ax.text(v+0.3, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Overall RMSE"); ax.set_title("Ablation Study", fontsize=12, fontweight="bold"); ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig5_ablation.pdf"), bbox_inches="tight", dpi=300); plt.show()

# %%
# Fig 6: RUL prediction + uncertainty (using proposed model)
ckpt_path = os.path.join(RESULTS, "proposed_seed42.pt")
if os.path.exists(ckpt_path):
    model = build_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, fd in zip(axes, ["FD001", "FD004"]):
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        loader = make_loader(npz["X_test"], npz["y_test"], cfg["training"]["batch_size"])
        mp, sp, tr = mc_predict(model, loader, DEVICE, cfg["model"]["mc_samples"])
        idx = np.argsort(tr); t, m, s = tr[idx], mp[idx], sp[idx]
        ax.plot(t, t, "k--", lw=1.2, alpha=0.5, label="Perfect"); ax.plot(t, m, color="#2E86C1", lw=1.5, label="Predicted (mean)")
        ax.fill_between(t, m-2*s, m+2*s, alpha=0.25, color="#2E86C1", label="95% CI")
        ax.set_xlabel("True RUL"); ax.set_ylabel("Predicted RUL"); ax.set_title(f"{fd}", fontweight="bold"); ax.legend(fontsize=9)
    fig.suptitle("MC-Dropout Uncertainty-Aware RUL Prediction", fontweight="bold", fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig6_rul_uncertainty.pdf"), bbox_inches="tight", dpi=300); plt.show()
else:
    print("No proposed checkpoint found, skipping Fig 6")

# %%
# Fig 7: Convergence curves
fig, ax = plt.subplots(figsize=(10, 5))
for name, color in PALETTE.items():
    if name == "centralised": continue
    p = os.path.join(RESULTS, f"{name}_convergence_seed42.csv")
    if not os.path.exists(p): continue
    conv = pd.read_csv(p)
    if "rmse" in conv.columns: ax.plot(conv["round"], conv["rmse"], color=color, lw=2, label=name.capitalize())
ax.set_xlabel("Communication Round"); ax.set_ylabel("Global RMSE")
ax.set_title("Federated Training Convergence", fontsize=13, fontweight="bold"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig7_convergence.pdf"), bbox_inches="tight", dpi=300); plt.show()

# %%
# Fig 8: Statistical significance — Wilcoxon
from scipy.stats import wilcoxon
proposed_df = load_exp("proposed")
if proposed_df is not None:
    print("=== Wilcoxon Signed-Rank Test (proposed vs. baselines) ===")
    proposed_rmse = proposed_df["overall_rmse"].values
    for name in ["centralised", "fedavg", "fedprox"]:
        df = load_exp(name)
        if df is None: continue
        br = df["overall_rmse"].values; n = min(len(proposed_rmse), len(br))
        try:
            stat, p = wilcoxon(proposed_rmse[:n], br[:n])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  vs {name:<15} proposed={proposed_rmse[:n].mean():.4f}  baseline={br[:n].mean():.4f}  p={p:.4f}  {sig}")
        except Exception as e: print(f"  vs {name}: {e}")

print("\n✅ All figures generated")
