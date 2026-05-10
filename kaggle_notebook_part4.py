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
for seed in cfg["evaluation"]["seeds"]:
    row, centralised_model = train_centralised(cfg, seed, epochs_central)
    log_results(row, csv_central)
print(f"\n✅ Centralised results saved → {csv_central}")

# %% [markdown]
# ## 5 — Federated Training (E2–E4)

# %%
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# --- Server strategies ---
def _weighted_average(results, weights):
    n_layers = len(results[0][0])
    return [sum(w * np.array(params[i]) for w, (params, _) in zip(weights, results)) for i in range(n_layers)]

def _agg_metrics(metrics):
    if not metrics: return {}
    total = sum(n for n, _ in metrics)
    keys = metrics[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}

def build_fedavg_strategy(cfg):
    return FedAvg(min_fit_clients=cfg["federation"]["min_clients"],
                  min_evaluate_clients=cfg["federation"]["min_clients"],
                  min_available_clients=cfg["federation"]["min_clients"])

class _BaseCustomStrategy(fl.server.strategy.Strategy):
    def __init__(self, cfg, initial_params):
        self.cfg = cfg; self.min_clients = cfg["federation"]["min_clients"]
        self._initial = ndarrays_to_parameters(initial_params)
    def initialize_parameters(self, cm): return self._initial
    def configure_fit(self, sr, p, cm):
        return [(c, fl.common.FitIns(p, {"round": sr})) for c in cm.sample(self.min_clients)]
    def configure_evaluate(self, sr, p, cm):
        return [(c, fl.common.EvaluateIns(p, {"round": sr})) for c in cm.sample(self.min_clients)]
    def aggregate_evaluate(self, sr, results, failures):
        if not results: return None, {}
        total = sum(r.num_examples for _, r in results)
        wl = sum(r.loss * r.num_examples for _, r in results) / total
        agg = _agg_metrics([(r.num_examples, r.metrics) for _, r in results])
        return wl, agg
    def evaluate(self, sr, p): return None
    def aggregate_fit(self, sr, results, failures): raise NotImplementedError

class FedProxStrategy(_BaseCustomStrategy):
    def aggregate_fit(self, sr, results, failures):
        if not results: return None, {}
        pl = [(parameters_to_ndarrays(fr.parameters), fr.num_examples) for _, fr in results]
        total = sum(n for _, n in pl)
        ws = np.array([n/total for _, n in pl])
        agg = _weighted_average(pl, ws)
        return ndarrays_to_parameters(agg), _agg_metrics([(n, fr.metrics) for _, fr in results if fr.metrics])

class SimilarityWeightedStrategy(_BaseCustomStrategy):
    def __init__(self, cfg, initial_params, kl_matrix_path):
        super().__init__(cfg, initial_params)
        kl = pd.read_csv(kl_matrix_path, index_col=0)
        self.fd_keys = cfg["data"]["clients"]
        mean_kl = np.array([kl.loc[fd, [f for f in self.fd_keys if f != fd]].mean() for fd in self.fd_keys])
        sims = np.exp(-mean_kl); self.weights = sims / sims.sum()
        print("\n[SimilarityWeighted] Aggregation weights:")
        for fd, w, k in zip(self.fd_keys, self.weights, mean_kl):
            print(f"  {fd}: weight={w:.4f}  (mean_KL={k:.4f})")

    def aggregate_fit(self, sr, results, failures):
        if not results: return None, {}
        cd = [(parameters_to_ndarrays(fr.parameters), fr.num_examples, self.weights[int(p.cid)]) for p, fr in results]
        rw = np.array([w for _, _, w in cd]); nw = rw / rw.sum()
        nl = len(cd[0][0])
        agg = [sum(nw_i * np.array(params[i]) for (params, _, _), nw_i in zip(cd, nw)) for i in range(nl)]
        return ndarrays_to_parameters(agg), _agg_metrics([(n, fr.metrics) for _, fr in results if fr.metrics])

def build_strategy(name, cfg, init_params):
    if name == "fedavg": return build_fedavg_strategy(cfg)
    elif name == "fedprox": return FedProxStrategy(cfg, init_params)
    elif name == "similarity_weighted":
        return SimilarityWeightedStrategy(cfg, init_params, os.path.join(cfg["data"]["output_dir"], "kl_divergence_matrix.csv"))
    else: raise ValueError(f"Unknown strategy: {name}")

print("✅ FL strategies defined")
