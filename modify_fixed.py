# %% [markdown]
# ## 5 — Federated Training (E2–E4)

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
    """
    Novelty 2: weights via exp(-mean_KL).
    Uses fd string from fit_res.metrics['fd'] to look up weights
    (NOT proxy.cid which is a hash in Flower >= 1.x).
    """
    def __init__(self, cfg, initial_params, kl_matrix_path):
        super().__init__(cfg, initial_params)
        kl = pd.read_csv(kl_matrix_path, index_col=0)
        self.fd_keys = cfg["data"]["clients"]
        # weights_dict keyed by fd string (e.g. "FD001") — NOT by index
        mean_kl = {fd: kl.loc[fd, [f for f in self.fd_keys if f != fd]].mean() for fd in self.fd_keys}
        sims = {fd: np.exp(-mean_kl[fd]) for fd in self.fd_keys}
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

        # FIX 2: inter-round cosine LR decay
        round_num    = int(config.get("round", 1))
        total_rounds = int(config.get("num_rounds", self.cfg["federation"].get("num_rounds", 50)))
        cos_factor   = 0.5 * (1.0 + math.cos(math.pi * (round_num - 1) / max(total_rounds - 1, 1)))
        effective_lr = self.lr * (0.1 + 0.9 * cos_factor)

        gp = [p.clone().detach() for p in self.model.parameters()]
        if self.use_simulation:
            Xs, ys = self.simulator.generate(self.cfg["simulation"]["n_trajectories"])
            Xtr = np.concatenate([self.X_train, Xs]); ytr = np.concatenate([self.y_train, ys])
            # FIX 1: round-specific shuffle seed — avoids identical shuffle every round
            idx = np.random.default_rng(self._base_seed + round_num * 7_919).permutation(len(Xtr))
            Xtr, ytr = Xtr[idx], ytr[idx]
        else:
            Xtr, ytr = self.X_train, self.y_train
        loader = make_loader(Xtr, ytr, self.batch_size, shuffle=True)
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
        # Include "fd" + "lr" in metrics
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

# --- Tracking wrapper with eval_every ---
class _TrackingStrategy(fl.server.strategy.Strategy):
    def __init__(self, w, eval_every=5):
        self._w = w; self.eval_every = max(1, eval_every)
        self.round_metrics = []; self.last_params = None
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
        if params is not None: self.last_params = parameters_to_ndarrays(params)
        return params, met

    def aggregate_evaluate(self, server_round=None, results=None, failures=None, **kwargs):
        loss, met = self._w.aggregate_evaluate(server_round, results, failures)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if met:
            rec = dict(met); rec["round"] = server_round; self.round_metrics.append(rec)
            try:
                total_r = self._w.cfg["federation"]["num_rounds"]
            except Exception:
                total_r = "?"
            kv = "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "round")
            print(f"  Round {server_round:>3}/{total_r}  {kv}", flush=True)

        return loss, met

# %%
# --- Main federated run function ---
MODE_CONFIG = {
    "fedavg":   (False, False, "fedavg"),
    "fedprox":  (False, True,  "fedprox"),
    "proposed": (True,  False, "similarity_weighted"),
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
    strategy = _TrackingStrategy(base, eval_every=eval_every)
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
        client_resources={"num_cpus": 1, "num_gpus": 0.5},
    )
    print(f"\n  Simulation finished in {(time.time()-t0)/60:.1f} min")

    if strategy.last_params is None: raise RuntimeError("No params captured!")

    # Load final model
    model = build_model(cfg).to(device)
    sd = model.state_dict()
    ns = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in zip(sd.keys(), strategy.last_params)}
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

# %%
# proposed modes with seed 42

import ray
if ray.is_initialized():
    ray.shutdown()
ray.init(ignore_reinit_error=True, num_gpus=2, num_cpus=4)

completed = []
failed = []

for mode in ["proposed"]:
    csv_path = os.path.join(cfg["evaluation"]["results_dir"], f"{mode}.csv")
    for seed in [42]:
        print(f"\n{'='*60}")
        print(f"  Starting: {mode.upper()} | seed={seed}")
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
