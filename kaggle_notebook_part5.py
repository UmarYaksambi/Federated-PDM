# %% [markdown]
# ### 5.1 — FL Client & Tracking + Run Function

# %%
# --- FL Client ---
class PDMClient(fl.client.NumPyClient):
    def __init__(self, fd, cfg, use_simulation=True, use_fedprox=False, device=DEVICE):
        self.fd, self.cfg, self.use_simulation, self.use_fedprox, self.device = fd, cfg, use_simulation, use_fedprox, device
        tc = cfg["training"]
        self.epochs, self.batch_size, self.lr, self.wd = tc["epochs_local"], tc["batch_size"], tc["learning_rate"], tc["weight_decay"]
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        self.X_train, self.y_train, self.X_test, self.y_test = npz["X_train"], npz["y_train"], npz["X_test"], npz["y_test"]
        if use_simulation:
            with open(os.path.join(cfg["data"]["output_dir"], "weibull_params.json")) as f: params = json.load(f)
            self.simulator = WeibullSimulator(k=params[fd]["k"], lam=params[fd]["lambda"], cfg=cfg, seed=cfg["reproducibility"]["seed"])
        self.criterion = HybridRULLoss(cfg["loss"]["lambda_physics"])
        self.fedprox_loss = FedProxLoss(mu=cfg["federation"]["fedprox_mu"])
        self.model = build_model(cfg).to(device)
        print(f"  Client {fd} | train={len(self.X_train):,} test={len(self.X_test):,} sim={'on' if use_simulation else 'off'}")

    def get_parameters(self, config): return [p.cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        ns = {k: torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in zip(sd.keys(), parameters)}
        self.model.load_state_dict(ns, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        gp = [p.clone().detach() for p in self.model.parameters()]
        if self.use_simulation:
            Xs, ys = self.simulator.generate(self.cfg["simulation"]["n_trajectories"])
            Xtr = np.concatenate([self.X_train, Xs]); ytr = np.concatenate([self.y_train, ys])
            idx = np.random.default_rng(self.cfg["reproducibility"]["seed"]).permutation(len(Xtr))
            Xtr, ytr = Xtr[idx], ytr[idx]
        else:
            Xtr, ytr = self.X_train, self.y_train
        loader = make_loader(Xtr, ytr, self.batch_size, shuffle=True)
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler(enabled=USE_AMP)
        self.model.train(); tl = 0.0
        for _ in range(self.epochs):
            el = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=USE_AMP):
                    pred, hi = self.model.forward_with_hi(Xb)
                    loss, _ = self.criterion(pred, yb, hi)
                    if self.use_fedprox: loss = self.fedprox_loss(loss, self.model, gp)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); el += loss.item()
            sched.step(); tl = el / max(len(loader), 1)
        return self.get_parameters({}), len(Xtr), {"train_loss": tl}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters); self.model.eval()
        loader = make_loader(self.X_test, self.y_test, self.batch_size)
        p, t = [], []
        with torch.no_grad():
            for X, y in loader: p.append(self.model(X.to(self.device)).cpu().numpy()); t.append(y.numpy())
        m = compute_metrics(np.concatenate(p), np.concatenate(t))
        return float(m["rmse"]), len(self.X_test), m


def make_client_fn(cfg, use_sim, use_fp, device):
    fdk = cfg["data"]["clients"]
    def fn(cid): return PDMClient(fdk[int(cid)], cfg, use_sim, use_fp, device)
    return fn

# --- Tracking wrapper ---
class _TrackingStrategy(fl.server.strategy.Strategy):
    def __init__(self, w):
        self._w = w; self.round_metrics = []; self.last_params = None
    def initialize_parameters(self, cm): return self._w.initialize_parameters(cm)
    def configure_fit(self, sr, p, cm): return self._w.configure_fit(sr, p, cm)
    def configure_evaluate(self, sr, p, cm): return self._w.configure_evaluate(sr, p, cm)
    def evaluate(self, sr, p): return self._w.evaluate(sr, p)
    def aggregate_fit(self, sr, res, fail):
        params, met = self._w.aggregate_fit(sr, res, fail)
        if params is not None: self.last_params = parameters_to_ndarrays(params)
        return params, met
    def aggregate_evaluate(self, sr, res, fail):
        loss, met = self._w.aggregate_evaluate(sr, res, fail)
        if met:
            rec = dict(met); rec["round"] = sr; self.round_metrics.append(rec)
            if sr % 10 == 0 or sr == 1:
                kv = "  ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "round")
                print(f"  Round {sr:>3}  {kv}")
        return loss, met

# %%
# --- Main federated run function ---
MODE_CONFIG = {
    "fedavg":   (False, False, "fedavg"),
    "fedprox":  (False, True,  "fedprox"),
    "proposed": (True,  False, "similarity_weighted"),
}

def run_federated(cfg, mode, seed):
    set_seeds(seed)
    device = DEVICE
    use_sim, use_fp, sname = MODE_CONFIG[mode]
    nc = len(cfg["data"]["clients"]); nr = cfg["federation"]["num_rounds"]
    rd = cfg["evaluation"]["results_dir"]; os.makedirs(rd, exist_ok=True)
    print(f"\n[Federated | {mode.upper()}] seed={seed} rounds={nr} device={device}")

    init_model = build_model(cfg)
    init_params = [p.detach().numpy() for p in init_model.parameters()]
    base = build_strategy(sname, cfg, init_params)
    strategy = _TrackingStrategy(base)
    client_fn = make_client_fn(cfg, use_sim, use_fp, device)

    gpu_frac = (1.0 / nc) if torch.cuda.is_available() else 0.0
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=nc,
        config=fl.server.ServerConfig(num_rounds=nr),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": gpu_frac},
    )

    if strategy.last_params is None: raise RuntimeError("No params captured!")

    # Load final model
    model = build_model(cfg).to(device)
    sd = model.state_dict()
    ns = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in zip(sd.keys(), strategy.last_params)}
    model.load_state_dict(ns, strict=True)

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
# Run all 3 federated experiments
for mode in ["fedavg", "fedprox", "proposed"]:
    csv_path = os.path.join(cfg["evaluation"]["results_dir"], f"{mode}.csv")
    for seed in cfg["evaluation"]["seeds"]:
        row, _ = run_federated(cfg, mode, seed)
        log_results(row, csv_path)
    print(f"✅ {mode.upper()} results saved → {csv_path}")

print("\n✅ All federated experiments complete")
