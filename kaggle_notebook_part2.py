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
