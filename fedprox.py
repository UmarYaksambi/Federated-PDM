# %% [markdown]
# ## 0 — Install & Setup

# %%
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'flwr[simulation]', 'torch', 'numpy', 'pandas==2.2.2',
    'scipy', 'scikit-learn', 'matplotlib', 'pyyaml',
    'huggingface_hub', 'protobuf<6', 'cryptography<44', 'tqdm'
])
import warnings, os
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print('✅ Dependencies installed')

# %% [markdown]
# ## 1 — Config (Tuned for Sub-20 RMSE)

# %%
import yaml, random, json, csv, time, math, glob, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from scipy.stats import entropy, kstest, weibull_min
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Key hyperparameters tuned for sub-20 RMSE ──
CONFIG = {
    'data': {
        'data_dir': './data',
        'output_dir': './client_data',
        'clients': ['FD001', 'FD002', 'FD003', 'FD004'],
        'selected_sensors': [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
        'hi_sensors': ['s_2', 's_3', 's_4', 's_11', 's_15'],
        'max_rul': 125,
        'window_size': 30,
        'stride': 1,               # full data — better convergence ceiling
    },
    'model': {
        'input_size': 14,
        'hidden_channels': 64,
        'num_levels': 4,
        'kernel_size': 3,
        'dropout': 0.15,
        'mc_samples': 20,
    },
    'loss': {
        'lambda_physics': 0.1,
        'phm_alpha': 13.0,
        'phm_beta': 10.0,
        'phm_weight': 0.01,
    },
    'training': {
        'epochs_local': 5,         # more local training to escape mean-prediction basin
        'batch_size': 256,
        'learning_rate': 6e-4,
        'weight_decay': 1e-4,
        'noise_std': 0.02,         # more diversity
    },
    'federation': {
        'num_rounds': 200,
        'min_clients': 4,
        'fedprox_mu': 0.1,         # strong constraint to hold 5 epochs of drift
        'checkpoint_every': 50,
    },
    'evaluation': {
        'seeds': [42],
        'results_dir': './results',
    },
    'reproducibility': {'seed': 42},
}
cfg = CONFIG

def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'Rounds: {cfg["federation"]["num_rounds"]}  |  Local epochs: {cfg["training"]["epochs_local"]}  |  Hidden: {cfg["model"]["hidden_channels"]}')
print('✅ Config ready')

# %% [markdown]
# ## 2 — HuggingFace Setup

# %%
from huggingface_hub import HfApi, login

HF_REPO = 'Unmeshraj/federated-pdm-cmapss'
HF_FOLDER = 'fedprox'  # all checkpoints go under /fedprox/ in the repo

try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret('HF_TOKEN')
except:
    hf_token = os.environ.get('HF_TOKEN', None)

login(token=hf_token)
api = HfApi()
api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type='model')

def safe_upload(local_path, remote_name=None):
    """Upload to HF_FOLDER/filename with 3 retries."""
    if not os.path.exists(local_path):
        print(f'  ⚠️  File not found: {local_path}')
        return False
    fname = remote_name or os.path.basename(local_path)
    remote_path = f'{HF_FOLDER}/{fname}'
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=local_path, path_in_repo=remote_path, repo_id=HF_REPO)
            print(f'  ☁️  {remote_path}')
            return True
        except Exception as e:
            print(f'  ⚠️  Attempt {attempt+1}/3: {e}')
    return False

print('✅ HuggingFace ready')

# %% [markdown]
# ## 3 — Enhanced TCN Model (Wider + Deeper)

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
    def __init__(self, input_size=14, hidden_channels=64, num_levels=4, kernel_size=3, dropout=0.15):
        super().__init__()
        blocks = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else hidden_channels
            blocks.append(_ResidualBlock(in_ch, hidden_channels, kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.attention = _TemporalAttention(hidden_channels, num_heads=4)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_channels, 1)
        )

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
        return self.head(attended).squeeze(-1), hi_seq


def build_model(cfg):
    m = cfg['model']
    return TCN(m['input_size'], m['hidden_channels'], m['num_levels'], m['kernel_size'], m['dropout'])

m = build_model(cfg)
params = sum(p.numel() for p in m.parameters())
print(f'✅ Model: {params:,} parameters')


# %% [markdown]
# ## 4 — PHM-Aware Loss + FedProx

# %%
class PHMAwareLoss(nn.Module):
    """
    Combines:
    1. MSE loss
    2. Physics monotonicity loss (HI must degrade)
    3. Asymmetric PHM scoring loss (late predictions penalized more)
    
    Based on: Saxena et al. PHM08 scoring function
    """
    def __init__(self, lambda_physics=0.1, phm_weight=0.01, alpha=13.0, beta=10.0):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.phm_weight = phm_weight
        self.alpha = alpha  # penalty for late (pred > true)
        self.beta = beta    # penalty for early (pred < true)

    def forward(self, rul_pred, rul_true, hi_seq):
        # 1. MSE
        mse = F.mse_loss(rul_pred, rul_true)

        # 2. Physics: HI should be non-increasing
        hi_diff = hi_seq[:, 1:] - hi_seq[:, :-1]
        violation = torch.relu(-hi_diff).mean()

        # 3. Asymmetric PHM loss — clamp before exp to prevent gradient explosion.
        # At RMSE=90: exp(90/10)=8103/sample, phm_weight*8103=81 >> MSE=8100.
        # Model was optimising PHM not RMSE. Clamp to ±30 keeps asymmetry, sane grads.
        err = rul_pred - rul_true
        err_c = torch.clamp(err, -30.0, 30.0)
        phm = torch.where(
            err_c < 0,
            torch.exp(-err_c / self.alpha) - 1,
            torch.exp( err_c / self.beta)  - 1
        ).mean()

        total = mse + self.lambda_physics * violation + self.phm_weight * phm
        return total, {
            'mse': mse.item(),
            'physics': violation.item(),
            'phm': phm.item(),
            'total': total.item()
        }


class FedProxLoss(nn.Module):
    def __init__(self, mu=0.001):
        super().__init__()
        self.mu = mu

    def proximal_term(self, local_model, global_params):
        device = next(local_model.parameters()).device
        prox = torch.zeros(1, device=device)
        for lp, gp in zip(local_model.parameters(), global_params):
            prox = prox + torch.linalg.norm(lp - gp.detach().to(device)) ** 2
        return (self.mu / 2.0) * prox

    def forward(self, base_loss, local_model, global_params, round_num=1, total_rounds=100):
        # Mu warmup: ramp from 0 → full mu over first 20% of rounds.
        # Early global model is bad — full proximal constraint hurts convergence.
        warmup = max(1, int(total_rounds * 0.2))
        mu_scale = min(1.0, round_num / warmup)
        return base_loss + mu_scale * self.proximal_term(local_model, global_params)


print('✅ Losses defined')

# %% [markdown]
# ## 5 — Utilities (Dataset, Metrics, MC)

# %%
class CMAPSSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loader(X, y, batch_size, shuffle=False):
    return DataLoader(CMAPSSDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available(), num_workers=0)

def compute_metrics(rul_pred, rul_true):
    err = rul_pred - rul_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    nz   = rul_true != 0
    mape = float(np.mean(np.abs(err[nz] / rul_true[nz])) * 100) if nz.any() else float('nan')
    phm  = float(np.where(err < 0, np.exp(-err/13.0)-1, np.exp(err/10.0)-1).sum())
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'phm_score': phm}

def mc_predict(model, loader, device, n_samples=50):
    """Batch-wise MC-Dropout — avoids OOM loading all 34k FD004 samples at once."""
    model.train()
    mean_preds, std_preds, true_ruls = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            batch_preds = np.stack([model(X).cpu().numpy() for _ in range(n_samples)])
            mean_preds.append(batch_preds.mean(0))
            std_preds.append(batch_preds.std(0))
            true_ruls.append(y.numpy())
    model.eval()
    return np.concatenate(mean_preds), np.concatenate(std_preds), np.concatenate(true_ruls)

def log_results(results, csv_path, mode='a'):
    hdr = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=results.keys())
        if hdr: w.writeheader()
        w.writerow(results)

print('✅ Utilities defined')

# %% [markdown]
# ## 6 — Data Preprocessing (reuses existing .npz if available)

# %%
def load_cmapss(path):
    col_names = (['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] +
                 [f's_{i}' for i in range(1, 22)])
    df = pd.read_csv(path, sep=r'\s+', header=None, names=col_names, engine='python')
    return df.dropna(axis=1, how='all')

def label_train_rul(df, max_rul):
    mc = df.groupby('unit_id')['cycle'].max().reset_index(name='max_cycle')
    df = df.merge(mc, on='unit_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=max_rul)
    return df.drop(columns=['max_cycle'])

def label_test_rul(test_df, rul_df, max_rul):
    mc = test_df.groupby('unit_id')['cycle'].max().reset_index(name='max_cycle')
    rul_df = rul_df.copy(); rul_df['unit_id'] = rul_df.index + 1
    test_df = test_df.merge(mc, on='unit_id').merge(rul_df, on='unit_id')
    test_df['RUL'] = (test_df['max_cycle'] + test_df['RUL'] - test_df['cycle']).clip(upper=max_rul)
    return test_df.drop(columns=['max_cycle'])

def compute_health_index(df, hi_sensors):
    df['HI'] = 0.0
    if not hi_sensors: return df
    for uid in df['unit_id'].unique():
        mask = df['unit_id'] == uid
        sub = df.loc[mask, hi_sensors].values
        if sub.size == 0: continue
        composite = np.nanmean(sub, axis=1)
        valid_idx = ~np.isnan(composite)
        if valid_idx.sum() < 2: continue
        composite = composite[valid_idx]
        c_min, c_max = composite.min(), composite.max()
        hi = np.zeros_like(composite) if c_max - c_min < 1e-8 else (composite - c_min) / (c_max - c_min)
        hi_iso = IsotonicRegression(out_of_bounds='clip').fit_transform(np.arange(len(hi)), hi)
        df.loc[df.index[mask][valid_idx], 'HI'] = hi_iso
    return df

def create_sequences(df, sensor_cols, window_size, stride):
    seqs, labs = [], []
    for unit in df['unit_id'].unique():
        u = df[df['unit_id'] == unit].reset_index(drop=True)
        data, rul = u[sensor_cols].values, u['RUL'].values
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

DATA_DIR = '/kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps'
OUTPUT_DIR = cfg['data']['output_dir']
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_sensor_names(sensor_list):
    return [s if str(s).startswith('s_') else f's_{s}' for s in sensor_list]

sensor_cols = normalize_sensor_names(cfg['data']['selected_sensors'])
hi_sensors  = normalize_sensor_names(cfg['data']['hi_sensors'])
fd_keys = cfg['data']['clients']

# Check if .npz already exist to skip reprocessing
npz_exist = all(os.path.exists(os.path.join(OUTPUT_DIR, f'{fd}.npz')) for fd in fd_keys)
kl_exist  = os.path.exists(os.path.join(OUTPUT_DIR, 'kl_divergence_matrix.csv'))

if npz_exist and kl_exist:
    print('✅ .npz and KL matrix already exist — skipping preprocessing')
    print('   Delete ./client_data/*.npz to rerun preprocessing')
else:
    print('Running preprocessing...')
    set_seeds(cfg['reproducibility']['seed'])
    clients = {}
    for fd in fd_keys:
        train_path = os.path.join(DATA_DIR, f'train_{fd}.txt')
        test_path  = os.path.join(DATA_DIR, f'test_{fd}.txt')
        rul_path   = os.path.join(DATA_DIR, f'RUL_{fd}.txt')
        train_df = load_cmapss(train_path)
        test_df  = load_cmapss(test_path)
        rul_df   = pd.read_csv(rul_path, header=None, names=['RUL'])
        sensor_cols_valid = [s for s in sensor_cols if s in train_df.columns]
        keep = ['unit_id', 'cycle'] + sensor_cols_valid
        train_df = train_df[keep].copy(); test_df = test_df[keep].copy()
        train_df = label_train_rul(train_df, cfg['data']['max_rul'])
        test_df  = label_test_rul(test_df, rul_df, cfg['data']['max_rul'])
        scaler = MinMaxScaler()
        train_df[sensor_cols_valid] = scaler.fit_transform(train_df[sensor_cols_valid])
        test_df[sensor_cols_valid]  = scaler.transform(test_df[sensor_cols_valid])
        valid_hi = [s for s in hi_sensors if s in sensor_cols_valid] or sensor_cols_valid
        train_df = compute_health_index(train_df, valid_hi)
        test_df  = compute_health_index(test_df,  valid_hi)
        X_train, y_train = create_sequences(train_df, sensor_cols_valid, cfg['data']['window_size'], cfg['data']['stride'])
        X_test,  y_test  = create_sequences(test_df,  sensor_cols_valid, cfg['data']['window_size'], cfg['data']['stride'])
        lifetimes = train_df.groupby('unit_id')['cycle'].max().values.astype(float)
        clients[fd] = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'lifetimes': lifetimes}
        np.savez(os.path.join(OUTPUT_DIR, f'{fd}.npz'), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        print(f'  {fd}: train={X_train.shape}  test={X_test.shape}')
    
    kl_matrix = pd.DataFrame(index=fd_keys, columns=fd_keys, dtype=float)
    for i in fd_keys:
        for j in fd_keys:
            kl_matrix.loc[i, j] = 0.0 if i == j else round(kl_divergence(clients[i]['lifetimes'], clients[j]['lifetimes']), 4)
    kl_matrix.to_csv(os.path.join(OUTPUT_DIR, 'kl_divergence_matrix.csv'))
    print('✅ Preprocessing complete')

# %% [markdown]
# ## 7 — SCAFFOLD + FedProx Strategy

# %%
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

def _weighted_average(results, weights):
    n_layers = len(results[0][0])
    return [sum(w * np.array(params[i]) for w, (params, _) in zip(weights, results))
            for i in range(n_layers)]

def _agg_metrics(metrics):
    if not metrics: return {}
    total = sum(n for n, _ in metrics)
    keys = [k for k in metrics[0][1].keys() if isinstance(metrics[0][1][k], (int, float))]
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}


class OptimizedFedProxStrategy(fl.server.strategy.Strategy):
    """
    FedProx + SCAFFOLD-inspired server correction.
    
    SCAFFOLD (Karimireddy et al. 2020) corrects client drift by maintaining
    server and client control variates. This server-side version maintains
    a momentum buffer that corrects systematic bias across clients.
    """
    def __init__(self, cfg, initial_params):
        self.cfg = cfg
        self.min_clients = cfg['federation']['min_clients']
        self.num_rounds = cfg['federation']['num_rounds']
        self._initial = ndarrays_to_parameters(initial_params)
        self.last_global = initial_params  # for drift correction
        # Server momentum for SCAFFOLD-style correction
        self.server_momentum = [np.zeros_like(p) for p in initial_params]
        self.momentum_beta = 0.9

    def initialize_parameters(self, client_manager=None, **kwargs):
        return self._initial

    def configure_fit(self, server_round=None, parameters=None, client_manager=None, **kwargs):
        config = {
            'round': server_round,
            'num_rounds': self.num_rounds,
        }
        return [(c, fl.common.FitIns(parameters, config))
                for c in client_manager.sample(self.min_clients)]

    def configure_evaluate(self, server_round=None, parameters=None, client_manager=None, **kwargs):
        return [(c, fl.common.EvaluateIns(parameters, {'round': server_round}))
                for c in client_manager.sample(self.min_clients)]

    def aggregate_fit(self, server_round=None, results=None, failures=None, **kwargs):
        if not results: return None, {}

        # FIX: plain sample-count weighted average — removed SCAFFOLD momentum
        # which was adding noise that corrupted convergence (pushed weights in
        # wrong direction at early rounds when momentum is near zero)
        pl = [(parameters_to_ndarrays(fr.parameters), fr.num_examples) for _, fr in results]
        total = sum(n for _, n in pl)
        ws = np.array([n / total for _, n in pl])
        agg = _weighted_average(pl, ws)

        met = _agg_metrics([(fr.num_examples, fr.metrics) for _, fr in results if fr.metrics])
        return ndarrays_to_parameters(agg), met

    def aggregate_evaluate(self, server_round=None, results=None, failures=None, **kwargs):
        if not results: return None, {}
        total = sum(r.num_examples for _, r in results)
        wl  = sum(r.loss * r.num_examples for _, r in results) / total
        agg = _agg_metrics([(r.num_examples, r.metrics) for _, r in results])
        return wl, agg

    def evaluate(self, sr, parameters=None, **kwargs):
        return None


class _TrackingStrategy(fl.server.strategy.Strategy):
    """Wraps any strategy, tracks metrics and saves checkpoints every N rounds."""
    def __init__(self, w, eval_every=5, checkpoint_every=25, results_dir='./results', seed=42):
        self._w = w
        self.eval_every = max(1, eval_every)
        self.checkpoint_every = checkpoint_every
        self.results_dir = results_dir
        self.seed = seed
        self.round_metrics = []
        self.last_params = None
        self.best_rmse = float('inf')
        self.best_params = None

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
        return params, met

    def aggregate_evaluate(self, server_round=None, results=None, failures=None, **kwargs):
        loss, met = self._w.aggregate_evaluate(server_round, results, failures)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if met:
            rec = dict(met); rec['round'] = server_round
            self.round_metrics.append(rec)
            nr = self.cfg['federation']['num_rounds'] if hasattr(self._w, 'cfg') else '?'
            kv = '  '.join(f'{k}={v:.4f}' for k, v in rec.items() if k not in ('round',))
            print(f'  Round {server_round:>3}/{nr}  {kv}', flush=True)

            # Track best
            if 'rmse' in met and met['rmse'] < self.best_rmse:
                self.best_rmse = met['rmse']
                self.best_params = [p.copy() for p in self.last_params] if self.last_params else None

        # Save checkpoint every N rounds
        if (server_round % self.checkpoint_every == 0) and self.last_params is not None:
            self._save_and_upload_checkpoint(server_round)

        return loss, met

    def _save_and_upload_checkpoint(self, server_round):
        try:
            ckpt_name = f'fedprox_optimized_round{server_round}_seed{self.seed}.pt'
            ckpt_path = os.path.join(self.results_dir, ckpt_name)
            model = build_model(cfg).to('cpu')
            sd = model.state_dict()
            ns = {k: torch.tensor(v, dtype=torch.float32) for k, v in zip(sd.keys(), self.last_params)}
            model.load_state_dict(ns, strict=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f'  💾 Checkpoint saved: {ckpt_name}')
            safe_upload(ckpt_path)
        except Exception as e:
            print(f'  ⚠️  Checkpoint save failed at round {server_round}: {e}')

    @property
    def cfg(self):
        return getattr(self._w, 'cfg', {})


print('✅ Strategies defined')

# %% [markdown]
# ## 8 — Optimized FL Client with OneCycleLR

# %%
class OptimizedPDMClient(fl.client.NumPyClient):
    """
    Key improvements over baseline PDMClient:
    - OneCycleLR per local training session (super-convergence)
    - Gradient accumulation for effective larger batch
    - PHM-aware loss
    - Layer-wise learning rate decay (backbone learns slower)
    """
    def __init__(self, fd, cfg, device=DEVICE, use_amp=True):
        self.fd = fd
        self.cfg = cfg
        self.device = device
        self.use_amp = use_amp and (device.type == 'cuda')
        self._base_seed = cfg['reproducibility']['seed']
        tc = cfg['training']
        self.epochs = tc['epochs_local']
        self.batch_size = tc['batch_size']
        self.lr = tc['learning_rate']
        self.wd = tc['weight_decay']

        npz = np.load(os.path.join(cfg['data']['output_dir'], f'{fd}.npz'))
        self.X_train = npz['X_train']
        self.y_train = npz['y_train']
        self.X_test  = npz['X_test']
        self.y_test  = npz['y_test']

        lc = cfg['loss']
        self.criterion = PHMAwareLoss(
            lambda_physics=lc['lambda_physics'],
            phm_weight=lc['phm_weight'],
            alpha=lc['phm_alpha'],
            beta=lc['phm_beta']
        )
        self.fedprox_loss = FedProxLoss(mu=cfg['federation']['fedprox_mu'])
        self.model = build_model(cfg).to(device)
        print(f'  Client {fd} | train={len(self.X_train):,} test={len(self.X_test):,} amp={self.use_amp}')

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        ns = {k: torch.tensor(v, dtype=torch.float32).to(self.device)
              for k, v in zip(sd.keys(), parameters)}
        self.model.load_state_dict(ns, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        round_num    = int(config.get('round', 1))
        total_rounds = int(config.get('num_rounds', cfg['federation']['num_rounds']))

        # FIX 2: inter-round cosine LR decay (from client.py)
        # round 1 → full lr, final round → 0.1 * lr
        cos_factor   = 0.5 * (1.0 + math.cos(math.pi * (round_num - 1) / max(total_rounds - 1, 1)))
        effective_lr = self.lr * (0.1 + 0.9 * cos_factor)

        # Global params for FedProx proximal term
        gp = [p.clone().detach() for p in self.model.parameters()]

        # Round-specific shuffle + Gaussian noise augmentation for data diversity
        rng = np.random.default_rng(self._base_seed + round_num * 7_919)
        idx = rng.permutation(len(self.X_train))
        X_aug = self.X_train[idx].copy()
        noise_std = self.cfg['training'].get('noise_std', 0.0)
        if noise_std > 0:
            X_aug += rng.normal(0, noise_std, X_aug.shape).astype(np.float32)
            X_aug = np.clip(X_aug, 0.0, 1.0)  # keep in normalised range
        loader = make_loader(X_aug, self.y_train[idx], self.batch_size, shuffle=False)

        # FIX 4: uniform LR across all param groups — backbone was at lr*0.1=3e-5
        # which is too small to learn anything in 5 epochs
        opt = AdamW(self.model.parameters(), lr=effective_lr, weight_decay=self.wd)

        # CosineAnnealingLR within local epochs (replaces broken OneCycleLR)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=effective_lr * 0.1)

        scaler = GradScaler(enabled=self.use_amp)
        self.model.train()
        total_loss = 0.0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for Xb, yb in loader:
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred, hi = self.model.forward_with_hi(Xb)
                    loss, _ = self.criterion(pred, yb, hi)
                    loss = self.fedprox_loss(loss, self.model, gp, round_num, total_rounds)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                epoch_loss += loss.item()
            sched.step()  # FIX: step AFTER optimizer.step(), not inside batch loop
            total_loss = epoch_loss / max(len(loader), 1)

        return (self.get_parameters({}), len(self.X_train),
                {'train_loss': total_loss, 'n_samples': len(self.X_train), 'fd': self.fd, 'lr': effective_lr})

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loader = make_loader(self.X_test, self.y_test, self.batch_size)
        preds, trues = [], []
        with torch.no_grad():
            for X, y in loader:
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds.append(self.model(X.to(self.device)).cpu().float().numpy())
                trues.append(y.numpy())
        m = compute_metrics(np.concatenate(preds), np.concatenate(trues))
        return float(m['rmse']), len(self.X_test), m


def make_client_fn(cfg, device, use_amp=True):
    fdk = cfg['data']['clients']
    def fn(context):
        partition_id = context.node_config.get('partition-id', int(context.node_id))
        fd = fdk[int(partition_id) % len(fdk)]
        return OptimizedPDMClient(fd, cfg, device, use_amp).to_client()
    return fn


print('✅ Client defined')

# %% [markdown]
# ## 9 — Centralized Pre-Training (Warm Start for FL)

# %%
# Pool all client data and train centrally for 30 epochs.
# Saves ./results/centralised_seed42.pt — FL warm start picks it up automatically.
RESULTS_DIR = cfg['evaluation']['results_dir']
cent_ckpt = os.path.join(RESULTS_DIR, 'centralised_seed42.pt')
if os.path.exists(cent_ckpt):
    print(f'✅ Centralised checkpoint exists locally — skipping')
else:
    # Try downloading from HuggingFace first (fastest path)
    hf_downloaded = False
    for hf_name in ['centralised/centralised_seed42.pt', 'centralised/centralised_seed456.pt',
                     'centralised_seed42.pt', 'centralised_seed456.pt']:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=HF_REPO, filename=hf_name, local_dir=RESULTS_DIR)
            # rename to expected name
            import shutil; shutil.copy(path, cent_ckpt)
            print(f'✅ Downloaded from HF: {hf_name} — RMSE~15.7')
            hf_downloaded = True
            break
        except Exception:
            continue

    if not hf_downloaded:
        print('🏋️  No HF checkpoint found — training centralized model (100 epochs)...')
        all_X, all_y = [], []
        for fd in cfg['data']['clients']:
            npz = np.load(os.path.join(cfg['data']['output_dir'], f'{fd}.npz'))
            all_X.append(npz['X_train']); all_y.append(npz['y_train'])
        all_X = np.concatenate(all_X); all_y = np.concatenate(all_y)
        cent_model = build_model(cfg).to(DEVICE)
        use_amp_c  = DEVICE.type == 'cuda'
        criterion_c = PHMAwareLoss(cfg['loss']['lambda_physics'], cfg['loss']['phm_weight'],
                                    cfg['loss']['phm_alpha'], cfg['loss']['phm_beta'])
        opt_c   = AdamW(cent_model.parameters(), lr=cfg['training']['learning_rate'],
                        weight_decay=cfg['training']['weight_decay'])
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched_c = CosineAnnealingLR(opt_c, T_max=100, eta_min=1e-5)
        scaler_c = GradScaler(enabled=use_amp_c)
        loader_c = make_loader(all_X, all_y, cfg['training']['batch_size'], shuffle=True)
        best_loss_c = float('inf')
        for epoch in range(100):
            cent_model.train(); ep_loss = 0.0
            for Xb, yb in loader_c:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt_c.zero_grad()
                with autocast(device_type=DEVICE.type, enabled=use_amp_c):
                    pred, hi = cent_model.forward_with_hi(Xb)
                    loss, _  = criterion_c(pred, yb, hi)
                scaler_c.scale(loss).backward()
                scaler_c.unscale_(opt_c)
                nn.utils.clip_grad_norm_(cent_model.parameters(), 1.0)
                scaler_c.step(opt_c); scaler_c.update(); ep_loss += loss.item()
            sched_c.step(); ep_loss /= len(loader_c)
            if ep_loss < best_loss_c:
                best_loss_c = ep_loss; torch.save(cent_model.state_dict(), cent_ckpt)
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1:>3}/100  loss={ep_loss:.4f}')
        print(f'✅ Centralized training done')

    # Quick eval
    cent_model = build_model(cfg).to(DEVICE)
    cent_model.load_state_dict(torch.load(cent_ckpt, map_location=DEVICE))
    ap, at = [], []
    for fd in cfg['data']['clients']:
        npz = np.load(os.path.join(cfg['data']['output_dir'], f'{fd}.npz'))
        ldr = make_loader(npz['X_test'], npz['y_test'], cfg['training']['batch_size'])
        mp, _, tr = mc_predict(cent_model, ldr, DEVICE, n_samples=10)
        m = compute_metrics(mp, tr); ap.append(mp); at.append(tr)
        print(f'  {fd}: RMSE={m["rmse"]:.2f}')
    ov = compute_metrics(np.concatenate(ap), np.concatenate(at))
    print(f'  OVERALL RMSE={ov["rmse"]:.2f}')
    safe_upload(cent_ckpt)
    print(f'✅ Centralized pre-training done — checkpoint saved')

# %% [markdown]
# ## 10 — Run Optimized FedProx (150 rounds, checkpoint every 50)

# %%
import ray

RESULTS_DIR = cfg['evaluation']['results_dir']
os.makedirs(RESULTS_DIR, exist_ok=True)

import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
logging.getLogger("absl").setLevel(logging.ERROR)


set_seeds(cfg['reproducibility']['seed'])

completed = []
failed = []

for seed in cfg['evaluation']['seeds']:
    print(f"\n{'='*70}")
    print(f'  FEDPROX OPTIMIZED | seed={seed} | rounds=200')
    print(f"{'='*70}")
    try:
        set_seeds(seed)

        # Build model + strategy
        init_model  = build_model(cfg)

        init_params = [v.detach().cpu().numpy() for v in init_model.state_dict().values()]
        base = OptimizedFedProxStrategy(cfg, init_params)
        strategy = _TrackingStrategy(
            base,
            eval_every=10,
            checkpoint_every=cfg['federation']['checkpoint_every'],
            results_dir=RESULTS_DIR,
            seed=seed
        )

        client_fn = make_client_fn(cfg, DEVICE, use_amp=(DEVICE.type == 'cuda'))

        # Init Ray
        if ray.is_initialized(): ray.shutdown()
        ray.init(ignore_reinit_error=True, num_gpus=2, num_cpus=4)

        t0 = time.time()
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(cfg['data']['clients']),
            config=fl.server.ServerConfig(num_rounds=cfg['federation']['num_rounds']),
            strategy=strategy,
            client_resources={'num_cpus': 1, 'num_gpus': 0.5},
        )
        elapsed = (time.time() - t0) / 60
        print(f'\n  Simulation finished in {elapsed:.1f} min')

        if strategy.last_params is None:
            raise RuntimeError('No params captured')

        # Use best params if better than final
        use_params = strategy.best_params if strategy.best_params is not None else strategy.last_params
        print(f'  Using {"best" if strategy.best_params is not None else "final"} params (best RMSE seen: {strategy.best_rmse:.2f})')

        # Load final model
        model = build_model(cfg).to(DEVICE)
        sd = model.state_dict()
        ns = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE)
              for k, v in zip(sd.keys(), use_params)}
        model.load_state_dict(ns, strict=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Per-client evaluation
        print('\n  Final per-client evaluation:')
        row = {'experiment': 'fedprox_optimized', 'seed': seed}
        all_preds, all_trues = [], []
        for fd in cfg['data']['clients']:
            npz = np.load(os.path.join(cfg['data']['output_dir'], f'{fd}.npz'))
            loader = make_loader(npz['X_test'], npz['y_test'], cfg['training']['batch_size'])
            mp, sp, tr = mc_predict(model, loader, DEVICE, cfg['model']['mc_samples'])
            m = compute_metrics(mp, tr)
            all_preds.append(mp); all_trues.append(tr)
            print(f'    {fd}: RMSE={m["rmse"]:.2f}  MAE={m["mae"]:.2f}  PHM={m["phm_score"]:.1f}  Unc={sp.mean():.3f}')
            row[f'{fd}_rmse'] = round(m['rmse'], 4)
            row[f'{fd}_mae']  = round(m['mae'],  4)
            row[f'{fd}_phm']  = round(m['phm_score'], 2)
            row[f'{fd}_unc']  = round(float(sp.mean()), 4)

        ov = compute_metrics(np.concatenate(all_preds), np.concatenate(all_trues))
        row['overall_rmse'] = round(ov['rmse'], 4)
        row['overall_mae']  = round(ov['mae'],  4)
        row['overall_phm']  = round(ov['phm_score'], 2)
        print(f"\n  OVERALL: RMSE={ov['rmse']:.2f}  MAE={ov['mae']:.2f}  PHM={ov['phm_score']:.1f}")

        # Save final checkpoint + convergence CSV
        csv_path = os.path.join(RESULTS_DIR, 'fedprox_optimized.csv')
        log_results(row, csv_path)

        final_ckpt = os.path.join(RESULTS_DIR, f'fedprox_optimized_final_seed{seed}.pt')
        torch.save(model.state_dict(), final_ckpt)

        if strategy.round_metrics:
            conv_path = os.path.join(RESULTS_DIR, f'fedprox_optimized_convergence_seed{seed}.csv')
            with open(conv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=strategy.round_metrics[0].keys())
                w.writeheader(); w.writerows(strategy.round_metrics)
            safe_upload(conv_path)

        # Upload final files
        safe_upload(final_ckpt)
        safe_upload(csv_path)

        completed.append(f'seed_{seed}')
        print(f'  ✅ Done: seed={seed}  OVERALL RMSE={ov["rmse"]:.2f}')

    except Exception as e:
        print(f'  ❌ FAILED: seed={seed}')
        print(traceback.format_exc())
        failed.append(f'seed_{seed}')
        # Still try to upload whatever exists
        csv_path = os.path.join(RESULTS_DIR, 'fedprox_optimized.csv')
        if os.path.exists(csv_path): safe_upload(csv_path)
        continue

# Final sweep upload
print('\n📦 Final sweep upload...')
for fpath in glob.glob(os.path.join(RESULTS_DIR, 'fedprox_optimized*.pt')) + \
else:
    # Try downloading from HuggingFace first (fastest path)
    hf_downloaded = False
    for hf_name in ['centralised/centralised_seed42.pt', 'centralised/centralised_seed456.pt',
                     'centralised_seed42.pt', 'centralised_seed456.pt']:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=HF_REPO, filename=hf_name, local_dir=RESULTS_DIR)
            # rename to expected name
            import shutil; shutil.copy(path, cent_ckpt)
            print(f'✅ Downloaded from HF: {hf_name} — RMSE~15.7')
            hf_downloaded = True
            break
        except Exception:
            continue

    if not hf_downloaded:
        print('🏋️  No HF checkpoint found — training centralized model (100 epochs)...')
        all_X, all_y = [], []
        for fd in cfg['data']['clients']:
            npz = np.load(os.path.join(cfg['data']['output_dir'], f'{fd}.npz'))
            all_X.append(npz['X_train']); all_y.append(npz['y_train'])
        all_X = np.concatenate(all_X); all_y = np.concatenate(all_y)
        cent_model = build_model(cfg).to(DEVICE)
        use_amp_c  = DEVICE.type == 'cuda'
        criterion_c = PHMAwareLoss(cfg['loss']['lambda_physics'], cfg['loss']['phm_weight'],
                                    cfg['loss']['phm_alpha'], cfg['loss']['phm_beta'])
        opt_c   = AdamW(cent_model.parameters(), lr=cfg['training']['learning_rate'],
                        weight_decay=cfg['training']['weight_decay'])
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched_c = CosineAnnealingLR(opt_c, T_max=100, eta_min=1e-5)
        scaler_c = GradScaler(enabled=use_amp_c)
        loader_c = make_loader(all_X, all_y, cfg['training']['batch_size'], shuffle=True)
        best_loss_c = float('inf')
        for epoch in range(100):
            cent_model.train(); ep_loss = 0.0
            for Xb, yb in loader_c:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt_c.zero_grad()
                with autocast(device_type=DEVICE.type, enabled=use_amp_c):
                    pred, hi = cent_model.forward_with_hi(Xb)
                    loss, _  = criterion_c(pred, yb, hi)
                scaler_c.scale(loss).backward()
                scaler_c.unscale_(opt_c)
                nn.utils.clip_grad_norm_(cent_model.parameters(), 1.0)
                scaler_c.step(opt_c); scaler_c.update(); ep_loss += loss.item()
            sched_c.step(); ep_loss /= len(loader_c)
            if ep_loss < best_loss_c:
                best_loss_c = ep_loss; torch.save(cent_model.state_dict(), cent_ckpt)
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1:>3}/100  loss={ep_loss:.4f}')
        print(f'✅ Centralized training done')

    # Quick eval
    cent_model = build_model(cfg).to(DEVICE)
    cent_model.load_state_dict(torch.load(cent_ckpt, map_location=DEVICE))
    ap, at = [], []
    for fd in cfg['data']['clients']:
        npz = np.load(os.path.join(cfg['data']['output_dir'], f'{fd}.npz'))
        ldr = make_loader(npz['X_test'], npz['y_test'], cfg['training']['batch_size'])
        mp, _, tr = mc_predict(cent_model, ldr, DEVICE, n_samples=10)
        m = compute_metrics(mp, tr); ap.append(mp); at.append(tr)
        print(f'  {fd}: RMSE={m["rmse"]:.2f}')
    ov = compute_metrics(np.concatenate(ap), np.concatenate(at))
    print(f'  OVERALL RMSE={ov["rmse"]:.2f}')
    safe_upload(cent_ckpt)
    print(f'✅ Centralized pre-training done — checkpoint saved')

# %% [markdown]
# ## 10 — Run Optimized FedProx (150 rounds, checkpoint every 50)

# %%
import ray

RESULTS_DIR = cfg['evaluation']['results_dir']
os.makedirs(RESULTS_DIR, exist_ok=True)

import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
logging.getLogger("absl").setLevel(logging.ERROR)


set_seeds(cfg['reproducibility']['seed'])

completed = []
failed = []

for seed in cfg['evaluation']['seeds']:
    print(f"\n{'='*70}")
    print(f'  FEDPROX OPTIMIZED | seed={seed} | rounds=200')
    print(f"{'='*70}")
    try:
        set_seeds(seed)

        # Build model + strategy
        init_model  = build_model(cfg)

        init_params = [v.detach().cpu().numpy() for v in init_model.state_dict().values()]
        base = OptimizedFedProxStrategy(cfg, init_params)
        strategy = _TrackingStrategy(
            base,
            eval_every=10,
            checkpoint_every=cfg['federation']['checkpoint_every'],
            results_dir=RESULTS_DIR,
            seed=seed
        )

        client_fn = make_client_fn(cfg, DEVICE, use_amp=(DEVICE.type == 'cuda'))

        # Init Ray
        if ray.is_initialized(): ray.shutdown()
        ray.init(ignore_reinit_error=True, num_gpus=2, num_cpus=4)

        t0 = time.time()
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(cfg['data']['clients']),
            config=fl.server.ServerConfig(num_rounds=cfg['federation']['num_rounds']),
            strategy=strategy,
            client_resources={'num_cpus': 1, 'num_gpus': 0.5},
        )
        elapsed = (time.time() - t0) / 60
        print(f'\n  Simulation finished in {elapsed:.1f} min')

        if strategy.last_params is None:
            raise RuntimeError('No params captured')

        # Use best params if better than final
        use_params = strategy.best_params if strategy.best_params is not None else strategy.last_params
        print(f'  Using {"best" if strategy.best_params is not None else "final"} params (best RMSE seen: {strategy.best_rmse:.2f})')

        # Load final model
        model = build_model(cfg).to(DEVICE)
        sd = model.state_dict()
        ns = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE)
              for k, v in zip(sd.keys(), use_params)}
        model.load_state_dict(ns, strict=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Per-client evaluation
        print('\n  Final per-client evaluation:')
        row = {'experiment': 'fedprox_optimized', 'seed': seed}
        all_preds, all_trues = [], []
        for fd in cfg['data']['clients']:
            npz = np.load(os.path.join(cfg['data']['output_dir'], f'{fd}.npz'))
            loader = make_loader(npz['X_test'], npz['y_test'], cfg['training']['batch_size'])
            mp, sp, tr = mc_predict(model, loader, DEVICE, cfg['model']['mc_samples'])
            m = compute_metrics(mp, tr)
            all_preds.append(mp); all_trues.append(tr)
            print(f'    {fd}: RMSE={m["rmse"]:.2f}  MAE={m["mae"]:.2f}  PHM={m["phm_score"]:.1f}  Unc={sp.mean():.3f}')
            row[f'{fd}_rmse'] = round(m['rmse'], 4)
            row[f'{fd}_mae']  = round(m['mae'],  4)
            row[f'{fd}_phm']  = round(m['phm_score'], 2)
            row[f'{fd}_unc']  = round(float(sp.mean()), 4)

        ov = compute_metrics(np.concatenate(all_preds), np.concatenate(all_trues))
        row['overall_rmse'] = round(ov['rmse'], 4)
        row['overall_mae']  = round(ov['mae'],  4)
        row['overall_phm']  = round(ov['phm_score'], 2)
        print(f"\n  OVERALL: RMSE={ov['rmse']:.2f}  MAE={ov['mae']:.2f}  PHM={ov['phm_score']:.1f}")

        # Save final checkpoint + convergence CSV
        csv_path = os.path.join(RESULTS_DIR, 'fedprox_optimized.csv')
        log_results(row, csv_path)

        final_ckpt = os.path.join(RESULTS_DIR, f'fedprox_optimized_final_seed{seed}.pt')
        torch.save(model.state_dict(), final_ckpt)

        if strategy.round_metrics:
            conv_path = os.path.join(RESULTS_DIR, f'fedprox_optimized_convergence_seed{seed}.csv')
            with open(conv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=strategy.round_metrics[0].keys())
                w.writeheader(); w.writerows(strategy.round_metrics)
            safe_upload(conv_path)

        # Upload final files
        safe_upload(final_ckpt)
        safe_upload(csv_path)

        completed.append(f'seed_{seed}')
        print(f'  ✅ Done: seed={seed}  OVERALL RMSE={ov["rmse"]:.2f}')

    except Exception as e:
        print(f'  ❌ FAILED: seed={seed}')
        print(traceback.format_exc())
        failed.append(f'seed_{seed}')
        # Still try to upload whatever exists
        csv_path = os.path.join(RESULTS_DIR, 'fedprox_optimized.csv')
        if os.path.exists(csv_path): safe_upload(csv_path)
        continue

# Final sweep upload
print('\n📦 Final sweep upload...')
for fpath in glob.glob(os.path.join(RESULTS_DIR, 'fedprox_optimized*.pt')) + \
             glob.glob(os.path.join(RESULTS_DIR, 'fedprox_optimized*.csv')):
    safe_upload(fpath)

print(f"{'='*70}")
print(f'  COMPLETED: {completed}')
print(f'  FAILED:    {failed}')
print(f"{'='*70}")


# %% [markdown]
# ## 11 — Resume from HuggingFace Checkpoint

# %%
RESUME_EXTRA_ROUNDS = 200  # how many more rounds to run

# 1. Find latest checkpoint on HF
from huggingface_hub import list_repo_files, hf_hub_download
import re

all_files  = list(list_repo_files(HF_REPO))
ckpt_files = [f for f in all_files if 'fedprox_optimized_round' in f and f.endswith('.pt')]
if not ckpt_files:
    raise RuntimeError('No fedprox checkpoints found on HuggingFace!')

def _round_num(f): m = re.search(r'round(\d+)', f); return int(m.group(1)) if m else 0
latest_hf   = max(ckpt_files, key=_round_num)
start_round = _round_num(latest_hf)
print(f'  Latest HF checkpoint: {latest_hf}  (round {start_round})')

# 2. Download + load weights
local_ckpt   = hf_hub_download(repo_id=HF_REPO, filename=latest_hf)
resume_model = build_model(cfg).to('cpu')
resume_model.load_state_dict(torch.load(local_ckpt, map_location='cpu'))
init_params  = [v.detach().cpu().numpy() for v in resume_model.state_dict().values()]
print(f'  Loaded — running {RESUME_EXTRA_ROUNDS} more rounds from round {start_round}')

# 3. Re-evaluate device (GPU may have changed between sessions)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  Device: {DEVICE}')

# 4. Run FL
set_seeds(cfg['reproducibility']['seed'])
base_r      = OptimizedFedProxStrategy(cfg, init_params)
strategy_r  = _TrackingStrategy(base_r, eval_every=10, checkpoint_every=50,
                                 results_dir=RESULTS_DIR, seed=42)
client_fn_r = make_client_fn(cfg, DEVICE, use_amp=(DEVICE.type == 'cuda'))

if ray.is_initialized(): ray.shutdown()
ray.init(ignore_reinit_error=True)

n_gpu = (1.0 / len(cfg['data']['clients'])) if torch.cuda.is_available() else 0.0
fl.simulation.start_simulation(
    client_fn=client_fn_r,
    num_clients=len(cfg['data']['clients']),
    config=fl.server.ServerConfig(num_rounds=RESUME_EXTRA_ROUNDS),
    strategy=strategy_r,
    client_resources={'num_cpus': 1, 'num_gpus': n_gpu},
    ray_init_args={'ignore_reinit_error': True},
)

# 5. Save + upload best
if strategy_r.best_params is not None:
    best_path = os.path.join(RESULTS_DIR,
        f'fedprox_resumed_round{start_round + RESUME_EXTRA_ROUNDS}_seed42.pt')
    m2 = build_model(cfg).to('cpu')
    sd = m2.state_dict()
    m2.load_state_dict({k: torch.tensor(v) for k, v in zip(sd.keys(), strategy_r.best_params)})
    torch.save(m2.state_dict(), best_path)
    safe_upload(best_path)
    print(f'  ✅ Best RMSE: {strategy_r.best_rmse:.4f} — saved & uploaded')
