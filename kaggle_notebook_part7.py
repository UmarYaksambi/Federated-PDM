# %% [markdown]
# ## 8 — Upload Models to Hugging Face Hub
#
# Set your HF token as a Kaggle secret named `HF_TOKEN`, or paste it below.
# Change `HF_REPO` to your own repo name.

# %%
from huggingface_hub import HfApi, login

HF_REPO = "YOUR_USERNAME/federated-pdm-cmapss"   # <-- CHANGE THIS

# Login — uses Kaggle secrets or manual token
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    hf_token = os.environ.get("HF_TOKEN", None)

if hf_token:
    login(token=hf_token)
    api = HfApi()

    # Create repo if not exists
    try:
        api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload all checkpoints + config
    results_dir = cfg["evaluation"]["results_dir"]
    files_to_upload = []
    for f in os.listdir(results_dir):
        if f.endswith(".pt") or f.endswith(".csv"):
            files_to_upload.append(os.path.join(results_dir, f))
    files_to_upload.append("config.yaml")

    for fpath in files_to_upload:
        if os.path.exists(fpath):
            fname = os.path.basename(fpath)
            api.upload_file(path_or_fileobj=fpath, path_in_repo=fname, repo_id=HF_REPO)
            print(f"  ✅ Uploaded {fname}")

    # Upload a model card
    model_card = """---
tags:
  - predictive-maintenance
  - federated-learning
  - remaining-useful-life
  - cmapss
  - tcn
datasets:
  - behrad3d/nasa-cmaps
---

# Federated Predictive Maintenance — CMAPSS

Uncertainty-aware TCN models trained via Federated Learning on NASA CMAPSS turbofan dataset.

## Models included
- `centralised_seed*.pt` — Upper-bound centralised baseline (E1)
- `fedavg_seed*.pt` — Standard FedAvg (E2)
- `fedprox_seed*.pt` — FedProx baseline (E3)
- `proposed_seed*.pt` — Full proposed system: Weibull sim + Similarity-weighted aggregation + Physics loss (E4)

## Architecture
- Temporal Convolutional Network (TCN) with multi-head temporal self-attention
- MC-Dropout for uncertainty estimation
- Physics-constrained hybrid loss (MSE + monotonicity)

## How to load
```python
import torch
from models.tcn import build_model
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
model = build_model(cfg)
model.load_state_dict(torch.load("proposed_seed42.pt", map_location="cpu"))
model.eval()
```
"""
    card_path = os.path.join(results_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(model_card)
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md", repo_id=HF_REPO)
    print("  ✅ Uploaded model card")

    print(f"\n✅ All models uploaded to https://huggingface.co/{HF_REPO}")
else:
    print("⚠️  No HF_TOKEN found. Set it as a Kaggle secret or environment variable to upload models.")
    print("   You can still download the checkpoints from the ./results/ directory.")

# %% [markdown]
# ## 9 — Summary Table

# %%
print("\n" + "="*80)
print("  FINAL RESULTS SUMMARY")
print("="*80)

for exp_name in ["centralised", "fedavg", "fedprox", "proposed"]:
    df = load_exp(exp_name)
    if df is None: continue
    rmse_m, rmse_s = df["overall_rmse"].mean(), df["overall_rmse"].std()
    mae_m = df["overall_mae"].mean()
    print(f"  {exp_name:<15}  RMSE = {rmse_m:.2f} ± {rmse_s:.2f}   MAE = {mae_m:.2f}")

print()
abl = load_exp("ablation")
if abl is not None:
    print("  ABLATION:")
    for _, row in abl.iterrows():
        print(f"    {row['variant']:<20} RMSE={row['overall_rmse']:.2f}  — {row['description']}")

print("\n" + "="*80)
print("✅ Notebook complete! All experiments, figures, and uploads done.")
