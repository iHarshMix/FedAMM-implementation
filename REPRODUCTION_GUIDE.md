# FedAMM — Local Reproduction Guide (WSL / Linux)

> **Branch:** `localWSL` only — never touch `main`.  
> **Paper:** *FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities* (MICCAI 2024 Workshop)  
> **Base model:** RFNet (ICCV 2021) — the backbone is re-used almost verbatim.

---

## 1. What This Paper Actually Does (Mental Model)

| Concept | Plain English |
|---|---|
| **Setting** | 4 hospitals (clients) each holding brain MRI scans. Every hospital is missing some modalities (FLAIR, T1, T1ce, T2). No hospital shares raw data. |
| **Problem** | Standard FedAvg collapses when clients have wildly different modality distributions. |
| **Fix — 3 components** | ① *Modal balance loss* (KL + prototype loss per modality) ② *Modal combination loss* (cross-client EMA prototype alignment via `global_loss`) ③ *Modality-weighted aggregation* (`avg_encoder_weights` in `fl_utils.py`) |
| **Backbone** | RFNet encoder-decoder with Region-aware Modal Fusion (RFM) and Prototype Region Maps (PRM). |
| **Dataset** | BraTS2020 — 369 cases, 4 modalities, 4-class segmentation (BG, NCR, ED, ET). |
| **Key metric** | Dice score on Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET) across all 15 modality-mask combinations. |

---

## 2. Repo & Branch Setup (do this first)

```bash
# Clone YOUR fork (replace URL with yours)
git clone https://github.com/iHarshMix/FedAMM-implementation.git
cd FedAMM-implementation

# Always verify you're on localWSL
git checkout localWSL
git branch   # should show * localWSL

# Set upstream to the original repo to pull paper author's future fixes
git remote add upstream https://github.com/<original-author>/FedAMM.git
git fetch upstream

# Never merge upstream into main from here; only cherry-pick into localWSL if needed
```

### Golden Git Rules for this project
```
feat/   — new feature or experiment
fix/    — bug fix
chore/  — env setup, dependency pins
exp/    — experiment results / ablation
```

**Commit convention:**
```
feat(training): add gradient clipping to local_training loop
fix(data): handle missing npy files gracefully in Brats_loadall
chore(env): pin torch==2.1.0 in requirements.txt
exp(results): round 300 dice WT=0.87 TC=0.79 ET=0.74
```

**Never commit:**  
- `.npy` data files  
- Model checkpoints > 50 MB (use Git LFS or just `.gitignore`)  
- `results/` directory contents (add to `.gitignore`)

---

## 3. Environment Setup

### 3a. Recommended: Conda environment

```bash
conda create -n fedamm python=3.9 -y
conda activate fedamm

# PyTorch — pick the right CUDA version for your GPU
# CUDA 11.8 (most common on WSL2):
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# CPU-only fallback (very slow, only for debugging):
# pip install torch==2.1.0 torchvision==0.16.0

# Core dependencies
pip install \
  numpy==1.24.4 \
  nibabel \
  medpy \
  scikit-learn \
  pandas \
  scipy \
  matplotlib \
  tensorboard \
  setproctitle \
  tqdm
```

### 3b. Save the environment

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "chore(env): pin all dependencies"
git push origin localWSL
```

### 3c. Verify GPU in WSL2

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX XXXX
```

If `False`: In WSL2, you need the Windows NVIDIA driver (≥ 527.41) + CUDA toolkit inside WSL. Do NOT install a Linux NVIDIA driver inside WSL — it breaks things.

---

## 4. Data Preparation

### 4a. Get BraTS2020

Register and download from: https://www.med.upenn.edu/cbica/brats2020/data.html  
You'll get `MICCAI_BraTS2020_TrainingData.zip` (~9 GB).

```bash
# Unzip to a dedicated location (NOT inside the repo)
unzip MICCAI_BraTS2020_TrainingData.zip -d ~/datasets/BraTS/BRATS2020_Training_Data
```

### 4b. Preprocess to `.npy`

The script is at `utils/preprocessing/preprocess_brats.py`. Edit the paths at the top:

```python
src_path = os.path.expanduser("~/datasets/BraTS/BRATS2020_Training_Data")
tar_path = os.path.expanduser("~/datasets/BraTS/BRATS2020_Training_none_npy")
```

Then run:
```bash
python utils/preprocessing/preprocess_brats.py
# Takes ~20–30 min; produces vol/*.npy and seg/*.npy
```

What it does: crops non-zero bounding box (≥128³), normalizes each modality by its brain-mask mean/std, stacks as `(H,W,D,4)`.

### 4c. Train/Val/Test split

```bash
python utils/preprocessing/data_split.py
# Edit datarootPath inside the file first to point to tar_path above
# Produces train.txt, val.txt, test.txt under tar_path/
```

### 4d. Per-client federated split with imbalanced modality rates

```bash
# First create the 4-client text splits (not in repo, do manually):
mkdir -p ~/datasets/BraTS/BRATS2020_Training_none_npy/client_train_split_seed_1234
# Split train.txt into 4 equal parts:
python - <<'EOF'
import random, os
random.seed(1234)
root = os.path.expanduser("~/datasets/BraTS/BRATS2020_Training_none_npy")
with open(f"{root}/train.txt") as f:
    names = [l.strip() for l in f]
random.shuffle(names)
n = len(names)
out = f"{root}/client_train_split_seed_1234"
os.makedirs(out, exist_ok=True)
for i, part in enumerate(range(4)):
    chunk = names[i*n//4:(i+1)*n//4]
    with open(f"{out}/client_part_{i+1}.txt", "w") as f:
        f.write("\n".join(chunk))
print("Done. Splits:", [len(names[i*n//4:(i+1)*n//4]) for i in range(4)])
EOF

# Then generate the Dirichlet-imbalanced modality CSV files:
cd utils/preprocessing
python generate_dir_imb_mr.py   # uses alpha=0.001 by default (high imbalance)
# Outputs: datalist/dir_brats_split/1024_0.001/client_part_{1-4}_imb.csv
```

### 4e. Update `options.py` paths

Open `options.py` and change the defaults:

```python
parser.add_argument('--datapath', 
    default=os.path.expanduser('~/datasets/BraTS/BRATS2020_Training_none_npy/'), type=str)

parser.add_argument('--train_file', type=dict, default={
    1: "./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_1_imb.csv",
    2: "./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_2_imb.csv",
    3: "./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_3_imb.csv",
    4: "./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_4_imb.csv"})

parser.add_argument('--valid_file', type=str,
    default=os.path.expanduser('~/datasets/BraTS/BRATS2020_Training_none_npy/val.txt'))
```

---

## 5. Known Bugs & Fixes (do before first run)

### Bug 1: `trian.py` is misspelled — rename to `train.py`

```bash
git mv trian.py train.py
git commit -m "fix(train): rename trian.py -> train.py"
```

### Bug 2: `run.sh` calls `train.py`, so update it:

```bash
# run.sh already calls train.py (correct after rename)
# Just fix the device_ids to match your actual GPU count:
# If you have 1 GPU: --device_ids 0,0,0,0
# If you have 4 GPUs: --device_ids 0,1,2,3
```

### Bug 3: `EMA_cls_Fs` crashes when `prior_Fs` has `None` values

In `utils/criterions.py`, the `EMA_cls_Fs` function iterates `prior_Fs.items()` but skips checking for `None` values:

```python
# CURRENT (broken):
def EMA_cls_Fs(prior_Fs, glb_protos):
    if prior_Fs is None:
        return glb_protos
    alpha = 0.999
    for key, value in prior_Fs.items():
        glb_Fs = alpha * value.numpy() + (1-alpha)* glb_protos[key].numpy()
        glb_protos[key] = torch.from_numpy(glb_Fs)
    return glb_protos

# FIX:
def EMA_cls_Fs(prior_Fs, glb_protos):
    if prior_Fs is None:
        return glb_protos
    alpha = 0.999
    for key, value in prior_Fs.items():
        if value is None or glb_protos.get(key) is None:   # <-- add this
            continue
        glb_Fs = alpha * value.numpy() + (1-alpha) * glb_protos[key].numpy()
        glb_protos[key] = torch.from_numpy(glb_Fs)
    return glb_protos
```

### Bug 4: `rfnet.py` forward returns 7 values but `predict.py` unpacks only 1

In `predict.py`, `model(x_input, mask)` is called during test — but when `is_training=False`, `rfnet.Model.forward` returns `F.softmax(fuse_pred, dim=1)` and 6 more items (the training losses). Actually re-check: during inference `is_training=False` short-circuits before the loss block, but the return statement at the end of `forward()` is always hit and always returns 7 values even when not training. **Fix in `rfnet.py`:**

```python
# At the very end of Model.forward(), replace:
return F.softmax(fuse_pred, dim=1), prm_loss, sep_loss, kl_loss, proto_loss, dist, gt

# With:
if self.is_training:
    return F.softmax(fuse_pred, dim=1), prm_loss, sep_loss, kl_loss, proto_loss, dist, gt
else:
    return F.softmax(fuse_pred, dim=1)
```

And in `predict.py`, the call `pred_part = model(x_input, mask)` stays as is (already correct after the fix above).

---

## 6. `.gitignore` Setup

```bash
cat > .gitignore << 'EOF'
# Data & checkpoints
*.npy
*.pth
*.pt
results/
__pycache__/
*.pyc
*.egg-info/
.ipynb_checkpoints/
datalist/dir_brats_split/

# Logs
*.log
TBlog/

# IDE
.vscode/
.idea/
EOF

git add .gitignore
git commit -m "chore(git): add .gitignore"
git push origin localWSL
```

---

## 7. Running Training

### Single GPU (debugging — use `--use_multiprocessing False`)

```bash
conda activate fedamm
cd ~/FedAMM-implementation

python train.py \
  --client_num 4 \
  --c_rounds 300 \
  --local_ep 1 \
  --round_per_train 10 \
  --batch_size 1 \
  --lr 2e-4 \
  --temp 4.0 \
  --mask_type idt \
  --region_fusion_start_epoch 0 \
  --device_ids 0,0,0,0 \
  --gpus 0 \
  --use_multiprocessing False \
  --version debug_run_$(date +%m%d_%H%M)
```

### Multi-GPU (4 GPUs, as in paper)

```bash
python train.py \
  --client_num 4 \
  --c_rounds 1000 \
  --local_ep 1 \
  --round_per_train 100 \
  --device_ids 0,1,2,3 \
  --use_multiprocessing True \
  --version full_run_$(date +%m%d_%H%M)
```

### Monitor with TensorBoard

```bash
tensorboard --logdir results/ --port 6006
# Open http://localhost:6006 in browser
```

---

## 8. Expected Results (from paper Table 2)

| Modality combo | WT Dice | TC Dice | ET Dice |
|---|---|---|---|
| All 4 modalities | ~0.880 | ~0.810 | ~0.750 |
| 3 modalities (avg) | ~0.860 | ~0.790 | ~0.720 |
| 2 modalities (avg) | ~0.820 | ~0.740 | ~0.670 |
| 1 modality (avg) | ~0.750 | ~0.650 | ~0.580 |
| **Overall avg** | **~0.840** | **~0.770** | **~0.710** |

FedAMM outperforms FedAvg and FedProx most significantly in **low-modality (1–2) scenarios**.

---

## 9. Experiment Tracking Workflow (best practice)

After each notable run, commit a results summary:

```bash
# Save a quick results note
cat > results/run_$(date +%m%d_%H%M)_summary.txt << 'EOF'
Round: 300
Config: alpha=0.001, temp=4.0, lr=2e-4, mask_type=idt
Avg Dice: WT=X.XXX TC=X.XXX ET=X.XXX
Notes: ...
EOF

git add results/run_*_summary.txt
git commit -m "exp(results): round 300 baseline run WT=X.XX TC=X.XX ET=X.XX"
git push origin localWSL
```

---

## 10. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `CUDA out of memory` | `patch_size=80`, batch=1 still OOM | Reduce `basic_dims` in `rfnet.py` from 8 to 4 |
| `RuntimeError: Expected all tensors to be on the same device` | multiprocessing CUDA context clash | Set `--use_multiprocessing False` |
| `KeyError` in `EMA_cls_Fs` | None values not handled | Apply Bug 3 fix above |
| `NaN loss` | Learning rate too high or no grad clipping | Lower `--lr` to `1e-4`; add `torch.nn.utils.clip_grad_norm_` |
| `ValueError: normalization type reflect is not supported` | `padding_mode='reflect'` unsupported for some Conv3d configs | Change to `padding_mode='zeros'` in `blocks.py` |
| Slow training on single GPU | All 4 clients run sequentially | Expected; ~4x slower than 4-GPU setup |

---

## 11. File Map (quick reference)

```
train.py                  ← Main FL training loop (was trian.py)
run.sh                    ← Launch script
options.py                ← All hyperparameters
models/
  rfnet.py                ← Full model: 4 encoders + fused decoder + sep decoder
  blocks.py               ← RFM, PRM, attention, conv blocks
  mask.py                 ← Mask generation for cross-attention
  layers.py               ← Duplicate of some blocks (legacy)
dataset/
  datasets_nii.py         ← Main dataset classes used in training
  datasets.py             ← Older dataset classes (mostly unused)
  transforms.py           ← 3D augmentations
utils/
  criterions.py           ← All losses + EMA prototype clustering
  predict.py              ← Test-time sliding window inference + Dice eval
  fl_utils.py             ← FedAvg + modality-weighted aggregation
  lr_scheduler.py         ← Poly LR schedule
  preprocessing/
    preprocess_brats.py   ← Raw NIfTI → .npy
    data_split.py         ← train/val/test text files
    generate_dir_imb_mr.py← Dirichlet modality-imbalanced client CSV files
datalist/                 ← CSV files for each client (generated, gitignored)
results/                  ← Checkpoints + TBlog (gitignored)
```

---

## 12. Next Steps After First Successful Run

1. **Ablation**: Disable each of the 3 components one at a time (`global_loss=0`, `kl_loss=0`, encoder weight averaging → uniform) to reproduce Table 3.
2. **Baseline comparison**: The paper compares against FedAvg, FedProx, and PASSION. FedAvg is just removing `global_loss` and using uniform aggregation.
3. **Visualization**: `predict.py` has `visualize_and_save()` — pass `method_name='FedAMM'` and `output_dir` to generate segmentation overlays matching Fig. 2.
