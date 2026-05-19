# FedAMM Reproduction — Session Context File
> **Read this first in every new chat session.**  
> Goal: Reproduce FedAMM paper results on local WSL machine, then move to college server for full training.

---

## The Big Picture

**Paper:** FedAMM — Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities (MICCAI 2024)  
**Repo (my fork):** https://github.com/iHarshMix/FedAMM-implementation  
**Active branch:** `localWSL` — ALL work happens here. Never touch `main`.  
**College server:** 2 GPUs, 98GB RAM — used for full training later (Phase 2).

### What FedAMM Does (30 seconds)
- 4 hospitals (clients) each have brain MRI scans but are missing some modalities (FLAIR, T1, T1ce, T2)
- They train together via Federated Learning without sharing raw data
- FedAMM's 3 innovations over vanilla FedAvg:
  1. **Modal balance loss** — KL + prototype loss per modality (handles intra-client imbalance)
  2. **Modal combination loss** — cross-client EMA prototype alignment via `global_loss` (handles inter-client differences)
  3. **Modality-weighted aggregation** — hospitals with more of a modality have more say in that modality's encoder weights
- Dataset: BraTS2020 (369 cases, 4 modalities, 4-class segmentation)
- Metric: Dice score on Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET) across all 15 modality-mask combinations

---

## My Machine Setup

| Item | Detail |
|---|---|
| OS | WSL2, Ubuntu |
| GPU | NVIDIA GTX 1660 Ti, 6GB VRAM |
| CUDA | 12.1 |
| Python | 3.9.25 |
| Conda env | `fedamm` (always activate before working) |
| PyTorch | 2.1.0+cu121 |
| NumPy | 1.26.4 (must stay <2, else PyTorch breaks) |

---

## Git Workflow (strict — never deviate)

```
main          ← never touch
localWSL      ← stable, production branch, push here after milestones
fix/*         ← feature/fix branches, merge into localWSL when done
exp/*         ← experiment result branches
```

**Step by step every time:**
```bash
# 1. Always start here
conda activate fedamm
cd ~/FedAMM/FedAMM-implementation
git checkout localWSL
git pull origin localWSL

# 2. Create a feature branch for new work
git checkout -b fix/your-fix-name

# 3. Make changes, then commit at every small step
git add <file>
git commit -m "fix(scope): what you did"

# 4. When milestone is done, merge back
git checkout localWSL
git merge fix/your-fix-name
git push origin localWSL
```

**Commit message format:**
```
fix(rfnet): short description
feat(training): short description
chore(env): short description
exp(results): round 300 WT=0.87 TC=0.79 ET=0.74
```

---

## Current State of the Repo

### Milestone 1 ✅ — Code Fixes (DONE, pushed)
All fixes are on `localWSL` and pushed to GitHub.

| Fix | File | What was wrong | What we did |
|---|---|---|---|
| Rename | `trian.py` → `train.py` | Typo in filename | `git mv` |
| .gitignore | `.gitignore` | No gitignore existed | Created it |
| Inference crash | `models/rfnet.py` | `forward()` always returned 7 values even at test time, crashing `predict.py` | Added `if self.is_training` check, return only prediction tensor during inference |
| None crash | `utils/criterions.py` | `EMA_cls_Fs()` tried to do math on `None` cluster centers | Added `if value is None or glb_protos.get(key) is None: continue` |
| run.sh syntax | `run.sh` | Trailing `\` on last line caused bash error; paths wrong | Fixed syntax, set `--device_ids 0,0,0,0 --use_multiprocessing False` for single GPU |
| Hardcoded paths | `options.py` | Paths pointed to paper author's server `/root/autodl-tmp/` | Replaced with `os.path.expanduser('~/datasets/...')` |

### Milestone 2 ✅ — Environment Setup (DONE, pushed)
```bash
conda activate fedamm   # always do this first
```

| Package | Version |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.1.0+cu121 |
| NumPy | 1.26.4 |
| nibabel | 5.3.3 |
| medpy | 0.5.2 |
| scikit-learn | 1.6.1 |
| pandas | 2.3.3 |
| scipy | 1.13.1 |
| matplotlib | 3.9.4 |
| tensorboard | 2.20.0 |
| setproctitle | 1.3.7 |
| tqdm | 4.67.3 |

`requirements.txt` is committed and pushed.

### Milestone 3 ⏳ — Data Preparation (NEXT)
Steps to complete:
1. Register and download BraTS2020 from https://www.med.upenn.edu/cbica/brats2020/data.html
2. Unzip to `~/datasets/BraTS/BRATS2020_Training_Data/`
3. Run `utils/preprocessing/preprocess_brats.py` → produces `~/datasets/BraTS/BRATS2020_Training_none_npy/vol/*.npy` and `seg/*.npy`
4. Run `utils/preprocessing/data_split.py` → produces `train.txt`, `val.txt`, `test.txt`
5. Manually split `train.txt` into 4 client parts
6. Run `utils/preprocessing/generate_dir_imb_mr.py` → produces per-client CSV files with Dirichlet-imbalanced modality masks

### Milestone 4 ⏳ — Debug Training Run (TODO)
- Single GPU, small run to verify no crashes
- `--c_rounds 5 --round_per_train 1 --use_multiprocessing False`

### Milestone 5 ⏳ — College Server Setup (TODO)
- Copy repo + requirements.txt to server
- Install same environment
- Update `run.sh` for 2 GPUs: `--device_ids 0,1,0,1 --use_multiprocessing True`
- Full training run: `--c_rounds 1000`

### Milestone 6 ⏳ — Reproduce Paper Results (TODO)
- Target: WT~0.84, TC~0.77, ET~0.71 average across all 15 modality combos
- Run ablation: disable each of 3 components one at a time

---

## Key Files Map

```
train.py                     ← Main FL training loop (was trian.py, now fixed)
run.sh                       ← Launch script (fixed)
options.py                   ← All hyperparameters (paths fixed)
models/
  rfnet.py                   ← Full model: 4 encoders + fused decoder (inference fixed)
  blocks.py                  ← RFM, PRM, attention, conv blocks
  mask.py                    ← Mask generation for cross-attention
dataset/
  datasets_nii.py            ← Main dataset class used in training
  transforms.py              ← 3D augmentations (RandCrop, RandomFlip, etc.)
utils/
  criterions.py              ← All losses + EMA prototype clustering (None fix applied)
  predict.py                 ← Test-time sliding window inference + Dice eval
  fl_utils.py                ← FedAvg + modality-weighted aggregation
  lr_scheduler.py            ← Polynomial LR schedule
  preprocessing/
    preprocess_brats.py      ← Raw NIfTI → .npy (run once)
    data_split.py            ← train/val/test text file splits (run once)
    generate_dir_imb_mr.py   ← Dirichlet modality-imbalanced client CSV files (run once)
```

---

## Important Things to Never Forget

1. **Always `conda activate fedamm`** before any python or git work
2. **Never work on `main`** — only `localWSL` and feature branches
3. **NumPy must stay <2** — `pip install "numpy<2"` if it ever upgrades
4. **Commit at every small step** — don't batch multiple changes into one commit
5. **`run.sh` has single GPU config** — change for college server before full run
6. **Data files never go into git** — `.gitignore` covers `*.npy`, `*.pth`, `results/`

---

## How to Continue in a New Chat

1. Open a new chat in the same Claude project (so source files are attached)
2. Paste this entire file as your first message
3. Also attach `REPRODUCTION_GUIDE.md` from the repo
4. Say: "Continue from where we left off — next step is Milestone X"

Claude will have full context and pick up exactly where we stopped.
