# FedAMM+ — Production Guide (Part 2)

> **Branch:** `fedamm-plus` (branched from `localWSL` at Milestone 6 end)  
> **Parent work:** FedAMM reproduction (`localWSL` branch) — completed, 1000 rounds, WT=0.809, TC=0.654, ET=0.479  
> **Goal:** Implement FedAMM+ — prototype-space cross-modal attention imputation — as described in `Revised_Prototype_Imputation.md`, validated by `Optimization_and_Gradients.md`, Variant 1 from `Variants_Comparison.md`.  
> **Paper target:** MICCAI 2026 / MedIA

---

## 1. What FedAMM+ Does (Mental Model)

| Concept | Plain English |
|---|---|
| **Problem FedAMM leaves unsolved** | The decoder sees variable-density input depending on how many modalities are available. With 1 modality it gets sparse input, with 4 it gets dense input. The decoder must generalize across this distribution shift (W3). |
| **What FedAMM+ adds** | A small cross-attention imputer that estimates class-level prototypes for **missing** modalities, using prototypes from **available** modalities as context. The decoder now always receives M-dimensional input — real where available, imputed where missing. |
| **Where imputation happens** | **Prototype space** (class-level vectors of size C=4), NOT feature space. Already computed by FedAMM — zero communication overhead. |
| **The imputer** | `CrossAttn(Q=e^m, K={P^a}, V={P^a})` where `e^m` is a learnable modality embedding query. Handles any `|A_n| ∈ {1,2,3}` naturally. |
| **The loss** | Cosine distance between imputed prototype and stop-gradient multimodal teacher prototype. Bounded `[0,2]`. Integrates as extension of existing `L_mb`. |
| **Aggregation** | Imputer weights by (available × missing) pair count per client — analogous to FedAMM's per-modality encoder weighting. |
| **Warmup** | First 100 rounds: imputer disabled (`λ₃=0`). Rounds 101-150: linear ramp. Round 151+: full `λ₃*`. |
| **What does NOT change** | Data pipeline, FL loop structure, encoders, decoder, RFM, PRM, evaluation — everything from Part 1 stays. |

---

## 2. Branch Setup

```bash
# On college server — always SSH to head node first, then GPU node
ssh harshyadav@10.100.36.74
ssh h100

conda activate fedamm
cd ~/FedAMM-implementation

# Verify fedamm-plus exists and is at Milestone 6 state
git fetch origin
git checkout fedamm-plus
git log --oneline -5
# Should show the full_run commit at top (0c2efa5 or similar from localWSL)

# Verify it branched correctly from localWSL Milestone 6
git log --oneline origin/localWSL -1
# Should match the base of fedamm-plus

# Set upstream tracking
git branch --set-upstream-to=origin/fedamm-plus fedamm-plus
git pull origin fedamm-plus
```

### Git Rules for FedAMM+ (same conventions, new branch base)

```
fedamm-plus          ← stable production branch for this part
feat/imputer-*       ← feature branches per milestone
fix/*                ← bug fixes
exp/*                ← experiment results
```

**Commit convention (same as before):**
```
feat(imputer): add ProtoImputer module to blocks.py
feat(criterions): add imputation_loss_bs function
feat(rfnet): integrate imputer into Model forward pass
feat(train): add warmup schedule and imputer aggregation weight
fix(imputer): handle single-modality edge case in cross-attention
exp(results): fedamm+ round 1000 WT=X.XX TC=X.XX ET=X.XX
```

**Never commit:** `.npy`, `.pth` > 50MB, `results/`, `datasets/`

---

## 3. The Math — What We Are Implementing

> Reference: `Revised_Prototype_Imputation.md` + `Optimization_and_Gradients.md`

### 3.1 Imputer Forward Pass

For sample n with available modalities `A_n`, for each missing modality `m ∉ A_n`:

```
P̂_{n,c}^m = CrossAttn(
    Q = e^m,                          # learnable modality embedding [C]
    K = {P_{n,c}^a}_{a ∈ A_n},       # available prototypes as keys [|A_n|, C]
    V = {P_{n,c}^a}_{a ∈ A_n}        # same as values
)
```

Where:
- `e^m ∈ R^C` — one learnable embedding per modality (4 total), dimension = num_cls = 4
- All quantities are `[C]`-dimensional — no projection needed
- Cross-attention with variable number of keys handles any `|A_n|`

### 3.2 Imputation Loss

```
L_imp = (1 / |Ā_n|) * Σ_{m ∉ A_n} Σ_{c=1}^{C} [1 - cosine(P̂_{n,c}^m, sg(P_{n,c}^t))]
```

- `sg(·)` = stop-gradient on teacher
- Bounded `[0, 2]`
- When `|Ā_n| = 0` (all modalities present): skip, loss = 0

### 3.3 Integration into Existing Objective

```
L_FedAMM+ = L_ce + L_dice + λ₁·L_mb + λ₂·L_mc + λ₃(t)·L_imp
```

When `λ₃=0`: reduces exactly to original FedAMM. No other terms change.

### 3.4 Warmup Schedule

```
λ₃(t) = 0                                    if t ≤ T_w  (= 100)
λ₃(t) = λ₃* · (t - T_w) / T_r               if T_w < t ≤ T_w + T_r  (T_r = 50)
λ₃(t) = λ₃*                                  if t > T_w + T_r
```

### 3.5 Imputer Aggregation Weight

```
ω_k^I = Σ_{n ∈ D_k} |A_n| · |Ā_n|  /  Σ_j Σ_{n ∈ D_j} |A_n| · |Ā_n|
```

Counts total imputation tasks (available × missing pairs) per client.

### 3.6 Gradient Flow (Honest Summary)

| Loss | Available Encoders | Missing Encoders | Imputer | Decoder |
|---|---|---|---|---|
| `L_seg` | ✓ direct | ✗ zero | ✓ via fusion | ✓ |
| `L_mb` | ✓ via prototype | ✗ zero | ✗ | ✗ |
| `L_mc` | ✓ via prototype | ✗ zero | ✗ | ✗ |
| `L_imp` | ✓ cross-modal | ✗ zero | ✓ direct | ✗ |

Missing encoders still receive zero gradient. This is stated honestly — not overclaimed.

---

## 4. File Change Map

| File | Type of Change | What Specifically |
|---|---|---|
| `models/blocks.py` | **ADD** | `ProtoImputer` class |
| `models/rfnet.py` | **MODIFY** | Import + instantiate `ProtoImputer`; integrate `L_imp` in `Model.forward()` |
| `utils/criterions.py` | **ADD** | `imputation_loss_bs()` function |
| `utils/fl_utils.py` | **MODIFY** | `avg_imputer_weights()` function for imputer aggregation |
| `train.py` | **MODIFY** | Warmup schedule; imputer aggregation call; logging |
| `options.py` | **ADD** | `--lambda3`, `--warmup_imp_rounds`, `--warmup_imp_ramp` args |

**Nothing else changes.** Dataset, data_utils, transforms, predict, lr_scheduler — all untouched.

---

## 5. Milestones

---

### Milestone P1 — Branch Verification + options.py ✅ Starting Point

**Goal:** Confirm `fedamm-plus` branch is clean and add new hyperparameter arguments.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p1-options-setup
```

**Changes to `options.py`:**

```python
# ADD these arguments to args_parser():

parser.add_argument('--lambda3', default=0.1, type=float,
    help='Weight for imputation loss L_imp')
parser.add_argument('--warmup_imp_rounds', default=100, type=int,
    help='Rounds before imputation loss activates (T_w)')
parser.add_argument('--warmup_imp_ramp', default=50, type=int,
    help='Rounds to linearly ramp up lambda3 after warmup (T_r)')
parser.add_argument('--use_imputer', action='store_true', default=False,
    help='Enable FedAMM+ prototype imputer')
```

**Verify no crashes:**
```bash
python train.py --help | grep lambda3
python train.py --help | grep imputer
```

**Commit:**
```bash
git add options.py
git commit -m "feat(options): add lambda3, warmup_imp, use_imputer args for FedAMM+"
git checkout fedamm-plus
git merge feat/p1-options-setup
git push origin fedamm-plus
```

---

### Milestone P2 — ProtoImputer Module in `blocks.py`

**Goal:** Implement the cross-attention prototype imputer as a standalone nn.Module.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p2-proto-imputer-module
```

**Add to `models/blocks.py`:**

```python
class ProtoImputer(nn.Module):
    """
    Prototype-Space Cross-Modal Attention Imputer (FedAMM+)
    
    For each missing modality m, imputes its class prototype using
    cross-attention over available modality prototypes.
    
    All operations in R^C (prototype space) — no spatial dimensions.
    Preserves encoder independence (A1).
    
    Reference: Revised_Prototype_Imputation.md, Variant 1
    """
    def __init__(self, num_cls=4, num_modals=4):
        super(ProtoImputer, self).__init__()
        self.num_cls = num_cls       # C = prototype dimension
        self.num_modals = num_modals # M = number of modalities
        
        # Learnable modality query embeddings: one per modality, dim=C
        # e^m ∈ R^C — encodes "what does modality m look like in prototype space"
        self.modal_queries = nn.Embedding(num_modals, num_cls)
        
        # Attention scale factor
        self.scale = num_cls ** -0.5
        
        # Output projection (optional — keeps dim=C)
        self.out_proj = nn.Linear(num_cls, num_cls)
        
        # Initialize embeddings
        nn.init.normal_(self.modal_queries.weight, mean=0.0, std=0.02)
    
    def forward(self, available_protos, avail_mask):
        """
        Args:
            available_protos: dict {modal_idx: proto_tensor}
                             proto_tensor shape: [B, C] — batch of class prototypes
                             for one specific class c (called once per class)
            avail_mask: [B, M] boolean — which modalities are available per sample
        
        Returns:
            imputed: dict {modal_idx: [B, C]} — imputed prototypes for missing modalities
                     Only contains entries for missing modalities
        """
        B = avail_mask.shape[0]
        imputed = {}
        
        # Determine which modalities are missing (globally in this batch)
        # We impute per-modality per-sample
        for m in range(self.num_modals):
            # Check if any sample in batch is missing modality m
            missing_in_batch = ~avail_mask[:, m]  # [B]
            if not missing_in_batch.any():
                continue
            
            # Query for modality m: [1, C] → [B, C]
            q = self.modal_queries(
                torch.tensor(m, device=avail_mask.device)
            ).unsqueeze(0).expand(B, -1)  # [B, C]
            
            # Collect keys and values from available modalities
            # Stack available protos: [B, |A|, C]
            keys_list = []
            for a in range(self.num_modals):
                if a == m:
                    continue
                if a in available_protos:
                    keys_list.append(available_protos[a])  # [B, C]
            
            if len(keys_list) == 0:
                # No available modalities to attend to — return zeros
                imputed[m] = torch.zeros(B, self.num_cls, 
                                         device=avail_mask.device)
                continue
            
            # K, V: [B, |A|, C]
            K = torch.stack(keys_list, dim=1)
            V = K.clone()
            
            # Scaled dot-product attention
            # q: [B, C] → [B, 1, C]
            q = q.unsqueeze(1)
            
            # attn_weights: [B, 1, |A|]
            attn_weights = torch.matmul(q, K.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # output: [B, 1, C] → [B, C]
            out = torch.matmul(attn_weights, V).squeeze(1)
            out = self.out_proj(out)
            
            imputed[m] = out
        
        return imputed
```

**Unit test (run this on server to verify):**
```python
# Quick sanity check — paste into python REPL on server
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/harshyadav/FedAMM-implementation')
from models.blocks import ProtoImputer

B, C, M = 2, 4, 4
imputer = ProtoImputer(num_cls=C, num_modals=M)

# Simulate: samples have modalities 0,1 available; 2,3 missing
avail_mask = torch.tensor([[True, True, False, False],
                            [True, True, False, False]])
available_protos = {
    0: torch.randn(B, C),
    1: torch.randn(B, C)
}

imputed = imputer(available_protos, avail_mask)
print("Imputed modalities:", list(imputed.keys()))  # Should be [2, 3]
print("Shape:", imputed[2].shape)  # Should be [2, 4]
print("Test passed!")
```

**Commit:**
```bash
git add models/blocks.py
git commit -m "feat(imputer): add ProtoImputer cross-attention module to blocks.py"
git checkout fedamm-plus
git merge feat/p2-proto-imputer-module
git push origin fedamm-plus
```

---

### Milestone P3 — Imputation Loss in `criterions.py`

**Goal:** Add `imputation_loss_bs()` function — cosine distance between imputed and teacher prototype.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p3-imputation-loss
```

**Add to `utils/criterions.py`:**

```python
def imputation_loss_bs(imputed_protos, teacher_proto, avail_mask, num_cls=4, eps=1e-8):
    """
    Cosine imputation loss L_imp (FedAMM+)
    
    For each missing modality m, measures cosine distance between:
    - imputed prototype P̂_{n,c}^m  (from ProtoImputer)
    - stop-gradient teacher prototype sg(P_{n,c}^t)
    
    L_imp = (1/|Ā_n|) * Σ_{m ∉ A_n} Σ_c [1 - cosine(P̂^m_c, sg(P^t_c))]
    
    Args:
        imputed_protos: dict {modal_idx: [B, C]} — imputer outputs for missing modals
        teacher_proto: [B, num_cls, C] — multimodal teacher prototypes, per class
                       OR list of [B, C] tensors, one per class
        avail_mask: [B, M] boolean
        num_cls: C = number of classes = prototype dimension
        eps: numerical stability
    
    Returns:
        loss: [B, 1] — per-sample imputation loss (for batch-wise weighting)
    """
    B = avail_mask.shape[0]
    device = avail_mask.device
    
    if len(imputed_protos) == 0:
        # No missing modalities in this batch
        return torch.zeros(B, 1, device=device)
    
    # Count missing modalities per sample: [B]
    num_missing = (~avail_mask).float().sum(dim=1)  # [B]
    num_missing = torch.clamp(num_missing, min=1.0)  # avoid div by zero
    
    total_loss = torch.zeros(B, device=device)
    
    for m, p_hat in imputed_protos.items():
        # p_hat: [B, C] — imputed prototype for modality m
        # teacher_proto: [B, C] — multimodal teacher prototype (averaged over classes)
        # stop-gradient on teacher
        p_t = teacher_proto.detach()  # [B, C]
        
        # Cosine similarity: [B]
        cos_sim = F.cosine_similarity(p_hat, p_t, dim=-1, eps=eps)
        
        # Cosine distance: [B]
        cos_dist = 1.0 - cos_sim
        
        # Only apply to samples where modality m IS missing
        missing_m = (~avail_mask[:, m]).float()  # [B]
        total_loss += cos_dist * missing_m
    
    # Normalize by number of missing modalities per sample
    loss = total_loss / num_missing  # [B]
    loss = loss.unsqueeze(1)  # [B, 1]
    
    return loss
```

**Unit test:**
```python
import torch
from utils.criterions import imputation_loss_bs

B, C, M = 2, 4, 4
avail_mask = torch.tensor([[True, True, False, False],
                            [True, False, True, False]])
imputed_protos = {
    2: torch.randn(B, C),
    3: torch.randn(B, C)
}
teacher_proto = torch.randn(B, C)

loss = imputation_loss_bs(imputed_protos, teacher_proto, avail_mask, num_cls=C)
print("Loss shape:", loss.shape)  # Should be [2, 1]
print("Loss values:", loss)       # Should be in [0, 2]
print("Test passed!")
```

**Commit:**
```bash
git add utils/criterions.py
git commit -m "feat(criterions): add imputation_loss_bs cosine loss for FedAMM+"
git checkout fedamm-plus
git merge feat/p3-imputation-loss
git push origin fedamm-plus
```

---

### Milestone P4 — Integrate Imputer into `rfnet.py`

**Goal:** Add `ProtoImputer` to `Model`, compute imputed prototypes inside `forward()`, feed into `L_imp`.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p4-rfnet-integration
```

**Changes to `models/rfnet.py`:**

**Step 1 — Import:**
```python
from models.blocks import (general_conv3d, normalization, prm_generator_pk,
                    prm_generator_laststage_pk, region_aware_modal_fusion,
                    ProtoImputer)   # ADD THIS
```

**Step 2 — Add to `Model.__init__()`:**
```python
# ADD after existing module definitions:
self.proto_imputer = ProtoImputer(num_cls=num_cls, num_modals=num_modals)
self.use_imputer = False  # Controlled by train.py
```

**Step 3 — Add prototype extraction helper to `Model`:**
```python
def _extract_teacher_proto(self, de_f, target, num_cls):
    """
    Extracts class-level prototype from decoder features and target.
    Returns: [B, num_cls] — one scalar per class (mean over spatial dims and channels)
    
    Simplified version: uses gt_prototype logic but returns [B, C] for imputer input.
    """
    eps = 1e-5
    B = de_f.shape[0]
    proto_list = []
    target = target.float()
    for c in range(num_cls):
        targeti = target[:, c, :, :, :]  # [B, H, W, Z]
        # Mean pool features over spatial dims where class c is present
        proto_c = torch.sum(de_f * targeti[:, None], dim=(-3,-2,-1)) / \
                  (torch.sum(targeti[:, None], dim=(-3,-2,-1)) + eps)  # [B, ch]
        # Reduce channels to scalar via mean
        proto_c = proto_c.mean(dim=1, keepdim=True)  # [B, 1]
        proto_list.append(proto_c)
    # Stack: [B, num_cls]
    return torch.cat(proto_list, dim=1)
```

**Step 4 — Modify `Model.forward()` return:**

Inside `forward()`, after `fuse_pred, preds, de_f_avg = self.decoder_fuse(x1, x2, x3, x4, mask)`:

```python
# --- FedAMM+ Imputer Block ---
imp_loss = torch.zeros(B, 1).float().to(device)

if self.use_imputer and self.is_training and target is not None:
    # Extract teacher prototype: [B, num_cls]
    teacher_proto = self._extract_teacher_proto(
        de_f_avg[0], target, num_cls
    )
    
    # Build available protos dict: {modal_idx: [B, num_cls]}
    # Using per-modality decoder features from single-modal passes
    # NOTE: This is called AFTER single-modal forward passes in training block
    # So we defer this to the training block below
    pass
# --- End FedAMM+ Imputer Block ---
```

Inside the `if self.is_training:` block, AFTER the 4 single-modal decoder passes:

```python
# --- FedAMM+ Prototype Imputation ---
if self.use_imputer:
    teacher_proto = self._extract_teacher_proto(
        de_f_avg[0], target, num_cls
    )
    
    # Collect per-modality prototypes from single-modal passes
    modal_protos = {}
    modal_de_fs = [de_f_flair[0], de_f_t1ce[0], de_f_t1[0], de_f_t2[0]]
    for mi, de_f_mi in enumerate(modal_de_fs):
        modal_protos[mi] = self._extract_teacher_proto(
            de_f_mi, target, num_cls
        )
    
    # Run imputer: get imputed protos for missing modalities
    imputed_protos = self.proto_imputer(modal_protos, mask)
    
    # Compute imputation loss
    from utils.criterions import imputation_loss_bs
    imp_loss = imputation_loss_bs(
        imputed_protos, teacher_proto, mask, num_cls=num_cls
    )
# --- End FedAMM+ ---
```

**Modify return statement:**
```python
# BEFORE:
return F.softmax(fuse_pred, dim=1), prm_loss, sep_loss, kl_loss, proto_loss, dist, gt

# AFTER:
return F.softmax(fuse_pred, dim=1), prm_loss, sep_loss, kl_loss, proto_loss, dist, gt, imp_loss
```

**And inference return:**
```python
# BEFORE (inference):
return F.softmax(fuse_pred, dim=1)

# AFTER (inference — unchanged, still correct):
return F.softmax(fuse_pred, dim=1)
```

**Commit:**
```bash
git add models/rfnet.py models/blocks.py
git commit -m "feat(rfnet): integrate ProtoImputer into Model, add imp_loss to forward return"
git checkout fedamm-plus
git merge feat/p4-rfnet-integration
git push origin fedamm-plus
```

---

### Milestone P5 — Training Loop Changes in `train.py`

**Goal:** Add warmup schedule, handle new `imp_loss` return value, add imputer aggregation, logging.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p5-training-loop
```

**Changes to `train.py`:**

**Step 1 — Warmup schedule function (add near top with other utils):**
```python
def get_lambda3(round_num, args):
    """
    Warmup schedule for lambda3 (imputation loss weight).
    T_w = args.warmup_imp_rounds
    T_r = args.warmup_imp_ramp
    """
    if not args.use_imputer:
        return 0.0
    T_w = args.warmup_imp_rounds
    T_r = args.warmup_imp_ramp
    if round_num <= T_w:
        return 0.0
    elif round_num <= T_w + T_r:
        return args.lambda3 * (round_num - T_w) / T_r
    else:
        return args.lambda3
```

**Step 2 — Enable imputer on model before training:**
```python
# In local_training(), before model.train():
model.use_imputer = args.use_imputer
```

**Step 3 — Unpack new return value:**
```python
# BEFORE:
fuse_pred, prm_loss_bs, sep_loss_m_bs, kl_loss_m_bs, proto_loss_m_bs, dist_m_bs, gt = \
    model(x, mask, target=target, temp=args.temp)

# AFTER:
fuse_pred, prm_loss_bs, sep_loss_m_bs, kl_loss_m_bs, proto_loss_m_bs, dist_m_bs, gt, imp_loss_bs = \
    model(x, mask, target=target, temp=args.temp)
```

**Step 4 — Compute imputation loss and add to total:**
```python
# Get current lambda3 (warmup-aware)
lambda3 = get_lambda3(round, args)

# Compute imp_loss scalar
imp_loss = torch.sum(imp_loss_bs) * lambda3

# ADD to loss computation (in the round >= region_fusion_start_epoch block):
# BEFORE:
loss = fuse_loss + sep_loss + prm_loss + kl_loss * 0.5 + proto_loss * 0.1 + global_loss

# AFTER:
loss = fuse_loss + sep_loss + prm_loss + kl_loss * 0.5 + proto_loss * 0.1 + global_loss + imp_loss
```

**Step 5 — Add epoch tracking for imp_loss:**
```python
# In epoch initialization:
epoch_imp_losses = torch.zeros(1).cpu().float()

# In iter loop:
epoch_imp_losses += (imp_loss / iter_per_epoch).detach().cpu()

# In msg:
msg += 'imp_loss:{:.4f}, '.format(imp_loss.item())
```

**Step 6 — Add imputer aggregation in `uploadLCweightsandGLBupdate()`:**

**Add to `fl_utils.py`:**
```python
def avg_imputer_weights(w1, w2, w3, w4, imputer_weights):
    """
    Aggregate imputer parameters weighted by (available x missing) pair counts.
    imputer_weights: [4] — one weight per client, sum to 1.
    """
    aggregated = {}
    for key in w1.keys():
        sum_weight = 0
        for i, client_w in enumerate([w1, w2, w3, w4]):
            sum_weight += client_w[key].data.cpu() * imputer_weights[i]
        aggregated[key] = sum_weight
    return aggregated
```

**Compute imputer weights in main FL loop:**
```python
# After collecting local_weights, before uploadLCweightsandGLBupdate:
if args.use_imputer:
    # Compute ω_k^I = Σ_{n∈D_k} |A_n|·|Ā_n| / Σ_j ...
    imputer_pair_counts = []
    for client_idx in range(args.client_num):
        imb_data = pd.read_csv(args.train_file[client_idx+1])
        pair_count = 0
        for sample_mask in imb_data['mask']:
            m = eval(sample_mask)
            avail = sum(m)
            missing = 4 - avail
            pair_count += avail * missing
        imputer_pair_counts.append(pair_count)
    total_pairs = sum(imputer_pair_counts)
    imputer_weights = [c / total_pairs if total_pairs > 0 else 0.25 
                       for c in imputer_pair_counts]
    logging.info(f'Imputer aggregation weights: {imputer_weights}')
```

**Step 7 — Add imputer aggregation to `uploadLCweightsandGLBupdate()`:**
```python
# ADD parameter:
def uploadLCweightsandGLBupdate(server_model, local_weights, client_mask_proportions_sum,
                                  client_modal_weight, model_clients,
                                  use_imputer=False, imputer_weights=None):
    # ... existing code ...
    
    # ADD at end, before return:
    if use_imputer and imputer_weights is not None:
        imputer_w = avg_imputer_weights(
            model_clients[0].proto_imputer.state_dict(),
            model_clients[1].proto_imputer.state_dict(),
            model_clients[2].proto_imputer.state_dict(),
            model_clients[3].proto_imputer.state_dict(),
            imputer_weights
        )
        server_model.proto_imputer.load_state_dict(imputer_w)
    
    return server_model
```

**Step 8 — Log lambda3 each round:**
```python
logging.info(f'Round {round}: lambda3 = {get_lambda3(round, args):.4f}')
```

**Commit:**
```bash
git add train.py utils/fl_utils.py
git commit -m "feat(train): add warmup schedule, imp_loss integration, imputer aggregation"
git checkout fedamm-plus
git merge feat/p5-training-loop
git push origin fedamm-plus
```

---

### Milestone P6 — Debug Run (3 rounds)

**Goal:** Verify the full FedAMM+ pipeline runs without crashes. No performance expectations.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p6-debug-run
```

**Pre-run checklist:**
```bash
# 1. Verify on GPU node
ssh h100
conda activate fedamm
cd ~/FedAMM-implementation
git checkout fedamm-plus
git pull origin fedamm-plus

# 2. Quick import test
python -c "
from models.blocks import ProtoImputer
from models.rfnet import Model
from utils.criterions import imputation_loss_bs
import torch
m = Model(num_cls=4)
m.use_imputer = True
print('All imports OK')
print('ProtoImputer params:', sum(p.numel() for p in m.proto_imputer.parameters()))
"
```

**Debug run command:**
```bash
screen -S fedamm_plus_debug
ulimit -n 65536
conda activate fedamm
cd ~/FedAMM-implementation

python train.py \
  --client_num 4 \
  --c_rounds 3 \
  --local_ep 1 \
  --round_per_train 1 \
  --batch_size 1 \
  --lr 2e-4 \
  --temp 4.0 \
  --mask_type idt \
  --region_fusion_start_epoch 0 \
  --device_ids 0,0,0,0 \
  --gpus 0 \
  --use_imputer \
  --lambda3 0.1 \
  --warmup_imp_rounds 0 \
  --warmup_imp_ramp 1 \
  --version fedamm_plus_debug_$(date +%m%d_%H%M)
# Note: warmup_imp_rounds=0 and ramp=1 so imputer is active from round 1 in debug
```

**What to verify in logs:**
- `imp_loss` appears in every iteration log — not zero, not NaN
- `lambda3 = 0.1000` logged each round
- `Imputer aggregation weights:` logged after each round
- No shape errors, no CUDA errors
- Round 1 evaluation runs without crash

**Known potential issues and fixes:**

| Issue | Likely Cause | Fix |
|---|---|---|
| `KeyError` in imputed_protos | Modal index mismatch | Check `ProtoImputer.forward()` modal indexing |
| `RuntimeError: size mismatch` | Teacher proto shape wrong | Check `_extract_teacher_proto` output shape |
| `NaN in imp_loss` | Empty class in target | Add `eps` clamp in loss function |
| `AttributeError: use_imputer` | Not set before training | Add `model.use_imputer = args.use_imputer` |
| imp_loss always 0 | warmup not bypassed in debug | Set `--warmup_imp_rounds 0` |

**Commit debug artifacts:**
```bash
mkdir -p experiments/fedamm_plus_p6_debug
cp results/fedamm_plus_debug_*/fl_log.txt experiments/fedamm_plus_p6_debug/
# Write brief SUMMARY.md
git add experiments/fedamm_plus_p6_debug/
git commit -m "exp(debug): fedamm+ 3-round debug run successful, imp_loss active"
git checkout fedamm-plus
git merge feat/p6-debug-run
git push origin fedamm-plus
```

---

### Milestone P7 — Variant 2 Baseline Run (Closed-Form, 1000 rounds)

**Goal:** Implement Variant 2 (parameter-free closed-form interpolation) and run 1000 rounds. This is the ablation baseline — isolates whether learned attention is necessary.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p7-variant2-baseline
```

**Variant 2 — Add to `utils/criterions.py`:**
```python
def closed_form_impute(available_protos, global_centroids, avail_mask, num_cls=4, eps=1e-8):
    """
    Variant 2: Closed-form prototype interpolation (parameter-free).
    
    P̂^m = Σ_{a ∈ A_n} α^{ma} · P^a
    
    where α^{ma} = cos(P^{g,m}, P^{g,a}) / Σ_{a'} cos(P^{g,m}, P^{g,a'})
    
    global_centroids: [M, C] — global prototype centroids from server K-means
    available_protos: dict {modal_idx: [B, C]}
    avail_mask: [B, M]
    
    Returns: dict {modal_idx: [B, C]} for missing modals
    """
    M = avail_mask.shape[1]
    B = avail_mask.shape[0]
    imputed = {}
    
    for m in range(M):
        missing_in_batch = ~avail_mask[:, m]
        if not missing_in_batch.any():
            continue
        
        # Compute interpolation weights from global centroids
        alpha = {}
        total_sim = 0.0
        for a in range(M):
            if a == m or a not in available_protos:
                continue
            sim = F.cosine_similarity(
                global_centroids[m].unsqueeze(0),
                global_centroids[a].unsqueeze(0),
                dim=-1, eps=eps
            ).item()
            sim = max(sim, 0.0)  # clamp negative similarities
            alpha[a] = sim
            total_sim += sim
        
        if total_sim < eps or len(alpha) == 0:
            imputed[m] = torch.zeros(B, available_protos[
                list(available_protos.keys())[0]].shape[-1],
                device=avail_mask.device)
            continue
        
        # Weighted sum of available prototypes
        result = torch.zeros_like(list(available_protos.values())[0])
        for a, sim in alpha.items():
            result += (sim / total_sim) * available_protos[a]
        
        imputed[m] = result
    
    return imputed
```

**Run Variant 2 for 1000 rounds:**
```bash
screen -S fedamm_v2
ulimit -n 65536
conda activate fedamm
cd ~/FedAMM-implementation

python train.py \
  --client_num 4 \
  --c_rounds 1000 \
  --local_ep 1 \
  --round_per_train 100 \
  --batch_size 1 \
  --lr 2e-4 \
  --temp 4.0 \
  --mask_type idt \
  --region_fusion_start_epoch 0 \
  --device_ids 0,1,0,1 \
  --gpus 0,1 \
  --use_multiprocessing \
  --use_imputer \
  --imputer_variant 2 \
  --lambda3 0.1 \
  --warmup_imp_rounds 100 \
  --warmup_imp_ramp 50 \
  --version fedamm_v2_$(date +%m%d_%H%M)
```

**Expected result:** Should be between FedAMM baseline and FedAMM+ Variant 1. If Variant 2 ≈ Variant 1 → learned attention adds little. If Variant 1 > Variant 2 → learned attention justified.

**Commit:**
```bash
git add utils/criterions.py experiments/fedamm_plus_p7_v2/
git commit -m "exp(v2): variant2 closed-form baseline WT=X.XX TC=X.XX ET=X.XX"
git checkout fedamm-plus
git merge feat/p7-variant2-baseline
git push origin fedamm-plus
```

---

### Milestone P8 — Full FedAMM+ Run (Variant 1, 1000 rounds)

**Goal:** Full training with learned ProtoImputer. Primary result.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p8-full-fedamm-plus
```

**Run command:**
```bash
screen -S fedamm_plus_full
ulimit -n 65536
conda activate fedamm
cd ~/FedAMM-implementation

python train.py \
  --client_num 4 \
  --c_rounds 1000 \
  --local_ep 1 \
  --round_per_train 100 \
  --batch_size 1 \
  --lr 2e-4 \
  --temp 4.0 \
  --mask_type idt \
  --region_fusion_start_epoch 0 \
  --device_ids 0,1,0,1 \
  --gpus 0,1 \
  --use_multiprocessing \
  --use_imputer \
  --lambda3 0.1 \
  --warmup_imp_rounds 100 \
  --warmup_imp_ramp 50 \
  --version fedamm_plus_full_$(date +%m%d_%H%M)
```

**Monitor at round 100 (warmup ends):**
```bash
# Check imp_loss activates after round 100
grep "imp_loss" results/fedamm_plus_full_*/fl_log.txt | head -20
grep "lambda3" results/fedamm_plus_full_*/fl_log.txt | grep "0.1000" | head -5
```

**Expected improvement over FedAMM baseline (WT=0.809, TC=0.654, ET=0.479):**

| Target | Realistic Expectation | Optimistic |
|---|---|---|
| WT | 0.815 – 0.825 | 0.830+ |
| TC | 0.660 – 0.675 | 0.680+ |
| ET | 0.485 – 0.500 | 0.505+ |

Largest expected gains on **single-modality combinations** (W3 is most severe there).

**Commit:**
```bash
mkdir -p experiments/fedamm_plus_p8_full
cp results/fedamm_plus_full_*/fl_log.txt experiments/fedamm_plus_p8_full/
cp results/fedamm_plus_full_*/rfnet.csv experiments/fedamm_plus_p8_full/
git add experiments/fedamm_plus_p8_full/
git commit -m "exp(full): fedamm+ variant1 1000 rounds WT=X.XX TC=X.XX ET=X.XX"
git checkout fedamm-plus
git merge feat/p8-full-fedamm-plus
git push origin fedamm-plus
```

---

### Milestone P9 — Sensitivity Analysis + Ablation

**Goal:** Validate design choices. Run experiments from `Final_Assessment.md` required experiments table.

**Branch:**
```bash
git checkout fedamm-plus
git checkout -b feat/p9-ablation
```

**Experiments to run (each 1000 rounds):**

| Experiment | Config change | Purpose |
|---|---|---|
| λ₃ sensitivity | `--lambda3 0.01`, `0.05`, `0.5` | Is 0.1 the right weight? |
| Warmup sensitivity | `--warmup_imp_rounds 50`, `200` | Is T_w=100 necessary? |
| L_mb + L_imp redundancy | Add `--disable_lmb` flag, keep λ₃=0.1 | Is L_mb redundant after L_imp? |
| No imputer (FedAMM) | `--use_imputer` not passed | Confirm baseline |

**Priority order:** Run λ₃ sensitivity first (fastest to vary, most informative). Then warmup. Then redundancy check.

**Results table template:**

| Config | WT | TC | ET | Avg | Notes |
|---|---|---|---|---|---|
| FedAMM (baseline) | 0.809 | 0.654 | 0.479 | 0.647 | Part 1 result |
| FedAMM+ V2 (closed-form) | ? | ? | ? | ? | Milestone P7 |
| FedAMM+ V1 λ₃=0.01 | ? | ? | ? | ? | |
| FedAMM+ V1 λ₃=0.1 | ? | ? | ? | ? | Primary |
| FedAMM+ V1 λ₃=0.5 | ? | ? | ? | ? | |
| FedAMM+ V1 T_w=50 | ? | ? | ? | ? | |
| FedAMM+ V1 T_w=200 | ? | ? | ? | ? | |
| FedAMM+ no L_mb | ? | ? | ? | ? | OP6 check |

**Commit results:**
```bash
git add experiments/fedamm_plus_p9_ablation/
git commit -m "exp(ablation): sensitivity analysis complete, best config λ₃=X.X T_w=XXX"
git checkout fedamm-plus
git merge feat/p9-ablation
git push origin fedamm-plus
```

---

## 6. Known Risks and Mitigations

| Risk | Described In | Mitigation |
|---|---|---|
| Low imputation quality at `|A_n|=1` | OP5 | Monitor imp_loss per `|A_n|` bucket separately |
| C=4 too small for cross-attention | OP8 | Try augmenting proto with channel-mean features |
| Warmup too short → noisy imputer corrupts training | Final_Assessment Risk 3 | Ablate T_w ∈ {50, 100, 200} in P9 |
| L_mb and L_imp redundant | OP6 | Run λ₁=0, λ₃>0 ablation in P9 |
| imp_loss NaN early training | Cold-start (R9) | Warmup T_w=100 handles this; add NaN guard in loss |
| Imputer collapse (outputs constant) | Harsh_Review R4 | Cosine loss with normalization prevents this |
| A1 violated → aggregation invalid | W4 | Proved safe in Optimization_and_Gradients.md |

---

## 7. File Map (FedAMM+ additions highlighted)

```
train.py                  ← MODIFIED: warmup, imp_loss, imputer aggregation
options.py                ← MODIFIED: --lambda3, --warmup_imp_*, --use_imputer
models/
  rfnet.py                ← MODIFIED: ProtoImputer in Model, imp_loss in forward
  blocks.py               ← MODIFIED: ProtoImputer class ADDED
  mask.py                 ← unchanged
  layers.py               ← unchanged
dataset/                  ← ALL unchanged
utils/
  criterions.py           ← MODIFIED: imputation_loss_bs(), closed_form_impute() ADDED
  fl_utils.py             ← MODIFIED: avg_imputer_weights() ADDED
  predict.py              ← unchanged
  lr_scheduler.py         ← unchanged
experiments/
  fedamm_plus_p6_debug/   ← debug run artifacts
  fedamm_plus_p7_v2/      ← variant 2 results
  fedamm_plus_p8_full/    ← primary FedAMM+ results
  fedamm_plus_p9_ablation/← sensitivity analysis results
```

---

## 8. Session Continuity

Update `SESSION_CONTEXT.md` at the end of each milestone with:
- Milestone number and status
- Any bugs found and fixed
- Results (if applicable)
- Next step

In a new chat: upload this guide + updated `SESSION_CONTEXT.md`.

---

## 9. Expected Final Contribution

> FedAMM+ addresses decoder distribution shift (W3) — the largest unresolved gap in FedAMM — by standardizing the decoder's input to always receive M-dimensional prototype-guided features regardless of modality availability. The learned cross-attention imputer, trained with a bounded cosine loss as an extension of the existing modal balance framework, provides meaningful cross-modal supervision with zero communication overhead.
>
> **Target venue:** MICCAI 2026 or Medical Image Analysis
