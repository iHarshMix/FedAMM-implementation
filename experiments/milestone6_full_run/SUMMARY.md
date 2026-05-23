# Milestone 6 — Full Training Run Results

## Config
- Rounds: 1000, local_ep: 1, batch_size: 1
- lr: 2e-4, temp: 4.0, mask_type: idt
- device_ids: 0,1,0,1, use_multiprocessing: True
- Clients: 4, Dataset: BraTS2020 (259 train, 36 val)
- Dirichlet alpha: 0.01, seed: 1024

## Results (Avg across all 15 modality masks)

| Round | WT     | TC     | ET     | Avg    |
|-------|--------|--------|--------|--------|
| 100   | 0.696  | 0.529  | 0.351  | 0.525  |
| 200   | 0.762  | 0.603  | 0.408  | 0.591  |
| 300   | 0.788  | 0.629  | 0.433  | 0.617  |
| 400   | 0.794  | 0.643  | 0.450  | 0.629  |
| 500   | 0.802  | 0.658  | 0.461  | 0.640  |
| 600   | 0.806  | 0.651  | 0.462  | 0.640  |
| 700   | 0.810  | 0.659  | 0.475  | 0.648  |
| 800   | 0.805  | 0.653  | 0.476  | 0.645  |
| 900   | 0.809  | 0.657  | 0.474  | 0.647  |
| 1000  | 0.809  | 0.654  | 0.479  | 0.647  |

## Paper Targets
| WT ~0.84 | TC ~0.77 | ET ~0.71 |

## Notes
- Model converged around round 500-600
- Gap from paper likely due to smaller training set (259 vs ~335 cases)
- Bug fixed during run: group_cluster_and_select empty gt_data_cl crash at round 70
- ~46 sec/round on 2x H100 with multiprocessing
