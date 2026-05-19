import os

import medpy.io as medio
import numpy as np

# ── ONLY THESE TWO LINES CHANGED FROM ORIGINAL ──────────────────────────────
# src_path: points to the actual case folders inside the Kaggle download
# tar_path: points to our preprocessed output folder
currentdirPath = os.path.dirname(os.path.abspath(__file__))
relativePath = '../../datasets'
datarootPath = os.path.abspath(os.path.join(currentdirPath, relativePath))

src_path = os.path.join(datarootPath, 'BraTS/BRATS2020_Training_Data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
tar_path = os.path.join(datarootPath, 'BraTS/BRATS2020_Training_none_npy')
# ─────────────────────────────────────────────────────────────────────────────

name_list = os.listdir(src_path)

def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print ('#' * 100)
        ecart = int((128-(xmax-xmin))/2)
        xmax = xmax+ecart+1
        xmin = xmin-ecart
    if xmin < 0:
        xmax-=xmin
        xmin=0
    return xmin, xmax

def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max

def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol

if not os.path.exists(os.path.join(tar_path, 'vol')):
    os.makedirs(os.path.join(tar_path, 'vol'))

if not os.path.exists(os.path.join(tar_path, 'seg')):
    os.makedirs(os.path.join(tar_path, 'seg'))

skipped = []
processed = 0

for file_name in name_list:
    # ── try/except added to skip corrupt cases like BraTS20_Training_355 ────
    try:
        print(file_name)
        ### BraTS2020 Rename in RFNet
        num = file_name.split('_')[2]
        HLG = 'HG_' if int(num) <= 259 or int(num) >= 336 else 'LG_'

        # ── .nii.gz → .nii (Kaggle files are uncompressed) ──────────────────
        flair, flair_header = medio.load(os.path.join(src_path, file_name, file_name+'_flair.nii'))
        t1ce,  t1ce_header  = medio.load(os.path.join(src_path, file_name, file_name+'_t1ce.nii'))
        t1,    t1_header    = medio.load(os.path.join(src_path, file_name, file_name+'_t1.nii'))
        t2,    t2_header    = medio.load(os.path.join(src_path, file_name, file_name+'_t2.nii'))
        # ─────────────────────────────────────────────────────────────────────

        vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
        x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
        vol1 = normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
        vol1 = vol1.transpose(1,2,3,0)
        print(vol1.shape)

        # ── .nii.gz → .nii ──────────────────────────────────────────────────
        seg, seg_header = medio.load(os.path.join(src_path, file_name, file_name+'_seg.nii'))
        # ─────────────────────────────────────────────────────────────────────
        seg = seg.astype(np.uint8)
        seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
        seg1[seg1==4]=3

        ### BraTS2020 Rename in RFNet
        np.save(os.path.join(tar_path, 'vol', HLG+file_name+'_vol.npy'), vol1)
        np.save(os.path.join(tar_path, 'seg', HLG+file_name+'_seg.npy'), seg1)

        processed += 1

    except Exception as e:
        print(f"SKIPPING {file_name}: {e}")
        skipped.append(file_name)
        continue
    # ─────────────────────────────────────────────────────────────────────────

print(f"\nDone. Processed: {processed}, Skipped: {len(skipped)}")
if skipped:
    print(f"Skipped cases: {skipped}")
