import torch.nn.functional as F
import torch
import logging
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np

__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss', 'dice_loss']

cross_entropy = F.cross_entropy


def dice_loss_bs(output, target, num_cls=5, eps=1e-7, up_op=None):
    target = target.float()
    if up_op:
        output = up_op(output)
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:], dim=(1,2,3))
        l = torch.sum(output[:,i,:,:,:], dim=(1,2,3))
        r = torch.sum(target[:,i,:,:,:], dim=(1,2,3))
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
        dice_loss = (1.0 - 1.0 * dice / num_cls).unsqueeze(1)
    return dice_loss

def softmax_weighted_loss_bs(output, target, num_cls=5, up_op=None):
    target = target.float()
    if up_op:
        output = up_op(output)
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss, dim=(1,2,3)).unsqueeze(1)
    return cross_loss

def temp_kl_loss_bs(logit_s, logit_t, target, num_cls=5, temp=1.0, up_op=None):
    pred_s = F.softmax(logit_s/temp, dim=1)
    pred_t = F.softmax(logit_t/temp, dim=1)
    if up_op:
        pred_s = up_op(pred_s)
        pred_t = up_op(pred_t)
    pred_s = torch.clamp(pred_s, min=0.005, max=1)
    pred_t = torch.clamp(pred_t, min=0.005, max=1)
    pred_s = torch.log(pred_s)
    kl_loss = temp * temp * torch.mul(pred_t, torch.log(pred_t)-pred_s)
    kl_loss = torch.mean(kl_loss, dim=(1,2,3,4)).unsqueeze(1)
    return kl_loss

def prototype_loss_bs(feature_s, feature_t, target, logit_s, logit_t, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    N = len(feature_s.size()) - 2
    s = []
    t = []
    st = []
    logit_ss = []
    logit_tt = []
    proto_fs = torch.zeros_like(feature_s).cuda().float()

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_fs += proto_s[:,:,None,None,None] * targeti[:,None]
            proto_map_s = F.cosine_similarity(feature_s,proto_s[:,:,None,None,None],dim=1,eps=eps)
            proto_map_t = F.cosine_similarity(feature_t,proto_t[:,:,None,None,None],dim=1,eps=eps)
            s.append(proto_map_s.unsqueeze(1))
            t.append(proto_map_t.unsqueeze(1))
            logit_ss.append(logit_s[:, i, :, :, :].unsqueeze(1))
            logit_tt.append(logit_t[:, i, :, :, :].unsqueeze(1))
    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_st = F.cosine_similarity(proto_fs,proto_t[:,:,None,None,None],dim=1,eps=eps)
            st.append(proto_map_st.unsqueeze(1))
    sim_map_s = torch.cat(s,dim=1)
    sim_map_t = torch.cat(t,dim=1)
    proto_loss = torch.mean((sim_map_s-sim_map_t)**2, dim=(1,2,3,4)).unsqueeze(1)

    dist = torch.mean(torch.sqrt((sim_map_s-sim_map_t)**2), dim=(1,2,3,4)).unsqueeze(1)

    return proto_loss, dist


def prototype_pmr_loss(feature_s, feature_t, target, logit_s, logit_t, num_cls=5, temp=1.0, up_op=None):
    target = target.float()
    eps = 1e-5
    N = len(feature_s.size()) - 2
    ss = []
    gt = []

    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        if (torch.sum(targeti,dim=(-3,-2,-1))>0).all():
            proto_s =  torch.sum(feature_s*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
            proto_map_ss = -torch.sqrt(torch.sum((feature_s-proto_s[:,:,None,None,None])**2, dim=1))
            ss.append(proto_map_ss.unsqueeze(1))
            gt.append(targeti[:,None])

    softmax_s = torch.nn.Softmax(dim=1)(torch.cat(ss,dim=1))
    gt = torch.cat(gt,dim=1)

    proto_distri_s = torch.sum(softmax_s*gt, dim=1)
    proto_loss = torch.mean(-torch.log(torch.clamp(proto_distri_s, min=0.005, max=1)))
    kl_loss = torch.mean(proto_distri_s)

    return proto_loss, kl_loss


def gt_prototype(feature_t, target, num_cls=4):
    eps = 1e-5
    gt = []
    for i in range(num_cls):
        targeti = target[:, i, :, :, :]
        proto_t =  torch.sum(feature_t*targeti[:,None],dim=(-3,-2,-1))/(torch.sum(targeti[:,None],dim=(-3,-2,-1))+eps)
        gt.append(proto_t)
    gt = torch.cat(gt,dim=0)
    return gt


def cluster_and_select(client_gt_list):
    gt_data = [item[1].numpy().astype('float32') for item in client_gt_list]
    cluster_centers_cl = []
    for cl in range(4):
        gt_data_cl = [item[cl] for item in gt_data]
        kmeans = KMeans(n_clusters=15, random_state=0).fit(gt_data_cl)
        
        cluster_centers = []
        for i in range(15):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            cluster_data = [client_gt_list[j] for j in cluster_indices]
            
            masks = [item[0] for item in cluster_data]
            masks = [tuple(item[0].numpy().flatten().tolist()) for item in cluster_data]

            most_common_mask = max(set(masks), key=masks.count)
            
            filtered_data = [item for item in cluster_data if tuple(item[0].numpy().flatten().tolist()) == most_common_mask]
            
            if not filtered_data:
                continue
            
            sub_gt_data = [item[1][cl].numpy().astype('float32') for item in filtered_data]
            sub_kmeans = KMeans(n_clusters=1, random_state=0).fit(sub_gt_data)
            
            cluster_centers.append((most_common_mask, sub_kmeans.cluster_centers_[0]))
        cluster_centers_cl.append(cluster_centers)
    
    return cluster_centers_cl

def test_clustering(client_gt_list, cluster_centers_cl):
    correct = 0
    total = 0
    for cl in range(4):
        for item in client_gt_list:
            mask, gt = item
            mask = tuple(mask.numpy().flatten().tolist())
            min_distance = float('inf')
            closest_center = None
            for center in cluster_centers_cl[cl]:
                distance = np.linalg.norm(gt[cl] - center[1])
                if distance < min_distance:
                    min_distance = distance
                closest_center = center
        
            if mask == closest_center[0]:
                correct += 1
            total += 1
        
        accuracy = correct / total
        print(f'class {cl} accuracy: {accuracy:.2f}')
    return accuracy


def group_cluster_and_select(client_gt_list, masks_test):
    mask_groups = {tuple(mask): [] for mask in masks_test}
    for mask_key, gt in client_gt_list:
        if mask_key in mask_groups:
            mask_groups[mask_key].append(gt.numpy().astype('float32'))
    
    cluster_centers_dict = {tuple(mask): [] for mask in masks_test}
    for mask, gt_data in mask_groups.items():
        cluster_centers = []
        
        if len(gt_data) < 1:
            cluster_centers_dict[mask] = None
            continue
        
        skip = False
        for cl in range(4):
            gt_data_cl = [item[cl] for item in gt_data if not np.all(item[cl] == 0)]
            if len(gt_data_cl) == 0:
                cluster_centers_dict[mask] = None
                skip = True
                break
            n_clusters = 1
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gt_data_cl)
            cluster_centers.append(kmeans.cluster_centers_[0])
        if skip:
            continue
        clu_cl = np.stack(cluster_centers, axis=0)
        clu_cl = torch.from_numpy(clu_cl)
        cluster_centers_dict[mask] = clu_cl
    
    return cluster_centers_dict

def EMA_cls_Fs(prior_Fs, glb_protos):
    if prior_Fs is None:
        return glb_protos
    alpha = 0.999
    for key, value in prior_Fs.items():
        if value is None or glb_protos.get(key) is None:
            continue
        glb_Fs = alpha * value.numpy() + (1-alpha) * glb_protos[key].numpy()
        glb_protos[key] = torch.from_numpy(glb_Fs)
    return glb_protos


# ── FedAMM+ ────────────────────────────────────────────────────────────────────

def imputation_loss_bs(imputed_protos, teacher_proto, avail_mask, num_cls=4, eps=1e-8):
    """
    Cosine imputation loss L_imp (FedAMM+)

    For each missing modality m, measures cosine distance between:
    - imputed prototype P_hat_{n,c}^m  (from ProtoImputer)
    - stop-gradient teacher prototype sg(P_{n,c}^t)

    L_imp = (1/|A_bar_n|) * sum_{m not in A_n} [1 - cosine(P_hat^m, sg(P^t))]

    The teacher prototype is already in [B, C] form (extracted by
    _extract_teacher_proto in rfnet.py, which pools decoder features
    per class and averages channel-wise).

    Args:
        imputed_protos: dict {modal_idx: [B, C]} -- imputer outputs for missing modals
        teacher_proto:  [B, C] -- multimodal teacher prototype
        avail_mask:     [B, M] boolean tensor -- which modalities are available
        num_cls:        C = prototype dimension (= number of seg classes = 4)
        eps:            numerical stability for cosine_similarity

    Returns:
        loss: [B, 1] -- per-sample imputation loss, values in [0, 2]
    """
    B = avail_mask.shape[0]
    device = avail_mask.device

    if len(imputed_protos) == 0:
        # All modalities present — no imputation needed this batch
        return torch.zeros(B, 1, device=device)

    # Count how many modalities are missing per sample: [B]
    num_missing = (~avail_mask).float().sum(dim=1)      # [B]
    num_missing = torch.clamp(num_missing, min=1.0)     # avoid div-by-zero when all present

    total_loss = torch.zeros(B, device=device)

    # Stop-gradient on teacher: gradients must NOT flow back through the teacher.
    # This prevents the teacher (fused decoder features) from collapsing toward
    # the imputed student output.
    p_t = teacher_proto.detach()                        # [B, C]

    for m, p_hat in imputed_protos.items():
        # p_hat: [B, C] — imputed prototype for modality m

        # Cosine similarity in prototype space: [B]
        cos_sim = F.cosine_similarity(p_hat, p_t, dim=-1, eps=eps)

        # Cosine distance (bounded [0, 2]): [B]
        cos_dist = 1.0 - cos_sim

        # Apply only to samples where modality m IS actually missing.
        # ProtoImputer already skips present modalities, but this mask
        # makes the function safe if imputed_protos contains extra keys.
        missing_m = (~avail_mask[:, m]).float()         # [B]
        total_loss = total_loss + cos_dist * missing_m

    # Normalize by number of missing modalities per sample
    loss = total_loss / num_missing                     # [B]
    loss = loss.unsqueeze(1)                            # [B, 1]

    return loss
