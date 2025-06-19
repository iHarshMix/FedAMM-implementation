import pandas as pd
import torch.backends.cudnn as cudnn
import csv
import torch
import os
import random
import numpy as np
import time
import copy
from torch import nn
from tqdm import tqdm
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.data_utils import init_fn

from models import rfnet
from utils.fl_utils import avg_local_weights,avg_encoder_weights
from utils.lr_scheduler import LR_Scheduler
from utils import criterions
from dataset.datasets import Brats_test, Brats_train, GLB_Brats_train
from dataset.datasets_nii import Brats_loadall_train_nii_idt,Brats_loadall_test_nii,Brats_loadall_val_nii
from options import args_parser
from utils.predict import test_dice_softmax,AverageMeter

from multiprocessing import Pool

import setproctitle
setproctitle.setproctitle("donot use 0123 gpu!")

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))


def local_training(args, device_id, mask, dataloader, model, client_idx,round,cluster_centers,client_mask_id_proportions):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"Training on GPU{device_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    model.train()
    model = model.to(device)
    start = time.time()
    

    # Set Optimizer for the local model update
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
        optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    elif args.optimizer == 'adamw':
        train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
        optimizer = torch.optim.AdamW(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    
    # writer.add_scalar('lr_lc', step_lr, global_step=round)
    # logging.info('############# client_{} local training ############'.format(client_idx+1))

    iter_per_epoch = len(dataloader)
    train_iter = iter(dataloader)

    ### IDT Init-Imb-Weight Setting
    imb_mr_csv_data = pd.read_csv(args.train_file[client_idx+1])
    modal_num = torch.tensor((0,0,0,0), requires_grad=False).cuda().float()
    for sample_mask in imb_mr_csv_data['mask']:
        modal_num += torch.tensor(eval(sample_mask), requires_grad=False).cuda().float()        
    logging.info('Training Imperfect Datasets with Mod.Flair-{:d}, Mod.T1c-{:d}, Mod.T1-{:d}, Mod.T2-{:d}'\
    .format(int(modal_num[0].item()), int(modal_num[1].item()), int(modal_num[2].item()), int(modal_num[3].item())))
    
    modal_weight = torch.tensor((1,1,1,1), requires_grad=False).cuda().float()
    modal_weight = (iter_per_epoch/modal_num).cuda().float()
    imb_beta = torch.tensor((1,1,1,1), requires_grad=False).cuda().float()
    eta = 0.01
    eta_ext = 1.5

    for epoch in range(args.local_ep):
        step_lr = lr_schedule(optimizer, round)
        #writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        epoch_fuse_losses = torch.zeros(1).cpu().float()
        epoch_sep_losses = torch.zeros(1).cpu().float()
        epoch_prm_losses = torch.zeros(1).cpu().float()
        epoch_kl_losses = torch.zeros(1).cpu().float()
        epoch_proto_losses = torch.zeros(1).cpu().float()
        epoch_global_losses = torch.zeros(1).cpu().float()
        # epoch_dist_losses = torch.zeros(1).cpu().float()
        epoch_losses = torch.zeros(1).cpu().float()
        epoch_sep_m = torch.zeros(4).cpu().float()
        epoch_kl_m = torch.zeros(4).cpu().float()
        epoch_proto_m = torch.zeros(4).cpu().float()
        epoch_dist_m = torch.zeros(4).cpu().float()

        b = time.time()
        client_gt = []
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(dataloader)
                data = next(train_iter)
            x, target, mask, name = data
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask_id = masks_test.index(mask[0].tolist())
            mask = mask.cuda(non_blocking=True)

            model.is_training = True

            kl_loss_m = torch.zeros(4).cuda().float()
            sep_loss_m = torch.zeros(4).cuda().float()
            proto_loss_m = torch.zeros(4).cuda().float()
            dist_m = torch.zeros(4).cuda().float()
            prm_loss = torch.zeros(1).cuda().float()
            # fuse_loss = torch.zeros(1).cuda().float()
            rp_iter = torch.zeros(4).cuda().float()

            cluster_mask = tuple(mask.cpu().numpy().flatten().tolist())
            fuse_pred, prm_loss_bs, sep_loss_m_bs, kl_loss_m_bs, proto_loss_m_bs, dist_m_bs,gt = model(x, mask, target=target, temp=args.temp)
            
            client_gt.append((cluster_mask,gt.detach().cpu()))
            
            global_loss = torch.zeros(1).cuda().float()
            if cluster_centers and cluster_centers[cluster_mask] is not None:
                cluster_gt = cluster_centers[cluster_mask]
                cluster_gt = cluster_gt.cuda().float()
                global_loss = torch.mean((cluster_gt-gt)**2, dim=1)
                non_zero_mask = torch.any(gt != 0, dim=1)  # 找出第一个维度上不全为 0 的行
                global_loss = global_loss[non_zero_mask].sum()
                global_loss = global_loss * client_mask_id_proportions[mask_id]
                logging.info('global_loss:{:.4f}'.format(global_loss.item()))
            
            ###Loss compute
            # fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            # fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            # fuse_loss += fuse_cross_loss + fuse_dice_loss
            fuse_loss_bs = criterions.softmax_weighted_loss_bs(fuse_pred, target, num_cls=args.num_class) + criterions.dice_loss_bs(fuse_pred, target, num_cls=args.num_class)
            fuse_loss = torch.sum(fuse_loss_bs)
            prm_loss = torch.sum(prm_loss_bs)
            
            # masks_sum = torch.clamp(torch.sum(mask, dim=0).float(), min=0.005, max=args.batch_size)
            sep_loss_m = torch.sum(sep_loss_m_bs*mask, dim=0)
            kl_loss_m = torch.sum(kl_loss_m_bs*mask, dim=0)
            proto_loss_m = torch.sum(proto_loss_m_bs*mask, dim=0)
            dist_m = torch.sum(dist_m_bs*mask, dim=0)

            for bs in range(x.size(0)):
                dist_avg_bs = sum(dist_m_bs[bs])/sum(mask[bs])
                rp_iter += mask[bs]*(dist_m_bs[bs]/dist_avg_bs-1)
            rp_mask = rp_iter > 0

            kl_loss = (imb_beta * modal_weight * kl_loss_m).sum()
            proto_loss = (rp_mask * modal_weight * proto_loss_m).sum()
            # dist_loss = (rp_mask * imb_beta * modal_weight * dist_m).sum()

            ## warmup with shared sep-decoder like rfnet
            if round < args.region_fusion_start_epoch:
                sep_loss = (imb_beta * modal_weight * sep_loss_m).sum()
                loss = fuse_loss * 0.0 + sep_loss + prm_loss * 0.0 + kl_loss * 0.0 + proto_loss * 0.0
            else:
                sep_loss = (rp_mask * imb_beta * modal_weight * sep_loss_m).sum()
                loss = fuse_loss + sep_loss + prm_loss + kl_loss * 0.5 + proto_loss * 0.1 + global_loss

            # ## without warmup and without shared sep-decoder
            # sep_loss = (rp_mask * imb_beta * modal_weight * sep_loss_m).sum()
            # loss = fuse_loss + sep_loss * 0.0 + prm_loss + kl_loss * 0.5 + proto_loss * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses += (loss/iter_per_epoch).detach().cpu()
            epoch_fuse_losses += (fuse_loss/iter_per_epoch).detach().cpu()
            epoch_prm_losses += (prm_loss/iter_per_epoch).detach().cpu()
            epoch_sep_losses += (sep_loss/iter_per_epoch).detach().cpu()
            epoch_kl_losses += (kl_loss/iter_per_epoch).detach().cpu()
            epoch_proto_losses += (proto_loss/iter_per_epoch).detach().cpu()
            # epoch_dist_losses += (dist_loss/iter_per_epoch).detach().cpu()
            epoch_global_losses += (global_loss/iter_per_epoch).detach().cpu()

            if args.mask_type == 'idt':
                epoch_kl_m += (kl_loss_m/modal_num).detach().cpu()
                epoch_sep_m += (sep_loss_m/modal_num).detach().cpu()
                epoch_proto_m += (proto_loss_m/modal_num).detach().cpu()
                epoch_dist_m += (dist_m/modal_num).detach().cpu()
            else:
                epoch_kl_m += (kl_loss_m/iter_per_epoch).detach().cpu()
                epoch_sep_m += (sep_loss_m/iter_per_epoch).detach().cpu()
                epoch_proto_m += (proto_loss_m/iter_per_epoch).detach().cpu()
                epoch_dist_m += (dist_m/iter_per_epoch).detach().cpu()

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.local_ep, (i+1), iter_per_epoch, loss.item())
            msg += 'fuse_loss:{:.4f}, prm_loss:{:.4f}, '.format(fuse_loss.item(), prm_loss.item())
            msg += 'sep_loss:{:.4f}, '.format(sep_loss.item())
            msg += 'kl_loss:{:.4f}, proto_loss:{:.4f},'.format(kl_loss.item(), proto_loss.item())
            msg += 'seplist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(sep_loss_m[0].item(), sep_loss_m[1].item(), sep_loss_m[2].item(), sep_loss_m[3].item())
            msg += 'kllist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(kl_loss_m[0].item(), kl_loss_m[1].item(), kl_loss_m[2].item(), kl_loss_m[3].item())
            msg += 'protolist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(proto_loss_m[0].item(), proto_loss_m[1].item(), proto_loss_m[2].item(), proto_loss_m[3].item())
            msg += 'distlist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(dist_m[0].item(), dist_m[1].item(), dist_m[2].item(), dist_m[3].item())
            for bs_n in range(x.size(0)):
                msg += '{:>20}, '.format(name[bs_n])
            msg += 'kl_w[{:.2f},{:.2f},{:.2f},{:.2f}] '.format(modal_weight[0].item(), modal_weight[1].item(), modal_weight[2].item(), modal_weight[3].item())
            logging.info(msg)
        b_train = time.time()
        logging.info('train time per epoch: {}'.format(b_train - b))

        epoch_dist_avg = (sum(epoch_dist_m)/4.0).cpu().float()
        rp_epoch = ((epoch_dist_avg - epoch_dist_m) / epoch_dist_avg)
        if round < args.region_fusion_start_epoch:
            imb_beta = imb_beta.cuda()
        else:
            if round % 100 == 0:
                eta = eta * eta_ext
            imb_beta = imb_beta.cpu() - eta * rp_epoch
            imb_beta = torch.clamp(imb_beta, min=0.1, max=4.0)
            imb_beta = 2 * imb_beta / (sum(imb_beta**2)**(0.5))
            imb_beta = imb_beta.cuda()


        logging.info('epoch_global_losses:{:.4f}'.format(epoch_global_losses.item()))

        logging.info('rp_epoch:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(rp_epoch[0].item(), rp_epoch[1].item(), rp_epoch[2].item(), rp_epoch[3].item()))
        logging.info('imb_beta:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(imb_beta[0].item(), imb_beta[1].item(), imb_beta[2].item(), imb_beta[3].item()))

        epoch_loss = {'epoch_losses':epoch_losses, 'epoch_fuse_losses':epoch_fuse_losses, 'epoch_prm_losses':epoch_prm_losses, 'epoch_sep_losses':epoch_sep_losses,
                        'epoch_kl_losses':epoch_kl_losses, 'epoch_proto_losses':epoch_proto_losses, 'epoch_kl_m':epoch_kl_m, 'epoch_sep_m':epoch_sep_m,
                        'epoch_proto_m':epoch_proto_m, 'epoch_dist_m':epoch_dist_m, 'rp_epoch':rp_epoch, 'epoch_global_losses':epoch_global_losses,"lr":step_lr}
    
    msg = 'client_{} local training total time: {:.4f} hours'.format(client_idx+1, (time.time() - start)/3600)
    print(msg)
    logging.info(msg)
    model = model.cpu()
        
    return model,epoch_loss,client_gt


def log_client_train(writer, client_i, local_losses, round):
    
    writer.add_scalar('Client_{}/epoch_losses'.format(client_i+1), local_losses['epoch_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_fuse_losses'.format(client_i+1), local_losses['epoch_fuse_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_prm_losses'.format(client_i+1), local_losses['epoch_prm_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_sep_losses'.format(client_i+1), local_losses['epoch_sep_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_kl_losses'.format(client_i+1), local_losses['epoch_kl_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_proto_losses'.format(client_i+1), local_losses['epoch_proto_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_global_losses'.format(client_i+1), local_losses['epoch_global_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/lr'.format(client_i+1), local_losses['lr'].item(), global_step=round)
    for m in range(4):
        writer.add_scalar('Client_{}/kl_m{}'.format(client_i+1, m), local_losses['epoch_kl_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/sep_m{}'.format(client_i+1, m), local_losses['epoch_sep_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/proto_m{}'.format(client_i+1, m), local_losses['epoch_proto_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/dist_m{}'.format(client_i+1, m), local_losses['epoch_dist_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/rp_m{}'.format(client_i+1, m), local_losses['rp_epoch'][m].item(), global_step=round)
    
def uploadLCweightsandGLBupdate(server_model,local_weights,client_mask_proportions_sum,client_modal_weight,model_clients):
    glb_w = avg_local_weights(local_weights[0], local_weights[1], local_weights[2], local_weights[3],client_mask_proportions_sum)
    server_model.load_state_dict(glb_w)
    flair_encoder = avg_encoder_weights(model_clients[0].flair_encoder.state_dict(), model_clients[1].flair_encoder.state_dict(), model_clients[2].flair_encoder.state_dict(), model_clients[3].flair_encoder.state_dict(),client_modal_weight.T[0])
    t1ce_encoder = avg_encoder_weights(model_clients[0].t1ce_encoder.state_dict(), model_clients[1].t1ce_encoder.state_dict(), model_clients[2].t1ce_encoder.state_dict(), model_clients[3].t1ce_encoder.state_dict(),client_modal_weight.T[1])
    t1_encoder = avg_encoder_weights(model_clients[0].t1_encoder.state_dict(), model_clients[1].t1_encoder.state_dict(), model_clients[2].t1_encoder.state_dict(), model_clients[3].t1_encoder.state_dict(),client_modal_weight.T[2])
    t2_encoder = avg_encoder_weights(model_clients[0].t2_encoder.state_dict(), model_clients[1].t2_encoder.state_dict(), model_clients[2].t2_encoder.state_dict(), model_clients[3].t2_encoder.state_dict(),client_modal_weight.T[3])
    server_model.flair_encoder.load_state_dict(flair_encoder)
    server_model.t1ce_encoder.load_state_dict(t1ce_encoder)
    server_model.t1_encoder.load_state_dict(t1_encoder)
    server_model.t2_encoder.load_state_dict(t2_encoder)
    return server_model

def downloadGLBweights(server_model, model_clients):
    for i in range(len(model_clients)):
        model_clients[i].load_state_dict(server_model.state_dict())
    return model_clients

if __name__ == '__main__':

    args = args_parser()
    
    args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    args.save_path = args.save_root + '/' + str(args.version)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    args.modelfile_path = os.path.join(args.save_path, 'model_files')
    if not os.path.exists(args.modelfile_path):
        os.makedirs(args.modelfile_path)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=args.save_path + '/fl_log.txt')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
    
    writer = SummaryWriter(os.path.join(args.save_path, 'TBlog'))
    
    ########## setting seed for deterministic
    if args.deterministic:
        # cudnn.enabled = False
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########## setting device and gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device_ids = list(map(int,args.device_ids.split(',')))
    args.local_devices = args.device_ids

    ########## setting global and local model
    server_model = rfnet.Model(num_cls=args.num_class)
    server_model.mask_type = args.mask_type
    best_dices = [0.0, 0.0, 0.0, 0.0]
    best_dice = 0.0
    cluster_centers = None  
    if args.reload_from_checkpoint:
        ckpt = torch.load(args.checkpoint_path + '/last.pth')
        server_model.load_state_dict(ckpt["server"])
        args.start_round = ckpt['round']
        best_dice = ckpt['best_dice']
        best_dices = ckpt['best_dices']
        cluster_centers = ckpt['cluster_centers']
        print("load best result: {}, {}, {}, {}.".format(best_dice, best_dices[0], best_dices[1], best_dices[2]))


    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    ########## FL setting ##########
    # define dataset, model, optimizer for each clients 
    dataloader_clients, validloader_clients, testloader_clients = [], [], []
    model_clients = []
    optimizer_clients = []
    client_counts, client_weights = [], []     ### FedAvg Setting
    modal_list = ['flair', 't1ce', 't1', 't2']
    logging.info(str(args))
    client_modal_weight = []
    mask_id_count_list = []
    client_mask_id_proportions = []
    for client_idx in range(args.client_num):
        lc_train_file = args.train_file[client_idx+1]
        data_set = Brats_loadall_train_nii_idt(transforms=args.train_transforms,root=args.datapath, num_cls=args.num_class, train_file=lc_train_file)
        data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
        dataloader_clients.append(data_loader)
        logging.info('Client-{} : the length of Brats train_dataset is {}'.format(client_idx+1, len(data_set)))
        net = copy.deepcopy(server_model)   # .to(device)  # .to(args.device)
        model_clients.append(net)
    ####15种mask的分布
        imb_mr_csv_data = pd.read_csv(args.train_file[client_idx+1])

        clinet_modal_num = np.zeros(4, dtype=np.float32)  # 初始化为零的NumPy数组
        for sample_mask in imb_mr_csv_data['mask']:
            clinet_modal_num += np.array(eval(sample_mask), dtype=np.float32)  # 将字符串转换为NumPy数组并累加
        client_modal_weight.append(clinet_modal_num)

        mask_id_count = [0] * 15
        for mask_id in imb_mr_csv_data['mask_id']:
            mask_id_count[mask_id] += 1
        logging.info('Mask ID Count: {}'.format(mask_id_count))
        mask_id_count_list.append(mask_id_count)
    total = [sum(x) for x in zip(*mask_id_count_list)]
    for mask_id_count in mask_id_count_list:
        client_mask_id_proportions.append([mask_id_count[i] / total[i] if total[i] != 0 else 0 for i in range(len(mask_id_count))])
    client_mask_proportions_sum = [sum(client_mask_id_proportions[i])/len(client_mask_id_proportions[i]) for i in range(len(client_mask_id_proportions))]
    logging.info(f'client_mask_proportions_sum: {client_mask_proportions_sum}')
    logging.info(f'client_mask_id_proportions: {client_mask_id_proportions}')
    ####15种mask的分布
    
    client_modal_weight = np.array(client_modal_weight)
    all_client_modal_weight = client_modal_weight.sum(axis=0)
    all_client_modal_weight = np.where(all_client_modal_weight == 0, 1, all_client_modal_weight)
    client_modal_weight = client_modal_weight / all_client_modal_weight

    logging.info(f'client_modal_weight: {client_modal_weight}')


    valid_set = Brats_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, train_file=args.valid_file)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)
    
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=args.valid_file)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=3, pin_memory=True)
    logging.info('the length of Brats dataset is {} : {}'.format(len(valid_set), len(test_set)))

    #validloader_clients.append(valid_loader)
    #testloader_clients.append(test_loader)
          
    ########## FL Training ##########
    for round in tqdm(range(args.start_round, args.c_rounds+1)):
        start = time.time()
        ##### local training
        local_weights, local_losses, local_protos = [], {}, {}
        logging.info(f'\n | Global Training Round : {round} |')
        start = time.time()

        result = []
        torch.cuda.empty_cache()
        if args.use_multiprocessing:
            ctx = torch.multiprocessing.get_context("spawn")
            pool = ctx.Pool(args.client_num)
        for client_i in range(4):
            if args.use_multiprocessing:
                result.append(pool.apply_async(local_training, args=(args, args.local_devices[client_i], masks_torch[client_i], dataloader_clients[client_i], model_clients[client_i], client_i,round,cluster_centers,client_mask_id_proportions[client_i])))
            else:
                result.append(local_training(args, args.local_devices[client_i], masks_torch[client_i], dataloader_clients[client_i], model_clients[client_i], client_i, round,cluster_centers,client_mask_id_proportions[client_i]))
        if args.use_multiprocessing:
            pool.close()
            pool.join()
        
        logging.info("local client training: {}".format(time.time() - start))
        client_gt_list = []
        for client_i, i in enumerate(result):
            if args.use_multiprocessing:
                m, loss,client_gt = i.get()
            else:
                m, loss,client_gt = i
            local_weights.append(copy.deepcopy(m.state_dict()))
            local_losses = copy.deepcopy(loss)
            model_clients[client_i] = m              
            log_client_train(writer, client_i, local_losses, round)
            client_gt_list+=client_gt


        #cluster_centers = criterions.cluster_and_select(client_gt_list)
        new_cluster_centers = criterions.group_cluster_and_select(client_gt_list,masks_test)
        cluster_centers = criterions.EMA_cls_Fs(cluster_centers, new_cluster_centers)
        #criterions.test_clustering(client_gt_list, cluster_centers)    

        # global Aggre and Fusion
        server_model = uploadLCweightsandGLBupdate(server_model,local_weights,client_mask_proportions_sum,client_modal_weight,model_clients)
        downloadGLBweights(server_model, model_clients)
        ##### Eval the model after aggregation and 10 round
        if (round+1)%args.round_per_train == 0:# and round>200:
            logging.info('-'*20 + 'Test All the Models per 10 round'+ '-'*20)
            test_dice_score = AverageMeter()
            #test_hd95_score = AverageMeter()
            csv_name = os.path.join(args.save_path, '{}.csv'.format('rfnet'))
            with torch.no_grad():
                file = open(csv_name, "a+")
                csv_writer = csv.writer(file)
                csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95' 'ETPro HD95'])
                file.close()
                for i, mask in enumerate(masks_test[::-1]):
                    logging.info('{}'.format(mask_name[::-1][i]))
                    file = open(csv_name, "a+")
                    csv_writer = csv.writer(file)
                    csv_writer.writerow([mask_name[::-1][i]])
                    file.close()
                    dice_score, class_evaluation = test_dice_softmax(
                                    test_loader,
                                    server_model,
                                    dataname = args.dataname,
                                    feature_mask = mask,
                                    mask_name = mask_name[::-1][i],
                                    csv_name = csv_name,
                                    device = args.device
                                    )
                    for clev in range(len(class_evaluation)):
                        writer.add_scalar('{}/Eval_dice_{}'.format(mask_name[::-1][i], class_evaluation[clev]), dice_score[clev], round)
                        #writer.add_scalar('Eval_hd95_{}_{}'.format(mask_name[::-1][i], class_evaluation[clev]), hd95_score[clev], round)
                    test_dice_score.update(dice_score)
                    #test_hd95_score.update(hd95_score)

                logging.info('Avg Dice scores: {} avg:{}'.format(test_dice_score.avg,(test_dice_score.avg[0]+test_dice_score.avg[1]+test_dice_score.avg[2])/3))
                #logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))
                for clev in range(len(class_evaluation)):
                    writer.add_scalar('Eval_AvgDice/{}'.format(class_evaluation[clev]), test_dice_score.avg[clev], round)

        logging.info('*'*10+'FL train a round total time: {:.4f} hours'.format((time.time() - start)/3600)+'*'*10)
        #for c in range(args.client_num):
        #    print("bbbbbbbbbbbbbbb", model_clients[c].decoder_fuse.d3_c1.conv.weight[10,10,1,1])
        if (round+1)%args.round_per_train == 0:
            torch.save({
            'round': round + 1,
            'server': server_model.state_dict(),
            'cluster_centers': cluster_centers,
            'best_dice': best_dice,
            'best_dices': best_dices
            }, args.modelfile_path + '/last.pth')
            
    writer.close()    