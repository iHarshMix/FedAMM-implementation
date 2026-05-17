import argparse
import os

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--datapath', default=os.path.expanduser('~/datasets/BraTS/BRATS2020_Training_none_npy/'), type=str)
    parser.add_argument('--dataname', default='BRATS2020', type=str)
    parser.add_argument('--chose_modal', default='all', type=str)
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('--save_root', default='results', type=str)
    parser.add_argument('--reload_from_checkpoint', action='store_true', default=False, help='reload from checkpoint')
    parser.add_argument('--checkpoint_path', default='./results/debug', type=str)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--visualize', default=True)
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=42, type=int)


    # Settings
    parser.add_argument('--temp', default=4.0, type=float, help='knowledge-distillation temperature')
    parser.add_argument('--mask_type', default='idt', type=str, help='training settings: pdt idt or idt_drop')
    parser.add_argument('--round_per_train', type=int, default=3, help="validate the model per X rounds")
    parser.add_argument('--region_fusion_start_epoch', default=0, type=int, help='warm-up epochs used in rfnet')
    parser.add_argument('--use_multiprocessing', default=False, help='whether use multiprocessing')

    # FL Settings
    parser.add_argument('--gpus', default='0', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--c_rounds', type=int, default=300, help="number of rounds of training and communication")
    parser.add_argument('--start_round', type=int, default=0, help="number of rounds of training and communication")

    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--global_ep', type=int, default=1, help="the number of global epochs: E")
    parser.add_argument('--client_num', type=int, default=4, help="number of users: K")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    # files
    
    
    parser.add_argument('--train_file', type=dict, 
                default={ 
                1:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_1_imb.csv", 
                2:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_2_imb.csv", 
                3:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_3_imb.csv", 
                4:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_4_imb.csv"})
    parser.add_argument('--valid_file', type=str, default=os.path.expanduser("~/datasets/BraTS/BRATS2020_Training_none_npy/val.txt"))
    
    
    parser.add_argument('--test_file', type=str, default="./datalist/BRATS2020_Training_none_npy/test.txt")
    parser.add_argument("--device_ids", type=str, default='0,0,0,0')

    # 说明
    parser.add_argument('--version', type=str, default='debug', help='to explain the experiment set up')

    args = parser.parse_args()
    return args