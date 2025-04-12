import os
import os
import pprint
from collections import OrderedDict, defaultdict
import pickle

import numpy as np
import torch
from torch import nn,optim
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer,show_cam
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics,get_reload_weight
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from tools.function import get_model_log_path
from clipS import clip
from clipS.model import *


# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

set_seed(605)
part_order=[[30,31,32,33,34],[0,1,2,3,4],[25,26,28,29,27,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]]
part_words=[[0,1,2,3,16],[10,18,19,30,15],[4,5,20,22,17,7,9,11,14,21,26,29,32,33,34],[6,8,12,25,27,31,13,23,24,28]]
group_num_start=[0,5,10,25,35]
# device = "cuda" if torch.cuda.is_available() else "cpu"
# ViT_model, ViT_preprocess = clip.load("ViT-L/14", device=device,download_root='/media/backup/**/pretrained/') 
def main(args):
    log_dir = os.path.join('logs', args.dataset)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, 'vit')
    save_model_path = os.path.join(model_dir, f'ckpt_max_{time_str()}.pth')
    
    
    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    select_gpus(args.gpus)
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    model = TransformerClassifier(args,train_set.attr_num,attr_words=train_set.label_word)
    if args.reload:
        model,ViT_model = get_reload_weight(model_dir, model,pth='ckpt_max_0506-clipclsmaskprompt-same-vitgppluspcam-lr3e5-20ep-gpmaskprompt-plusP.pth')

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    
    clip_params=[]
    for name, param in ViT_model.named_parameters():
        if any(keyword in name for keyword in args.clip_update_parameters):
            print(name, param.requires_grad)
            clip_params.append(param)
        else:
            param.requires_grad = False

    # prompt_optimizer = optim.SGD(clip_params, args.clip_lr, momentum=0.9, weight_decay=args.clip_weight_decay)
    # prompt_scheduler = make_scheduler(prompt_optimizer,num_epochs=args.epoch,lr=args.clip_lr,warmup_t=10)
    
    lr = args.lr
    epoch_num = args.epoch

    # optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW([{'params':clip_params[0]},{'params':model.parameters()}],lr=lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)
    # scheduler = CosineAnnealingWarmupRestarts(
    #         optimizer,
    #         first_cycle_steps=3026,
    #         cycle_mult=1.0,
    #         max_lr=lr,
    #         min_lr=0,
    #         warmup_steps=100
    #     )

    best_metric, epoch_i,epoch_j,epoch_k,best_metric_ma, epoch_ma_i,epoch_ma_j,epoch_ma_k,best_metric_f1, epoch_f1_i,epoch_f1_j,epoch_f1_k = valider(epoch=epoch_num,
                                 model=model,
                                 ViT_model=ViT_model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 #criterion_p=criterion_part,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 path=save_model_path)

    if args.local_rank == 0:
        print(f'vit,  best_metrc : {best_metric} in epoch{epoch_i},{epoch_j},{epoch_k}')
        print(f'vit,  best_metrc : {best_metric_ma} in epoch{epoch_ma_i},{epoch_ma_j},{epoch_ma_k}')
        print(f'vit,  best_metrc : {best_metric_f1} in epoch{epoch_f1_i},{epoch_f1_j},{epoch_f1_k}')
    



def valider(epoch, model,ViT_model,train_loader, valid_loader, criterion, optimizer, scheduler, path):#criterion_p,
    maximum = float(-np.inf)
    best_epoch = 0
    maximum_ma = float(-np.inf)
    best_epoch_ma = 0
    maximum_f1 = float(-np.inf)
    best_epoch_f1 = 0

    result_list = defaultdict()

    result_path = path
    result_path = result_path.replace('ckpt_max', 'metric')
    result_path = result_path.replace('pth', 'pkl')
    show_cam(model=model,ViT_model=ViT_model,valid_loader=valid_loader,criterion=criterion)
    # # for i in range(6):
    # #     for j in range(6):
    # #         for k in range(6):
    
    # # for k in range(1,11):        
    # valid_loss, valid_gt, valid_probs,valid_loss_g,valid_loss_p,valid_loss_patch = valid_trainer(
    #     model=model,
    #     ViT_model=ViT_model,
    #     valid_loader=valid_loader,
    #     criterion=criterion,
    #     # c1=i/5.0,
    #     # c2=j/5.0,
    #     # c3=k/5.0
        
    #     #criterion_p=criterion_p
    # )#

    # valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
    # i,j,k=1,1,1
    # # i,j=1,1

    # print(f'Evaluation on test set, \n','i:{:.1f},j:{:.1f},k:{:.1f}\n'.format(i,j,k),
    #         'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
    #             valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
    #         'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
    #             valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
    #             valid_result.instance_f1))
    # print(f'loss_g:{valid_loss_g:.4f}',f'loss_p:{valid_loss_p:.4f}',f'loss_patch:{valid_loss_patch:.4f}',f'train_loss:{valid_loss:.4f}')#,

    # cur_metric = valid_result.ma + valid_result.instance_f1
    # if cur_metric > maximum:
    #     maximum = cur_metric
    #     best_epoch_i = i
    #     best_epoch_j = j
    #     best_epoch_k = k
        
    
    # cur_metric = valid_result.ma
    # if cur_metric > maximum_ma:
    #     maximum_ma = cur_metric
    #     best_epoch_ma_i = i
    #     best_epoch_ma_j = j
    #     best_epoch_ma_k = k

    # cur_metric = valid_result.instance_f1
    # if cur_metric > maximum_f1:
    #     maximum_f1 = cur_metric
    #     best_epoch_f1_i = i
    #     best_epoch_f1_j = j
    #     best_epoch_f1_k = k

                       
    maximum, best_epoch_i,best_epoch_j,best_epoch_k,maximum_ma, best_epoch_ma_i,best_epoch_ma_j,best_epoch_ma_k,maximum_f1, best_epoch_f1_i,best_epoch_f1_j,best_epoch_f1_k=0,0,0,0,0,0,0,0,0,0,0,0
    return maximum, best_epoch_i,best_epoch_j,best_epoch_k,maximum_ma, best_epoch_ma_i,best_epoch_ma_j,best_epoch_ma_k,maximum_f1, best_epoch_f1_i,best_epoch_f1_j,best_epoch_f1_k


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()