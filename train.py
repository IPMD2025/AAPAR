import os
import os
import pprint
from collections import OrderedDict, defaultdict
import pickle

import numpy as np
import torch
from torch import nn,optim
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from loss.cross_entropy_loss import *
from loss.hard_mine_triplet_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler,make_scheduler
from tools.function import get_model_log_path
from clipS import clip
from clipS.model import *
import random
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
# part_order=[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18],[19,20,21,22,23,24,25]]
part_order=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24],[25,26,27,28,29],[30,31,32,33,34]]
set_seed(605)

device = "cuda" if torch.cuda.is_available() else "cpu"
ViT_model, ViT_preprocess = clip.load("ViT-L/14", device=device,download_root='/media/backup/**/pretrained/') 
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
    des_num= train_set.des_num
    model = TransformerClassifier(args,train_set.attr_num,train_set.label_word,des_num)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    # criterion_p = []
    # for i in range(len(part_order)):
    #     criterion_p.append(CEL_Sigmoid(sample_weight[part_order[i]],attr_idx=len(part_order[i])))
    # sample_weight1 = torch.ones(train_set.attr_num) / train_set.attr_num
    # sample_weight1 = sample_weight1.cpu().numpy()
    # criterion_clip=CEL_Sigmoid(sample_weight1, attr_idx=train_set.attr_num)
    criterion_clip=CrossEntropyLoss(num_classes=des_num)#
    # criterion_clip=TripletLoss(margin=0.3)
    # clip_params=[]
    # for name, param in ViT_model.named_parameters():
    #     if any(keyword in name for keyword in args.clip_update_parameters):
    #         print(name, param.requires_grad)
    #         clip_params+= [{
    #         "params": [param],
    #         "lr": args.clip_lr,
    #         "weight_decay": args.clip_weight_decay
    #         }]
    #     else:
    #         param.requires_grad = False
    clip_params=[]
    for name, param in ViT_model.named_parameters():
        if any(keyword in name for keyword in args.clip_update_parameters):
            print(name, param.requires_grad)
            clip_params.append(param)
        else:
            param.requires_grad = False
    
    # parameters_other=[]
    # for name, param in model.named_parameters():
    #     if any(keyword in name for keyword in ["blocks_t"]):
    #         param.requires_grad = False
    #         print(name, param.requires_grad)
    #         # parameters_c.append(param)
    #     else:
    #         parameters_other.append(param)
    
    # prompt_optimizer = optim.SGD(clip_params, args.clip_lr, momentum=0.9, weight_decay=args.clip_weight_decay)
    # prompt_scheduler = make_scheduler(prompt_optimizer,num_epochs=args.epoch,lr=args.clip_lr,warmup_t=10)
    
    lr = args.lr
    epoch_num = args.epoch

    # optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW([{'params':params} for params in clip_params]+[{'params':model.parameters()}], lr=lr, weight_decay=args.weight_decay)#{'params':params} for params in parameters_other
    # optimizer = optim.AdamW([{'params':clip_params[0]},{'params':model.parameters()}],lr=lr, weight_decay=args.weight_decay)#,{'params':clip_params[1]},{'params':clip_params[2]},
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)
    

    best_metric, epoch,best_metric_ma, epoch_ma,best_metric_f1, epoch_f1 = trainer(epoch=epoch_num,
                                 model=model,
                                 ViT_model=ViT_model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 criterion_c=criterion_clip,
                                #  criterion_p = criterion_p,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 des_non=train_set.des_non,
                                #  prompt_scheduler=prompt_scheduler,
                                #  prompt_optimizer=prompt_optimizer,
                                 path=save_model_path,
                                 a=args.a)
    if args.local_rank == 0:
        print(f'vit,  best_metrc : {best_metric} in epoch{epoch}')
        print(f'vit,  best_metrc : {best_metric_ma} in epoch{epoch_ma}')
        print(f'vit,  best_metrc : {best_metric_f1} in epoch{epoch_f1}')
    
def trainer(epoch, model,ViT_model,train_loader, valid_loader, criterion, criterion_c,optimizer, scheduler, des_non,path,a):#criterion_p,,prompt_scheduler,prompt_optimizer
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
    for i in range(1, epoch+1):
        scheduler.step(i)
        # prompt_scheduler.step(i)
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            criterion=criterion,
            criterion_c=criterion_c,
            optimizer=optimizer,
            des_non=des_non,
            a=a
            # group_arr=group_index
        )#, prompt_optimizer=prompt_optimizer

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            ViT_model=ViT_model,
            valid_loader=valid_loader,
            criterion=criterion,
            # group_arr=group_index
            #criterion_p=criterion_p
        )
        
        valid_result,_ = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print('-' * 60)
        # if i % args.epoch_save_ckpt:
        #     save_ckpt(model, os.path.join(path, f'ckpt_{time_str()}_{i}.pth'), i, valid_result)

        cur_metric = valid_result.ma + valid_result.instance_f1
        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            
        
        cur_metric = valid_result.ma
        if cur_metric > maximum_ma:
            maximum_ma = cur_metric
            best_epoch_ma = i
            # save_ckpt(model, path, i, maximum)

        cur_metric = valid_result.instance_f1
        if cur_metric > maximum_f1:
            maximum_f1 = cur_metric
            best_epoch_f1 = i
            save_ckpt(model,ViT_model, path, i, maximum)#,group_index

        result_list[i] = {
            # 'train_result': train_result,  # 'train_map': train_map,
            'valid_result': valid_result,  # 'valid_map': valid_map,
            'train_gt': train_gt, 'train_probs': train_probs,
            'valid_gt': valid_gt, 'valid_probs': valid_probs,
            # 'train_imgs': train_imgs, 'valid_imgs': valid_imgs
        }


        with open(result_path, 'wb') as f:
            pickle.dump(result_list, f)

    return maximum, best_epoch,maximum_ma, best_epoch_ma,maximum_f1, best_epoch_f1


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()