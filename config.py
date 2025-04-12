import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="RAP")
    
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=40)#100
    parser.add_argument("--height", type=int, default=224)#256
    parser.add_argument("--width", type=int, default=224)#128
    parser.add_argument("--lr", type=float, default=1e-3)#3e-5
    parser.add_argument("--weight_decay", type=float, default=1e-4)#5e-2
    parser.add_argument("--category_index", type=int, default=None)
    parser.add_argument("--head_fusion", type=str, default="max", choices=['mean', 'min','max'])
    parser.add_argument("--discard_ratio", type=float, default=0.9)
    
    parser.add_argument("--clip_lr", type=float, default=4e-3)
    parser.add_argument("--clip_weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_update_parameters", type=list, default=["prompt_deep","softmax_model"])#,"part_class_embedding","agg_bn",]
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    
    parser.add_argument('--gpus', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=5)
    parser.add_argument("--reload", action='store_true')

    parser.add_argument("--datapath", type=str, default='/media/backup/**/PETA/')
    parser.add_argument("--a", type=float, default=0.5)
    return parser
