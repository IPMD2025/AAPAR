import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *
from clipS import clip
from clipS.model import *
import numpy as np
from models.layers import ResidualAttention,TransformerDecoder
from models.pre_peta_random import petabaseDataset
import copy
part_words=[[0,1,2,3,16],[10,18,19,30,15],[4,5,20,22,17,7,9,11,14,21,26,29,32,33,34],[6,8,12,25,27,31,13,23,24,28]]
# word_order=[0,1,2,3,16,10,18,19,30,15,4,5,20,22,17,7,9,11,14,21,26,29,32,33,34,6,8,12,25,27,31,13,23,24,28]
group_num_start=[1,6,11,26,36]
part_order_part=[[30,31,32,33,34],[0,1,2,3,4],[25,26,28,29,27,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]]
# part_order=[[0,1,2,3,4],[5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20],[21,22,23,24],[25,26,27,28,29],[30,31,32,33],[34]]
# part_order=[[0,1,2,3,4,5,6],[7,8],[9,10,11,12,13,14,15,16,17,18,21,24],[19,20,22,23,25]]
part_order=[[0],[1,2,3],[4,5,6],[7,8],[9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25]]
part_attr=[[1,2],[4,0],[3,6],[7,5]]

class TransformerClassifier(nn.Module):
    def __init__(self,args, attr_num,attr_words, dim=768, pretrain_path='/media/backup/**/pretrained/jx_vit_base_p16_224-80ecf9dd.pth'):#
        super().__init__()
        self.attr_num = attr_num
        self.dim=dim
        self.word_embed = nn.Linear(512, dim)
        self.visual_embed= nn.Linear(512, dim)
        self.feat_cam=Global_CAM()
        # self.clip_model, _ =  clip.load("ViT-B/16", device='cuda',download_root='/media/backup/**/pretrained')
        # self.clip_model=self.clip_model.float()
        self.attributes=attr_words
        self.lmbd=8
        self.vit = vit_base()
        self.vit.load_param(pretrain_path)
        self.blocks = self.vit.blocks[-1:]
        self.blocks_t =nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=dim,
                                       n_head=12,
                                       att_type='cross')])
        self.norm = nn.LayerNorm(self.dim)
        
        self.blocks_p=[]
        self.norm_p=[]
        self.head_p=[]
        self.bn_p=[]
        self.blocks_pt=[]
        self.norm_pt=[]
        self.head_pt=[]
        self.bn_pt=[]
        for i in range(self.lmbd):
            # self.weight_layer_part.append(nn.ModuleList([nn.Linear(dim, 1) for j in range(len(part_order[i]))]))
            self.blocks_p.append(nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=768,
                                       n_head=12,
                                       att_type='cross')]))
            self.blocks_pt.append(nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=768,
                                       n_head=12,
                                       att_type='cross')]))
            self.norm_p.append(nn.LayerNorm(self.dim))
            self.head_p.append(nn.Linear(dim, len(part_order[i])))
            self.bn_p.append(nn.LayerNorm(len(part_order[i])))
            self.head_p[i].apply(self._init_weights)
            self.norm_pt.append(nn.LayerNorm(self.dim))
            self.head_pt.append(nn.Linear(dim, len(part_order[i])))
            self.bn_pt.append(nn.LayerNorm(len(part_order[i])))
            self.head_pt[i].apply(self._init_weights)
        self.norm = nn.LayerNorm(self.dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.LayerNorm(self.attr_num)
        self.bn_g=nn.LayerNorm(self.attr_num)
        self.bn_f=nn.LayerNorm(self.attr_num)

       
        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.text = clip.tokenize(attr_words).cuda()

        self.head = nn.Linear(dim, self.attr_num)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        self.head_f=nn.Linear(dim, self.attr_num)
        self.head_f.apply(self._init_weights)
        self.pos_embed = nn.Parameter(torch.zeros(1,self.attr_num , self.dim))#128
        trunc_normal_(self.pos_embed, std=.02)
        self.cls_part_token=nn.Parameter(torch.zeros(1, self.lmbd, dim))
        
        # self.weight_layer_part=[]
        # self.blocks_part=[]
        # self.norm_part=[]
        # for i in range(4):
        #     self.weight_layer_part.append(nn.ModuleList([nn.Linear(dim, 1) for j in range(len(part_order[i]))]))
        #     self.blocks_part.append(copy.deepcopy(self.blocks))
        #     self.norm_part.append(copy.deepcopy(self.norm))
        
        # self.bn_part=nn.LayerNorm(self.attr_num)
        # self.descrip=petabaseDataset(args.datapath)
        # self.apply(self._init_weights)

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imgs,ViT_model):
        # features,all_x_cls = self.vit(imgs)
        features,all_x_cls = ViT_model.encode_image(imgs)
        features=(features.float())#self.visual_embed
        word_embed=ViT_model.encode_text(self.text).cuda().float()
        word_embed = (word_embed).expand(features.shape[0], -1, -1)#self.word_embed
        B,N,_=features.shape
        part_embed=[]
        for j in range(self.lmbd):
            part_embed.append(torch.gather(word_embed,dim=1,index=torch.tensor(part_order[j]).unsqueeze(0).unsqueeze(-1).expand(B,-1,self.dim).cuda()))
        
        x=features
        
        for blk in self.blocks:    
            x = blk(x)
        x = self.norm(x)
        
        feat_g=self.head(x[:,0])
        feat_g=self.bn_g(feat_g)
       
        attn_clip=ViT_model.visual.transformer.attn_weights#self.vit.attn_weights
        attn_vit=[]
        attn_vit.append(self.blocks[0].attn_w.float())
        
        attn=attn_clip + (attn_vit)

        logits=[]
        logits.append(feat_g)
        
        global_cam = self.feat_cam(attn,x)
        tex_embed = word_embed #+ self.tex_embed
        vis_embed = global_cam#+ self.pos_embed  #+ self.vis_embed #features_cls_temp #
       
        for blk in self.blocks_t:     
            z = blk(vis_embed,tex_embed) 
        z = self.norm(z)
        feat_f=self.head_f(z.squeeze(1))
        feat_f=self.bn_f(feat_f)
        logits.append(feat_f)
        
        vit_cls_g,vit_cls_p=0,0

        part_tokens= x[:,0].unsqueeze(1) + self.cls_part_token #features[:,0]
        patch_embed = x[:,1:] #+ self.tex_embed
        token_embed = part_tokens #+ self.vis_embed

        features_p=[]
        # part_cam=[]
        logits_p=[]
        logits_pt=[]
        # partlist=[[0,N-1],[0,(N-1)//2],[(N-1)//4,(N-1)*3//4],[(N-1)//2,N-1]]
        for i in range(self.lmbd):
            self.blocks_p[i].cuda()
            self.norm_p[i].cuda()
            self.head_p[i].cuda()
            self.bn_p[i].cuda()
            for blk in self.blocks_p[i]:#
                t = blk(token_embed[:,i].unsqueeze(1),patch_embed)#[:,partlist[i][0]:partlist[i][1]]
            t = self.norm_p[i](t)
            features_p.append(t)
            feat_p=self.head_p[i](t.squeeze(1))
            feat_p=self.bn_p[i](feat_p)
            attn_part=self.blocks_p[i][0].attn_weight##
            part_cam=attn_part@patch_embed#[:,partlist[i][0]:partlist[i][1]]
            
            tex_embed = part_embed[i] #+ self.tex_embed
            vis_embed = part_cam #+ self.pos_embed #+ self.vis_embed #features_cls_temp #features[:,5:]
            self.blocks_pt[i].cuda()
            self.norm_pt[i].cuda()
            self.head_pt[i].cuda()
            self.bn_pt[i].cuda()
            for blk in self.blocks_pt[i]:     
                y = blk(vis_embed,tex_embed) 
            y = self.norm_pt[i](y)
            feat_pt=self.head_pt[i](y.squeeze(1))
            feat_pt=self.bn_pt[i](feat_pt)
            #part_cam.append(self.feat_cam(attn_part,patch_embed[:,partlist[i][0]:partlist[i][1]]))
            logits_p.append(feat_p)
            logits_pt.append(feat_pt)
        features_p=torch.cat(features_p,dim=1)
        # part_cam=torch.cat(part_cam,dim=1)
        # logits_p=torch.cat((logits_p[1],logits_p[2][:,5:],logits_p[3],logits_p[2][:,:5],logits_p[0]),dim=1)
        logits_p=torch.cat(logits_p,dim=1)
        logits_pt=torch.cat(logits_pt,dim=1)
        logits.append(logits_p)

        logits.append(logits_pt)
        # b= torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        # logits.append(self.bn(b))

        return logits,all_x_cls,vit_cls_g,vit_cls_p#loss_itc,loss_itc_part,



class Global_CAM(nn.Module):
    def __init__(self):
        super(Global_CAM, self).__init__()

    def forward(self, x,features):
        length = len(x)
        # feat_cam=[]
        # N=features.shape[1]
        # feats_patch=[features[:,1:],features[:,1:],features[:,1:((N-1)//2 + 1)],features[:,((N-1)//4 + 1):(3*(N-1)//4 + 1)],features[:,((N-1)//2 + 1):]]
        last_map =x[0].float()
        for i in range(1, length):
            last_map = torch.matmul(x[i].float(), last_map)
        
        last_map1 = last_map[:,0,1:].unsqueeze(1)
        feat_cam=last_map1@F.relu(features[:,1:])
        
        return feat_cam  
        
        
         

