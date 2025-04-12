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
# part_order=[[30,31,32,33,34],[0,1,2,3,4],[25,26,28,29,27,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24]]
# part_order_group=[[0],[1,2,3],[7,8],[9,10,11,12],[4,5,6],[25],[13,14,15,16,17,18],[19,20,21,22,23,24]]
# part_order=[[0,1,2,3,4,5,6],[7,8,13,14,15,16,17,18],[9,10,11,12,21,24],[19,20,22,23,25]]
part_order=[[0,1,2,3,4,5,6],[7,8],[9,10,11,12,13,14,15,16,17,18],[19,20,21,22,23,24,25]]
part_attr=[[1,2],[4,0],[3,6],[7,5]]

class TransformerClassifier(nn.Module):
    def __init__(self,args, attr_num,attr_words, dim=512, pretrain_path='/media/backup/**/pretrained/jx_vit_base_p16_224-80ecf9dd.pth'):#
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(512, dim)
        self.visual_embed= nn.Linear(512, dim)
        
        self.part_cam=Part_CAM()
        self.global_cam=Global_CAM()
        self.clip_model, _ =  clip.load("ViT-B/16", device='cuda',download_root='/media/backup/**/pretrained')
        self.clip_model=self.clip_model.float()
        self.attributes=attr_words
        
        self.blocks =nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=512,
                                       n_head=8,
                                       att_type='cross')])
        self.blocks_p =nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=512,
                                       n_head=8,
                                       att_type='cross')])

        self.dim=dim
        self.norm = nn.LayerNorm(self.dim)
        self.norm_p = nn.LayerNorm(self.dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.LayerNorm(self.attr_num)

        self.weight_layer_part = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn_part = nn.LayerNorm(self.attr_num)

        
        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.text = clip.tokenize(attr_words).cuda()
        self.dim=dim
       
        self.descrip=petabaseDataset(args.datapath)
        

        self.head = nn.Linear(dim, self.attr_num)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        self.head_p=[]
        self.bn_p=[]
        for i in range(4):
            self.head_p.append(nn.Linear(dim, len(part_order[i])))
            self.bn_p.append(nn.LayerNorm(len(part_order[i])))
            self.head_p[i].apply(self._init_weights)
        self.pos_embed = nn.Parameter(torch.zeros(1,self.attr_num , self.dim))#128
        trunc_normal_(self.pos_embed, std=.02)
        self.cls_part_token=nn.Parameter(torch.zeros(1, 4, dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imgs, word_vec, label=None,description=None):
        # features,all_x_cls = self.vit(imgs)
        features,all_x_cls = self.clip_model.visual(imgs.type(self.clip_model.dtype))
        B,N,_=features.shape
        
        attn=self.clip_model.visual.transformer.attn_weights#self.vit.attn_weights
        part_cam=self.part_cam(attn,features)
        global_cam=self.global_cam(attn,features)
        logits=[]
        feat_g=self.head(features[:,0])
        feat_g=self.bn(feat_g)
        logits.append(feat_g)
        
        logits_p=[]
        for i in range(4):
            self.head_p[i].cuda()
            self.bn_p[i].cuda()
            feat_p=self.head_p[i](features[:,i+1])
            feat_p=self.bn_p[i](feat_p)
            logits_p.append(feat_p)
        # logits_p=torch.cat((logits_p[0],logits_p[1][:,:2],logits_p[2][:,:4],logits_p[1][:,2:],logits_p[3][:,:2],logits_p[2][:,4].unsqueeze(-1),
        # logits_p[3][:,2:4],logits_p[2][:,-1].unsqueeze(-1),logits_p[3][:,-1].unsqueeze(-1)),dim=1)
        logits_p=torch.cat(logits_p,dim=1)
        logits.append(logits_p)
        # part_order=[[0,1,2,3,4,5,6],[7,8,13,14,15,16,17,18],[9,10,11,12,21,24],[19,20,22,23,25]]

        word_embed = (word_vec).expand(features.shape[0], word_vec.shape[0], features.shape[-1])
        part_embed=[]
        for j in range(4):
            part_embed.append(torch.gather(word_embed,dim=1,index=torch.tensor(part_order[j]).unsqueeze(0).unsqueeze(-1).expand(B,-1,self.dim).cuda()))

        # gpnocls_cam = self.feat_cam(attn,features)
        spec=global_cam#features[:,1:]
        
        tex_embed = word_embed #+ self.tex_embed
        vis_embed = spec#+ self.pos_embed  #+ self.vis_embed #features_cls_temp #
       
        for blk in self.blocks:     
            x = blk(tex_embed,vis_embed) 
        x = self.norm(x)
        b= torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits.append(self.bn(b))

        vit_cls_p=[]
        for i in range(4):
            spec=part_cam#patch_embed[:,partlist[i][0]:partlist[i][1]]#globle_part_cam
            
            tex_embed = part_embed[i] #+ self.tex_embed
            vis_embed = spec #+ self.pos_embed #+ self.vis_embed #features_cls_temp #features[:,5:]
            # self.blocks_part[i].cuda()
            # self.norm_part[i].cuda()
            for blk in self.blocks_p:#self.blocks_part[i]     
                x = blk(tex_embed,vis_embed) 
            x = self.norm_p(x)#self.norm_part[i](x)
            vit_cls_p.append(x)
               
        # vit_cls_p=torch.cat((vit_cls_p[0],vit_cls_p[1][:,:2],vit_cls_p[2][:,:4],vit_cls_p[1][:,2:],vit_cls_p[3][:,:2],vit_cls_p[2][:,4].unsqueeze(1),
        # vit_cls_p[3][:,2:4],vit_cls_p[2][:,-1].unsqueeze(1),vit_cls_p[3][:,-1].unsqueeze(1)),dim=1)
        vit_cls_p=torch.cat(vit_cls_p,dim=1)
        b=torch.cat([self.weight_layer_part[i](vit_cls_p[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits.append(self.bn_part(b))
        
        
        vit_cls_g,vit_cls_p=0,0

        return logits,all_x_cls,vit_cls_g,vit_cls_p#loss_itc,loss_itc_part,



class Global_CAM(nn.Module):
    def __init__(self):
        super(Global_CAM, self).__init__()

    def forward(self, x,features):
        length = len(x)
        feat_cam=[]
        B,N,dim=features.shape
        C=5
        idx_token=np.array(range(N))
        idx_token=torch.cat((torch.tensor(idx_token[np.where(idx_token==0)]),torch.tensor(idx_token[np.where(idx_token >C-1)]))).cuda()


        # feats_patch=[features[:,1:],features[:,1:],features[:,1:((N-1)//2 + 1)],features[:,((N-1)//4 + 1):(3*(N-1)//4 + 1)],features[:,((N-1)//2 + 1):]]
        partlist=[[C,N],[C,C+(N-C)//2],[C+(N-C)//4,C+(N-C)*3//4],[C+(N-C)//2,N]]
        
        last_map_temp=torch.gather(x[0],dim=1,index=idx_token.unsqueeze(0).unsqueeze(-1).expand(B,-1,N))
        last_map=torch.gather(last_map_temp,dim=2,index=idx_token.unsqueeze(0).unsqueeze(0).expand(B,N-C+1,-1))
        
        for i in range(1, length):
            
            temp_map=torch.gather(x[i],dim=1,index=idx_token.unsqueeze(0).unsqueeze(-1).expand(B,-1,N).cuda())
            middle_map=torch.gather(temp_map,dim=2,index=idx_token.unsqueeze(0).unsqueeze(0).expand(B,N-C+1,-1).cuda())
            last_map = torch.matmul(middle_map, last_map)
        
        last_map1 = last_map[:,0,1:].unsqueeze(1)
        for k in range(4):
            feature_map=F.relu(features[:,partlist[k][0]:partlist[k][1]])
            feat_cam.append(last_map1[:,:,partlist[k][0]-C:partlist[k][1]-C]@feature_map)
        return torch.cat(feat_cam,dim=1) 

class Part_CAM(nn.Module):
    def __init__(self):
        super(Part_CAM, self).__init__()

    def forward(self, x,features):
        length = len(x)
        C=5
        
        b=x[0].shape[0]
        att_tt=[]
        N=features.shape[1]
        feat_cam=[]
        # partlist=[[0,N-4],[0,(N-4)//2],[(N-4)//4,(N-4)*3//4],[(N-4)//2,N-4]]
        for d in range(length):
            att_ts=[]
            for e in range(C):
                att_tk=x[d][:,e]
                att_pt=x[d][:,C:]
                
                att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
                att_tk2=att_t1[:,:,e]
                
                att_pt2=att_t1[:,:,C:]
                
                att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
                att_ts.append(att)
            att_tt.append(att_ts)
        att_map=[]
        for f in range(C):
            last_map =att_tt[0][f]
            for i in range(1, length):
                last_map = torch.matmul(att_tt[i][f], last_map)
            att_map.append(last_map)
        for k in range(C): 
            last_map1 = att_map[k][:,0,1:].unsqueeze(1)
            feat_map=F.relu(features[:,C:])
            feat_cam.append(last_map1@feat_map)
        feat_cam=torch.cat(feat_cam,dim=1)
        return feat_cam[:,1:]

class Part_CAM_mask(nn.Module):
    def __init__(self):
        super(Part_CAM, self).__init__()

    def forward(self, x,features):
        length = len(x)
        max_p=[]
        max_inx=[]
        b=x[0].shape[0]
        att_tt=[]
        N=features.shape[1]
        C=5
        feat_cam=[]
        partlist=[[C,N],[C,N],[C,C+(N-C)//2],[C+(N-C)//4,C+(N-C)*3//4],[C+(N-C)//2,N]]
        for d in range(length):
            att_ts=[]
            for e in range(C):
                att_tk=x[d][:,e]
                if e==0 or e==1 :
                    att_pt=x[d][:,C:]
                elif e==2:
                    att_pt=x[d][:,C:C+(N-C)//2]
                elif e==3:
                    att_pt=x[d][:,(C+(N-C)//4):(C+(N-C)*3//4)]
                else:
                    att_pt=x[d][:,C+(N-C)//2:]
                att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
                att_tk2=att_t1[:,:,e]
                if e==0 or e==1: 
                    att_pt2=att_t1[:,:,C:]
                elif e==2:
                    att_pt2=att_t1[:,:,C:C+(N-C)//2]
                elif e==3:
                    att_pt2=att_t1[:,:,(C+(N-C)//4):(C+(N-C)*3//4)]
                else:
                    att_pt2=att_t1[:,:,C+(N-C)//2:]
                att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
                att_ts.append(att)
            att_tt.append(att_ts)
        att_map=[]
        for f in range(C):
            last_map =att_tt[0][f]
            for i in range(1, length):
                last_map = torch.matmul(att_tt[i][f], last_map)
            att_map.append(last_map)
        for k in range(C): 
            last_map1 = att_map[k][:,0,1:].unsqueeze(1)
            feam_map=F.relu(features[:,partlist[k][0]:partlist[k][1]])
            feat_cam.append(last_map1@feam_map)
        feat_cam=torch.cat(feat_cam,dim=1)
        return feat_cam[:,1:] 
        
        
         

