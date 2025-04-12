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
import math
from functools import reduce
from operator import mul
part_order=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24],[25,26,27,28,29],[30,31,32,33,34]]
# part_order=[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18],[19,20,21,22,23,24,25]]
# part_order=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24,25,26],[27,28,29,30,31,32,33,34],[35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]]
class TransformerClassifier(nn.Module):
    def __init__(self,args, attr_num,attr_words,des_num, dim=768, pretrain_path='/media/backup/**/pretrained/jx_vit_base_p16_224-80ecf9dd.pth'):#
        super().__init__()
        self.attr_num = attr_num
        self.dim=dim
        self.word_embed = nn.Linear(dim, dim)
       
        # self.group_vice=self.get_groupvice(group_order)
        
        self.lmbd=4
        self.patch=256
        self.get_image_mask(self.patch,1+self.lmbd)
        vit = vit_base()
        vit.load_param(pretrain_path)
        self.blocks = vit.blocks[-1:]
        self.norm = vit.norm
        self.head = nn.Linear(dim, self.attr_num)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
        self.bn_g=nn.BatchNorm1d(self.attr_num)
        # self.feat_cam=Global_CAM(self.lmbd)
        
        self.head_p=nn.ModuleList([nn.Linear(dim, len(part_order[i])) for i in range(self.lmbd)]) 
        self.bn_p=nn.BatchNorm1d(self.attr_num)
        
        vit1=vit_base()
        vit1.load_param(pretrain_path)
        self.blocks_t=vit1.blocks[-1:]
        self.norm_t=vit1.norm#copy.deepcopy(self.norm)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn=nn.BatchNorm1d(self.attr_num)

        self.text = clip.tokenize(attr_words).cuda()
        self.cls_part_token=nn.Parameter(torch.zeros(1, self.lmbd, dim))#
        val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + dim))
        nn.init.uniform_(self.cls_part_token.data, -val,val)
        
        
        
        self.bn_des = nn.BatchNorm1d(des_num)#
        
       

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imgs,ViT_model,des=None,gt_label=None):#,group_arr
        # features,all_x_cls = self.vit(imgs)
        features = ViT_model.encode_image(imgs)
        features=(features.float())#self.visual_embed
        B,N,_=features.shape
        word_embed_temp=ViT_model.encode_text(self.text).cuda().float()
        
        if des!=None:
            loss_itc ,_ = ViT_model.forward_aggregate(features[:,0], des)
            loss_itc = self.bn_des(loss_itc)
        
        word_embed = self.word_embed(word_embed_temp).expand(features.shape[0], -1, -1)#self.word_embed
        part_tokens= self.cls_part_token.expand(B,-1,-1).cuda() #features[:,0].unsqueeze(1) + 
        patch_embed = features[:,1:] #+ self.tex_embed
       
        features_all=torch.cat((features[:,0].unsqueeze(1),part_tokens,patch_embed),dim=1)
        
        x=features_all
        image_mask_last=self.image_mask
        for blk in self.blocks:
            blk.attn_mask=image_mask_last    
            x = blk(x)
        img_ft = self.norm(x)
        
        feat_g=self.head(img_ft[:,0])#
        logits_g=self.bn_g(feat_g)
        
        # V_c=self.head_c(features[:,1:]).permute([0,2,1])
        # A_c=V_c.softmax(dim=-1)
        # spec_c=A_c@features[:,1:]
        
        # V=self.head_v(img_ft[:,1+self.lmbd:]).permute([0,2,1])
        # A=V.softmax(dim=-1)
        # spec_v=A@img_ft[:,1+self.lmbd:]
        
        # feat_cam = torch.cat((spec_c,spec_v),dim=1)
        P=50
        # 
        # attn_clip=ViT_model.visual.transformer.attn_weights#self.vit.attn_weights
        # att_len=len(attn_clip)
        # attn_vit=[]
        # attn_vit.append(self.blocks[0].attn_w.float())
        # attn=attn_clip + (attn_vit)
        # feat_cam_g = self.feat_cam(attn,img_ft)
        # last_map = attn_clip[-1][:,0,1:].unsqueeze(1).float()
        # feat_map_g=F.relu(features[:,1:])#+self.lmbd 
        # feat_cam_c=last_map@feat_map_g
        
       ###########################
        att_vit=self.blocks[0].attn_w.float()
        att_vit_cls=torch.cat((att_vit[:,0].unsqueeze(1),att_vit[:,1+self.lmbd:]),dim=1)#+ P 
        att_vit_cls=torch.cat((att_vit_cls[:,:,0].unsqueeze(-1),att_vit_cls[:,:,1+self.lmbd:]),dim=-1)#+ P
        last_map = att_vit_cls[:,0,1:].unsqueeze(1)
        feat_map_g=F.relu(img_ft[:,1+self.lmbd:])#+self.lmbd 
        feat_cam_g=last_map@feat_map_g


        img_msk_start=[0,self.patch // 2,self.patch // 4,0]
        img_msk_end=[self.patch // 2,self.patch +1,self.patch * 3 // 4,self.patch +1]
       
        feat_cam_tmp=[]
        for i in range(self.lmbd):
            att_vit_temp=torch.cat((att_vit[:,i+1].unsqueeze(1),att_vit[:,1+self.lmbd + P + img_msk_start[i] : 1+self.lmbd + P + img_msk_end[i]]),dim=1)#,att_vit[:,1+self.lmbd:1+self.lmbd + P]
            att_vit_p=torch.cat((att_vit_temp[:,:,i+1].unsqueeze(-1),att_vit_temp[:,:,1+self.lmbd +P+ img_msk_start[i] : 1+self.lmbd + P + img_msk_end[i]]),dim=-1)#,att_vit_temp[:,:,1+self.lmbd:1+self.lmbd + P]
            last_map = att_vit_p[:,0,1:].unsqueeze(1)
            img_patch= img_ft[:,1+self.lmbd + P + img_msk_start[i] : 1+self.lmbd + P + img_msk_end[i]]#torch.cat((img_ft[:,1+self.lmbd:1+self.lmbd + P],),dim=1)#
            feat_map_p=F.relu(img_patch)
            feat_cam_tmp.append(last_map@feat_map_p)
        feat_cam_p=torch.cat(feat_cam_tmp,dim=1)
        
        feat_cam=torch.cat((feat_cam_g,feat_cam_p),dim=1)#img_ft[:,0].unsqueeze(1),
        
        # att_vit=self.blocks[0].attn_w.float()
        # feat_cam_tmp=[]
        # for i in range(1+self.lmbd):
        #     last_map = att_vit[:,i,1+self.lmbd:].unsqueeze(1)
        #     feat_map=F.relu(img_ft[:,1+self.lmbd:])#+self.lmbd 
        #     feat_cam_tmp.append(last_map@feat_map)
        # feat_cam=torch.cat(feat_cam_tmp,dim=1)
        logits=[]
        logits.append(logits_g)
        
        
        logits_p=[]
        for i in range(self.lmbd):
            self.head_p[i].cuda()
            feat_p=self.head_p[i](img_ft[:,i+1])
            logits_p.append(feat_p)
        
        logits_p=torch.cat(logits_p,dim=1)
        logits.append( self.bn_p( logits_p))#  
        
        
        
        tex_embed =word_embed#torch.gather(word_embed,dim=1,index=torch.tensor(self.group_vice).unsqueeze(0).unsqueeze(-1).expand(B,-1,self.dim).cuda())## 
        vis_embed = feat_cam#features#torch.cat((img_ft[:,:1+self.lmbd],),dim=1)#img_ft[:,1+self.lmbd:]#global_cam[:,0].unsqueeze(1)#+ self.pos_embed #+ self.vis_embed #features_cls_temp #features[:,5:]
        x=torch.cat((tex_embed,vis_embed),dim=1)
        for blk in self.blocks_t:     
            x=blk(x)
        ptext_ft = self.norm_t(x)
        d= torch.cat([self.weight_layer[i](ptext_ft[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits.append(self.bn(d))#
        
        
        if des!=None:
            logits.append(loss_itc)
        return logits#loss_itc,loss_itc_part,

    def get_image_mask(self,N,C):
        # partlist=[[0,(N-1)//2],[0,(N-1)//2],[(N-1)//2,N-1],[(N-1)//2,N-1],[(N-1)//4,(N-1)*3//4],[0,N-1],[0,N-1]]
        P=50
        self.image_mask = torch.zeros(C+P+N, C+P+N)
    
        self.image_mask[:C,:C].fill_(float("-inf"))     #8个cls token   
        self.image_mask[1][C+P+N//2:].fill_(float("-inf"))   #0-hair， 1th，2th，3th块保留  1-age whole attention 2-gender whole attention
        self.image_mask[2][:C+P+N//2].fill_(float("-inf"))
        self.image_mask[3][:C+P+N//4].fill_(float("-inf"))   #3-carry 3,4,5,6块保留 [2*2*14+8,6*2*14+8]
        self.image_mask[3][C+P+N*3//4:].fill_(float("-inf"))  #4-accessory 1,2,3,4,5,6块保留 [6*2*14+8]  
 
        for i in range(C): 
            if i!=0:
                self.image_mask[i][C:P+C].fill_(float("-inf"))#
            self.image_mask[i][0].fill_(0)#
            self.image_mask[i][i].fill_(0)
    
    def get_groupvice(self,grouporder):
        length=len(grouporder)
        group_vice=[]
        for i in range(length):
            for j in range(length):
                if i==grouporder[j]:
                    group_vice.append(j)
        return group_vice
    # def get_image_mask(self,N,C):
    #     # partlist=[[0,(N-1)//2],[0,(N-1)//2],[(N-1)//2,N-1],[(N-1)//2,N-1],[(N-1)//4,(N-1)*3//4],[0,N-1],[0,N-1]]
    #     P=50
    #     self.image_mask = torch.zeros(C+P+N, C+P+N)
    
    #     self.image_mask[:C,:C].fill_(float("-inf"))     #8个cls token   
    #     self.image_mask[1][C+P+N//5:].fill_(float("-inf"))   #0-hair， 1th，2th，3th块保留  1-age whole attention 2-gender whole attention
    #     self.image_mask[2][:C+P+N//5].fill_(float("-inf"))
    #     self.image_mask[2][C+P+2*N//5:].fill_(float("-inf"))
    #     self.image_mask[3][:C+P+3*N//5].fill_(float("-inf"))
    #     self.image_mask[3][C+P+4*N//5:].fill_(float("-inf"))   #3-carry 3,4,5,6块保留 [2*2*14+8,6*2*14+8]
    #     self.image_mask[4][:C+P+4*N//5].fill_(float("-inf"))  #4-accessory 1,2,3,4,5,6块保留 [6*2*14+8]  
    #     self.image_mask[5][:C+P+2*N//5].fill_(float("-inf"))
    #     self.image_mask[5][C+P+3*N//5:].fill_(float("-inf"))
        
    #     for i in range(C): 
    #         if i!=0:
    #             self.image_mask[i][C:P+C].fill_(float("-inf"))#
    #         self.image_mask[i][0].fill_(0)#
    #         self.image_mask[i][i].fill_(0)


    def get_groupvice(self,grouporder):
        length=len(grouporder)
        group_vice=[]
        for i in range(length):
            for j in range(length):
                if i==grouporder[j]:
                    group_vice.append(j)
        return group_vice

def get_grouparr(grouporder,grouparr):
    length=len(grouparr)
    group_arr=[]
    for i in range(length):
        group_arr.append(grouporder[grouparr[i]])
    array_1d = [item for sublist in group_arr for item in sublist]
    return array_1d
    
def get_contrastive_loss(image_feat, text_feat,model, idx=None):
        # assert image_feat.size(-1) == self.embed_dim
        # assert text_feat.size(-1) == self.embed_dim
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        image_feat_all = image_feat#allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = text_feat#allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() 
        logits=logits* model.logit_scale.exp()
        # bsz = image_feat_all.shape[0]

        # if idx is None:
        #     labels = torch.arange(bsz, device=image_feat.device)
        #     loss_i2t = F.cross_entropy(logits, labels)
        #     loss_t2i = F.cross_entropy(logits.t(), labels)
        #     return (loss_i2t + loss_t2i) / 2
        # else:
        #     idx = idx.view(-1, 1)
        #     assert idx.size(0) == image_feat.size(0)
        #     idx_all = idx#allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
        #     pos_idx = torch.eq(idx_all, idx_all.t()).float()
        #     labels = pos_idx / pos_idx.sum(1, keepdim=True)

        #     loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
        #     loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
        return logits#(loss_i2t + loss_t2i) / 2
# class Global_CAM(nn.Module):
#     def __init__(self):
#         super(Global_CAM, self).__init__()

#     def forward(self, x,features):
#         length = len(x)
#         C=1
#         # feat_cam=[]
#         # N=features.shape[1]
#         # feats_patch=[features[:,1:],features[:,1:],features[:,1:((N-1)//2 + 1)],features[:,((N-1)//4 + 1):(3*(N-1)//4 + 1)],features[:,((N-1)//2 + 1):]]
#         last_map =x[0].float()
#         for i in range(1, length):
#             last_map = torch.matmul(x[i].float(), last_map)
        
#         last_map1 = last_map[:,0,1:].unsqueeze(1)
#         feat_cam=last_map1@F.relu(features[:,C:])
        
#         return feat_cam 
    
class Global_CAM(nn.Module):
    def __init__(self,lmbd):
        super(Global_CAM, self).__init__()
        self.lmbd = lmbd
    def forward(self, x,features):
        length = len(x)
        C=1 + self.lmbd
        P=50
        # feat_cam=[]
        # N=features.shape[1]
        # feats_patch=[features[:,1:],features[:,1:],features[:,1:((N-1)//2 + 1)],features[:,((N-1)//4 + 1):(3*(N-1)//4 + 1)],features[:,((N-1)//2 + 1):]]
        attn_vit=x[-1]
        attn_vit = torch.cat((attn_vit[:,0].unsqueeze(1),attn_vit[:,C:]),dim=1)
        attn_vit = torch.cat((attn_vit[:,:,0].unsqueeze(-1),attn_vit[:,:,C:]),dim=2)
        x[-1] = attn_vit
        last_map =x[0].float()
        for i in range(1, length):
            last_map = torch.matmul(x[i].float(), last_map)
        
        last_map1 = last_map[:,0,1:1+P].unsqueeze(1)
        last_map1 = torch.cat((last_map1,attn_vit[:,0,1+P:].unsqueeze(1)),dim=2)
        feat_cam=last_map1@F.relu(features[:,C:])
        
        return feat_cam 