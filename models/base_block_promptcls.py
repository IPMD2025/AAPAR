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

# part_order=[[0],[1,2,3],[4,5,6],[7,8],[9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24,25]]
# part_order=[[0],[1,2,3],[4,5,6],[7,8],[9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25]]
# part_order=[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18],[19,20,21,22,23,24,25]]
part_order=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[15,16,17,18,19,20,21,22,23,24],[25,26,27,28,29],[30,31,32,33,34]]
# part_order_arr=[0,1,2,3,4,5,6,7,8,13,14,15,16,17,18,9,10,11,12,19,20,21,22,23,24,25]
group_order = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]

class TransformerClassifier(nn.Module):
    def __init__(self,args, attr_num,attr_words, dim=768, pretrain_path='/media/backup/**/pretrained/jx_vit_base_p16_224-80ecf9dd.pth'):#
        super().__init__()
        self.attr_num = attr_num
        self.dim=dim
        self.word_embed = nn.Linear(dim, dim)
        self.word_embed.apply(self._init_weights)
        # self.visual_embed= nn.Linear(512, dim)
        self.group_vice=self.get_groupvice(group_order)
        self.lmbd=4
        self.feat_cam=Part_CAM(self.lmbd)#Global_CAM()
        # self.clip_model, _ =  clip.load("ViT-B/16", device='cuda',download_root='/media/sdb/**/pretrained')
        # self.clip_model=self.clip_model.float()
        self.attributes=attr_words
        
        self.patch=256
        self.get_image_mask(self.patch,2+self.lmbd)
        vit = vit_base()
        vit.load_param(pretrain_path)
        self.blocks = vit.blocks[-1:]
        self.norm = vit.norm
        self.head = nn.Linear(dim, self.attr_num)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
        self.bn_g=nn.BatchNorm1d(self.attr_num)
       

        self.head_p=nn.ModuleList([nn.Linear(dim, len(part_order[i])) for i in range(self.lmbd)])
        # self.bn_p=nn.ModuleList([nn.BatchNorm1d(len(part_order[i])) for i in range(self.lmbd)])
        for i in range(self.lmbd):
            # self.bn_p.append(nn.LayerNorm(len(part_order[i])))
            self.head_p[i].apply(self._init_weights)
        self.bn_p=nn.BatchNorm1d(self.attr_num)
        
        vit1=vit_base()
        vit1.load_param(pretrain_path)
        self.blocks_t=vit1.blocks[-1:]#copy.deepcopy(self.blocks)
        # self.blocks_t=nn.ModuleList([ResidualAttention(num_layers=1,
        #                                d_model=dim,
        #                                n_head=12,
        #                                att_type='cross')])
        self.norm_t=vit1.norm#copy.deepcopy(self.norm)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn=nn.BatchNorm1d(self.attr_num)
       
        # self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        # self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.text = clip.tokenize(attr_words).cuda()
        # self.cls_part_token=nn.Parameter(torch.zeros(1, self.lmbd, dim))#
        # val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + dim))
        # nn.init.uniform_(self.cls_part_token.data, -val,val)

        # nn.init.normal_(self.cls_part_token)
        # self.head_t = nn.Linear(self.attr_num, self.attr_num)
        # self.head_t.apply(self._init_weights)
        # self.bn_t=nn.BatchNorm1d(self.attr_num)
        
        
      

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imgs,ViT_model):
        features = ViT_model.encode_image(imgs)#,all_x_cls
        features=(features.float())#self.visual_embed
        B,N,_=features.shape
        word_embed=ViT_model.encode_text(self.text).cuda().float()
        word_embed = self.word_embed(word_embed).expand(features.shape[0], -1, -1)#
        # part_tokens= self.cls_part_token.expand(B,-1,-1).cuda() #features[:,0]features[:,0].unsqueeze(1) + 
        # patch_embed = features[:,1:] #+ self.tex_embed
       
        # features_all=torch.cat((features[:,0].unsqueeze(1),part_tokens,patch_embed),dim=1)
        
        # part_embed=[]
        # for j in range(self.lmbd):
        #     part_embed.append(torch.gather(word_embed,dim=1,index=torch.tensor(part_order[j]).unsqueeze(0).unsqueeze(-1).expand(B,-1,self.dim).cuda()))
        
        x=features#_all
        image_mask_last=self.image_mask
        for blk in self.blocks:
            blk.attn_mask=image_mask_last    
            x = blk(x)
        img_ft = self.norm(x)
        
        feat_g=self.head(img_ft[:,1])
        feat_g=self.bn_g(feat_g)
       
        attn_clip=ViT_model.visual.transformer.attn_weights#self.vit.attn_weights
        attn_vit=[]
        attn_vit.append(self.blocks[0].attn_w.float())
        attn=attn_clip + (attn_vit)
        feat_cam = self.feat_cam(attn,img_ft)
        P=50
        # attn_clip=ViT_model.visual.transformer.attn_weights#self.vit.attn_weights
        # att_len=len(attn_clip)
        # att_vit=self.blocks[0].attn_w.float()
        # att_vit=torch.cat((att_vit[:,1].unsqueeze(1),att_vit[:,2+self.lmbd:]),dim=1)#+P
        # att_vit=torch.cat((att_vit[:,:,1].unsqueeze(-1),att_vit[:,:,2+self.lmbd:]),dim=-1)
        # # attn=(attn_clip[att_len-1] + att_vit)/2
        # # last_map = attn[:,0,1:].unsqueeze(1)
        # last_map = att_vit[:,0,1:].unsqueeze(1)
        # feat_map=F.relu(img_ft[:,2+self.lmbd:])
        # feat_cam=last_map@feat_map

        logits=[]
        logits.append(feat_g)
        
       
        logits_p=[]
        for i in range(self.lmbd):
            self.head_p[i].cuda()
            # self.bn_p[i].cuda()
            feat_p=self.head_p[i](img_ft[:,i+2])
            # feat_p=self.bn_p[i](feat_p)
            logits_p.append(feat_p)

        # logits_arrange=torch.cat(logits_p,dim=1)
        # logits_part=torch.gather(logits_arrange,dim=1,index=torch.tensor(self.group_vice).unsqueeze(0).expand(B,-1).cuda())
        # logits.append(self.bn_p(logits_part))

        logits_p=torch.cat(logits_p,dim=1)
        logits.append(self.bn_p(logits_p))

        tex_embed = word_embed #+ self.tex_embed
        vis_embed = feat_cam#global_cam[:,0].unsqueeze(1)#+ self.pos_embed  #+ self.vis_embed #features_cls_temp #
       
        x=torch.cat((tex_embed,vis_embed),dim=1)
        for blk in self.blocks_t:     
            # ptext_ft = blk(vis_embed,tex_embed) 
            x=blk(x)
        ptext_ft = self.norm_t(x)
        d= torch.cat([self.weight_layer[i](ptext_ft[:, i, :]) for i in range(self.attr_num)], dim=1)
        # pp=torch.gather(self.bn(d),dim=1,index=torch.tensor(self.group_vice).unsqueeze(0).expand(B,-1).cuda())
        # feat_pt=self.head_pt(ptext_ft.squeeze(1))
        # d=self.bn_pt(feat_pt)
        logits.append(self.bn(d))#pp

       
        return logits#loss_itc,loss_itc_part,

    def get_image_mask(self,N,C):
        # partlist=[[0,(N-1)//2],[0,(N-1)//2],[(N-1)//2,N-1],[(N-1)//2,N-1],[(N-1)//4,(N-1)*3//4],[0,N-1],[0,N-1]]
        P=50
        self.image_mask = torch.zeros(C+P+N, C+P+N)
        
        self.image_mask[2:C,2:C].fill_(float("-inf"))     #8个cls token   
        self.image_mask[2][C+P+N//2:].fill_(float("-inf"))   #0-hair， 1th，2th，3th块保留  1-age whole attention 2-gender whole attention
        self.image_mask[3][:C+P+N//2].fill_(float("-inf"))
        self.image_mask[4][:C+P+N//4].fill_(float("-inf"))
        self.image_mask[4][C+P+N*3//4:].fill_(float("-inf"))
        
        # self.image_mask[7][:C+P+N//2].fill_(float("-inf"))   #3-carry 3,4,5,6块保留 [2*2*14+8,6*2*14+8]
        # self.image_mask[8][:C+P+N//2].fill_(float("-inf"))  #4-accessory 1,2,3,4,5,6块保留 [6*2*14+8]  
 
        for i in range(C): 
            if i>1:
                self.image_mask[i][C:P+C].fill_(float("-inf"))#
            # self.image_mask[i][0].fill_(0)
            self.image_mask[i][1].fill_(0)
            self.image_mask[i][i].fill_(0)
        self.image_mask[1:C,0].fill_(float("-inf"))#
        # self.image_mask[0][1:P+C].fill_(float("-inf"))
            
    
    # def get_image_mask(self,N,C):
    #     # partlist=[[0,(N-1)//2],[0,(N-1)//2],[(N-1)//2,N-1],[(N-1)//2,N-1],[(N-1)//4,(N-1)*3//4],[0,N-1],[0,N-1]]
    #     P=50
    #     self.image_mask = torch.zeros(C+P+N, C+P+N)
        
    #     self.image_mask[1:,:C].fill_(float("-inf"))     #8个cls token   
    #     self.image_mask[4][C+P+N//2:].fill_(float("-inf"))   #0-hair， 1th，2th，3th块保留  1-age whole attention 2-gender whole attention
    #     self.image_mask[5][:C+P+N//4].fill_(float("-inf"))
    #     self.image_mask[5][C+P+N*3//4:].fill_(float("-inf"))
    #     self.image_mask[6][C+P+N//2:].fill_(float("-inf"))
    #     self.image_mask[7][:C+P+N//2].fill_(float("-inf"))   #3-carry 3,4,5,6块保留 [2*2*14+8,6*2*14+8]
    #     self.image_mask[8][:C+P+N//2].fill_(float("-inf"))  #4-accessory 1,2,3,4,5,6块保留 [6*2*14+8]  
 
    #     for i in range(C): 
    #         self.image_mask[i][C:P+C].fill_(float("-inf"))
    #         self.image_mask[i][i].fill_(0)
    #         self.image_mask[i][0].fill_(0)
    def get_image_mask_old(self,N,C):
        # partlist=[[0,(N-1)//2],[0,(N-1)//2],[(N-1)//2,N-1],[(N-1)//2,N-1],[(N-1)//4,(N-1)*3//4],[0,N-1],[0,N-1]]
        self.image_mask = torch.zeros(C+N, C+N)
        self.image_mask[:,:C].fill_(float("-inf"))     #8个cls token   
        self.image_mask[4][C+N//2:].fill_(float("-inf"))   #0-hair， 1th，2th，3th块保留  1-age whole attention 2-gender whole attention
        self.image_mask[5][:C+N//4].fill_(float("-inf"))   #3-carry 3,4,5,6块保留 [2*2*14+8,6*2*14+8]
        self.image_mask[5][C+N*3//4:].fill_(float("-inf"))
        self.image_mask[6][C+N//2:].fill_(float("-inf"))
        self.image_mask[7][:C+N//2].fill_(float("-inf"))
        self.image_mask[8][:C+N//2].fill_(float("-inf"))
          #4-accessory 1,2,3,4,5,6块保留 [6*2*14+8]  
 
        for i in range(C): 
            self.image_mask[i][i].fill_(0)
          
    def get_groupvice(self,grouporder):
        length=len(grouporder)
        group_vice=[]
        for i in range(length):
            for j in range(length):
                if i==grouporder[j]:
                    group_vice.append(j)
        return group_vice
            
    


class Global_CAM(nn.Module):
    def __init__(self):
        super(Global_CAM, self).__init__()

    def forward(self, x,features):
        length = len(x)
        C=8
        # feat_cam=[]
        # N=features.shape[1]
        # feats_patch=[features[:,1:],features[:,1:],features[:,1:((N-1)//2 + 1)],features[:,((N-1)//4 + 1):(3*(N-1)//4 + 1)],features[:,((N-1)//2 + 1):]]
        last_map =x[0].float()
        for i in range(1, length):
            last_map = torch.matmul(x[i].float(), last_map)
        
        last_map1 = last_map[:,0,1:].unsqueeze(1)
        feat_cam=last_map1@F.relu(features[:,C:])
        
        return feat_cam 

class Global_CAM_2(nn.Module):
    def __init__(self,lmbd):
        super(Global_CAM, self).__init__()
        self.lmbd=lmbd

    def forward(self, x,features):
        length = len(x)
        C=1+self.lmbd
        
        b=x[0].shape[0]
        att_tt=[]
        N=features.shape[1]
        feat_cam=[]
        # partlist=[[0,N-4],[0,(N-4)//2],[(N-4)//4,(N-4)*3//4],[(N-4)//2,N-4]]
        for d in range(length): 
            att_tk=x[d][:,0]
            att_pt=x[d][:,C:]
            att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
            att_tk2=att_t1[:,:,0]
            att_pt2=att_t1[:,:,C:]
            att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
            att_tt.append(att)
       
        last_map =att_tt[0].float()
        for i in range(1, length):
            last_map = torch.matmul(att_tt[i].float(), last_map)
        
        last_map1 = last_map[:,0,1:].unsqueeze(1)
        feat_map=F.relu(features[:,C:])
        feat_cam=last_map1@feat_map
        return feat_cam

class Part_CAM_p(nn.Module):
    def __init__(self,lmbd):
        super(Part_CAM, self).__init__()
        self.lmbd=lmbd

    def forward(self, x,features):
        length = len(x)
        C=1+self.lmbd
        b=x[0].shape[0]
        att_tt=[]
        N=features.shape[1]
        feat_cam=[]
        for d in range(length):
            if d==length-1:
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
            else:
                att_tt.append(x[d])
        
        last_map =att_tt[0].float()
        for i in range(1, length-1):
            last_map = torch.matmul(att_tt[i].float(), last_map)

        att_map=[]
        for f in range(C):
            last_map_gpcls = torch.matmul(att_tt[length-1][f].float(), last_map)
            att_map.append(last_map_gpcls)

        for k in range(C): 
            last_map1 = att_map[k][:,0,1:].unsqueeze(1)
            feat_map=F.relu(features[:,C:])
            feat_cam.append(last_map1@feat_map)
        feat_cam=torch.cat(feat_cam,dim=1)
        return feat_cam

class Part_CAM(nn.Module):
    def __init__(self,lmbd):
        super(Part_CAM, self).__init__()
        self.lmbd=lmbd

    def forward(self, x,features):
        length = len(x)
        C=2+self.lmbd
        
        b=x[0].shape[0]
        att_tt=[]
        N=features.shape[1]
        feat_cam=[]
        # partlist=[[0,N-4],[0,(N-4)//2],[(N-4)//4,(N-4)*3//4],[(N-4)//2,N-4]]
        for d in range(length):
            
            # if d==length-1:
            att_vit=torch.cat((x[d][:,1].unsqueeze(1),x[d][:,C:]),dim=1)# + P
            att_vit=torch.cat((att_vit[:,:,1].unsqueeze(-1),att_vit[:,:,C:]),dim=-1)
            att_tt.append(att_vit)
            # else:
            #     # att_clip=torch.cat((x[d][:,0].unsqueeze(1),x[d][:,1+P:]),dim=1)# + P
            #     # att_clip=torch.cat((att_clip[:,:,0].unsqueeze(-1),att_clip[:,:,1+P :]),dim=-1)
            #     # att_tt.append(att_clip)
            #     att_tt.append(x[d])
        
        last_map =att_tt[0].float()
        for i in range(1, length):
            last_map = torch.matmul(att_tt[i].float(), last_map)
        
        last_map1 = last_map[:,0,1:].unsqueeze(1)
        feat_map=F.relu(features[:,C:])
        feat_cam=last_map1@feat_map
        return feat_cam
        
        
         

