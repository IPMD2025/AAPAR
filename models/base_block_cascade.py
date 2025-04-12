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
part_order_group=[[0],[1,2,3],[7,8],[9,10,11,12],[4,5,6],[25],[13,14,15,16,17,18],[19,20,21,22,23,24]]
part_order=[[0,1,2,3,4,5,6],[7,8],[9,10,11,12,13,14,15,16,17,18,21,24],[19,20,22,23,25]]
part_attr=[[1,2],[4,0],[3,6],[7,5]]

class TransformerClassifier(nn.Module):
    def __init__(self,args, attr_num,attr_words, dim=512, pretrain_path='/media/sdb/**/pretrained/jx_vit_base_p16_224-80ecf9dd.pth'):#
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(512, dim)
        self.visual_embed= nn.Linear(512, dim)
        # self.part_select=Part_Attention()
        self.feat_cam=Global_CAM()
        self.clip_model, _ =  clip.load("ViT-B/16", device='cuda',download_root='/media/sdb/**/pretrained')
        self.clip_model=self.clip_model.float()
        self.attributes=attr_words
        self.lmbd=8

        # self.vit = vit_base()
        # self.vit.load_param(pretrain_path)
        
        # self.blocks_g = self.vit.blocks[-1:]
        self.blocks =nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=512,
                                       n_head=8,
                                       att_type='cross')])

        self.blocks_p =nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=512,
                                       n_head=8,
                                       att_type='cross')])

        self.blocks_part =nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=512,
                                       n_head=8,
                                       att_type='cross')])
        
        self.blocks_patch =nn.ModuleList([ResidualAttention(num_layers=1,
                                       d_model=512,
                                       n_head=8,
                                       att_type='cross')])

        self.dim=dim
        self.norm = nn.LayerNorm(self.dim)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.LayerNorm(self.attr_num)

        self.norm_p = nn.LayerNorm(self.dim)
        self.weight_layer_p = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn_p = nn.LayerNorm(self.attr_num)

        self.norm_patch = nn.LayerNorm(self.dim)
        self.weight_layer_patch = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn_patch = nn.LayerNorm(self.attr_num)
        
        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.text = clip.tokenize(attr_words).cuda()
        self.dim=dim
        # self.weight_layer_part=[]
        # self.blocks_part=[]
        # self.norm_part=[]
        # for i in range(4):
        #     self.weight_layer_part.append(nn.ModuleList([nn.Linear(dim, 1) for j in range(len(part_order[i]))]))
        #     self.blocks_part.append(copy.deepcopy(self.blocks))
        #     self.norm_part.append(copy.deepcopy(self.norm))
        
        # self.bn_part=nn.LayerNorm(self.attr_num)
        self.descrip=petabaseDataset(args.datapath)
        # self.apply(self._init_weights)

        self.head = nn.Linear(dim, self.attr_num)#nn.Conv1d(self.dim, self.attr_num, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)
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
        # word_embed=self.clip_model.encode_text(clip.tokenize(self.attributes).to(imgs.device)).to(imgs.device).float() #ViT_model.encode_text(self.text).float()
        #features=self.visual_embed(features)

        # cls_token_out = features[:, 0]
        # # Average pool
        # fp = features[:, 1:5]
        
        # # Add global cls token to each local token 
        # for i in range(4):
        #     out = torch.mul(fp[:, i, :], self.lmbd)
        #     features[:,i+1,:] = torch.div(torch.add(cls_token_out,out), 1+self.lmbd)
        attn=self.clip_model.visual.transformer.attn_weights#self.vit.attn_weights
        global_cam = self.feat_cam(attn,features)
        # t=features
        # for blk in self.blocks_g:
        #     t=blk(t)
        # features_g=t[:,0]
        part_tokens= features[:,0].unsqueeze(1) + self.cls_part_token #features[:,0]
        patch_embed = features[:,1:] #+ self.tex_embed
        token_embed = part_tokens #+ self.vis_embed

        features_p=[]
        part_cam=[]
        partlist=[[0,N-1],[0,(N-1)//2],[(N-1)//4,(N-1)*3//4],[(N-1)//2,N-1]]
        for i in range(4):
            for blk in self.blocks_p:#_part[k-1]
                t = blk(token_embed[:,i].unsqueeze(1),patch_embed[:,partlist[i][0]:partlist[i][1]])
            features_p.append(t)
            attn_part=self.blocks_p[0].attn_weight##
            part_cam.append(attn_part@patch_embed[:,partlist[i][0]:partlist[i][1]])
            #part_cam.append(self.feat_cam(attn_part,patch_embed[:,partlist[i][0]:partlist[i][1]]))
        features_p=torch.cat(features_p,dim=1)
        part_cam=torch.cat(part_cam,dim=1)
        

        
       
        word_embed = (word_vec).expand(features.shape[0], word_vec.shape[0], features.shape[-1])
        
        part_embed=[]
        for j in range(4):
            part_embed.append(torch.gather(word_embed,dim=1,index=torch.tensor(part_order[j]).unsqueeze(0).unsqueeze(-1).expand(B,-1,self.dim).cuda()))

        
       
            
        # attn=self.clip_model.visual.transformer.attn_weights

        

        logits=[]
        logits_part=[]
        vit_cls_p=[]
        for k in range(5):
            if k==0:
                # tt=[]
                
                # for i in range(B):
                #     emb_select_cls=features[i,part_inxd[k][i]]
                #     t=torch.cat((features[i,k].unsqueeze(0),emb_select_cls),dim=0)#
                #     tt.append(t)
                   
                # features_cls_temp=torch.stack(tt)
                
                tex_embed = word_embed #+ self.tex_embed
                vis_embed = features[:,0].unsqueeze(1)#features_cls_temp #+ self.vis_embed  features_g
                # x = torch.cat([tex_embed, vis_embed], dim=1)
                # image_mask_last=torch.zeros(x.size(1), x.size(1))
                for blk in self.blocks:
                    # blk.attn_mask=image_mask_last
                    x = blk(tex_embed,vis_embed)
                    # x = blk(x)
                x = self.norm(x)
                vit_cls_g=x[:,:self.attr_num,:]
                b= torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
                logits.append(self.bn(b))
            else:
                # tt=[]
                
                # for i in range(B):
                #     # part_select=list(set(part_select))
                #     emb_select_part=features[i,part_inxd[k][i]]
                #     t=torch.cat((features[i,k].unsqueeze(0),emb_select_part),dim=0)
                #     tt.append(t)
                    
                # features_temp=torch.stack(tt)
                
                
                tex_embed = part_embed[k-1] #+ self.tex_embed
                vis_embed = features_p[:,k-1].unsqueeze(1) #+ self.vis_embed

                # x = torch.cat([tex_embed, vis_embed], dim=1)
                # image_mask_last=torch.zeros(x.size(1), x.size(1))
                # self.blocks_part[k-1].cuda()
                for blk in self.blocks_part:#_part[k-1]
                    # blk.attn_mask=image_mask_last
                    x = blk(tex_embed,vis_embed)
                    # x=blk(x)
                x = self.norm_p(x)
                # vit_cls_p.append(x[:,:len(part_order[k-1]),:])
                vit_cls_p.append(x)
               
        vit_cls_p=torch.cat((vit_cls_p[1],vit_cls_p[2][:,5:],vit_cls_p[3],vit_cls_p[2][:,:5],vit_cls_p[0]),dim=1)
        b=torch.cat([self.weight_layer_p[i](vit_cls_p[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits.append(self.bn_p(b))

        # V=self.head(features[:,1:]).permute([0,2,1])
        # A=V.softmax(dim=-1)
        # spec=A@features[:,1:]
        spec=torch.cat((global_cam,part_cam),dim=1)
        
        # feat_g=self.head_g(features[:,0])
        # feat_g=self.bn_g(feat_g)
        # logits.append(feat_g)

        tex_embed = word_embed #+ self.tex_embed
        vis_embed = spec  #+ self.vis_embed #features_cls_temp #+ self.pos_embed
       
        for blk in self.blocks:     
            x = blk(tex_embed,vis_embed) 
        x = self.norm(x)
        b= torch.cat([self.weight_layer_patch[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits.append(self.bn(b))

       

        # cls_descrip=[]
        # group_descrip=[description[i].split(';')[:8] for i in range(len(description))]
        # [cls_descrip.append(''.join(tx)) for tx in group_descrip]
        # tk_description = clip.tokenize(cls_descrip, truncate=True)
        # cap_emb=self.clip_model.encode_text(tk_description.to(imgs.device)).to(imgs.device).float()
        # cap_emb=self.word_embed(cap_emb)
        # loss_itc = get_contrastive_loss(cls_token_out, cap_emb,self.clip_model, idx=None) + get_contrastive_loss(features_select_cls, cap_emb,self.clip_model, idx=None)
        # loss_itc_part=0
        # for i in range(4):
        #     part_slect=np.array(group_descrip)[:,part_attr[i]].tolist()
        #     part_descrip=[''.join(tx) for tx in part_slect]
        #     tk_part = clip.tokenize(part_descrip, truncate=True)
        #     part_emb=self.clip_model.encode_text(tk_part.to(imgs.device)).to(imgs.device).float()
        #     part_emb=self.word_embed(part_emb)
        #     loss_itc_part += get_contrastive_loss(features[:,i+1], part_emb,self.clip_model, idx=None) + get_contrastive_loss(features_select_part, part_emb,self.clip_model, idx=None)


        return logits,all_x_cls,vit_cls_g,vit_cls_p#loss_itc,loss_itc_part,


        
    
class Part_Attention_clip(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        max_p=[]
        max_inx=[]
        b=x[0].shape[0]
        att_tt=[]
        for d in range(length):
            att_ts=[]
            for e in range(5):
                att_tk=x[d][:,e]
                if e==0 or e==1:
                    att_pt=x[d][:,5:]
                elif e==2:
                    att_pt=x[d][:,5:103]
                elif e==3:
                    att_pt=x[d][:,54:152]
                else:
                    att_pt=x[d][:,103:]
                att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
                att_tk2=att_t1[:,:,e]
                if e==0 or e==1: 
                    att_pt2=att_t1[:,:,5:]
                elif e==2:
                    att_pt2=att_t1[:,:,5:103]
                elif e==3:
                    att_pt2=att_t1[:,:,54:152]
                else:
                    att_pt2=att_t1[:,:,103:]
                att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
                att_ts.append(att)
            att_tt.append(att_ts)
        att_map=[]
        for f in range(5):
            last_map =att_tt[0][f]
            for i in range(1, length):
                last_map = torch.matmul(att_tt[i][f], last_map)
            att_map.append(last_map)
        for k in range(5): 
            last_map1 = att_map[k][:,0,1:]
            max_g, max_inx_g = last_map1.topk(1,dim=-1, sorted=False)
            max_p.append(max_g)
            if k==0 or k==1: 
                max_inx.append(max_inx_g+5)
            elif k==2:
                max_inx.append(max_inx_g+5)
            elif k==3:
                max_inx.append(max_inx_g+54)
            else:
                max_inx.append(max_inx_g+103)
        return torch.stack(max_p,dim=0), torch.stack(max_inx,dim=0)   

class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        max_p=[]
        max_inx=[]
        b=x[0].shape[0]
        att_tt=[]
        for d in range(length):
            att_ts=[]
            for e in range(5):
                att_tk=x[d][:,e]
                if e==0 or e==1:
                    att_pt=x[d][:,5:]
                elif e==2:
                    att_pt=x[d][:,5:69]
                elif e==3:
                    att_pt=x[d][:,37:101]
                else:
                    att_pt=x[d][:,69:]
                att_t1=torch.cat((att_tk.unsqueeze(1),att_pt),dim=1)
                att_tk2=att_t1[:,:,e]
                if e==0 or e==1: 
                    att_pt2=att_t1[:,:,5:]
                elif e==2:
                    att_pt2=att_t1[:,:,5:69]
                elif e==3:
                    att_pt2=att_t1[:,:,37:101]
                else:
                    att_pt2=att_t1[:,:,69:]
                att=torch.cat((att_tk2.unsqueeze(2),att_pt2),dim=2)
                att_ts.append(att)
            att_tt.append(att_ts)
        att_map=[]
        for f in range(5):
            last_map =att_tt[0][f]
            for i in range(1, length):
                last_map = torch.matmul(att_tt[i][f], last_map)
            att_map.append(last_map)
        for k in range(5): 
            last_map1 = att_map[k][:,0,1:]
            max_g, max_inx_g = last_map1.topk(1,dim=-1, sorted=False)
            max_p.append(max_g)
            if k==0 or k==1: 
                max_inx.append(max_inx_g+5)
            elif k==2:
                max_inx.append(max_inx_g+5)
            elif k==3:
                max_inx.append(max_inx_g+37)
            else:
                max_inx.append(max_inx_g+69)
        return torch.stack(max_p,dim=0), torch.stack(max_inx,dim=0)   

class Part_Attention1(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        max_p=[]
        max_inx=[]
        last_map =x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        for k in range(5): 
            last_map1 = last_map[:,k,5:]
            max_g, max_inx_g = last_map1.topk(1,dim=-1, sorted=False)
            max_p.append(max_g)
            max_inx.append(max_inx_g+5)
        return torch.stack(max_p,dim=0), torch.stack(max_inx,dim=0) 

class Global_CAM(nn.Module):
    def __init__(self):
        super(Global_CAM, self).__init__()

    def forward(self, x,features):
        length = len(x)
        # feat_cam=[]
        # N=features.shape[1]
        # feats_patch=[features[:,1:],features[:,1:],features[:,1:((N-1)//2 + 1)],features[:,((N-1)//4 + 1):(3*(N-1)//4 + 1)],features[:,((N-1)//2 + 1):]]
        last_map =x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        
        last_map1 = last_map[:,0,1:].unsqueeze(1)
        feat_cam=last_map1@features[:,1:]
        
        return feat_cam  
        
        
         

def get_contrastive_loss(image_feat, text_feat,model, idx=None):
        # assert image_feat.size(-1) == self.embed_dim
        # assert text_feat.size(-1) == self.embed_dim
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        image_feat_all = image_feat#allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = text_feat#allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() 
        logits=logits* model.logit_scale.exp()
        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            return (loss_i2t + loss_t2i) / 2
        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = idx#allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
            return (loss_i2t + loss_t2i) / 2
     
