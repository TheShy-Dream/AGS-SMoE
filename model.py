import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, FusionTrans, Encoder, \
    SelfAttention, CrossAttention, CrossAttentionWithMOE, CLUB, Projector, SelfAttentionAudio, SelfAttentionVision
from modules.InfoNCE import InfoNCE
from transformers import BertModel, BertConfig,BertTokenizer




class MMIM(nn.Module):

    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()

        def cal_avg_grad_norm(module, grad_input, grad_output):
            #grad_input_=grad_input
            avg_grad_norm = 0
            param_count = 0
            for grad in grad_input:
                if grad is not None:
                    avg_grad_norm += grad.norm(2).item()
                    param_count += torch.numel(grad)
            avg_grad_norm /= param_count
            if "Language" in module.__class__.__name__: #算的当前批次的倾向性分数
                self.text_grad_list.append(avg_grad_norm)
                if len(self.audio_grad_list) > 1 and len(self.vision_grad_list) > 1 and len(self.text_grad_list) > 1:
                    if self.text_grad_list[-1] > (self.audio_grad_list[-1] + self.vision_grad_list[-1]) / 2:
                        ratio = self.text_grad_list[-1] / ((self.audio_grad_list[-1] + self.vision_grad_list[-1]) / 2)
                        reversed_ratio = 1.0 / ratio
                        final_ratio = 1.0 / (1.0 + reversed_ratio)
                        factor=1.0 - math.tanh(self.hp.hyperalpha * final_ratio)
                        grad_input = tuple(x * factor for x in grad_input)
                        return grad_input
                    else:
                        grad_input = grad_input
                        return grad_input
                else:
                    pass
            elif "Audio" in module.__class__.__name__: #计算之前批次的倾向性分数
                if len(self.audio_grad_list) > 1 and len(self.vision_grad_list) > 2 and len(self.text_grad_list) > 1:
                    if self.audio_grad_list[-1] > (self.text_grad_list[-1] + self.vision_grad_list[-2]) / 2:
                        ratio = self.audio_grad_list[-1] / ((self.text_grad_list[-1] + self.vision_grad_list[-2]) / 2)
                        reversed_ratio = 1.0 / ratio
                        final_ratio = 1.0 / (1.0 + reversed_ratio)
                        factor=1.0 - math.tanh(self.hp.hyperalpha * final_ratio)
                        grad_input = tuple(x * factor for x in grad_input)
                        return grad_input
                    else:
                        grad_input = grad_input
                        return grad_input
                else:
                    pass
                self.audio_grad_list.append(avg_grad_norm)
            elif "Vision" in module.__class__.__name__: #计算之前批次的倾向性分数 最先更新的
                if len(self.vision_grad_list) > 1 and len(self.text_grad_list) > 1 and len(self.audio_grad_list) > 1:
                    if self.vision_grad_list[-1] > (self.text_grad_list[-1] + self.audio_grad_list[-1]) / 2:
                        ratio = self.vision_grad_list[-1] / ((self.text_grad_list[-1] + self.audio_grad_list[-1]) / 2)
                        reversed_ratio = 1.0 / ratio
                        final_ratio = 1.0 / (1.0 + reversed_ratio)
                        factor=1.0 - math.tanh(self.hp.hyperalpha * final_ratio)
                        grad_input = tuple(x * factor for x in grad_input)
                        return grad_input
                    else:
                        grad_input = grad_input
                        return grad_input
                else:
                    pass
                self.vision_grad_list.append(avg_grad_norm)
            #print(f'Average gradient norm for {module.__class__.__name__}: {avg_grad_norm}')
        self.hp = hp

        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin
        self.text_grad_list = []
        self.audio_grad_list = []
        self.vision_grad_list = []

        self.uni_text_enc = LanguageEmbeddingLayer(hp)  # BERT Encoder
        self.uni_visual_enc = RNNEncoder(  # 视频特征提取
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.uni_acoustic_enc = RNNEncoder(  # 音频特征提取
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        # For MI maximization   互信息最大化
        # Modality Mutual Information Lower Bound（MMILB）
        self.mi_tv = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_vout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        if hp.add_va:  # 一般是tv和ta   若va也要MMILB
            self.mi_va = MMILB(
                x_size=hp.d_vout,
                y_size=hp.d_aout,
                mid_activation=hp.mmilb_mid_activation,
                last_activation=hp.mmilb_last_activation
            )

        # CPC MI bound   d_prjh是什么？？？
        self.cpc_zt = CPC(
            x_size=hp.d_tout,  # to be predicted  各个模态特征提取后得到的维度
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size=hp.d_aout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        self.uni_audio_encoder = SelfAttentionAudio(hp, d_in=hp.d_ain, d_model=hp.model_dim_self,
                                                   nhead=hp.num_heads_self,
                                                   dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                   num_layers=hp.num_layers_self)
        self.uni_vision_encoder = SelfAttentionVision(hp, d_in=hp.d_vin, d_model=hp.model_dim_self,
                                                    nhead=hp.num_heads_self,
                                                    dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                    num_layers=hp.num_layers_self)
        if self.hp.dataset == "sims":
            self.audio_classifer = SubNet(in_size=hp.d_aout, hidden_size=hp.d_aout*2,
                                    n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_ain]
            self.vision_classifer = SubNet(in_size=hp.d_vout, hidden_size=hp.d_vout*2,
                                     n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_vin]
            self.text_classifer = SubNet(in_size=hp.d_tin, hidden_size=hp.d_tin*2,
                                   n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_tin]

        self.projector_ta=Projector(hp.model_dim_cross,hp.model_dim_cross*2,hp.model_dim_cross)
        self.projector_tv = Projector(hp.model_dim_cross, hp.model_dim_cross * 2, hp.model_dim_cross)

        self.audio_mlp=SubNet(in_size=hp.d_ain+hp.model_dim_cross,hidden_size=hp.audio_mlp_hidden_size,n_class=None,dropout=hp.dropout_prj,output_size=hp.d_ain)#[bs,seq_len,d_ain]
        self.vision_mlp=SubNet(in_size=hp.d_vin+hp.model_dim_cross,hidden_size=hp.vision_mlp_hidden_size,n_class=None,dropout=hp.dropout_prj,output_size=hp.d_vin)#[bs,seq_len,d_vin]
        self.text_mlp=SubNet(in_size=hp.d_tin+hp.model_dim_cross,hidden_size=hp.text_mlp_hidden_size,n_class=None,dropout=hp.dropout_prj,output_size=hp.d_tin)#[bs,seq_len,d_tin]

        # 用MULT融合 每个模块的输出都是[bs,query_length,model_dim_cross]
        self.ta_cross_attn=CrossAttentionWithMOE(hp,d_modal1=hp.d_tin,d_modal2=hp.d_ain,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross,num_experts=hp.num_experts,top_k=hp.top_k)
        self.tv_cross_attn=CrossAttentionWithMOE(hp,d_modal1=hp.d_tin,d_modal2=hp.d_vin,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross,num_experts=hp.num_experts,top_k=hp.top_k)

        self.fusion_mlp_for_regression = SubNet(in_size=hp.model_dim_cross*2,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)


    def gen_mask(self, a, length=None):
        if length is None:
            msk_tmp = torch.sum(a, dim=-1)
            # 特征全为0的时刻加mask
            mask = (msk_tmp == 0)
            return mask
        else:
            b = a.shape[0]
            l = a.shape[1]
            msk = torch.ones((b, l))
            x = []
            y = []
            for i in range(b):
                for j in range(length[i], l):
                    x.append(i)
                    y.append(j)
            msk[x, y] = 0
            return (msk == 0)

    def D(self,p, z): #距离度量函数
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    def SimSiamLoss(self,p1, z1, p2, z2): #损失函数
        return self.D(p1, z2) / 2 + self.D(p2, z1) / 2

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None,
                mem=None,v_mask=None,a_mask=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        sentences: torch.Size([0, 32])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        For Bert input, the length of text is "seq_len + 2"
        """
        # bs=visual.size(1)
        # compensate_matrix_a = self.learned_queries_a.unsqueeze(0).expand(bs, -1, -1).permute(1,0,2)
        # compensate_matrix_v = self.learned_queries_v.unsqueeze(0).expand(bs, -1, -1).permute(1,0,2)
        #compensate_matrix = torch.zeros((self.hp.learned_query_size,bs, self.hp.model_dim_cross),requires_grad=False).cuda()
        with torch.no_grad():
            maskT = (bert_sent_mask == 0)
            maskV = self.gen_mask(visual.transpose(0,1),v_len)
            maskA = self.gen_mask(acoustic.transpose(0,1),a_len)

        enc_word= self.uni_text_enc(sentences, bert_sent, bert_sent_type,bert_sent_mask)  # 32*50*768 (batch_size, seq_len, emb_size)
        text_trans = enc_word.transpose(0, 1)  # torch.Size([50, 32, 768]) (seq_len, batch_size,emb_size)
        loss_vq=torch.tensor(0)
        loss_commitment=torch.tensor(0)

        acoustic = self.uni_audio_encoder(acoustic)  # [seq_len,bs,dim] 自注意力
        visual = self.uni_vision_encoder(visual)  # [seq_len,bs,dim] 自注意力

        vision_trans = visual
        audio_trans = acoustic
        # 2. 跨模态注意力部分
        cross_tv = self.tv_cross_attn(text_trans, vision_trans,Tmask=maskT,Vmask=maskV).mean(dim=0)
        cross_ta = self.ta_cross_attn(text_trans, audio_trans,Tmask=maskT,Amask=maskA).mean(dim=0)
        fusion, preds = self.fusion_mlp_for_regression(torch.cat([cross_ta, cross_tv], dim=1))  # 32*128,32*1
        if self.training:
            text = enc_word[:,0,:] # 32*768 (batch_size, emb_size)
            acoustic = self.uni_acoustic_enc(acoustic, a_len)  # 32*16
            visual = self.uni_visual_enc(visual, v_len)  # 32*16

            if self.hp.dataset == "sims":
                _,acoustic_pred = self.audio_classifer(acoustic)
                _,visual_pred = self.vision_classifer(visual)
                _,text_pred = self.text_classifer(text)
            else:
                text_pred=torch.tensor(0.0,device="cuda:0")
                visual_pred=torch.tensor(0.0,device="cuda:0")
                acoustic_pred=torch.tensor(0.0,device="cuda:0")

            if y is not None:
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
                # for ablation use
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
            else:  # 默认进这
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)  # mi_tv 模态互信息
                # lld_tv:-2.1866  tv_pn:{'pos': None, 'neg': None}  H_tv:0.0
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)

            nce_t = self.cpc_zt(text, fusion)  # 3.4660
            nce_v = self.cpc_zv(visual, fusion)  # 3.4625
            nce_a = self.cpc_za(acoustic, fusion)  # 3.4933

            nce = nce_t + nce_v + nce_a  # 10.4218  CPC loss

            pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn if self.add_va else None}
            # {'tv': {'pos': None, 'neg': None}, 'ta': {'pos': None, 'neg': None}, 'va': None}
            lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)  # -5.8927
            H = H_tv + H_ta + (H_va if self.add_va else 0.0)
        if self.training:
            return lld, nce, preds, pn_dic, H,loss_vq,loss_commitment,None,text_pred,acoustic_pred,visual_pred
        else:
            return None,None, preds, None, None,None,None,None,None,None,None



if __name__=="__main__":
    net=Encoder(4, 8, 2,32,0.1,'relu',2)
    data=torch.randn(30,32,4)
    data_mask=pad_sequence([torch.zeros(torch.FloatTensor(sample).size(0)) for sample in data])
    data_mask[:,4:].fill_(float(1.0))
    output=net(data,data_mask.transpose(1,0))
    print(data_mask,data)