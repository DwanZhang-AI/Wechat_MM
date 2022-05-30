import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from models.lxrt import LXRTModel, GeLU, BertLayerNorm
from models.uniter import UniterModel

from category_id_map import CATEGORY_ID_LIST


class MultiModal_a(nn.Module):
    def __init__(self, args, num_class):
        super().__init__()
        self.num_class = num_class
        self.bert = UniterModel.from_pretrained(args.bert_dir, img_dim=768)
        hid_dim = self.bert.config.hidden_size

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_class)
        )
        # self.logit_fc.apply(self.bert.init_bert_weights)
        init_weights_a(self.logit_fc)

    def forward(self, batch, inference=False):
        _, pooled_output = self.bert(
            batch['title_input'], batch['frame_input'], batch['title_mask'], batch['frame_mask']
        )
        logit = self.logit_fc(pooled_output) # [B, cat_len]

        if inference:
            return torch.argmax(logit, dim=1)
        else:
            return logit


class MultiModal_b(nn.Module):
    def __init__(self, args, num_class):
        super().__init__()
        self.num_class = num_class
        self.bert = UniterModel.from_pretrained(args.bert_dir, img_dim=768)
        hid_dim = self.bert.config.hidden_size

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_class)
        )
        self.logit_fc.apply(self.bert.init_bert_weights)

    def forward(self, batch, inference=False):
        _, pooled_output = self.bert(
            batch['title_input'], batch['frame_input'], batch['title_mask'], batch['frame_mask']
        )
        logit = self.logit_fc(pooled_output) # [B, cat_len]

        if inference:
            return torch.argmax(logit, dim=1)
        else:
            return logit

class MultiModal(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    def forward(self, inputs, unlabel_inputs):
        output_a = self.model_a(inputs)
        output_b = self.model_b(inputs)

        un_output_a = self.model_a(unlabel_inputs)
        un_output_b = self.model_b(unlabel_inputs)

        un_gt_a = torch.argmax(un_output_b, dim=1).unsqueeze(1)
        un_gt_b = torch.argmax(un_output_a, dim=1).unsqueeze(1)

        loss_a, accuracy_a, pred_label_id_a, label_a = self.cal_loss(output_a, inputs['label'])
        loss_b, accuracy_b, pred_label_id_b, label_b = self.cal_loss(output_b, inputs['label'])
        
        
        un_loss_a, un_accuracy_a, un_pred_label_id_a, un_label_a = self.cal_loss(un_output_a, un_gt_a)
        un_loss_b, un_accuracy_b, un_pred_label_id_b, un_label_b = self.cal_loss(un_output_b, un_gt_b)

        label_loss = loss_a + loss_b
        unlabel_loss = un_loss_a + un_loss_b

        return loss_a, un_loss_a, loss_b, un_loss_b, accuracy_a, accuracy_b



# class NeXtVLAD(nn.Module):
#     def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
#         super().__init__()
#         self.feature_size = feature_size
#         self.output_size = output_size
#         self.expansion_size = expansion
#         self.cluster_size = cluster_size
#         self.groups = groups
#         self.drop_rate = dropout

#         self.new_feature_size = self.expansion_size * self.feature_size // self.groups

#         self.dropout = torch.nn.Dropout(self.drop_rate)
#         self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
#         self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
#         self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
#                                               bias=False)
#         self.cluster_weight = torch.nn.Parameter(
#             torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
#         self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

#     def forward(self, inputs, mask):
#         # todo mask
#         inputs = self.expansion_linear(inputs)
#         attention = self.group_attention(inputs)
#         attention = torch.sigmoid(attention)
#         attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
#         reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
#         activation = self.cluster_linear(reshaped_input)
#         activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
#         activation = torch.softmax(activation, dim=-1)
#         activation = activation * attention
#         a_sum = activation.sum(-2, keepdim=True)
#         a = a_sum * self.cluster_weight
#         activation = activation.permute(0, 2, 1).contiguous()
#         reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
#         vlad = torch.matmul(activation, reshaped_input)
#         vlad = vlad.permute(0, 2, 1).contiguous()
#         vlad = F.normalize(vlad - a, p=2, dim=1)
#         vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
#         vlad = self.dropout(vlad)
#         vlad = self.fc(vlad)
#         return vlad


# class SENet(nn.Module):
#     def __init__(self, channels, ratio=8):
#         super().__init__()
#         self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
#         self.relu = nn.ReLU()
#         self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         gates = self.sequeeze(x)
#         gates = self.relu(gates)
#         gates = self.excitation(gates)
#         gates = self.sigmoid(gates)
#         x = torch.mul(x, gates)

#         return x


# class ConcatDenseSE(nn.Module):
#     def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
#         super().__init__()
#         self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
#         self.fusion_dropout = nn.Dropout(dropout)
#         self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

#     def forward(self, inputs):
#         embeddings = torch.cat(inputs, dim=1)
#         embeddings = self.fusion_dropout(embeddings)
#         embedding = self.fusion(embeddings)
#         embedding = self.enhance(embedding)

#         return embedding

def init_weights_a(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def init_weights_b(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

