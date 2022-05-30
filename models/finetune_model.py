import sys
sys.path.append('../')
from config import args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .lxrt import LXRTModel, GeLU, BertLayerNorm
from .uniter import UniterModel

class ClassificationModel(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.bert = LXRTModel.from_pretrained(args.bert_dir)
        hid_dim = 768

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_class)
        )
        self.logit_fc.apply(self.bert.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None,
                category_label=None, inference=False):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            visual_feats=visual_feats, visual_attention_mask=visual_attention_mask
        )
        logit = self.logit_fc(pooled_output) # [B, cat_len]

        if inference:
            return torch.argmax(logit, dim=1)
        else:
            return self.cal_loss(logit, category_label)

    @staticmethod
    def cal_loss(prediction, label):
        focal = FocalLoss(class_num=prediction.size(1))
        label = label.squeeze(dim=1)
        # loss = F.cross_entropy(prediction, label)
        loss = focal(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class FinetuneUniterModel(nn.Module):

    def __init__(self, num_class):
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None,
                category_label=None, inference=False):
        _, pooled_output = self.bert(
            input_ids, visual_feats, attention_mask, visual_attention_mask
        )
        logit = self.logit_fc(pooled_output) # [B, cat_len]

        if inference:
            return torch.argmax(logit, dim=1)
        else:
            return self.cal_loss(logit, category_label)

    @staticmethod
    def cal_loss(prediction, label):
        focal = FocalLoss(class_num=prediction.size(1))
        label = label.squeeze(dim=1)
        #loss = F.cross_entropy(prediction, label)
        loss = focal(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        '''
        inputs: [N, C]
        '''
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        P = P.clamp(min=0.0001, max=1.0)

        class_mask = inputs.data.new(N, C).fill_(0) # a new all-0s tensor having same type and device as inputs
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# test Focal loss
if __name__ == "__main__":
    inputs = torch.rand(size=(32, 10)).cuda()
    target = torch.argmax(torch.rand(size=(32, 10)), dim=1, keepdim=False).cuda()
    focal = FocalLoss(class_num=10)
    print(focal(inputs, target))
