r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn.functional as F
from torchvision.models import resnet

from model.base.feature import extract_feat_res
from model.base.correlation import Correlation
from model.ifsl import iFSLModule
from model.module.hsnet import HPNLearner

from einops import rearrange


class HypercorrSqueezeNetwork(iFSLModule):
    def __init__(self, args):
        super(HypercorrSqueezeNetwork, self).__init__(args)

        # 1. Backbone network initialization
        self.backbone_type = args.backbone

        if args.backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif args.backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()

        # explicitly freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.learner = HPNLearner(list(reversed(nbottlenecks[-3:])), args.way)

    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, 3, H, W]
        support_masks.shape : [bsz, way, H, W]
        '''
        # extract the backbone (e.g. ResNet50) features without fine-tuning.
        with torch.no_grad():
            # NOTE: default implementation only support 1-shot inference, to do multiple shots, we need to modify the implementation.
            support_img = rearrange(batch['support_imgs'], 'b s c h w -> (b s) c h w')
            support_mask = rearrange(batch['support_masks'], 'b s h w -> (b s) h w')
            # support_img = rearrange(batch['support_imgs'], 'b n s c h w -> (b n) s c h w')
            # support_mask = rearrange(batch['support_masks'], 'b n s h w -> (b n) s h w')
            query_img = batch['query_img']

            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.mask_feature(support_feats, support_mask.clone())
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids, self.args.way)

        shared_masks = self.learner(corr)

        # B, N, 2, H, W
        shared_masks = torch.log_softmax(shared_masks, dim=2)
        return shared_masks

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        cls_score_agg = 0
        support_imgs = batch['support_imgs'].clone()
        support_masks = batch['support_masks'].clone()
        for s_idx in range(nshot):
            batch['support_imgs'] = support_imgs[:, :, s_idx]
            batch['support_masks'] = support_masks[:, :, s_idx]
            shared_masks = self.forward(batch)
            pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)
            cls_score_agg += pred_cls.clone()
            logit_mask_agg += logit_seg.clone()
            if nshot == 1:
                return pred_cls, pred_seg
        # voting for nshot prediction
        pred_cls = (cls_score_agg / float(nshot)) >= 0.5
        pred_seg = (logit_mask_agg / float(nshot)).argmax(dim=1)
        return pred_cls, pred_seg

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def configure_optimizers(self):
        return torch.optim.Adam([{"params": self.parameters(), "lr": self.args.lr}])
