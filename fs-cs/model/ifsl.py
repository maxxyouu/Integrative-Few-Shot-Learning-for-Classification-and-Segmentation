import math
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

class iFSLModule(pl.LightningModule):
    """
    """
    def __init__(self, args):
        super(iFSLModule, self).__init__()

        self.args = args
        self.way = self.args.way
        self.weak = args.weak
        self.range = torch.arange(args.way + 1, requires_grad=False).view(1, args.way + 1, 1, 1)
        self.learner = None
        self.dice = args.dice

    def forward(self, batch):
        pass

    def train_mode(self):
        pass

    def configure_optimizers(self):
        pass

    def predict_mask_nshot(self, batch, nshot):
        pass

    def training_step(self, batch, batch_idx):
        """
        batch.keys()
        > dict_keys(['query_img', 'query_mask', 'query_name', 'query_ignore_idx', 'org_query_imsize', 'support_imgs', 'support_masks', 'support_names', 'support_ignore_idxs', 'class_id'])

        batch['query_img'].shape : [bsz, 3, H, W]
        batch['query_mask'].shape : [bsz, H, W]
        batch['query_name'].len : [bsz]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['org_query_imsize'].len : [bsz]
        batch['support_imgs'].shape : [bsz, way, shot, 3, H, W]
        batch['support_masks'].shape : [bsz, way, shot, H, W]
        # FYI: this support_names' shape is transposed so keep in mind for vis
        batch['support_names'].shape : [bsz, shot, way]
        batch['support_ignore_idxs'].shape: [bsz, way, shot, H, W]
        batch['support_classes'].shape : [bsz]
        batch['support_classes'].shape : [bsz, way] (torch.int64)
        batch['query_class_presence'].shape : [bsz, way] (torch.bool)
        # FYI: K-shot is always fixed to 1 for training
        """

        split = 'trn' if self.training else 'val'
        shared_masks = self.forward(batch)
        pred_cls, pred_seg, logit_seg = self.predict_cls_and_mask(shared_masks, batch)

        # if self.weak:
        #     loss = self.compute_cls_objective(shared_masks, batch['query_class_presence'])
        # else:
        if self.dice:
            loss = self.compute_soft_dice_loss(logit_seg, batch['query_mask'])
        else:
            loss = self.compute_seg_objective(logit_seg, batch['query_mask'])

        with torch.no_grad():
            self.average_meter.update_cls(pred_cls, batch['query_class_presence'])
            self.average_meter.update_seg(pred_seg, batch, loss.item())

            self.log(f'{split}/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
        return loss

    def on_train_epoch_end(self):
        self._shared_epoch_end()

    def validation_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self.training_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self._shared_epoch_end()

    def _shared_epoch_end(self):
        split = 'trn' if self.training else 'val'
        miou = self.average_meter.compute_iou()
        er = self.average_meter.compute_cls_er()
        loss = self.average_meter.avg_seg_loss()

        dict = {f'{split}/loss': loss,
                f'{split}/miou': miou,
                f'{split}/er': er}

        for k in dict:
            self.log(k, dict[k], on_epoch=True, logger=True)

        space = '\n\n' if split == 'val' else '\n'
        print(f'{space}[{split}] ep: {self.current_epoch:>3}| {split}/loss: {loss:.3f} | {split}/miou: {miou:.3f} | {split}/er: {er:.3f}')

    def test_step(self, batch, batch_idx):
        pred_cls, pred_seg = self.predict_mask_nshot(batch, self.args.shot)
        er_b = self.average_meter.update_cls(pred_cls, batch['query_class_presence'], loss=None)
        iou_b = self.average_meter.update_seg(pred_seg, batch, loss=None)

        if self.args.vis:
            print(batch_idx, 'qry:', batch['query_name'])
            print(batch_idx, 'spt:', batch['support_names'])
            if self.args.shot > 1: raise NotImplementedError
            if self.args.weak:
                batch['support_masks'] = torch.zeros(1, self.way, 400, 400).cuda()
            from common.vis import Visualizer
            Visualizer.initialize(True, self.way)
            Visualizer.visualize_prediction_batch(batch['support_imgs'].squeeze(2),
                                                batch['support_masks'].squeeze(2),
                                                batch['query_img'],
                                                batch['query_mask'],
                                                batch['org_query_imsize'],
                                                pred_seg,
                                                batch_idx,
                                                iou_b=iou_b,
                                                er_b=er_b,
                                                to_cpu=True)

    def test_epoch_end(self, test_step_outputs):
        miou = self.average_meter.compute_iou()
        er = self.average_meter.compute_cls_er()
        length = 16
        dict = {'benchmark'.ljust(length): self.args.benchmark,
                'fold'.ljust(length): self.args.fold,
                'test/miou'.ljust(length): miou.item(),
                'test/er'.ljust(length): er.item()}

        for k in dict:
            self.log(k, dict[k], on_epoch=True)

    def predict_cls_and_mask(self, shared_masks, batch):
        logit_seg = self.merge_bg_masks(shared_masks)
        logit_seg = self.upsample_logit_mask(logit_seg, batch)

        with torch.no_grad():
            pred_cls = self.collect_class_presence(shared_masks)
            pred_seg = logit_seg.argmax(dim=1) # either 0 or 1 for background and foreground

        return pred_cls, pred_seg, logit_seg

    def collect_class_presence(self, logit_mask):
        ''' logit_mask: B, N, 2, H, W '''
        # since logit_mask is log-softmax-ed, we use torch.log(0.5) for the threshold
        class_activation = logit_mask[:, :, 1].max(dim=-1)[0].max(dim=-1)[0] >= math.log(0.5)
        return class_activation.type(logit_mask.dtype).detach()

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch['query_img'].shape[-2:]
        else:
            spatial_size = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        return F.interpolate(logit_mask, spatial_size, mode='bilinear', align_corners=True)

    def compute_seg_objective(self, logit_mask, gt_mask):
        ''' supports 1-way training '''
        return F.nll_loss(logit_mask, gt_mask.long())
    
    def compute_soft_dice_loss(self, logit_mask, gt_mask):
        return self._dice_loss(logit_mask, gt_mask)

    def _dice_loss(self, input, target):
        """
        input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input
        
        code obtained from https://github.com/pytorch/pytorch/issues/1249
        and https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708

        """
        ## ------------------ matching the target tensor shape to the input tensor shape START -------------------------
        # for safety only, target.requires_grad is false
        target_fore = target.clone().detach()
        target_back = target.clone().detach()

        # perform label flipping to get the background groundtruth
        target_back[target_back > 0.] = 2. # foregound is set to 2 for future replacement 
        target_back[target_back < 1.] = 1. # background is set to 1. as the groundtruth label
        target_back[target_back == 2.] = 0. # reset the foreground labels of the target oject as background
        target = torch.stack([target_back, target_fore], dim=1)
        ## ------------------ matching the target tensor shape to the input tensor shape END -------------------------

        assert input.size() == target.size(), "Input sizes must be equal."
        assert input.dim() == 4, "Input must be a 4D Tensor."
        uniques=np.unique(target.cpu().numpy())
        assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

        probs=F.softmax(input)
        num=probs*target#b,c,h,w--p*g
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)
        

        den1=probs*probs#--p^2
        den1=torch.sum(den1,dim=3)#b,c,h
        den1=torch.sum(den1,dim=2)
        

        den2=target*target#--g^2
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c
        
        dice=2*(num/(den1+den2))
        dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

        # since we want to minimize the loss function, we put a negative here to get
        # the equivalent operation as maximize the dice score.
        dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
        return dice_total
    
    # def dice_loss_V2(self, input, target):
    #     # smooth = 1.
    #     smooth = 0.
    #     iflat = input.view(-1)
    #     tflat = target.view(-1)
    #     intersection = (iflat * tflat).sum()
        
    #     # return 1 - ((2. * intersection + smooth) /
    #     #         (iflat.sum() + tflat.sum() + smooth))
    #     return ((2. * intersection + smooth) /
    #             (iflat.sum() + tflat.sum() + smooth))

    def compute_cls_objective(self, shared_masks, gt_presence):
        ''' supports 1-way training '''
        # B, N, 2, H, W -> B, N, 2 -> B, 2
        prob_avg = shared_masks.mean(dim=[-1, -2]).squeeze(1)
        return F.nll_loss(prob_avg, gt_presence.long().squeeze(-1))

    def merge_bg_masks(self, shared_fg_masks):
        # resulting shape: B, N-way, H, W;
        logit_fg = shared_fg_masks[:, :, 1]
        # resulting shape B, 1, H, W
        logit_episodic_bg = shared_fg_masks[:, :, 0].mean(dim=1)
        # B, (1 + N), H, W  NOTE: there is no difference for 1-way
        logit_mask = torch.cat((logit_episodic_bg.unsqueeze(1), logit_fg), dim=1)
        return logit_mask

    def get_progress_bar_dict(self):
        # to stop to show the version number in the progress bar
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
