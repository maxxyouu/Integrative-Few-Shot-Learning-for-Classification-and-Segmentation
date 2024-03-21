from functools import reduce
from operator import add
import math
from torch import nn
import torch
from torchvision.models import resnet
import torch.nn.functional as F

from einops import rearrange
from model.base.feature import extract_feat_res
from model.ifsl import iFSLModule
from model.module.selective_kernel import SelectiveKernel
from model.module.scnet import SCBottleneck

#universeg related
from .universeg.nn import CrossConv2d, CrossSKConv2d, CrossSCConv2d
from .universeg.nn import reset_conv2d_parameters
from .universeg.nn import Vmap, vmap
from .universeg.validation import (Kwargs, as_2tuple, size2t, validate_arguments,
                        validate_arguments_init)
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
    if nonlinearity is None:
        return nn.Identity()
    if nonlinearity == "Softmax":
        # For Softmax, we need to specify the channel dimension
        return nn.Softmax(dim=1)
    if hasattr(nn, nonlinearity):
        return getattr(nn, nonlinearity)()
    raise ValueError(f"nonlinearity {nonlinearity} not found")


#@validate_arguments_init
#@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):

    # in_channels: int
    # out_channels: int
    # kernel_size: size2t = 3
    # nonlinearity: Optional[str] = "LeakyReLU"
    # init_distribution: Optional[str] = "kaiming_normal"
    # init_bias: Union[None, float, int] = 0.0

    def __init__(self, 
                in_channels: size2t, 
                out_channels: int, 
                kernel_size: size2t = 3, 
                nonlinearity: Optional[str] = "LeakyReLU", 
                init_distribution: Optional[str] = "kaiming_normal", 
                init_bias: Union[None, float, int] = 0.0,
                use_sc: bool = False,
                use_sk: bool = False,
                sk_split_input: bool =True,
                last_layer: bool =False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.init_distribution = init_distribution
        self.init_bias = init_bias

        # assert use_sc != use_sk
        if use_sc:
            self.conv = SCBottleneck(
                self.in_channels,
                self.out_channels,
                act_layer=nn.LeakyReLU if nonlinearity == 'LeakyReLU' else None,
                # TODO: might need to specify a padding, double check this.
                bias=True,
                last_layer=last_layer
            )
        elif use_sk: # for selective kernel option 
            self.conv = SelectiveKernel(
                self.in_channels,
                self.out_channels,
                act_layer=nn.LeakyReLU if nonlinearity == 'LeakyReLU' else None,
                # TODO: might need to specify a padding, double check this.
                bias=True,
                split_input=sk_split_input
            #padding_mode by default is "zeros"
            )
        else:
            self.conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode="zeros",
                bias=True,
            )

            if self.nonlinearity is not None:
                self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

#@validate_arguments_init
#@dataclass(eq=False, repr=False)
class CrossOp(nn.Module):

    # in_channels: size2t
    # out_channels: int
    # kernel_size: size2t = 3
    # nonlinearity: Optional[str] = "LeakyReLU"
    # init_distribution: Optional[str] = "kaiming_normal"
    # init_bias: Union[None, float, int] = 0.0

    def __init__(self, 
                in_channels: size2t, 
                out_channels: int, 
                kernel_size: size2t = 3, 
                nonlinearity: Optional[str] = "LeakyReLU", 
                init_distribution: Optional[str] = "kaiming_normal", 
                init_bias: Union[None, float, int] = 0.0,
                use_sk: bool = False,
                use_sc: bool = False,
                sk_split_input = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.init_distribution = init_distribution
        self.init_bias = init_bias

        # assert use_sc == True and use_sc != use_sk
        
        if use_sc:
            self.cross_conv = CrossSCConv2d(
                in_channels=as_2tuple(self.in_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
            )
        elif use_sk:
            self.cross_conv = CrossSKConv2d(
                in_channels=as_2tuple(self.in_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                sk_split_input = sk_split_input
                # padding=self.kernel_size // 2,
            )
            # make sure no more nonlinearity as internally in the CrossSKConv2d there is nonlinearity already
            assert self.nonlinearity is None
        else:
            self.cross_conv = CrossConv2d(
                in_channels=as_2tuple(self.in_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            )

            if self.nonlinearity is not None:
                self.nonlin = get_nonlinearity(self.nonlinearity)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, target, support):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)

        new_target = interaction.mean(dim=1, keepdims=True)

        return new_target, interaction
    

# #@validate_arguments_init
#@dataclass(eq=False, repr=False)
class CrossBlock(nn.Module):

    def __init__(self, 
                in_channels: size2t, 
                cross_features: int, 
                conv_features: Optional[int] = None, 
                cross_kws: Optional[Dict[str, Any]] = None, 
                conv_kws: Optional[Dict[str, Any]] = None):
        super().__init__()

        self.in_channels = in_channels
        self.cross_features = cross_features
        self.conv_features = conv_features
        self.cross_kws = cross_kws
        self.conv_kws = conv_kws

        assert 'use_sk' in self.conv_kws and 'use_sk' in self.cross_kws
        assert 'use_sc' in self.conv_kws and 'use_sc' in self.cross_kws

        conv_features = self.conv_features or self.cross_features
        cross_kws = self.cross_kws or {}
        conv_kws = self.conv_kws or {}

        self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)
        self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))

    def forward(self, target, support):
        target, support = self.cross(target, support)
        target = self.target(target)
        support = self.support(support)
        return target, support
    

class PPM(nn.Module):
    """
    this module involve the pyramid pooling convolution that encode contextual information
    and a bottlenect module that summarize the channel dimention.
    """
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.LeakyReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        return out

class CrossBlockSequential(nn.Sequential):
    def __init__(self, *args):
        super(CrossBlockSequential, self).__init__(*args)

    def forward(self, target, support):
        for module in self:
            target, support = module(target, support)
        return target, support

#@validate_arguments_init
# @dataclass(eq=False, repr=False)
class UniverSeg(iFSLModule):
    """
    main universeg model that inherit the pytorch lightning module.
    """
    
    def __init__(self, args):
        super(UniverSeg, self).__init__(args)
        self.encoder_blocks: List[size2t] = [64, 64, 64, 64]
        self.decoder_blocks: Optional[List[size2t]] = None

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        encoder_blocks = list(map(as_2tuple, self.encoder_blocks))
        decoder_blocks = self.decoder_blocks or encoder_blocks[-2::-1]
        decoder_blocks = list(map(as_2tuple, decoder_blocks))

        # toggle the selective kernel module in here
        conv_kws = dict(
                use_sk=True if args.use_sk else False,
                use_sc=True if args.use_sc else False,
                sk_split_input=True if args.sk_split_input else False)
        block_kws = dict(
            cross_kws=dict(
                nonlinearity=None, 
                use_sk=True if args.use_sk else False,
                use_sc=True if args.use_sc else False,
                sk_split_input=True if args.sk_split_input else False),
            conv_kws=conv_kws
        )

        # in_ch = (1, 2) #NOTE: this is for 1d image slice
        # out_channels = 1 #NOTE: for using sigmoid only

        in_ch = (3, 4) #NOTE: this is for rgb images (input dimension of the query images, input dimension of support images + input dimention of the support mask)
        out_channels = 2 #NOTE: for using logsoftmax + negative log likelihood

        # Encoder
        skip_outputs = []
        for j, (cross_ch, conv_ch) in enumerate(encoder_blocks):
            if j == 0 and args.use_sk:
                # a additional crossblock in the first layer to make sure
                # there are enough feature maps split the path evenly
                # otherwise there are 7 paths to split
                # TODO: see if necessary to use the approach used in scnet
                block = CrossBlockSequential(
                    CrossBlock(in_ch, cross_ch, conv_ch, **dict(
                        cross_kws=dict(
                            # nonlinearity=None, 
                            use_sk=False,
                            use_sc=False),
                        conv_kws=dict(
                            use_sk=False,
                            use_sc=False)
                    )),
                    CrossBlock(conv_ch, cross_ch, conv_ch, **block_kws)
                )
            else:
                block = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # a pyramid pooling module that only applied at the end of the encoder
        self.use_ppm = args.use_ppm
        self.bins = args.bins
        self.dropout = args.dropout
        if self.use_ppm:
            self.fea_dim = self.encoder_blocks[-1] # the same as the the last number in self.encoder_blocks
            self.ppm = PPM(self.fea_dim, int(self.fea_dim/len(self.bins)), self.bins)
            self.fea_dim *= 2
            self.bottleneck_dim = self.encoder_blocks[-1] # TODO: if this does not work, then try smaller than 64 but also need to modify the decoder dimension

            self.bottleneck = nn.Sequential(
                nn.Conv2d(self.fea_dim, self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout2d(p=self.dropout))

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(in_ch + skip_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.dec_blocks.append(block)

        out_activation = None
        self.out_conv = ConvOp(
            in_ch, out_channels, kernel_size=1, **conv_kws
        )

    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        batch['support_imgs'].shape : [bsz, way, shot, 3, H, W] [2 batch, 40 shots, channel, 128 h, 128 w]
        batch['support_masks'].shape : [bsz, way, shot, H, W]
        '''
        # NOTE: the following code is needed for the pascal_origin.py file
        # support_images = rearrange(batch['support_imgs'], 'b n s c h w -> (b n) s c h w') # resulting shape = [b, shot,c, h, w]
        # support_labels = None if self.weak else rearrange(batch['support_masks'], 'b n s h w -> (b n) s 1 h w') # resulting shape = [b, shot,c, h, w]

        # NOTE: the following is needed for the pascal.py file obtained from L_Seg paper
        support_images = batch['support_imgs']# resulting shape = [b, shot,c, h, w]
        support_labels = None if self.weak else rearrange(batch['support_masks'], 'b s h w -> b s 1 h w') # resulting shape = [b, shot,c, h, w]
        target_image = batch['query_img'] #[b, c, h, w]

        target = rearrange(target_image, "B C H W -> B 1 C H W") # #[b, 1, c, h, w]
        support = torch.cat([support_images, support_labels], dim=2)

        pass_through = []

        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.encoder_blocks) - 1  and not self.use_ppm:
                break

            # for the last encoder block, if ppm is used => encode the contexual information both the target the support features
            if i == len(self.encoder_blocks) - 1 and self.use_ppm:
                target = self.ppm(target.squeeze(1))
                target = self.bottleneck(target)
                target = target.unsqueeze(1)

                _, num_sprts, *_ = support.shape
                sprts = []
                for s in range(num_sprts):
                    sprt = support[:, s, :]
                    sprt_ppm = self.ppm(sprt)
                    sprt_ppm = self.bottleneck(sprt_ppm)
                    sprts.append(sprt_ppm)
                support = torch.stack(sprts, dim=1)
                break

            pass_through.append((target, support))

            # after each encoder block (EXCEPT THE LAST ENCODER BLOCK), downsample it
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        for decoder_block in self.dec_blocks:
            target_skip, support_skip = pass_through.pop()

            # skip connection with upsampled features before the decoder module
            target = torch.cat([vmap(self.upsample, target), target_skip], dim=2)
            support = torch.cat([vmap(self.upsample, support), support_skip], dim=2)

            # concatenate the skipped encoded features with the upsampled features
            target, support = decoder_block(target, support)

        target = rearrange(target, "B 1 C H W -> B C H W")
        logits = self.out_conv(target)
        
        # activation of the last layer and compute loss should be for dimension of foreground and background
        shared_masks = torch.log_softmax(logits, dim=1).unsqueeze(1) # B 1-way C H W

        # NOTE: for out_channels = 1: we have to use log sigmoid + bceCE and not log softmax + negative log likelihood 
        # log_sigmoid = nn.LogSigmoid()
        # shared_masks = log_sigmoid(logits)
        return shared_masks

    def train_mode(self):
        self.train()

    def configure_optimizers(self):
        return torch.optim.Adam([{"params": self.parameters(), "lr": self.args.lr}])

    def predict_mask_nshot(self, batch, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        # essentially batch prediction

        pass
