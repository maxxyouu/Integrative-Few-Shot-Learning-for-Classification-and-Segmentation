import torch
import torch.nn as nn
import torch.nn.functional as F

class SCNet(nn.Module):
    """this is the self calibration module for spatial neuron activation"""
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer, bias):
        """inplanes: number of input channels; planes: number of output channels"""
        super(SCNet, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=bias),
                    # norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=bias),
                    # norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=bias),
                    # norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                    cardinality=1, bottleneck_width=32,
                    avd=False, dilation=1, is_first=False,
                    norm_layer=None, act_layer=nn.LeakyReLU, bias=False, last_layer=False):
        """note for small model bias should be large but for large model bias does not matter
        
        k1 and scnet with default kernel size of 3 
        the intial conv1_a and conv1_b with kernel size of 1
        """
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=bias)
        self.bn1_a = norm_layer(group_width) if norm_layer else None
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=bias)
        self.bn1_b = norm_layer(group_width) if norm_layer else None

        self.avd = avd and (stride > 1 or is_first)
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=bias),
                    # norm_layer(group_width), # NOTE: check if this is exists if it is None
                    )

        self.scconv = SCNet(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer, bias=bias)

        # self.conv3 = nn.Conv2d(
        #     group_width * 2, planes * 4, kernel_size=1, bias=bias)
        # self.bn3 = norm_layer(planes*4)

        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.last_layer = last_layer

    def forward(self, x):
        # residual = x

        # perform convolution with equal splited input channels
        out_a= self.conv1_a(x)
        if self.bn1_a:
            out_a = self.bn1_a(out_a)
        
        out_b = self.conv1_b(x)
        if self.bn1_b:
            out_b = self.bn1_b(out_b)
        
        out_a = self.act(out_a)
        out_b = self.act(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)

        # NOTE: to make sure that the last layer does not have nonlinear activation
        if not self.last_layer:
            out_a = self.act(out_a)
            out_b = self.act(out_b)

        # if self.avd:
        #     out_a = self.avd_layer(out_a)
        #     out_b = self.avd_layer(out_b)
        out = torch.cat([out_a, out_b])

        # out = self.conv3(out, dim=1)
        # out = self.bn3(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        return out
