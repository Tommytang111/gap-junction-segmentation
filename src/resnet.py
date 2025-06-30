import torch
import torch.nn as nn
def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1, three=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    ) if not three else nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride = 1, three=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False) if not three else nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample=None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer= None,
        three=False
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.three = three
        self.conv1 = conv3x3(inplanes, planes, stride, three=self.three)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, three=self.three)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample= None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
        three = False
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.three = three
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, three=self.three)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer= None,
        three=False,
        membrane=False
    ):
        super().__init__()
        print("---USING RESNET--")
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.three = three
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(layers[0], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) if not three else nn.Conv3d(layers[0], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if not three else nn.MaxPool3d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(block, 64, layers[0], three=three)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=2, dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.membrane = membrane
        if self.membrane:
            self.mem_conv1 = nn.Conv2d(layers[0], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) if not three else nn.Conv3d(layers[0], self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) 
            self.mem_bn1 = norm_layer(self.inplanes)
            self.mem_relu = nn.ReLU(inplace=True)
            self.mem_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if not three else nn.MaxPool3d(kernel_size=3, stride=2, padding=1) 

            self.mem_layer1 = self._make_layer(block, 64, layers[0], three=three)
            self.mem_layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate= False,
        three=False
    ):
        norm_layer = nn.BatchNorm2d if not three else nn.BatchNorm3d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, three=three),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, three=three
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if hasattr(self, "membrane") and self.membrane:
            x, x_mem = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        skip_out_1 = self.layer1(x)
        if hasattr(self, "three") and self.three: 
            skip_out_1 = skip_out_1.squeeze(2)
        skip_out_2_o = self.layer2(skip_out_1)
        
        #add_membrane:
        if hasattr(self, "membrane") and self.membrane:
            x_mem = self.mem_conv1(x_mem)
            x_mem = self.mem_bn1(x_mem)
            x_mem = self.mem_relu(x_mem)
            x_mem = self.mem_maxpool(x_mem)

            skip_mem1 = self.mem_layer1(x_mem)
            x_mem = self.mem_layer2(skip_mem1)

            skip_out_2 = x_mem + skip_out_2_o
            skip_out_1 = skip_out_1 + skip_mem1 # NEW ADDITION 
        else: skip_out_2 = skip_out_2_o

        skip_out_3 = self.layer3(skip_out_2)
        skip_out_4 = self.layer4(skip_out_3)
        x = self.layer5(skip_out_4)
        
        return x, skip_out_1, skip_out_2, skip_out_3, skip_out_4

    def forward(self, x):
        return self._forward_impl(x)

if __name__ == "__main__":
    img = torch.rand(1, 3, 512, 512)
    ResNet(BasicBlock, [1, 4, 6, 3, 3]).to("cuda")(img.to("cuda"))
