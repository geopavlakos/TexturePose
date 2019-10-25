import torch
import torch.nn as nn
import math
import torchvision.models.resnet as resnet

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SMPLPoseNet(nn.Module):

    def __init__(self, block, layers, npose):
        self.inplanes = 64
        super(SMPLPoseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, init_pose, init_shape, init_cam):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = []
        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        xf1 = torch.cat([xf,init_pose,init_shape,init_cam],1)
        xf1 = self.fc1(xf1)
        xf1 = self.drop1(xf1)
        xf1 = self.fc2(xf1)
        xf1 = self.drop2(xf1)
        xpose1 = self.decpose(xf1) + init_pose
        xshape1 = self.decshape(xf1) + init_shape
        xcam1 = self.deccam(xf1) + init_cam
        out.append(xpose1)
        out.append(xshape1)
        out.append(xcam1)

        xf2 = torch.cat([xf,xpose1,xshape1,xcam1],1)
        xf2 = self.fc1(xf2)
        xf2 = self.drop1(xf2)
        xf2 = self.fc2(xf2)
        xf2 = self.drop2(xf2)
        xpose2 = self.decpose(xf2) + xpose1
        xshape2 = self.decshape(xf2) + xshape1
        xcam2 = self.deccam(xf2) + xcam1
        out.append(xpose2)
        out.append(xshape2)
        out.append(xcam2)

        xf3 = torch.cat([xf,xpose2,xshape2,xcam2],1)
        xf3 = self.fc1(xf3)
        xf3 = self.drop1(xf3)
        xf3 = self.fc2(xf3)
        xf3 = self.drop2(xf3)
        xpose3 = self.decpose(xf3) + xpose2
        xshape3 = self.decshape(xf3) + xshape2
        xcam3 = self.deccam(xf3) + xcam2
        out.append(xpose3)
        out.append(xshape3)
        out.append(xcam3)

        return out

def smplresnet50(npose=144, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SMPLPoseNet(Bottleneck, [3, 4, 6, 3], npose, **kwargs)
    if pretrained:
        model_full = resnet.resnet50(pretrained=pretrained)
        model.load_state_dict(model_full.state_dict(),strict=False)
    return model
