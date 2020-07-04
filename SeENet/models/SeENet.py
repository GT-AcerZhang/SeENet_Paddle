import paddle.fluid as fluid
from .modeling.SeEM import SeEM
from .modeling.SeEM_FS import SeEM_FS
from .backbone.resnet import resnet50, resnet101, resnet152


class BaseNet(fluid.dygraph.Layer):
    def __init__(self, nclass=21, pretrained=False, backbone="resnet101"):
        super(BaseNet, self).__init__()
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained)
        elif backbone == "resnet101":
            self.backbone = resnet101(pretrained)
        elif backbone == "resnet152":
            self.backbone = resnet152(pretrained)

    def base_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)

        return c2, c3, c4, c5


class SeENet(BaseNet):
    def __init__(self, nclass=21, pretrained=False, backbone="resnet101"):
        super(SeENet, self).__init__(nclass, pretrained, backbone)
        self.head = SeENet_Head(out_channels=nclass, norm_layer=fluid.dygraph.BatchNorm)

    def forward(self, x):
        _, _, h, w = x.shape
        c2, c3, c4, c5 = self.base_forward(x)
        output = []
        feat = self.head(c3, c4, c5)
        output.append(fluid.layers.resize_bilinear(feat, out_shape=(h, w)))
        return tuple(output)


class SeENet_Head(fluid.dygraph.Layer):
    def __init__(self, in_channels=(512, 1024, 2048), out_channels=21, norm_layer=None):
        super(SeENet_Head, self).__init__()
        self.seem_1 = SeEM(in_channels=in_channels[0], atrous_rates=(1, 2, 4, 8),
                           norm_layer=norm_layer)
        self.seem_2 = SeEM(in_channels=in_channels[1], atrous_rates=(3, 6, 9, 12),
                           norm_layer=norm_layer)
        self.conv3 = fluid.dygraph.Conv2D(num_channels=in_channels[1]+in_channels[0], num_filters=in_channels[0],
                                          filter_size=1, stride=1, padding=0, bias_attr=False)

        self.seem_fs = SeEM_FS(in_channels=in_channels[2], atrous_rates=(7, 13, 19, 25), norm_layer=norm_layer)
        self.conv4 = fluid.dygraph.Conv2D(num_channels=in_channels[2]+in_channels[1], num_filters=in_channels[1],
                                          filter_size=1, stride=1, padding=0, bias_attr=False)

        self.conv = fluid.dygraph.Sequential(
            fluid.dygraph.Conv2D(num_channels=512,
                                 num_filters=256,
                                 filter_size=3, stride=1, padding=1, bias_attr=False),
            norm_layer(256),
            fluid.dygraph.Conv2D(num_channels=256,
                                 num_filters=out_channels, filter_size=1, stride=1, padding=0, bias_attr=False)
        )

    def forward(self, x3, x4, x5):
        _, _, h, w = x.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        seem_fs = fluid.layers.resize_bilinear(self.seem_fs(x5), out_shape=(h4, w4))
        conv4 = self.conv4(fluid.layers.concat([seem_fs, x4], axis=1))

        seem_2 = fluid.layers.resize_bilinear(self.seem_2(conv4), out_shape=(h3, w3))
        conv3 = self.conv3(fluid.layers.concat([seem_2, x3], axis=1))

        seem_1 = self.seem_1(conv3)

        return self.conv(seem_1)


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np
        x = fluid.dygraph.to_variable(np.ones(shape=(4, 3, 512, 512), dtype="float32"))
        # x2 = fluid.dygraph.to_variable(np.ones(shape=(4, 256, 128, 128), dtype="float32"))
        # x3 = fluid.dygraph.to_variable(np.ones(shape=(4, 512, 64, 64), dtype="float32"))
        # x4 = fluid.dygraph.to_variable(np.ones(shape=(4, 1024, 32, 32), dtype="float32"))
        # x5 = fluid.dygraph.to_variable(np.ones(shape=(4, 2048, 16, 16), dtype="float32"))
        # model = SeENet_Head(norm_layer=fluid.dygraph.BatchNorm)
        # out = model(x3, x4, x5)
        # print(out.shape)
        model = SeENet(pretrained=False, backbone="resnet50")
        out = model(x)
        print(out[-1].shape)