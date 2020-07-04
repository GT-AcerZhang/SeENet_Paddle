import paddle.fluid as fluid


class SeEM(fluid.dygraph.Layer):
    """Semantic Enhancement Module."""
    def __init__(self, in_channels, atrous_rates=(1, 2, 4, 8), norm_layer=None):
        super(SeEM, self).__init__()
        self.branches = fluid.dygraph.LayerList()
        for atrous_rate in atrous_rates:
            self.branches.append(
                fluid.dygraph.Sequential(
                    fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=128, filter_size=1, padding=0,
                                         bias_attr=False),
                    norm_layer(128),
                    DwAtrousConv(in_channels=128, out_channels=128, atrous_rate=atrous_rate, padding=atrous_rate, norm_layer=norm_layer),

                )
            )
        self.branches_conv1x1 = fluid.dygraph.LayerList()
        for l in range(4):
            self.branches_conv1x1.append(
                fluid.dygraph.Conv2D(num_channels=128, num_filters=128, filter_size=1, padding=0,
                                     bias_attr=False))
        self.conv = fluid.dygraph.Conv2D(num_channels=in_channels+512, num_filters=in_channels, filter_size=1, padding=0, stride=1, bias_attr=False)
        
    def forward(self, x):
        _, _, h, w = x.shape
        feats = []
        for branch, branch_conv1x1 in zip(self.branches, self.branches_conv1x1):
            feats.append(branch_conv1x1(fluid.layers.resize_bilinear(branch(x), out_shape=(h, w))))
        feats.append(x)
        return self.conv(fluid.layers.concat(feats, axis=1))
        

class DwAtrousConv(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, filter_size=3, padding=1, stride=1, bias=False, atrous_rate=1, norm_layer=fluid.dygraph.BatchNorm):
        super(DwAtrousConv, self).__init__()
        self.conv1 = fluid.dygraph.Conv2D(
            num_channels=in_channels,
            num_filters=in_channels,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=atrous_rate,
            groups=in_channels,
            bias_attr=bias)
        self.bn = norm_layer(in_channels)
        self.point_wise = fluid.dygraph.Conv2D(
            num_channels=in_channels,
            num_filters=out_channels,
            filter_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias_attr=bias)

    def forward(self, x):
        return self.point_wise(self.bn(self.conv1(x)))


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np
        x = fluid.dygraph.to_variable(np.ones(shape=(4, 1024, 32, 32), dtype="float32"))
        model = SeEM(in_channels=1024, norm_layer=fluid.dygraph.BatchNorm)
        out = model(x)
        print(out.shape)
