import paddle.fluid as fluid

from SeENet_Paddle.models.modeling.SeEM import DwAtrousConv


class SeEM_FS(fluid.dygraph.Layer):
    """Semantic Enhancement Module."""

    def __init__(self, in_channels, atrous_rates=(1, 2, 4, 8), norm_layer=None):
        super(SeEM_FS, self).__init__()
        self.branches = fluid.dygraph.LayerList()
        for atrous_rate in atrous_rates:
            self.branches.append(
                fluid.dygraph.Sequential(
                    fluid.dygraph.Conv2D(num_channels=in_channels, num_filters=128, filter_size=1, padding=0,
                                         bias_attr=False),
                    norm_layer(128),
                    fluid.dygraph.Conv2D(num_channels=128, num_filters=128, filter_size=3, padding=1,
                                         bias_attr=False),
                    DwAtrousConv(in_channels=128, out_channels=128, atrous_rate=atrous_rate, padding=atrous_rate,
                                 norm_layer=norm_layer),

                )
            )
        self.branches_conv1x1 = fluid.dygraph.LayerList()
        for l in range(4):
            self.branches_conv1x1.append(
                fluid.dygraph.Conv2D(num_channels=128, num_filters=128, filter_size=1, padding=0,
                                     bias_attr=False))
        self.conv = fluid.dygraph.Conv2D(num_channels=in_channels + 512, num_filters=in_channels, filter_size=1,
                                         padding=0, stride=1, bias_attr=False)

    def forward(self, x):
        _, _, h, w = x.shape
        feats = []
        for branch, branch_conv1x1 in zip(self.branches, self.branches_conv1x1):
            # print(branch(x).shape)
            feats.append(branch_conv1x1(fluid.layers.resize_bilinear(branch(x), out_shape=(h, w))))
        feats.append(x)
        return self.conv(fluid.layers.concat(feats, axis=1))


if __name__ == '__main__':
    with fluid.dygraph.guard():
        import numpy as np
        x = fluid.dygraph.to_variable(np.ones(shape=(4, 2048, 16, 16), dtype="float32"))
        model = SeEM_FS(in_channels=2048, norm_layer=fluid.dygraph.BatchNorm)
        out = model(x)
        print(out.shape)
