# Towards Bridging Semantic Gap to Improve Semantic Segmentation 

说明：这个仓库是该论文的非官方复现

模型结构图：

<img src="C:\Users\wuyang\Desktop\Paddle\SeENet_Paddle\src\SeENet.png" style="zoom:75%;" />

子模块结构图：

<img src="C:\Users\wuyang\Desktop\Paddle\SeENet_Paddle\src\Sub-Modules.png" style="zoom:75%;" />

注意：原文使用带有dilated convolution的resnet做模型backbone，由于显卡算力有限，在这里我们只使用原始resnet作为backbone。

实验结果：

| Method | backbone  | mIoU % | pixAcc % |
| :----: | :-------: | :----: | :------: |
|  FCN   | resnet50  |        |          |
| SeENet | resnet50  |        |          |
| SeENet | resnet101 |        |          |

