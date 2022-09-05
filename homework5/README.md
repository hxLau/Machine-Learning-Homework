# 机器学习课程第五次作业
本作业是针对FashionMNIST的一个图片数据集进行图片分类任务，训练出一个利用全连接层构建的模型对此数据集进行分类，此数据集包含了10个类别的图像，分别是：t-shirt（T恤），trouser（牛仔裤），pullover（套衫），dress（裙子），coat（外套），sandal（凉鞋），shirt（衬衫），sneaker（运动鞋），bag（包），ankle boot（短靴）。

### 作业步骤

1.运行main.py文件并理解代码含义。

2.了解和熟悉读取数据、建立模型、训练模型、测试模型的过程。

3.学会保存和读取训练好的模型，并利用模型进行图片分类的预测。

4.理解模型NeuralNetwork处理数据的过程

注意：如果出现类似如下的错误
anaconda3/lib/python3.7/site-packages/torchvision/transforms/functional.py", line 5, in module 

from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION

ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'

问题就是PIL版本太新。解决方式为可参考[此链接](https://blog.csdn.net/weixin_45021364/article/details/104600802)