# 机器学习课程第六次作业
本作业是针对MNIST数据集进行的数字图片分类任务，训练出一个卷积网络对此数据集进行分类。

### 作业步骤

1.运行main.py文件并理解代码含义，并截图保存结果。

2.了解和熟悉读取数据、建立模型、训练模型、测试模型的过程。

3.理解模型ConvNet，了解其中卷积层和池化层的作用和处理数据的方法，可尝试与全连接层进行比较。

4.理解dropout的作用

注意：如果出现类似如下的错误
anaconda3/lib/python3.7/site-packages/torchvision/transforms/functional.py", line 5, in module 

from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION

ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'

问题就是PIL版本太新。解决方式为可参考[此链接](https://blog.csdn.net/weixin_45021364/article/details/104600802)