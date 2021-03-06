#### 本模型是一个单隐层神经网络，包含2个输入单元，4个隐藏单元和1个输出单元

![models](https://github.com/weekdawn/DL/blob/master/twoLayerNN/img/models.png)

#### 第一层采用的激活函数为tanh，第二层采用的激活函数为sigmoid

![formulas](https://github.com/weekdawn/DL/blob/master/twoLayerNN/img/formulas.jpg)

#### 首先介绍关于建立一个神经网络通用过程

Step1：设计网络结构，例如多少层，每层有多少神经元等。

Step2：初始化模型的参数

Step3：循环

    Step3.1：前向传播计算
    
    Step3.2：计算代价函数
    
    Step3.3：反向传播计算
    
    Step3.4：更新参数