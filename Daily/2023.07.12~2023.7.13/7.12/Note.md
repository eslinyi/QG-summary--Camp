## 7.12学习记录

#### **GNN**

整体的学习分为3个环节

+ 第一个是初始了解，通过视频[零基础多图详解图神经网络（GNN/GCN）【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1iT4y1d7zP/?spm_id_from=333.337.search-card.all.click&vd_source=19cf4c72428c6a8d5f94d949e36643c7)
+ 第二个是自行探索-通过https://distill.pub/2021/gnn-intro/理解原理
+ 第三个是扩展和代码框架，通过视频[20. 6-网络结构定义模块_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1184y1x71H/?p=20&spm_id_from=pageDriver&vd_source=19cf4c72428c6a8d5f94d949e36643c7)

通过图的特征来进行神经网络的学习

+ 不同点
+ 不同边
+ 整个图

定义1：邻接矩阵：表示不同点之间的关系

知识点1：消息传递的方法：具体方法和神经网络的方法一样，类似加权的这种形式，但是有一点特殊的是考虑了图之间邻居的关系，比如说最大化，sum，mean等方法

知识点2：GNN的本质是更新部分特征，输入是特征，输出也是特征，也就是关系的结构不发生改变，但是其中多层网络中，经过感受野的方法来表示其中的全值关系

知识点3：GCN--图卷积：无核函数，是靠邻居来进行卷积操作 ---ps：传统的机器学习只能是固定结构，但是GCN不需要固定结构

+ GCN需要各节点的输入特征和网络结构图
+ 不需要全部标签，用少量便签也行，计算损失只用有标签的---一个人做好，需要全部做好 --半监督任务
+ GCN的基本思想 --针对某个特征来平均其邻居和自身特征来传入神经网络，和CNN类似，不同传入，用ReLu或者Sofmax等不断传下去
+ GCN的基本计算 --图的构成 
  + G图
  + A邻接矩阵
  + D各节点的度
  + F每个节点的特征
  + 计算可以是邻接矩阵与特征矩阵乘法操作，表示聚合邻居信息
  + 但是上述情况没考虑自己，故可在邻接矩阵加上自身后，再乘度函数，度函数要平均化处理再开根号 --防治产生多余操作

知识点4：GCN变换原理

+ 归一化原理是防止出现邻居之间影响太大(单独个体)
+ 基本公式:![](https://gitee.com/eslinyi/picture/raw/master/img/20230712102756.png)
+ 一般GCN的层数不会太多

操作1：安装PYG

操作2：基础框架搭建

+ GCN原理：![](https://gitee.com/eslinyi/picture/raw/master/img/20230712112712.png)
+ 实际使用---代码成立 --project1
+ 对该函数源代码理解 ----[torch_geometric.nn.conv.gcn_conv — pytorch_geometric documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv)

操作3：实现分类任务

代码 --project2

效果：![](https://gitee.com/eslinyi/picture/raw/master/img/20230712144454.png)

![](https://gitee.com/eslinyi/picture/raw/master/img/20230712144524.png)

操作4 ：构建自己的图数据集

代码 -- project3

InMemoryDataset函数理解

知识点5：网络架构

+ topkpooling剪枝操作
+  embedding操作 -- 特征编码 扩张维度
+ gap 平均--对向量平均，或者可以max

#### **Attention 图的注意力机制**

原理 --权重的设置

对于1，2节点进行映射后，再拼接后再乘可训练的权重最后得到一个新的权重

对邻接矩阵进行加权

补充：![](https://gitee.com/eslinyi/picture/raw/master/img/20230712160449.png)

上述来自[(15条消息) 深度学习算法--Attention（注意力机制）_attention算法_西欧阿哥的博客-CSDN博客](https://blog.csdn.net/Western_europe/article/details/109611695)

涉及三个机制

+ Query  --目标字
+ Key      --上下文的字，也就是序列值   --Query和Key的相似性看成权重
+ Value    --最后将权重和原始Value融合

还有Self-Attention   Multi-head Self-Attention机制等机制

#### **T-GNN --序列图神经网络**

图跟时间有关系

往RNN进行迁移 --时间序列(GRU)

LSTM可以把序列网络套用进来

#### **RNN**

深度学习这本书有点难以理解，明天再整理笔记



