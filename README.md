# Introduction

将百度DeepSpeech的keras后端由theano改为tensorflow，整合mozilla解码模块进行中文语音识别模型部署，以下称deepspeech-enhance模型。


@[toc]

----

## 项目背景
这是我调整了整整一年后落地的项目，效果能和科大讯飞媲美，不如讯飞的点是识别结果中没有标点符号，在特定领域准确率极高。

国内中文语音识别相关的有用资料很少，技术相对封闭，搜索引擎中能找到的方法基本上都是10年前的传统方法，早已过时。

在这个过程中踩了很多坑（开始两个月尝试先转拼音再转文字，fail），请教了很多人，很多个夜晚睡不着觉，想尽一切办法积累数据，有很多次会想放弃，持续专注的做了一年，最终呈现一个效果还不错的结果。

很赞同季逸超的观点，互联网领域的idea不值钱，实现也不值钱，值钱的是“**经过沉淀的idea + 反复推敲地执行**”

[相关博客，点我](https://blog.csdn.net/qq_30262201/category_9398117.html)

----

# 一、和百度deepspeech 2 的不同点
## 1.框架选择

背景：2019年3月12号接受了新采购的GPU机器一台，由于新机器适配的驱动版本太高（2019年2月发布），deepspeech 2转写模型使用的深度学习框架theano偏学术研究，theano的开发团队在17年就加入了google，已经停止维护，theano不支持分布式，相比之下tensorflow框架更偏工程，已经是主流框架，支持分布式，支持新硬件，我们有必要对转写工程做框架调整。

deepspeech 2模型框架：theano_0.8.2、keras_1.1.0

deepspeech-enhance模型框架：tensorflow_1.13.1、keras_2.2.4

分析：根据调研资料显示，tensorflow新版本相比theano可以带来性能上一倍的提升，同时需要更大的内存。

## 2.声学模型结构
在模型结构上主要做了6项调整，分析了每个调整项带来的影响：

|调整项	| deepspeech 2模型	| deepspeech-enhance模型	| 准确率 | 	性能 | 	资源占用|说明|
|----|----|----|----|----|-----|---|
|网络结构|	1D_CNN+3*GRU|	1_DCNN+3*BiGRU	|有提升|	降低近一倍|	更大的内存| 现在双向网络已是主流，transformer、bert等都是双向网络
|损失函数	|warp-ctc（baidu出品）	|tensorflow-ctc（google出品）	|不确定|	降低一点	|不确定| 前者是batch纬度计算损失，并行度高，但是训练阶段容易出现长尾问题；后者是样本纬度计算损失，训练过程不会出现长尾问题
|输出节点数|	26个英文字母+2|	4563个常用汉字|	降低	|降低一点|	增加| 汉字共有6000左右，统计发现有一千五百多个生僻字出现在日常对话中的概率极低
|语音帧长|	20ms	|25ms	|有一点提升|	提升一点|	更小的内存|-
|采样率|	16k	| 8k	|有一点降低|	提升 |	更小的内存和磁盘|工业使用中在保证效果的前提下节省一半空间


> 参考论文：http://proceedings.mlr.press/v48/amodei16.pdf

> [论文博客,点我](https://blog.csdn.net/qq_30262201/article/details/102654708)

## 3.其他调整项

（1）卷积层输出处理：忽略卷积层的前两位输出，因为它们通常无意义，且会影响模型最后的输出；

（2）BN层处理：最后一次训练冻结BN层，传入加载模型（纯开源数据训练的）的移动均值和方差。
调整后准确率平均提升2个百分点

## 4.增加beam search和n-gram组合解码模块（这里是重点）

- deepspeech 2 模型是贪婪搜索解码
- deepspeech-enhance的解码模块使用现在GitHub 上比较热门的mozilla基金会实现的beam search解码模型，n-gram的作用就是进一步纠错。

**关于解码**
> 为了在解码过程中整合语言模型信息，Graves＆Jaitly（2014）使用其经过CTC训练的神经网络对由基于HMM的最新系统生成的晶格或n最佳假设列表进行评分。 这引入了潜在的混淆因素，因为n最佳列表会很大程度上限制可能的转录集。 另外，它导致整个系统仍然依靠HMM语音识别基础结构来获得最终结果。 相反，<font face="微软雅黑" color=blue>我们提出的首遍解码结果使用神经网络和语言模型从头开始解码，而不是对现有假设进行重新排序</font>。

> 以上来自论文：https://arxiv.org/pdf/1408.2873.pdf

> [相关博客,点我](https://blog.csdn.net/qq_30262201/article/details/102653937)


# 二、测试结果
## 1.开源数据
100条数据堂电话语音数据上平均字错率0.02，句错率0.06
详细见./test_result/recongnnize_result.txt

列举几个准确率（1-编辑距离）不等于1的case，可以看到语言模型真正起到了纠错的作用。

语音|label   | predict  |  acc|
---|---|---|---|
G2425/session01/T0055G2425S0213.wav |我不是发给你了==么==  |   我不是发给你了==吗==  | 0.875
 G1978/session01/T0055G1978S0487.wav|你==那==有什么服务价格多少    | 你==那里==有什么服务价格多少 |   0.90909091
G2863/session01/T0055G2863S0065.wav|周末大扫除没空==啊==    | 周末大扫除没空  |  0.875
G1940/session01/T0055G1940S0496.wav|那你也不==给==我说一声    |  那你也不==跟==我说一声  |  0.88888889

## 2.垂直领域效果
贴上自己录音的识别效果：

<img src="https://img-blog.csdnimg.cn/20200108164816392.jpeg" width="100%" >
<img src="https://img-blog.csdnimg.cn/20200108164830722.jpeg" width="60%" >

# 三、deepspeech 环境搭建

新建虚拟环境：`conda create -n tensorflow python=3.6`

激活虚拟环境：`source activate tensorflow`

1.安装tensorflow：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.13.1`

2.安装keras：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.2.4`

3.安装语音流处理模块：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple soundfile==0.10.2`

训练环境安装前三个就可以，测试环境需要后面两个

4.安装beam search解码模块（解码模块使用mozilla项目里面的）：`pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp36-cp36m-manylinux1_x86_64.whl`

报错platform不支持的话在mozilla的DeepSpeech里面执行进行安装：`pip install $(python util/taskcluster.py --decoder)`

gpu版：`pip install https://index.taskcluster.net/v1/task/project.deepspeech.deepspeech.native_client.v0.5.0-alpha.11.cpu-ctc/artifacts/public/ds_ctcdecoder-0.5.0a11-cp35-cp35m-manylinux1_x86_64.whl`

5.读字节流：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pydub`

# 四、训练调参经验
参数|调整|分析
---|---|---|
|数据过度|先开源数据，后业务数据|先用开源数据做预训练使模型收敛，再用垂直领域的业务数据做fine-tune
|batch size| 16|尝试过16、32、64，设置16的情况下训练效果最好。batch size变小，数据拟合能力更好，训练时长会更长
|验证集大小| 3000|放大验证集，结果更有说服力，训练时长会更长
|优化器|adam|尝试过adam、SGD、adam+SGD、NAG。adam学习率自适应，比较智能
|学习率| 2e-4 - 4e-6|前期学习率大一点，后期小一点。loss出现nan，一般来说是学习率太大，应该减小学习率

# 五、失败的尝试
调整|失败原因
---|---|
声学模型映射1400个带声调的拼音| 声调的特征不是独立占位的，其特征包含在拼音的位置里面，而ctc损失计算的本质是给每一帧做分类，所以这样训练的效果并不好。
声学模型映射410个不带声调的拼音| 映射单元减少了准确率提升特别大，单纯音频转拼音这个环节的准确率提升很大，准确率很高，beam search解top-n能保证极高的召回率，但是拼音进一步转汉字的阶段效果很差，主要是在短句上的效果不好，长句上的效果还不错，这个阶段和拼音输入法的原理一样。



----

# 贡献者名单
姓名|属性|主要贡献|
|----|----|----|
陶瑞 | 项目负责人| 声学模型调整、声学模型数据收集和调整，模型训练及整体技术选型|
盛长霞  |团队成员| 语言模型调整、语言模型数据收集和调整、语音端点检测算法实现、mozilla工程梳理|
刘尧  |  团队成员| 工程运维、数据收集|
蒋志宇 | 团队成员| 服务部署|
袁文杰 | 外援| 协助|
赵若琪 | 团队成员| 参与deepspeech2梳理|
张瑞雄 | 外援|答疑解惑|
柠檬博主 | 外援|答疑解惑|

其他帮我答疑的同学暂时想不起来，想起来再补充，这里再次感谢以上同学对该项目的贡献。


