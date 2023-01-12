# NLP-Prompt
基于prompt learning的NLP算法，涉及文本分类、信息抽取等  

---

本项目集成了基于 prompt learning 相关的NLP任务，对 prompt learning 相关的理论知识可见[Prompt Learning|深入浅出提示学习要旨与常用方法](https://zhuanlan.zhihu.com/p/595178668) 。  

基于 pytorch + transformers 框架，其中[transformers](https://huggingface.co/docs/transformers/index) 是 huggingface 开源的近年非常火的开源框架，支持非常方便的加载/训练 transformer 模型，详见[这里](https://huggingface.co/docs/transformers/quicktour)  ；看到该库的安装方法和入门级调用，该库也能支持用户非常便捷的[微调一个属于自己的模型](https://huggingface.co/docs/transformers/training)  。    

本项目抽象于实际业务，并集成了一些主流的NLP任务，如有需要对应的任务，可将代码中的`训练数据集`更换成`你自己任务下的数据集`从而训练一个符合你自己任务下的模型。

<br>

目前已经实现的NLP任务如下（更新中）：

#### 1. 信息抽取（Information Extraction）

> 在给定的文本中抽取目标信息，多用于：`命名实体识别（NER）`，`实体关系抽取（RE）` 等任务。

| 模型  | 传送门  |
|---|---|
| 通用信息抽取（Universe Information Extraction, UIE）  | [[入口]](https://github.com/xuyingjie521/NLP-Prompt/tree/main/UIE_prompt) |

<br>

#### 2. 文本分类（Text Classification）

> 对给定文本进行分类，用于：`情感识别`，`文本质检识别` 等任务。

| 模型  | 传送门  |
|---|---|
| 基于 BERT 的分类器  | [[]]() |

<br>
