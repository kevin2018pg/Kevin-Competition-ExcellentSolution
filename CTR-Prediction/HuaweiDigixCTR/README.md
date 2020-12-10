# digix2020_ctr
华为digix算法大赛2020机器学习赛道-ctr预估

## 项目环境

Python 3.7

lightgbm

gensim

sklearn

pandas

numpy

tqdm

networkx

## 处理流程

在ctr下创建data文件夹，并将训练集、测试集A、测试集B的csv文件放在ctr/data/

运行reduce/reduce.py进行数据压缩

运行full.py进行全特征模型的训练和推理

运行win.py进行滑窗模型的训练和推理

运行nounique.py进行部分特征模型的训练和推理

运行result/fusion.py得到三个模型结果的融合

result文件夹中可得到最终结果文件submission_f.csv
