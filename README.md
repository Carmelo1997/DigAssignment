### 学院：计算机学院&emsp;学号：3120191079&emsp;姓名：周泳宇
# 互评作业4_离群点分析与异常检测
**所选数据集**
- abalone_benchmarks
- wine_benchmarks
以上两个数据集均来自于：[Anomaly Detection Meta-Analysis Benchmarks](https://ir.library.oregonstate.edu/concern/datasets/47429f155?locale=en "Anomaly Detection Meta-Analysis Benchmarks")
------------

**目录文件介绍：目录【互评作业4_离群点分析与异常检测】下为本次作业的相关文件**
- **挖掘过程报告**：
	- abalone.html，wine.html分别为两个数据集的分析过程报告；
	- abalone_result.csv，wine_result.csv分别为两个数据集的分析结果：包含了用六种离群点检测算法对数据进行离群点分析时，在roc和prn这两个指标上的结果。
- **程序**：
	- abalone.ipynb，abalone.py为第一个数据集的程序；
	- wine.ipynb，wine.py为第二个数据集的程序；
	- .ipynb后缀的是使用 Jupyter Notebook 来编写Python程序时的文件；
	- .py后缀的是源代码文件。

------------

**使用说明**
1. 将数据集中的文件导入与程序同一级目录下
2. 安装以下python库
```python
import numpy as np
import pandas as pd
import pyod
import sklearn
import tqdm
```
