<h3 align = "right">学院：计算机学院&emsp;学号：3120191079&emsp;姓名：周泳宇</h3>
# 互评作业1: 数据探索性分析与数据预处理
**所选数据集**
- [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews "Wine Reviews")
- [Consumer & Visitor Insights For Neighborhoods](https://www.kaggle.com/safegraph/visit-patterns-by-census-block-group "Consumer & Visitor Insights For Neighborhoods")

------------

**目录文件介绍**
- **分析过程报告**：Wine Reviews.html，Consumer & Visitor Insights For Neighborhoods.html分别为两个数据集的分析过程报告
- **程序**：
	- Wine Reviews.ipynb，Wine Reviews.py为第一个数据集的程序；
	- Consumer & Visitor Insights For Neighborhoods.ipynb，Consumer & Visitor Insights For Neighborhoods.py为第二个数据集的程序；
	- .ipynb后缀的是使用 Jupyter Notebook 来编写Python程序时的文件；
	- .py后缀的是源代码文件。

------------

**使用说明**
1. 将数据集中的文件导入与程序同一级目录下
2. 安装以下python库
```python
import seaborn as sns
import numpy as np
from fancyimpute import KNN
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
```
