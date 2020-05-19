### 学院：计算机学院&emsp;学号：3120191079&emsp;姓名：周泳宇
# 互评作业2_频繁模式与关联规则挖掘
**所选数据集**
- [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews "Wine Reviews")

------------

**目录文件介绍：目录【互评作业2_频繁模式与关联规则挖掘】下为本次作业的相关文件**
- **分析过程报告**：Wine Reviews.html是挖掘过程的报告
- **程序**：
	- Wine Reviews.ipynb，Wine Reviews.py为整个挖掘过程的程序；
	- .ipynb后缀的是使用 Jupyter Notebook 来编写Python程序时的文件；
	- .py后缀的是源代码文件。

------------

**使用说明**
1. 将数据集中的文件导入与程序同一级目录下
2. 安装以下python库
```python
import numpy as np
import pandas as pd
from tqdm import tqdm
from efficient_apriori import apriori
from matplotlib import pyplot as plt
```
