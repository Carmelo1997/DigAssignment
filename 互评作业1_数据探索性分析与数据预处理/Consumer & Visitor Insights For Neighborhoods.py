#!/usr/bin/env python
# coding: utf-8

# <h2 align = "right">学院：计算机学院&emsp;学号：3120191079&emsp;姓名：周泳宇</h2>

# # Consumer & Visitor Insights For Neighborhoods数据集

# ## 一、数据可视化和摘要

# In[6]:


import seaborn as sns
import numpy as np
from fancyimpute import KNN
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', 10000)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',50)


# In[3]:


wine_df = pd.read_csv('cbg_patterns.csv')


# In[67]:


def fiveNumber(nums):
    # 五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum = min(nums)
    Maximum = max(nums)
    Q1 = np.percentile(nums, 25)
    Median = np.median(nums)
    Q3 = np.percentile(nums, 75)

    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR  # 下限值
    upper_limit = Q3 + 1.5 * IQR  # 上限值

    return Minimum, Q1, Median, Q3, Maximum, lower_limit, upper_limit


# ### 1.1 数据摘要

# #### 标称属性：以related_same_day_brand、top_brands为例（频数描述）

# In[8]:


province = wine_df['related_same_day_brand']
value_count = province.value_counts()
print(value_count)


# In[9]:


country = wine_df['top_brands']
value_count = country.value_counts()
print(value_count)


# #### 数值型属性：distance_from_home, raw_visit_count, raw_visitor_count（统计缺失值+五数概括）

# In[11]:


# 统计缺失值
price = wine_df['raw_visit_count']
na_count = price.shape[0] - price.count()
print(na_count)

# 统计缺失值
price = wine_df['raw_visitor_count']
na_count = price.shape[0] - price.count()
print(na_count)

# 统计缺失值
price = wine_df['distance_from_home']
na_count = price.shape[0] - price.count()
print(na_count)


wine_df.info()
# wine_df.head()


# #### 可以看到，数据总数为220735条，在数值型数据中，raw_visit_count缺失了106条数据，raw_visitor_count缺失了106条数据，distance_from_home缺失了217条数据，数据缺失原因：数据库进行数据写入时数据丢失，或者价格本身就是缺失数据，没有被统计到。

# In[12]:


# 五数概括
wine_df.describe()


# ### 1.2 数据可视化（以属性distance_from_home, raw_visit_count, raw_visitor_count为例）

# #### 直方图（数据分布）

# In[13]:


sns.distplot(wine_df['raw_visit_count'])
plt.show()
sns.distplot(wine_df['raw_visitor_count'])
plt.show()
sns.distplot(wine_df['distance_from_home'])
plt.show()


# #### 盒图（离群点）

# In[14]:


sns.boxplot(data=wine_df['raw_visit_count'])
plt.show()
sns.boxplot(data=wine_df['raw_visitor_count'])
plt.show()
sns.boxplot(data=wine_df['distance_from_home'])
plt.show()


# ## 二、数据缺失值的处理（因标称型属性无缺失，故这里不进行处理）

# In[15]:


#显示所有列
pd.set_option('display.max_columns', 100)
#显示所有行
pd.set_option('display.max_rows', 1000)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',1000)


# ### 2.1 将缺失部分剔除

# #### 处理之前的数据集

# In[20]:


sns.pairplot(wine_df, vars=["distance_from_home","raw_visit_count","raw_visitor_count"])
plt.show()
print(wine_df['distance_from_home'])


# #### 处理之后的数据集

# In[21]:


wine_df_after = wine_df.dropna()
sns.pairplot(wine_df_after, vars=["distance_from_home","raw_visit_count","raw_visitor_count"])
plt.show()
print(wine_df_after['distance_from_home'])


# ### 2.2 用最高频率值来填补缺失值

# #### 处理之前的数据集

# In[22]:


wine_df2 = wine_df.copy(deep=True)
sns.pairplot(wine_df2, vars=["distance_from_home","raw_visit_count","raw_visitor_count"])
plt.show()
print(wine_df2['distance_from_home'])


# #### 处理之后的数据集

# In[23]:


mode = wine_df2['distance_from_home'].mode().iloc[0]
wine_df2['distance_from_home'] = wine_df2['distance_from_home'].fillna(mode)

mode = wine_df2['raw_visit_count'].mode().iloc[0]
wine_df2['raw_visit_count'] = wine_df2['raw_visit_count'].fillna(mode)

mode = wine_df2['raw_visitor_count'].mode().iloc[0]
wine_df2['raw_visitor_count'] = wine_df2['raw_visitor_count'].fillna(mode)

sns.pairplot(wine_df2, vars=["distance_from_home","raw_visit_count","raw_visitor_count"])
plt.show()
print(wine_df2['distance_from_home'])


# ### 2.3 通过属性的相关关系来填补缺失值（对于属性raw_visit_count和raw_visitor_count，先采用取众数的方式进行填充，再通过这两个属性和属性distance_from_home的相关关系来填充distance_from_home）

# #### 处理之前的数据集

# In[24]:


wine_df3 = wine_df.copy(deep=True)
sns.pairplot(wine_df3, vars=["distance_from_home","raw_visit_count","raw_visitor_count"])
plt.show()
print(wine_df3['distance_from_home'])


# #### 处理之后的数据集

# In[26]:


def set_missing_ages(df):
    # 把数值型特征都放到随机森林里面去
    age_df = df[["distance_from_home","raw_visit_count","raw_visitor_count"]]
    known_age = age_df[age_df.distance_from_home.notnull()].iloc[:,:].values
    unknown_age = age_df[age_df.distance_from_home.isnull()].iloc[:,:].values
    y = known_age[:, 0]  # y是price，第一列数据
    x = known_age[:, 1:]  # x是特征属性值，后面几列
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # 根据已有数据去拟合随机森林模型
    rfr.fit(x, y)
    # 预测缺失值
    predictedAges = rfr.predict(unknown_age[:, 1:])
    # 填补缺失值
    df.loc[(df.distance_from_home.isnull()), 'distance_from_home'] = predictedAges

    return df

mode = wine_df3['raw_visit_count'].mode().iloc[0]
wine_df3['raw_visit_count'] = wine_df3['raw_visit_count'].fillna(mode)

mode = wine_df3['raw_visitor_count'].mode().iloc[0]
wine_df3['raw_visitor_count'] = wine_df3['raw_visitor_count'].fillna(mode)

wine_df3 = set_missing_ages(wine_df3)

sns.pairplot(wine_df3, vars=["distance_from_home","raw_visit_count","raw_visitor_count"])
plt.show()
print(wine_df3['distance_from_home'])


# ### 2.4 通过数据对象之间的相似性来填补缺失值（用KNN来衡量对象之间的相似性）

# #### 处理之前的数据集

# In[33]:


wine_df4 = wine_df.copy(deep=True)
sns.pairplot(wine_df4, vars=["distance_from_home","raw_visit_count","raw_visitor_count"])
plt.show()
print(wine_df4['distance_from_home'][-1000:])


# #### 处理之后的数据集（因计算资源原因，只处理局部数据）

# In[34]:


new_data = wine_df4[["distance_from_home","raw_visit_count","raw_visitor_count"]][-1000:]
fill_knn = KNN(k=3).fit_transform(new_data)
print(fill_knn[-1000:])


# In[ ]:




