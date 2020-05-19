#!/usr/bin/env python
# coding: utf-8

# <h2 align = "right">学院：计算机学院&emsp;学号：3120191079&emsp;姓名：周泳宇</h2>

# # 互评作业2: 频繁模式与关联规则挖掘

# ## 本数据集由许多关于葡萄酒的点评构成，这里将分析红酒产地country与品种variety的关系。

# ## 1.对数据集进行处理，转换成适合进行关联规则挖掘的形式

# ### 读取数据集

# In[1]:


import pandas as pd


# In[66]:


wine_df = pd.read_csv("winemag-data_first150k.csv")


# In[15]:


wine_df.head()


# ### 将数据转换成多条事务，适合于频繁项挖掘

# In[16]:


from tqdm import tqdm

transactions = []
for index, row in tqdm(wine_df.iterrows()):
    transactions += [(row['country'], row['variety'])]

transactions[:20]


# ## 2.找出频繁模式（采用Apriori算法，support>0.03，confidence>0.1）

# ```
# Aprior算法流程
#   输入：数据集合D，支持度阈值α
#   输出：最大的频繁k项集
#   1）扫描整个数据集，得到所有出现过的数据，作为候选频繁1项集。k=1，频繁0项集为空集。
# 　2）挖掘频繁k项集
# 　   a) 扫描数据计算候选频繁k项集的支持度
#      b) 去除候选频繁k项集中支持度低于阈值的数据集,得到频繁k项集。如果得到的频繁k项集为空，则直接返回频繁k-1项集的集合作为算法结果，算法结束。如果得到的频繁k项集只有一项，则直接返回频繁k项集的集合作为算法结果，算法结束。
#      c) 基于频繁k项集，连接生成候选频繁k+1项集。
#   3） 令k=k+1，转入步骤2。
# ```

# In[17]:


from efficient_apriori import apriori

itemsets, rules = apriori(transactions, min_support=0.03,  min_confidence=0.1)


# ### 频繁模式如下所示：所有的频繁k项集已在下面的字典中列出

# In[23]:


itemsets


# ## 3.导出关联规则，计算其支持度和置信度

# In[44]:


for rule in sorted(rules, key=lambda rule: rule.lift):
#     print(dir(rule))
#     print(rule)
    print("关联规则：" + "{" + rule.lhs[0] + "}" + " -> " + "{" + rule.rhs[0] + "}" + "\t" + 
         "支持度：" + str(rule.support) + "\t" + "置信度：" + str(rule.confidence))


# ## 4.使用Lift、Kulc系数对规则进行评价

# ### 4.1 相关性系数Lift
# #### 对于规则A—>B或者B—>A，lift(A,B)=P(A交B)/(P(A)*P(B))，如果lift(A,B)>1表示A、B呈正相关，lift(A,B)<1表示A、B呈负相关，lift(A,B)=1表示A、B不相关（独立）。

# In[45]:


for rule in sorted(rules, key=lambda rule: rule.lift):
#     print(dir(rule))
#     print(rule)
    print("关联规则：" + "{" + rule.lhs[0] + "}" + " -> " + "{" + rule.rhs[0] + "}" + "\t" + 
         "相关性系数Lift：" + str(rule.lift))


# #### 可以看到，关联规则{Bordeaux-style Red Blend} -> {France}的相关性系数较高：说明品种Bordeaux-style Red Blend与产地France存在较大的关联关系，可以通过原始数据集来验证一下，如下所示（可以看出，法国的确比较多）：

# In[46]:


wine_df[wine_df['variety'] == 'Bordeaux-style Red Blend'].sample(20)


# #### 绘制直方图进一步验证：

# In[48]:


wine_df[wine_df['variety'] == 'Bordeaux-style Red Blend']['country'].value_counts().plot(kind='bar')


# ### 4.2 Kulc系数
# #### Kulc系数就是对两个置信度做一个平均处理：kulc(A,B)=(confidence(A—>B)+confidence(B—>A))/2 

# In[54]:


res = []
for rule1 in sorted(rules, key=lambda rule: rule.lift):
    conf1 = rule1.confidence
    for rule2 in sorted(rules, key=lambda rule: rule.lift):
        if rule2.lhs[0] == rule1.rhs[0] and rule2.rhs[0] == rule1.lhs[0]:
            conf2 = rule2.confidence
    kulc = (conf1 + conf2) / 2
    res.append("关联规则：" + "{" + rule1.lhs[0] + "}" + " -> " + "{" + rule1.rhs[0] + "}    " + 
         "Kulc系数：" + str(kulc))

res


# ## 5.对挖掘结果进行分析

# ### 如在上一部分所示，通过Lift系数，已经对规则{Bordeaux-style Red Blend} -> {France}进行了分析和验证，类似的，规则{Pinot Noir} -> {US}和{Cabernet Sauvignon} -> {US}的Lift系数也较大，说明这两个葡萄酒品种极有可能来自于US，最后，规则{Chardonnay} -> {US}的Lift系数为1.357417922402187，比较接近1，说明品种Chardonnay与US成正相关关系，但可能关联性并没有那么强。

# ## 6.可视化展示

# In[63]:


import matplotlib.pyplot as plt

def plot_bar(rules, data, title):
    plt.title(title)
    plt.xticks(range(len(data)),rules,rotation=90)
    plt.bar(range(len(data)), data, color = 'B')
    plt.show()

def visualization(big_rule_list):
    rules = []
    conf = []
    support = []
    lift = []
    kulc = []
    for i in range(len(big_rule_list)):
        rule = big_rule_list[i][0]
        rules.append(rule)
        conf.append(big_rule_list[i][1])
        support.append(big_rule_list[i][2])
        lift.append(big_rule_list[i][3])
        kulc.append(big_rule_list[i][4])
    plot_bar(rules, support, 'rule-support figure')
    plot_bar(rules, conf, 'rule-confidence figure')
    plot_bar(rules, lift, 'rule-lift figure')
    plot_bar(rules, kulc, 'rule-kulc figure')


# In[64]:


big_rule_list = []

for rule1 in sorted(rules, key=lambda rule: rule.lift):
    conf1 = rule1.confidence
    for rule2 in sorted(rules, key=lambda rule: rule.lift):
        if rule2.lhs[0] == rule1.rhs[0] and rule2.rhs[0] == rule1.lhs[0]:
            conf2 = rule2.confidence
    kulc = (conf1 + conf2) / 2
    big_rule_list.append(["{" + rule1.lhs[0] + "}" + "=>" + "{" + rule1.rhs[0] + "}", rule1.confidence, 
                          rule1.support, rule1.lift, kulc])


# In[65]:


visualization(big_rule_list)


# In[ ]:




