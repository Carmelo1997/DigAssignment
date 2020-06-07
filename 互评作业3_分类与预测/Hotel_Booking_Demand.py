
# coding: utf-8

# ## 学院：计算机学院&emsp;学号：3120191079&emsp;姓名：周泳宇

# # Hotel booking demand, 酒店预订需求

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('hotel_bookings.csv')


# ## 查看数据

# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# ## 查看缺失值

# In[6]:


df.isnull().sum()[df.isnull().sum()!=0]


# ### 有四项信息存在缺失值，company缺失较多，可以考虑删除，children、country和agent较少，可以考虑填充。
# 
# ### 处理方法：假设agent中缺失值代表未指定任何机构，即nan=0，country使用其字段内众数填充，childred使用其字段内众数填充，company因缺失数值过大，所以直接删除

# In[7]:


df_new = df.copy(deep = True)
df_new.drop("company", axis=1, inplace=True)


# In[8]:


df_new["agent"].fillna(0, inplace=True)
df_new["children"].fillna(df_new["children"].mode()[0], inplace=True)
df_new["country"].fillna(df_new["country"].mode()[0], inplace=True)


# ## 再次查看数据，可以看到已经无缺失数据

# In[10]:


df_new.info()


# ## 处理异常值

# In[11]:


df_new["children"] = df_new["children"].astype(int)
df_new["agent"] = df_new["agent"].astype(int)

# 将 变量 adults + children + babies == 0 的数据删除
zero_guests = list(df_new["adults"] +
                  df_new["children"] +
                  df_new["babies"] == 0)
# hb_new.info()
df_new.drop(df_new.index[zero_guests], inplace=True)


# ## 1.基本情况：城市酒店和假日酒店预订需求和入住率比较；

# In[28]:


ch_count = len(df_new[df_new["hotel"]=="City Hotel"])
rh_count = len(df_new[df_new["hotel"]=="Resort Hotel"])

print("城市酒店预定需求：" + str(ch_count))
print("假日酒店预定需求：" + str(rh_count))

ch_live_count = len(df_new[(df_new["hotel"]=="City Hotel") & (df_new["is_canceled"]==0)])
rh_live_count = len(df_new[(df_new["hotel"]=="Resort Hotel") & (df_new["is_canceled"]==0)])

print("==============")

print("城市酒店入住率：%.4f" % (float(ch_live_count)/ch_count))
print("假日酒店入住率：%.4f" % (float(rh_live_count)/rh_count))


# ## 2.用户行为：提前预订时间、入住时长、预订间隔、餐食预订情况；

# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### 提前预定时长

# In[39]:


lead_cancel_data = pd.DataFrame(df_new.groupby("lead_time")["is_canceled"].describe())
# lead_cancel_data
# 因为lead_time中值范围大且数量分布不匀，所以选取lead_time>10次的数据（<10的数据不具代表性）
lead_cancel_data_10 = lead_cancel_data[lead_cancel_data["count"]>10]

y = list(round(lead_cancel_data_10["mean"], 4) * 100)

plt.figure(figsize=(12, 8))
sns.regplot(x=list(lead_cancel_data_10.index),
           y=y)
plt.title("提前预定时长对取消的影响", fontsize=16)
plt.xlabel("提前预定时长", fontsize=16)
plt.ylabel("取消数 [%]", fontsize=16)
plt.show()


# ### 可以明显看到：不同的提前预定时长确定对旅客是否取消预定有一定影响；通常，越早预订，越容易取消酒店房间预定。

# ### 入住时长

# In[45]:


full_data_guests = df_new.loc[df_new["is_canceled"] == 0] # only actual gusts
full_data_guests["total_nights"] = full_data_guests["stays_in_weekend_nights"] + full_data_guests["stays_in_week_nights"]

# 新建字段：total_nights_bin——居住时长区间
full_data_guests["total_nights_bin"] = "住1晚"
full_data_guests.loc[(full_data_guests["total_nights"]>1)&(full_data_guests["total_nights"]<=5), "total_nights_bin"] = "2-5晚"
full_data_guests.loc[(full_data_guests["total_nights"]>5)&(full_data_guests["total_nights"]<=10), "total_nights_bin"] = "6-10晚"
full_data_guests.loc[(full_data_guests["total_nights"]>10), "total_nights_bin"] = "11晚以上"

ch_nights_count = full_data_guests["total_nights_bin"][full_data_guests.hotel=="City Hotel"].value_counts()
rh_nights_count = full_data_guests["total_nights_bin"][full_data_guests.hotel=="Resort Hotel"].value_counts()

ch_nights_index = full_data_guests["total_nights_bin"][full_data_guests.hotel=="City Hotel"].value_counts().index
rh_nights_index = full_data_guests["total_nights_bin"][full_data_guests.hotel=="Resort Hotel"].value_counts().index

ch_nights_data = pd.DataFrame({"hotel": "城市酒店",
                               "nights": ch_nights_index,
                              "guests": ch_nights_count})
rh_nights_data = pd.DataFrame({"hotel": "假日酒店",
                               "nights": rh_nights_index,
                              "guests": rh_nights_count})
# 绘图数据
nights_data = pd.concat([ch_nights_data, rh_nights_data], ignore_index=True)
order = ["住1晚", "2-5晚", "6-10晚", "11晚以上"]
nights_data["nights"] = pd.Categorical(nights_data["nights"], categories=order, ordered=True)

plt.figure(figsize=(12, 8))
sns.barplot(x="nights", y="guests", hue="hotel", data=nights_data)
plt.title("旅客居住时长分布", fontsize=16)
plt.xlabel("居住时长", fontsize=16)
plt.ylabel("旅客数", fontsize=16)

plt.legend()
plt.show()


# ### 预定间隔

# ### days_in_waiting_list字段：Number of days the booking was in the waiting list before it was confirmed to the customer，该字段即可表示预定间隔

# In[60]:


df_new.hist('days_in_waiting_list')
plt.show()


# ### 餐食预定

# In[46]:


meal_data = df_new[["hotel", "is_canceled", "meal"]]
# meal_data

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.pie(meal_data.loc[meal_data["is_canceled"]==0, "meal"].value_counts(), 
        labels=meal_data.loc[meal_data["is_canceled"]==0, "meal"].value_counts().index, 
       autopct="%.2f%%")
plt.title("未取消预订旅客餐食选择", fontsize=16)
plt.legend(loc="upper right")

plt.subplot(122)
plt.pie(meal_data.loc[meal_data["is_canceled"]==1, "meal"].value_counts(), 
        labels=meal_data.loc[meal_data["is_canceled"]==1, "meal"].value_counts().index, 
       autopct="%.2f%%")
plt.title("取消预订旅客餐食选择", fontsize=16)
plt.legend(loc="upper right")
plt.show()


# ### 很明显，取消预订旅客和未取消预订旅客有基本相同的餐食选择

# ## 3.一年中最佳预订酒店时间；

# In[66]:


def merge_date(month, day):
    return month + '-' + str(day)


# ### 将到达日期合并成具体的月份和天，进行统计

# In[67]:


df_new['arrive_time'] = df_new.apply(lambda row: merge_date(row['arrival_date_month'], row['arrival_date_day_of_month']), axis=1)


# In[68]:


df_new['arrive_time'].value_counts()


# ### 从统计可以看出，若想在人多的时候预定酒店，那么一年中最佳预定酒店时间为10月16号；若想在人少的时候预定酒店，那么一年中最佳预定酒店时间为上述出现的日期之外的时间。

# ## 4.利用Logistic预测酒店预订（我理解为预测旅客是否取消队酒店的预定）

# In[55]:


# for ML:
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# In[56]:


#手动选择要包括的列
#为了使模型更通用并防止泄漏，排除了一些列
#（到达日期、年份、指定房间类型、预订更改、预订状态、国家/地区，等待日列表）
#包括国家将提高准确性，但它也可能使模型不那么通用
num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled","agent",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]

cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]
#分离特征和预测值
features = num_features + cat_features
X = df_new.drop(["is_canceled"], axis=1)[features]
y = df_new["is_canceled"]


# In[57]:


#预处理数值特征：
#对于大多数num cols，除了日期，0是最符合逻辑的填充值
#这里没有日期遗漏。
num_transformer = SimpleImputer(strategy="constant")

# 分类特征的预处理：
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

# 数值和分类特征的束预处理：
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                               ("cat", cat_transformer, cat_features)])

# 定义要测试的模型：
base_models = [("LR_model", LogisticRegression(random_state=42,n_jobs=-1))]

#将数据分成“kfold”部分进行交叉验证，
#使用shuffle确保数据的随机分布：
kfolds = 4 # 4 = 75% train, 25% validation
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)

#对每个模型进行预处理、拟合、预测和评分：
for name, model in base_models:
    #将数据和模型的预处理打包到管道中：
    model_steps = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    #获取每个模型的交叉验证分数：
    cv_results = cross_val_score(model_steps, 
                                 X, y, 
                                 cv=split,
                                 scoring="accuracy",
                                 n_jobs=-1)
    # output:
    min_score = round(min(cv_results), 4)
    max_score = round(max(cv_results), 4)
    mean_score = round(np.mean(cv_results), 4)
    std_dev = round(np.std(cv_results), 4)
    print(f"{name} cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")


# ### 可以看到：使用逻辑回归预测旅客是否取消队酒店的预定时，可以达到80%左右的准确率

# In[ ]:




