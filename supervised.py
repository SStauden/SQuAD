#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np, pandas as pd
import ast 
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import warnings
warnings.filterwarnings('ignore')
import spacy
from nltk import Tree
en_nlp = spacy.load('en')
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# In[2]:


data = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)

# In[3]:


data.shape

# In[4]:


data.head(3)

# In[5]:


ast.literal_eval(data["sentences"][0])

# In[6]:


data = data[data["sentences"].apply(lambda x: len(ast.literal_eval(x)))<11].reset_index(drop=True)

# In[7]:


def create_features(data):
    train = pd.DataFrame()
     
    for k in range(len(data["euclidean_dis"])):
        dis = ast.literal_eval(data["euclidean_dis"][k])
        for i in range(len(dis)):
            train.loc[k, "column_euc_"+"%s"%i] = dis[i]
    
    print("Finished")
    
    for k in range(len(data["cosine_sim"])):
        dis = ast.literal_eval(data["cosine_sim"][k].replace("nan","1"))
        for i in range(len(dis)):
            train.loc[k, "column_cos_"+"%s"%i] = dis[i]
            
    train["target"] = data["target"]
    return train

# In[8]:


train = create_features(data)

# In[9]:


del data

# In[10]:


train.head(3)

# In[11]:


# train.fillna(10000, inplace=True)

# In[12]:


train.head(3).transpose()

# ### Fitting Multinomial Logistic Regression

# ### Standardize

# In[13]:


train.apply(max, axis = 0)

# In[14]:


subset1 = train.iloc[:,:10].fillna(60)
subset2 = train.iloc[:,10:].fillna(1)

# In[15]:


 subset1.head(3)

# In[16]:


 subset2.head(3)

# In[17]:


train2 = pd.concat([subset1, subset2],axis=1, join_axes=[subset1.index])

# In[18]:


train2.head(3)

# In[19]:


train2.apply(max, axis = 0)

# In[20]:


scaler = MinMaxScaler()
X = scaler.fit_transform(train2.iloc[:,:-1])

# In[21]:


X

# In[22]:


train_x, test_x, train_y, test_y = train_test_split(X,
train.iloc[:,-1], train_size=0.8, random_state = 5)

# In[23]:


mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))


# ### Logistic-Regression with Root Match feature

# In[24]:


predicted = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)

# In[25]:


predicted = predicted[predicted["sentences"].apply(lambda x: len(ast.literal_eval(x)))<11].reset_index(drop=True)

# In[26]:


predicted.shape

# In[27]:


def get_columns_from_root(train):
    
    for i in range(train.shape[0]):
        if len(ast.literal_eval(train["root_match_idx"][i])) == 0: pass
        
        else:
            for item in ast.literal_eval(train["root_match_idx"][i]):
                train.loc[i, "column_root_"+"%s"%item] = 1
    return train

# In[28]:


predicted = get_columns_from_root(predicted)

# In[29]:


predicted.head(3).transpose()

# In[30]:


subset3 = predicted[["column_root_0","column_root_1","column_root_2","column_root_3","column_root_4","column_root_5",\
             "column_root_6","column_root_7","column_root_8","column_root_9"]]

# In[31]:


subset3.fillna(0, inplace=True)

# In[32]:


train3 = pd.concat([subset3, train2],axis=1, join_axes=[subset3.index])

# In[33]:


train3.head(3).transpose()

# In[34]:


train3 = train3[["column_root_0","column_root_1","column_root_2","column_root_3","column_root_4","column_root_5",\
             "column_root_6","column_root_7","column_root_8","column_root_9", "column_cos_0","column_cos_1",\
           "column_cos_2","column_cos_3","column_cos_4","column_cos_5",\
             "column_cos_6","column_cos_7","column_cos_8","column_cos_9", "target"]]

# In[35]:


train_x, test_x, train_y, test_y = train_test_split(train3.iloc[:,:-1],
train3.iloc[:,-1], train_size=0.8, random_state = 5)

# In[36]:


mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))


# ### Random Forest

# In[64]:


rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
rf.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, rf.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, rf.predict(test_x)))

# ### XgBoost

# In[72]:


model = xgb.XGBClassifier()
param_dist = {"max_depth": [3,5,10],
              "min_child_weight" : [1,5,10],
              "learning_rate": [0.07, 0.1,0.2],
               }

# run randomized search
grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 
                                   verbose=5, n_jobs=-1)
grid_search.fit(train_x, train_y)

# In[73]:


grid_search.best_estimator_

# In[74]:


xg = xgb.XGBClassifier(max_depth=5)
xg.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, xg.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, xg.predict(test_x)))

# In[69]:


xgb.XGBClassifier??

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



