#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np, pandas as pd
import json
import ast 
from textblob import TextBlob
import nltk
import torch
import pickle
from scipy import spatial
import warnings
warnings.filterwarnings('ignore')
import spacy
from nltk import Tree
en_nlp = spacy.load('en')
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

# In[2]:


# !conda update pandas --y

# In[3]:


train = pd.read_csv("data/train.csv")

# In[4]:


train.shape

# ### Loading Embedding dictionary

# In[5]:


with open("data/dict_embeddings1.pickle", "rb") as f:
    d1 = pickle.load(f)

# In[6]:


with open("data/dict_embeddings2.pickle", "rb") as f:
    d2 = pickle.load(f)

# In[7]:


dict_emb = dict(d1)
dict_emb.update(d2)

# In[8]:


len(dict_emb)

# In[9]:


del d1, d2

# ## Data Processing

# In[10]:


def get_target(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]: idx = i
    return idx

# In[11]:


train.head(3)

# In[12]:


train.shape

# In[13]:


train.dropna(inplace=True)

# In[14]:


train.shape

# In[15]:


def process_data(train):
    
    print("step 1")
    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])
    
    print("step 2")
    train["target"] = train.apply(get_target, axis = 1)
    
    print("step 3")
    train['sent_emb'] = train['sentences'].apply(lambda x: [dict_emb[item][0] if item in\
                                                           dict_emb else np.zeros(4096) for item in x])
    print("step 4")
    train['quest_emb'] = train['question'].apply(lambda x: dict_emb[x] if x in dict_emb else np.zeros(4096) )
        
    return train   

# In[16]:


train = process_data(train)

# In[17]:


train.head(3)

# In[ ]:




# ## Predicted Cosine & Euclidean Index

# In[18]:


def cosine_sim(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item,x["quest_emb"][0]))
    return li   

# In[19]:


def pred_idx(distances):
    return np.argmin(distances)   

# In[20]:


def predictions(train):
    
    train["cosine_sim"] = train.apply(cosine_sim, axis = 1)
    train["diff"] = (train["quest_emb"] - train["sent_emb"])**2
    train["euclidean_dis"] = train["diff"].apply(lambda x: list(np.sum(x, axis = 1)))
    del train["diff"]
    
    print("cosine start")
    
    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: pred_idx(x))
    train["pred_idx_euc"] = train["euclidean_dis"].apply(lambda x: pred_idx(x))
    
    return train
    

# In[21]:


predicted = predictions(train)

# In[22]:


predicted.head(3)

# In[23]:


predicted["cosine_sim"][0]

# In[24]:


predicted["euclidean_dis"][0]

# ## Accuracy

# In[25]:


def accuracy(target, predicted):
    
    acc = (target==predicted).sum()/len(target)
    
    return acc

# ### Accuracy for  euclidean Distance

# In[26]:


print(accuracy(predicted["target"], predicted["pred_idx_euc"]))

# ### Accuracy for Cosine Similarity

# In[27]:


print(accuracy(predicted["target"], predicted["pred_idx_cos"]))

# In[44]:


predicted.to_csv("train_detect_sent.csv", index=None)

# In[34]:


predicted.iloc[75207,:]

# In[38]:


ct,k = 0,0
for i in range(predicted.shape[0]):
    if predicted.iloc[i,10] != predicted.iloc[i,5]:
        k += 1
        if predicted.iloc[i,11] == predicted.iloc[i,5]:
            ct += 1

# In[39]:


ct, k

# ### Combining Accuracy

# In[47]:


label = []
for i in range(predicted.shape[0]):
    if predicted.iloc[i,10] == predicted.iloc[i,11]:
        label.append(predicted.iloc[i,10])
    else:
        label.append((predicted.iloc[i,10],predicted.iloc[i,10]))

# In[66]:


ct = 0
for i in range(75206):
    item = predicted["target"][i]
    try:
        if label[i] == predicted["target"][i]: ct +=1
    except:
        if item in label[i]: ct +=1
            

# In[68]:


ct/75206

# ### Root Match

# In[2]:


predicted = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)

# In[39]:


doc = en_nlp(predicted.iloc[0,1])

# In[40]:


predicted.iloc[0,1]

# In[49]:


predicted.iloc[0,2]

# In[41]:


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

# In[42]:


[to_nltk_tree(sent.root).pretty_print()  for sent in en_nlp(predicted.iloc[0,2]).sents]

# In[50]:


[to_nltk_tree(sent.root) .pretty_print() for sent in doc.sents][5]

# In[44]:


for sent in doc.sents:
    roots = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]
    print(roots)

# In[45]:


def match_roots(x):
    question = x["question"].lower()
    sentences = en_nlp(x["context"].lower()).sents
    
    question_root = st.stem(str([sent.root for sent in en_nlp(question).sents][0]))
    
    li = []
    for i,sent in enumerate(sentences):
        roots = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

        if question_root in roots: 
            for k,item in enumerate(ast.literal_eval(x["sentences"])):
                if str(sent) in item.lower(): 
                    li.append(k)
    return li

# In[46]:


predicted["question"][21493]

# In[47]:


predicted["context"][21493]

# In[48]:


predicted["root_match_idx"] = predicted.apply(match_roots, axis = 1)

# In[ ]:




# In[51]:


predicted["root_match_idx_first"]= predicted["root_match_idx"].apply(lambda x: x[0] if len(x)>0 else 0)

# In[52]:


(predicted["root_match_idx_first"]==predicted["target"]).sum()/predicted.shape[0]

# In[55]:


predicted.to_csv("train_detect_sent.csv", index=None)

# In[53]:


predicted[(predicted["sentences"].apply(lambda x: len(ast.literal_eval(x)))<11) &  (predicted["root_match_idx_first"]>10)]       



# In[23]:


len(ast.literal_eval(predicted.iloc[21493,4]))

# In[59]:


question = predicted["question"][21493].lower()
sentences = en_nlp(predicted["context"][21493].lower()).sents
    
question_root = st.stem(str([sent.root for sent in en_nlp(question).sents][0]))
    
li = []
for i,sent in enumerate(sentences):
    roots = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]
    print(roots)

    if question_root in roots: li.append(i)

# In[4]:


ast.literal_eval(predicted["sentences"][21493])

# In[5]:


predicted["context"][21493]

# In[14]:


en_nlp = spacy.load('en')
sentences = en_nlp(predicted["context"][21493].lower()).sents

# In[15]:


for item in sentences:
    print(item)

# In[ ]:




# In[6]:


TfidfVectorizer(predicted["sentences"][0], ngram_range=(1,2))

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



