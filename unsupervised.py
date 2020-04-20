
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



def get_target(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]: idx = i
    return idx

def process_data(train, dict_emb):
    
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

def cosine_sim(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item,x["quest_emb"][0]))
    return li   

def pred_idx(distances):
    return np.argmin(distances)   

def predictions(train):
    
    train["cosine_sim"] = train.apply(cosine_sim, axis = 1)
    train["diff"] = (train["quest_emb"] - train["sent_emb"])**2
    train["euclidean_dis"] = train["diff"].apply(lambda x: list(np.sum(x, axis = 1)))
    del train["diff"]
    
    print("cosine start")
    
    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: pred_idx(x))
    train["pred_idx_euc"] = train["euclidean_dis"].apply(lambda x: pred_idx(x))
    
    return train

def accuracy(target, predicted):
    
    acc = (target==predicted).sum()/len(target)
    
    return acc

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

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

def run_unsupervised(csv_data="data/train.csv", emb_data="data/dict_embeddings.pickle"):

    # read csv data
    train = pd.read_csv(csv_data)

    # read embeddings
    with open(emb_data, "rb") as f:
        dict_emb = dict(pickle.load(f))

    # drop NA values
    train.dropna(inplace=True)

    # read embeddings of data
    train = process_data(train, dict_emb)

    # get closest distance representations
    predicted = predictions(train)

    # Accuracy for  euclidean Distance
    print(accuracy(predicted["target"], predicted["pred_idx_euc"]))

    # Accuracy for Cosine Similarity
    print(accuracy(predicted["target"], predicted["pred_idx_cos"]))

    predicted.to_csv("train_detect_sent.csv", index=None)

  
if __name__ == "__main__":
    run_unsupervised(csv_data="data/train.csv", 
                     emb_data="data/dict_embeddings.pickle")
