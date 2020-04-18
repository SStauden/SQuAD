
import warnings
warnings.filterwarnings('ignore')
from InferSent.models import InferSent
import pickle
import numpy as np
import pandas as pd
import json
from textblob import TextBlob
import nltk
from scipy import spatial
import torch
import spacy
import os

# en_nlp = spacy.load('en')

def gen_dict_embeddings(datafile="data/train-v1.1.json"):
        
    # Convert Json to Pandas Dataframe
    data = pd.read_json(datafile)

    contexts = []
    questions = []
    answers_text = []
    answers_start = []

    for i in range(data.shape[0]):
        topic = data.iloc[i,0]['paragraphs']
        for sub_para in topic:
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                answers_start.append(q_a['answers'][0]['answer_start'])
                answers_text.append(q_a['answers'][0]['text'])
                contexts.append(sub_para['context'])   
    df = pd.DataFrame({"context":contexts, "question": questions, "answer_start": answers_start, "text": answers_text})

    outfile = os.path.basename(datafile).split(".")[0]
    df.to_csv("data/{}.csv".format(outfile), index = None)

    print("Saved Data as CSV to {}".format("data/{}.csv".format(outfile)))

    # Create dictionary of sentence embeddings for faster computation
    paras = list(df["context"].drop_duplicates().reset_index(drop= True))
    blob = TextBlob(" ".join(paras))
    sentences = [item.raw for item in blob.sentences]

    # # infersent = torch.load('InferSent/encoder/infersent.allnli.pickle', map_location=lambda storage, loc: storage, pickle_module=pickle)
    # infersent = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
    # infersent.set_glove_path("InferSent/dataset/GloVe/glove.840B.300d.txt")

    V = 2
    MODEL_PATH = 'Infersent/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = 'Infersent/fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)

    infersent.build_vocab(sentences, tokenize=True)

    dict_embeddings = {}
    for i in range(len(sentences)):
        print("processing sentence {}/{}\r".format(i, len(sentences)), end="")
        dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)

    questions = list(df["question"])

    for i in range(len(questions)):
        print("processing question {}/{}\r".format(i, len(questions)), end="")
        dict_embeddings[questions[i]] = infersent.encode([questions[i]], tokenize=True)

    with open('data/{}_dict_embeddings.pickle'.format(outfile), 'wb') as handle:
        pickle.dump(dict_embeddings, handle)

    print("Saved Embeddings to {}".format('data/{}_dict_embeddings.pickle'.format(outfile)))

    del dict_embeddings

if __name__ == "__main__":

    gen_dict_embeddings(datafile="data/train-v1.1.json")
    gen_dict_embeddings(datafile="data/dev-v1.1.json")
