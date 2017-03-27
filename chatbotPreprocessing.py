# -*- coding: utf-8 -*-


import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle

os.chdir("D:\semicolon\Deep Learning\chatbot");
model = gensim.models.Word2Vec.load('word2vec.bin');
path2="corpus";
file=open(path2+'/conversation.json');
data = json.load(file)
cor=data["conversations"];

x=[]
y=[]

path2="corpus";

for i in range(len(cor)):
    for j in range(len(cor[i])):
        if j<len(cor[i])-1:
            x.append(cor[i][j]);
            y.append(cor[i][j+1]);

tok_x=[]
tok_y=[]
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))
    
    

sentend=np.ones((300L,),dtype=np.float32) 

vec_x=[]
for sent in tok_x:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sentvec)
    
vec_y=[]
for sent in tok_y:
    sentvec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sentvec)           
    
    
for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)    
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)             
            
            
with open('conversation.pickle','w') as f:
    pickle.dump([vec_x,vec_y],f)                
