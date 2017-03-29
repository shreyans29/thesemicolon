# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 09:11:32 2017

@author: Shreyans
"""

import os
from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model


import theano
theano.config.optimizer="None"


model=load_model('LSTM5000.h5')
mod = gensim.models.Word2Vec.load('word2vec.bin');
while(True):
    x=raw_input("Enter the message:");
    sentend=np.ones((300L,),dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [mod[w] for w in sent if w in mod.vocab]

    sentvec[14:]=[]
    sentvec.append(sentend)
    if len(sentvec)<15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend) 
    sentvec=np.array([sentvec])
    
    predictions = model.predict(sentvec)
    outputlist=[mod.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    output=' '.join(outputlist)
    print output
            