#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import random
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


# In[ ]:


lemmatzer=WordNetLemmatizer()

#loading the data we have previously saved through creating model 

intensts=json.loads(open('C:/Users/Sakib Ahmed/Desktop/chatbot v1 qna.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

#loading model that we have created and saved
model=load_model('chatbot_model_v1.model')


# In[ ]:


#tokenizing and lemmatizing the sentence here
def clean_up_sentence(sentence):
    sentence_word=nltk.word_tokenize(sentence)
    sentence_word=[lemmatizer.lemmatize(sentence)for sentence in sentence_word]
    return sentence_word


# In[ ]:


#encoding those categorical data to numerical data
def bag_of_words(sentence):
    sentence_word=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if(word==w):
                bag[i]=1
    return np.array(bag)            


# In[ ]:


def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    #if data is grater then threshold then it will be added 
    results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    #sorting list through assending or dessinging order for highest probability among 100 percent
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list    


# In[ ]:


while True:
    message=input("")
    ints=predict_class(message)
    res=get_responce(ints,intents)
    print(res)

