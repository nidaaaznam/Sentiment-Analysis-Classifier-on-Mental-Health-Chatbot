# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:21:34 2020

@author: Aznida
"""
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
import re
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle
import numpy as np
import pandas as pd


from keras.models import load_model
model = load_model('chatbot_model.h5')
from keras.models import load_model
model_sent = load_model('cnn_w2v.h5')
import json
import random

nltk.download('punkt')

intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
max_seq_len = 500

data_train = pd.read_csv('data/data_train.csv', encoding='utf-8')
data_test = pd.read_csv('data/data_test.csv', encoding='utf-8')
data = data_train.append(data_test, ignore_index=True)

global emotion_array
emotion_array = []

def clean_text(data):
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    # tekenization using nltk
    data = word_tokenize(data)
    
    return data

texts = [' '.join(clean_text(text)) for text in data.Text]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def sentiment(sent_msg):
    seq = tokenizer.texts_to_sequences(sent_msg)
    padded = pad_sequences(seq, maxlen=max_seq_len)
    pred = model_sent.predict(padded)
    return pred

def sentiment_response(sent_msg):
    pred = sentiment(sent_msg)
    emotion_pred  = class_names[np.argmax(pred)]
    
    if emotion_pred=='joy':
        emotion_cond = "joy"
    elif emotion_pred=='fear':
        emotion_cond = "fear"
    elif emotion_pred=='anger':
        emotion_cond = "anger"
    elif emotion_pred=='sadness':
        emotion_cond = "sadness"
    elif emotion_pred=='neutral':
        emotion_cond = "neutral"
    else:
        emotion_cond = "neutral"
        
    return emotion_cond

def mental_state(emotion_array):
    """
    NE=0
    msg_count=0
    mental=''
    i=0
    while i<5:
        i=i+1
        msg_count=msg_count+1
        if emotion_pred[0] or emotion_pred[4]:
            NE=0
        else:
            NE=NE+1
            

    
    return mental
    """
    PE=0
    NE=0
    for emotion in emotion_array:
        if emotion == "joy" or emotion == "neutral":
            PE=PE+1
        elif emotion == "sadness" or emotion == "anger" or emotion == "fear":
            NE=NE+1
    
    p_p = PE/len(emotion_array)
    n_p = NE/len(emotion_array)
    
    if n_p<0.2 and p_p>0.8:
        mental="Emotionally Stable 🙂"
    elif (n_p>=0.2 and n_p<=0.4) and (p_p>=0.6 and p_p<=0.8):
        mental="Slightly Stressed 😐"
    elif (n_p>0.4 and n_p<=0.6) and (p_p>=0.4 and p_p<0.6):
        mental="Highly Stressed 😓"
    elif (n_p>0.6 and n_p<=0.75) and (p_p>=0.25 and p_p<0.4):
        mental="Slightly Depressed 😰"
    elif n_p>0.75 and p_p<0.25:
        mental="Highly Depressed 😭"
    else:
        mental="Pending"
    
            
    print(emotion_array)
    print(NE)
    print(n_p)
    print(PE)
    print(p_p)
    print(mental)
    
    
    return mental
        
   