# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:30:46 2020

@author: Aznida
"""
from chatbot_function import *
import flask
import time
from flask import Flask, render_template, request


global emotion_array
emotion_array = []

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chatbot_response(userText))

@app.route("/getsentiment")
def get_bot_sentiment():
    time.sleep(2)
    userText = request.args.get('msg')
    sent_msg = [userText]
    emotion_pred = sentiment_response(sent_msg)
    emotion_array.append(emotion_pred)
    mental = mental_state(emotion_array)

    if emotion_pred=="joy":
        emotion_cond = "Joy 😄"
    elif emotion_pred=="fear":
        emotion_cond = "Fear 😰"
    elif emotion_pred=="anger":
        emotion_cond = "Angry 😡"
    elif emotion_pred=="sadness":
        emotion_cond = "Sadness 😢"
    elif emotion_pred=="neutral":
        emotion_cond = "Neutral 😐"
    else:
        emotion_cond = "Neutral 😐"

    return str('Based on my sentiment calculation you are currently feeling: <br><br> <i>Emotion:</i> <b>"'+emotion_cond+'"</b>  <br> <i>Mental State:</i> <b>"'+mental+'"<b>')

if __name__ == "__main__":
    app.debug = True
    app.run(host = '0.0.0.0',port=80)
    
    