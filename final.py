#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
nltk.download('wordnet')

import warnings
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
import re
#!pip install rake_nltk
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# loading the trained model
pickle_in = open('model12.pkl', 'rb') 
model = pickle.load(pickle_in)

pickle_in = open('vectorizer.pkl', 'rb') 
vectorizer = pickle.load(pickle_in)
# Title of the application 
st.title('Text Analysis Tool\n', )
st.header("Sentiment Analysis Tool")
st.subheader("Enter the review that you want to analyze")

input_text = st.text_area("Enter review", height=50)

# Sidebar options
option = st.sidebar.selectbox('Navigation',['Sentiment Analysis','Word Cloud'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Sentiment Analysis":
    
    
    
    if st.button("Predict sentiment"):
        st.write("Number of words in Review:", len(input_text.split()))
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open('model12.pkl', 'rb') 
        model = pickle.load(pickle_in)
        pickle_in = open('vectorizer.pkl', 'rb') 
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])
        
        if model.predict(transformed_input) ==0:
            st.write("Input review has Negative Sentiment.:sad:")
        elif model.predict(transformed_input) ==2:
            st.write("Input review has Positive Sentiment.:smile:")
        else:
            st.write(" Input review has Neutral Sentiment.😐")
         

elif option == "Word Cloud":
    st.header("Word cloud")
    if st.button("Generate Wordcloud"):
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        wordcloud = WordCloud().generate(text)
        plt.figure(figsize=(40, 30))
        plt.imshow(wordcloud) 
        plt.axis("off")
        
        st.pyplot()


# In[ ]:




