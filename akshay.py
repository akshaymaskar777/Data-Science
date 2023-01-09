#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[2]:


review=pd.read_excel(r"hotel_reviews.xlsx",sheet_name='hotel_reviews')
review


# # Information about dataset

# In[3]:


review.shape


# In[4]:


review.info()


# In[5]:


review[review.duplicated()]  ### To see the Duplicate Reviews


# In[6]:


review.isnull().sum()


# In[7]:


review["Rating"].value_counts()


# In[8]:


review.groupby("Rating").count().plot.pie(y="Review",autopct="%.2f%%",figsize=(6,6)); #rating in form of pie chart


# In[9]:


review['Rating_sentiment'] = None
for index, rows in review.iterrows():
    if(rows['Rating'] >= 1 and rows['Rating'] < 3):
        review.at[index, 'Rating_sentiment'] = 'Negitive'
    elif(rows['Rating'] == 3):
        review.at[index,'Rating_sentiment'] = 'Neutral'
    elif(rows['Rating'] > 3 and rows['Rating'] <= 5 ):
        review.at[index,'Rating_sentiment'] = 'Positive'
review.head()


# In[10]:


sns.countplot(x='Rating_sentiment',data=review ).set_title("Frequency of Ratings")


# In[11]:


review.groupby("Rating_sentiment").count().plot.pie(y="Review",autopct="%.2f%%",figsize=(6,6));


# In[12]:


review["Rating_sentiment"].value_counts()


# # TEXT PRE-PROCESSING

# Exploratory Data Analusis (EDA)

# Expanding Contractions
# 

# In[13]:


import contractions
review['no_contract'] = review['Review'].apply(lambda x: [contractions.fix(word) for word in x.split()])
review['no_contract']


# In[14]:


review['content_str'] = [' '.join(map(str, l)) for l in review['no_contract']]
review['content_str']


# # Removing Punctuations

# In[15]:


review['no_punc'] = review['content_str'].str.replace('[^\w\s]','')
review


# # Tokenization

# In[16]:


from nltk.tokenize import word_tokenize
review['tokenized'] = review['no_punc'].apply(word_tokenize)
review.iloc[:,2:]


# # Normalize the data

# In[17]:


review['lower'] = review['tokenized'].apply(lambda x: [word.lower() for word in x])
review.iloc[:,4:]


# # Removing Stopwords

# In[18]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
review['no_stopword'] = review['lower'].apply(lambda x: [word for word in x if word not in stop_words])
review.iloc[:,4:]


# # Number of words

# In[19]:


#Number of Words in single review
review['no_word'] = review['no_stopword'].apply(lambda x: len(str(x).split(" ")))
review.iloc[:,6:]


# # Number of Characters

# In[20]:


#Number of characters in single review 
review['char_count'] = review['no_punc'].str.len() 
review.iloc[:,6:]


# # Stemming

# In[21]:


review['stp_rem_str'] = [' '.join(map(str, l)) for l in review['no_stopword']]
#from nltk.stem import PorterStemmer
#st = PorterStemmer()
#review['stemmi']=review['stp_rem_str'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
#review.iloc[:,-1]


# # Lemmatization

# In[22]:


from textblob import Word
review['lemma'] = review['stp_rem_str'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# # Removing Numbers

# In[23]:



import re
# Removing numbers form text
review['cleaned']=review['lemma'].apply(lambda x: re.sub('\w*\d\w*','', x))
review.iloc[:,-2:]


# # advance cleaning 

# In[24]:


def cleantext(text):
    text = re.sub(r"â€™", "", text) # Remove Mentions
    text = re.sub(r"#", "", text) # Remove Hashtags Symbol
    text = re.sub(r"\w*\d\w*", "", text) # Remove numbers
    text = re.sub(r"https?:\/\/\S+", "", text) # Remove The Hyper Link
    text = re.sub(r"______________", "", text) # Remove _____
    
    
    return text


# In[25]:


review['cleaned'] = review['cleaned'].apply(cleantext)
review.head()


# In[26]:


review['cleaned']


# # TFIDF - Term frequency inverse Document Frequency

# In[27]:



from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True) #keep top 1000 words
TFIDF= vectorizer.fit_transform(review["cleaned"])
names_features = vectorizer.get_feature_names()
dense= TFIDF.todense()
denselist =dense.tolist()
df = pd.DataFrame(denselist, columns = names_features)
df


# # N-gram

# In[28]:


#Bi-gram
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[29]:


top2_words = get_top_n2_words(review["cleaned"], n=200) #top 200
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df


# In[30]:


#Bi-gram plot
import matplotlib.pyplot as plt
import seaborn as sns
top20_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])
plt.show()


# In[31]:


#tri-gram
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[32]:


top3_words = get_top_n3_words(review["cleaned"], n=200) #top 200
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["tri-gram", "Freq"]
top3_df


# In[33]:


top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_trigram["tri-gram"],y=top20_trigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_trigram["tri-gram"])
plt.show()


# # Visualizing positive and negative words using wordcloud

# In[34]:


# Finding total words in Cleaned review
clean_review_words = " ".join(review['cleaned'])
clean_review_words = clean_review_words.split()
len(clean_review_words)# 1997097 words are present  


# In[35]:


clean_review_words[0:10]


# In[36]:


string_Total = " ".join(review["cleaned"])


# In[37]:


from wordcloud import WordCloud
wordcloud_all = WordCloud(background_color='black', width=1800, height=1400, max_words=100).generate(string_Total)        


# In[38]:


# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_all) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show();


# # Emotion Mining

# In[39]:


#Sentiment analysis
afinn = pd.read_csv('Afinn.csv', sep=',', encoding='latin-1')
afinn.shape


# In[40]:


afinn.head()


# In[41]:


affinity_scores = afinn.set_index('word')['value'].to_dict()


# In[42]:


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
import spacy
nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score


# In[ ]:





# In[43]:


review['sentiment_value'] = review['Review'].apply(calculate_sentiment)
review['sentiment_value']


# In[44]:


review['sentiment_value'].describe()


# In[45]:


review['emotion_sentiment'] = None
for index, rows in review.iterrows():
    if(rows['sentiment_value'] >= -50 and rows['sentiment_value'] < 0):
        review.at[index, 'emotion_sentiment'] = 'Negitive'
    elif(rows['sentiment_value'] >= 0 and rows['sentiment_value'] < 10):
        review.at[index,'emotion_sentiment'] = 'Neutral'
    elif(rows['sentiment_value'] > 10 and rows['sentiment_value'] <= 300 ):
        review.at[index,'emotion_sentiment'] = 'Positive'
review.head()
review['emotion_sentiment']


# In[46]:


review.groupby("emotion_sentiment").count().plot.pie(y="Review",autopct="%.2f%%",figsize=(6,6));


# In[47]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(review['sentiment_value'])


# In[48]:


import nltk
nltk.download('vader_lexicon')


# In[49]:


# add sentiment anaylsis columns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
review["sentiments"] = review["Review"].apply(lambda x: sid.polarity_scores(x))

review = pd.concat([review.drop(['sentiments'], axis=1), review['sentiments'].apply(pd.Series)], axis=1)

review.head()


# In[50]:


reviews=np.array(review['cleaned'])
s=str(reviews)
token = word_tokenize(s)
print(token)


# In[51]:


from nltk.probability import FreqDist
mostcommon_1 = FreqDist(token).most_common(15)
x, y = zip(*mostcommon_1)
plt.figure(figsize=(50,25))
plt.margins(0.02)
plt.bar(x, y)
plt.xlabel('Words', fontsize=20)
plt.ylabel('Frequency of Words', fontsize=40)
plt.yticks(fontsize=40)
plt.xticks(rotation=60, fontsize=40)
plt.title('Frequency of 10 Most Common Words', fontsize=40)
plt.show()


# In[52]:


final_data=review[['cleaned','Rating_sentiment']]


# In[53]:


final_data


# In[54]:


final_data['Rating_sentiment'] = final_data['Rating_sentiment'].replace({'Negitive': 0})
final_data['Rating_sentiment'] = final_data['Rating_sentiment'].replace({'Positive': 2})
final_data['Rating_sentiment'] = final_data['Rating_sentiment'].replace({'Neutral': 1})


# In[55]:


final_data


# In[56]:


y=final_data['Rating_sentiment']
y


# In[57]:


review['sentiment_value']


# In[58]:


corpus = final_data['cleaned'].tolist()


# In[59]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
df_tf_idf= vectorizer.fit_transform(corpus).toarray()


# In[60]:


x_tfidf = pd.DataFrame(df_tf_idf)
x_tfidf


# In[61]:


ydf=pd.DataFrame(y)
posi,neg,neu=ydf.value_counts()
posi


# In[62]:


import pickle
pickle_out=open('vectorizer.pkl','wb')
pickle.dump(vectorizer,pickle_out)
pickle_out.close()


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# In[64]:


x_traintfidf, x_testtfidf,y_traintfidf,y_testtfidf = train_test_split(x_tfidf,ydf, test_size=0.35,random_state=0)
x_traintfidf.shape,y_traintfidf.shape, x_testtfidf.shape,y_testtfidf.shape


# # Balancing the splited data using SMOTE method

# In[65]:


#!pip install imblearn


# In[66]:


from imblearn.over_sampling import SMOTE


# In[67]:


oversample = SMOTE()
x_traintfidf1, y_traintfidf1 = oversample.fit_resample(x_traintfidf, y_traintfidf)


# In[68]:


classifier2 = LogisticRegression(solver='lbfgs', max_iter=500, multi_class='multinomial')
classifier2.fit(x_traintfidf, y_traintfidf)
log_pred_test = classifier2.predict(x_testtfidf)
log_pred_train=classifier2.predict(x_traintfidf)
acc_log_train2=accuracy_score(y_traintfidf,log_pred_train)*100
acc_log_test2 = accuracy_score(y_testtfidf, log_pred_test) * 100
print('Accuracy of Training data =',acc_log_train2)
print("Accuracy of Test data =", acc_log_test2)


# In[69]:


classifier4 = RandomForestClassifier(n_estimators=400 ,max_depth=10)
classifier4.fit(x_traintfidf, y_traintfidf)
pred_test = classifier4.predict(x_testtfidf)
pred_train=classifier4.predict(x_traintfidf)
acc_train4=accuracy_score(y_traintfidf,pred_train)*100
acc_test4 = accuracy_score(y_testtfidf, pred_test) * 100
print('Accuracy of Training data =',acc_train4)
print("Accuracy of Test data =", acc_test4)


# In[70]:


model = LinearSVC()
model.fit(x_traintfidf, y_traintfidf)
pred_test = model.predict(x_testtfidf)
pred_train=model.predict(x_traintfidf)
acc_train6=accuracy_score(y_traintfidf,pred_train)*100
acc_test6 = accuracy_score(y_testtfidf, pred_test) * 100
print('Accuracy of Training data =',acc_train6)
print("Accuracy of Test data =", acc_test6)


# In[71]:


model12 = MultinomialNB()
model12.fit(x_traintfidf, y_traintfidf)
pred_test = model12.predict(x_testtfidf)
pred_train=model12.predict(x_traintfidf)
acc_train10=accuracy_score(y_traintfidf,pred_train)*100
acc_test10 = accuracy_score(y_testtfidf, pred_test) * 100
print('Accuracy of Training data =',acc_train10)
print("Accuracy of Test data =", acc_test10)


# In[72]:


Ada_model=AdaBoostClassifier()
Ada_model.fit(x_traintfidf, y_traintfidf)
pred_test = Ada_model.predict(x_testtfidf)
pred_train=Ada_model.predict(x_traintfidf)
add_train=accuracy_score(y_traintfidf,pred_train)*100
add_test= accuracy_score(y_testtfidf, pred_test) * 100
print('Accuracy of Training data =',add_train)
print("Accuracy of Test data =", add_test)


# In[73]:


xg_model=XGBClassifier()
xg_model.fit(x_traintfidf, y_traintfidf)
pred_test = xg_model.predict(x_testtfidf)
pred_train=xg_model.predict(x_traintfidf)
xg_train=accuracy_score(y_traintfidf,pred_train)*100
xg_test= accuracy_score(y_testtfidf, pred_test) * 100
print('Accuracy of Training data =',xg_train)
print("Accuracy of Test data =", xg_test)


# In[74]:


AS={'Models':['Logistic Regression (TFIDF)', 'Random Forest(TFIDF)', 'LinearSVC(TFIDF)', 'Multinomial Naive Bayes(TFIDF)','Add Boost(TFIDF)','XGBM(TFIDF)'],
    'Train Accuracy':[acc_log_train2,acc_train4,acc_train6,acc_train10,add_train,xg_train]
    ,'Test Accuracy':[acc_log_test2,acc_test4,acc_test6,acc_test10,add_test,xg_test]}


# In[75]:


Model=pd.DataFrame(AS)
Model


# In[76]:


def clean_text(a):
    print("Number of words in Review:", len(a.split()))
    text=re.sub('[^A-za-z0-9]',' ',a)
    text=text.lower()
    text=text.split(' ')
    text = ' '.join(text)
    top2_words = get_top_n2_words([text],n=5) 
    df = pd.DataFrame(top2_words)
    df.columns=["Bi-gram", "Freq"]
    print(df)
   
    return text


# In[77]:


def expression_check(prediction_input):
    if prediction_input == -1:
        print("Input statement has Negative Sentiment.")
    elif prediction_input == 1:
        print("Input statement has Positive Sentiment.")
    else:
        print(" Input statement has Neutral Sentiment.")


# In[78]:


# function to take the input statement and perform the same transformations we did earlier
def sentiment_predictor(input):
    input = clean_text(input)
    transformed_input = vectorizer.transform([input])
    prediction = model12.predict(transformed_input)
    expression_check(prediction)


# In[79]:


a=('it was amazing')


# In[80]:


sentiment_predictor(a)


# In[81]:


b=('the worst hotel')


# In[82]:


sentiment_predictor(b)


# In[83]:


pickle_out=open('model12.pkl','wb')
pickle.dump(model12,pickle_out)
pickle_out.close()


# In[ ]:




