#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup 
import csv
import pandas as pd
import numpy as np
import os
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


nltk.download('punkt')


# In[4]:


def call_crawler(url, max_count):
    Que = [url] 
    iter_ = 0
    results=[]  #list to append records
    while(Que!=[] and iter_ < max_count):
        start_url = Que.pop(0)    
        print("Accessing " + start_url)
        
        #Reading HTML text and parsing it so that data can be extracted
        code = requests.get(start_url)
        plain = code.text            
        soup = BeautifulSoup(plain, "html.parser")  
           
        #extracting information from div tag
        record=soup.find_all("div", class_="result-container")

        for paper in record:
            rec={}
            #title of publication
            title_paper = paper.find("h3", class_="title")
            #link of the publication
            link_paper = paper.find("a", class_="link")
            #date of publication published
            date_paper=paper.find("span", class_="date")
            
            #printing extracted information
            print("Title of the Publication: ",title_paper.text)
            print("Link to the publication: ",link_paper.get('href'))
            print("Paper was published on: ",date_paper.text)

            #reading CU authors' details
            auth_name= []   #initialising list to store names of authors
            #to store links to authors' pureportal profile
            auth_link=[]
            
            for auth in paper.find_all("a", class_="link person"):
                #fetching author's profile links
                portal_link = auth.get('href')
                #fetching author's name
                name=auth.string
                auth_name.append(name)
                auth_link.append(portal_link)
                print(name)
                print(portal_link)

            #writing all the fetched details about each publication in dictionary
            rec['Name of Publication']=title_paper.text
            rec['Publication Link']= link_paper.get('href')
            rec['Date of Publish']= date_paper.text
            rec['CU Author']=auth_name
            rec['Pureportal Profile Link']=auth_link
            
            #appending all rows in list
            results.append(rec)
        
        #searching link for next page and storing in queue so that it can be traversed next        
        pg_link=soup.find("a", class_="nextLink")
        #It will store links till last page only
        if(pg_link!= None):
            #getting link to next page
            url_next_page=pg_link.get('href')
            
            #Normalisation
            baseurl = "https://pureportal.coventry.ac.uk"
            link_next_page= baseurl+url_next_page
            print(link_next_page)
            
            #appending link to next page in queue
            Que.append(link_next_page) 

        #writing information to csv file
        file = open('record.csv', 'w', encoding="utf-8")
        field_names=['Name of Publication','Publication Link','Date of Publish','CU Author','Pureportal Profile Link']
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for rec in results:
            writer.writerow(rec)
        file.close()
        
        iter_+= 1
        
    print("Crawling completed ") 
        


# In[5]:


#Calling crawler function
run_crawler=input("Do you want to run a crawler (y/n): ").lower()
if (run_crawler=='y'):
    call_crawler('https://pureportal.coventry.ac.uk/en/organisations/school-of-life-sciences/publications/',18)
else:
    if(os.path.isfile('record.csv')):
        df=pd.read_csv("record.csv", names=['Name of Publication','Publication Link','Date of Publish',
                                            'CU Author','Pureportal Profile Link'],
                                       encoding= 'unicode_escape') 
        print("Search started")
    else: 
        print("No Crawler output exists, you need to run crawler first!")


# In[6]:


#Reading csv file and storing contents in dataframe
df=pd.read_csv("record.csv")

print("No. of papers found: ", df.count())
df.head()


# In[7]:


#Pre-processing
import re
import string
processed_title = []
for title in df['Name of Publication']:
    # Removing Unicode
    clean_doc = re.sub(r'[^\x00-\x7F]+', '', title)
    # Removing Mentions
    clean_doc = re.sub(r'@\w+', '', clean_doc)
    # Converting into lowercase
    clean_doc = clean_doc.lower()
    # Removing punctuations
    clean_doc = re.sub(r'[%s]' % re.escape(string.punctuation), '', clean_doc)
    #removing stop words
    st = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = st.sub('', clean_doc)
    processed_title.append(text)
    
print(processed_title)


# In[8]:


title_list=[]
for i in processed_title:
    token_words=word_tokenize(i)
    stem_words=[]
    for word in token_words:
        #Stemming and removing stop words
        if(word.isalpha()):
            stemmer = PorterStemmer()
            stem_words.append(stemmer.stem(word))
            query=' '.join(stem_words)
    title_list.append(query)

print(title_list)


# In[9]:


#Tfidf model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(title_list)
features = vectorizer.get_feature_names()
X = vectors.todense()
dense_list = X.tolist()
tf_idf = pd.DataFrame(dense_list, columns=features)
# print the full sparse matrix
tf_idf


# In[11]:



#Query processor and Indexer
searchstring=input('enter a string')
#Pre-processing input string
searchstring=searchstring.lower()
#tokenization
token_words=word_tokenize(searchstring)
stem_words=[]
for word in token_words:
    #Stemming and removing stop words, punctualtion
    if word not in stopwords.words("english"):
        if(word.isalpha()):
            stem_words.append(stemmer.stem(word))
            query=' '.join(stem_words)
processed_query=query.split()
print(processed_query)    #Preprocessed search string

match=list()

for i in range(len(title_list)):
    matches=0
    for j in range(len(processed_query)):
        try:
            t=processed_query[j]
            matches+=tf_idf.iloc[i][t]
        except KeyError:
            matches+=0
    match.insert(i,matches)

boolean = all(element == 0 for element in match)

if(boolean):
    print("No related research paper found.")
    
else:
    a=( sorted( [(x,i) for (i,x) in enumerate(match)], reverse=True ) [:5])
    pd.set_option('display.max_colwidth', None)
    for i in range(len(a)):
        max_index=a[i][1]
        print(df.iloc[max_index])
        print("***********************************************")


# In[12]:


import feedparser
news=[]
#extracting news from BBC website using RSS feed
Politics = feedparser.parse("http://feeds.bbci.co.uk/news/politics/rss.xml")
Education = feedparser.parse("http://feeds.bbci.co.uk/news/education/rss.xml")
Science = feedparser.parse("http://feeds.bbci.co.uk/news/science_and_environment/rss.xml")

for i in Politics.entries:
    pol= i.summary
    news.append(pol)

for i in Education.entries:
    edu= i.summary
    news.append(edu)

for i in Science.entries:
    sci= i.summary
    news.append(sci)

print("No of news headlines retrieved: ", len(news))
print(news)


# In[13]:


#Preprocessing
ps = PorterStemmer()
processed_news = []
for i in news:
    tokens = word_tokenize(i.lower())
    n = ""
    for w in tokens:
        if w not in stopwords.words("english"):
            if (w.isalpha()):
                n += ps.stem(w) + " "
    processed_news.append(n)
print(processed_news)


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_news)
print(X.todense())


# In[15]:


from sklearn.cluster import KMeans
K = 3
cluster_model = KMeans(n_clusters=K)
cluster_model.fit(X)

print(cluster_model.labels_)


# In[16]:


#Taking input from user
query = input("Enter a string: ")

#pre-processing user input
tokens = word_tokenize(query)
x = ""
for w in tokens:
    if w not in stopwords.words("english"):
        x += ps.stem(w) + " "
print(x)   #preprocessed user query

#predicting cluster
Y = vectorizer.transform([x])
prediction = cluster_model.predict(Y)
print("cluster is: ", prediction)

