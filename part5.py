import requests
from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from datetime import datetime
import justext
import pandas as pd

def download_content(resp):
    '''
        Downloads content from the web page of the URL given.

        Params:
            resp = response from HTTP request
        
        Return:
            content = content of the web page
            download_datetime = date & time the web page was downloaded
    '''
    stop_words = set(stopwords.words('english'))

    # download content from the web page
    content = ""


    paragraphs = justext.justext(resp.content, justext.get_stoplist("English"))
    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            # this asseration makes sure we catch string and unicode only
            assert isinstance(paragraph.text, str)
            # https://portingguide.readthedocs.io/en/latest/strings.html
            # convert byte string to readable text
            if type(paragraph.text) == bytes:
                paragraph_text = paragraph.text.decode('utf8', 'ignore')
            else:
                paragraph_text = paragraph.text

            # tokenize text
            paragraph_tokens = regexp_tokenize(paragraph_text, r'\s+', gaps=True)

            # remove stop words
            filtered_paragraph = [w for w in paragraph_tokens if not w.lower() in stop_words]
            # join string back together
            filtered_paragraph = ' '.join(filtered_paragraph)

            # add to final content string for the file
            content += str(filtered_paragraph) + '\n'

    # print(content)
    content = content.lower()
    return [content]

def vectorize_text(text, train_tfidf_vocab):

    document = pd.DataFrame(text, columns=['text'])
    vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', vocabulary=train_tfidf_vocab)
    
    tfidf_text = vectorizer.fit_transform(document['text'])

    return tfidf_text

def predict_link(classifier, train_tfidf_vocab):

    #Get URL from user
    url = input("Enter a URL: ")

    resp = requests.get(url)

    #If response is not successful, print error and exit
    if resp.status_code != 200:
        print(f"HTTP code: {resp.status_code} for URL: {url}.")
        quit()

    # Download text content from URL, preprocess and vectorize text
    content = download_content(resp)

    #Vectorize content
    X_input = vectorize_text(content, train_tfidf_vocab)

    #Get list of labels
    class_folders = os.listdir('data')
    if class_folders[0] == '.DS_Store':
        class_folders = class_folders[1:]

    # print(class_folders)

    # Make predictions
    predicted_label_idx = classifier.predict(X_input)[0]
    predicted_label = class_folders[predicted_label_idx]

    # print(classifier.predict(X_input))
    # print(classifier.predict_proba(X_input))

    confidence_percentage = classifier.predict_proba(X_input)[0][predicted_label_idx] * 100

    # Print predicted label and confidence score
    print(f"<{predicted_label}, {confidence_percentage:.2f}%>")
    


    
def run():
    #load saved model from part4:
    with open('classifier.model', 'rb') as f:
            loaded_model = pickle.load(f)

        
    classifier = loaded_model['classifier']
    train_tfidf_vocab = loaded_model['vectorizer']

    #use saved model to predict link 
    predict_link(classifier, train_tfidf_vocab)
