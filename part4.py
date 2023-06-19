import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import nltk
from nltk.corpus import stopwords
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, class_names, classifier_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix for {classifier_name}")
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.show()
    
def train_classifier():
    # Load and preprocess data
    data_path = 'data'
    class_folders = os.listdir(data_path)
    if class_folders[0] == '.DS_Store':
        class_folders = class_folders[1:]
    documents = []
    labels = []
    
    for class_label, class_folder in enumerate(class_folders):
        folder_path = os.path.join(data_path, class_folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read().lower())
            labels.append(class_label)


    documents = pd.DataFrame(documents, columns=['text'])

    # Convert documents to a matrix of TF-IDF features
    # Use TfidfVectorizer's built-in tokenizer (default)
    vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode')
    X = vectorizer.fit_transform(documents['text'])
    y = np.array(labels)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classifiers
    classifiers = {
        'SVM': SVC(kernel='linear', probability=True),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy'),
        'Random Forest': RandomForestClassifier(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Nearest Neighbours': KNeighborsClassifier()
    }

    # Train and evaluate classifiers to choose best
    for name, classifier in classifiers.items():

        # classifier.fit(X_train, y_train)
        # y_pred = classifier.predict(X_test)
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"{name} Classifier:")
        # print(f"Accuracy: {accuracy:.4f}")
        # print(classification_report(y_test, y_pred, target_names=class_folders))


        #Multinomial Naive Bayes performed best with our testing
        if name == 'Random Forest':
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Classifier:")
            print(f"Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred, target_names=class_folders))
            plot_confusion_matrix(y_test, y_pred, class_folders, name)
            classifier = classifier.fit(X,y)
            

            #save classifier
            pickle.dump({'classifier': classifier, 'vectorizer': vectorizer.vocabulary_}, open('classifier.model', 'wb'))
        


