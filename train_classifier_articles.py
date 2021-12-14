# import libraries
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy
from pandas import DataFrame
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sqlite3

# import libraries for ML
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer

# import libraries from sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report

# import grid search
from sklearn.model_selection import GridSearchCV

# import pickle to load models
import pickle


def load_data(database_filepath):
    # connect to the database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', conn)
    # create X and y variables to be later used in classification
    X: DataFrame = df['message']
    Y: DataFrame = df.drop(['message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize text
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^a-zA-A0-9]', ' ', text)
    # tokenize the text
    text = word_tokenize(text)
    # Remove stop words
    text = [w for w in text if w not in stop_words]

    # Lemmatization
    # Reduce words to their root form
    # lemmatize for nouns
    text = [lemmatizer.lemmatize(x) for x in text]
    # lemmatize for verbs
    text = [lemmatizer.lemmatize(x, pos='v') for x in text]

    return text


def build_model():

    # create pipeline for the model
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # define parameters to tweak through grid search
    parameters = {
        'features__text_pipeline__count_vect__binary': [True, False],
        'features__text_pipeline__tfidf__smooth_idf': [True, False],
        'clf__estimator__learning_rate': [1, 0.7]
    }

    # create a grid search object based on define pipeline and parameters above
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    # predict Y (categories) based on the test data
    Y_pred = model.predict(X_test)

    # loop fro each column/category name and produce a classification report
    for col, col_name in enumerate(category_names):
        print("Classification report for column " + str(col + 1) + " is:\n" + classification_report(Y_test.iloc[:, col], Y_pred[:, col]))


def save_model(model, model_filepath):
    # Save to file in the current working directory
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()