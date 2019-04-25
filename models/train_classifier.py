import sys
# import libraries
import pandas as pd
import re
import numpy as np
import pickle

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Method to load in the required data from a SQL DB.
    
    Args:
    database_filepath: Path to the DB file
    Returns:
    X pandas_dataframe: dataframe
    Y pandas_dataframe: dataframe
    category_names list: labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df['message'] 
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    return X, Y, Y.columns


def tokenize(text):
    """
    Method to tokenize the text.
    
    Args:
    test: Text to tokenize.
    Returns:
    lemmed : tokenized text
    """
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed


def build_model():
    """
    Method to build the model/pipeline.
    
    Returns:
    pipeline sklearn_pipeline: pipeline built
    """ 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0)
    }

    return GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Method to evaluate a model.
    
    Args:
    model: model to evaluate
    X_test: data to preid_ct
    Y_test: expected predictions
    category_names: name of the categories
_   """
    y_pred = model.predict(X_test)
    y_test = Y_test.values
    for i in range(1, 37):
        print(classification_report(y_test[i], y_pred[i]))


def save_model(model, model_filepath):
    """
    Method to save the model to a pickle file.
    
    Args:
    model: Model to save.
    model_filepath: File to save the model to.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Main method to train a classifier.
    """
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