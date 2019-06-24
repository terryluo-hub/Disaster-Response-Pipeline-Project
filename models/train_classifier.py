import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib


import re
import sys


nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')


def load_data(database_filepath):
    
    '''

    Connect to the database, then retrieve the data from database

    '''

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)

    inspector = inspect(engine)

    # inspector.get_table_names()

    for table_name in inspector.get_table_names():
        # print(table_name)
        continue

        for column in inspector.get_columns(table_name):
            # print("Column: %s" % column['name'])
            continue

    df = pd.read_sql('select * from Message', con=engine)

    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = list(Y)

    return X, Y, category_names


def tokenize(text):
    
    '''
    
    Convert the text into tokens for NLP pipeline

    '''

    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''
    Build pipeline for MultiOutputClassifier

    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
                    'clf__estimator__n_estimators': [5, 10, 20],
                    'clf__estimator__learning_rate': [0.1, 0.2, 0.5]
                }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluate the model using classification_report

    '''

    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names = category_names))


def save_model(model, model_filepath):
    
    '''
    Save the model to the classifier.pkl

    '''

    joblib.dump(model,model_filepath)


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
