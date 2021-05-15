import sys

# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# libraries for text mining
import nltk
nltk.download(['punkt','wordnet','stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re

# libraries for ML Flow
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# libraries for export
import pickle


def load_data(database_filepath):
    """ load_data - a function that loads data from the SQL database
    
    input: datbase_filepath - the location of the database
    output: X, Y sets of data, the column names """
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # the input that we are using to predict the response is the message field
    X = df['message']

    # we want to predict our 36 categories - i.e. related, request, weather_related, etc.
    Y =df.drop(['id','message','original','genre'],axis=1)

    # store column names for ref
    category_names=list(Y.columns)
    
    return X,Y,category_names


def tokenize(text):
    """ function to clean tokens, remove stop words as well as punctuation 
    
    input: text - this is the messages column
    
    output: clean_tokens - the work tokens that are cleaned and can then be used in modelling """
        
    # regex to remove punctuation     
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
      
        
    tokens = word_tokenize(text)
    tokens2= [w for w in tokens if w not in stopwords.words('english')]
    
    lemmatizer= WordNetLemmatizer()
    
    clean_tokens=[]
      
    
    for tok in tokens2:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens 


def build_model():
    """ build_model - a function that runs a pipeline to build and classify the messages
    
    Inputs: None
    Output: cv - the cross-validated model """
    
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'clf__estimator__n_estimators': [10,25,50,100]}
                
    cv = GridSearchCV(pipeline,param_grid=parameters)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """" evaluate_model - a function to output model accuracy as well as precision and F1 for each category
    
    input: model - model is the object created from the build_model() function (i.e. the returned cv)
           X_test - the test set for the predictors, the messages
           Y_test - the test set for the respnse categories
           category_names - labels for the classification_report
           
    output: None """
    
    y_pred = model.predict(X_test)

    print("Overall Model Accuracy: "+ str((y_pred==Y_test).mean()), '\n')

    # iterate through the index of our column names - this will ensure the table is interpretable
    for ind, cols in enumerate(category_names):
    
        # print the column name along with the associated report for that index position
        print(cols,classification_report(Y_test.iloc[:,ind], y_pred[:,ind] ))
    
    


def save_model(model, model_filepath):
    """ save_model - a function to save the model as a pickle file
    
    Input:  model - the fitted model
            model_filepath - the location of where the pickle file will be saved
    
    output: None """
    
    classifier = open(model_filepath,"wb")
    pickle.dump(model, classifier)
    classifier.close()


    
def main():
    """ main - a function that runs all previous functions
        load_data (to load the data from the database)
        splits to test and train
        build_model (to build the model pipeline)
        applied model.fit (to fit the pipeline to the training data)
        evaluate_model (runs the evaluation function)
        save_model (saves the trained and tested model to a pickle file)
        
        Inputs: None
        Outputs: None """
        
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