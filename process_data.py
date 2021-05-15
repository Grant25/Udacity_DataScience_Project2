# import the required libaries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    ''' load_data - a function that loads two files and merges together to create one dataframe
    
    input:  messages_filepath - the location of the messages file
            categories_filepath - the location of the categories file
            
    output: df - the merged file of messages and categories '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id', how='left')
  
    # return the df so it can be used in subsequent functions
    return df


def clean_data(df):
    ''' clean_data - a function that cleans and tidies the data for modelling. Create the response columns and remove duplicates
    
    input - df - the input is the dataframe created from load_data
    
    output - a df that has been tidied and with duplicates removed '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames=row.apply(lambda row: row[0:-2])  
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
    # adapt the lambda function from earlier to select the last value
        cols = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = cols.astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep='last',inplace=True)
      
    # return the df so it can be used in the following functions
    return df


def save_data(df, database_filename):
    ''' save_data - a function that saves the dataframe to the sql database
    input - df: the dataframe that will be added to the sql database
    output - database_filename: the name of the database '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    """ main - a function to run all previous functions in order
    
    load_data (loads the data from the input files, creating a df)
    clean_data (cleans the data, removes duplicates)
    save_data (saves the df to a SQL database)
    
    Inputs: None
    Outputs: None """
    
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()