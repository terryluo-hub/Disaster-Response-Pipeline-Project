import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine




def load_data(messages_filepath, categories_filepath):
    
    '''
	Load message and categories file, then merge them into one dataframe

	'''


    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    
    '''
	Convert the categorical columns into numerical columns, then use 
	categorical value as column names. After that, convert the outliner, Finally, 
	remove the duplicate.

	'''


    categories = df['categories'].str.split(pat=';',expand=True)
    
    row = categories.iloc[0]
    
    category_colnames = row.apply(lambda x: x[0:-2])
    
    categories.columns = category_colnames

    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    
    for column in categories:
    
      categories[column] = categories[column].apply(lambda x: x[-1:])
      categories[column] = pd.to_numeric(categories[column])
    
    df.drop(['categories'],axis=1,inplace=True)
    
    df = pd.concat([df, categories],axis=1)
    
    df = df.drop_duplicates()


    return df


def save_data(df, database_filename):
    
    '''
	Save the data to the 'Message' table in sqlite database

	'''

    engine = create_engine('sqlite:///' + database_filename)

    df.to_sql('Message', engine, index=False)


def main():
    if len(sys.argv) == 4:

        # print(sys.argv)

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # print(df.head())

        print('Cleaning data...')
        df = clean_data(df)

        # print(df.head())

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
