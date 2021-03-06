import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Method to load in the required data from csv files.
    
    Args:
    messages_filepath: Path to the messages csv file.
    categories_filepath: Path to the messages csv file
    Returns:
    df pandas_dataframe: dataframe of merged files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=['id'])
    
    categories = categories["categories"].str.split(pat=";", expand=True)
    row = categories.iloc[1]
    category_colnames = row.transform(lambda c: c[:-2]).tolist()
    categories.columns = category_colnames
    
    for col in categories:
        categories[col] = categories[col].transform(lambda c: c[-1:]).tolist()
        categories[col] = pd.to_numeric(categories[col])
        
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    return df

def clean_data(df):
    """
    Method to clean the data.
    
    Args:
    df: dataframe to clean.
    Returns:
    df pandas_dataframe: cleaned dataframe
    """
    df.drop_duplicates(inplace=True)
    df = df.dropna()
    
    return df


def save_data(df, database_filename):
    """
    Method to save the data into a db file.
    
    Args:
    df: dataframe to save.
    database_filename: Path to save the db to.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    """
    Method to clean the process data
    """
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