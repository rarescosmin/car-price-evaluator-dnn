import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import logging

TARGET_COLUMN = 'price'


# return csv dataframe
def load_csv_data(path, fieldnames):
    raw_dataset = pd.read_csv(path, names=fieldnames, encoding='latin-1')
    return raw_dataset

# one hot encodes features using pandas get_dummies
def one_hot_encode_categorical_features(dataframe, columns_to_encode):
    logging.info('One Hot Encoding make, model, fueltype columns... ')
    encoded_dataset = pd.get_dummies(
        dataframe, columns=columns_to_encode, prefix='', prefix_sep='')
    return encoded_dataset


# normalizes label between min and max
def normalize_values(dataset):
    cylinders_min_value = dataset['cylinders'].min()
    print('min cylinders', cylinders_min_value)
    cylinders_max_value = dataset['cylinders'].max()
    print('max cylinders', cylinders_max_value)
    for cylinders_value in dataset['cylinders']:
        normalized_cylinders_value = (cylinders_value - cylinders_min_value) / (cylinders_max_value - cylinders_min_value)
        dataset['cylinders'].replace(cylinders_value, normalized_cylinders_value, inplace=True)


    engineCapacity_min_value = dataset['engineCapacity'].min()
    print('min engineCapacity', engineCapacity_min_value)
    engineCapacity_max_value = dataset['engineCapacity'].max()
    print('max engineCapacity', engineCapacity_max_value)
    for engineCapacity_value in dataset['engineCapacity']:
        normalized_engineCapacity_value = (engineCapacity_value - engineCapacity_min_value) / (engineCapacity_max_value - engineCapacity_min_value)
        dataset['engineCapacity'].replace(engineCapacity_value, normalized_engineCapacity_value, inplace=True)


    mileage_min_value = dataset['mileage'].min()
    print('min mileage', mileage_min_value)
    mileage_max_value = dataset['mileage'].max()
    print('max mileage', mileage_max_value)
    for mileage_value in dataset['mileage']:
        normalized_mileage_value = (mileage_value - mileage_min_value) / (mileage_max_value - mileage_min_value)
        dataset['mileage'].replace(mileage_value, normalized_mileage_value, inplace=True)


    year_min_value = dataset['year'].min()
    print('min year', year_min_value)
    year_max_value = dataset['year'].max()
    print('max year', year_max_value)
    for year_value in dataset['year']:
        normalized_year_value = (year_value - year_min_value) / (year_max_value - year_min_value)
        dataset['year'].replace(year_value, normalized_year_value, inplace=True)


    price_min_value = dataset['price'].min()
    print('min price', price_min_value)
    price_max_value = dataset['price'].max()
    print('max price', price_max_value)
    for price_value in dataset['price']:
        normalized_price_value = (price_value - price_min_value) / (price_max_value - price_min_value)
        dataset['price'].replace(price_value, normalized_price_value, inplace=True)

    return dataset


if __name__ == '__main__':
    logging.basicConfig(filename='car-predict.log',
                        level=logging.DEBUG, filemode='w')

    input_car = {
        'make': 'audi',
        'model': 'a5',
        'year': 2013,
        'mileage': 200000,
        'fuelType': 'diesel',
        'engineCapacity': 1968,
        'cylinders': 4
    }

    raw_dataset = load_csv_data(
        path='./clean-csv-data/cars_train.csv',
        fieldnames=['make', 'model', 'year', 'mileage',
            'fuelType', 'engineCapacity', 'cylinders', 'price']
    )

    encoded_dataset = one_hot_encode_categorical_features(
        raw_dataset, ['make', 'model', 'fuelType']
    )

    encoded_dataset.pop('price')

    encoded_dataset_columns = encoded_dataset.columns
    logging.debug('dataframe columns: \n %s \n', encoded_dataset_columns)

    encoded_dataset_row = encoded_dataset.iloc[0]
    logging.debug('encoded dataset row: \n %s \n', encoded_dataset_row)
    for item in encoded_dataset_row:
        encoded_dataset_row.replace(item, 0, inplace=True)
    
    input_dataset_row = encoded_dataset_row.copy()


    input_dataset_row['year'] = input_car['year']
    input_dataset_row['mileage'] = input_car['mileage']
    input_dataset_row['engineCapacity'] = input_car['engineCapacity']
    input_dataset_row['cylinders'] = input_car['cylinders']

    for item in encoded_dataset_columns:
        if item == input_car['make']:
            input_dataset_row[item] = 1
        
        if item == input_car['model']:
            input_dataset_row[item] = 1
        
        if item == input_car['fuelType']:
            input_dataset_row[item] = 1
    
    
    logging.debug('encoded dataset row: \n %s \n', input_dataset_row)

    prediction_dataframe = pd.DataFrame.from_records(input_dataset_row, columns=encoded_dataset_columns)
    logging.debug('prediction dataframe: \n %s \n', prediction_dataframe)

    
    print('finished prediction')