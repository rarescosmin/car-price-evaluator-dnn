import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import logging
import sys

TARGET_COLUMN = 'price'


# return csv dataframe
def load_csv_data(path, fieldnames):
    raw_dataset = pd.read_csv(path, names=fieldnames, encoding='latin-1')
    return raw_dataset

# one hot encodes features using pandas get_dummies
def one_hot_encode_categorical_features(dataframe, columns_to_encode):
    logging.info('One Hot Encoding make, model, fueltype columns... ')
    encoded_dataset = pd.get_dummies(
        dataframe, columns=columns_to_encode, prefix='', prefix_sep='', dtype=float)
    return encoded_dataset


def normalize_value(value, min, max):
    print('value ', value)
    normalized_value = (value - min) / (max - min)
    print('normalized value ', normalized_value)
    return normalized_value



def reverse_normalization(normalized_value, min, max):
    actual_value = normalized_value * (max - min) + min
    print('actual_value ', actual_value)
    return actual_value



# normalizes label between min and max
def get_min_max_values(dataset):
    cylinders_min_value = dataset['cylinders'].min()
    print('min cylinders', cylinders_min_value)
    cylinders_max_value = dataset['cylinders'].max()
    print('max cylinders', cylinders_max_value)
    

    engineCapacity_min_value = dataset['engineCapacity'].min()
    print('min engineCapacity', engineCapacity_min_value)
    engineCapacity_max_value = dataset['engineCapacity'].max()
    print('max engineCapacity', engineCapacity_max_value)
    

    mileage_min_value = dataset['mileage'].min()
    print('min mileage', mileage_min_value)
    mileage_max_value = dataset['mileage'].max()
    print('max mileage', mileage_max_value)
    

    year_min_value = dataset['year'].min()
    print('min year', year_min_value)
    year_max_value = dataset['year'].max()
    print('max year', year_max_value)
    

    price_min_value = dataset['price'].min()
    print('min price', price_min_value)
    price_max_value = dataset['price'].max()
    print('max price', price_max_value)
    
    return cylinders_min_value, cylinders_max_value, engineCapacity_min_value, engineCapacity_max_value, mileage_min_value, mileage_max_value, year_min_value, year_max_value, price_min_value, price_max_value



def plotMostListedCars(treshhold):
    cars_data = load_csv_data(
        path='./clean-csv-data/cars_train.csv',
        fieldnames=['make', 'model', 'year', 'mileage',
            'fuelType', 'engineCapacity', 'cylinders', 'price']
    )
    car_makes = cars_data['make']

    car_makes_set = set({})
    for make in car_makes:
        car_makes_set.add(make)

    plot_data = {}
    for car_set_value in car_makes_set:
        count = 0
        for make in car_makes:
            if car_set_value == make:
                count = count + 1
        if count > treshhold:
            plot_data[car_set_value] = count

    plot_data_car_makes = list(plot_data.keys())
    plot_data_car_makes_count = list(plot_data.values())

    objects = plot_data_car_makes
    y_pos = np.arange(len(objects))
    performance = plot_data_car_makes_count

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of cars listed')
    plt.xlabel('Car makes')
    plt.title('Most listed cars on Autovit')
    plt.autoscale()
    plt.show()



if __name__ == '__main__':
    logging.basicConfig(filename='car-predict.log',
                        level=logging.DEBUG, filemode='w')
    
    print('sys args', sys.argv)

    input_car = {
        'make': sys.argv[1],
        'model': sys.argv[2],
        'year': int(sys.argv[3]),
        'mileage': int(sys.argv[4]),
        'fuelType': sys.argv[5],
        'engineCapacity': int(sys.argv[6]),
        'cylinders': int(sys.argv[7])
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
    logging.debug('input dataset row: \n %s \n', input_dataset_row)

    cylinders_min_value, cylinders_max_value, engineCapacity_min_value, engineCapacity_max_value, mileage_min_value, mileage_max_value, year_min_value, year_max_value, price_min_value, price_max_value = get_min_max_values(raw_dataset)


    input_dataset_row['year'] = normalize_value(input_car['year'], year_min_value, year_max_value)
    input_dataset_row['mileage'] = normalize_value(input_car['mileage'], mileage_min_value, mileage_max_value)
    input_dataset_row['engineCapacity'] = normalize_value(input_car['engineCapacity'], engineCapacity_min_value, engineCapacity_max_value)
    input_dataset_row['cylinders'] = normalize_value(input_car['cylinders'], cylinders_min_value, cylinders_max_value)

    for item in encoded_dataset_columns:
        if item == input_car['make']:
            input_dataset_row[item] = 1
        
        if item == input_car['model']:
            input_dataset_row[item] = 1
        
        if item == input_car['fuelType']:
            input_dataset_row[item] = 1
    
    
    logging.debug('normalized and encoded dataset row: \n %s \n', input_dataset_row)

    reloaded = tf.keras.models.load_model('car-eval-model')
    normalized_prediction = reloaded.predict(np.array(input_dataset_row).reshape(1, 553))
    print('normalized prediction price is ', normalized_prediction)
    prediction = reverse_normalization(normalized_prediction, price_min_value, price_max_value)
    print('price is: ', prediction)

    #plotMostListedCars(500)

    print('finished prediction')