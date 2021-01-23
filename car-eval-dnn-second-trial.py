import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import logging


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Check tensorflow version
print('TensorFlow Ver.. ', tf.__version__)

# return csv dataframe
def load_csv_data(path, fieldnames):
    raw_dataset = pd.read_csv(path, names=fieldnames, encoding='latin-1')
    return raw_dataset

# one hot encodes features using pandas get_dummies
def one_hot_encode_categorical_features(dataframe, columns_to_encode):
    logging.info('One Hot Encoding make, model, fueltype columns... ')
    encoded_dataset = pd.get_dummies(dataframe, columns=columns_to_encode, prefix='', prefix_sep='')
    return encoded_dataset


if __name__ == '__main__':
    logging.basicConfig(filename='car-eval-dnn.log', level=logging.DEBUG, filemode='w')

    logging.info('Reading csv... ')
    # first we load the data into a data frame
    raw_dataset = load_csv_data(
        path='./clean-csv-data/cars_train.csv',
        fieldnames=['make', 'model', 'year', 'mileage', 'fuelType', 'engineCapacity', 'cylinders', 'price']
    )
    logging.debug('Initial dataset: \n %s \n', raw_dataset.head())

    encoded_dataset = one_hot_encode_categorical_features(raw_dataset, ['fuelType'])
    logging.debug('Dataset has been encoded looks like this: \n %s \n', encoded_dataset)
    logging.debug('Column names after encoding: \n %s \n', encoded_dataset.columns)
    


