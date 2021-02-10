import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras import backend as K
from keras.losses import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import logging

TARGET_COLUMN = 'price'

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Check tensorflow version
print('TensorFlow Ver.. ', tf.__version__)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))




def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([-1, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Price]')
    plt.legend()
    plt.grid(True)
    plt.show()


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
    logging.basicConfig(filename='car-eval-dnn.log',
                        level=logging.DEBUG, filemode='w')

    # first we load the data into a data frame
    logging.info('Reading csv... ')
    raw_dataset = load_csv_data(
        path='./clean-csv-data/cars_train.csv',
        fieldnames=['make', 'model', 'year', 'mileage',
            'fuelType', 'engineCapacity', 'cylinders', 'price']
    )
    logging.debug('Initial dataset: \n %s \n', raw_dataset.head())

    logging.info('Normalizing values... ')
    dataset = normalize_values(raw_dataset)
    logging.debug('Normalized labels look like this: \n %s \n', dataset.head())

    # encoding categorical features
    encoded_dataset = one_hot_encode_categorical_features(
        dataset, ['make', 'model', 'fuelType']
    )
    logging.debug(
        'Dataset has been encoded looks like this: \n %s \n', encoded_dataset)
    logging.debug('Column names after encoding: \n %s \n',
                  encoded_dataset.columns)

    # split dataset into train and test
    logging.info('Splitting dataset into train and test...')
    train_dataset = encoded_dataset.sample(frac=0.8, random_state=0)
    test_dataset = encoded_dataset.drop(train_dataset.index)
    logging.debug('Train dataset size %s : \n %s \n',
                  len(train_dataset), train_dataset.head())
    logging.debug('Test dataset size %s : \n %s \n',
                  len(test_dataset), test_dataset.head())
    

    logging.info('General stats about the data: \n %s \n',
                 train_dataset.describe().transpose())

    logging.info('Spliting features from labels... ')
    train_features = train_dataset.copy()
    train_labels = train_features.pop(TARGET_COLUMN)
    test_features = test_dataset.copy()
    test_labels =  test_features.pop(TARGET_COLUMN)
    logging.debug('Train features: \n %s \n', train_features.head())
    logging.debug('Train labels: \n %s \n', train_labels.head())
    logging.debug('Test features: \n %s \n', test_features.head())
    logging.debug('Test labels: \n %s \n', test_labels.head())


    # logging.info('Adding normalizer...')
    # normalizer = preprocessing.Normalization()
    # normalizer.adapt(np.array(train_features))

    # first = np.array(train_features[:1])
    # with np.printoptions(precision=2, suppress=True):
    #     print('First example:', first)
    #     print()
    #     print('Normalized:', normalizer(first).numpy())

    model = keras.Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    #print('summary ', model.summary())

    model.compile(loss=mean_squared_error,
                    optimizer=tf.keras.optimizers.Adam(0.001))
    
    
    history = model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=1, epochs=10
    )
    plot_loss(history)

    test_results = {}
    test_results['model'] = model.evaluate(test_features, test_labels, verbose=1)

    model.save('car-eval-model')

