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
    raw_dataset = pd.read_csv(path, names=fieldnames)
    return raw_dataset


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10000])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Price]')
    plt.legend()
    plt.grid(True)
    plt.show()




# build and compiles a model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    return model



if __name__ == '__main__':
    logging.basicConfig(filename='car-eval-dnn.log', level=logging.DEBUG, filemode='w')

    # first we load the data into a data frame
    raw_dataset = load_csv_data(
        path='./train-test-data/cars_train.csv',
        fieldnames=['make', 'model', 'year', 'mileage', 'fuelType', 'engineCapacity', 'cylinders', 'price']
    )

    dataset = raw_dataset.copy()
    logging.debug('Initial dataset head looks like this: \n %s \n', dataset.head())
    logging.debug('Initial dataset data types are: \n %s \n', dataset.dtypes)

    logging.info('Splitting data into train and test... \n')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    logging.debug('Train dataset has %s data_types  \n and head looks like this: \n %s \n', train_dataset.dtypes, train_dataset.head())

    test_dataset = dataset.drop(train_dataset.index)
    logging.debug('Test dataset has %s data_types \n and head looks like this: \n %s \n', test_dataset.dtypes, test_dataset.head())
    
    logging.debug('General information about the train dataset variables: \n %s \n', train_dataset.describe().transpose())
    logging.info('Displaying Train Dataset mean and deviation: \n %s \n', train_dataset.describe().transpose()[['mean', 'std']])

    logging.info('Splitting features from labels.. \n')

    train_features = train_dataset.copy()
    train_labels = train_features.pop('price')
    logging.debug('Train dataset features head look like this: \n %s \n', train_features.head())
    logging.debug('Train dataset labels head look like this: \n %s \n', train_labels.head())

    test_features = test_dataset.copy()
    test_labels = test_features.pop('price')
    logging.debug('Test dataset features head look like this: \n %s \n', test_features.head())
    logging.debug('Test dataset labels head look like this: \n %s \n \n', test_labels.head())

    logging.info('Adding normalization layer... \n')
    normalizer = preprocessing.Normalization()

    logging.info('Adapting normalizer to train featuers... \n')
    normalizer.adapt(np.array(train_features))

    logging.info('Testing application of normalizer...')
    first = np.array(train_features[:1])
    logging.debug('First example: \n %s', first)
    logging.debug('Normalized: \n %s', normalizer(first).numpy())
    logging.info('Finished testing for normalizer... \n \n')

    logging.info('Creating DNN model... \n')
    dnn_model = build_and_compile_model(normalizer)

    logging.info('Training model... \n')
    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=200
    )

    plot_loss(history)


    logging.info('<<--- Model trained! Gathering results... --->>\n')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    logging.debug('Model last epochs values \n %s \n', hist.tail())

    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
    
    logging.debug('logging results \n %s \n ', pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

    print('Car Eval with DNN finished running')