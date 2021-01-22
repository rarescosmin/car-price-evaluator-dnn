import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import logging

TARGET_COLUMN = 'price'

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Check tensorflow version
print('TensorFlow Ver.. ', tf.__version__)

# return csv dataframe
def load_csv_data(path, fieldnames):
    raw_dataset = pd.read_csv(path, names=fieldnames, encoding='latin-1')
    return raw_dataset


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(TARGET_COLUMN)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    # Prepare a Dataset that only yields our feature.
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices.
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))




if __name__ == '__main__':
    logging.basicConfig(filename='car-eval-text-data-dnn.log', level=logging.DEBUG, filemode='w')
    
    # first we load the data into a data frame
    logging.info('Reading dataset... \n')
    dataframe = load_csv_data(
        path='./clean-csv-data/cars_train.csv',
        fieldnames=['make', 'model', 'year', 'mileage', 'fuelType', 'engineCapacity', 'cylinders', 'price']
    )
    logging.debug('dataframe.head() \n %s \n', dataframe.head())

    logging.info('Splitting dataframe into train, test and eval \n')
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    logging.debug('%s train examples', len(train))
    logging.debug('%s validation examples', len(val))
    logging.debug('%s test examples \n', len(test))


    logging.info('Creating new dataset with larger batch \n')
    batch_size = 10
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    
    all_inputs = []
    encoded_features = []

    logging.info('Encoding numerical features [year, mileage, engineCapacity]... \n')
    # Numeric features.
    for header in ['year', 'mileage', 'engineCapacity']:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)
    
    logging.info('Encoding categorical features encoded as integers[cylinders] \n')
    # Categorical features encoded as integers.
    cylinders_col = tf.keras.Input(shape=(1,), name='cylinders', dtype='int64')
    encoding_layer = get_category_encoding_layer('cylinders', train_ds, dtype='int64',
                                                max_tokens=5)
    encoded_cylinders_col = encoding_layer(cylinders_col)
    all_inputs.append(cylinders_col)
    encoded_features.append(encoded_cylinders_col)

    logging.info('Encoding categorical features encoded as strings')
    # Categorical features encoded as string.
    categorical_cols = ['make', 'model', 'fuelType']
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                    max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                loss="mse", metrics=["mse"])
    
    model.fit(train_ds, epochs=50, validation_data=val_ds)
    result = model.evaluate(test_ds)

    print("Result ", result)
    print('FINISHED!')

