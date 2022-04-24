import tensorflow as tf
import pandas as pd
import numpy as np
import gzip

_DEFAULT_TRAIN_FILEPATH = ['train_data_1.gz', 'train_data_2.gz', 'train_data_3.gz', 'train_data_4.gz']
_DEFAULT_TEST_FILEPATH = 'test_data.gz'
_DEFAULT_VALIDATION_FILEPATH = 'validation_data.gz'
_IMAGE_DIM = 48
EMOTION_CLASSIFICATION = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def _merge_train_data(train_data_file_paths: list):
    train_data_list = []
    for file_path in train_data_file_paths:
        train_data = pd.read_csv(file_path, compression='gzip')
        train_data_list.append(train_data)
    return pd.concat(train_data_list)

def _split_data(df: pd.DataFrame):
    # convert pixel string to list of ints
    df['pixels'] = df['pixels'].apply(lambda pixel_seq: [int(pixel) for pixel in pixel_seq.split()])
    x = tf.convert_to_tensor(np.array(df['pixels'].tolist(), dtype='float32').reshape(-1, _IMAGE_DIM, _IMAGE_DIM, 1) / 255.0)
    # one hot enconding emotion labels
    y = tf.convert_to_tensor(tf.keras.utils.to_categorical(df['emotion'], len(EMOTION_CLASSIFICATION)))
    return x, y


def get_train_data(train_data_file_paths: list = _DEFAULT_TRAIN_FILEPATH,
                val_data_file_path: str = _DEFAULT_VALIDATION_FILEPATH):
    
    train_data = _merge_train_data(train_data_file_paths)
    val_data = pd.read_csv(val_data_file_path, compression='gzip')
    x_train, y_train = _split_data(train_data)
    x_val, y_val = _split_data(val_data)
    return x_train, y_train, x_val, y_val


def get_test_data(test_data_file_path: str = _DEFAULT_TEST_FILEPATH):
    test_data = pd.read_csv(test_data_file_path, compression='gzip')
    x_test, y_test = _split_data(test_data)
    return x_test, y_test


def get_data(train_data_file_paths: list = _DEFAULT_TRAIN_FILEPATH,
             val_data_file_path: str = _DEFAULT_VALIDATION_FILEPATH,
             test_data_file_path: str = _DEFAULT_TEST_FILEPATH):
    x_train, y_train, x_val, y_val = get_train_data(train_data_file_paths, val_data_file_path)
    x_test, y_test = get_test_data(test_data_file_path)

    return x_train, y_train, x_val, y_val, x_test, y_test

def shuffle(x: tf.Tensor, y: tf.Tensor, limit: int = None):
    if not limit:
        limit = x.shape[0]
    indices = tf.random.shuffle(tf.range(start=0, limit=limit))
    return tf.gather(x, indices), tf.gather(y, indices)
