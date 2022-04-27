import tensorflow as tf
import pandas as pd
import numpy as np
import gzip

train_filepath = ['train_data_1.gz', 'train_data_2.gz', 'train_data_3.gz', 'train_data_4.gz']
test_filepath = 'test_data.gz'
validation_path = 'validation_data.gz'
image_hw = 48
EMOTION_CLASSIFICATION = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
CONVERSION_DICT = {3: 0, 4: 1, 6: 2}
EMOTION_CLASSIFICATION_REDUCED = {0: 'Happy', 1: 'Sad', 2: 'Neutral'}


def _split_data(df: pd.DataFrame):
    # convert pixel string to list of ints
    df['pixels'] = df['pixels'].apply(lambda pixel_seq: [int(pixel) for pixel in pixel_seq.split()])
    x = tf.convert_to_tensor(np.array(df['pixels'].tolist(), dtype='float32').reshape(-1, image_hw, image_hw, 1) / 255.0)
    # one hot enconding emotion labels
    y = tf.convert_to_tensor(tf.keras.utils.to_categorical(df['emotion'], len(EMOTION_CLASSIFICATION_REDUCED)))
    return x, y


def get_data(train_data_file_paths: list = train_filepath,
             val_data_file_path: str = validation_path,
             test_data_file_path: str = test_filepath):
    #train data
    #merging list of training data
    train_data_list = []
    for file_path in train_data_file_paths:
        train_data = pd.read_csv(file_path, compression='gzip')
        train_data_list.append(train_data)
    train_data = pd.concat(train_data_list)
    val_data = pd.read_csv(val_data_file_path, compression='gzip')
    
    
    train_data = train_data[train_data['emotion'].isin([3, 4, 6])]
    train_data['emotion'] = train_data['emotion'].replace(CONVERSION_DICT)

    val_data = val_data[val_data['emotion'].isin([3, 4, 6])]
    val_data['emotion'] = val_data['emotion'].replace(CONVERSION_DICT)

    
    x_train, y_train = _split_data(train_data)
    x_val, y_val = _split_data(val_data)

    #test data
    test_data = pd.read_csv(test_data_file_path, compression='gzip')

    test_data = test_data[test_data['emotion'].isin([3, 4, 6])]
    test_data['emotion'] = test_data['emotion'].replace(CONVERSION_DICT)

    x_test, y_test = _split_data(test_data)

    return x_train, y_train, x_val, y_val, x_test, y_test

def shuffle(x: tf.Tensor, y: tf.Tensor, limit: int = None):
    if not limit:
        limit = x.shape[0]
    indices = tf.random.shuffle(tf.range(start=0, limit=limit))
    return tf.gather(x, indices), tf.gather(y, indices)


get_data()