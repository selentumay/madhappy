
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from regularization import DataAgumentationGenerator
from tensorboard_utils import \
    ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

IMG_DIM = 48
EMOTION_CLASSIFICATION = {0: 'Angry', 1: 'Digust', 2: 'Fear',
                          3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
EMOTION_CLASSIFICATION= {0: 'Happy', 1: 'Neutral', 2: 'Sad', 3: 'Surprise'}


def generateModel(num_emotion=4):
    return Sequential(
        [
            layers.InputLayer(input_shape=(IMG_DIM, IMG_DIM, 1)),

            layers.Conv2D(filters=64, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.Conv2D(filters=64, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=128, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.Conv2D(filters=128, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=256, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.Conv2D(filters=256, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.Conv2D(filters=512, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.Conv2D(filters=512, kernel_size=(3, 3),
                          strides=(1, 1), padding='SAME'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            layers.SpatialDropout2D(rate=0.1),

            layers.Flatten(),

            layers.Dense(units=512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.2),

            layers.Dense(units=128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.2),

            layers.Dense(units=num_emotion, activation='softmax')
        ]
    )


def trainModel(model, x_train, y_train, x_val, y_val, epochs=35, batch_size=64):
    data_augmentation_gen = DataAgumentationGenerator()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            update_freq='batch',
            profile_batch=0),
        CustomModelSaver('checkpoints', 1, 3)
    ]

    

    history = model.fit_generator(
        data_augmentation_gen.flow(x_train, y_train, batch_size),
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val))

    return model, history


def testModel(model, x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    return y_pred, y_true


def saveModel(model, file_name='cnn_weights.h5'):
    model.save(file_name)


def loadModel(filepath='cnn_weights.h5'):
    return keras.models.load_model(filepath)


def predictEmotion(model, image):
    predicition = model.predict(image)[0]
    prob = np.max(predicition)
    emotion_index = np.argmax(predicition)
    emotion_label = EMOTION_CLASSIFICATION[emotion_index]
    return emotion_index, emotion_label, prob
