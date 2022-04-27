from cnn import generateModel, trainModel, testModel, saveModel
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import visualkeras

import sys
sys.path.append('../data')
from preprocess import get_data

_DEFAULT_TRAIN_FILEPATH = ['train_data_1.gz',
                           'train_data_2.gz', 'train_data_3.gz', 'train_data_4.gz']
_DEFAULT_TEST_FILEPATH = 'test_data.gz'
_DEFAULT_VALIDATION_FILEPATH = 'validation_data.gz'
EMOTIONS = ['Angry', 'Digust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTIONS = ['Happy', 'Angry', 'Sad', 'Surprise']

def viz_training_results(history, epochs=50):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='indigo')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='violet')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='indigo')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='violet')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('graphs/train_results.png')


def viz_test_confusion_matrix(y_pred, y_true, labels):

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Normalized confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.3f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    
    fig.savefig('graphs/test_confusion_matrix.png')

def viz_model_summary(model):
    with open('graphs/model_summary.txt', 'w+') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def viz_model(model):
    visualkeras.layered_view(model, to_file='graphs/model.png')

def main():
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(
                                                    _DEFAULT_TRAIN_FILEPATH, 
                                                    _DEFAULT_VALIDATION_FILEPATH, 
                                                    _DEFAULT_TEST_FILEPATH)
    model = generateModel()
    viz_model_summary(model=model)
    viz_model(model=model)

    model, history = trainModel(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, epochs=50)
    viz_training_results(history=history, epochs=50)

    y_pred, y_true = testModel(model=model, x_test=x_test, y_test=y_test)
    test_acc = accuracy_score(y_true, y_pred)
    print('Model Test Accuracy: ', test_acc)
    viz_test_confusion_matrix(y_pred=y_pred, y_true=y_true, labels=EMOTIONS)

    saveModel(model=model, file_name='cnn-newarc-3emots-2.h5')

if __name__ == "__main__":
    main()
