from keras.datasets import imdb
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import time

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def plot_loss(history, title='Loss'):
    """Plot loss relative to epochs"""
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.figure()
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def plot_accuracy(history, title='Accuracy'):
    """Plot accuracy relative to epochs"""
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

def build_model(depth=3, width=16):
    model = models.Sequential()
    model.add(layers.Dense(width, activation='relu', input_shape=(10000,)))
    for i in range(depth - 2):
        model.add(layers.Dense(width, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    return model

def train_model(model, x_train, y_train):
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
    return history

def prepare_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    return x_train, x_test, y_train, y_test

def run_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    print(y_predict[:10])
    print(y_test[:10])

def DoIt():
    x_train, x_test, y_train, y_test = prepare_data()
    model_2 = build_model(depth=3, width=16)
    history_2 = train_model(model_2, x_train, y_train)
    plot_loss(history_2, 'Loss 2')
    plot_accuracy(history_2, 'Acc 2')
    model_3 = build_model(depth=3, width=32)
    history_3 = train_model(model_3, x_train, y_train)
    plot_loss(history_3, 'Loss 3')
    plot_accuracy(history_3, 'Acc 3')
    model_4 = build_model(depth=3, width=64)
    history_4 = train_model(model_4, x_train, y_train)
    plot_loss(history_4, 'Loss 4')
    plot_accuracy(history_4, 'Acc 4')
    plt.show()
    run_model(model_2, x_test, y_test)

if __name__ == '__main__':
    DoIt()