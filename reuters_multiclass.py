"""Following Section 3.5 in Deep Learning with Python."""

from keras.datasets import reuters
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

def decode_to_text(word_index_array):
    """Decode an array of word indices to text."""
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    word_array = [reverse_word_index.get(i - 3, '?') for i in word_index_array]
    return ' '.join(word_array)

def vectorize_sequences(sequences, dimension=10000):
    """Convert a sequence of words to a binary presence vector."""
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension):
    """Convert a label to a one-hot vector."""
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def prepare_data():
    """Load and prepare data from reuters data set."""
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = to_one_hot(train_labels, 46)
    y_test = to_one_hot(test_labels, 46)
    return x_train, x_test, y_train, y_test

def build_model(depth=3, width=64, final_width=46):
    model = models.Sequential()
    model.add(layers.Dense(width, activation='relu', input_shape=(10000,)))
    for i in range(depth - 2):
        model.add(layers.Dense(width, activation='relu'))
    model.add(layers.Dense(final_width, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    return model

def train_model(model, x_train, y_train, epochs=20):
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]
    history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=512, validation_data=(x_val, y_val))
    return history

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
    print(history_dict.keys())
    acc = history_dict['categorical_accuracy']
    val_acc = history_dict['val_categorical_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

def main():
    x_train, x_test, y_train, y_test = prepare_data()
    model = build_model()
    history = train_model(model, x_train, y_train, epochs=100)
    model_narrow = build_model(depth=3, width=4, final_width=46)
    history_narrow = train_model(model_narrow, x_train, y_train, epochs=100)
    plot_loss(history)
    plot_accuracy(history)
    plot_loss(history_narrow, title='Loss Narrow')
    plot_accuracy(history_narrow, title='Acc Narrow')
    plt.show()

if __name__ == '__main__':
    main()